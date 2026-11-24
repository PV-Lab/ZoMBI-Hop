"""Golden bytes and DB effects of today's LEGACY serial path (lanes port, task T9).

DESIGN.md section 4.4 (benchmarks/zoom_fix/lanes/report/review/impl/design/): BEFORE any
Phase-2 edit, pin exactly what the legacy ZoMBI side puts on the wire and into its sqlite
mailboxes, so that Phase 2 (lanes4 functions in scripts/communication.py,
scripts/lanes_io.py, the scripts/main.py and scripts/serial_only.py flags, the
run_zombi_main lanes branch) can prove that protocol="legacy" is unchanged.

Sections (fixtures: tests/fixtures/lanes_legacy/<section>.json):
  writer_sender  run_zombi_main._write_line_pair -> sql/compositions.db (DROP + CREATE of
                 seven tables) -> communication.composition_sender -> the exact frames on the
                 wire; identical re-sends every tick; no frame while any table is empty; the
                 write-error path (reset_output_buffer; the next frame is NOT led by "\\n").
  receiver       communication.objective_receiver fed DiSCO's exact line bytes ->
                 sql/objective.db + sql/objective_memory.db after every line: ts de-dup,
                 r / k / rail ignored, caps / selftest / NULL notice / all-NaN / malformed
                 lines dropped, the one-slot mailbox overwrite, value de-dup without ts.
  hygiene        get_y_measurements on each received packet: NaN y, all-zero composition
                 rows, torn and comps-less packets. This is the reference that DESIGN 6.3's
                 lanes_io.clean_packet must reproduce (without the waits).
  handshake      run_zombi_main.objective(): call order write -> reset -> wait/read -> reset
                 -> log, the DB state after every step, the composition_log record and the
                 returned tensors; plus reset_objective, clear_in_flight_objective_state and
                 the scripts/main.py new-trial reset (initialize_db + resets), which must
                 leave every other file in sql/ alone (reviewer correction R2.3).
  selftest       serial_link_selftest ping and pong frames.
  orchestrator   start_serial_dual_io_shared_port with pyserial's Serial replaced by an
                 in-memory port (no COM port is opened): the real reader / receiver / sender
                 threads reproduce writer_sender's frame and the synchronous receiver's DB.

Everything runs in-process with cwd = a per-case folder under tmp_path (the legacy code uses
cwd-relative ./sql/ paths, as tests/test_axial_init_hardware.py relies on). An autouse fixture
restores the CPU/CUDA RNG states, the three random_simplex attributes and every module global
the legacy path touches (REVIEW F9).

Phase 2 must keep this file green WITHOUT re-capturing. Re-capture only to accept a
deliberate change of the legacy path, and say so in that change:
    PYTHONPATH=. python -B tests/test_lanes_legacy_bytes.py --capture [--force]
The capture refuses to overwrite existing fixtures without --force, runs every case twice
(determinism check) and checks the cross-section claims before writing anything.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import queue
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time as _real_time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:                      # script (capture) mode
    sys.path.insert(0, str(REPO))

import numpy as np
import pytest
import serial
import torch

import scripts.run_zombi_main as rzm
from scripts import communication
from scripts.initialize_databases import initialize_db

FIXDIR = Path(__file__).resolve().parent / "fixtures" / "lanes_legacy"
SECTIONS = ("writer_sender", "receiver", "hygiene", "handshake", "selftest", "orchestrator")
SCHEMA = 1

OBJ_DB = "./sql/objective.db"
MEM_DB = "./sql/objective_memory.db"
COMP_DB = "./sql/compositions.db"
# Sender / receiver rate in the replays. Production runs 1 Hz (scripts/main.py:75-80); the
# rate only paces the loops (sleep / queue timeout), it never changes a byte.
HZ = 20.0
# run_zombi_main.NUM_EXPERIMENTS at capture; passed explicitly so the golden pins 24 rows.
N_EXP = 24
# run_zombi_main's frozen time.time() base (the ts of each composition write and log record).
T_CLOCK = 1793900000.25

# Capture inputs (read-only; the sandbox must stay byte-identical). Only --capture reads them;
# the fixtures carry every input, so the tests never touch benchmarks/.
_RID = REPO / "benchmarks" / "zoom_fix" / "lanes" / "result_id" / "fixtures"
_SRC_FILES = ("scripts/communication.py", "scripts/run_zombi_main.py",
              "scripts/initialize_databases.py", "scripts/main.py", "scripts/serial_only.py")


# ─────────────────────────────────────────────────────────────────────────────
# small utilities (work with and without pytest)
# ─────────────────────────────────────────────────────────────────────────────

class _Patches:
    """Minimal monkeypatch usable outside pytest (capture mode)."""

    _MISSING = object()

    def __init__(self):
        self._undo = []

    def set(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name, self._MISSING)))
        setattr(obj, name, value)

    def undo(self):
        while self._undo:
            obj, name, old = self._undo.pop()
            if old is self._MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, old)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.undo()
        return False


class FrozenClock:
    """Stand-in for run_zombi_main's ``time`` module: time() is frozen, the rest is real.

    _write_line_pair stamps compositions.db with ``time.time()`` (RZM 876) and
    log_compositions stamps its record the same way (RZM 188); both read the module global
    ``time`` of scripts.run_zombi_main at call time.
    """

    def __init__(self, t):
        self._t = float(t)

    def time(self):
        return self._t

    def __getattr__(self, name):
        return getattr(_real_time, name)


class BoundedSleep:
    """time.sleep stand-in while get_y_measurements runs. Its waits are 1 s polls: on the
    golden inputs it polls only in the torn / comps-less cases (29 re-reads before the
    zero-fill fallback, RZM 718-720). A changed legacy path that would make it wait for a
    packet that never comes fails here instead of hanging the suite."""

    def __init__(self, limit=200):
        self.limit = limit
        self.calls = 0

    def __call__(self, _s=0.0):
        self.calls += 1
        if self.calls > self.limit:
            raise AssertionError("get_y_measurements kept waiting: the packet was not "
                                 "delivered to it as the legacy path does")


def plain(x):
    """JSON-ready, exact (floats keep their repr; bytes kept as latin-1 text)."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        return {"tensor": plain(t.tolist()), "dtype": str(t.dtype), "device": x.device.type,
                "shape": list(t.shape)}
    if isinstance(x, np.ndarray):
        return {"nd": plain(x.tolist()), "dtype": str(x.dtype), "shape": list(x.shape)}
    if isinstance(x, (bytes, bytearray)):
        return {"bytes": bytes(x).decode("latin-1")}
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    return x


def canon(x) -> str:
    return json.dumps(plain(x), sort_keys=True, allow_nan=True, separators=(",", ":"))


def _short(v, n=160):
    s = canon(v)
    return s if len(s) <= n else s[:n] + "..."


def first_diff(a, b, path="$"):
    """Human-readable location of the first difference between two plain structures."""
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return (f"{path}: keys differ (only in got: {sorted(set(a) - set(b))}, "
                    f"only in golden: {sorted(set(b) - set(a))})")
        for k in sorted(a):
            d = first_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        for i, (u, v) in enumerate(zip(a, b)):
            d = first_diff(u, v, f"{path}[{i}]")
            if d:
                return d
        if len(a) != len(b):
            return f"{path}: length {len(a)} (got) != {len(b)} (golden)"
        return None
    if canon(a) != canon(b):
        return f"{path}: got {_short(a)} != golden {_short(b)}"
    return None


def assert_golden(got, expect, what):
    g = json.loads(canon(got))
    if canon(g) != canon(expect):
        raise AssertionError(f"{what}: the legacy serial path changed: {first_diff(g, expect)}")


def dump_db(path) -> dict:
    """Every table of a sqlite file: CREATE sql + rows in rowid order (exact floats)."""
    p = Path(path)
    if not p.exists():
        return {"exists": False}
    conn = sqlite3.connect(str(p), timeout=10.0)
    try:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        out = {}
        for name, sql in tables:
            rows = conn.execute(f'SELECT * FROM "{name}" ORDER BY rowid').fetchall()
            out[name] = {"sql": sql, "rows": [list(r) for r in rows]}
        other = conn.execute(
            "SELECT type, name FROM sqlite_master WHERE type != 'table' ORDER BY name").fetchall()
    finally:
        conn.close()
    return {"exists": True, "tables": out, "other": [list(o) for o in other]}


def read_flag(db=OBJ_DB):
    if not Path(db).exists():
        return None
    conn = sqlite3.connect(db, timeout=10.0)
    try:
        row = conn.execute("SELECT new_objective_available FROM handshake WHERE id = 1").fetchone()
    except sqlite3.Error:
        row = None
    finally:
        conn.close()
    return None if row is None else int(row[0])


def _count_rows(db, table):
    try:
        conn = sqlite3.connect(db, timeout=10.0)
        try:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error:
        return -1


def receiver_state():
    return {"objective.db": dump_db(OBJ_DB), "objective_memory.db": dump_db(MEM_DB)}


def _sha1(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def _line(s: str) -> bytes:
    """Fixture text -> the exact bytes (stored as latin-1 so any byte round-trips)."""
    return s.encode("latin-1")


@contextlib.contextmanager
def in_dir(name):
    """A fresh per-case folder with sql/ as the cwd (the legacy code uses ./sql/...)."""
    old = os.getcwd()
    d = Path(old) / name
    d.mkdir(parents=True, exist_ok=False)
    (d / "sql").mkdir()
    os.chdir(d)
    try:
        yield d
    finally:
        os.chdir(old)


def _drain(q):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


@contextlib.contextmanager
def legacy_sandbox(workdir: Path):
    """cwd = workdir; on exit restore cwd, RNG states, the three random_simplex attributes
    and every module global of scripts.communication / scripts.run_zombi_main the legacy
    path touches (REVIEW F9). Also keeps a non-UTF-8 console from turning the receiver's
    "✔" log line into a write retry (see the T9 report)."""
    from src.core import linebo as _lb
    from src.core import zombihop as _zh
    from src.utils import domain as _dom
    simplex = {m: m.random_simplex for m in (_dom, _zh, _lb)}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_rng = np.random.get_state()
    comm_names = ("_last_objective_ts", "_last_objective_vals", "_last_objective_comps",
                  "_objective_writing", "_read_queue")
    comm_saved = {k: getattr(communication, k) for k in comm_names}
    rzm_saved = {k: getattr(rzm, k) for k in ("OPTIMIZING_DIMS", "_COMP_LOG", "time")}
    serial_cls = serial.Serial
    flag_was_set = communication._shutdown_flag.is_set()
    streams = []
    for s in (sys.stdout, sys.stderr):
        enc = (getattr(s, "encoding", "") or "").lower().replace("-", "").replace("_", "")
        if enc != "utf8" and hasattr(s, "reconfigure"):
            try:
                streams.append((s, s.errors))
                s.reconfigure(errors="backslashreplace")
            except Exception:
                streams.pop()
    old_cwd = os.getcwd()
    Path(workdir).mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)
    communication._shutdown_flag.clear()
    _drain(communication._read_queue)
    try:
        yield Path(workdir)
    finally:
        os.chdir(old_cwd)
        _drain(communication._read_queue)
        for k, v in comm_saved.items():
            setattr(communication, k, v)
        for k, v in rzm_saved.items():
            setattr(rzm, k, v)
        serial.Serial = serial_cls
        if flag_was_set:
            communication._shutdown_flag.set()
        else:
            communication._shutdown_flag.clear()
        for m, f in simplex.items():
            m.random_simplex = f
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(np_rng)
        for s, err in streams:
            try:
                s.reconfigure(errors=err)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# in-memory serial handles (no COM port is ever opened)
# ─────────────────────────────────────────────────────────────────────────────

class WirePort:
    """What composition_sender writes to. Stops the sender after `stop_after` frames by
    raising communication._shutdown_flag (the loop condition), optionally failing the first
    `fail_first` writes with a SerialException."""

    def __init__(self, stop_after=1, fail_first=0):
        self.stop_after = stop_after
        self.fail_left = fail_first
        self.frames, self.failed, self.events = [], [], []

    def write(self, data):
        data = bytes(data)
        if self.fail_left > 0:
            self.fail_left -= 1
            self.failed.append(data)
            self.events.append(["write_failed", len(self.failed) - 1])
            raise serial.SerialException("in-memory port: injected write failure")
        self.frames.append(data)
        self.events.append(["write", len(self.frames) - 1])
        if len(self.frames) >= self.stop_after:
            communication._shutdown_flag.set()
        return len(data)

    def flush(self):
        self.events.append(["flush"])

    def reset_output_buffer(self):
        self.events.append(["reset_output_buffer"])


class SelftestPort:
    """Answers the first ping with a fixed DiSCO byte stream, delivered in read(n) chunks."""

    def __init__(self, replies: bytes):
        self.timeout = 0.1
        self.out = []
        self._replies = replies
        self._in = bytearray()
        self.reset_input_calls = 0

    def write(self, data):
        self.out.append(bytes(data))
        if len(self.out) == 1:
            self._in += self._replies
        return len(data)

    def flush(self):
        pass

    def read(self, n=1):
        chunk = bytes(self._in[:n])
        del self._in[:n]
        return chunk

    def reset_input_buffer(self):
        self.reset_input_calls += 1


class LoopbackPort:
    """Replaces serial.Serial inside start_serial_dual_io_shared_port: reads return the
    preloaded DiSCO bytes in read(n) chunks, writes are recorded."""

    preload = b""
    opened: list = []

    def __init__(self, port, baudrate, timeout=None, write_timeout=None, **kw):
        self.args = {"port": port, "baudrate": baudrate, "timeout": timeout,
                     "write_timeout": write_timeout, "extra_kwargs": sorted(kw)}
        self.timeout = timeout
        self.is_open = True
        self._in = bytearray(type(self).preload)
        self._lock = threading.Lock()
        self.frames, self.calls = [], []
        type(self).opened.append(self)

    def pending_in(self):
        with self._lock:
            return len(self._in)

    def read(self, n=1):
        with self._lock:
            chunk = bytes(self._in[:n])
            del self._in[:n]
        if not chunk:
            _real_time.sleep(0.005)
        return chunk

    def write(self, data):
        with self._lock:
            self.frames.append(bytes(data))
        return len(data)

    def flush(self):
        pass

    def reset_input_buffer(self):
        self.calls.append("reset_input_buffer")

    def reset_output_buffer(self):
        self.calls.append("reset_output_buffer")

    def cancel_read(self):
        self.calls.append("cancel_read")

    def close(self):
        self.is_open = False
        self.calls.append("close")


class _DrainingQueue(queue.Queue):
    """objective_receiver's input queue for a synchronous replay: once it is empty it
    raises communication._shutdown_flag, so the receiver loop returns."""

    def get(self, block=True, timeout=None):
        try:
            return super().get(block=False)    # (get_nowait would re-enter this override)
        except queue.Empty:
            communication._shutdown_flag.set()
            raise


# ─────────────────────────────────────────────────────────────────────────────
# drivers of the legacy functions
# ─────────────────────────────────────────────────────────────────────────────

def write_pair(dims, endpoints, ts):
    """run_zombi_main._write_line_pair (the writer behind objective() and the init path)."""
    with _Patches() as p:
        p.set(rzm, "OPTIMIZING_DIMS", list(dims))
        p.set(rzm, "time", FrozenClock(ts))
        ref = {}
        ep = torch.tensor(endpoints, dtype=torch.float64, device="cpu")
        x_main, x_cache = rzm._write_line_pair(ep, ref, N_EXP)
    return x_main, x_cache, ref


def run_sender(stop_after=1, fail_first=0, max_s=10.0):
    """composition_sender exactly as start_serial_dual_io_shared_port starts it
    (args ser, hz, comp_db, chaos, verbose), on an in-memory port."""
    port = WirePort(stop_after, fail_first)
    communication._shutdown_flag.clear()
    guard = threading.Timer(max_s, communication._shutdown_flag.set)
    guard.daemon = True
    guard.start()
    try:
        communication.composition_sender(port, HZ, COMP_DB, False, True)
    finally:
        guard.cancel()
        communication._shutdown_flag.clear()
    return port


def fresh_receiver():
    """The receiver's de-dup memory as a newly started serial process has it."""
    communication._last_objective_ts = None
    communication._last_objective_vals = None
    communication._last_objective_comps = None


def receive(lines):
    """Feed raw lines (bytes incl. the "\\n", as _serial_reader queues them) through the real
    objective_receiver (args hz, obj_db, mem_db, verbose as the orchestrator passes them)."""
    q = _DrainingQueue()
    for ln in lines:
        q.put(ln)
    with _Patches() as p:
        p.set(communication, "_read_queue", q)
        communication._shutdown_flag.clear()
        # Safety net: the receiver catches every exception and loops; never hang a test.
        guard = threading.Timer(30.0, communication._shutdown_flag.set)
        guard.daemon = True
        guard.start()
        try:
            communication.objective_receiver(HZ, OBJ_DB, MEM_DB, True)
        finally:
            guard.cancel()
            communication._shutdown_flag.clear()
        if not q.empty():
            raise AssertionError("objective_receiver did not consume every line (timeout)")


# -- section runners: input dict -> result dict (the golden "expect") ---------------------

def run_writer(inp):
    initialize_db()
    x_main, x_cache, ref = write_pair(inp["dims"], inp["endpoints"], inp["ts"])
    db = dump_db(COMP_DB)
    port = run_sender(stop_after=2)
    return {"x_main": x_main, "x_cache": x_cache, "endpoints_log_ref": ref,
            "compositions.db": db, "frames": port.frames, "events": port.events}


def run_writer_overwrite(inp):
    initialize_db()
    for w in inp["writes"]:
        write_pair(w["dims"], w["endpoints"], w["ts"])
    return {"compositions.db": dump_db(COMP_DB)}


def run_sender_write_failure(inp):
    initialize_db()
    write_pair(inp["dims"], inp["endpoints"], inp["ts"])
    port = run_sender(stop_after=1, fail_first=1)
    return {"events": port.events, "failed": port.failed, "frames": port.frames}


def run_sender_no_frame(inp):
    initialize_db()
    if inp.get("write"):
        w = inp["write"]
        write_pair(w["dims"], w["endpoints"], w["ts"])
    if inp.get("then") == "launch_reset":            # scripts/main.py:191-193
        initialize_db()
        communication.reset_objective()
        communication.reset_compositions()
    elif inp.get("then") == "empty_cache_table":
        conn = sqlite3.connect(COMP_DB)
        conn.execute("DELETE FROM compositions_cache")
        conn.commit()
        conn.close()
    port = run_sender(stop_after=1, max_s=inp["listen_s"])
    return {"frames": port.frames, "compositions.db": dump_db(COMP_DB)}


def run_receiver_rk(inp):
    out = {}
    for key in ("plain", "with_rk"):
        with in_dir(key):
            initialize_db()
            fresh_receiver()
            receive([_line(inp[key])])
            out[key] = receiver_state()
    return out


def consume():
    """What objective() does with a delivered packet: get_y_measurements lowers the flag
    (RZM 796-804, same SQL), then objective() calls reset_objective() (RZM 913). After it,
    flag == 1 after a line means exactly "the receiver ingested that line"."""
    conn = sqlite3.connect(OBJ_DB, timeout=10.0)
    conn.execute("UPDATE handshake SET new_objective_available = 0 WHERE id = 1")
    conn.commit()
    conn.close()
    communication.reset_objective()


def run_receiver_sequence(inp):
    """One receiver session (de-dup memory carried over, as in the serial process). Before
    each line the previous packet is consumed, except where a step keeps it unread."""
    initialize_db()
    fresh_receiver()
    states = []
    for step in inp["steps"]:
        if step.get("consume_before", True):
            consume()
        receive([_line(step["line"])])
        states.append({"step": step["name"], "flag": read_flag(), **receiver_state()})
    return {"after_each_line": states}


def run_hygiene(inp):
    initialize_db()
    fresh_receiver()
    receive([_line(inp["line"])])
    out = {"flag_after_receive": read_flag()}
    if out["flag_after_receive"] == 1:
        with _Patches() as p:
            p.set(rzm, "OPTIMIZING_DIMS", list(inp["dims"]))
            # torn / comps-less packets: the 1 s re-reads are skipped, never a hang
            p.set(_real_time, "sleep", BoundedSleep())
            y, x, idx = rzm.get_y_measurements(None, ready_for_objectives=True,
                                               return_indices=True,
                                               accept_empty=inp["accept_empty"])
        out.update({"y": y, "x_meas": x, "valid_indices": idx, "flag_after_read": read_flag()})
    out["objective.db"] = dump_db(OBJ_DB)
    return out


def run_handshake(inp):
    """run_zombi_main.objective() as LineBO calls it (RZM 973), every collaborator recorded.
    DiSCO's result is fed through the real receiver right after the first reset_objective,
    i.e. while objective() waits in get_y_measurements (the order on hardware)."""
    steps = []
    initialize_db()
    fresh_receiver()
    if inp.get("stale_line"):
        receive([_line(inp["stale_line"])])
        steps.append({"step": "stale_result_received", "objective.db": dump_db(OBJ_DB)})
    o_write, o_reset = communication.write_compositions, communication.reset_objective
    o_gety, o_log = rzm.get_y_measurements, rzm.log_compositions
    n_reset = [0]

    def write_rec(*a, **kw):
        r = o_write(*a, **kw)
        steps.append({"step": "write_compositions", "n_args": len(a), "kwargs": sorted(kw),
                      "compositions.db": dump_db(COMP_DB), "objective.db": dump_db(OBJ_DB)})
        return r

    def reset_rec(*a, **kw):
        r = o_reset(*a, **kw)
        n_reset[0] += 1
        steps.append({"step": "reset_objective", "args": list(a), "kwargs": kw,
                      "objective.db": dump_db(OBJ_DB)})
        if n_reset[0] == 1:
            receive([_line(inp["line"])])
            steps.append({"step": "disco_result_received", "objective.db": dump_db(OBJ_DB)})
        return r

    def gety_rec(x, *a, **kw):
        with _Patches() as q:          # never sleeps on the golden inputs; fail, don't hang
            q.set(_real_time, "sleep", BoundedSleep(limit=0))
            out = o_gety(x, *a, **kw)
        steps.append({"step": "get_y_measurements", "x": x, "n_args": len(a), "kwargs": kw,
                      "returned": out, "objective.db": dump_db(OBJ_DB)})
        return out

    def log_rec(*a, **kw):
        steps.append({"step": "log_compositions", "n_args": len(a), "kwargs": sorted(kw)})
        return o_log(*a, **kw)

    comp_log = {"path": None, "buffer": [], "call": 0}
    with _Patches() as p:
        p.set(rzm, "OPTIMIZING_DIMS", list(inp["dims"]))
        p.set(rzm, "time", FrozenClock(inp["ts"]))
        p.set(rzm, "_COMP_LOG", comp_log)
        p.set(communication, "write_compositions", write_rec)
        p.set(communication, "reset_objective", reset_rec)
        p.set(rzm, "get_y_measurements", gety_rec)
        p.set(rzm, "log_compositions", log_rec)
        ref = {}
        ep = torch.tensor(inp["endpoints"], dtype=torch.float64, device="cpu")
        x, y = rzm.objective(ep, ready_for_objectives=True, endpoints_log_ref=ref)
    return {"steps": steps, "x": x, "y": y, "endpoints_log_ref": ref,
            "composition_log": comp_log}


def _foreign_db(path, table):
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE {table} (k INTEGER PRIMARY KEY, v TEXT)")
    conn.execute(f"INSERT INTO {table} (v) VALUES ('kept')")
    conn.commit()
    conn.close()


def run_primitive(inp):
    op = inp["op"]
    if op == "clear_in_flight_missing_db":
        communication.clear_in_flight_objective_state()
        return {"objective.db": dump_db(OBJ_DB), "sql_files": sorted(os.listdir("sql"))}
    if op == "reset_objective_without_table":
        conn = sqlite3.connect(OBJ_DB)
        conn.execute("CREATE TABLE handshake (id INTEGER PRIMARY KEY CHECK (id = 1), "
                     "new_objective_available INTEGER DEFAULT 0)")
        conn.execute("INSERT INTO handshake VALUES (1, 1)")
        conn.commit()
        conn.close()
        communication.reset_objective()
        return {"objective.db": dump_db(OBJ_DB)}
    initialize_db()
    fresh_receiver()
    receive([_line(inp["line"])])
    if op == "reset_objective":
        communication.reset_objective()
        return {"objective.db": dump_db(OBJ_DB)}
    if op == "clear_in_flight":
        communication.clear_in_flight_objective_state()
        return {"objective.db": dump_db(OBJ_DB)}
    if op == "launch_reset":
        w = inp["write"]
        write_pair(w["dims"], w["endpoints"], w["ts"])
        _foreign_db("sql/compositions4.db", "slots")
        _foreign_db("sql/objective_queue.db", "queue")
        Path("sql/serial_protocol.json").write_text('{"protocol": "lanes4"}')
        foreign = ("compositions4.db", "objective_queue.db", "serial_protocol.json")
        before = {f: _sha1(Path("sql") / f) for f in foreign}
        initialize_db()                               # scripts/main.py:191-193 (new trial)
        communication.reset_objective()
        communication.reset_compositions()
        after = {f: _sha1(Path("sql") / f) for f in foreign}
        return {"sql_files": sorted(os.listdir("sql")),
                "foreign_files_unchanged": before == after,
                "compositions.db": dump_db(COMP_DB), "objective.db": dump_db(OBJ_DB),
                "objective_memory.db": dump_db(MEM_DB)}
    raise ValueError(f"unknown primitive {op!r}")


def run_selftest(inp):
    port = SelftestPort(_line(inp["replies"]))
    ok = communication.serial_link_selftest(port, role="zombi", peer="disco", timeout=5.0,
                                            ping_period=0.5, grace=0.05)
    later = port.out[2:]
    return {"ok": ok, "first_two_frames": port.out[:2],
            "later_frames_are_pings": all(json.loads(f)["type"] == "selftest_ping" for f in later),
            "timeout_after": port.timeout, "reset_input_buffer_calls": port.reset_input_calls}


def run_orchestrator(inp, wait_s=20.0):
    """start_serial_dual_io_shared_port end to end on a LoopbackPort, as scripts/main.py
    starts it (defaults for the three DB paths), selftest off, stopped by parent_shutdown."""
    lines = [_line(s) for s in inp["lines"]]
    w = inp["write"]
    with in_dir("sync_reference"):                  # same lines, synchronous receiver
        initialize_db()
        fresh_receiver()
        receive(lines)
        sync_state = receiver_state()
    with in_dir("threads"), _Patches() as p:
        initialize_db()
        write_pair(w["dims"], w["endpoints"], w["ts"])
        fresh_receiver()
        p.set(LoopbackPort, "preload", b"".join(lines))
        p.set(LoopbackPort, "opened", [])
        p.set(serial, "Serial", LoopbackPort)
        stop = threading.Event()
        exit_codes, errors = [], []
        kw = dict(obj_hz=HZ, comp_hz=HZ, chaos=False, selftest=False, parent_shutdown=stop)
        # Phase 2 adds protocol= (DESIGN 6.3): replay explicitly as legacy once it exists.
        if "protocol" in inspect.signature(
                communication.start_serial_dual_io_shared_port).parameters:
            kw["protocol"] = "legacy"

        def target():
            try:
                communication.start_serial_dual_io_shared_port("COM_LOOPBACK", 115200, **kw)
            except SystemExit as e:                      # its finally ends with sys.exit(0)
                exit_codes.append(e.code)
            except BaseException as e:                   # pragma: no cover - reported below
                errors.append(repr(e))

        th = threading.Thread(target=target, daemon=True, name="legacy-serial-orchestrator")
        th.start()
        deadline = _real_time.time() + wait_s
        port = None
        while _real_time.time() < deadline:
            port = LoopbackPort.opened[0] if LoopbackPort.opened else None
            if (port is not None and port.pending_in() == 0
                    and communication._read_queue.empty() and len(port.frames) >= 3
                    and read_flag() == 1
                    and _count_rows(OBJ_DB, "objective") == inp["expect_rows"]):
                break
            _real_time.sleep(0.02)
        _real_time.sleep(0.3)        # let the receiver finish the trailing (ignored) lines
        stop.set()
        th.join(timeout=30.0)
        alive = th.is_alive()
        state = receiver_state()
        frames = list(port.frames) if port is not None else []
    communication._shutdown_flag.clear()
    return {"thread_finished": not alive, "exit_codes": exit_codes, "errors": errors,
            "serial_args": port.args if port is not None else None,
            "at_least_3_frames": len(frames) >= 3,
            "distinct_frames": sorted(set(frames)),
            "cleanup_calls": port.calls if port is not None else None,
            "port_closed": (port is not None and not port.is_open),
            "sync_receiver_equal": canon(state) == canon(sync_state),
            **state}


RUNNERS = {
    "writer": run_writer,
    "writer_overwrite": run_writer_overwrite,
    "sender_write_failure": run_sender_write_failure,
    "sender_no_frame": run_sender_no_frame,
    "receiver_rk": run_receiver_rk,
    "receiver_sequence": run_receiver_sequence,
    "hygiene": run_hygiene,
    "handshake": run_handshake,
    "primitive": run_primitive,
    "selftest": run_selftest,
    "orchestrator": run_orchestrator,
}


def run_case(case):
    with in_dir(case["name"]):
        return RUNNERS[case["kind"]](case["input"])


# ─────────────────────────────────────────────────────────────────────────────
# pytest: replay every golden case against today's code
# ─────────────────────────────────────────────────────────────────────────────

def _load(section):
    path = FIXDIR / f"{section}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


_GOLD = {s: _load(s) for s in SECTIONS}
_ALL = [(s, c) for s in SECTIONS for c in ((_GOLD[s] or {}).get("cases", []))]


def _case(section, name):
    for c in _GOLD[section]["cases"]:
        if c["name"] == name:
            return c
    raise KeyError(f"{section}/{name}")


@pytest.fixture(autouse=True)
def _isolated(tmp_path):
    """REVIEW F9: RNG states, random_simplex attributes and the legacy module globals are
    restored after every test; every file is written under tmp_path."""
    with legacy_sandbox(tmp_path):
        yield


def test_fixtures_present():
    missing = [s for s in SECTIONS if _GOLD[s] is None]
    assert not missing, (f"missing legacy golden fixtures {missing} in {FIXDIR}; they are "
                         f"captured once, before Phase 2 (see this module's docstring)")
    for s in SECTIONS:
        assert _GOLD[s]["_meta"]["schema"] == SCHEMA
        assert _GOLD[s]["cases"], s


@pytest.mark.parametrize("section,case", _ALL, ids=[f"{s}/{c['name']}" for s, c in _ALL])
def test_legacy_golden(section, case):
    got = run_case(case)
    assert_golden(got, case["expect"], f"{section}/{case['name']}")


# -- claims the goldens encode (also checked at capture time) ------------------------------

def test_golden_receiver_ignores_r_k_rail():
    for c in _GOLD["receiver"]["cases"]:
        if c["kind"] == "receiver_rk":
            assert canon(c["expect"]["plain"]) == canon(c["expect"]["with_rk"]), c["name"]


_INGESTED = {"pkt0", "nan_rows", "pkt1_with_r_k_rail_overwrites_unread_nan_rows",
             "zero_composition_rows", "no_comps_key", "empty_comps_list", "invalid_utf8_prefix",
             "no_ts_a", "no_ts_b", "pkt0_again_after_other_ts"}


def test_golden_receiver_ingest_pattern():
    """Which DiSCO lines the legacy receiver takes (flag raised) and which it drops: same-ts
    re-sends (even with r / k added), caps, selftest pings, blank / truncated / non-numeric /
    2-D / composition lines, the NULL notice, all-NaN packets and a value-identical re-send
    without ts."""
    seq = [c for c in _GOLD["receiver"]["cases"] if c["kind"] == "receiver_sequence"][0]
    flags = {s["step"]: s["flag"] for s in seq["expect"]["after_each_line"]}
    assert {k for k, v in flags.items() if v == 1} == _INGESTED
    assert {k for k, v in flags.items() if v != 1} == set(flags) - _INGESTED
    assert all(v in (0, 1) for v in flags.values())


def test_golden_second_write_replaces_every_table():
    c = _case("writer_sender", "overwrite_69af_then_96a9")
    last = _case("writer_sender", "pair_96a9_call0_6d")
    assert canon(c["expect"]["compositions.db"]) == canon(last["expect"]["compositions.db"])


def test_golden_resends_are_identical_and_unprefixed():
    for c in _GOLD["writer_sender"]["cases"]:
        if c["kind"] == "writer":
            f = c["expect"]["frames"]
            assert len(f) == 2 and f[0] == f[1], c["name"]
            assert f[0]["bytes"].endswith("\n") and not f[0]["bytes"].startswith("\n")
        if c["kind"] == "sender_write_failure":
            assert c["expect"]["failed"] == c["expect"]["frames"]


def test_golden_orchestrator_matches_sync_paths():
    c = _GOLD["orchestrator"]["cases"][0]
    e = c["expect"]
    assert e["thread_finished"] and e["exit_codes"] == [0] and not e["errors"]
    assert e["sync_receiver_equal"] is True and e["port_closed"] is True
    writer = _case("writer_sender", c["input"]["writer_case"])
    assert e["distinct_frames"] == [writer["expect"]["frames"][0]]


def test_golden_zero_composition_threshold():
    """Row 2 sums to 1e-13 (dropped as "no readback"), row 5 to 1e-6 (kept)."""
    idx = _case("hygiene", "near_zero_composition_rows")["expect"]["valid_indices"]["nd"]
    assert 2 not in idx and 5 in idx


def test_golden_hygiene_accept_empty_is_neutral_on_usable_packets():
    a = _case("hygiene", "pkt0_6d")["expect"]
    b = _case("hygiene", "pkt0_6d_accept_empty")["expect"]
    assert canon(a) == canon(b)


# ─────────────────────────────────────────────────────────────────────────────
# capture (script mode): inputs from the result_id fixtures, expectations from today's code
# ─────────────────────────────────────────────────────────────────────────────

def _dumps_line(pkt) -> str:
    """DiSCO objective_sender's frame: (json.dumps(pkt) + "\\n") (DiSCO communication.py:2936)."""
    return json.dumps(pkt) + "\n"


def build_inputs():
    lines = [json.loads(l) for l in (_RID / "composition_lines.jsonl").read_text().splitlines()]
    pk = [json.loads(l)["packet"] for l in
          (_RID / "objective_packets_96a9.jsonl").read_text().splitlines()]

    def rec(run, call):
        for r in lines:
            if r["run"] == run and r["call"] == call:
                return r
        raise KeyError((run, call))

    def ep(r, extra=None):
        return [r["main"], r["cache"]] + ([extra] if extra is not None else [])

    D3, D4, D6, D10 = [0, 8, 9], [0, 2, 8, 9], [0, 2, 3, 4, 8, 9], list(range(10))
    r9dff, r69af, r96a9 = rec("run_9dff", 3), rec("run_69af", 2), rec("run_96a9", 0)
    r96a9_last = [r for r in lines if r["run"] == "run_96a9"][-1]
    assert r9dff["dims"] == D4 and r69af["dims"] == D6 and r96a9["dims"] == D6
    rng = np.random.default_rng(20261113)
    ends10 = rng.dirichlet(np.ones(10), size=4).tolist()
    vertices = [[[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]], [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]]

    pair = {
        "pair_9dff_call3_4d": dict(dims=D4, endpoints=ep(r9dff), ts=T_CLOCK + 1.0),
        "pair_69af_call2_6d": dict(dims=D6, endpoints=ep(r69af), ts=T_CLOCK + 2.0),
        "pair_96a9_call0_6d": dict(dims=D6, endpoints=ep(r96a9), ts=T_CLOCK + 3.0),
        "pair_vertices_3d": dict(dims=D3, endpoints=vertices, ts=T_CLOCK + 4.0),
        "pair_dirichlet_10d": dict(dims=D10, endpoints=[ends10[:2], ends10[2:]],
                                   ts=T_CLOCK + 5.0),
    }

    def with_rk(p, r, k, rail=None):
        q = dict(p, r=r, k=k)
        if rail is not None:
            q["rail"] = rail
        return q

    def mod(p, ts=None, values=None, comps=None, drop=()):
        q = json.loads(json.dumps(p))
        if ts is not None:
            q["ts"] = ts
        if values is not None:
            q["values"] = values(q["values"])
        if comps is not None:
            q["comps"] = comps(q["comps"])
        for k in drop:
            q.pop(k, None)
        return q

    def set_none(idx):
        return lambda v: [None if j in idx else x for j, x in enumerate(v)]

    def zero_rows(idx):
        return lambda c: [[0.0] * 10 if j in idx else row for j, row in enumerate(c)]

    t0 = pk[0]["ts"]
    nan_rows = mod(pk[4], values=set_none({0, 5, 6}))
    zero_comp = mod(pk[5], comps=zero_rows({2, 7}))
    nan_and_zero = mod(pk[6], values=set_none({1, 9}), comps=zero_rows({1, 4}))
    all_zero = mod(pk[7], comps=lambda c: [[0.0] * 10 for _ in c])
    torn = mod(pk[8], comps=lambda c: c[:-3])
    no_comps = mod(pk[9], drop=("comps",))
    empty_comps = mod(pk[16], comps=lambda c: [])
    nan_comp_entry = mod(pk[11], comps=lambda c: [([None] + row[1:]) if j == 3 else row
                                                  for j, row in enumerate(c)])
    # _ZERO_COMP_TOL = 1e-12 (RZM 94): a row summing to 1e-13 is dropped, 1e-6 is kept
    near_zero = mod(pk[19], comps=lambda c: [([1e-13] + [0.0] * 9) if j == 2 else
                                             ([1e-6] + [0.0] * 9) if j == 5 else row
                                             for j, row in enumerate(c)])
    all_nan = {"type": "objective", "values": [None, None, None],
               "comps": pk[3]["comps"][:3], "ts": t0 + 2.0}
    null_notice = {"type": "objective", "values": [], "comps": [], "ts": t0 + 1.0,
                   "r": 17, "k": [17, None]}
    single = {"type": "objective", "values": [0.7],
              "comps": [[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.5]], "ts": t0 + 7.0}
    no_ts_a = {"type": "objective", "values": [0.61, 0.62, 0.63], "comps": pk[3]["comps"][:3]}
    no_ts_b = {"type": "objective", "values": [0.71, 0.72, 0.73], "comps": pk[3]["comps"][3:6]}
    caps = {"type": "caps", "v": 1, "slots": 4, "rid": 1, "boot": "1a2b3c4d",
            "k": [None, None], "t": t0 + 0.5}
    disco_ping = {"type": "selftest_ping", "from": "disco", "seq": 3, "pad": "x" * 64}
    comp_echo = {"type": "composition", "ts": 1.0, "start": [0.1] * 10, "end": [0.1] * 10,
                 "comps": [[0.1] * 10], "start_cache": [0.1] * 10, "end_cache": [0.1] * 10,
                 "compositions_cache": [[0.1] * 10]}
    L = _dumps_line

    writer_cases = [{"name": n, "kind": "writer", "input": v} for n, v in pair.items()]
    writer_cases += [
        {"name": "overwrite_69af_then_96a9", "kind": "writer_overwrite",
         "input": {"writes": [pair["pair_69af_call2_6d"], pair["pair_96a9_call0_6d"]]}},
        {"name": "write_failure_then_resend", "kind": "sender_write_failure",
         "input": pair["pair_9dff_call3_4d"]},
        {"name": "no_frame_after_initialize_db", "kind": "sender_no_frame",
         "input": {"listen_s": 0.3}},
        {"name": "no_frame_after_launch_reset", "kind": "sender_no_frame",
         "input": {"write": pair["pair_69af_call2_6d"], "then": "launch_reset", "listen_s": 0.3}},
        {"name": "no_frame_with_empty_cache_table", "kind": "sender_no_frame",
         "input": {"write": pair["pair_69af_call2_6d"], "then": "empty_cache_table",
                   "listen_s": 0.3}},
    ]

    receiver_cases = [
        {"name": "pkt0_r_k_rail", "kind": "receiver_rk",
         "input": {"plain": L(pk[0]), "with_rk": L(with_rk(pk[0], 17, [17, 16], "front"))}},
        {"name": "pkt1_r_k_rail", "kind": "receiver_rk",
         "input": {"plain": L(pk[1]), "with_rk": L(with_rk(pk[1], 18, [17, 18], "back"))}},
        {"name": "pkt2_r_k_null", "kind": "receiver_rk",
         "input": {"plain": L(pk[2]), "with_rk": L(with_rk(pk[2], 3, [None, 3]))}},
        {"name": "sequence", "kind": "receiver_sequence", "input": {"steps": [
            {"name": "pkt0", "line": L(pk[0])},
            {"name": "pkt0_resend_same_ts", "line": L(pk[0])},
            {"name": "pkt0_with_r_k_same_ts", "line": L(with_rk(pk[0], 17, [17, None]))},
            {"name": "caps_packet", "line": L(caps)},
            {"name": "disco_selftest_ping", "line": L(disco_ping)},
            {"name": "blank_line", "line": "\n"},
            {"name": "truncated_json", "line": '{"type": "objective", "values": [0.1, 0.2\n'},
            {"name": "null_notice_with_r", "line": L(null_notice)},
            {"name": "all_nan_values", "line": L(all_nan)},
            {"name": "values_not_numeric",
             "line": L({"type": "objective", "values": "n/a", "comps": [], "ts": t0 + 3.0})},
            {"name": "values_2d", "line": L({"type": "objective", "values": [[0.5, 0.6]],
                                             "comps": pk[3]["comps"][:1], "ts": t0 + 4.0})},
            {"name": "composition_packet_echo", "line": L(comp_echo)},
            {"name": "nan_rows", "line": L(nan_rows)},
            # nan_rows is still unread (not consumed): the one-slot mailbox is overwritten
            # (obj_empty never becomes False, COM 715-733)
            {"name": "pkt1_with_r_k_rail_overwrites_unread_nan_rows",
             "line": L(with_rk(pk[1], 18, [17, 18], "back")), "consume_before": False},
            {"name": "zero_composition_rows", "line": L(zero_comp)},
            {"name": "no_comps_key", "line": L(no_comps)},
            {"name": "empty_comps_list", "line": L(empty_comps)},
            {"name": "invalid_utf8_prefix", "line": "\xff\xfe" + L(pk[2])},
            {"name": "no_ts_a", "line": L(no_ts_a)},
            {"name": "no_ts_a_again", "line": L(no_ts_a)},
            {"name": "no_ts_b", "line": L(no_ts_b)},
            {"name": "pkt0_again_after_other_ts", "line": L(pk[0])},
        ]}},
    ]

    def hy(name, line, dims=D6, accept_empty=False):
        return {"name": name, "kind": "hygiene",
                "input": {"line": line, "dims": dims, "accept_empty": accept_empty}}

    hygiene_cases = [
        hy("pkt0_6d", L(pk[0])),
        hy("pkt0_6d_accept_empty", L(pk[0]), accept_empty=True),
        hy("pkt1_6d_with_r_k", L(with_rk(pk[1], 18, [17, 18], "back"))),
        hy("pkt2_6d", L(pk[2])),
        hy("pkt3_3d", L(pk[3]), dims=D3),
        hy("pkt10_10d", L(pk[10]), dims=D10),
        hy("nan_rows", L(nan_rows)),
        hy("zero_composition_rows", L(zero_comp)),
        hy("nan_and_zero_rows", L(nan_and_zero)),
        hy("all_zero_rows_accept_empty", L(all_zero), accept_empty=True),
        hy("torn_comps_shorter", L(torn)),
        hy("no_comps_accept_empty", L(no_comps), accept_empty=True),
        hy("single_row_3d", L(single), dims=D3),
        hy("nan_inside_a_composition", L(nan_comp_entry)),
        hy("near_zero_composition_rows", L(near_zero)),
        hy("all_nan_dropped_by_receiver", L(all_nan)),
        hy("null_notice_dropped_by_receiver", L(null_notice)),
    ]

    handshake_cases = [
        {"name": "regular_96a9_three_lines", "kind": "handshake",
         "input": dict(dims=D6, endpoints=ep(r96a9_last, r69af["main"]), ts=T_CLOCK + 11.0,
                       line=L(pk[12]))},
        {"name": "nan_and_zero_rows_69af", "kind": "handshake",
         "input": dict(dims=D6, endpoints=ep(r69af), ts=T_CLOCK + 12.0,
                       line=L(mod(pk[13], values=set_none({0, 3}), comps=zero_rows({3, 8}))))},
        {"name": "stale_result_before_call_9dff", "kind": "handshake",
         "input": dict(dims=D4, endpoints=ep(r9dff), ts=T_CLOCK + 13.0,
                       stale_line=L(pk[14]), line=L(pk[15]))},
        {"name": "vertices_3d", "kind": "handshake",
         "input": dict(dims=D3, endpoints=vertices, ts=T_CLOCK + 14.0, line=L(pk[16]))},
        {"name": "prim_reset_objective", "kind": "primitive",
         "input": {"op": "reset_objective", "line": L(pk[17])}},
        {"name": "prim_clear_in_flight", "kind": "primitive",
         "input": {"op": "clear_in_flight", "line": L(pk[17])}},
        {"name": "prim_clear_in_flight_missing_db", "kind": "primitive",
         "input": {"op": "clear_in_flight_missing_db"}},
        {"name": "prim_reset_objective_without_table", "kind": "primitive",
         "input": {"op": "reset_objective_without_table"}},
        {"name": "prim_launch_reset_new_trial", "kind": "primitive",
         "input": {"op": "launch_reset", "line": L(pk[18]),
                   "write": pair["pair_69af_call2_6d"]}},
    ]

    replies = (b"not json at all\n"
               + (json.dumps({"type": "selftest_ping", "from": "zombi", "seq": 99,
                              "pad": "z" * 8}) + "\n").encode()
               + (json.dumps({"type": "selftest_ping", "from": "disco", "seq": 7,
                              "pad": "d" * 48}) + "\n").encode()
               + (json.dumps({"type": "selftest_pong", "from": "disco", "ack": 1,
                              "pad": "p" * 48}) + "\n").encode())
    selftest_cases = [{"name": "ping_pong", "kind": "selftest",
                       "input": {"replies": replies.decode("latin-1")}}]

    orch_lines = [L(pk[0]), L(pk[0]), L(with_rk(pk[0], 17, [17, None])), L(caps),
                  L(disco_ping), "\n"]
    orchestrator_cases = [{"name": "loopback_threads", "kind": "orchestrator", "input": {
        "writer_case": "pair_69af_call2_6d", "write": pair["pair_69af_call2_6d"],
        "lines": orch_lines, "expect_rows": len(pk[0]["values"])}}]

    return {"writer_sender": writer_cases, "receiver": receiver_cases,
            "hygiene": hygiene_cases, "handshake": handshake_cases,
            "selftest": selftest_cases, "orchestrator": orchestrator_cases}


def _git(*args):
    try:
        r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=30)
        return r.stdout.strip()
    except Exception as e:                       # pragma: no cover
        return f"unavailable: {e!r}"


def _meta(section):
    return {
        "schema": SCHEMA, "section": section, "task": "lanes port T9 (DESIGN.md 4.4)",
        "captured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_head": _git("rev-parse", "HEAD"),
        "working_copy_status": _git("status", "--porcelain", "--", *_SRC_FILES).splitlines(),
        "working_copy_sha1": {f: _sha1(REPO / f) for f in _SRC_FILES},
        "inputs_from": [str(p.relative_to(REPO)).replace("\\", "/") for p in
                        (_RID / "composition_lines.jsonl", _RID / "objective_packets_96a9.jsonl")],
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "torch": torch.__version__, "sqlite": sqlite3.sqlite_version,
                     "pyserial": getattr(serial, "__version__", "?")},
        "replay": {"hz": HZ, "n_exp": N_EXP, "clock_base": T_CLOCK},
        "note": ("Provenance only; the tests never compare these hashes. Regenerate only to "
                 "accept a deliberate legacy change: python -B tests/test_lanes_legacy_bytes.py "
                 "--capture --force"),
    }


def _check_claims(gold):
    """The cross-section claims the pytest checks assert, verified before writing."""
    _GOLD.clear()
    _GOLD.update(gold)
    for fn in (test_golden_receiver_ignores_r_k_rail, test_golden_receiver_ingest_pattern,
               test_golden_second_write_replaces_every_table,
               test_golden_resends_are_identical_and_unprefixed,
               test_golden_orchestrator_matches_sync_paths, test_golden_zero_composition_threshold,
               test_golden_hygiene_accept_empty_is_neutral_on_usable_packets):
        fn()


def capture(force=False):
    existing = [s for s in SECTIONS if (FIXDIR / f"{s}.json").exists()]
    if existing and not force:
        raise SystemExit(f"refusing to overwrite {existing} in {FIXDIR} (pass --force)")
    inputs = build_inputs()
    gold = {}
    with tempfile.TemporaryDirectory(prefix="lanes_legacy_capture_") as td:
        for s in SECTIONS:
            cases = []
            for c in inputs[s]:
                results = []
                for rep in range(2):                     # determinism check
                    with legacy_sandbox(Path(td) / s / f"rep{rep}"):
                        results.append(json.loads(canon(run_case(c))))
                if canon(results[0]) != canon(results[1]):
                    raise SystemExit(f"{s}/{c['name']} is not deterministic: "
                                     f"{first_diff(results[0], results[1])}")
                cases.append({**c, "expect": results[0]})
                print(f"[capture] {s}/{c['name']}: ok")
            gold[s] = {"_meta": _meta(s), "cases": cases}
    _check_claims(gold)
    FIXDIR.mkdir(parents=True, exist_ok=True)
    for s in SECTIONS:
        text = json.dumps(plain(gold[s]), indent=1, allow_nan=True, sort_keys=False)
        (FIXDIR / f"{s}.json").write_text(text + "\n", encoding="utf-8")
        print(f"[capture] wrote {FIXDIR / (s + '.json')} ({len(text) // 1024} KB)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--capture", action="store_true", help="capture the golden fixtures")
    ap.add_argument("--force", action="store_true", help="overwrite existing fixtures")
    args = ap.parse_args(argv)
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="backslashreplace")
        except Exception:
            pass
    if not args.capture:
        ap.error("nothing to do: pass --capture (the replay runs under pytest)")
    capture(force=args.force)


if __name__ == "__main__":
    main()
