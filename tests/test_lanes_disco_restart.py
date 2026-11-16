"""DiSCO restart while ZoMBI keeps running, lanes mode (critic gap G2).

The signal: DiSCO's startup self-test ping {"type": "selftest_ping", "from": "disco", "seq", "pad"}
(DiSCO communication.py serial_link_selftest, run once per DiSCO start by dual_serial_zombi, after
DiSCO's startup wiped compositions.db). DiSCO's live link never pings. ZoMBI side:
scripts/communication.py DiscoRestartLink (receiver / sender / startup self-test), scripts/lanes_hw.py
LanesHardwareDriver._on_disco_restart, src/lanes/wire.py ResultRouter.on_disco_restart(issued_before=).

Checks (temp folders and in-memory ports only, no COM port; a fake engine except in the last test):
  * receiver: ONE 'disco_restart' link event per DiSCO session (a 0.5 s ping train is one episode, the
    startup self-test plus its live tail is one, a new start after a quiet gap is a new one); the
    objective queue (table q) is unchanged; without a link nothing is recorded (legacy);
  * sender (lanes branch only): answers the newest DiSCO ping with the startup self-test's pong bytes;
    holds the four-slot map until a map newer than the restart carries no pre-restart id; the legacy
    branch is byte-identical and never answers; a tick whose slots4 read fails (lock) while four-slot
    is active or not known to be off is skipped, never today's two-slot packet (review B1);
  * the serial child end to end (start_serial_dual_io_shared_port on an in-memory port);
  * driver: router.on_disco_restart for recipes issued before the restart; a lane whose main is unread
    gets its unread lines re-issued (same content, NEW ids), a lane whose main was read decides its next
    pair; one full map with a new ts; one console line; no previously issued id in the new map or in any
    later map; results of pre-restart slides routed to their own line ('late') or journaled and not
    ingested (unknown / duplicate), never attributed to another line; old events; resume cursor;
  * wire: on_disco_restart(issued_before=...) closes only older recipes; the default is unchanged;
  * end to end with the real engine and a fake four-slot DiSCO that restarts mid-run.

Run: PYTHONPATH=. python -B -m pytest -q -p no:cacheprovider tests/test_lanes_disco_restart.py
"""
from __future__ import annotations

import importlib.util
import json
import queue
import sqlite3
import sys
import threading
import time as _real_time
from pathlib import Path

import numpy as np
import pytest
import serial
import torch

import src.lanes
from scripts import communication, lanes_hw
from src.lanes import wire

REPO = Path(__file__).resolve().parents[1]
HZ = 50.0
Q_DB = "./sql/objective_queue.db"
COMP_DB = "./sql/compositions.db"
OBJ_DB = "./sql/objective.db"
MEM_DB = "./sql/objective_memory.db"
D = 3
OPT = [0, 1, 2]
M = 60.0


# ── fixtures / helpers ──────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _iso(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sql").mkdir()
    for g in ("_last_objective_ts", "_last_objective_vals", "_last_objective_comps",
              "_last_queued_ts", "_last_queued_raw"):
        monkeypatch.setattr(communication, g, None)
    communication._shutdown_flag.clear()
    yield
    communication._shutdown_flag.clear()


class Clock:
    """Settable clock. As lanes_hw.time it also needs sleep (journal retries only)."""

    def __init__(self, t):
        self.t = float(t)

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += float(s)


class TickClock(Clock):
    """Advances 0.5 s at every read: DiSCO's self-test ping period."""

    def time(self):
        t = self.t
        self.t += 0.5
        return t


def ping(seq, frm="disco"):
    return {"type": "selftest_ping", "from": frm, "seq": seq, "pad": "x" * 64}


def L(pkt) -> bytes:
    return (json.dumps(pkt) + "\n").encode()


def comp10(seed):
    v = np.random.default_rng(seed).dirichlet(np.ones(D))
    return [float(x) for x in np.pad(v, (0, 10 - D))]


def obj_line(ts, rid, vals, k=(1, 2)):
    return wire.encode_objective(vals, [comp10(i) for i in range(len(vals))], ts, rid=rid, last_read=tuple(k))


class DrainingQueue(queue.Queue):
    def get(self, block=True, timeout=None):
        try:
            return super().get(block=False)
        except queue.Empty:
            communication._shutdown_flag.set()
            raise


def receive(lines, monkeypatch, link):
    q = DrainingQueue()
    for ln in lines:
        q.put(ln)
    monkeypatch.setattr(communication, "_read_queue", q)
    communication._shutdown_flag.clear()
    guard = threading.Timer(30.0, communication._shutdown_flag.set)
    guard.daemon = True
    guard.start()
    try:
        communication.objective_receiver(HZ, OBJ_DB, MEM_DB, False, link=link)
    finally:
        guard.cancel()
        communication._shutdown_flag.clear()
    assert q.empty()


class Port:
    def __init__(self, stop_after=None):
        self.stop_after = stop_after
        self.frames = []

    def write(self, data):
        self.frames.append(bytes(data))
        if self.stop_after is not None and len(self.frames) >= self.stop_after:
            communication._shutdown_flag.set()
        return len(data)

    def flush(self):
        pass

    def reset_output_buffer(self):
        pass


def run_sender(link, n_frames=None, seconds=0.3):
    """composition_sender on COMP_DB until n_frames frames went out (or for `seconds` when None)."""
    port = Port(n_frames)
    communication._shutdown_flag.clear()
    guard = threading.Timer(10.0 if n_frames is not None else seconds, communication._shutdown_flag.set)
    guard.daemon = True
    guard.start()
    try:
        communication.composition_sender(port, HZ, COMP_DB, False, False, link=link)
    finally:
        guard.cancel()
        communication._shutdown_flag.clear()
    return port


def vec(seed):
    r = np.random.default_rng(seed).random(10)
    return [float(x) for x in np.round(r / r.sum(), 3)]


def enc(ts, entries):
    return wire.encode_slots(ts, {k: (r, a, b) for k, r, a, b in entries}, full=True)


def pong_bytes(ack):
    """What serial_link_selftest writes as its reply (scripts/communication.py, startup self-test)."""
    return (json.dumps({"type": "selftest_pong", "from": "zombi", "ack": ack, "pad": "x" * 3500}) + "\n").encode()


def kinds(frames):
    out = []
    for f in frames:
        p = json.loads(f.decode().strip())
        out.append("pong" if p.get("type") == "selftest_pong" else ("map" if "s" in p else p.get("type")))
    return out


# ── receiver: one event per DiSCO session ───────────────────────────────────
def test_receiver_records_one_restart_event_per_disco_session(monkeypatch):
    clk = TickClock(1.8e9)
    link = communication.DiscoRestartLink(queue_path=Q_DB, clock=clk.time)
    a, b, c = obj_line(100.5, 7, [0.1, 0.2]), obj_line(101.5, 8, [0.3]), obj_line(102.5, 9, [0.4])
    disco_pong = {"type": "selftest_pong", "from": "disco", "ack": 1, "pad": "p" * 16}
    # DiSCO session 1: its self-test ping train (0.5 s apart), mixed with ZoMBI's own ping (an echo),
    # a DiSCO pong and a result packet: ONE event
    receive([a, L(ping(1)), L(ping(2)), L(ping(1, "zombi")), L(disco_pong), b, L(ping(3)), L(ping(4))],
            monkeypatch, link)
    clk.t += 60.0                                   # DiSCO restarted again a minute later: a new session
    receive([c, L(ping(1)), L(ping(2))], monkeypatch, link)

    q = communication.read_objective_queue(0, path=Q_DB)
    assert [p["r"] for _, p in q] == [7, 8, 9]       # the objective queue is untouched by the events
    qid = {p["r"]: i for i, p in q}
    ev = communication.read_link_events(0, path=Q_DB)
    assert [(e["type"], e["src"], e["seq"], e["episode"]) for _, _, e in ev] == [
        ("disco_restart", "live", 1, 1), ("disco_restart", "live", 1, 2)]
    assert [after for _, after, _ in ev] == [qid[7], qid[9]]     # arrival order relative to q
    assert ev[0][2]["t_ping"] == 1.8e9 and ev[1][2]["t_ping"] > 1.8e9 + 60.0
    assert communication.max_link_event_id(Q_DB) == ev[-1][0]
    assert communication.read_link_events(ev[0][0], path=Q_DB) == ev[1:]
    assert link.take_pong()[0] == 2 and link.take_pong() is None   # newest live ping, answered once


def test_receiver_without_link_records_nothing(monkeypatch):
    receive([L(ping(1)), obj_line(5.5, 3, [0.5])], monkeypatch, None)
    assert communication.read_link_events(0, path=Q_DB) == []
    with sqlite3.connect(Q_DB) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert communication.LINK_EVENTS_TABLE not in names
    assert [p["r"] for _, p in communication.read_objective_queue(0, path=Q_DB)] == [3]


class SelftestPort:
    """Answers the first ping with a fixed DiSCO byte stream (as tests/test_lanes_legacy_bytes.py)."""

    def __init__(self, replies: bytes):
        self.timeout = 0.1
        self.out = []
        self._replies = replies
        self._in = bytearray()

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
        pass


def test_startup_selftest_ping_is_one_event_with_its_live_tail():
    replies = (L(ping(7)) + L({"type": "selftest_pong", "from": "disco", "ack": 1, "pad": "p" * 48})
               + L(ping(99, "zombi")))
    plain = SelftestPort(replies)
    assert communication.serial_link_selftest(plain, role="zombi", peer="disco", timeout=5.0,
                                              ping_period=0.5, grace=0.05)
    clk = Clock(1.8e9)
    link = communication.DiscoRestartLink(queue_path=Q_DB, clock=clk.time)
    seen = []

    def cb(p):
        seen.append(p)
        link.note_ping(p, "startup_selftest", defer=True)

    port = SelftestPort(replies)
    assert communication.serial_link_selftest(port, role="zombi", peer="disco", timeout=5.0,
                                              ping_period=0.5, grace=0.05, on_peer_ping=cb)
    assert port.out[:2] == plain.out[:2]             # the callback changes no byte of the self-test
    assert port.out[1] == pong_bytes(7)
    assert [p["seq"] for p in seen] == [7]           # only DiSCO's ping
    assert communication.read_link_events(0, path=Q_DB) == []      # nothing written inside the test
    assert link.take_pong() is None                  # the self-test answered it itself
    assert link.flush() == 1
    ev = communication.read_link_events(0, path=Q_DB)
    assert [(e["src"], e["seq"]) for _, _, e in ev] == [("startup_selftest", 7)]
    clk.t += 2.0                                     # DiSCO's 2 s grace tail reaches the live receiver
    link.note_ping(ping(9), "live")
    assert len(communication.read_link_events(0, path=Q_DB)) == 1
    assert link.take_pong()[0] == 9                  # still answered
    clk.t += 60.0                                    # a new DiSCO start
    link.note_ping(ping(1), "live")
    assert len(communication.read_link_events(0, path=Q_DB)) == 2


def test_event_kept_and_retried_when_the_queue_db_is_locked(monkeypatch):
    clk = Clock(1.8e9)
    link = communication.DiscoRestartLink(queue_path=Q_DB, clock=clk.time)
    real = communication.append_link_event
    calls = []

    def flaky(ev, path):
        calls.append(ev["seq"])
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        return real(ev, path)

    monkeypatch.setattr(communication, "append_link_event", flaky)
    link.note_ping(ping(4), "live")
    assert communication.read_link_events(0, path=Q_DB) == [] and len(link.unsaved) == 1
    clk.t += 0.5
    link.note_ping(ping(5), "live")                  # same episode: no second event, the first is retried
    ev = communication.read_link_events(0, path=Q_DB)
    assert [e["seq"] for _, _, e in ev] == [4] and link.unsaved == []


# ── sender: pong + hold (lanes) / unchanged (legacy) ────────────────────────
OLD = [(0, 11, vec(1), vec(2)), (1, 12, vec(3), vec(4)), (2, 13, vec(5), vec(6)), (3, 14, vec(7), vec(8))]


def test_sender_answers_and_holds_old_ids_until_the_reissued_map(capsys):
    ts_old = communication.write_slots4(OLD, ts=1000.0)
    clk = Clock(2000.0)
    link = communication.DiscoRestartLink(queue_path=Q_DB, clock=clk.time)
    assert run_sender(link, n_frames=1).frames == [enc(ts_old, OLD)]
    assert link.last_sent_rids == {11, 12, 13, 14}

    link.note_ping(ping(3), "live")                  # DiSCO restarted at t = 2000
    f = run_sender(link).frames
    assert f == [pong_bytes(3)]                      # answered once, byte-identical to the self-test reply
    assert f[0] == communication.selftest_pong_frame(3)
    # a map that still carries a pre-restart id, or that is older than the restart: never sent
    communication.write_slots4([(0, 21, vec(9), vec(10)), (1, 12, vec(3), vec(4))], ts=2500.0)
    assert run_sender(link).frames == []
    communication.write_slots4([(0, 21, vec(9), vec(10))], ts=1500.0)
    assert run_sender(link).frames == []
    clk.t = 2031.0                                   # still held 31 s later: one warning line
    assert run_sender(link).frames == []
    assert "four-slot map HELD" in capsys.readouterr().out
    # the driver's re-issued map (new ids, new ts) goes out and ends the hold
    new = [(1, 22, vec(3), vec(4)), (2, 23, vec(5), vec(6)), (3, 24, vec(7), vec(8))]
    ts_new = communication.write_slots4(new, ts=2600.0)
    assert run_sender(link, n_frames=2).frames == [enc(ts_new, new)] * 2
    assert link.hold is None and "released" in capsys.readouterr().out
    # a new DiSCO session: its hold blocks exactly the ids of the last map sent before it
    clk.t = 4000.0
    link.note_ping(ping(1), "live")
    assert run_sender(link).frames == [pong_bytes(1)]
    assert link.hold["blocked"] == {22, 23, 24}


def test_sender_legacy_branch_unchanged_and_silent_with_a_pending_ping():
    a = np.array(vec(11)); b = np.array(vec(12)); grid = np.linspace(a, b, 24)
    communication.write_compositions(a, b, grid, 1793900000.25, b, a, grid[::-1].copy(), db_path=COMP_DB)
    plain = run_sender(None, n_frames=2).frames
    link = communication.DiscoRestartLink(queue_path=Q_DB, clock=Clock(2000.0).time)
    link.note_ping(ping(3), "live")
    assert run_sender(link, n_frames=2).frames == plain          # no pong, same legacy bytes
    assert link.take_pong() is not None                          # never consumed by the legacy branch
    communication.clear_slots4(COMP_DB)                          # lanes tables present but inactive
    link.note_ping(ping(4), "live")
    assert run_sender(link, n_frames=2).frames == plain


# ── sender: compositions.db unreadable while four-slot is active (review B1) ─────────────────
class _ShortTimeoutSqlite:
    """Stand-in for communication's sqlite3 module: every connect() timeout is divided by 20, so the
    sender's 10 s sqlite timeouts become 0.5 s and a real EXCLUSIVE lock of 0.7 s reproduces the
    review's 12 s lock (> _slots4_state's timeout, < that plus read_all's) in under a second."""

    def __getattr__(self, name):
        return getattr(sqlite3, name)

    @staticmethod
    def connect(path, timeout=5.0, **kw):
        return sqlite3.connect(path, timeout=timeout / 20.0, **kw)


def _lock_comp_db(seconds):
    """EXCLUSIVE-lock COMP_DB for `seconds` from a thread (a writer, or Dropbox); returns once locked."""
    locked = threading.Event()

    def hold():
        h = sqlite3.connect(COMP_DB, timeout=5.0, isolation_level=None)
        try:
            h.execute("BEGIN EXCLUSIVE")
            locked.set()
            _real_time.sleep(seconds)
            h.execute("COMMIT")
        finally:
            h.close()

    th = threading.Thread(target=hold, daemon=True)
    th.start()
    assert locked.wait(5)
    return th


def _init_proposal():
    """The last axial-init (legacy two-slot) proposal: it stays in compositions.db for a whole new run."""
    a = np.array(vec(41)); b = np.array(vec(42)); grid = np.linspace(a, b, 5)
    communication.write_compositions(a, b, grid, 123.0, b, a, grid[::-1].copy(), db_path=COMP_DB)


def _start_sender(port, link):
    th = threading.Thread(target=communication.composition_sender, args=(port, HZ, COMP_DB, False, False),
                          kwargs=dict(link=link), daemon=True)
    th.start()
    return th


def _stop_sender(th):
    communication._shutdown_flag.set()
    th.join(timeout=10.0)
    communication._shutdown_flag.clear()
    assert not th.is_alive()


def test_slots4_state_tells_inactive_from_unreadable(monkeypatch):
    S = communication
    assert S._slots4_state(COMP_DB) == (S.SLOTS4_INACTIVE, None)            # no compositions.db
    _init_proposal()
    assert S._slots4_state(COMP_DB) == (S.SLOTS4_INACTIVE, None)            # legacy: no slots4_meta
    ts = S.write_slots4(OLD, ts=1000.0)
    state, payload = S._slots4_state(COMP_DB)
    assert state == S.SLOTS4_ACTIVE and payload == S._read_slots4(COMP_DB) and payload[0] == ts
    assert sorted(payload[1]) == [0, 1, 2, 3] and payload[1][2][0] == 13
    S.clear_slots4(COMP_DB)
    assert S._slots4_state(COMP_DB) == (S.SLOTS4_INACTIVE, None)            # active = 0
    S.write_slots4(OLD, ts=1001.0)
    with sqlite3.connect(COMP_DB) as c:                                      # meta readable, a row is not
        c.execute("UPDATE slots4 SET start = '{' WHERE slot = 1")
    assert S._slots4_state(COMP_DB) == (S.SLOTS4_UNREADABLE, True)
    assert S._read_slots4(COMP_DB) is None                                   # the old API is unchanged
    S.write_slots4(OLD, ts=1002.0)
    monkeypatch.setattr(S, "sqlite3", _ShortTimeoutSqlite())
    lk = _lock_comp_db(1.0)
    assert S._slots4_state(COMP_DB) == (S.SLOTS4_UNREADABLE, None)          # the mode itself is unreadable
    lk.join()
    assert S._slots4_state(COMP_DB)[0] == S.SLOTS4_ACTIVE


def test_sender_never_falls_back_to_the_legacy_packet_while_four_slot_is_active(monkeypatch, capsys):
    """Review B1 (scripts/communication.py composition_sender): a failed slots4 read used to look like
    'four-slot inactive', so the sender sent the stale two-slot init proposal; it bypassed the G2 hold
    and, on DiSCO, re-armed the legacy slots. Now such a tick is skipped."""
    monkeypatch.setattr(communication, "sqlite3", _ShortTimeoutSqlite())
    _init_proposal()
    ts_old = communication.write_slots4(OLD, ts=1000.0)

    # (a) normal lanes operation: the lock comes while the sender runs (last readable read: active)
    port = Port()
    th = _start_sender(port, None)
    try:
        assert _wait(lambda: len(port.frames) >= 3)
        _lock_comp_db(0.7).join()
        n = len(port.frames)
        assert _wait(lambda: len(port.frames) >= n + 3)                      # four-slot resumes
    finally:
        _stop_sender(th)
    assert "slots4 read error" in capsys.readouterr().out                    # the lock really bit
    assert set(port.frames) == {enc(ts_old, OLD)}, kinds(port.frames)

    # (b) the review's case: hold active (DiSCO restarted), the sender (re)starts into the lock
    clk = Clock(2000.0)
    link = communication.DiscoRestartLink(queue_path=Q_DB, clock=clk.time)
    link.note_map_sent({k: (r, a, b) for k, r, a, b in OLD})
    link.note_ping(ping(7), "live")
    lk = _lock_comp_db(0.7)
    port = run_sender(link, seconds=1.5)
    lk.join()
    assert "slots4 read error" in capsys.readouterr().out
    assert port.frames == [pong_bytes(7)], kinds(port.frames)                # answered; no map, no legacy

    # (c) meta says active but the map cannot be read: skipped too, and DiSCO's ping is still answered
    with sqlite3.connect(COMP_DB) as c:
        c.execute("UPDATE slots4 SET start = '{' WHERE slot = 1")
    link.note_ping(ping(8), "live")
    assert run_sender(link, seconds=0.3).frames == [pong_bytes(8)]
    assert run_sender(None, seconds=0.3).frames == []


def test_sender_legacy_branch_unchanged_under_a_lock(monkeypatch):
    """Four-slot known off (legacy session): an unreadable tick still runs today's path."""
    monkeypatch.setattr(communication, "sqlite3", _ShortTimeoutSqlite())
    _init_proposal()
    plain = run_sender(None, n_frames=1).frames[0]
    link = communication.DiscoRestartLink(queue_path=Q_DB, clock=Clock(2000.0).time)
    link.note_ping(ping(3), "live")
    port = Port()
    th = _start_sender(port, link)
    try:
        assert _wait(lambda: len(port.frames) >= 3)
        _lock_comp_db(0.7).join()
        n = len(port.frames)
        assert _wait(lambda: len(port.frames) >= n + 3)
    finally:
        _stop_sender(th)
    assert set(port.frames) == {plain}                                       # no pong, same legacy bytes
    assert link.take_pong() is not None


# ── serial child end to end (in-memory port, no COM port) ───────────────────
class FeedPort:
    """serial.Serial stand-in: bytes fed by the test are read in chunks; writes are recorded."""

    opened = []

    def __init__(self, port, baudrate, timeout=None, write_timeout=None, **kw):
        self.timeout = timeout
        self.is_open = True
        self._in = bytearray()
        self._lock = threading.Lock()
        self.frames = []
        type(self).opened.append(self)

    def feed(self, data):
        with self._lock:
            self._in += data

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
        pass

    def reset_output_buffer(self):
        pass

    def cancel_read(self):
        pass

    def close(self):
        self.is_open = False


def _wait(cond, s=10.0):
    t_end = _real_time.time() + s
    while _real_time.time() < t_end:
        if cond():
            return True
        _real_time.sleep(0.01)
    return False


def test_serial_child_end_to_end_answers_holds_and_releases(monkeypatch):
    monkeypatch.setattr(FeedPort, "opened", [])
    monkeypatch.setattr(serial, "Serial", FeedPort)
    ts_old = communication.write_slots4(OLD)
    stop = threading.Event()
    exits = []

    def target():
        try:
            communication.start_serial_dual_io_shared_port("COM_TEST", 115200, obj_hz=HZ, comp_hz=HZ,
                                                          selftest=False, parent_shutdown=stop)
        except SystemExit as e:
            exits.append(e.code)

    th = threading.Thread(target=target, daemon=True)
    th.start()
    try:
        assert _wait(lambda: FeedPort.opened and enc(ts_old, OLD) in FeedPort.opened[0].frames)
        port = FeedPort.opened[0]
        port.feed(L(ping(5)) + L(ping(6)))           # DiSCO restarted: its self-test pings us
        assert _wait(lambda: "pong" in kinds(list(port.frames)))
        _real_time.sleep(0.3)
        frames = list(port.frames)
        i_pong = kinds(frames).index("pong")
        assert "map" not in kinds(frames[i_pong:])   # held from the restart on
        ev = communication.read_link_events(0, path=Q_DB)
        assert [(e["src"], e["seq"]) for _, _, e in ev] == [("live", 5)]
        new = [(0, 31, vec(1), vec(2)), (1, 32, vec(3), vec(4)), (2, 33, vec(5), vec(6)), (3, 34, vec(7), vec(8))]
        ts_new = communication.write_slots4(new)     # what the lanes driver writes after re-issuing
        assert ts_new > ev[0][2]["t_ping"]
        assert _wait(lambda: enc(ts_new, new) in port.frames)
        after = list(port.frames)[i_pong:]
        for f in after:
            p = json.loads(f.decode().strip())
            if "s" in p:
                assert not {e[1] for e in p["s"]} & {11, 12, 13, 14}, p["s"]
    finally:
        stop.set()
        th.join(timeout=30.0)
        communication._shutdown_flag.clear()
    assert not th.is_alive()


# ── driver with a fake engine ───────────────────────────────────────────────
class FakeLine:
    def __init__(self, rid, lane, act, zoom, ends):
        self.ends = np.asarray(ends, float)
        self.tags = dict(rid=rid, lane=lane, act=act, zoom=zoom, kind="lane", bridge=False)

    @property
    def rid(self):
        return self.tags["rid"]


class FakeLane:
    def __init__(self, eng, name):
        self.eng, self.name, self.code = eng, name, name[0].upper()
        self.act, self.zoom, self.n = 0, 0, 0
        self.decisions, self.picked, self.results = [], [], []

    def _line(self):
        rid = f"{self.code}{self.n}"
        rng = np.random.default_rng(self.n + (0 if self.name == "front" else 1000))
        self.n += 1
        ln = FakeLine(rid, self.name, self.act, self.zoom, [rng.dirichlet(np.ones(D)), rng.dirichlet(np.ones(D))])
        self.eng.lines[rid] = ln
        return ln

    def decide(self, now=0.0):
        m, b = self._line(), self._line()
        self.decisions.append(dict(t=now, main=m.rid, backup=b.rid))
        return m, b, {"main": dict(m.tags), "backup": dict(b.tags)}

    def refresh(self, now=0.0):
        return None

    def on_pickup(self, rid, now=None):
        self.picked.append(rid)

    def on_result(self, p, tags=None, now=None):
        self.eng.ingest(p, tags)
        self.results.append(None if tags is None else tags.get("rid"))


class FakeDH:
    def __init__(self):
        self.X_all_actual = torch.zeros(0, D, dtype=torch.float64)
        self.X_all_expected = torch.zeros(0, D, dtype=torch.float64)
        self.Y_all = torch.zeros(0, 1, dtype=torch.float64)


class FakeEngine:
    def __init__(self):
        self.lines = {}
        self.lanes = {n: FakeLane(self, n) for n in ("front", "back")}
        self.dh = FakeDH()
        self.packets, self.events, self.zones = [], [], []

    def tags_of(self, rid):
        ln = self.lines.get(rid)
        return None if ln is None else dict(ln.tags)

    def ingest(self, p, tags):
        if p.get("_ingested"):                       # both lanes see a packet; ingested once
            return
        p["_ingested"] = True
        x = torch.as_tensor(p["x"], dtype=torch.float64).reshape(-1, D)
        y = torch.as_tensor(p["y"], dtype=torch.float64).reshape(-1, 1)
        dh = self.dh
        dh.X_all_actual = torch.cat([dh.X_all_actual, x])
        dh.X_all_expected = torch.cat([dh.X_all_expected, x])
        dh.Y_all = torch.cat([dh.Y_all, y])
        t = tags or {}
        self.packets.append(dict(pid=len(self.packets), cls="own" if tags else "shared", n=int(x.shape[0]),
                                 lane=t.get("lane"), act=t.get("act"), zoom=t.get("zoom"), rid=t.get("rid")))

    def close(self):
        pass


def make_driver(tmp_path, monkeypatch, clk, run="run_g2"):
    monkeypatch.setattr(lanes_hw, "time", clk)
    eng = FakeEngine()
    monkeypatch.setattr(src.lanes, "build_lane_engine", lambda *a, **k: eng)
    rng = np.random.default_rng(0)
    Xa = torch.tensor(rng.dirichlet(np.ones(D), 4), dtype=torch.float64)
    Y = torch.tensor(rng.random((4, 1)), dtype=torch.float64)
    logs = []
    drv = lanes_hw.LanesHardwareDriver(tmp_path / "runs" / run, Xa, Xa.clone(), Y,
                                       np.stack([np.zeros(D), np.ones(D)]), {}, preset="leader",
                                       log=logs.append, opt_dims=OPT, y_to_optimizer=lambda v: v)
    return drv, eng, logs


def spy_writes(monkeypatch, clk):
    """write_slots4 with ts on the test clock (production: both processes read the same wall clock, and
    a write always comes strictly after the ping it reacts to; the fake clock does not move by itself)."""
    real = communication.write_slots4
    writes = []

    def spy(slots, ts=None, path=COMP_DB):
        t = max(clk.t + 0.001, writes[-1][0] + 0.001) if writes else clk.t + 0.001
        writes.append((t, [tuple(s) for s in slots]))
        return real(slots, ts=t, path=path)

    monkeypatch.setattr(communication, "write_slots4", spy)
    return writes


def queue_pkt(pkt):
    assert communication._queue_objective_packet(json.dumps(pkt), pkt, OBJ_DB)


def ack_pkt(ts, k):
    """A result without "r" (e.g. a calibration coupon) still carries DiSCO's read acks."""
    return json.loads(wire.encode_objective([0.5], [comp10(99)], ts, rid=None, last_read=k))


def result_pkt(rid, ts, seed, k=(None, None), n=4):
    rng = np.random.default_rng(seed)
    vals = [float(v) for v in rng.random(n)]
    return json.loads(wire.encode_objective(vals, [comp10(seed * 10 + i) for i in range(n)], ts, rid=rid, last_read=k))


def journal(drv):
    return lanes_hw._read_jsonl(drv.dir / "journal.jsonl")


T0 = 1.8e9


def _restart_scenario(tmp_path, monkeypatch, read_front_backup=False):
    """start -> the front rail reads its main (and optionally its backup) -> DiSCO restarts."""
    clk = Clock(T0)
    writes = spy_writes(monkeypatch, clk)
    drv, eng, logs = make_driver(tmp_path, monkeypatch, clk)
    drv.start()
    old = {k: drv.slots[k] for k in range(4)}
    link = communication.DiscoRestartLink(queue_path=Q_DB, clock=lambda: clk.t)
    ts_before, map_before = communication._read_slots4(COMP_DB)
    link.note_map_sent(map_before)                    # the serial child put this map on the wire
    clk.t = T0 + 10 * M
    queue_pkt(ack_pkt(clk.t, (old[0].wid, None)))
    drv.poll_once()
    assert drv.slots[0].state == "read" and drv.slots[1].state == "unread"
    if read_front_backup:                             # the backup was read too: front replenished
        clk.t = T0 + 11 * M
        queue_pkt(ack_pkt(clk.t, (old[1].wid, None)))
        drv.poll_once()
        assert drv.slots[0].wid not in {o.wid for o in old.values()}
    pre = {k: drv.slots[k] for k in range(4)}
    clk.t = T0 + 20 * M
    link.note_ping(ping(1), "live")                   # DiSCO restarted: its first self-test ping
    n_logs, n_writes = len(logs), len(writes)
    drv.poll_once()
    return dict(clk=clk, drv=drv, eng=eng, logs=logs, writes=writes, old=old, pre=pre, link=link,
                ts_before=ts_before, map_before=map_before, restart_logs=logs[n_logs:],
                restart_writes=writes[n_writes:], t_ping=T0 + 20 * M)


def test_driver_restart_reissues_new_ids_in_one_full_map(tmp_path, monkeypatch):
    S = _restart_scenario(tmp_path, monkeypatch)
    drv, old, pre = S["drv"], S["old"], S["pre"]
    old_ids = {s.wid for s in old.values()}
    # one console line, one full map (written once, new ts)
    assert len(S["restart_logs"]) == 1 and S["restart_logs"][0].startswith("[lanes] DiSCO restarted"), S["restart_logs"]
    assert len(S["restart_writes"]) == 1
    ts_after, map_after = communication._read_slots4(COMP_DB)
    assert ts_after > S["ts_before"] and ts_after > S["t_ping"]
    assert sorted(map_after) == [0, 1, 2, 3]
    new_ids = {v[0] for v in map_after.values()}
    assert not new_ids & old_ids                      # no previously issued id in the new map
    # back: main unread -> both lines kept, same content, new ids
    for k in (2, 3):
        assert drv.slots[k].line is pre[k].line and drv.slots[k].wid != pre[k].wid
        assert map_after[k][1] == pre[k].start and map_after[k][2] == pre[k].end
    # front: main read -> the lane decided its next pair (the unread backup superseded, never printed)
    assert drv.slots[0].line.rid not in {pre[0].line.rid, pre[1].line.rid}
    assert drv.slots[1].line.rid not in {pre[0].line.rid, pre[1].line.rid}
    assert S["eng"].lanes["front"].decisions[-1]["main"] == drv.slots[0].line.rid
    # the router: pre-restart recipes closed, the new ones open
    st = {rid: r.status for rid, r in drv.router.ids.by_id.items()}
    assert st[old[0].wid] == "lost"
    assert all(st[old[k].wid] == "unprinted" for k in (1, 2, 3))
    assert all(st[i] == "issued" for i in new_ids)
    assert drv.router.last_read == {"front": None, "back": None}
    # journal: the restart is durable before anything is acted on
    jr = journal(drv)
    ev = [r["ev"] for r in jr]
    i_r, i_d = ev.index("disco_restart"), ev.index("disco_restart_done")
    i_w = [i for i, r in enumerate(jr) if r["ev"] == "slot_write" and r["reason"] == "disco_restart"]
    assert len(i_w) == 1 and i_r < i_w[0] < i_d
    done = jr[i_d]
    assert done["lost"] == [old[0].wid] and done["decided"] == ["front"]
    assert {int(a): b for a, b in done["reissued"].items()} == {pre[2].wid: drv.slots[2].wid,
                                                                pre[3].wid: drv.slots[3].wid}
    reiss = [r for r in jr if r["ev"] == "issue" and r.get("kind") == "restart_reissue"]
    assert {r["supersedes"] for r in reiss} == {pre[2].wid, pre[3].wid}
    assert [r["kind"] for r in jr if r["ev"] == "decision"][-1] == "restart"
    # the serial-side hold releases on exactly this map, never on the old one
    link = S["link"]
    assert link.map_allowed(S["ts_before"], S["map_before"]) is False
    assert link.map_allowed(ts_after, map_after) is True


def test_driver_restart_keeps_unread_plan_of_both_lanes(tmp_path, monkeypatch):
    clk = Clock(T0)
    spy_writes(monkeypatch, clk)
    drv, eng, logs = make_driver(tmp_path, monkeypatch, clk)
    drv.start()
    pre = {k: drv.slots[k] for k in range(4)}
    clk.t = T0 + 5 * M
    communication.DiscoRestartLink(queue_path=Q_DB, clock=lambda: clk.t).note_ping(ping(2), "live")
    n0 = sum(len(ln.decisions) for ln in eng.lanes.values())
    drv.poll_once()
    assert sum(len(ln.decisions) for ln in eng.lanes.values()) == n0      # no decision needed
    for k in range(4):
        assert drv.slots[k].line is pre[k].line and drv.slots[k].wid not in {p.wid for p in pre.values()}
        assert drv.slots[k].start == pre[k].start and drv.slots[k].end == pre[k].end
        assert drv.router.ids.by_id[drv.slots[k].wid].line["reissue_of"] == pre[k].wid
        assert drv.router.ids.by_id[drv.slots[k].wid].line["rid"] == pre[k].line.rid
    assert len([x for x in logs if x.startswith("[lanes] DiSCO restarted")]) == 1


def test_pre_restart_results_route_to_their_own_line_or_are_journaled(tmp_path, monkeypatch):
    S = _restart_scenario(tmp_path, monkeypatch)
    drv, eng, old, clk = S["drv"], S["eng"], S["old"], S["clk"]
    old_ids = {s.wid for s in old.values()}
    reissued_back_main = drv.slots[2]
    clk.t = T0 + 75 * M
    p_front = result_pkt(old[0].wid, clk.t, 1)                   # printed by the old session, read before
    p_back = result_pkt(old[2].wid, clk.t + 1, 2)                # ZoMBI saw it unread; the old session printed it
    p_unknown = result_pkt(777, clk.t + 2, 3)                    # never issued by this run
    p_dup = dict(p_front, ts=clk.t + 3)                          # the same slide delivered again
    for p in (p_front, p_back, p_unknown, p_dup):
        queue_pkt(p)
    drv.poll_once()
    jr = journal(drv)
    q = {p["ts"]: i for i, p in communication.read_objective_queue(0, path=Q_DB)}
    routes = {r["qid"]: r for r in jr if r["ev"] == "route"}
    ingested = {r["qid"] for r in jr if r["ev"] == "ingested"}
    issue = {}
    for r in jr:
        if r["ev"] == "issue":
            issue[r["wid"]] = r
    rf, rb = routes[q[p_front["ts"]]], routes[q[p_back["ts"]]]
    assert (rf["cls"], rf["wid"], rf["lane"], rf["engine_rid"]) == ("late", old[0].wid, "front", old[0].line.rid)
    assert (rb["cls"], rb["wid"], rb["lane"], rb["engine_rid"]) == ("late", old[2].wid, "back", old[2].line.rid)
    assert rb["engine_rid"] == reissued_back_main.line.rid      # the same line as its re-issue, nothing else
    assert q[p_front["ts"]] in ingested and q[p_back["ts"]] in ingested
    assert [p["rid"] for p in eng.packets] == [old[0].line.rid, old[2].line.rid]
    for qid in (q[p_unknown["ts"]], q[p_dup["ts"]]):
        assert routes[qid]["disposition"] == wire.LOG and qid not in ingested
    assert routes[q[p_unknown["ts"]]]["cls"] == "unknown" and routes[q[p_dup["ts"]]]["cls"] == "duplicate"
    # every ingested result went to the line its own id was issued for
    for qid in ingested:
        r = routes[qid]
        assert r["engine_rid"] == issue[r["wid"]]["engine_rid"] and r["lane"] == issue[r["wid"]]["lane"]
    # the new session's own result for the re-issued line: matched, same line
    clk.t = T0 + 100 * M
    p_new = result_pkt(reissued_back_main.wid, clk.t, 4, k=(None, reissued_back_main.wid))
    queue_pkt(p_new)
    drv.poll_once()
    rn = {r["qid"]: r for r in journal(drv) if r["ev"] == "route"}[
        {p["ts"]: i for i, p in communication.read_objective_queue(0, path=Q_DB)}[p_new["ts"]]]
    assert (rn["cls"], rn["engine_rid"]) == ("matched", old[2].line.rid)
    # no pre-restart id was ever written again
    t_restart = [r for r in journal(drv) if r["ev"] == "disco_restart"][0]["t"]
    for r in journal(drv):
        if r["ev"] == "slot_write" and r["t"] >= t_restart:
            assert not {s[1] for s in r["slots"]} & old_ids, r


def test_old_id_never_reenters_the_slots(tmp_path, monkeypatch):
    S = _restart_scenario(tmp_path, monkeypatch)
    drv, old, clk = S["drv"], S["old"], S["clk"]
    old_ids = {s.wid for s in old.values()}
    # the allocator wraps onto the old ids: they stay blocked by the reuse horizon
    drv.router.ids.next = min(old_ids)
    for i in range(6):
        clk.t += M
        drv._decide("front" if i % 2 else "back", "regular")
        drv._write_full("test")
        ids_now = {v[0] for v in communication._read_slots4(COMP_DB)[1].values()}
        assert not ids_now & old_ids
    for w in old_ids:
        assert not drv.router.ids._free(w, clk.t)
    # a retired slot object put back by mistake is withdrawn, not re-sent
    retired = old[3]
    assert retired.retired
    drv.slots[3] = retired
    drv._write_full("bug")
    assert 3 not in communication._read_slots4(COMP_DB)[1]
    assert [r for r in journal(drv) if r["ev"] == "retired_slot_dropped"][0]["wid"] == retired.wid
    assert any("CRITICAL" in x for x in S["logs"])


def test_restart_with_both_front_slots_read_after_replenish(tmp_path, monkeypatch):
    """The front rail read main and backup (-> replenished pair, unread): its new pair is kept."""
    S = _restart_scenario(tmp_path, monkeypatch, read_front_backup=True)
    drv, pre = S["drv"], S["pre"]
    for k in range(4):
        assert drv.slots[k].line is pre[k].line and drv.slots[k].wid != pre[k].wid
    st = {rid: r.status for rid, r in drv.router.ids.by_id.items()}
    assert st[S["old"][0].wid] == "lost" and st[S["old"][1].wid] == "lost"


def test_old_events_new_run_resume_and_joint_restart(tmp_path, monkeypatch):
    clk = Clock(T0)
    spy_writes(monkeypatch, clk)
    # an event from before this run (e.g. an earlier DiSCO restart): a new run skips it
    communication.DiscoRestartLink(queue_path=Q_DB, clock=lambda: T0 - 9000.0).note_ping(ping(1), "live")
    drv, eng, logs = make_driver(tmp_path, monkeypatch, clk)
    assert drv.after_event_id == 1
    # a recipe left open by a previous session of this run (read before the restart below)
    clk.t = T0 - 3000.0
    w_prev = drv.router.try_issue(0, wire.make_line("lane", lane="front", rid="F_prev", role="main", act=0, zoom=0),
                                  clk.t)
    drv.router.on_read_ack((w_prev, None), clk.t + 60)
    # DiSCO and ZoMBI restarted together: the startup self-test saw DiSCO's ping before our first write
    communication.DiscoRestartLink(queue_path=Q_DB, clock=lambda: T0 - 100.0).note_ping(ping(1), "startup_selftest")
    clk.t = T0
    drv.start()
    before = {k: drv.slots[k].wid for k in range(4)}
    n_logs = len(logs)
    drv.poll_once()
    assert {k: drv.slots[k].wid for k in range(4)} == before      # nothing predates it: nothing re-issued
    jr = journal(drv)
    assert [r["eid"] for r in jr if r["ev"] == "disco_restart"] == [2]
    assert not [r for r in jr if r["ev"] == "slot_write" and r["reason"] == "disco_restart"]
    assert drv.router.ids.by_id[w_prev].status == "lost"            # the previous session's read: closed
    assert all(drv.router.ids.by_id[w].status == "issued" for w in before.values())
    assert len(logs) - n_logs == 1 and "nothing re-issued" in logs[-1]
    # resume: the handled event is not handled again
    drv2, _, _ = make_driver(tmp_path, monkeypatch, clk)
    assert drv2.resumed
    drv2.resume_cursor()
    assert drv2.after_event_id == 2
    drv2.start()
    drv2.poll_once()
    assert len([r for r in journal(drv2) if r["ev"] == "disco_restart"]) == 1


# ── wire: on_disco_restart(issued_before=) ──────────────────────────────────
def _rows(v):
    return dict(values=[v, v + 0.01], comps=[comp10(1), comp10(2)])


def test_router_on_disco_restart_issued_before():
    mk = lambda rid: wire.make_line("lane", lane="front", rid=rid, role="main", act=0, zoom=0)  # noqa: E731
    R = wire.ResultRouter(wire.RecipeIds.for_run())
    a = R.try_issue(0, mk("F0"), 100.0)
    b = R.try_issue(1, mk("F1"), 200.0)
    c = R.try_issue(2, mk("B0"), 300.0)
    R.on_read_ack((b, None), 210.0)
    assert R.on_disco_restart(400.0, issued_before=250.0) == [b]
    st = {rid: r.status for rid, r in R.ids.by_id.items()}
    assert st == {a: "unprinted", b: "lost", c: "issued"}
    assert R.last_read == {"front": None, "back": None}
    res = R.on_objective(dict(rid=b, last_read=None, **_rows(0.5)), 200.0 + 3600.0)
    assert res["cls"] == "late" and res["line"]["rid"] == "F1"      # still routed to its own line
    # default: every recipe (the sandbox behaviour)
    R2 = wire.ResultRouter(wire.RecipeIds.for_run())
    x = R2.try_issue(0, mk("F0"), 100.0)
    y = R2.try_issue(2, mk("B0"), 300.0)
    R2.on_read_ack((x, None), 150.0)
    assert R2.on_disco_restart(400.0) == [x]
    assert R2.ids.by_id[y].status == "unprinted"


# ── end to end: real engine, fake four-slot DiSCO that restarts mid-run ─────
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_e2e_real_engine_disco_restart_mid_run(tmp_path, monkeypatch):
    SM = _load("_lanes_hw_smoke_for_g2", REPO / "tests" / "test_lanes_hw_smoke.py")
    from src.lanes.engine import engine_hparams

    device = "cuda" if torch.cuda.is_available() else "cpu"
    comp_db = str(tmp_path / "sql" / "compositions.db")
    obj_db = str(tmp_path / "sql" / "objective.db")
    q_db = str(tmp_path / "sql" / "objective_queue.db")
    clk = SM.Clock(1.8e9)
    monkeypatch.setattr(lanes_hw, "time", clk)
    writes = spy_writes(monkeypatch, clk)
    link = communication.DiscoRestartLink(queue_path=q_db, clock=clk.time)
    T_RESTART = clk.t + 3 * 3600.0
    BOOT_S = 120.0

    class RestartingDisco(SM.FakeDisco4):
        """hw_smoke's fake DiSCO, fed through the serial child's hold, restarting once at T_RESTART:
        compositions.db wiped, "k" reset, the slides in flight keep going (their results arrive after
        the restart with their old "r"), DiSCO's self-test pings ZoMBI, the rails wait while it boots."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.restarted, self.boot_until = False, None
            self.armed = []            # (t, rid, before_restart)
            self.in_flight = []        # (rid, rail) printing when DiSCO restarted

        def _sync(self):
            got = communication._read_slots4(self.comp_db)
            if got is None:
                return
            ts, slots = got
            if not link.map_allowed(ts, slots):
                return
            link.note_map_sent(slots)
            dec = wire.decode_slots(communication._lanes_wire().encode_slots(ts, slots, full=True))
            self.sent_frames += 1
            for k, act in self.slots.apply(dec).items():
                if act == "replace":
                    self.armed.append((self.clk.t, dec["slots"][k][0], not self.restarted))

        def step(self, t):
            if not self.restarted and t >= T_RESTART:
                self.restarted, self.boot_until = True, t + BOOT_S
                self.slots = wire.DiscoSlots()
                self.last_read = {r: None for r in wire.RAILS}
                for r in wire.RAILS:
                    st = self.rail[r]
                    if st["cur"] is not None:
                        self.in_flight.append((st["cur"][1], r))
                    st["nxt"] = None
                link.note_ping({"type": "selftest_ping", "from": "disco", "seq": 1, "pad": "x"}, "live")
            if self.boot_until is not None and t < self.boot_until:
                return
            super().step(t)

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    Xa, Xe, Y = SM._init_data(rng, device)
    hp = engine_hparams(SM.D)
    hp.update(raw=64, n_restarts=8, nat_grad_max_steps=60)
    run_dir = tmp_path / "runs" / "run_g2e2e"
    run_dir.mkdir(parents=True)
    disco = RestartingDisco(comp_db, obj_db, clk, rng)
    clk.hook = disco.step
    t_end = clk.t + 30 * 3600.0
    logs = []
    lanes_hw.run_lanes_hardware(
        run_dir, Xa, Xe, Y, np.stack([np.zeros(SM.D), np.ones(SM.D)]), hp, preset="leader", log=logs.append,
        opt_dims=SM.OPT, y_to_optimizer=lambda v: v,
        stop=lambda: (len([p for p in disco.posted if p[0] > T_RESTART + BOOT_S]) >= 6
                      and SM._all_seen(run_dir, len(disco.posted))) or clk.t > t_end)

    jr = lanes_hw._read_jsonl(run_dir / "lanes" / "journal.jsonl")
    assert not [r for r in jr if r["ev"] in ("packet_error", "loop_error", "link_event_error")]
    assert disco.restarted
    rs = [r for r in jr if r["ev"] == "disco_restart"]
    assert len(rs) == 1 and len([r for r in jr if r["ev"] == "disco_restart_done"]) == 1
    assert len([x for x in logs if x.startswith("[lanes] DiSCO restarted")]) == 1
    t_ping = rs[0]["t_ping"]
    issue_t = {}
    issue = {}
    for r in jr:
        if r["ev"] == "issue":
            issue_t[r["wid"]] = r["t"]
            issue[r["wid"]] = r
    # the fresh DiSCO armed only ids issued after the restart, none it had armed before
    before = {rid for _, rid, b in disco.armed if b}
    after = [(t, rid) for t, rid, b in disco.armed if not b]
    assert after, "the restarted DiSCO never received a map"
    for t, rid in after:
        assert issue_t[rid] >= t_ping and rid not in before, (t, rid)
    # every result routed to the line of its own id; the in-flight slides arrive late, still attributed
    queue_rows = communication.read_objective_queue(0, path=q_db)
    routes = {r["qid"]: r for r in jr if r["ev"] == "route"}
    for qid, p in queue_rows:
        rt = routes[qid]
        assert rt["wid"] == p["r"]
        assert rt["lane"] == issue[p["r"]]["lane"] and rt["engine_rid"] == issue[p["r"]]["engine_rid"]
        assert rt["disposition"] == wire.LANE, rt
    flight = {rid for rid, _ in disco.in_flight}
    assert flight
    late = {routes[q]["wid"] for q, p in queue_rows if p["r"] in flight}
    assert late == flight and all(routes[q]["cls"] == "late" for q, p in queue_rows if p["r"] in flight)
    # after the restart, no map written by the driver carries a pre-restart id
    pre_ids = {w for w, t in issue_t.items() if t < t_ping}
    for t, entries in writes:
        if t >= t_ping:
            assert not {e[1] for e in entries} & pre_ids
    # the rails never waited with both of their slots empty after DiSCO booted
    assert not [v for v in disco.violations if v[0] >= T_RESTART + BOOT_S], disco.violations
    print("results", len(disco.posted), "in flight at restart", disco.in_flight, "armed after", len(after),
          "violations", disco.violations, "restart line", [x for x in logs if "DiSCO restarted" in x])
