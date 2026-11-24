"""Lanes serial path in scripts/communication.py (four-slot sender + objective queue).

Checks, with temp folders and in-memory ports only (no COM port is ever opened):
  * write_slots4 -> composition_sender puts on the wire exactly the bytes of the hw_test
    encoder (tests/zombi_side_test.py, the cable-tested GREEN encoder) for the same slots,
    and the same bytes as src/lanes/wire.encode_slots;
  * a new ts on every write_slots4 call; withdrawn (absent) slots are absent on the wire;
  * when slots4 is inactive (never written, or clear_slots4) the sender sends today's
    legacy packet, byte-identical to the frame sent before any lanes table existed;
  * a failed four-slot write makes the next four-slot frame start with "\\n" (resync);
  * objective_receiver appends every NEW objective packet (ts de-dup) to
    sql/objective_queue.db and read_objective_queue returns them in id order, while the
    legacy objective.db write still happens.

Run: PYTHONPATH=. python -m pytest -q tests/test_lanes_serial_rush.py
"""
from __future__ import annotations

import importlib.util
import json
import queue
import sqlite3
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import serial

from scripts import communication

REPO = Path(__file__).resolve().parents[1]
HZ = 50.0


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod                          # @dataclass needs the module registered
    spec.loader.exec_module(mod)
    return mod


HW = _load("_hw_zombi_side_test", REPO / "tests" / "zombi_side_test.py")
WIRE = _load("_prod_lanes_wire", REPO / "src" / "lanes" / "wire.py")


# ── fixtures / fakes ────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _iso(tmp_path, monkeypatch):
    """cwd = tmp (legacy code uses ./sql/ paths); fresh module state; flag cleared."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sql").mkdir()
    for g in ("_last_objective_ts", "_last_objective_vals", "_last_objective_comps",
              "_last_queued_ts", "_last_queued_raw"):
        monkeypatch.setattr(communication, g, None)
    communication._shutdown_flag.clear()
    yield
    communication._shutdown_flag.clear()


class Port:
    def __init__(self, stop_after=1, fail_first=0):
        self.stop_after, self.fail_left = stop_after, fail_first
        self.frames, self.events = [], []

    def write(self, data):
        if self.fail_left > 0:
            self.fail_left -= 1
            self.events.append("fail")
            raise serial.SerialException("injected")
        self.frames.append(bytes(data))
        self.events.append("write")
        if len(self.frames) >= self.stop_after:
            communication._shutdown_flag.set()
        return len(data)

    def flush(self):
        pass

    def reset_output_buffer(self):
        self.events.append("reset")


def run_sender(comp_db, stop_after=1, fail_first=0):
    port = Port(stop_after, fail_first)
    communication._shutdown_flag.clear()
    guard = threading.Timer(10.0, communication._shutdown_flag.set)
    guard.daemon = True
    guard.start()
    try:
        communication.composition_sender(port, HZ, comp_db, False, False)
    finally:
        guard.cancel()
        communication._shutdown_flag.clear()
    return port


class DrainingQueue(queue.Queue):
    def get(self, block=True, timeout=None):
        try:
            return super().get(block=False)
        except queue.Empty:
            communication._shutdown_flag.set()
            raise


def receive(lines, monkeypatch):
    q = DrainingQueue()
    for ln in lines:
        q.put(ln)
    monkeypatch.setattr(communication, "_read_queue", q)
    communication._shutdown_flag.clear()
    guard = threading.Timer(30.0, communication._shutdown_flag.set)
    guard.daemon = True
    guard.start()
    try:
        communication.objective_receiver(HZ, "./sql/objective.db", "./sql/objective_memory.db", False)
    finally:
        guard.cancel()
        communication._shutdown_flag.clear()
    assert q.empty()


def vec(seed):
    r = np.random.default_rng(seed).random(10)
    return [float(x) for x in np.round(r / r.sum(), 3)]


SLOTS = [(0, 417, vec(1), vec(2)), (1, 418, vec(3), vec(4)), (2, 5, vec(5), vec(6)), (3, 999, vec(7), vec(8))]
COMP_DB = "./sql/compositions.db"


def hw_bytes(ts, slots):
    slots4 = [None] * 4
    for k, rid, a, b in slots:
        slots4[k] = (rid, a, b)
    return HW.encode_slots(ts, slots4, False)


def write_legacy(ts=1793900000.25):
    a = np.array(vec(11)); b = np.array(vec(12))
    grid = np.linspace(a, b, 24)
    communication.write_compositions(a, b, grid, ts, b, a, grid[::-1].copy(), db_path=COMP_DB)


# ── sender ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("subset", [(0, 1, 2, 3), (0, 2), (1, 3), ()])
def test_sender_four_slot_bytes_match_hw_encoder(subset):
    write_legacy()                                   # legacy tables present too: four-slot must win
    slots = [s for s in SLOTS if s[0] in subset]
    ts = communication.write_slots4(slots, ts=1793900123.5)
    assert ts == 1793900123.5
    port = run_sender(COMP_DB, stop_after=2)
    expect = hw_bytes(ts, slots)
    assert port.frames == [expect, expect]           # 1 Hz re-sends are identical
    assert expect == WIRE.encode_slots(ts, {k: (r, a, b) for k, r, a, b in slots}, full=True)
    dec = WIRE.decode_slots(port.frames[0])
    assert sorted(dec["slots"]) == list(subset) and dec["full"]
    for k, r, a, b in slots:
        assert dec["slots"][k] == (r, a, b)


def test_write_slots4_unordered_input_and_new_ts_each_call():
    t1 = communication.write_slots4(list(reversed(SLOTS)))
    t2 = communication.write_slots4(SLOTS)
    t3 = communication.write_slots4(SLOTS[:1])
    assert t1 < t2 < t3
    port = run_sender(COMP_DB)
    assert port.frames == [hw_bytes(t3, SLOTS[:1])]


def test_write_slots4_rejects_bad_input():
    for bad in ([(4, 1, vec(1), vec(2))], [(0, 1000, vec(1), vec(2))], [(0, 1, vec(1)[:9], vec(2))],
                [(0, 1, vec(1), vec(2)), (0, 2, vec(3), vec(4))], [(0, 1, [-0.1] + vec(1)[1:], vec(2))]):
        with pytest.raises(ValueError):
            communication.write_slots4(bad)


def test_legacy_unchanged_when_inactive_and_after_clear():
    write_legacy()
    legacy = run_sender(COMP_DB).frames
    a = np.array(vec(11)); b = np.array(vec(12)); grid = np.linspace(a, b, 24)
    assert legacy == [WIRE.encode_composition_legacy(1793900000.25, a.tolist(), b.tolist(), grid.tolist(),
                                                     b.tolist(), a.tolist(), grid[::-1].tolist())]
    communication.clear_slots4(COMP_DB)              # no lanes tables yet: creates them inactive
    assert run_sender(COMP_DB).frames == legacy
    ts = communication.write_slots4(SLOTS)
    assert run_sender(COMP_DB).frames == [hw_bytes(ts, SLOTS)]
    communication.clear_slots4(COMP_DB)
    assert run_sender(COMP_DB).frames == legacy
    # legacy writer rewriting its own tables leaves the lanes tables alone
    write_legacy()
    with sqlite3.connect(COMP_DB) as c:
        assert c.execute("SELECT COUNT(*) FROM slots4").fetchone()[0] == 4
        assert c.execute("SELECT active FROM slots4_meta").fetchone()[0] == 0


def test_four_slot_resync_newline_after_failed_write():
    ts = communication.write_slots4(SLOTS)
    port = run_sender(COMP_DB, stop_after=2, fail_first=1)
    assert port.events[:3] == ["fail", "reset", "write"]
    assert port.frames == [b"\n" + hw_bytes(ts, SLOTS), hw_bytes(ts, SLOTS)]


# ── receiver -> queue ───────────────────────────────────────────────────────
def obj_line(ts, rid, vals, k=(1, 2), comps=None):
    comps = comps if comps is not None else [vec(i) for i in range(len(vals))]
    return WIRE.encode_objective(vals, comps, ts, rid=rid, last_read=tuple(k))


def test_receiver_appends_new_packets_and_read_returns_them_in_order(monkeypatch):
    a = obj_line(100.5, 7, [0.1, 0.2, 0.3])
    b = obj_line(101.5, 8, [0.4, 0.5])
    nan = obj_line(102.5, 9, [float("nan")] * 2)     # legacy ignores it; the lanes queue keeps it
    lines = [a, a, b"not json\n", json.dumps({"type": "selftest_ping", "from": "disco"}).encode() + b"\n",
             b, b, nan, a]                           # a again after b: its ts differs from the last queued one
    receive(lines, monkeypatch)
    rows = communication.read_objective_queue(0, path="./sql/objective_queue.db")
    got = [(p["ts"], p.get("r")) for _, p in rows]
    assert got == [(100.5, 7), (101.5, 8), (102.5, 9), (100.5, 7)]
    ids = [i for i, _ in rows]
    assert ids == sorted(ids) and len(set(ids)) == 4
    assert rows[0][1] == json.loads(a)
    assert communication.read_objective_queue(ids[1], path="./sql/objective_queue.db") == rows[2:]
    assert communication.read_objective_queue(ids[-1], path="./sql/objective_queue.db") == []
    with sqlite3.connect("./sql/objective_queue.db") as c:
        assert [r[0] for r in c.execute("SELECT ts FROM q ORDER BY id")] == [100.5, 101.5, 102.5, 100.5]
    # legacy path still ran: objective.db holds the last VALID packet (a, re-sent last with a new-vs-last ts)
    with sqlite3.connect("./sql/objective.db") as c:
        vals = [r[0] for r in c.execute("SELECT val FROM objective")]
    assert vals == [0.1, 0.2, 0.3]


def test_receiver_queue_survives_restart_dedup_and_missing_db():
    assert communication.read_objective_queue(0, path="./sql/objective_queue.db") == []
    line = obj_line(5.0, 1, [0.5]).decode().strip()
    pkt = json.loads(line)
    assert communication._queue_objective_packet(line, pkt, "./sql/objective.db")
    assert not communication._queue_objective_packet(line, pkt, "./sql/objective.db")   # same ts
    assert not communication._queue_objective_packet("x", {"type": "caps"}, "./sql/objective.db")
    assert not communication._queue_objective_packet("[1]", [1], "./sql/objective.db")
    rows = communication.read_objective_queue(0)     # default path ./sql/objective_queue.db
    assert [p["r"] for _, p in rows] == [1]
