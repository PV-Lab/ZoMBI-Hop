"""A result first received WITHOUT its recipe id (DiSCO link restarting) is queued again, exactly once,
when DiSCO re-sends the same table WITH "r"; every other same-ts re-send stays de-duplicated."""
import json
import sqlite3

from scripts import communication as comm


def _rows(qpath):
    con = sqlite3.connect(qpath)
    try:
        return [json.loads(r[0]) for r in con.execute("SELECT raw FROM q ORDER BY id")]
    finally:
        con.close()


def test_r_upgrade_queued_once(tmp_path, monkeypatch):
    monkeypatch.setattr(comm, "_last_queued_ts", None)
    monkeypatch.setattr(comm, "_last_queued_raw", None)
    monkeypatch.setattr(comm, "_last_queued_has_r", False)
    obj = str(tmp_path / "objective.db")
    q = str(tmp_path / "objective_queue.db")
    base = {"type": "objective", "values": [0.5], "comps": [[0.1] * 10]}
    frames = [dict(base, ts=1.0), dict(base, ts=1.0), dict(base, ts=1.0, r=7, k=[7, None]),
              dict(base, ts=1.0, r=7, k=[7, None]), dict(base, ts=1.0), dict(base, ts=2.0, r=8),
              dict(base, ts=2.0, r=8), dict(base, ts=3.0), dict(base, ts=3.0)]
    got = [comm._queue_objective_packet(json.dumps(f), f, obj, q) for f in frames]
    assert got == [True, False, True, False, False, True, False, True, False]
    rows = _rows(q)
    assert [(r["ts"], r.get("r")) for r in rows] == [(1.0, None), (1.0, 7), (2.0, 8), (3.0, None)]
