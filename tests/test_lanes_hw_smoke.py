"""DB-level smoke of the lanes hardware path (scripts/lanes_hw.py + scripts/communication.py).

A fake four-slot DiSCO reads compositions.db through the REAL sender path (communication._read_slots4 ->
wire.encode_slots -> wire.decode_slots -> wire.DiscoSlots), runs two rails with the hardware's cross-wired
order (each rail reads its NEXT recipe ~1 min before its own result is reported), 'prints' the recipe,
scores the printed line on a synthetic 6-D function and appends the result packet (with "r" and "k") to
objective_queue.db through communication._queue_objective_packet, exactly as objective_receiver does.
run_lanes_hardware is driven end to end on a fake clock (its sleep advances the clock and steps DiSCO).
No serial port is opened; every file lives in tmp_path.
"""
import json
import os

import numpy as np
import pytest
import torch

from scripts import communication, lanes_hw
from src.lanes import wire
from src.lanes.engine import engine_hparams
from src.utils.axial_init import axial_design

D = 6
OPT = list(range(D))
C1 = np.array([0.35, 0.25, 0.15, 0.10, 0.10, 0.05])
N_PTS = 12
N_RESULTS = 12
PRINT_S = 70 * 60.0          # read -> result (under the router's 90-min lost-after-read)
CLEAR_S = 5 * 60.0           # read -> archerfish clear (the print itself)
PREFETCH_S = 60.0            # the rail reads its next recipe this long before its own result
BACK_START_S = 35 * 60.0     # the back rail starts half a period later (rails alternate)
TICK_S = 60.0


def f6(X):
    X = np.asarray(X, float).reshape(-1, D)
    return np.exp(-((X - C1) ** 2).sum(axis=1) / (2 * 0.2 ** 2))


class Clock:
    def __init__(self, t0):
        self.t = float(t0)
        self.hook = None

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += TICK_S
        if self.hook is not None:
            self.hook(self.t)


def measure(a10, b10, rng):
    a, b = np.asarray(a10, float)[OPT], np.asarray(b10, float)[OPT]
    t = np.linspace(0, 1, N_PTS)[:, None]
    X = np.clip(a + t * (b - a) + rng.normal(0, 0.005, (N_PTS, D)), 0, None)
    X = X / X.sum(axis=1, keepdims=True)
    y = f6(X) + rng.normal(0, 0.02, N_PTS)
    C = np.zeros((N_PTS, 10))
    C[:, OPT] = X
    return [float(v) for v in y], C.tolist()


class FakeDisco4:
    """Two rails on the DiscoSlots rules; reads compositions.db the way composition_sender does."""

    def __init__(self, comp_db, obj_db, clk, rng):
        self.comp_db, self.obj_db, self.clk, self.rng = comp_db, obj_db, clk, rng
        self.slots = wire.DiscoSlots()
        self.last_read = {r: None for r in wire.RAILS}
        self.rail = {r: dict(cur=None, nxt=None, t_read=None, cleared=False,
                             start=clk.t + (0.0 if r == "front" else BACK_START_S)) for r in wire.RAILS}
        self.posted = []          # (ts, rid, rail, slot)
        self.violations = []      # a rail wanted a recipe and both of its slots were empty
        self.sent_frames = 0

    def _sync(self):
        got = communication._read_slots4(self.comp_db)
        if got is None:
            return
        ts, slots = got
        raw = communication._lanes_wire().encode_slots(ts, slots, full=True)
        self.sent_frames += 1
        self.slots.apply(wire.decode_slots(raw))

    def _read(self, r, t):
        got = self.slots.read(r)
        if got is None:
            base = 0 if r == "front" else 2
            self.violations.append((t, r, [self.slots.slots[k].state for k in (base, base + 1)]))
            return None
        self.last_read[r] = got[1]
        return got

    def step(self, t):
        self._sync()                                   # the 1 Hz sender -> DiSCO apply
        for r in wire.RAILS:
            st = self.rail[r]
            if t < st["start"]:
                continue
            if st["cur"] is None:
                got = st["nxt"] if st["nxt"] is not None else self._read(r, t)
                st["nxt"] = None
                if got is None:
                    continue
                st.update(cur=got, t_read=t, cleared=False)
            if not st["cleared"] and t >= st["t_read"] + CLEAR_S:
                self.slots.clear(r)                    # archerfish: READ -> CLEARED after the print
                st["cleared"] = True
            due = st["t_read"] + PRINT_S
            if st["nxt"] is None and st["cleared"] and t >= due - PREFETCH_S:
                st["nxt"] = self._read(r, t)           # cross-wire: next read ~1 min before own result
            if t >= due:
                slot, rid, a, b = st["cur"]
                y, C = measure(a, b, self.rng)
                k = (self.last_read["front"], self.last_read["back"])
                line = wire.encode_objective(y, C, t, rid, k).decode("utf-8").strip()
                ok = communication._queue_objective_packet(line, json.loads(line), self.obj_db)
                assert ok, "queue append failed"
                self.posted.append((t, rid, r, slot))
                st.update(cur=None, t_read=None, cleared=False)
                if st["nxt"] is not None:
                    st.update(cur=st["nxt"], nxt=None, t_read=t, cleared=False)


def _init_data(rng, device):
    design = axial_design(D, lower=np.zeros(D), upper=np.ones(D), drop="auto")
    Xa, Xe, Ys = [], [], []
    for tr in design.traces:
        L, R = np.asarray(tr.end, float), np.asarray(tr.start, float)
        y, C = measure(np.pad(L, (0, 10 - D)), np.pad(R, (0, 10 - D)), rng)
        Xa.append(np.asarray(C)[:, OPT])
        t = np.linspace(0, 1, N_PTS)[:, None]
        Xe.append(L + t * (R - L))
        Ys.append(y)
    f = lambda a: torch.tensor(np.concatenate(a), dtype=torch.float64, device=device)  # noqa: E731
    return f(Xa), f(Xe), f([np.asarray(y) for y in Ys]).reshape(-1, 1)


def test_lanes_hw_db_level_smoke(tmp_path, monkeypatch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sql").mkdir()
    comp_db = str(tmp_path / "sql" / "compositions.db")
    obj_db = str(tmp_path / "sql" / "objective.db")
    q_db = str(tmp_path / "sql" / "objective_queue.db")
    monkeypatch.setattr(communication, "_last_queued_ts", None)
    monkeypatch.setattr(communication, "_last_queued_raw", None)

    clk = Clock(1.8e9)
    monkeypatch.setattr(lanes_hw, "time", clk)

    real_write = communication.write_slots4
    writes = []

    def spy_write(slots, ts=None, path=comp_db):
        slots = [tuple(s) for s in slots]
        writes.append(slots)
        out = real_write(slots, ts=ts, path=path)       # the real write path into compositions.db
        assert os.path.abspath(path) == os.path.abspath(comp_db)
        return out

    monkeypatch.setattr(communication, "write_slots4", spy_write)

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    Xa, Xe, Y = _init_data(rng, device)
    hp = engine_hparams(D)
    hp.update(raw=64, n_restarts=8, nat_grad_max_steps=60)
    run_dir = tmp_path / "runs" / "run_smoke"
    run_dir.mkdir(parents=True)

    disco = FakeDisco4(comp_db, obj_db, clk, rng)
    clk.hook = disco.step
    t_end = clk.t + 24 * 3600.0
    logs = []
    lanes_hw.run_lanes_hardware(
        run_dir, Xa, Xe, Y, np.stack([np.zeros(D), np.ones(D)]), hp, preset="leader", log=logs.append,
        opt_dims=OPT, y_to_optimizer=lambda v: v,
        stop=lambda: (len(disco.posted) >= N_RESULTS and _all_seen(run_dir, len(disco.posted))) or clk.t > t_end)

    jr = lanes_hw._read_jsonl(run_dir / "lanes" / "journal.jsonl")
    queue = communication.read_objective_queue(0, path=q_db)
    issue = {}
    for rec in jr:
        if rec["ev"] == "issue":
            issue[rec["wid"]] = rec        # ids are never reused within a run (24 h horizon)

    # sanity: the run happened
    assert len(disco.posted) >= N_RESULTS, (len(disco.posted), logs[-20:])
    assert len(queue) == len(disco.posted)
    assert not [r for r in jr if r["ev"] in ("packet_error", "loop_error")], \
        [r for r in jr if r["ev"] in ("packet_error", "loop_error")][:2]

    # 1) journal has every packet, raw, before acting
    pk = {r["qid"]: r for r in jr if r["ev"] == "packet"}
    assert set(pk) == {q for q, _ in queue}
    for q, p in queue:
        assert pk[q]["raw"] == p

    # 2) every result routed to the lane / line whose id it carries
    routes = {r["qid"]: r for r in jr if r["ev"] == "route"}
    ingested = {r["qid"]: r for r in jr if r["ev"] == "ingested"}
    for q, p in queue:
        rt = routes[q]
        assert rt["wid"] == p["r"]
        iss = issue[p["r"]]
        assert rt["lane"] == iss["lane"], (q, rt, iss)
        assert rt["engine_rid"] == iss["engine_rid"], (q, rt, iss)
        assert rt["disposition"] == wire.LANE, rt
        assert q in ingested, (q, rt)
    # the printed slot belonged to the lane of its rail (each rail reads only its own slots)
    for t, rid, rail, slot in disco.posted:
        assert wire.SLOT_RAIL[slot] == rail
        assert issue[rid]["lane"] == rail and issue[rid]["slot"] == slot

    # 3) the four-slot map is always the full state, and the DB holds exactly what was written
    assert writes
    for w in writes:
        assert sorted(s[0] for s in w) == [0, 1, 2, 3], w
    sw = [r for r in jr if r["ev"] == "slot_write"]
    assert len(sw) == len(writes)
    ts_db, slots_db = communication._read_slots4(comp_db)
    assert {k: v[0] for k, v in slots_db.items()} == {s[0]: s[1] for s in writes[-1]}

    # 4) no rail ever wanted a recipe with both of its slots empty
    assert not disco.violations, disco.violations

    # 5) the engine decided for both lanes (start + regular decisions on results)
    dec = [r for r in jr if r["ev"] == "decision" and r.get("result") != "none"]
    for lane in ("front", "back"):
        kinds = [r["kind"] for r in dec if r["lane"] == lane]
        assert "start" in kinds and "regular" in kinds, (lane, kinds)

    # durable side files
    assert (run_dir / "lanes" / "ids.json").exists()
    assert (run_dir / "lanes" / "engine_ckpt.pt").exists()
    rows = lanes_hw._read_jsonl(run_dir / "lanes" / "rows.jsonl")
    assert len(rows) == N_PTS * len(ingested)
    print("results", len(disco.posted), "writes", len(writes), "decisions", len(dec),
          "route classes", {c: sum(1 for q in routes if routes[q]["cls"] == c) for c in {routes[q]["cls"] for q in routes}},
          "decision kinds", [(r["lane"], r["kind"]) for r in dec],
          "pickups", sum(1 for r in jr if r["ev"] == "pickup"), "frames", disco.sent_frames)


def _all_seen(run_dir, n):
    jr = lanes_hw._read_jsonl(run_dir / "lanes" / "journal.jsonl")
    return sum(1 for r in jr if r["ev"] == "cursor") >= n
