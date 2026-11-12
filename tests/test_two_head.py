"""Two-head mode: rails protocol v2 (ZoMBI side), the rail broker, the two-head hub,
and a full two-head hardware run against a fake rails-v2 DiSCO.

The fake DiSCO follows docs/disco_independent_rails.md: it arms each slot on its own
per-slot timestamp, tags every result with (slot, slot_ts), and talks to ZoMBI's real
objective_receiver through the serial read queue (re-sending results, as DiSCO does).
"""

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.run_zombi_main as rzm
from scripts import communication as zcomm
from scripts.rail_broker import RailBroker
from test_axial_init_hardware import _Stop, _fast_hparams, _landscape, hw_env  # noqa: F401

OBJ_DB = "./sql/objective.db"
COMP_DB = "./sql/compositions.db"
MEM_DB = "./sql/objective_memory.db"


# ── fake rails-v2 DiSCO ──────────────────────────────────────────────────────

class FakeDiSCOv2(threading.Thread):
    """Two transporters, one printer; arms main/cache on their OWN slot ts.

    Results are sent as JSON lines into ZoMBI's serial read queue (so the real
    objective_receiver ingests them), each re-sent `resends` extra times."""

    def __init__(self, dims, *, print_s=0.1, return_s=0.05, lag_s=None, noise=0.01,
                 resends=2, seed=0, tag=True, fail=None):
        super().__init__(daemon=True)
        self.dims = list(dims)
        self.print_s, self.return_s = print_s, return_s
        # per-slot result lag: cache comes back FASTER than main, so results arrive
        # out of command order
        self.lag_s = lag_s or {"main": 0.9, "cache": 0.25}
        self.noise, self.resends, self.tag = noise, resends, tag
        self.fail = fail or (lambda k, slot: False)
        self.rng = np.random.default_rng(seed)
        self.stop_evt = threading.Event()
        self.prints = []          # (t, slot, slot_ts, rows)
        self.dropped_unread = []  # (slot, slot_ts) superseded before printing
        self._ts = 0.0

    def _read(self):
        import sqlite3
        try:
            conn = sqlite3.connect(COMP_DB, timeout=5)
            st = dict(conn.execute("SELECT slot, ts FROM slot_ts").fetchall())
            main = np.array(conn.execute("SELECT * FROM compositions").fetchall(), float)
            cache = np.array(conn.execute("SELECT * FROM compositions_cache").fetchall(), float)
            conn.close()
            return st, {"main": main, "cache": cache}
        except sqlite3.Error:
            return None

    def _flag(self):
        import sqlite3
        try:
            conn = sqlite3.connect(OBJ_DB, timeout=5)
            v = conn.execute("SELECT new_objective_available FROM handshake WHERE id=1").fetchone()
            conn.close()
            return v[0] if v else 0
        except sqlite3.Error:
            return 1

    def _emit(self, slot, slot_ts, rows, k):
        meas = rows + self.rng.normal(0, self.noise, rows.shape) * (rows > 0)
        meas = np.clip(meas, 0, None)
        meas /= meas.sum(axis=1, keepdims=True)
        y = _landscape(meas[:, self.dims])
        if self.fail(k, slot):
            y[:] = np.nan
        self._ts = max(time.time(), self._ts + 1e-3)
        pkt = {"type": "objective", "ts": self._ts,
               "values": [None if not np.isfinite(v) else float(v) for v in y],
               "comps": meas.tolist()}
        if self.tag:
            pkt.update(slot=slot, slot_ts=slot_ts)
        raw = (json.dumps(pkt) + "\n").encode()
        for _ in range(1 + self.resends):
            zcomm._read_queue.put(raw)

    def run(self):
        last = {"main": None, "cache": None}
        got = self._read()
        if got is not None:
            last = {s: got[0].get(s) for s in last}          # predates us
        slots = {"main": None, "cache": None}                # [rows, slot_ts, unread]
        free_at = {"front": 0.0, "back": 0.0}
        printer_free = 0.0
        due = []
        while not self.stop_evt.is_set():
            now = time.time()
            got = self._read()
            if got is not None:
                st, rows = got
                for s in ("main", "cache"):
                    if st.get(s) is not None and st.get(s) != last[s] and len(rows[s]):
                        if slots[s] is not None and slots[s][2]:
                            self.dropped_unread.append((s, slots[s][1]))
                        last[s] = st[s]
                        slots[s] = [rows[s], st[s], True]
            for t in free_at:
                if free_at[t] > now or printer_free > now:
                    continue
                s = "main" if slots["main"] and slots["main"][2] else \
                    "cache" if slots["cache"] and slots["cache"][2] else None
                if s is None:
                    continue
                slots[s][2] = False
                k = len(self.prints)
                self.prints.append((now, s, slots[s][1], slots[s][0]))
                printer_free = now + self.print_s
                free_at[t] = printer_free + self.return_s
                due.append((printer_free + self.lag_s[s], s, slots[s][1], slots[s][0], k))
            due.sort(key=lambda d: d[0])
            # Initialization (both-slot writes) reads the single-packet mailbox:
            # hold a result while the previous one is unread, as FakeDiSCO does.
            init_phase = last["main"] is not None and last["main"] == last["cache"]
            if due and due[0][0] <= now and not (init_phase and self._flag() == 1
                                                 and now - due[0][0] < 3.0):
                _, s, sts, rows, k = due.pop(0)
                self._emit(s, sts, rows, k)
            time.sleep(0.02)

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=10)


@pytest.fixture()
def receiver(hw_env):
    """ZoMBI's real objective_receiver on the serial read queue."""
    zcomm._shutdown_flag.clear()
    zcomm._last_objective_ts = zcomm._last_objective_vals = zcomm._last_objective_comps = None
    zcomm._seen_slot_objectives.clear()
    zcomm._objective_acks.update(main=None, cache=None)
    while not zcomm._read_queue.empty():
        zcomm._read_queue.get_nowait()
    t = threading.Thread(target=zcomm.objective_receiver, args=(20, OBJ_DB, MEM_DB), daemon=True)
    t.start()
    yield t
    zcomm._shutdown_flag.set()
    t.join(timeout=5)
    zcomm._shutdown_flag.clear()


def _line10(a, b, n=24):
    a, b = np.asarray(a, float), np.asarray(b, float)
    arr = np.array([a + t * (b - a) for t in np.linspace(0, 1, n)])
    return a, b, arr


def _wait(cond, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.05)
    return False


# ── protocol (ZoMBI side) ────────────────────────────────────────────────────

def test_write_slot_leaves_the_other_rail_alone(hw_env):
    import sqlite3
    e = np.eye(10)
    zcomm.write_compositions(*_line10(e[0], e[1])[:2], _line10(e[0], e[1])[2], 100.0,
                             *_line10(e[2], e[3]))
    assert zcomm.read_slot_ts(COMP_DB) == {"main": 100.0, "cache": 100.0}
    zcomm.write_slot("main", *_line10(e[4], e[5]), slot_ts=200.0)
    assert zcomm.read_slot_ts(COMP_DB) == {"main": 200.0, "cache": 100.0}
    conn = sqlite3.connect(COMP_DB)
    cache = np.array(conn.execute("SELECT * FROM compositions_cache").fetchall())
    main = np.array(conn.execute("SELECT * FROM compositions").fetchall())
    ts = conn.execute("SELECT ts FROM timestamp").fetchone()[0]
    conn.close()
    assert np.allclose(cache[0], e[2]) and np.allclose(main[0], e[4])
    assert ts == 200.0                       # legacy ts still bumps on every write


def test_receiver_queues_tagged_results_once_and_acks(hw_env, receiver):
    def pkt(ts, slot, sts, vals):
        return (json.dumps({"type": "objective", "ts": ts, "slot": slot, "slot_ts": sts,
                            "values": vals, "comps": [[0.5, 0.5] + [0] * 8] * len(vals)})
                + "\n").encode()
    # interleaved re-sends of two slots' results: each must be queued exactly once
    for raw in [pkt(1.0, "main", 10.0, [1, 2]), pkt(2.0, "cache", 11.0, [3]),
                pkt(1.0, "main", 10.0, [1, 2]), pkt(2.0, "cache", 11.0, [3]),
                pkt(3.0, "cache", 12.0, [None]),           # failed slide: still queued
                pkt(1.0, "main", 10.0, [1, 2])]:
        zcomm._read_queue.put(raw)
    assert _wait(lambda: len(zcomm.read_objective_queue(OBJ_DB)) >= 3)
    time.sleep(0.5)
    rows = zcomm.read_objective_queue(OBJ_DB)
    assert [(r["slot"], r["slot_ts"]) for r in rows] == [("main", 10.0), ("cache", 11.0),
                                                         ("cache", 12.0)]
    assert np.isnan(rows[2]["values"]).all()
    assert zcomm._objective_acks == {"main": 1.0, "cache": 3.0}


# ── rail broker ──────────────────────────────────────────────────────────────

def _payload(left, right, dims, n=24):
    x_sent, s10, e10, a10 = rzm._line_payload(np.asarray(left), np.asarray(right), n)
    return s10, e10, a10


def test_broker_routes_out_of_order_results_to_their_own_head(hw_env, receiver, monkeypatch):
    dims = [0, 8, 9]
    monkeypatch.setattr(rzm, "OPTIMIZING_DIMS", dims)
    disco = FakeDiSCOv2(dims, lag_s={"main": 1.2, "cache": 0.2})
    disco.start()
    br = RailBroker(dims, first_timeout_s=30, stall_floor_s=5, poll_s=0.05,
                    log_path=Path("rails_log.jsonl"))
    br.start()
    lines = {0: [([.8, .1, .1], [.1, .8, .1]), ([.7, .2, .1], [.2, .2, .6])],
             1: [([.1, .1, .8], [.4, .5, .1]), ([.3, .3, .4], [.6, .1, .3]),
                 ([.2, .7, .1], [.5, .2, .3])]}
    out = {0: [], 1: []}

    def head(h):
        for a, b in lines[h]:
            x, y = br.measure(h, np.array(a), np.array(b), _payload(a, b, dims))
            out[h].append((np.array(a), np.array(b), x, y))
    try:
        ts = [threading.Thread(target=head, args=(h,)) for h in (0, 1)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=60)
    finally:
        disco.stop()
    from src.utils.init_queue import _segment_distance
    for h in (0, 1):
        assert len(out[h]) == len(lines[h])
        for a, b, x, y in out[h]:
            assert len(y) == 24
            assert _segment_distance(x, a, b).mean() < 0.05     # it IS this head's line
    # every commanded line printed exactly once; the slow main rail was never re-armed
    # by the cache head's writes (v1 would have re-printed it)
    printed = [(s, sts) for _, s, sts, _ in disco.prints]
    assert len(printed) == len(set(printed)) == 5
    assert sum(s == "main" for s, _ in printed) == 2
    assert disco.dropped_unread == []


def test_broker_resends_a_lost_line_and_shares_its_late_twin(hw_env, receiver, monkeypatch):
    dims = [0, 8, 9]
    monkeypatch.setattr(rzm, "OPTIMIZING_DIMS", dims)
    # the first main print reports only after the stall timeout -> re-sent; both
    # copies eventually report: the second one is extra data
    disco = FakeDiSCOv2(dims, lag_s={"main": 2.2, "cache": 0.2})
    disco.start()
    extra = []
    br = RailBroker(dims, first_timeout_s=1.5, stall_floor_s=1.5, poll_s=0.05, max_sends=3,
                    on_extra_data=lambda h, x, y: extra.append((h, len(y))))
    br.start()
    try:
        a, b = [.8, .1, .1], [.1, .8, .1]
        x, y = br.measure(0, np.array(a), np.array(b), _payload(a, b, dims))
        assert len(y) == 24
        # the late twin is picked up whenever a head next waits on the broker
        assert _wait(lambda: (br.pump(), extra)[1], timeout=10)
    finally:
        disco.stop()
    assert extra and extra[0][0] == 0


def test_broker_failed_slide_returns_empty_at_once(hw_env, receiver, monkeypatch):
    dims = [0, 8, 9]
    monkeypatch.setattr(rzm, "OPTIMIZING_DIMS", dims)
    disco = FakeDiSCOv2(dims, fail=lambda k, s: True)
    disco.start()
    br = RailBroker(dims, first_timeout_s=60, stall_floor_s=60, poll_s=0.05)
    br.start()
    try:
        a, b = [.1, .1, .8], [.4, .5, .1]
        t0 = time.time()
        x, y = br.measure(1, np.array(a), np.array(b), _payload(a, b, dims))
    finally:
        disco.stop()
    assert len(y) == 0 and x.shape == (0, 3) and time.time() - t0 < 10


# ── data handler + hub ───────────────────────────────────────────────────────

def _dh(torch, tmp, uuid, resume=False):
    from src.utils.datahandler import DataHandler
    return DataHandler(directory=str(tmp), run_uuid=uuid, is_resume=resume,
                       device="cuda", dtype=torch.float64, d=3)


def _init(dh, torch):
    X = torch.tensor([[.2, .3, .5], [.6, .2, .2], [.1, .8, .1]], dtype=torch.float64,
                     device=dh.device)
    b = torch.zeros(2, 3, dtype=torch.float64, device=dh.device)
    b[1] = 1
    dh.save_init(X, X.clone(), torch.tensor([[1.], [2.], [3.]], dtype=torch.float64,
                                             device=dh.device), b)


def test_foreign_rows_are_gp_data_not_activation_data_and_persist(torch, tmp_path):
    dh = _dh(torch, tmp_path, "fr01")
    _init(dh, torch)
    dev = dict(device=dh.device, dtype=torch.float64)
    dh.activation_start_idx = 3
    own = torch.tensor([[.3, .3, .4]], **dev)
    other = torch.tensor([[.5, .4, .1], [.45, .45, .1]], **dev)
    dh.add_all_points(own, own, torch.tensor([[5.]], **dev))
    dh.add_all_points(other, other, torch.tensor([[9.], [8.]], **dev), foreign=True)
    assert dh.current_activation_mask().tolist() == [False] * 3 + [True, False, False]
    assert dh.get_gp_data()[0].shape[0] >= 5          # GP sees them (paring may merge two)
    dh.take_snapshot("s", activation=0, zoom=0, iteration=0)
    dh2 = _dh(torch, tmp_path, "fr01", resume=True)
    dh2.load_state()
    assert dh2.foreign_mask().tolist() == [False] * 4 + [True, True]
    assert dh2.current_activation_mask().tolist() == [False] * 3 + [True, False, False]


def test_regions_cross_only_at_the_activation_boundary(torch, tmp_path):
    from src.core.two_head import TwoHeadHub
    seen = []
    hub = TwoHeadHub(on_regions_changed=lambda regs: seen.append(len(regs)))
    heads = []
    for h in (0, 1):
        dh = _dh(torch, tmp_path, f"hb0{h}")
        _init(dh, torch)
        z = SimpleNamespace(data_handler=dh, _log=lambda m: None)
        hub.attach(h, z)
        heads.append(z)
    dev = dict(device=heads[0].data_handler.device, dtype=torch.float64)
    a, b = heads[0].data_handler, heads[1].data_handler
    # head0 measures and declares a needle; head1 is mid-activation
    p = torch.tensor([[.33, .33, .34]], **dev)
    heads[0].two_head_link.after_measurement(p, p, torch.tensor([7.], **dev))
    a.add_needle(p[0], 7.0, 0.05, 0, 0)
    heads[1].two_head_link.after_measurement(p * 0 + torch.tensor([.7, .2, .1], **dev),
                                             p, torch.tensor([1.], **dev))
    assert b.foreign_mask().sum().item() == 1          # point arrived with head1's next line
    assert b.needles.shape[0] == 0                     # ...but not the needle
    assert seen == [1]
    heads[1].two_head_link.at_activation_boundary()
    assert b.needles.shape[0] == 1 and b.needles_results[0]["origin_head"] == 0
    assert not b.get_penalty_mask(p).item()            # now penalised for head1 too
    # an imported region is never echoed back
    heads[0].two_head_link.at_activation_boundary()
    assert a.needles.shape[0] == 1 and len(hub.regions()) == 1
    # head1 finds the same needle independently: both are kept
    b.add_needle(p[0] + 1e-4, 7.1, 0.05, 0, 0)
    heads[0].two_head_link.at_activation_boundary()
    assert a.needles.shape[0] == 2


# ── full two-head hardware run (fake v2 DiSCO) ───────────────────────────────

class _Counter:
    """Lets each head measure `per_head` lines; a head that wants more waits until
    the other has had its quota too, then stops the run (deterministic stop)."""

    def __init__(self, per_head):
        self.per_head, self.n, self.done = per_head, {0: 0, 1: 0}, {0: 0, 1: 0}
        self.cv = threading.Condition()

    def wrap(self, real):
        def f(broker, head, endpoints, ref=None, **kw):
            with self.cv:
                if self.n[head] >= self.per_head:
                    # stop only once BOTH heads have their quota of lines back
                    # (stopping aborts the other head's in-flight measurement)
                    self.cv.wait_for(lambda: all(v >= self.per_head
                                                 for v in self.done.values()), timeout=180)
                    raise _Stop()
                self.n[head] += 1
            out = real(broker, head, endpoints, ref, **kw)
            with self.cv:
                self.done[head] += 1
                self.cv.notify_all()
            return out
        return f


@pytest.fixture()
def two_head_env(hw_env, receiver, monkeypatch):
    monkeypatch.setattr(rzm, "TWO_HEAD_FIRST_RESULT_TIMEOUT_MIN", 0.5)
    monkeypatch.setattr(rzm, "TWO_HEAD_STALL_FLOOR_MIN", 0.1)
    return hw_env


def test_two_head_run_end_to_end_and_resume(two_head_env, monkeypatch, torch):
    dims = [0, 8, 9]
    monkeypatch.setattr(rzm, "OPTIMIZING_DIMS", dims)
    ckpt = two_head_env / "ckpt"
    real = rzm.rail_objective
    disco = FakeDiSCOv2(dims)
    disco.start()
    try:
        c = _Counter(per_head=3)
        monkeypatch.setattr(rzm, "rail_objective", c.wrap(real))
        with pytest.raises(RuntimeError) as ei:
            rzm.run_zombi_main(optimizing_dims=dims, checkpoint_dir=str(ckpt),
                               hparams_path=_fast_hparams(two_head_env / "hp.json"),
                               new_run_uuid="th01", two_head=True)
        assert isinstance(ei.value.__cause__, _Stop)
        r0, r1 = ckpt / "run_th01", ckpt / "run_th01_h1"
        assert (r0 / rzm.TWO_HEAD_STATE_NAME).exists() and (r1 / "snapshots").exists()
        log = [json.loads(l) for l in (r0 / "rails_log.jsonl").read_text().splitlines()]
        res = [e for e in log if e["event"] == "result"]
        assert res and all(e["how"] == "slot_ts" for e in res)
        assert all(e["slot"] == ("main" if e["head"] == 0 else "cache") for e in log
                   if "head" in e)
        assert sum(e["head"] == 0 for e in res) == 3 and sum(e["head"] == 1 for e in res) == 3
        # optimisation-phase prints: one per commanded line, never a re-print
        sends = {(e["slot"], e["slot_ts"]) for e in log if e["event"] == "send"}
        printed = [(s, sts) for _, s, sts, _ in disco.prints if (s, sts) in sends]
        assert len(printed) == len(set(printed))

        from src.utils.datahandler import DataHandler
        dhs = []
        for u in ("th01", "th01_h1"):
            dh = DataHandler(directory=str(ckpt), run_uuid=u, is_resume=True,
                             device="cuda", dtype=torch.float64, d=3)
            dh.load_state()
            dhs.append(dh)
        # each head holds the other's points as foreign rows
        assert all(dh.foreign_mask().sum().item() > 0 for dh in dhs)
        n_init = dhs[0]._n_init_points
        assert n_init == dhs[1]._n_init_points > 0

        # resume: both heads come back, a couple more lines each
        c2 = _Counter(per_head=1)
        monkeypatch.setattr(rzm, "rail_objective", c2.wrap(real))
        with pytest.raises(RuntimeError):
            rzm.run_zombi_main(resume_uuid="th01", optimizing_dims=dims,
                               checkpoint_dir=str(ckpt))
        assert c2.n == {0: 1, 1: 1}
        with pytest.raises(ValueError):
            rzm.run_zombi_main(resume_uuid="th01_h1", optimizing_dims=dims,
                               checkpoint_dir=str(ckpt))
    finally:
        disco.stop()
