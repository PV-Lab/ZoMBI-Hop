"""Axial initialization through the hardware path (scripts/run_zombi_main.py).

A fake DiSCO plays the other end of the databases the way the real one behaves
(DiSCO Full_System_Integration: composition_receiver, archerfish.py, objective_sender),
at accelerated speed:
  * it mirrors ZoMBI's compositions.db on every new timestamp, REPLACING both slots;
  * each of two transporters takes `main` (else `cache`) only when it is at the
    printer and the printer is free, prints, and comes back;
  * every printed slide reports one objective packet some time later (stability),
    written exactly as communication.objective_receiver writes it - unless the test
    makes that slide fail, in which case nothing is ever sent (objective_sender never
    sends an empty packet).
Nothing on the ZoMBI side is mocked except where a test needs to stop the run.
"""

import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import scripts.run_zombi_main as rzm
from scripts import communication
from scripts.initialize_databases import initialize_db
from src.utils.axial_init import AxialTrace

OBJ_DB = "./sql/objective.db"
COMP_DB = "./sql/compositions.db"


def _landscape(x):
    x = np.asarray(x, float)
    c = np.full(x.shape[-1], 1.0 / x.shape[-1])
    return 0.5 + 0.5 * np.exp(-((x - c) ** 2).sum(-1) / 0.05)


class FakeDiSCO(threading.Thread):
    """Two transporters, one printer, lagged per-slide results (seconds, accelerated)."""

    def __init__(self, dims, *, print_s=0.15, return_s=0.1, lag_s=0.6,
                 transporters=("front", "back"), fail=None, nan_row_p=0.0,
                 all_nan=None, noise=0.01, seed=0, max_hold_s=2.0):
        super().__init__(daemon=True)
        self.dims = list(dims)
        self.print_s, self.return_s, self.lag_s = print_s, return_s, lag_s
        self.transporters = list(transporters)
        self.fail = fail or (lambda k, rows: False)          # silent failure of print k
        self.all_nan = all_nan or (lambda k, rows: False)    # report print k as all-NaN
        self.nan_row_p = nan_row_p
        self.noise = noise
        # Results are held back while ZoMBI has not consumed the previous packet (so
        # tests are deterministic), but only briefly: the real DiSCO never waits.
        self.max_hold_s = max_hold_s
        self.rng = np.random.default_rng(seed)
        self.stop_evt = threading.Event()
        self.writes, self.prints, self.results = [], [], []   # observations for asserts
        self.overwrites = 0

    # -- databases ---------------------------------------------------------
    def _read_zombi(self):
        try:
            conn = sqlite3.connect(COMP_DB, timeout=5)
            row = conn.execute("SELECT ts FROM timestamp").fetchone()
            if row is None:
                conn.close()
                return None
            main = np.array(conn.execute("SELECT * FROM compositions").fetchall(), float)
            cache = np.array(conn.execute("SELECT * FROM compositions_cache").fetchall(), float)
            conn.close()
            return float(row[0]), main, cache
        except sqlite3.Error:
            return None

    def _flag(self):
        try:
            conn = sqlite3.connect(OBJ_DB, timeout=5)
            v = conn.execute("SELECT new_objective_available FROM handshake WHERE id=1").fetchone()
            conn.close()
            return v[0] if v else 0
        except sqlite3.Error:
            return 1

    def _emit(self, k, rows):
        meas = rows + self.rng.normal(0, self.noise, rows.shape) * (rows > 0)
        meas = np.clip(meas, 0, None)
        meas /= meas.sum(axis=1, keepdims=True)
        y = _landscape(meas[:, self.dims])
        if self.all_nan(k, rows):
            y[:] = np.nan
        else:
            y[self.rng.random(len(y)) < self.nan_row_p] = np.nan
        # what objective_receiver does with a packet: rows, comps, then the flag
        communication._write_objective_row(y.tolist(), OBJ_DB)
        communication._write_compositions_matrix(meas, OBJ_DB)
        communication._set_objective_ready(OBJ_DB)
        self.results.append((time.time(), k))

    # -- main loop ---------------------------------------------------------
    def run(self):
        got = self._read_zombi()
        last_ts = got[0] if got is not None else None      # whatever is there predates us
        slots = {"main": None, "cache": None}               # [rows, unread]
        at_printer = {t: 0.0 for t in self.transporters}    # time the rail is back
        printer_free = 0.0
        due = []                                            # (time, print#, rows)
        while not self.stop_evt.is_set():
            now = time.time()
            got = self._read_zombi()
            if got is not None and got[0] != last_ts:
                last_ts = got[0]
                for s in ("main", "cache"):
                    if slots[s] is not None and slots[s][1]:
                        self.overwrites += 1
                slots = {"main": [got[1], True], "cache": [got[2], True]}
                self.writes.append((now, got[1], got[2]))
            for t in self.transporters:
                if at_printer[t] > now or printer_free > now:
                    continue
                s = "main" if slots["main"] and slots["main"][1] else \
                    "cache" if slots["cache"] and slots["cache"][1] else None
                if s is None:
                    continue
                rows = slots[s][0]
                slots[s][1] = False
                k = len(self.prints)
                self.prints.append((now, t, s, rows))
                printer_free = now + self.print_s
                at_printer[t] = printer_free + self.return_s
                if not self.fail(k, rows):
                    due.append((printer_free + self.lag_s, k, rows))
            due.sort(key=lambda d: d[0])
            if due and due[0][0] <= now and (self._flag() == 0
                                             or now - due[0][0] > self.max_hold_s):
                _, k, rows = due.pop(0)
                self._emit(k, rows)
            time.sleep(0.02)

    def stop(self):
        self.stop_evt.set()
        self.join(timeout=10)


@pytest.fixture()
def hw_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("sql").mkdir()
    initialize_db()
    import src.utils.needles_db as ndb
    monkeypatch.setattr(ndb, "DEFAULT_NEEDLES_DB_PATH", tmp_path / "sql" / "needles.db")
    monkeypatch.setattr(rzm, "_COMP_LOG", {"path": None, "buffer": [], "call": 0})
    # watchdog scaled to the accelerated DiSCO (seconds instead of hours)
    monkeypatch.setattr(rzm, "INIT_FIRST_RESULT_TIMEOUT_MIN", 0.25)
    monkeypatch.setattr(rzm, "INIT_STALL_FLOOR_MIN", 0.05)
    monkeypatch.setattr(rzm, "INIT_STALL_MULT", 3.0)
    monkeypatch.setattr(rzm, "INIT_MIN_WRITE_GAP_S", 0.5)
    monkeypatch.setattr(rzm, "INIT_MIN_RESULT_LATENCY_MIN", 0.0)   # results come back in ~1 s here
    return tmp_path


def _bounds(torch, d, hi=None):
    b = torch.zeros(2, d, dtype=torch.float64)
    b[1] = 1.0 if hi is None else torch.tensor(hi, dtype=torch.float64)
    return b


def _state(run_dir):
    return json.loads((run_dir / rzm.INIT_STATE_NAME).read_text())


def _trace_of_rows(state, rows, dims):
    """Index of the init trace a printed 10-col row block was sent for."""
    traces = [AxialTrace.from_dict(t) for t in state["queue"]["traces"]]
    x = rows[:, dims]
    d = [np.linalg.norm(x[0] - t.end) + np.linalg.norm(x[-1] - t.start) for t in traces]
    return int(np.argmin(d))


def _run_init(hw_env, torch, dims, hi=None, name="run_t", **disco_kw):
    disco = FakeDiSCO(dims, **disco_kw)
    disco.start()
    run_dir = hw_env / name
    run_dir.mkdir(exist_ok=True)
    try:
        t0 = time.time()
        out = rzm.run_axial_initialization(run_dir, name, _bounds(torch, len(dims), hi), "cpu")
        wall = time.time() - t0
    finally:
        disco.stop()
    return out, disco, run_dir, wall


def test_each_result_sends_the_next_queued_pair(hw_env, monkeypatch, torch):
    """6-dim campaign box: the next pair goes out on every single result - never
    after waiting for both rails - and every trace ends up measured."""
    dims, hi = [0, 2, 3, 4, 8, 9], [1, 1, 1, 0.3, 1, 1]
    monkeypatch.setattr(rzm, "OPTIMIZING_DIMS", dims)
    # the two rails' results arrive a full second apart (longer than the write gap)
    (Xa, Xe, Y), disco, run_dir, _ = _run_init(hw_env, torch, dims, hi, noise=0.0,
                                                print_s=1.0)
    st = _state(run_dir)
    assert st["complete"]
    assert all(s == "done" for s in st["queue"]["status"])
    assert Xa.shape[1] == 6 and Xa.shape[0] >= 6 * 24 and Xe.shape == Xa.shape
    assert (Xa.numpy() <= np.array(hi) + 1e-3).all()
    # ZoMBI's own record: every pair after the first follows a result, and the second
    # pair went out after the FIRST result - before the other rail had reported (a
    # both-rails rule would have waited for it)
    write_t = [w["t"] for w in st["queue"]["writes"]]
    res_t = st["queue"]["result_times"]
    assert len(write_t) >= 3
    for w0, w1 in zip(write_t, write_t[1:]):
        assert any(w0 < r <= w1 for r in res_t), "a pair was sent without a result"
    assert res_t[0] < write_t[1] < res_t[1]
    assert len(disco.writes) == len(write_t)             # DiSCO saw every one of them
    # every trace was printed, in print order face -> pure
    printed = {_trace_of_rows(st, rows, dims) for _, _, _, rows in disco.prints}
    assert printed == set(range(6))
    tr0 = AxialTrace.from_dict(st["queue"]["traces"][0])
    first_rows = disco.writes[0][1]
    assert first_rows[0, dims[tr0.component]] == pytest.approx(tr0.end[tr0.component], abs=2e-3)
    assert first_rows[-1, dims[tr0.component]] == pytest.approx(tr0.start[tr0.component], abs=2e-3)


def test_silent_failure_does_not_block(hw_env, monkeypatch, torch):
    """DiSCO sends nothing for a slide that fails completely. The init must not wait
    for it: the trace is re-sent and still gets measured."""
    dims = [0, 2, 8, 9]
    monkeypatch.setattr(rzm, "OPTIMIZING_DIMS", dims)
    failed = []

    def fail_second_print(k, rows):
        if k == 1:
            failed.append(rows)
            return True
        return False

    (Xa, _, _), disco, run_dir, wall = _run_init(hw_env, torch, dims, fail=fail_second_print)
    st = _state(run_dir)
    assert st["complete"] and all(s == "done" for s in st["queue"]["status"])
    lost = _trace_of_rows(st, failed[0], dims)
    assert st["queue"]["sends"][lost] >= 2                   # it was re-sent
    assert any(len(b["y"]) for b in st["queue"]["data"][lost])
    assert wall < 120


def test_trace_that_always_fails_is_skipped_not_waited_for(hw_env, monkeypatch, torch):
    dims = [0, 2, 8, 9]
    monkeypatch.setattr(rzm, "OPTIMIZING_DIMS", dims)
    target = {}

    def always_fail_one(k, rows):
        key = tuple(np.round(rows[0], 3))
        target.setdefault("key", key)              # the first trace printed always fails
        return key == target["key"]

    (Xa, _, _), disco, run_dir, wall = _run_init(hw_env, torch, dims, fail=always_fail_one)
    st = _state(run_dir)
    assert st["complete"]
    assert st["queue"]["status"].count("abandoned") == 1
    assert st["queue"]["status"].count("done") == 3
    assert max(st["queue"]["sends"]) == rzm.INIT_MAX_SENDS
    assert Xa.shape[0] >= 3 * 20
    assert wall < 180


def test_nan_rows_and_all_nan_packets_are_fine(hw_env, monkeypatch, torch):
    """Partial-NaN results keep their usable rows; an all-NaN packet (if DiSCO ever
    sends one) completes its trace with no data instead of holding the queue."""
    dims = [0, 2, 8, 9]
    monkeypatch.setattr(rzm, "OPTIMIZING_DIMS", dims)
    (Xa, _, Y), disco, run_dir, wall = _run_init(
        hw_env, torch, dims, nan_row_p=0.3, all_nan=lambda k, rows: k == 0)
    st = _state(run_dir)
    assert st["complete"] and all(s == "done" for s in st["queue"]["status"])
    assert np.isfinite(Y.numpy()).all() and np.isfinite(Xa.numpy()).all()
    empty = [i for i, blocks in enumerate(st["queue"]["data"])
             if blocks and not any(len(b["y"]) for b in blocks)]
    assert len(empty) == 1                                    # the all-NaN slide
    assert wall < 120


def test_stale_results_are_not_taken_for_init_traces(hw_env, monkeypatch, torch):
    """A result flagged before init starts is cleared; one arriving later for an
    unrelated line is ignored and never enters the init data."""
    dims = [0, 2, 3, 4, 8, 9]
    monkeypatch.setattr(rzm, "OPTIMIZING_DIMS", dims)
    stale = np.zeros((24, 10))
    stale[:, dims] = np.linspace([0.3, 0.3, 0.1, 0.1, 0.1, 0.1],
                                 [0.1, 0.1, 0.3, 0.1, 0.1, 0.3], 24)
    # (1) already flagged when the run starts
    communication._write_objective_row(np.full(24, 0.99).tolist(), OBJ_DB)
    communication._write_compositions_matrix(stale, OBJ_DB)
    communication._set_objective_ready(OBJ_DB)
    # (2) arrives a moment after the first pair is sent
    def late_stale():
        time.sleep(0.3)
        communication._write_objective_row(np.full(24, 0.98).tolist(), OBJ_DB)
        communication._write_compositions_matrix(stale, OBJ_DB)
        communication._set_objective_ready(OBJ_DB)
    threading.Thread(target=late_stale, daemon=True).start()
    # results take >= 1.6 s here; anything sooner after a send is a leftover
    monkeypatch.setattr(rzm, "INIT_MIN_RESULT_LATENCY_MIN", 1.0 / 60)
    (Xa, _, Y), disco, run_dir, _ = _run_init(hw_env, torch, dims, lag_s=1.5)
    st = _state(run_dir)
    assert st["complete"]
    assert not np.any(np.isclose(Y.numpy(), 0.99) | np.isclose(Y.numpy(), 0.98))
    assert all(s == "done" for s in st["queue"]["status"])


def test_regular_objective_still_returns_on_first_rail(hw_env, monkeypatch, torch):
    """The optimisation-loop objective() is unchanged: it returns on the first result."""
    dims = [0, 8, 9]
    monkeypatch.setattr(rzm, "OPTIMIZING_DIMS", dims)
    disco = FakeDiSCO(dims, lag_s=0.5)
    disco.start()
    try:
        ep = torch.tensor([[[1, 0, 0], [0, .5, .5]], [[0, 1, 0], [.5, 0, .5]]],
                          dtype=torch.float64)
        x, y = rzm.objective(ep, ready_for_objectives=True)
    finally:
        disco.stop()
    assert x.shape == (24, 3) and y.shape == (24,)


def test_get_y_measurements_timeout_and_empty(hw_env, monkeypatch):
    monkeypatch.setattr(rzm, "OPTIMIZING_DIMS", [0, 8, 9])
    t0 = time.time()
    assert rzm.get_y_measurements(None, ready_for_objectives=True, timeout_s=1.5) is None
    assert 1.0 < time.time() - t0 < 5.0
    communication._write_objective_row([float("nan")] * 24, OBJ_DB)
    communication._set_objective_ready(OBJ_DB)
    y, x, idx = rzm.get_y_measurements(None, ready_for_objectives=True, return_indices=True,
                                       timeout_s=5, accept_empty=True)
    assert len(y) == 0 and x.shape == (0, 3)
    # a packet holding a single usable value is read, not waited on forever
    one = np.zeros((1, 10))
    one[0, [0, 8, 9]] = [0.2, 0.3, 0.5]
    communication._write_objective_row([0.7], OBJ_DB)
    communication._write_compositions_matrix(one, OBJ_DB)
    communication._set_objective_ready(OBJ_DB)
    y, x = rzm.get_y_measurements(None, ready_for_objectives=True, timeout_s=5)
    assert y.tolist() == [0.7] and x.shape == (1, 3)


class _Stop(BaseException):
    pass


def _fast_hparams(path):
    path.write_text(json.dumps({"hparams": {
        "n_restarts": 4, "raw": 32, "nat_grad_max_steps": 10, "max_iterations": 2}}))
    return str(path)


def test_end_to_end_no_prediction_before_init_complete(hw_env, monkeypatch, torch):
    """Full new hardware run: the optimiser's first line is requested only after the
    queue is done, and its snapshot holds the whole design."""
    dims = [0, 2, 8, 9]
    seen = []

    def stop_at_first_prediction(endpoints, **kw):
        seen.append(_state(hw_env / "ckpt" / "run_e2e1")["complete"])
        raise _Stop()

    monkeypatch.setattr(rzm, "objective", stop_at_first_prediction)
    disco = FakeDiSCO(dims)
    disco.start()
    try:
        with pytest.raises(_Stop):
            rzm.run_zombi_main(optimizing_dims=dims, checkpoint_dir=str(hw_env / "ckpt"),
                               hparams_path=_fast_hparams(hw_env / "hp.json"),
                               new_run_uuid="e2e1")
    finally:
        disco.stop()
    assert seen == [True]
    run_dir = hw_env / "ckpt" / "run_e2e1"
    st = _state(run_dir)
    n_pts = sum(len(b["y"]) for blocks in st["queue"]["data"] for b in blocks)
    snaps = sorted((run_dir / "snapshots").iterdir())
    delta = torch.load(snaps[0] / "delta.pt", map_location="cpu", weights_only=False)
    assert delta["n_init_points"] == n_pts >= 4 * 20
    live = json.loads((run_dir / "live_plot_state.json").read_text())
    assert len(live["x_actual"]) == n_pts
    log = [json.loads(l) for l in (run_dir / rzm.COMPOSITION_LOG_NAME).read_text().splitlines()]
    assert len(log) >= 4 and all(any(len(r["measured"]) for r in rec["rails"]) for rec in log)


def test_end_to_end_resume_during_initialization(hw_env, monkeypatch, torch):
    """Stop mid-initialization, then resume through run_zombi_main: measured traces
    are kept and not printed again; the rest are completed."""
    dims = [0, 2, 3, 4, 8, 9]
    ckpt = hw_env / "ckpt"
    real_get_y = rzm.get_y_measurements
    n = [0]

    def stop_after_two_results(*a, **kw):
        out = real_get_y(*a, **kw)
        if out is not None:
            n[0] += 1
            if n[0] == 3:
                raise _Stop()
        return out

    monkeypatch.setattr(rzm, "get_y_measurements", stop_after_two_results)
    disco = FakeDiSCO(dims)
    disco.start()
    try:
        with pytest.raises(_Stop):
            rzm.run_zombi_main(optimizing_dims=dims, checkpoint_dir=str(ckpt),
                               hparams_path=_fast_hparams(hw_env / "hp.json"),
                               new_run_uuid="e2e2")
    finally:
        disco.stop()
    run_dir = ckpt / "run_e2e2"
    assert not (run_dir / "snapshots").exists()
    st = _state(run_dir)
    done_before = [i for i, s in enumerate(st["queue"]["status"]) if s == "done"]
    assert len(done_before) == 2

    monkeypatch.setattr(rzm, "get_y_measurements", real_get_y)
    monkeypatch.setattr(rzm, "objective", lambda *a, **k: (_ for _ in ()).throw(_Stop()))
    rzm._COMP_LOG.update({"path": None, "buffer": [], "call": 0})
    disco = FakeDiSCO(dims, seed=2)
    disco.start()
    try:
        with pytest.raises(_Stop):
            rzm.run_zombi_main(resume_uuid="e2e2", optimizing_dims=dims,
                               checkpoint_dir=str(ckpt),
                               hparams_path=_fast_hparams(hw_env / "hp.json"))
    finally:
        disco.stop()
    st = _state(run_dir)
    assert st["complete"] and all(s == "done" for s in st["queue"]["status"])
    reprinted = {_trace_of_rows(st, rows, dims) for _, _, _, rows in disco.prints}
    assert not (reprinted & set(done_before))
    assert (run_dir / "snapshots").exists()
