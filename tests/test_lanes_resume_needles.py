"""Lanes resume keeps the previous needles (exact penalty geometry) and mirrors them into needles.db (2026-11-15:
the lanes driver never wrote sql/needles.db, so DiSCO's visualization showed no stars).

Also the review findings of the same day: the mirror is opt-in (a test or harness never writes the live
sql/needles.db), a restart before the first decision / a torn latest.txt / a lost rows.jsonl line never loses the
needles, the timeline places restored zones in the process that declared them, and the newest needle's radial
median reaches needles.db after the lane's own-rows override."""
import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest

FIXTURE = (Path(__file__).resolve().parent.parent / "benchmarks" / "zoom_fix" / "lanes" / "report" / "review" / "impl"
           / "gui_plot" / "fixture_run")
DIMS = [0, 2, 3, 4, 8, 9]


@pytest.fixture
def drivers():
    """Builds lanes drivers and closes every engine at teardown (the engine installs the seeded random_simplex
    patch, Fix 4; a leaked one breaks tests/test_lanes_rng.py)."""
    built = []

    def make(run, **kw):
        import torch
        from scripts import lanes_hw
        ini = torch.load(Path(run) / "lanes" / "init.pt", map_location="cpu", weights_only=False)
        drv = lanes_hw.LanesHardwareDriver(str(run), ini["X_actual"], ini["X_expected"], ini["Y"], ini["bounds"],
                                           None, opt_dims=list(ini.get("opt_dims") or DIMS),
                                           y_to_optimizer=lambda v: v, seed=0, log=kw.pop("log", lambda *a, **k: None),
                                           **kw)
        built.append(drv)
        return drv

    yield make
    for drv in built:
        try:
            drv.engine.close()
        except Exception:
            pass


@pytest.fixture
def run_copy(tmp_path, monkeypatch):
    if not (FIXTURE / "lanes" / "engine_ckpt.pt").exists():
        pytest.skip("fixture_run not present")
    run = tmp_path / "run_f1x7"
    shutil.copytree(FIXTURE, run)
    db = tmp_path / "needles.db"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    from scripts import communication
    monkeypatch.setattr(communication, "write_slots4", lambda entries, *a, **k: None)
    monkeypatch.setattr(communication, "max_link_event_id", lambda *a, **k: 0)
    return run, db


def _saved(run):
    import torch
    ed = run / "lanes" / "engine" / "run_f1x7"
    snap = ed / "snapshots" / (ed / "latest.txt").read_text().strip()
    saved = torch.load(snap / "delta.pt", map_location="cpu", weights_only=False)
    if int(saved["needles"].shape[0]) == 0:
        pytest.skip("fixture snapshot has no needles")
    return saved


def _db(db):
    con = sqlite3.connect(db)
    try:
        return con.execute("select objective_value, radial_objective_value from needles order by rank").fetchall()
    finally:
        con.close()


def _same_geometry(dh, saved):
    import torch
    assert torch.allclose(dh.needles.cpu().double(), saved["needles"].double(), atol=1e-12)
    assert torch.allclose(dh.needle_penalty_radii.cpu().double(), saved["needle_penalty_radii"].double(), atol=1e-12)
    old, new = saved["penalty_mask"].reshape(-1).bool(), dh._penalty_mask.reshape(-1).bool().cpu()
    if old.numel() == new.numel():
        assert int((old != new).sum()) == 0                       # exact penalty geometry


def test_resume_restores_needles_and_mirrors_them(run_copy, drivers):
    run, db = run_copy
    saved = _saved(run)
    n_saved = int(saved["needles"].shape[0])
    drv = drivers(run, needles_db=str(db))
    dh = drv.engine.dh
    assert drv.n_needles_restored == n_saved
    assert int(dh.needles.shape[0]) == n_saved and len(dh.needles_results) == n_saved
    _same_geometry(dh, saved)
    vals = [r[0] for r in _db(db)]
    assert len(vals) == n_saved and vals == sorted(vals, reverse=True)
    # restored zone records are tagged with the process that declared them (timeline)
    nz = [z for z in drv.engine.zones if z["kind"] == "needle"]
    assert len(nz) == n_saved and all(z.get("restored") and z.get("t_proc") for z in nz)
    # persisted at once: a restart before the first decision still finds them
    assert (run / "lanes" / "needles_state.pt").exists()


def test_fresh_run_mirror_is_enabled_on_request(tmp_path, monkeypatch, drivers):
    """needles_db=True: ZOMBI_NEEDLES_DB (else the default path); empty table until the first needle."""
    db = tmp_path / "needles.db"
    monkeypatch.setenv("ZOMBI_NEEDLES_DB", str(db))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    if not (FIXTURE / "lanes" / "init.pt").exists():
        pytest.skip("fixture_run not present")
    from scripts import communication
    monkeypatch.setattr(communication, "max_link_event_id", lambda *a, **k: 0)
    run = tmp_path / "run_new1"
    (run / "lanes").mkdir(parents=True)
    shutil.copy(FIXTURE / "lanes" / "init.pt", run / "lanes" / "init.pt")
    drv = drivers(run, needles_db=True)
    assert drv.engine.dh._needles_db_path == str(db)
    con = sqlite3.connect(db)
    assert con.execute("select count(*) from needles").fetchone()[0] == 0
    con.close()


def test_mirror_is_off_by_default(run_copy, drivers):
    """A driver built without needles_db (tests, the xsys harness) never writes any needles.db: on 2026-11-15
    two driver tests emptied the live campaign's table through the default path."""
    import src.utils.needles_db as ndb
    run, db = run_copy
    _saved(run)
    drv = drivers(run)
    assert drv.needles_db_path is None and drv.engine.dh._needles_db_path is None
    assert drv.n_needles_restored > 0                     # the restore itself does not depend on the mirror
    assert not Path(ndb.DEFAULT_NEEDLES_DB_PATH).exists() and not db.exists()
    assert Path(os.environ["ZOMBI_NEEDLES_DB"]) == Path(ndb.DEFAULT_NEEDLES_DB_PATH)   # conftest isolation


def test_live_needles_db_is_refused_under_pytest(run_copy, drivers, tmp_path, monkeypatch):
    """Both live-file guards, exercised on a STAND-IN path (never the real file): the driver's (ZOMBIHOP_TESTING=1)
    and tests/conftest.py's write_needles / reset_needles_db wrappers."""
    import src.utils.needles_db as ndb
    from scripts import lanes_hw
    assert os.environ.get("ZOMBIHOP_TESTING") == "1"
    real_live = Path(__file__).resolve().parents[1] / "sql" / "needles.db"
    assert lanes_hw.live_needles_db_refused(real_live)
    assert ndb.write_needles.live == real_live and ndb.reset_needles_db.live == real_live
    stand_in = tmp_path / "live_stand_in" / "needles.db"
    monkeypatch.setattr(ndb.write_needles, "live", stand_in)
    monkeypatch.setattr(ndb.reset_needles_db, "live", stand_in)
    with pytest.raises(RuntimeError):
        ndb.write_needles([], DIMS, path=stand_in)
    with pytest.raises(RuntimeError):
        ndb.reset_needles_db(stand_in)
    monkeypatch.setattr(lanes_hw, "LIVE_NEEDLES_DB", stand_in)
    run, _ = run_copy
    drv = drivers(run, needles_db=str(stand_in))
    assert drv.engine.dh._needles_db_path is None
    assert not stand_in.exists()
    monkeypatch.setenv("ZOMBIHOP_TESTING", "0")
    assert not lanes_hw.live_needles_db_refused(stand_in)   # outside a test the path is an ordinary file


def test_double_restart_before_any_decision_keeps_needles(run_copy, drivers):
    """Restart, no decision (the new engine's 0001_init snapshot is 'latest', 0 needles), restart again."""
    run, db = run_copy
    saved = _saved(run)
    n_saved = int(saved["needles"].shape[0])
    a = drivers(run, needles_db=str(db))
    assert a.n_needles_restored == n_saved
    ed = run / "lanes" / "engine" / "run_f1x7"
    assert (ed / "latest.txt").read_text().strip().endswith("_init")
    a.engine.close()
    b = drivers(run, needles_db=str(db))
    assert b.n_needles_restored == n_saved
    _same_geometry(b.engine.dh, saved)
    assert len(_db(db)) == n_saved
    assert len([z for z in b.engine.zones if z["kind"] == "needle"]) == n_saved


def test_torn_latest_and_no_state_file_scans_the_snapshots(run_copy, drivers):
    run, db = run_copy
    saved = _saved(run)
    (run / "lanes" / "engine" / "run_f1x7" / "latest.txt").write_text("", encoding="utf-8")
    drv = drivers(run, needles_db=str(db))
    assert drv.n_needles_restored == int(saved["needles"].shape[0])
    _same_geometry(drv.engine.dh, saved)


def test_misaligned_rows_keep_the_penalty_geometry(run_copy, drivers):
    """rows.jsonl lost one early line: the saved row indices no longer point at the needles; the geometry is
    restored exactly and the indices re-derived (nearest row)."""
    import torch
    run, db = run_copy
    saved = _saved(run)
    n_init = int(torch.load(run / "lanes" / "init.pt", map_location="cpu", weights_only=False)["Y"].shape[0])
    rows_p = run / "lanes" / "rows.jsonl"
    lines = rows_p.read_text(encoding="utf-8").splitlines(True)
    recs = [json.loads(ln) for ln in lines[:6]]
    assert all(r.get("ev", "row") == "row" and r.get("x_actual") is not None for r in recs)
    gone = n_init + 5                                      # the global row index of the deleted line
    del lines[5]
    rows_p.write_text("".join(lines), encoding="utf-8")
    drv = drivers(run, needles_db=str(db))
    dh = drv.engine.dh
    assert drv.n_needles_restored == int(saved["needles"].shape[0])
    assert torch.allclose(dh.needles.cpu().double(), saved["needles"].double(), atol=1e-12)
    d = torch.norm(dh.X_all_actual[dh.needle_indices.reshape(-1)] - dh.needles, dim=1)
    assert float(d.max()) <= 1e-6                          # re-derived: each index is its needle's (shifted) row
    assert len(_db(db)) == int(saved["needles"].shape[0])
    # the penalty geometry itself: radii and ellipsoids exactly as saved ...
    assert torch.equal(dh.needle_penalty_radii.cpu().double().reshape(-1),
                       saved["needle_penalty_radii"].double().reshape(-1))
    has_M, M_stack = saved["needle_has_M"].reshape(-1), saved["needle_M_stack"]
    assert len(dh.needle_M_list) == int(has_M.numel()) == int(saved["needles"].shape[0])
    for i, M in enumerate(dh.needle_M_list):
        if bool(has_M[i]):
            assert M is not None and torch.equal(M.cpu().double(), M_stack[i].double())
        else:
            assert M is None
    # ... and the rebuilt penalty mask is the saved one without the deleted row
    old = saved["penalty_mask"].reshape(-1).bool()
    assert int(old.numel()) > gone
    expect = torch.cat([old[:gone], old[gone + 1:]])
    new = dh._penalty_mask.reshape(-1).bool().cpu()
    assert int(new.numel()) >= int(expect.numel())
    assert torch.equal(new[:int(expect.numel())], expect)
    assert int((~expect).sum()) > 0                        # the comparison covers penalised rows


def test_raw_rows_record_does_not_crash_the_resume(run_copy, drivers):
    run, db = run_copy
    _saved(run)
    with open(run / "lanes" / "rows.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": 0.0, "ev": "raw_rows", "qid": 999999, "wid": 1, "y_opt": [0.5]}) + "\n")
    logs = []
    drv = drivers(run, needles_db=str(db), log=lambda *a, **k: logs.append(" ".join(map(str, a))))
    assert drv.n_needles_restored > 0
    assert any("without a composition skipped" in s for s in logs)


def test_restored_zones_stay_in_their_process_on_the_timeline(run_copy, drivers):
    """The new process's engine_ckpt.pt carries the restored zones (old act ids): the timeline must not turn
    them into needle-ended dives of the new process (phantom blocks, an inflated clean count)."""
    from visualization import lanes_timeline as LT
    run, db = run_copy
    _saved(run)
    R0 = LT.load_lanes_run(run)
    before = sorted((nd["session"], nd["act"]) for nd in R0.needles)

    def tee(*a, **k):                                      # run_lanes_hardware's log is the GUI's run.log tee
        with open(run / "run.log", "a", encoding="utf-8") as fh:
            fh.write(" ".join(str(x) for x in a) + "\n")
    tee("=" * 80)
    tee("STARTING LANES OPTIMIZATION (preset=leader; four-slot protocol)")
    tee("=" * 80)
    drv = drivers(run, needles_db=str(db), log=tee)
    assert drv.n_needles_restored > 0
    drv._checkpoint()                                      # what the first result after the restart does
    R1 = LT.load_lanes_run(run)
    after = sorted((nd["session"], nd["act"]) for nd in R1.needles)
    assert R1.n_sessions == R0.n_sessions + 1
    assert after == before
    last = R1.n_sessions - 1
    assert not [k for k, v in R1.acts.items() if k[0] == last and v["reason"] == "needle"]


def test_new_needle_after_restore_persists_and_resyncs_the_median(run_copy, drivers):
    """A needle declared after the restore: lane.py overrides its median with the lane's own-rows median AFTER
    add_needle's sync; the driver resyncs needles.db, writes needles_state.pt and the zone record."""
    import torch
    run, db = run_copy
    saved = _saved(run)
    n0 = int(saved["needles"].shape[0])
    drv = drivers(run, needles_db=str(db))
    eng, dh, z = drv.engine, drv.engine.dh, drv.engine.z
    ln = drv.lanes["front"]
    unpen = torch.where(dh._penalty_mask.reshape(-1))[0]
    pick = int(unpen[int(dh.Y_all.reshape(-1)[unpen].argmax())])
    with eng.context(ln):
        z._declare_needle_from_point(dh, dh.X_all_actual[pick], dh.Y_all[pick], pick, zoom=2, iteration=0,
                                     reason="EI convergence")
    # persisted by add_needle itself (a kill before the decision's snapshot keeps it)
    st0 = torch.load(run / "lanes" / "needles_state.pt", map_location="cpu", weights_only=False)
    assert int(st0["tensors"]["needles"].shape[0]) == n0 + 1
    dh.needles_results[n0]["median_value"] = 123.0         # the lanes own-rows override (lane.py)
    dh.needles_results[n0]["lane"] = ln.name
    eng.add_zone("needle", ln, n0, pick=pick)
    drv._on_needles_maybe_changed()                        # what _decide runs after every decision
    assert int(dh.needle_indices.reshape(-1)[n0]) == pick
    assert 123.0 in [r[1] for r in _db(db)]
    st = torch.load(run / "lanes" / "needles_state.pt", map_location="cpu", weights_only=False)
    assert int(st["tensors"]["needles"].shape[0]) == n0 + 1 and st["records"][n0]["median_value"] == 123.0
    ck = torch.load(run / "lanes" / "engine_ckpt.pt", map_location="cpu", weights_only=False)
    assert any(zr["kind"] == "needle" and zr["index"] == n0 and not zr.get("restored") for zr in ck["zones"])
    # and a restart now (no decision snapshot since) restores all of them, the new one with its lane
    drv.engine.close()
    b = drivers(run, needles_db=str(db))
    assert b.n_needles_restored == n0 + 1
    assert b.engine.dh.needles_results[n0]["lane"] == "front"
    assert b.engine.dh.needles_results[n0]["median_value"] == 123.0


def test_kill_right_after_a_needle_declaration_keeps_it(run_copy, drivers):
    """ZoMBI stopped between add_needle and the decision's snapshot (main.py hard-terminates on Stop)."""
    import torch
    run, db = run_copy
    n0 = int(_saved(run)["needles"].shape[0])
    drv = drivers(run, needles_db=str(db))
    eng, dh, z = drv.engine, drv.engine.dh, drv.engine.z
    unpen = torch.where(dh._penalty_mask.reshape(-1))[0]
    pick = int(unpen[int(dh.Y_all.reshape(-1)[unpen].argmax())])
    with eng.context(drv.lanes["back"]):
        z._declare_needle_from_point(dh, dh.X_all_actual[pick], dh.Y_all[pick], pick, zoom=2, iteration=0,
                                     reason="EI convergence")
    mask = dh._penalty_mask.clone().cpu()
    drv.engine.close()                                     # killed here: no zone record, no snapshot, no hook
    b = drivers(run, needles_db=str(db))
    assert b.n_needles_restored == n0 + 1
    assert torch.equal(b.engine.dh._penalty_mask.cpu(), mask)
    assert len(_db(db)) == n0 + 1


def test_sync_tool_heals_a_wipe_and_never_downgrades(run_copy):
    from scripts import lanes_needles_sync as S
    run, db = run_copy
    saved = _saved(run)
    n = int(saved["needles"].shape[0])
    st, w = S.sync_once(run, str(db), None)
    assert w == n and len(_db(db)) == n
    st, w = S.sync_once(run, str(db), st)
    assert w is None                                       # the table already holds the set
    con = sqlite3.connect(db)
    con.execute("delete from needles")
    con.commit()
    con.close()
    st, w = S.sync_once(run, str(db), st)                  # an external wipe heals at the next tick
    assert w == n and len(_db(db)) == n
    ed = run / "lanes" / "engine" / "run_f1x7"
    # a snapshot with FEWER (but > 0) needles is 'latest' (a restarted process's early snapshots, a torn
    # pointer): nothing is written and the remembered set is kept
    fewer = None
    for sd in sorted((ed / "snapshots").iterdir()):
        k = int(json.loads((sd / "summary.json").read_text(encoding="utf-8")).get("n_needles") or 0)
        if 0 < k < n:
            fewer = sd.name
    assert fewer is not None
    (ed / "latest.txt").write_text(fewer, encoding="utf-8")
    st, w = S.sync_once(run, str(db), st)
    assert w is None and len(_db(db)) == n and len(st) == n
    # 0001_init is 'latest' (a restarted process's first seconds) AND the table is wiped: the remembered set heals it
    (ed / "latest.txt").write_text("0001_init", encoding="utf-8")
    st, w = S.sync_once(run, str(db), st)
    assert w is None and len(_db(db)) == n and len(st) == n
    con = sqlite3.connect(db)
    con.execute("delete from needles")
    con.commit()
    con.close()
    st, w = S.sync_once(run, str(db), st)
    assert w == n and len(_db(db)) == n
    # the table holds MORE rows than the set (a live mirror just added a needle no snapshot shows yet): never
    # downgraded
    extra = dict(st[0], value=0.001, median_value=0.001,
                 point=[min(1.0, float(v) + 0.01) for v in st[0]["point"]])
    S.write_needles(list(st) + [extra], S.run_dims(run), path=str(db))
    held = _db(db)
    assert len(held) == n + 1
    st, w = S.sync_once(run, str(db), st)
    assert w is None and _db(db) == held and len(st) == n


# ── hardening (2026-11-15 verification findings H1-H8) ──
def _journal(run, ev):
    p = run / "lanes" / "journal.jsonl"
    return [r for r in (json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
            if r.get("ev") == ev]


def _logger(logs):
    return lambda *a, **k: logs.append(" ".join(str(x) for x in a))


def test_needle_read_failure_never_stops_the_driver(run_copy, drivers, monkeypatch):
    """H1: an exception while reading the previous needles is journalled; the driver (= the run) still starts."""
    from scripts import lanes_hw
    run, db = run_copy
    _saved(run)

    def boom(self):
        raise RuntimeError("forced needle read failure")
    monkeypatch.setattr(lanes_hw.LanesHardwareDriver, "_read_needle_state", boom)
    logs = []
    drv = drivers(run, needles_db=str(db), log=_logger(logs))
    assert drv.n_needles_restored == 0 and drv._prev_needles is None
    assert any("forced needle read failure" in r["error"] for r in _journal(run, "needles_read_failed"))
    assert any("WARNING could not read the previous needles" in s for s in logs)


@pytest.mark.parametrize("latest", ["ok", "torn"])
def test_unreadable_snapshot_dir_never_stops_the_driver(run_copy, drivers, monkeypatch, latest):
    """H1: the snapshot scan's directory listing raising (a Dropbox / permission error) is a WARNING + journal
    entry; with a readable latest snapshot the needles are still restored from it."""
    import sys
    run, db = run_copy
    n = int(_saved(run)["needles"].shape[0])
    if latest == "torn":
        (run / "lanes" / "engine" / "run_f1x7" / "latest.txt").write_text("", encoding="utf-8")
    real_iterdir = Path.iterdir

    def iterdir(self):
        if self.name == "snapshots" and sys._getframe(1).f_code.co_filename.endswith("lanes_hw.py"):
            raise PermissionError(13, "forced: access is denied", str(self))
        return real_iterdir(self)
    monkeypatch.setattr(Path, "iterdir", iterdir)
    logs = []
    drv = drivers(run, needles_db=str(db), log=_logger(logs))
    assert _journal(run, "needles_scan_failed")
    assert any("could not scan the engine snapshots" in s for s in logs)
    if latest == "ok":
        assert drv.n_needles_restored == n
    else:
        assert drv.n_needles_restored == 0
        assert any("needles NOT restored" in s and f"name {n} needle(s)" in s for s in logs)
        assert _journal(run, "needles_not_restored")


def test_scan_restores_when_latest_is_init_state_missing_and_zones_stale(run_copy, drivers):
    """H2: latest.txt points at a 0001_init snapshot (0 needles), there is no needles_state.pt and the checkpoint's
    zone records are stale (written before the first needle): the always-on snapshot scan restores them all."""
    import torch
    run, db = run_copy
    saved = _saved(run)
    n = int(saved["needles"].shape[0])
    ed = run / "lanes" / "engine" / "run_f1x7"
    assert (ed / "snapshots" / "0001_init").is_dir() and not (run / "lanes" / "needles_state.pt").exists()
    (ed / "latest.txt").write_text("0001_init", encoding="utf-8")
    ck_p = run / "lanes" / "engine_ckpt.pt"
    ck = torch.load(ck_p, map_location="cpu", weights_only=False)
    ck["zones"] = []
    ck["events"] = [e for e in ck["events"] if e.get("kind") != "needle"]
    torch.save(ck, ck_p)
    drv = drivers(run, needles_db=str(db))
    assert drv.n_needles_restored == n
    _same_geometry(drv.engine.dh, saved)
    assert len(_db(db)) == n


def test_needle_state_write_retries_and_never_raises(run_copy, drivers, monkeypatch):
    """H3: torch.save on a file Dropbox holds raises RuntimeError: retried (<= 3 tries, <= 0.5 s of waiting),
    never raised into the decision path, and the set counts as persisted only once written (the next decision
    retries)."""
    import time

    import torch
    run, db = run_copy
    _saved(run)
    drv = drivers(run, needles_db=str(db))
    dh = drv.engine.dh
    st_p = run / "lanes" / "needles_state.pt"
    real_save = torch.save
    ctl = dict(calls=0, fail=0, slept=0.0)

    def flaky(obj, f, *a, **k):
        if str(f).endswith("needles_state.pt"):
            ctl["calls"] += 1
            if ctl["fail"] < 0 or ctl["calls"] <= ctl["fail"]:
                raise RuntimeError("PytorchStreamWriter failed writing file data/0: file write failed")
        return real_save(obj, f, *a, **k)

    def sleep(s):
        ctl["slept"] += float(s)
    monkeypatch.setattr(torch, "save", flaky)
    monkeypatch.setattr(time, "sleep", sleep)

    def medians():
        return [r["median_value"] for r in torch.load(st_p, map_location="cpu", weights_only=False)["records"]]
    # one failure, then success
    dh.needles_results[0]["median_value"] = 0.123
    ctl.update(calls=0, fail=1)
    drv._on_needles_maybe_changed()
    assert ctl["calls"] == 2 and medians()[0] == 0.123
    assert drv._needle_sig_last == drv._needle_sig()
    # always failing: no exception, the set is NOT marked persisted
    dh.needles_results[0]["median_value"] = 0.456
    ctl.update(calls=0, fail=-1, slept=0.0)
    drv._on_needles_maybe_changed()
    assert ctl["calls"] == 3 and ctl["slept"] <= 0.5
    assert drv._needle_sig_last != drv._needle_sig() and medians()[0] == 0.123
    assert drv._save_needle_state() is False and drv._persist_needles() is False
    assert _journal(run, "needle_state_write_failed")
    # the next decision retries and succeeds
    ctl.update(calls=0, fail=0)
    drv._on_needles_maybe_changed()
    assert ctl["calls"] == 1 and medians()[0] == 0.456 and drv._needle_sig_last == drv._needle_sig()


@pytest.mark.parametrize("how", ["raise", "zero"])
def test_failed_restore_keeps_the_unrestored_needles(run_copy, drivers, monkeypatch, how):
    """H4: a failed restore (an exception, or needles read but none restored) saves the unrestored state to NEW
    files before anything can overwrite lanes/needles_state.pt with a smaller set; CRITICAL + journal."""
    import torch
    from scripts import lanes_hw
    run, db = run_copy
    n = int(_saved(run)["needles"].shape[0])
    a = drivers(run, needles_db=str(db))
    assert a.n_needles_restored == n
    a.engine.close()
    st_p = run / "lanes" / "needles_state.pt"
    before = st_p.read_bytes()

    def broken(self, st):
        if how == "raise":
            raise RuntimeError("forced restore failure")
        return 0
    monkeypatch.setattr(lanes_hw.LanesHardwareDriver, "_apply_needle_state", broken)
    logs = []
    b = drivers(run, needles_db=str(db), log=_logger(logs))
    assert b.n_needles_restored == 0
    assert st_p.read_bytes() == before                     # not overwritten by the restart
    un = sorted((run / "lanes").glob("needles_state.unrestored_*.pt"))
    kept = [p for p in un if not p.name.endswith(".copy.pt")]
    copies = [p for p in un if p.name.endswith(".copy.pt")]
    assert len(kept) == 1 and len(copies) == 1
    assert copies[0].read_bytes() == before
    s = torch.load(kept[0], map_location="cpu", weights_only=False)
    assert int(s["tensors"]["needles"].shape[0]) == n and len(s["zones"]) == n
    assert any("CRITICAL" in x and str(kept[0]) in x for x in logs)
    assert [r["n"] for r in _journal(run, "needles_unrestored_saved")] == [n]
    b._on_needles_maybe_changed()                          # what every decision runs: nothing shrinks the file
    assert st_p.read_bytes() == before


def _diamonds(run):
    """(session, act) -> (star, declared) display positions of the needles (visualization/lanes_timeline.py)."""
    from visualization import lanes_timeline as LT
    L = LT.dive_layout(LT.load_lanes_run(run))
    return {(nd["session"], nd["act"]): (nd["star"], nd["declared"]) for nd in L.needles}


def test_restored_zones_keep_their_declaration_rows(run_copy, drivers):
    """H5: a restored zone carries its declaration's n_rows (from the checkpoint's needle event) into the new
    process's zones, needles_state.pt and engine_ckpt.pt, so the timeline's 'declared' diamonds do not jump to the
    end of their dive after a restart - also after a second restart."""
    import torch
    run, db = run_copy
    _saved(run)
    ck0 = torch.load(run / "lanes" / "engine_ckpt.pt", map_location="cpu", weights_only=False)
    ev = {(e["lane"], int(e["act"]), int(e["needle"])): e for e in ck0["events"] if e.get("kind") == "needle"}
    d0 = _diamonds(run)
    assert d0 and all(dc is not None for _, dc in d0.values())

    def tee(*a, **k):                                      # run_lanes_hardware's log is the GUI's run.log tee
        with open(run / "run.log", "a", encoding="utf-8") as fh:
            fh.write(" ".join(str(x) for x in a) + "\n")

    def restart():
        tee("=" * 80)
        tee("STARTING LANES OPTIMIZATION (preset=leader; four-slot protocol)")
        tee("=" * 80)
        return drivers(run, needles_db=str(db), log=tee)

    def check(zones):
        nz = [z for z in zones if z["kind"] == "needle"]
        assert len(nz) == len(ev)
        for z in nz:
            e = ev[(z["lane"], int(z["act"]), int(z["index"]))]
            assert z["n_rows"] == e["n_rows"] and z["value"] == e["value"]

    for _ in range(2):
        drv = restart()
        assert drv.n_needles_restored == len(ev)
        check(drv.engine.zones)
        check(torch.load(run / "lanes" / "needles_state.pt", map_location="cpu", weights_only=False)["zones"])
        drv._checkpoint()                                  # what the first result after the restart does
        check(torch.load(run / "lanes" / "engine_ckpt.pt", map_location="cpu", weights_only=False)["zones"])
        assert _diamonds(run) == d0
        drv.engine.close()


def test_short_name_spelling_of_the_live_file_is_refused(tmp_path, monkeypatch):
    """H7: an 8.3 short-name spelling of the live file is refused by both guards (lanes_hw.live_needles_db_refused
    and tests/conftest.py's write_needles / reset_needles_db wrappers) - on a STAND-IN, never the real file."""
    import ctypes

    import src.utils.needles_db as ndb
    from scripts import lanes_hw
    if os.name != "nt":
        pytest.skip("8.3 short names are a Windows feature")

    def short(p):
        buf = ctypes.create_unicode_buffer(1024)
        k = ctypes.windll.kernel32.GetShortPathNameW(str(p), buf, 1024)
        return buf.value if 0 < k < 1024 else None
    d = tmp_path / "live stand in folder with a long name"
    d.mkdir()
    stand_in = d / "needles.db"
    s_dir = short(d)
    if not s_dir or os.path.basename(s_dir).lower() == d.name.lower():
        pytest.skip("no 8.3 short names on this volume")
    monkeypatch.setattr(lanes_hw, "LIVE_NEEDLES_DB", stand_in)
    monkeypatch.setattr(ndb.write_needles, "live", stand_in)
    monkeypatch.setattr(ndb.reset_needles_db, "live", stand_in)
    assert os.environ.get("ZOMBIHOP_TESTING") == "1"
    # the stand-in does not exist yet (real-path comparison), then it does (samefile)
    for alias in (os.path.join(s_dir, "needles.db"), None):
        if alias is None:
            stand_in.write_bytes(b"")
            alias = short(stand_in)
        assert os.path.normcase(alias) != os.path.normcase(str(stand_in))
        assert lanes_hw.live_needles_db_refused(alias)
        with pytest.raises(RuntimeError):
            ndb.write_needles([], DIMS, path=alias)
        with pytest.raises(RuntimeError):
            ndb.reset_needles_db(alias)
    assert stand_in.read_bytes() == b""
    assert not lanes_hw.live_needles_db_refused(tmp_path / "other" / "needles.db")


@pytest.fixture(scope="module")
def _module_scope_view():
    """What a module-scoped fixture sees of the needles.db isolation (tests/conftest.py, H8)."""
    import src.utils.needles_db as ndb
    return dict(testing=os.environ.get("ZOMBIHOP_TESTING"), env_db=os.environ.get("ZOMBI_NEEDLES_DB"),
                default=Path(ndb.DEFAULT_NEEDLES_DB_PATH),
                guarded=[getattr(getattr(ndb, f), "live", None) for f in ("write_needles", "reset_needles_db")])


def test_module_scoped_fixtures_are_isolated_too(_module_scope_view):
    """H8: module / session fixtures run with ZOMBIHOP_TESTING=1, a temp ZOMBI_NEEDLES_DB, the default path on that
    temp file and the live-file-refusing wrappers (the per-test fixture only covers function scope)."""
    live = Path(__file__).resolve().parents[1] / "sql" / "needles.db"
    v = _module_scope_view
    assert v["testing"] == "1"
    assert v["env_db"] and Path(v["env_db"]) == v["default"] != live
    assert v["guarded"] == [live, live]


def test_unreadable_state_file_is_copied_before_a_snapshot_restore_rewrites_it(run_copy, drivers):
    """needles_state.pt unreadable at restart (a Dropbox lock, a torn file) while a snapshot restores the needles:
    the unreadable file is copied to needles_state.unrestored_<stamp>.copy.pt (it may hold a needle the snapshot
    does not) BEFORE the restored set rewrites it; a WARNING + journal (partial)."""
    import torch
    run, db = run_copy
    n = int(_saved(run)["needles"].shape[0])
    a = drivers(run, needles_db=str(db))
    assert a.n_needles_restored == n
    a.engine.close()
    st_p = run / "lanes" / "needles_state.pt"
    torn = b"PK\x03\x04 torn needles_state.pt"
    st_p.write_bytes(torn)
    logs = []
    b = drivers(run, needles_db=str(db), log=_logger(logs))
    assert b.n_needles_restored == n                       # from the engine snapshots
    copies = sorted((run / "lanes").glob("needles_state.unrestored_*.copy.pt"))
    assert len(copies) == 1 and copies[0].read_bytes() == torn
    assert not [p for p in (run / "lanes").glob("needles_state.unrestored_*.pt") if not p.name.endswith(".copy.pt")]
    rec = _journal(run, "needles_unrestored_saved")
    assert len(rec) == 1 and rec[0]["partial"] is True and rec[0]["path"] == str(copies[0])
    assert any("WARNING" in x and str(copies[0]) in x for x in logs)
    s = torch.load(st_p, map_location="cpu", weights_only=False)   # rewritten with the restored set
    assert int(s["tensors"]["needles"].shape[0]) == n
    assert len(_db(db)) == n


def test_add_needle_hook_tries_once_and_the_decision_retries(run_copy, drivers, monkeypatch):
    """A locked needles_state.pt costs a needle decision <= 0.3 s, not twice that: add_needle's hook
    (dh._sync_needles_db, called with no arguments) tries ONCE without waiting; the same decision's
    _on_needles_maybe_changed holds the only retry budget (3 tries, <= 0.3 s)."""
    import time

    import torch
    run, db = run_copy
    _saved(run)
    drv = drivers(run, needles_db=str(db))
    dh = drv.engine.dh
    real_save = torch.save
    ctl = dict(calls=0, slept=0.0)

    def locked(obj, f, *a, **k):
        if str(f).endswith("needles_state.pt"):
            ctl["calls"] += 1
            raise RuntimeError("PytorchStreamWriter failed writing file data/0: file write failed")
        return real_save(obj, f, *a, **k)

    def sleep(s):
        ctl["slept"] += float(s)
    monkeypatch.setattr(torch, "save", locked)
    monkeypatch.setattr(time, "sleep", sleep)
    dh.needles_results[0]["median_value"] = 0.321
    assert dh._sync_needles_db() is False                  # the hook add_needle calls
    assert ctl["calls"] == 1 and ctl["slept"] == 0.0
    drv._on_needles_maybe_changed()                        # the decision's retry budget
    assert ctl["calls"] == 4 and ctl["slept"] <= 0.3 + 1e-9
    assert drv._needle_sig_last != drv._needle_sig()       # not persisted: the next decision retries
