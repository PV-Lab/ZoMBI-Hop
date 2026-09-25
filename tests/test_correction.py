"""
Tests for objective correction (src/core/correction.py +
ZoMBIHop.redeclare_needles_after_correction).

Covers the pure data layer (results-DB parsing, composition matching, snapshot
Y-rewrite, run cloning, dim discovery) and an end-to-end lifecycle on a real
tiny CPU campaign: run → build a corrected DB → clone → rewrite Y → re-derive
needles, asserting the original is untouched, the corrected scores persist, and
the needle moves to the corrected optimum.
"""

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.core import correction


# =============================================================================
# Helpers
# =============================================================================

def _write_results_db(path: Path, rows, *, n_comp_cols: int = 10):
    """Write a minimal DiSCO-shaped results DB.

    Columns: Iteration, c0..c{n-1} (the optimizer composition space), X (unused
    trailing comp column), Objective. Each row in ``rows`` is
    ``(iteration, comp_list_len_n, objective_or_None)``.
    """
    comp_cols = [f"c{i}" for i in range(n_comp_cols)]
    cols = ["Iteration"] + comp_cols + ["X", "Objective"]
    con = sqlite3.connect(str(path))
    try:
        con.execute(f'CREATE TABLE results ({", ".join(c + " REAL" for c in cols)})')
        ph = ", ".join("?" for _ in cols)
        for it, comp, obj in rows:
            assert len(comp) == n_comp_cols
            con.execute(f'INSERT INTO results VALUES ({ph})',
                        [it] + list(comp) + [0.0, obj])
        con.commit()
    finally:
        con.close()


def _simplex3_to_10(a, b, c):
    """Embed a 3-simplex point into 10 comp columns at positions 0,1,2."""
    return [a, b, c] + [0.0] * 7


# =============================================================================
# load_corrected_objectives
# =============================================================================

def test_load_corrected_objectives_basic(tmp_path):
    db = tmp_path / "results.db"
    _write_results_db(db, [
        (0, _simplex3_to_10(0.6, 0.3, 0.1), 0.80),
        (0, _simplex3_to_10(0.5, 0.4, 0.1), 0.70),
        (1, _simplex3_to_10(0.2, 0.2, 0.6), None),      # NULL objective
        (1, _simplex3_to_10(0.6, 0.3, 0.1), 0.85),      # duplicate comp, new score
    ])
    cm = correction.load_corrected_objectives(db, [0, 1, 2], round_decimals=6)
    assert cm.n_rows == 4
    assert cm.n_valid == 3
    assert cm.n_null == 1
    assert cm.active_columns == ["c0", "c1", "c2"]
    # duplicate comp (0.6,0.3,0.1): last finite value wins
    assert cm.by_comp[(0.6, 0.3, 0.1)] == pytest.approx(0.85)
    assert cm.n_conflicting_keys == 1
    # NULL row recorded as None (does not shadow a finite value)
    assert cm.by_comp[(0.2, 0.2, 0.6)] is None
    assert cm.comp_array.shape == (3, 3)  # valid rows only


def test_load_corrected_objectives_dim_selection(tmp_path):
    db = tmp_path / "r.db"
    # comp columns c0..c9; put distinctive values so dim selection is checkable
    comp = [0.1, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    _write_results_db(db, [(0, comp, 0.5)])
    cm = correction.load_corrected_objectives(db, [0, 2], round_decimals=6,
                                              simplex_sum_tol=1.0)
    assert cm.active_columns == ["c0", "c2"]
    assert (0.1, 0.3) in cm.by_comp


def test_load_corrected_objectives_rejects_non_simplex(tmp_path):
    db = tmp_path / "bad.db"
    # comps that do not sum to ~1 across the 10 columns → layout guard trips
    _write_results_db(db, [(0, [0.9, 0.9, 0.0, 0, 0, 0, 0, 0, 0, 0], 0.5)] * 5)
    with pytest.raises(ValueError, match="sum to ~1|composition"):
        correction.load_corrected_objectives(db, [0, 1], round_decimals=6)


def test_load_corrected_objectives_missing_objective_column(tmp_path):
    db = tmp_path / "noobj.db"
    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE results (Iteration REAL, c0 REAL)")
    con.commit(); con.close()
    with pytest.raises(ValueError):
        correction.load_corrected_objectives(db, [0], round_decimals=6)


# =============================================================================
# match_corrected_Y
# =============================================================================

def test_match_corrected_Y_exact_and_changes(tmp_path):
    db = tmp_path / "r.db"
    _write_results_db(db, [
        (0, _simplex3_to_10(0.6, 0.3, 0.1), 0.90),   # changed  (old 0.80)
        (0, _simplex3_to_10(0.5, 0.4, 0.1), 0.70),   # unchanged
        (0, _simplex3_to_10(0.2, 0.3, 0.5), None),   # null → keep old
    ])
    cm = correction.load_corrected_objectives(db, [0, 1, 2], round_decimals=6)
    X = np.array([[0.6, 0.3, 0.1],
                  [0.5, 0.4, 0.1],
                  [0.2, 0.3, 0.5],
                  [0.9, 0.05, 0.05]])   # not in DB → unmatched
    Y = np.array([0.80, 0.70, 0.55, 0.40])
    corrected, rep = correction.match_corrected_Y(X, Y, cm)
    assert corrected[0] == pytest.approx(0.90)
    assert corrected[1] == pytest.approx(0.70)   # unchanged kept
    assert corrected[2] == pytest.approx(0.55)   # null → old kept
    assert corrected[3] == pytest.approx(0.40)   # unmatched → old kept
    assert rep.n_points == 4
    assert rep.n_matched == 2
    assert rep.n_changed == 1
    assert rep.n_unchanged == 1
    assert rep.n_null_objective == 1
    assert rep.n_unmatched == 1
    assert rep.max_abs_delta == pytest.approx(0.10)


def test_match_corrected_Y_nearest_neighbour_fallback(tmp_path):
    db = tmp_path / "r.db"
    _write_results_db(db, [(0, _simplex3_to_10(0.6, 0.3, 0.1), 0.95)])
    cm = correction.load_corrected_objectives(db, [0, 1, 2], round_decimals=6)
    # Point drifts past the rounding grid (distinct 6-dp key) but within nn_tol.
    X = np.array([[0.6001, 0.2999, 0.1]])
    Y = np.array([0.5])
    corrected, rep = correction.match_corrected_Y(X, Y, cm, nn_tol=1e-3)
    assert corrected[0] == pytest.approx(0.95)
    assert rep.n_nn_recovered == 1
    assert rep.n_matched == 1
    # Outside nn_tol → unmatched.
    corrected2, rep2 = correction.match_corrected_Y(
        np.array([[0.7, 0.2, 0.1]]), np.array([0.5]), cm, nn_tol=1e-4)
    assert rep2.n_unmatched == 1
    assert corrected2[0] == pytest.approx(0.5)


# =============================================================================
# rewrite_snapshot_Y
# =============================================================================

def _make_fake_run_with_deltas(torch, run_dir: Path, chunks):
    """Create snapshots/NNNN/delta.pt files whose Y_new slices concatenate to a
    known Y. ``chunks`` is a list of 1-D lists (per snapshot Y_new)."""
    snaps = run_dir / "snapshots"
    snaps.mkdir(parents=True)
    offset = 0
    for i, ch in enumerate(chunks, start=1):
        d = snaps / f"{i:04d}_s"
        d.mkdir()
        n = len(ch)
        torch.save({
            "delta_version": 1,
            "n_cumulative_points": offset + n,
            "n_prev_points": offset,
            "X_new": torch.zeros(n, 3, dtype=torch.float64),
            "Y_new": torch.tensor(ch, dtype=torch.float64).reshape(-1, 1),
        }, d / "delta.pt")
        offset += n


def test_rewrite_snapshot_Y_roundtrip(torch, tmp_path):
    run_dir = tmp_path / "run_x"
    _make_fake_run_with_deltas(torch, run_dir, [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]])
    new_Y = np.array([10., 20., 30., 40., 50., 60.])
    n = correction.rewrite_snapshot_Y(run_dir, new_Y)
    assert n == 3
    # Reconstruct by concatenating Y_new in sorted order.
    ys = []
    for p in sorted((run_dir / "snapshots").iterdir()):
        d = torch.load(str(p / "delta.pt"), weights_only=False)
        ys.append(d["Y_new"].reshape(-1).numpy())
    assert np.allclose(np.concatenate(ys), new_Y)


def test_rewrite_snapshot_Y_count_mismatch_raises(torch, tmp_path):
    run_dir = tmp_path / "run_x"
    _make_fake_run_with_deltas(torch, run_dir, [[1.0, 2.0], [3.0]])
    with pytest.raises(ValueError, match="does not match"):
        correction.rewrite_snapshot_Y(run_dir, np.array([1.0, 2.0]))  # 2 != 3


# =============================================================================
# clone_run / read_optimizing_dims
# =============================================================================

def test_clone_run_copies_all_and_stamps_uuid(tmp_path):
    src = tmp_path / "run_aaaa"
    (src / "snapshots" / "0001_init").mkdir(parents=True)
    (src / "snapshots" / "0001_init" / "delta.pt").write_bytes(b"x")
    (src / "config.json").write_text(json.dumps({"run_uuid": "aaaa", "d": 3}))
    (src / "run.log").write_text("hello")
    (src / "latest.txt").write_text("0001_init")

    dst = tmp_path / "run_bbbb"
    correction.clone_run(src, dst, "bbbb")
    assert (dst / "snapshots" / "0001_init" / "delta.pt").read_bytes() == b"x"
    assert (dst / "run.log").read_text() == "hello"
    assert json.loads((dst / "config.json").read_text())["run_uuid"] == "bbbb"
    # original untouched
    assert json.loads((src / "config.json").read_text())["run_uuid"] == "aaaa"
    # refuses to overwrite
    with pytest.raises(FileExistsError):
        correction.clone_run(src, dst, "cccc")


def test_read_optimizing_dims(tmp_path):
    rd = tmp_path / "run"
    rd.mkdir()
    (rd / "config.json").write_text(json.dumps({"dims": "1,4,5"}))
    assert correction.read_optimizing_dims(rd) == [1, 4, 5]
    (rd / "hw_config.json").write_text(json.dumps({"dims": "0,2,3,4,8,9"}))
    assert correction.read_optimizing_dims(rd) == [0, 2, 3, 4, 8, 9]  # hw wins
    assert correction.read_optimizing_dims(tmp_path / "nope") is None


# =============================================================================
# End-to-end: tiny CPU campaign → corrected DB → re-derive needles
# =============================================================================

@pytest.fixture()
def _cpu_default_device(torch):
    import src.core.zombihop  # noqa: F401 — trigger the import side effect first
    if torch.cuda.is_available():
        torch.set_default_device("cpu")
        yield
        torch.set_default_device("cuda")
    else:
        yield


def _run_tiny_campaign(torch, base_dir: Path, n_consecutive: int = 10):
    """Real ZoMBIHop campaign on CPU that records a convergence stream the
    correction pass can replay. With the default strict ``n_consecutive`` no
    needle is declared live (matching the retro test); the correction pass then
    re-derives under a loosened criterion, exactly like a resume."""
    from src.core.zombihop import ZoMBIHop
    dtype = torch.float64
    d = 3
    peak = torch.tensor([0.6, 0.3, 0.1], dtype=dtype, device="cpu")

    g = torch.Generator().manual_seed(0)
    X_init = torch.rand(6, d, generator=g, dtype=dtype, device="cpu")
    X_init = X_init / X_init.sum(dim=1, keepdim=True)
    Y_init = 1.0 - ((X_init - peak) ** 2).sum(dim=1, keepdim=True)

    calls = {"n": 0}

    def objective(X, bounds, acq):
        calls["n"] += 1
        k = calls["n"]
        base = torch.tensor(
            [[0.60, 0.30, 0.10], [0.55, 0.35, 0.10],
             [0.50, 0.30, 0.20], [0.45, 0.35, 0.20]], dtype=dtype, device="cpu")
        jitter = 0.013 * k * torch.tensor(
            [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0],
             [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], dtype=dtype, device="cpu")
        Xp = (base + jitter).clamp(min=0.001)
        Xp = Xp / Xp.sum(dim=1, keepdim=True)
        Y = 1.0 - ((Xp - peak) ** 2).sum(dim=1)
        return Xp.clone(), Xp.clone(), Y

    z = ZoMBIHop(
        objective=objective, X_init_actual=X_init.clone(),
        X_init_expected=X_init.clone(), Y_init=Y_init.clone(),
        device="cpu", dtype=dtype, max_zooms=1, max_iterations=2,
        n_consecutive_converged=n_consecutive, n_restarts=2, raw=32,
        nat_grad_max_steps=5, resume=False, checkpoint_dir=str(base_dir),
        verbose=False, min_zoom_for_needle=0, min_iters_per_zoom=2,
        input_noise=0.05,
    )
    z._check_convergence_to_needle = lambda *a, **k: (True, 1e-6, -13.8)
    z.run(max_activations=1, time_limit_hours=None)
    return z


def _dir_md5(root: Path) -> str:
    h = hashlib.md5()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def test_e2e_correction_lifecycle(torch, tmp_path, _cpu_default_device):
    from src.core.zombihop import ZoMBIHop
    from src.utils.datahandler import reconstruct_snapshot_tensors

    from src.core import retro

    base_dir = tmp_path / "runs"
    z = _run_tiny_campaign(torch, base_dir, n_consecutive=10)
    dh = z.data_handler
    src_dir = dh.run_dir
    assert dh.needles.shape[0] == 0  # strict criterion: nothing declared live

    # --- Reconstruct stored points and build a corrected DB (dims 0,1,2). ---
    latest = (src_dir / "latest.txt").read_text().strip()
    s = reconstruct_snapshot_tensors(src_dir, latest, device="cpu")
    X = s["X_all_actual"].numpy()
    Y = s["Y_all"].numpy().reshape(-1)

    # Force a NEW optimum onto a measured point INSIDE activation 0's row range
    # (the range the correction pass will search), so the re-derived needle must
    # land there under the corrected scores.
    ranges = retro.activation_point_ranges(src_dir)
    act0_rows = [i for (a, b) in ranges.get(0, []) for i in range(a, b)]
    assert act0_rows, "campaign produced no attributable activation-0 rows"
    target = act0_rows[-1]
    corrected_true = Y.copy()
    corrected_true[target] = 5.0

    db = tmp_path / "corrected.db"
    _write_results_db(db, [
        (0, _simplex3_to_10(float(X[i, 0]), float(X[i, 1]), float(X[i, 2])),
         float(corrected_true[i]))
        for i in range(X.shape[0])
    ])

    cm = correction.load_corrected_objectives(db, [0, 1, 2], round_decimals=6)
    corrected_Y, rep = correction.match_corrected_Y(X, Y, cm)
    assert rep.n_matched == X.shape[0]
    assert corrected_Y[target] == pytest.approx(5.0)

    src_md5_before = _dir_md5(src_dir)

    # --- Clone, rewrite Y, re-derive needles. ---
    dst_dir = base_dir / "run_cor0"
    correction.clone_run(src_dir, dst_dir, "cor0")
    n_deltas = correction.rewrite_snapshot_Y(dst_dir, corrected_Y)
    assert n_deltas >= 1

    _d0 = torch.zeros(0, 3, device="cpu", dtype=torch.float64)

    def _dummy(*_a, **_k):
        raise RuntimeError("resume construction must not call the objective")

    zc = ZoMBIHop(
        objective=_dummy, X_init_actual=_d0, X_init_expected=_d0,
        Y_init=torch.zeros(0, 1, device="cpu", dtype=torch.float64),
        device="cpu", dtype=torch.float64, run_uuid="cor0",
        checkpoint_dir=str(base_dir), verbose=False,
        min_zoom_for_needle=0, min_iters_per_zoom=2,   # match the tiny campaign
    )
    # Corrected Y is loaded from the rewritten deltas.
    loaded_Y = zc.data_handler.Y_all.cpu().numpy().reshape(-1)
    assert np.allclose(loaded_Y, corrected_Y)

    # Loosen the needle criterion (an operator config.json edit before resume),
    # so activation 0's recorded convergence qualifies.
    zc.data_handler.n_consecutive_converged = 2

    res = zc.redeclare_needles_after_correction(dry_run=False)
    assert res["applied"] is True
    assert res["n_declared"] >= 1
    # The re-declared needle sits at the forced corrected optimum.
    ni = zc.data_handler.needle_indices.reshape(-1).tolist()
    assert target in ni
    # Needle value reflects the corrected score.
    peak_row = ni.index(target)
    assert zc.data_handler.needle_vals.reshape(-1)[peak_row].item() == pytest.approx(5.0)
    # A retro/correction needle is indistinguishable from a live one.
    assert zc.data_handler.needles_results[peak_row]["reason"] == "EI convergence"

    # --- Permanent snapshot + advanced resume position. ---
    latest_new = (dst_dir / "latest.txt").read_text().strip()
    assert latest_new.endswith("_corrected_needles")
    assert (dst_dir / "snapshots" / latest_new / "permanent").exists()
    assert (zc.current_zoom, zc.current_iteration) == (0, 0)

    # --- Original run untouched. ---
    assert _dir_md5(src_dir) == src_md5_before

    # --- Corrected copy resumes cleanly cross-process. ---
    z2 = ZoMBIHop(
        objective=_dummy, X_init_actual=_d0, X_init_expected=_d0,
        Y_init=torch.zeros(0, 1, device="cpu", dtype=torch.float64),
        device="cpu", dtype=torch.float64, run_uuid="cor0",
        checkpoint_dir=str(base_dir), verbose=False,
    )
    assert np.allclose(z2.data_handler.Y_all.cpu().numpy().reshape(-1), corrected_Y)
    assert z2.data_handler.needles.shape[0] == zc.data_handler.needles.shape[0]


def test_redeclare_never_raises_without_run_dir(torch, tmp_path, _cpu_default_device):
    """The public wrapper must swallow errors (a hardware pass can't be poisoned)."""
    from src.core.zombihop import ZoMBIHop
    dtype = torch.float64
    d = 3
    X = torch.eye(3, dtype=dtype, device="cpu")
    Y = torch.ones(3, 1, dtype=dtype, device="cpu")
    z = ZoMBIHop(objective=lambda *a, **k: None, X_init_actual=X, X_init_expected=X,
                 Y_init=Y, device="cpu", dtype=dtype, resume=False,
                 checkpoint_dir=None, verbose=False)
    res = z.redeclare_needles_after_correction(dry_run=False)
    assert res["applied"] is False
    assert "error" in res
