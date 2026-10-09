"""
Tests for benchmarks/ablations — that each arm's patch actually bites, and that it
leaves nothing behind.

The point of these is narrow but important: the ablation arms are monkeypatches, so
the two ways they can silently produce a meaningless campaign are (a) the patch does
not change behaviour at all, in which case the "ablated" arm is a duplicate of the
baseline and the comparison reads as "no effect", and (b) the patch leaks past its
context manager, in which case every arm run afterwards in the same worker process is
contaminated. Neither failure mode raises anything on its own — a campaign would run to
completion and report confident numbers — so they have to be asserted directly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from benchmarks.ablations.arms import (
    ABLATIONS,
    ARMS,
    BASELINE_ARM,
    arm_context,
    arms_for,
    isotropize_precision,
)


# ─── Registry ────────────────────────────────────────────────────────────────────

def test_every_ablation_references_known_arms_with_baseline_first():
    for key, ablation in ABLATIONS.items():
        assert ablation.arms, f"{key} has no arms"
        assert ablation.arms[0] == BASELINE_ARM, (
            f"{key}'s first arm is the reference every paired statistic is computed "
            f"against, so it must be the baseline; got {ablation.arms[0]!r}")
        for name in ablation.arms:
            assert name in ARMS, f"{key} references unknown arm {name!r}"


def test_baseline_is_unmodified_and_unique():
    baselines = [a for a in ARMS.values() if a.is_baseline]
    assert len(baselines) == 1
    base = baselines[0]
    assert base.patches == (), "the baseline must run stock ZoMBI-Hop"
    assert base.hparam_overrides == {}
    assert base.runner == "single"


def test_arms_for_dedupes_the_shared_baseline():
    names = arms_for(list(ABLATIONS))
    assert names[0] == BASELINE_ARM, "baseline first, so it gets series slot 1"
    assert len(names) == len(set(names)), "the shared baseline must be queued once"
    # One baseline + one variant per ablation.
    assert len(names) == 1 + len(ABLATIONS)


# ─── A4: isotropic basins ────────────────────────────────────────────────────────

def test_isotropize_preserves_volume_and_is_spherical():
    M = torch.diag(torch.tensor([4.0, 100.0, 0.25], dtype=torch.float64))
    iso = isotropize_precision(M, max_radius=10.0)

    evals = torch.linalg.eigvalsh(iso)
    assert torch.allclose(evals, evals[0].expand_as(evals)), "not a sphere"
    # Volume of {u : u^T M u <= 1} scales as det(M)^(-1/2), so equal determinants
    # mean equal volume — this is what keeps A4 a test of SHAPE, not of size.
    assert torch.linalg.det(iso).item() == pytest.approx(
        torch.linalg.det(M).item(), rel=1e-9)


def test_isotropize_respects_the_semi_axis_cap():
    # Geometric mean here is 0.01 -> radius 10, well past the cap.
    M = torch.diag(torch.tensor([0.01, 0.01], dtype=torch.float64))
    iso = isotropize_precision(M, max_radius=0.5)
    radius = 1.0 / torch.linalg.eigvalsh(iso).sqrt()
    assert torch.all(radius <= 0.5 + 1e-12), (
        "a volume-matched sphere must not out-reach the ellipsoid's own cap")


def test_isotropic_arm_patches_and_restores_the_ellipsoid_call():
    from src.utils.gp_simplex import GPSimplex

    original = GPSimplex.determine_penalty_ellipsoid
    with arm_context(ARMS["isotropic_basins"]):
        assert GPSimplex.determine_penalty_ellipsoid is not original, "patch not applied"
    assert GPSimplex.determine_penalty_ellipsoid is original, "patch leaked"


def test_isotropic_arm_spheres_whatever_the_ellipsoid_call_returns():
    """The patch must isotropise the real return value, not just install a wrapper."""
    from src.utils.gp_simplex import GPSimplex

    anisotropic = torch.diag(torch.tensor([1.0, 64.0], dtype=torch.float64))
    basis = torch.eye(3, dtype=torch.float64)[:, :2]
    original = GPSimplex.determine_penalty_ellipsoid
    GPSimplex.determine_penalty_ellipsoid = (
        lambda self, needle, **kw: (anisotropic.clone(), basis.clone()))
    try:
        with arm_context(ARMS["isotropic_basins"]):
            M, B = GPSimplex.determine_penalty_ellipsoid(
                None, torch.zeros(3, dtype=torch.float64), max_radius=1.0)
        evals = torch.linalg.eigvalsh(M)
        assert torch.allclose(evals, evals[0].expand_as(evals))
        assert torch.allclose(B, basis), "the tangent basis must pass through unchanged"
    finally:
        GPSimplex.determine_penalty_ellipsoid = original


# ─── A3: random chords ───────────────────────────────────────────────────────────

def _linebo(dim: int = 3, num_lines: int = 8):
    from src.core.linebo import LineBO

    return LineBO(objective_function=lambda endpoints: (None, None),
                  dimensions=dim, num_lines=num_lines, device="cpu")


def test_random_chord_arm_returns_chords_through_the_candidate():
    lb = _linebo()
    x_tell = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    bounds = torch.stack([torch.zeros(3, dtype=torch.float64),
                          torch.ones(3, dtype=torch.float64)])

    with arm_context(ARMS["random_chords"]):
        left, right = lb.ranked_line_endpoints(x_tell, bounds, acquisition_function=None)

    assert left.shape[0] > 0 and left.shape == right.shape
    assert left.shape[0] <= lb.num_lines
    for xl, xr in zip(left, right):
        # x_tell must lie ON the segment: solve for t and check it round-trips and
        # sits inside [0, 1]. This is the property the arm is named for — a chord
        # that misses the candidate would be a different ablation entirely.
        direction = xr - xl
        k = int(torch.argmax(direction.abs()))
        t = ((x_tell[k] - xl[k]) / direction[k]).item()
        assert -1e-6 <= t <= 1 + 1e-6, f"candidate not inside the segment (t={t})"
        assert torch.allclose(xl + t * direction, x_tell, atol=1e-8), \
            "chord does not pass through the candidate"
        # Zero-sum directions keep every point of the chord on the simplex.
        assert xl.sum().item() == pytest.approx(1.0, abs=1e-9)
        assert xr.sum().item() == pytest.approx(1.0, abs=1e-9)


def test_random_chord_arm_ignores_the_acquisition_and_restores_the_method():
    from src.core.linebo import LineBO

    original = LineBO.ranked_line_endpoints
    x_tell = torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float64)
    bounds = torch.stack([torch.zeros(3, dtype=torch.float64),
                          torch.ones(3, dtype=torch.float64)])

    calls = []

    def spy_acquisition(batch):
        calls.append(batch)
        return torch.zeros(batch.shape[0], dtype=torch.float64)

    with arm_context(ARMS["random_chords"]):
        assert LineBO.ranked_line_endpoints is not original, "patch not applied"
        _linebo().ranked_line_endpoints(x_tell, bounds, spy_acquisition)
    assert not calls, "the random-chord arm must never consult the acquisition"
    assert LineBO.ranked_line_endpoints is original, "patch leaked"


def test_random_chord_arm_respects_a_tightened_box():
    lb = _linebo()
    x_tell = torch.tensor([0.4, 0.35, 0.25], dtype=torch.float64)
    bounds = torch.stack([torch.tensor([0.2, 0.2, 0.1], dtype=torch.float64),
                          torch.tensor([0.6, 0.5, 0.4], dtype=torch.float64)])

    with arm_context(ARMS["random_chords"]):
        left, right = lb.ranked_line_endpoints(x_tell, bounds)

    assert left.shape[0] > 0
    for pts in (left, right):
        assert torch.all(pts >= bounds[0] - 1e-9), "chord escapes the zoom box"
        assert torch.all(pts <= bounds[1] + 1e-9), "chord escapes the zoom box"


# ─── A2: no zooming ──────────────────────────────────────────────────────────────

def test_no_zoom_arm_pins_bounds_to_the_global_box():
    from src.utils.datahandler import DataHandler

    original = DataHandler.determine_new_bounds

    class _Stub:
        """Only the two attributes the patched method reads."""
        _full_bounds_ref = torch.stack([torch.zeros(4, dtype=torch.float64),
                                        torch.ones(4, dtype=torch.float64)])
        bounds = torch.stack([torch.full((4,), 0.4, dtype=torch.float64),
                              torch.full((4,), 0.6, dtype=torch.float64)])

    with arm_context(ARMS["no_zoom"]):
        assert DataHandler.determine_new_bounds is not original, "patch not applied"
        out = DataHandler.determine_new_bounds(_Stub())
        assert torch.allclose(out, _Stub._full_bounds_ref), \
            "the no-zoom arm must hand back the global box, not a contracted one"
        # The failure-retry path calls it with add_to_history=False; that signature
        # has to keep working or A2 dies partway through a run.
        assert torch.allclose(
            DataHandler.determine_new_bounds(_Stub(), add_to_history=False),
            _Stub._full_bounds_ref)
    assert DataHandler.determine_new_bounds is original, "patch leaked"


def test_no_zoom_arm_keeps_needles_declarable():
    arm = ARMS["no_zoom"]
    assert arm.hparam_overrides["max_zooms"] == 1
    # Without this the default min_zoom_for_needle=2 makes zoom level 1 unreachable
    # at max_zooms=1, and the arm would measure "no needles", not "no zooming".
    assert arm.hparam_overrides["min_zoom_for_needle"] == 0


# ─── A1: k restarts ──────────────────────────────────────────────────────────────

def test_restart_arm_caps_activations_without_touching_the_time_limit():
    from benchmarks.ablations.restarts import _cap_activations
    from src.core.zombihop import ZoMBIHop

    seen = []
    original = ZoMBIHop.run
    ZoMBIHop.run = lambda self, max_activations=5, time_limit_hours=None, **kw: \
        seen.append((max_activations, time_limit_hours))
    try:
        with _cap_activations(1):
            # Keyword form, as run_single_trial calls it.
            ZoMBIHop.run(None, max_activations=float("inf"), time_limit_hours=0.25)
            # Positional form, to prove the remaining arguments stay bound correctly.
            ZoMBIHop.run(None, 9, 0.5)
    finally:
        ZoMBIHop.run = original

    assert seen == [(1.0, 0.25), (1.0, 0.5)], (
        "the cap must clamp activations and leave every other argument alone")
    assert ZoMBIHop.run is original, "patch leaked"


def test_restart_arm_is_configured_as_plain_zombi():
    arm = ARMS["k_restarts"]
    assert arm.runner == "restarts"
    # 1 activation = zoom in, converge, declare one needle, stop. More than that and
    # each "restart" is a short ZoMBI-HOP run, which is not the thing A1 contrasts.
    assert arm.runner_kwargs["max_activations_per_restart"] == 1
    assert arm.runner_kwargs["fill_budget"] is True, (
        "without budget fill-in the arm can spend far less than the baseline and A1 "
        "measures the handicap instead of the strategy")


# ─── Cross-cutting ───────────────────────────────────────────────────────────────

def test_arms_do_not_contaminate_each_other():
    """Every arm runs in the same worker process, one after another."""
    from src.core.linebo import LineBO
    from src.utils.datahandler import DataHandler
    from src.utils.gp_simplex import GPSimplex

    before = (LineBO.ranked_line_endpoints,
              DataHandler.determine_new_bounds,
              GPSimplex.determine_penalty_ellipsoid)
    for name in ARMS:
        with arm_context(ARMS[name]):
            pass
    after = (LineBO.ranked_line_endpoints,
             DataHandler.determine_new_bounds,
             GPSimplex.determine_penalty_ellipsoid)
    assert before == after


def test_unknown_patch_name_fails_loudly():
    from benchmarks.ablations.arms import Arm

    bogus = Arm(name="bogus", label="Bogus", description="", patches=("nope",))
    with pytest.raises(KeyError, match="nope"):
        with arm_context(bogus):
            pass


def test_resolve_hparams_drops_keys_run_mobo_fixes():
    """A duplicate key is a TypeError inside ZoMBIHop(**ZOMBI_FIXED, **hp)."""
    import run_mobo as rm

    from benchmarks.ablations.runner import resolve_hparams

    fixed_key = next(iter(rm.ZOMBI_FIXED))
    hp = resolve_hparams({"max_zooms": 3, fixed_key: "clash"}, ARMS["no_zoom"])
    assert fixed_key not in hp
    assert hp["max_zooms"] == 1, "the arm's override must win over the base value"


def test_cell_seed_is_shared_across_arms_and_varies_across_cells():
    """Common random numbers: the seed depends on the cell, never on the arm."""
    from benchmarks.ablations.runner import cell_seed

    assert cell_seed(3, 2) == cell_seed(3, 2)
    assert cell_seed(3, 2) != cell_seed(3, 3)
    assert cell_seed(3, 2) != cell_seed(4, 2)
    assert cell_seed(3, 2, base=1) != cell_seed(3, 2, base=0)
    seeds = {cell_seed(ls, r) for ls in range(12) for r in range(1, 6)}
    assert len(seeds) == 60, "cells must not collide onto one seed"
    assert all(0 <= s < 2 ** 31 - 1 for s in seeds)


# ─── Summary statistics ──────────────────────────────────────────────────────────

def test_align_curves_holds_finished_cells_and_counts_active_ones():
    from benchmarks.ablations.summarize import align_curves

    matrix, n_active = align_curves([np.array([1.0, 2.0, 3.0]), np.array([5.0, 4.0])])
    assert matrix.shape == (2, 3)
    # The short cell really did end at 4.0; it is held, not dropped, so the mean at
    # iteration 3 stays a mean over the same two cells it was over at iteration 1.
    assert matrix[1].tolist() == [5.0, 4.0, 4.0]
    assert n_active.tolist() == [2, 2, 1]


def test_bootstrap_band_brackets_the_mean_and_degrades_safely():
    from benchmarks.ablations.summarize import bootstrap_band

    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(40, 5)) + np.arange(5)
    mean, lo, hi = bootstrap_band(matrix, seed=1)
    assert np.all(lo <= mean + 1e-12) and np.all(mean <= hi + 1e-12)
    assert np.allclose(mean, matrix.mean(axis=0))

    # Too few cells to resample: collapse to the mean rather than draw a band that
    # looks like evidence.
    mean2, lo2, hi2 = bootstrap_band(matrix[:2], seed=1)
    assert np.allclose(lo2, mean2) and np.allclose(hi2, mean2)


def test_sign_flip_p_separates_a_real_shift_from_noise():
    from benchmarks.ablations.summarize import sign_flip_p

    assert sign_flip_p(np.full(12, -0.4), n_perm=4000, seed=0) < 0.01
    rng = np.random.default_rng(3)
    assert sign_flip_p(rng.normal(size=40), n_perm=4000, seed=0) > 0.05
    # Never exactly zero: the observed arrangement is one of the permutations.
    assert sign_flip_p(np.full(30, -1.0), n_perm=1000, seed=0) > 0
