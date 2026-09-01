"""
optimize/hparam_space.py
========================
The ONE definition of the MOBO hyperparameter search space.

Lives in its own module — with no torch/botorch/matplotlib import — so that both
``run_mobo`` (which tunes over it) and ``pareto`` (which normalises recorded trials
against it for the number-line plot) can share a single dict cheaply. They used to
keep private copies, which drifted: ``input_noise_threshold_mult`` was dropped from
the tuned set in 56dd11d but survived in the copies, so a config file could appear
to set a knob the optimiser no longer reads.

Anything a trial records that is NOT a key here is not tuned; ``evaluate._split_hparams``
drops such keys before they reach ZoMBIHop, and it warns when it does.
"""

from __future__ import annotations

# ─── Hyperparameter search space ──────────────────────────────────────────────
# Each entry: (lo, hi, transform) — transform ∈ {"log", "linear", "int"}
# Normalised to [0, 1] for MOBO; unnormalised when calling ZoMBI.

#
# The bounds below were re-tightened on 2026-08-12 against the 200-run evidence in
# optimize/runs/showdown_6d_clamped_reps10 (5 configs × 5 landscapes × 10 repeats,
# 7 047 declared needles). The principle used: an axis is narrowed only where the
# extra range is provably *unreachable*, *inert*, or *pure cost on a minimised
# objective* — never where the good configurations genuinely disagree. Axes on
# which the showdown winners span the range (ucb_beta, output_noise_threshold_mult,
# needle_stop_noise_multiplier, paring_*) are Pareto trade-offs between the three
# minimised objectives and are deliberately left alone.

HPARAM_SPACE: dict[str, tuple] = {
    # Acquisition optimisation
    "nat_grad_step":               (0.01,   0.1,   "log"),
    "nat_grad_max_steps":          (10,     80,    "int"),
    "n_restarts":                  (100,    300,   "int"),
    "raw":                         (150,    400,   "int"),
    # Acquisition function
    "ucb_beta":                    (0.001,   3.0,   "linear"),
    # Zoom / convergence
    # Lower bound is 2: a needle can only be declared at zoom level 2+
    # (ZoMBIHop.min_zoom_for_needle, lowered to 1 post-6d-campaign), so max_zooms
    # must allow reaching it. Kept in sync with evaluate._force_zoom_floors(),
    # which derives the same floor from ZoMBIHop's own defaults.
    "max_zooms":                   (2,      6,     "int"),
    # Lower bound is 3 so at least min_iters_per_zoom (=3) lines can be sampled
    # per zoom level before the optimiser may advance or declare a needle.
    "max_iterations":              (3,      12,    "int"),
    "top_m_points":                (4,      16,    "int"),
    "n_consecutive_converged":     (2,      5,     "int"),
    # Lower bound is 0.1: the convergence test is EI < GP_output_noise × this,
    # so below ~0.1 it is effectively unsatisfiable — EI convergence stops
    # declaring needles at all and every needle has to come from the Jaccard
    # force-declare fallback, at the cost of the wasted zoom iterations.
    "output_noise_threshold_mult": (0.1,    2.0,   "linear"),
    # Penalisation & needle
    "max_penalty_radius":          (0.15,   1.5,   "linear"),
    "needle_shrink_factor":        (0.5,   0.95,  "linear"),
    "needle_stop_noise_multiplier":(1.0,    8.0,   "linear"),
    # Point paring (deduplication)
    "paring_spatial_halfnoise":    (0.1,    2.0,   "linear"),
    "paring_y_noise_multiplier":   (0.1,    5.0,   "linear"),
}
HPARAM_NAMES = list(HPARAM_SPACE.keys())
N_HPARAMS    = len(HPARAM_NAMES)
