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
# Rewritten for the BASIN FLOOD-FILL loop (src/core/zombihop.py). That loop decides
# zoom depth from the data (a basin box at most `zoom_volume_fraction` of the active
# box's volume) and declares a needle from two stability tests, so the whole
# EI-convergence / Jaccard-window apparatus it replaced is no longer tunable — nor
# even read. RETIRED from this space, and ignored by ZoMBIHop:
#
#   max_zooms, max_iterations         zoom depth is data-driven; the only budget an
#                                     activation has is max_lines_per_activation
#   top_m_points                      fed determine_new_bounds' sliding window
#   n_consecutive_converged           the EI-convergence counter
#   output_noise_threshold_mult       the EI-convergence noise floor
#   needle_shrink_factor,             already inert before this change: both were
#   needle_stop_noise_multiplier      stored on ZoMBIHop and never read
#
# The axes that survived (acquisition optimisation, ucb_beta, max_penalty_radius,
# paring_*) are unchanged; the bounds on them are the ones re-tightened on
# 2026-08-12 against the 200-run evidence in optimize/runs/showdown_6d_clamped_reps10.

HPARAM_SPACE: dict[str, tuple] = {
    # Acquisition optimisation
    "nat_grad_step":               (0.01,   0.1,   "log"),
    "nat_grad_max_steps":          (10,     80,    "int"),
    "n_restarts":                  (100,    300,   "int"),
    "raw":                         (150,    400,   "int"),
    # Acquisition function
    "ucb_beta":                    (0.001,   3.0,   "linear"),
    # Activation budget. The line loop is flat now, so this is the ONLY thing
    # bounding how long an activation may grind on one region — it replaces the
    # max_zooms x max_iterations product outright.
    "max_lines_per_activation":    (10,     60,    "int"),
    # Basin flood fill (step 2). z sets how permissive the UCB >= LCB admission test
    # is, k how far the fill can bridge between LineBO lines.
    "flood_ci_z":                  (1.0,    3.0,   "linear"),
    "flood_k":                     (3,      12,    "int"),
    # Zoom (step 3): the basin box must be at most this fraction of the active box's
    # VOLUME before the search zooms into it.
    "zoom_volume_fraction":        (0.1,    0.9,   "linear"),
    # Needle stability (step 4). needle_move_tol is an ABSOLUTE composition-L2
    # tolerance (0.10 = ten composition points), not a fraction of the box.
    "needle_move_tol":             (0.02,   0.25,  "linear"),
    "needle_ci_tol":               (0.02,   0.5,   "linear"),
    # Penalisation
    "max_penalty_radius":          (0.15,   1.5,   "linear"),
    # Point paring (deduplication). paring_spatial_halfnoise doubles as the radius of
    # the replicate neighbourhood the needle criterion's median CI is bootstrapped
    # over, so it now trades paring aggressiveness against that CI's sample size.
    "paring_spatial_halfnoise":    (0.1,    2.0,   "linear"),
    "paring_y_noise_multiplier":   (0.1,    5.0,   "linear"),
}
HPARAM_NAMES = list(HPARAM_SPACE.keys())
N_HPARAMS    = len(HPARAM_NAMES)
