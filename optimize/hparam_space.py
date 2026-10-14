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

Trials recorded before 2026-10-12 lack the axes added that day; ``upgrade_legacy``
fills them (and corrects two axes that were never wired up) with the values those
trials actually ran with, so old history keeps pooling.
"""

from __future__ import annotations

import os
import sys

# pareto.py / showdown.py import this with only optimize/ on sys.path.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# ─── Hyperparameter search space ──────────────────────────────────────────────
# Each entry: (lo, hi, transform) — transform ∈ {"log", "linear", "int"}
# Normalised to [0, 1] for MOBO; unnormalised when calling ZoMBI.
#
# The ranges are declared next to each knob in src/hparams.py (the one registry of
# every ZoMBI-Hop hyperparameter); this module only re-exports them. Ranges were
# re-tightened on 2026-08-12 against the 200-run evidence in
# optimize/runs/showdown_6d_clamped_reps10: an axis is narrowed only where the
# extra range is provably unreachable, inert, or pure cost on a minimised
# objective — never where the good configurations genuinely disagree.
#
# A box cannot express "n_consecutive_converged <= max_iterations", so MOBO maps
# each sampled point through repair_conflicts() before running it (see
# run_mobo.norm_to_hparams). The other structural constraints hold by
# construction: max_zooms >= 3 > min_zoom_for_needle <= 2, and
# max_iterations >= 3 >= min_iters_per_zoom.

from src.hparams import (  # noqa: E402,F401
    HPARAM_SPACE,
    LEGACY_EFFECTIVE,
    is_legacy,
    repair_conflicts,
    upgrade_legacy,
)

HPARAM_NAMES = list(HPARAM_SPACE.keys())
N_HPARAMS    = len(HPARAM_NAMES)
