"""
Canonical default ZoMBI-Hop hyperparameters — single source of truth.
==========================================================================

Both the GUI (``interface/app.py``) and the DB-backed hardware runner
(``scripts/run_zombi_main.py``) use these defaults when no ``--hparams`` /
hyperparameter JSON file is supplied. Edit them HERE and nowhere else.

``DEFAULT_HPARAMS`` is copied verbatim from ``optimize/hparams/6d_ensemble.json``
(the 6D MOBO ensemble winner). To adopt a different trial in the future, replace
the values below and update ``DEFAULT_HPARAMS_PROVENANCE``; every consumer picks
the change up automatically.
"""

from __future__ import annotations

# Human-readable origin of the values below. Used in UI hint text so it stays in
# sync with the actual defaults.
DEFAULT_HPARAMS_PROVENANCE = "optimize/hparams/6d_ensemble.json"

# The 16 tunable ZoMBI-Hop hyperparameters.
DEFAULT_HPARAMS: dict = {
    "nat_grad_step": 0.23242035,
    "nat_grad_max_steps": 400,
    "n_restarts": 300,
    "raw": 300,
    "ucb_beta": 3.0,
    "max_zooms": 3,
    "max_iterations": 3,
    "top_m_points": 8,
    "n_consecutive_converged": 5,
    "input_noise_threshold_mult": 3.0,
    "output_noise_threshold_mult": 2.0,
    "max_penalty_radius": 0.5020402,
    "needle_shrink_factor": 0.99,
    "needle_stop_noise_multiplier": 1.0,
    "paring_spatial_halfnoise": 5.0,
    "paring_y_noise_multiplier": 10.0,
}

# Known physical input noise (per-component composition std), measured by
# visualization/input_noise.py from runs/run_39af/composition_log.jsonl — the
# 6-dim hardware run of 2026-08-12, which records the composition the optimiser
# *sent* alongside the one the printer realised, so nothing is reconstructed.
#
# The previous value (0.064) came from data/2nd_real_run.db, where the sent
# composition was never logged and had to be inferred by fitting a line through
# each print line's realised endpoints. That fit absorbs the systematic
# requested-vs-realised offset, so it measures only the scatter about the line
# and under-reports by ~3x (0.040 vs 0.128 on run_39af's own data). ~87% of the
# residual on run_39af is *perpendicular* to the requested line, i.e. the printer
# genuinely misses the composition rather than merely mis-spacing along the line.
#
# Not a tuned hyperparameter, so it lives outside DEFAULT_HPARAMS; the hardware
# runner folds it into its defaults.
DEFAULT_INPUT_NOISE = 0.128
