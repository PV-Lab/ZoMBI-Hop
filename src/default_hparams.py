"""
Canonical default ZoMBI-Hop hyperparameters — single source of truth.
==========================================================================

Both the GUI (``interface/app.py``) and the DB-backed hardware runner
(``scripts/run_zombi_main.py``) use these defaults when no ``--hparams`` /
hyperparameter JSON file is supplied. Edit them HERE and nowhere else.

``DEFAULT_HPARAMS`` is copied verbatim from the strong 4D MOBO ensemble result
``optimize/runs/mobo_ensemble_4d_job17147232/trial_10`` (see ``trial.json``).
To adopt a different trial in the future, replace the values below and update
``DEFAULT_HPARAMS_PROVENANCE``; every consumer picks the change up automatically.
"""

from __future__ import annotations

# Human-readable origin of the values below. Used in UI hint text so it stays in
# sync with the actual defaults.
DEFAULT_HPARAMS_PROVENANCE = "mobo_ensemble_4d_job17147232/trial_10"

# The 16 tunable ZoMBI-Hop hyperparameters.
DEFAULT_HPARAMS: dict = {
    "nat_grad_step": 0.01800036,
    "nat_grad_max_steps": 90,
    "n_restarts": 173,
    "raw": 97,
    "ucb_beta": 3.0,
    "max_zooms": 10,
    "max_iterations": 2,
    "top_m_points": 2,
    "n_consecutive_converged": 5,
    "input_noise_threshold_mult": 0.5,
    "output_noise_threshold_mult": 0.01,
    "max_penalty_radius": 0.51882012,
    "needle_shrink_factor": 0.5531221,
    "needle_stop_noise_multiplier": 2.17946912,
    "paring_spatial_halfnoise": 5.0,
    "paring_y_noise_multiplier": 10.0,
}

# Known physical input noise (per-component composition std), measured as the
# average input noise of data/2nd_real_run.db (see visualization/input_noise.py).
# Not a tuned hyperparameter, so it lives outside DEFAULT_HPARAMS; the hardware
# runner folds it into its defaults.
DEFAULT_INPUT_NOISE = 0.064
