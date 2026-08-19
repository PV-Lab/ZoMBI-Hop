"""
Shared MOBO utilities for optimize/run_mobo.py.

No matplotlib import — safe on headless HPC nodes.
"""

from __future__ import annotations

import json
import math

import numpy as np
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated

# ─── Simulation / ZoMBI constants ─────────────────────────────────────────────

# Known input noise (per-component composition std), measured from runs/run_39af/composition_log.jsonl, the 6-dim hardware run of 2026-08-12 (109 lines / 2042 samples), which logs the sent composition directly: pooled per-component std 0.128, mean L2 0.271
# (see run_mobo.NOISE_LEVEL and default_hparams.DEFAULT_INPUT_NOISE).
NOISE_LEVEL = 0.128
NUM_EXPERIMENTS = 24
NUM_LINES = 10
N_INIT_LINES = 2

ZOMBI_FIXED = dict(
    max_gp_points=3000,
    acquisition_type="ucb",
    input_noise=NOISE_LEVEL,
    verbose=False,
)

# ─── Hyperparameter search space ──────────────────────────────────────────────

HPARAM_SPACE: dict[str, tuple] = {
    "nat_grad_step":               (0.001,  0.5,   "log"),
    "nat_grad_max_steps":          (10,     200,   "int"),
    "n_restarts":                  (20,     300,   "int"),
    "raw":                         (200,    2000,  "int"),
    "ucb_beta":                    (0.05,   3.0,   "linear"),
    "max_zooms":                   (2,      10,    "int"),
    "max_iterations":              (2,      10,    "int"),
    "top_m_points":                (2,      8,     "int"),
    "n_consecutive_converged":     (2,      10,    "int"),
    "input_noise_threshold_mult":  (0.5,    6.0,   "linear"),
    "output_noise_threshold_mult": (0.1,    2.0,   "linear"),
    "max_penalty_radius":          (0.2,    3.0,   "linear"),
    "needle_shrink_factor":        (0.55,   0.99,  "linear"),
    "needle_stop_noise_multiplier":(1.0,    8.0,   "linear"),
    "paring_spatial_halfnoise":    (0.1,    2.0,   "linear"),
    "paring_y_noise_multiplier":   (0.1,    3.0,   "linear"),
}
HPARAM_NAMES = list(HPARAM_SPACE.keys())
N_HPARAMS = len(HPARAM_NAMES)


def norm_to_hparams(x_norm: torch.Tensor) -> dict:
    params = {}
    for i, name in enumerate(HPARAM_NAMES):
        lo, hi, tfm = HPARAM_SPACE[name]
        v = float(x_norm[i].clamp(0.0, 1.0).item())
        if tfm == "log":
            params[name] = math.exp(math.log(lo) + v * (math.log(hi) - math.log(lo)))
        elif tfm == "int":
            params[name] = int(round(lo + v * (hi - lo)))
        else:
            params[name] = lo + v * (hi - lo)
    return params


# ─── Trial metrics (dimension-aware; see optimize/eval_metrics.py) ─────────────

from eval_metrics import (
    UNMATCHED_PENALTY,
    metric_dist_to_needles,
    metric_dup_fraction,
)


def save_running_summary(
    X_obs: list[torch.Tensor],
    Y_obs: list[torch.Tensor],
    save_path: str,
    *,
    n_init_trials: int,
) -> None:
    """Write a JSON summary of all trials completed so far."""
    n = len(Y_obs)
    if n == 0:
        return
    Y_stk = torch.stack(Y_obs)
    pareto_mask = is_non_dominated(Y_stk)

    metrics_all = [
        {
            "dist_to_needles": round(-Y_obs[i][0].item(), 6),
            "dup_fraction":    round(-Y_obs[i][1].item(), 6),
            "runtime_s":       round(-Y_obs[i][2].item(), 3),
        }
        for i in range(n)
    ]

    trials = [
        {
            "trial":   i + 1,
            "phase":   "sobol" if i < n_init_trials else "mobo",
            "pareto":  bool(pareto_mask[i].item()),
            "metrics": metrics_all[i],
            "hparams": {
                k: (round(v, 8) if isinstance(v, float) else v)
                for k, v in norm_to_hparams(X_obs[i]).items()
            },
        }
        for i in range(n)
    ]

    dists = [m["dist_to_needles"] for m in metrics_all]
    dups = [m["dup_fraction"] for m in metrics_all]
    runtimes = [m["runtime_s"] for m in metrics_all]

    summary = {
        "n_trials":  n,
        "n_pareto":  int(pareto_mask.sum().item()),
        "averages": {
            "dist_to_needles": round(float(np.mean(dists)), 6),
            "dup_fraction":    round(float(np.mean(dups)), 6),
            "runtime_s":       round(float(np.mean(runtimes)), 3),
        },
        "best_dist": {
            "value": round(min(dists), 6),
            "trial": int(np.argmin(dists)) + 1,
        },
        "trials": trials,
    }

    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"  [summary] {save_path}  ({n} trials, {summary['n_pareto']} Pareto)",
        flush=True,
    )
