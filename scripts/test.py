"""
Quick test of ZoMBIHop + LineBO with L2-distance objective on the simplex.

10D simplex with 3 minima (targets). Objective = -min(L2 distance to nearest target).
Uses 24 experiments per line. Artificially adds noise to inputs and outputs.
Convergence uses Probability of Improvement + stagnation window (same as main).
ZoMBIHop parameters match scripts/run_zombi_main.py (main.py).

Run: python -m scripts.test
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Tuple

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

from src import ZoMBIHop
from src.core.linebo import batch_line_simplex_segments, zero_sum_dirs

# Simplex dimension and experiments per line (same as run_zombi_main NUM_EXPERIMENTS)
NUM_EXPERIMENTS = 24
DIMENSIONS = 10  # 10D simplex with 3 L2 minima
NUM_TARGETS = 3  # number of minima (targets) in the objective
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Artificial noise (same scale as typical run; ZoMBIHop uses input/output noise thresholds)
INPUT_NOISE_STD = 0.001   # std of Gaussian noise added to inputs (observed vs requested)
OUTPUT_NOISE_STD = 0.001  # std of Gaussian noise added to outputs (objective values)

# ZoMBIHop maximizes the objective, so we minimize distance by returning objective = -distance.
MINIMIZE_DISTANCE_OBJECTIVE = True  # if True, return y = -distance so maximizer minimizes distance

# Print candidate, line endpoints, and Y stats every time the objective is called (set True to debug).
DEBUG_OBJECTIVE = False


def l2_objective_multi_target(
    ordered_endpoints: np.ndarray,
    targets: np.ndarray,
    num_experiments: int = NUM_EXPERIMENTS,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate -min_t ||x - target_t||_2 at points along the given lines (3 minima, L2).
    Adds artificial input noise (to x) and output noise (to y).

    ordered_endpoints: (2, 2, d) — 2 lines, each (left, right).
    targets: (num_targets, d) — fixed points on simplex (minima).
    Returns (x_meas, y) with x_meas (n, d) noisy inputs, y (n,) noisy -min_distance.
    """
    rng = rng or np.random.default_rng()
    x_list = []
    for line_idx in range(ordered_endpoints.shape[0]):
        left = ordered_endpoints[line_idx, 0]
        right = ordered_endpoints[line_idx, 1]
        for t in np.linspace(0, 1, num_experiments):
            x = left + t * (right - left)
            x_list.append(x)
    x_meas = np.array(x_list, dtype=np.float64)  # (2 * num_experiments, d)
    # Input noise: add Gaussian to each coordinate, then re-normalize to simplex
    x_meas = x_meas + rng.normal(0, INPUT_NOISE_STD, size=x_meas.shape).astype(np.float64)
    x_meas = np.clip(x_meas, 0.0, 1.0)
    x_meas = x_meas / x_meas.sum(axis=1, keepdims=True)
    # Distance to nearest target (L2)
    dists = np.array([np.linalg.norm(x_meas - t, axis=1) for t in targets])  # (num_targets, n)
    min_dist = np.min(dists, axis=0)  # (n,)
    y = (-min_dist.astype(np.float64)) if MINIMIZE_DISTANCE_OBJECTIVE else min_dist.astype(np.float64)
    y = y + rng.normal(0, OUTPUT_NOISE_STD, size=y.shape).astype(np.float64)
    return x_meas, y


def make_objective_wrapper(targets: np.ndarray, num_lines: int = 100, device=None, rng: np.random.Generator | None = None):
    """Build ZoMBIHop objective: LineBO + L2 multi-target objective. targets: (num_targets, d)."""
    targets = np.asarray(targets, dtype=np.float64)
    device = device or DEVICE
    rng = rng or np.random.default_rng()

    def objective_fn(ordered_endpoints: np.ndarray):
        return l2_objective_multi_target(ordered_endpoints, targets=targets, rng=rng)

    _call_count = [0]

    def wrapper(
        x_tell: torch.Tensor,
        bounds: torch.Tensor | None = None,
        acquisition_function=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _call_count[0] += 1
        candidate_np = x_tell.detach().cpu().numpy() if x_tell.dim() > 0 else x_tell.unsqueeze(0).cpu().numpy()
        if candidate_np.ndim > 1:
            candidate_np = candidate_np.reshape(-1)
        directions = zero_sum_dirs(num_lines, DIMENSIONS, device=device, dtype=torch.float64)
        x_left, x_right, t_min, t_max, mask = batch_line_simplex_segments(x_tell, directions)
        if x_left is None or x_left.shape[0] < 2:
            fallback = zero_sum_dirs(4, DIMENSIONS, device=device, dtype=torch.float64)
            x_left, x_right, t_min, t_max, mask = batch_line_simplex_segments(x_tell, fallback)
        if x_left is None or x_left.shape[0] < 2:
            x_left = x_tell.unsqueeze(0).repeat(2, 1)
            x_right = x_tell.unsqueeze(0).repeat(2, 1)
        x_left, x_right = x_left[:2], x_right[:2]
        ordered_endpoints = np.stack([x_left.cpu().numpy(), x_right.cpu().numpy()], axis=1)  # (2, 2, d)
        if DEBUG_OBJECTIVE:
            left0, right0 = ordered_endpoints[0, 0], ordered_endpoints[0, 1]
            left1, right1 = ordered_endpoints[1, 0], ordered_endpoints[1, 1]
            print(f"  [test objective #{_call_count[0]}] candidate (x_tell) = {candidate_np}")
            print(f"  [test objective #{_call_count[0]}] line0: left = {left0}, right = {right0}")
            print(f"  [test objective #{_call_count[0]}] line1: left = {left1}, right = {right1}")
        x_meas, y = objective_fn(ordered_endpoints)
        if DEBUG_OBJECTIVE:
            print(f"  [test objective #{_call_count[0]}] returned n={len(y)}, Y: min={float(np.min(y)):.4f}, max={float(np.max(y)):.4f}, mean={float(np.mean(y)):.4f}")
        X_actual = torch.tensor(x_meas, device=device, dtype=torch.float64)
        X_expected = X_actual.clone()
        Y = torch.tensor(y, device=device, dtype=torch.float64).reshape(-1)
        return X_actual, X_expected, Y

    return wrapper


def main():
    device = torch.device(DEVICE)
    dtype = torch.float64
    d = DIMENSIONS
    bounds = torch.zeros((2, d), device=device, dtype=dtype)
    bounds[0] = 0.0
    bounds[1] = 1.0

    # 3 minima (targets) on 10D simplex — objective = -min_t ||x - target_t||_2 (L2)
    # Place targets at different regions of the simplex so ZoMBIHop can find multiple needles
    rng = np.random.default_rng(42)
    targets = np.zeros((NUM_TARGETS, d), dtype=np.float64)
    targets[0] = np.ones(d) / d  # center: [1/d, ..., 1/d]
    targets[1] = np.zeros(d)
    targets[1][0], targets[1][1] = 0.5, 0.5  # first two coords
    targets[2] = np.zeros(d)
    targets[2][-2], targets[2][-1] = 0.5, 0.5  # last two coords
    for i in range(NUM_TARGETS):
        targets[i] = np.clip(targets[i], 0.0, 1.0)
        targets[i] = targets[i] / targets[i].sum()
    assert targets.shape == (NUM_TARGETS, d)
    for i in range(NUM_TARGETS):
        assert np.isclose(targets[i].sum(), 1.0), f"Target {i} must sum to 1"

    # Initial points on simplex; objective = -min distance to any target
    X_init = ZoMBIHop.random_simplex(2 * NUM_EXPERIMENTS, bounds[0].cpu(), bounds[1].cpu(), device=str(device))
    X_init = X_init.cpu().numpy()
    X_init = X_init + rng.normal(0, INPUT_NOISE_STD, size=X_init.shape).astype(np.float64)
    X_init = np.clip(X_init, 0.0, 1.0)
    X_init = X_init / X_init.sum(axis=1, keepdims=True)
    dists_init = np.array([np.linalg.norm(X_init - t, axis=1) for t in targets])
    min_dist_init = np.min(dists_init, axis=0)
    noise_init = rng.normal(0, OUTPUT_NOISE_STD, size=min_dist_init.shape).astype(np.float64)
    y_init = (-min_dist_init + noise_init) if MINIMIZE_DISTANCE_OBJECTIVE else (min_dist_init + noise_init)
    Y_init = torch.tensor(y_init, device=device, dtype=dtype).reshape(-1, 1)
    X_init_actual = torch.tensor(X_init[:, :d], device=device, dtype=dtype)
    X_init_expected = X_init_actual.clone()

    objective_wrapper = make_objective_wrapper(targets, num_lines=100, device=device, rng=rng)
    checkpoint_dir = Path("actual_runs") / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer = ZoMBIHop(
        objective=objective_wrapper,
        bounds=bounds,
        X_init_actual=X_init_actual,
        X_init_expected=X_init_expected,
        Y_init=Y_init,
        penalization_threshold=0.0005915,
        convergence_pi_threshold=0.001,
        input_noise_threshold_mult=float('inf'),
        output_noise_threshold_mult=1.0,
        n_consecutive_converged=2,
        max_zooms=5,
        max_iterations=7,
        top_m_points=max(d + 1, 4),
        n_restarts=50,
        raw=500,
        penalty_num_directions=10 * d,
        penalty_max_radius=0.33633,
        penalty_radius_step=None,
        max_gp_points=3000,
        device=device,
        dtype=dtype,
        checkpoint_dir=str(checkpoint_dir),
        verbose=True,
    )

    print(f"Targets (3 minima, L2): {[t.round(3).tolist() for t in targets]}")
    print(f"Objective = -min(L2 distance to nearest target). ZoMBIHop maximizer minimizes distance.")
    print(f"Running ZoMBIHop for multiple activations ({d}D simplex, 3 L2 minima, input/output noise)...")
    needles, needle_locs, needle_vals, X_all_actual, Y_all = optimizer.run(max_activations=5, time_limit_hours=None)

    # Needle locations: (n_needles, d); targets: (3, d). Assign needles to targets to minimize total distance.
    needle_locs_np = needle_locs.cpu().numpy() if torch.is_tensor(needle_locs) else np.asarray(needle_locs)
    targets_np = np.asarray(targets)
    n_needles = needle_locs_np.shape[0]
    n_targets = targets_np.shape[0]
    max_ok_dist = 0.7  # 10D simplex; allow reasonable tolerance per target

    if n_needles == 0:
        # No needles found: fall back to best point in data
        best_idx = Y_all.squeeze().argmax().item()
        best_x = X_all_actual[best_idx].cpu().numpy()
        dists_per_target = np.array([np.linalg.norm(best_x - t) for t in targets_np])
        print(f"\nNo needles found. Best point: {best_x.round(4)} (sum={best_x.sum():.4f})")
        print(f"Distances to targets: {dists_per_target.round(4).tolist()}")
        assert np.max(dists_per_target) < max_ok_dist, f"Expected each target within {max_ok_dist}, got max {np.max(dists_per_target):.4f}"
    elif linear_sum_assignment is not None and n_needles >= n_targets:
        # Cost[i,j] = distance(needle i, target j); assign needles to targets to minimize total distance
        cost = np.zeros((n_needles, n_targets))
        for i in range(n_needles):
            for j in range(n_targets):
                cost[i, j] = np.linalg.norm(needle_locs_np[i] - targets_np[j])
        # linear_sum_assignment minimizes sum; row_ind = needle indices, col_ind = target indices (one per target)
        needle_idx, target_idx = linear_sum_assignment(cost)
        n_pairs = len(needle_idx)
        total_dist = 0.0
        for k in range(n_pairs):
            ni, tj = needle_idx[k], target_idx[k]
            d = float(cost[ni, tj])
            total_dist += d
            print(f"  Target {tj}: needle {ni} at dist {d:.4f}  (needle sum={needle_locs_np[ni].sum():.4f})")
        print(f"\nOptimal assignment: total distance = {total_dist:.4f}")
        assert total_dist < n_targets * max_ok_dist, (
            f"Expected total distance < {n_targets * max_ok_dist}, got {total_dist:.4f}"
        )
        for k in range(n_pairs):
            assert cost[needle_idx[k], target_idx[k]] < max_ok_dist, (
                f"Target {target_idx[k]} assigned distance {cost[needle_idx[k], target_idx[k]]:.4f} >= {max_ok_dist}"
            )
    else:
        # No scipy or fewer needles than targets: for each target, use closest needle
        for j in range(n_targets):
            dists_j = np.array([np.linalg.norm(needle_locs_np[i] - targets_np[j]) for i in range(n_needles)])
            i_best = int(np.argmin(dists_j))
            d_best = float(dists_j[i_best])
            print(f"  Target {j}: closest needle {i_best} at dist {d_best:.4f}")
            assert d_best < max_ok_dist, f"Target {j} closest needle distance {d_best:.4f} >= {max_ok_dist}"
    print("test.py passed.")


if __name__ == "__main__":
    main()
