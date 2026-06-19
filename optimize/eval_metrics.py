"""
Evaluation metrics for ZoMBI-Hop runs.

Constants are tuned at the 3-simplex (d=3) and applied as-is at every dimension
(no dimension scaling).

  * ``unmatched_penalty(d)``     — ``UNMATCHED_PENALTY``
  * ``match_radius_ilr(d)``       — ILR L2 match radius
  * ``dup_threshold_ilr(d)``      — ILR duplicate radius
  * Matched composition distances — normalised by ``√2`` (simplex L2 diameter)
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str, rel_path: str):
    path = _ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_simplex = _load_module("_metric_simplex", "src/utils/simplex.py")
composition_to_ilr = _simplex.composition_to_ilr

UNMATCHED_PENALTY = 10.0
MATCH_RADIUS = 0.05
# ILR L2 radius at d=3, calibrated ≈ composition-L2 ``MATCH_RADIUS`` on typical needles.
MATCH_RADIUS_ILR = 0.12
NOISE_LEVEL_ILR = 0.03
SIMPLEX_L2_DIAMETER = math.sqrt(2.0)


def infer_metric_dim(
    dim: int | None,
    *arrays: np.ndarray | list | None,
) -> int:
    if dim is not None:
        return int(dim)
    for arr in arrays:
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            continue
        if a.ndim == 1:
            return int(a.shape[0])
        return int(a.shape[-1])
    return 3


def compositions_to_ilr(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return np.empty((0, max(infer_metric_dim(None, X) - 1, 0)), dtype=float)
    t = torch.as_tensor(X, dtype=torch.float64)
    return composition_to_ilr(t).numpy()


def unmatched_penalty(dim: int) -> float:
    """Per-miss penalty (fixed across dimensions)."""
    return UNMATCHED_PENALTY


def match_radius_ilr(dim: int) -> float:
    """Needle–optimum match radius in ILR L2 (fixed across dimensions)."""
    return MATCH_RADIUS_ILR


def dup_threshold_ilr(dim: int) -> float:
    """Duplicate-sample radius in ILR L2, half the input-noise scale (fixed across dimensions)."""
    return NOISE_LEVEL_ILR / 2.0


def metric_dist_to_needles(
    discovered: np.ndarray,
    true_optima: list[np.ndarray],
    *,
    dim: int | None = None,
    penalty: float | None = None,
) -> float:
    """Symmetric greedy matching distance (composition L2 / √2 + scaled penalty)."""
    d = infer_metric_dim(dim, discovered, true_optima)
    pen = float(penalty if penalty is not None else unmatched_penalty(d))
    n_opt = len(true_optima)
    if n_opt == 0:
        return 0.0
    n_disc = len(discovered)
    if n_disc == 0:
        return pen
    used: set[int] = set()
    total = 0.0
    for t in true_optima:
        best_d, best_j = float("inf"), -1
        for j, needle in enumerate(discovered):
            if j in used:
                continue
            dist = float(np.linalg.norm(np.asarray(needle) - np.asarray(t)))
            if dist < best_d:
                best_d, best_j = dist, j
        if best_j >= 0:
            used.add(best_j)
            total += best_d / SIMPLEX_L2_DIAMETER
        else:
            total += pen
    total += (n_disc - len(used)) * pen
    return total / max(n_disc, n_opt)


def metric_dup_fraction(
    X_all: np.ndarray,
    threshold: float | None = None,
    *,
    dim: int | None = None,
) -> float:
    """Fraction of samples with an ILR neighbour within ``dup_threshold_ilr(d)``."""
    X_all = np.asarray(X_all, dtype=float)
    n = len(X_all)
    if n <= 1:
        return 0.0
    d = infer_metric_dim(dim, X_all)
    thr = float(threshold if threshold is not None else dup_threshold_ilr(d))
    Z = compositions_to_ilr(X_all)
    diff = Z[:, None, :] - Z[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    return float((dists < thr).any(axis=1).mean())


def metric_pct_matched(
    discovered: np.ndarray,
    true_optima: list[np.ndarray],
    radius: float | None = None,
    *,
    dim: int | None = None,
) -> float:
    """Percentage of discovered needles within ILR ``match_radius_ilr(d)`` of a true optimum."""
    if len(discovered) == 0 or not true_optima:
        return 0.0
    d = infer_metric_dim(dim, discovered, true_optima)
    r = float(radius if radius is not None else match_radius_ilr(d))
    Z_disc = compositions_to_ilr(np.asarray(discovered, dtype=float))
    Z_opt = compositions_to_ilr(np.asarray(true_optima, dtype=float))
    valid = 0
    for z in Z_disc:
        if float(np.linalg.norm(Z_opt - z, axis=1).min()) <= r:
            valid += 1
    return 100.0 * valid / len(Z_disc)


def metric_avg_pairwise_dist(discovered: np.ndarray) -> float:
    """Average pairwise Euclidean distance between discovered needles (composition L2)."""
    disc = np.asarray(discovered, dtype=float)
    n = len(disc)
    if n < 2:
        return 0.0
    diff = disc[:, None, :] - disc[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    iu = np.triu_indices(n, k=1)
    return float(dists[iu].mean())
