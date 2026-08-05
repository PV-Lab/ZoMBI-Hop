"""
Dimension-aware evaluation metrics for ZoMBI-Hop runs.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

# Soft unmatched cost ≈ half the 3-simplex L2 diameter (√2 / 2).
# Legacy value was 10.0, which made |n_disc − n_opt| dominate geometry.
UNMATCHED_PENALTY = 0.7071067811865476  # ≈ √2 / 2
MATCH_RADIUS = 0.05
# Input-noise scale (per-component composition std) used for the duplicate-sample
# radius; matched to the measured average input noise of data/2nd_real_run.db.
NOISE_LEVEL = 0.064

# Floor for the zoom-scaled duplicate distance (composition L2). The duplicate
# radius is NOISE_LEVEL/2 at full domain but shrinks with the zoom-zone size (see
# metric_dup_fraction / zoom_size_fraction), so tightly-packed points inside a zoom
# aren't penalised — this rewards the optimiser for zooming. The floor stops it
# collapsing to ~0 at deep zooms (below it two points are the same sample). Tunable.
DUP_DIST_FLOOR = 0.005


def as_numpy(x, *, dtype=float) -> np.ndarray:
    """Coerce numpy arrays or CPU/CUDA tensors to a host ndarray."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=dtype)
    if arr.dtype == np.dtype(object):
        parts = [as_numpy(v, dtype=dtype) for v in arr.ravel()]
        if not parts:
            return np.empty((0,), dtype=dtype)
        if parts[0].ndim == 1:
            return np.stack([p.ravel() for p in parts], axis=0)
        return np.array(parts, dtype=dtype)
    return arr


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


def unmatched_penalty(dim: int) -> float:
    """Per-miss penalty (dimension-independent)."""
    del dim
    return UNMATCHED_PENALTY


def match_radius_comp(dim: int) -> float:
    """Needle–optimum match radius in composition L2."""
    del dim  # simplex L2 diameter does not grow with d
    return MATCH_RADIUS


def dup_threshold_comp(dim: int) -> float:
    """Duplicate-sample radius in composition L2 (half input-noise scale)."""
    del dim
    return NOISE_LEVEL / 2.0


def metric_dist_to_needles(
    discovered: np.ndarray,
    true_optima: list[np.ndarray],
    *,
    dim: int | None = None,
    penalty: float | None = None,
) -> float:
    """Symmetric greedy matching distance (composition L2 + fixed penalty)."""
    d = infer_metric_dim(dim, discovered, true_optima)
    pen = float(penalty if penalty is not None else UNMATCHED_PENALTY)
    n_opt = len(true_optima)
    if n_opt == 0:
        return 0.0
    n_disc = len(discovered)
    if n_disc == 0:
        return pen
    disc = as_numpy(discovered, dtype=float)
    used: set[int] = set()
    total = 0.0
    for t in true_optima:
        t_np = as_numpy(t, dtype=float).ravel()
        best_d, best_j = float("inf"), -1
        for j in range(len(disc)):
            if j in used:
                continue
            dist = float(np.linalg.norm(disc[j].ravel() - t_np))
            if dist < best_d:
                best_d, best_j = dist, j
        if best_j >= 0:
            used.add(best_j)
            total += best_d
        else:
            total += pen
    total += (n_disc - len(used)) * pen
    return total / max(n_disc, n_opt)


def zoom_size_fraction(zoom_bounds, full_bounds=None) -> float:
    """Linear size of a zoom box relative to the full domain, in (0, 1].

    Geometric mean of the per-axis extent ratios: ``(∏ᵢ extentᵢ_zoom /
    extentᵢ_full)^(1/d)``. This is the *linear* scale of the zone (a distance
    threshold scales with it), so a zoom to 1/8 of the domain *volume* gives
    ``s≈0.5`` in 3-D. Returns 1.0 at the full domain. ``full_bounds`` defaults to
    the unit box ``[0,1]^d`` (the ZoMBI global domain).
    """
    zb = as_numpy(zoom_bounds, dtype=float)          # (2, d)
    ext = np.clip(zb[1] - zb[0], 0.0, None)
    if full_bounds is None:
        full_ext = np.ones_like(ext)
    else:
        fb = as_numpy(full_bounds, dtype=float)
        full_ext = np.clip(fb[1] - fb[0], 1e-12, None)
    ratio = np.clip(ext / full_ext, 1e-12, 1.0)
    return float(np.exp(np.mean(np.log(ratio))))


def metric_dup_fraction(
    X_all: np.ndarray,
    threshold: float | None = None,
    *,
    dim: int | None = None,
    zoom_sizes: np.ndarray | None = None,
) -> float:
    """Fraction of samples with a composition-L2 neighbour within the dup radius.

    The base radius is ``dup_threshold_comp(d)`` (= NOISE_LEVEL/2). When
    ``zoom_sizes`` is given — a per-point array of zoom-zone linear sizes ``s`` in
    (0,1], one per row of ``X_all`` (see ``zoom_size_fraction``) — each point uses a
    *scaled* radius ``max(base * s, DUP_DIST_FLOOR)``, so points sampled inside a
    small zoom zone must be much closer to count as duplicates. With ``zoom_sizes``
    None the behaviour is the original single global radius (``s ≡ 1``).
    """
    X_all = as_numpy(X_all, dtype=float)
    n = len(X_all)
    if n <= 1:
        return 0.0
    d = infer_metric_dim(dim, X_all)
    base = float(threshold if threshold is not None else dup_threshold_comp(d))
    # A KD-tree nearest-neighbour query is O(N log N) time / O(N) memory; the
    # naive N×N×D difference array would peak at tens of GiB for N~20k samples.
    tree = cKDTree(X_all)
    nn_dist, _ = tree.query(X_all, k=2)  # k=2: self (dist 0) + nearest other
    nn = nn_dist[:, 1]
    if zoom_sizes is None:
        thr = base
    else:
        s = np.asarray(zoom_sizes, dtype=float).reshape(-1)
        if s.shape[0] != n:   # length mismatch → fall back to the global radius
            thr = base
        else:
            thr = np.maximum(base * s, DUP_DIST_FLOOR)
    return float((nn < thr).mean())


def metric_pct_matched_comp(
    discovered: np.ndarray,
    true_optima: list[np.ndarray],
    radius: float | None = None,
    *,
    dim: int | None = None,
) -> float:
    """% of discovered needles within composition-L2 ``match_radius_comp(d)`` of a true optimum."""
    if len(discovered) == 0 or not true_optima:
        return 0.0
    d = infer_metric_dim(dim, discovered, true_optima)
    r = float(radius if radius is not None else match_radius_comp(d))
    disc = as_numpy(discovered, dtype=float)
    opt = as_numpy(true_optima, dtype=float)
    valid = 0
    for needle in disc:
        if float(np.linalg.norm(opt - needle, axis=1).min()) <= r:
            valid += 1
    return 100.0 * valid / len(disc)


def metric_pct_matched(
    discovered: np.ndarray,
    true_optima: list[np.ndarray],
    radius: float | None = None,
    *,
    dim: int | None = None,
) -> float:
    """Alias for ``metric_pct_matched_comp``."""
    return metric_pct_matched_comp(discovered, true_optima, radius, dim=dim)


def metric_n_points_penalty(n_points: int) -> float:
    """Penalty on the total number of points sampled during a run (minimised).

    Sampling is expensive, so for a fixed solution quality fewer samples is
    better — the penalty therefore grows linearly with the sample count. A run
    that sampled *nothing* (``n_points == 0``) is not an efficient run, it is a
    broken one (ZoMBI never picked a single point), so it incurs an infinite
    penalty rather than the minimal score a plain count would give it. Callers
    feeding this to the GP must exclude the infinite case (it cannot be
    standardised); the finite branch equals the raw point count.
    """
    n = int(n_points)
    if n <= 0:
        return float("inf")
    return float(n)


def metric_avg_pairwise_dist(discovered: np.ndarray) -> float:
    """Average pairwise Euclidean distance between discovered needles (composition L2)."""
    disc = as_numpy(discovered, dtype=float)
    n = len(disc)
    if n < 2:
        return 0.0
    diff = disc[:, None, :] - disc[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    iu = np.triu_indices(n, k=1)
    return float(dists[iu].mean())
