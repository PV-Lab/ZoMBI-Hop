"""
Dimension-aware evaluation metrics for ZoMBI-Hop runs.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

# Cost charged for every unmatched member of the larger set, AND the radius past
# which a matched needle earns no credit (see metric_dist_to_needles).
#
# This was 10.0 until 2026-08-11. The problem with 10.0 is one of scale: measured
# matched-needle distances are 0.05-0.55 and composition L2 on the simplex cannot
# exceed its diameter ~1.414, so a 10.0 penalty is 20-200x anything it is averaged
# with. The consequence is not subtle — on an 8-peak landscape the localization term
# contributes 0.6% of the score, i.e. the metric is ~99% a comparison of how many
# needles were declared against how many exist, with the distances it is named for
# as noise on top. (That is also why ``metric_pct_matched`` read 0.0 everywhere.)
# At 0.5 — just above the largest matched distance actually observed — the two terms
# are commensurate, so the metric responds to WHERE needles landed and not only to
# how many there were.
#
# TRADEOFF, recorded because it is why 10.0 was chosen originally: the penalty is the
# whole anti-spam deterrent. Scoring a real run against the same run plus 200 needles
# jittered into the densest optima cluster gives a spam/honest ratio of 51.8x at 10.0
# but only 2.6x at 0.5. Over-declaration is still penalised — that is the
# ``(n_declared - n_true)`` term — just no longer to the exclusion of everything else.
#
# Changing this constant deliberately moves every consumer at once, which is the
# point: the same number is the failure sentinel in ``evaluate.run_once``,
# ``run_mobo._run_trial`` and ``run_mobo._failure_penalty_Y``, all of which mean "the
# worst attainable distance". At 10.0 those sentinels sat 20-50x outside the range of
# real measurements and dominated the standardisation of the MOBO GP's first
# objective; at 0.5 a failed trial scores exactly as badly as a run that declared
# nothing, which is what they were always documented to mean.
#
# Alternatives, if the domain answer differs: ~0.157 (input-noise L2 =
# NOISE_LEVEL*sqrt(d), the floor below which a needle cannot be localized at all), or
# 1.414 (simplex diameter — cap nothing, credit every distance).
UNMATCHED_PENALTY = 0.5

# The pre-2026-08-11 value, kept so old runs can be re-scored on their original
# scale (``summary_table --dist-cutoff 10``) without hunting for the number.
LEGACY_UNMATCHED_PENALTY = 10.0

MATCH_RADIUS = 0.05
# Input-noise scale (per-component composition std) used for the duplicate-sample
# radius; measured from runs/run_39af/composition_log.jsonl, the 6-dim hardware run of 2026-08-12 (109 lines / 2042 samples), which logs the sent composition directly: pooled per-component std 0.128, mean L2 0.271 (see run_mobo.NOISE_LEVEL).
NOISE_LEVEL = 0.128

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
    """Symmetric minimum-cost matching distance (composition L2 + fixed penalty).

    Needles and true optima are paired one-to-one at MINIMUM total cost, and the
    score is the mean over ``max(n_discovered, n_true)`` of the matched distances
    plus ``penalty`` for every unmatched member of the larger set. Lower is better;
    the range is ``[0, penalty]``.

    Two things changed here on 2026-08-11, and both make the number mean more than
    it used to:

    **The pairing is optimal, not greedy.** The previous implementation walked
    ``true_optima`` in list order and let each take its nearest unclaimed needle.
    That is order-dependent and can cost strictly more than the best pairing — an
    optimum early in the list takes the one needle a later optimum needed, sending
    the later one to the full penalty even though a perfectly good alternative
    existed. The intended quantity was always the minimum total matching cost, which
    ``scipy.optimize.linear_sum_assignment`` computes exactly (Hungarian, O(n^3) —
    trivial at the ~100 needles per run seen here). So this can only ever report a
    distance less than or equal to the old one, never more.

    **Every matched distance is capped at ``penalty``.** Without the cap a needle
    that landed 1.2 away would cost MORE than not declaring it at all, which makes
    the metric non-monotone in the thing it is supposed to reward. With it, the
    penalty reads as what it is: the radius past which a needle earns no credit.

    ``penalty`` defaults to ``UNMATCHED_PENALTY``; pass ``LEGACY_UNMATCHED_PENALTY``
    to score on the pre-2026-08-11 scale. Note that even at the legacy value this
    does NOT reproduce old stored numbers exactly, because the greedy pairing is
    gone — use the values already in ``metrics.json`` for that.
    """
    del dim  # the metric is scale-free in d; kept for call-site symmetry
    pen = float(penalty if penalty is not None else UNMATCHED_PENALTY)
    n_opt = len(true_optima)
    if n_opt == 0:
        return 0.0
    n_disc = len(discovered)
    if n_disc == 0:
        return pen
    disc = as_numpy(discovered, dtype=float).reshape(n_disc, -1)
    opt = np.asarray([as_numpy(t, dtype=float).ravel() for t in true_optima],
                     dtype=float)
    # Capped cost matrix, then the exact minimum-cost assignment. Rectangular is
    # fine: linear_sum_assignment pairs min(n_disc, n_opt) of them and the rest are
    # unmatched by construction.
    cost = np.minimum(np.linalg.norm(disc[:, None, :] - opt[None, :, :], axis=2), pen)
    rows, cols = linear_sum_assignment(cost)
    n = max(n_disc, n_opt)
    return float((cost[rows, cols].sum() + pen * (n - len(rows))) / n)


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


def metric_median_nn_spacing(
    X_all: np.ndarray,
    *,
    dim: int | None = None,
) -> float:
    """Median nearest-neighbour distance between a run's samples, in composition L2.

    The threshold-free counterpart to ``metric_dup_fraction``, and the reason it
    exists: dup fraction counts the samples whose nearest neighbour falls inside a
    radius, so its value depends entirely on that radius. ``NOISE_LEVEL`` moved from
    0.064 to 0.128 in a2deba7 (the input-noise measurement was redone against
    ``run_39af``, which logs the composition the optimiser *sent* rather than
    inferring it), which doubled the radius and left dup fractions from before and
    after the change answering two different questions. Nor can the two eras simply
    be re-scored at one common radius: at any radius wide enough to catch the older
    campaigns' duplicates the newer ones saturate near 1.0 and the metric separates
    nothing, and vice versa.

    This measures directly what dup fraction was a proxy for — how far apart the
    points a run actually sampled are. No radius, no noise level and no zoom scaling
    enter it, so runs from any era sit on one axis.

    **Direction is FLIPPED relative to dup fraction: higher is better.** Wide spacing
    means the run spread its samples out; small spacing means it kept re-measuring
    the same spot.

    Note that spacing scales as ``N**(-1/d)`` in the sample count, so it is only
    comparable at similar N — at d=6 a 7% difference in sample count moves it under
    1%, but an order of magnitude would not be negligible.

    Returns 0.0 for fewer than two samples (no spacing is defined).
    """
    X_all = as_numpy(X_all, dtype=float)
    n = len(X_all)
    if n <= 1:
        return 0.0
    infer_metric_dim(dim, X_all)   # validates shape the same way dup_fraction does
    # Same KD-tree query as metric_dup_fraction: the naive N x N x D difference array
    # would peak at tens of GiB for the N~20k samples these runs produce.
    tree = cKDTree(X_all)
    nn_dist, _ = tree.query(X_all, k=2)  # k=2: self (dist 0) + nearest other
    return float(np.median(nn_dist[:, 1]))


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
