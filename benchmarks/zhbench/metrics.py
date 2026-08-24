"""Metrics for multi-optimum search.

The old benchmark's headline was cumulative best-y, which is not what ZoMBI-Hop is
for, and its ``pct_matched`` counted a true optimum as "found" whenever ANY sample
landed within 0.05 of it. That rewards scattering points, which is why random
sampling won it. Nothing in it asked whether the method could say WHERE the optima
were.

So the headline here is the quality of the *declared* solution set:

    S = the optima the method claims to have found.

ZoMBI-Hop declares needles, so S is read straight out of its DataHandler. Standard
BO methods declare nothing, so S is extracted post hoc from their samples by the
best procedure available to them (:func:`posthoc_solution_set`) -- the same
courtesy ROBOT extends to single-solution baselines. Both are then scored
identically.

Three groups:

  A. Declared-optima quality at end of budget -- ``peak_ratio`` (recall),
     ``precision``, ``f1``, ``dist_to_needles``. Headline.
  B. Sample efficiency -- ``reached_ratio(t)`` and time-to-k-th-optimum. A true
     optimum counts as reached only when a sample lands both NEAR it and HIGH
     enough, which is what stops uniform scattering from claiming it.
  C. Cost and hygiene -- SnAKe input-change cost, duplicate fraction, wall clock.

Plus ``landscape_contrast``, which reports how discriminative an objective
actually is. This matters: the real 3-D campaign GP has 24 detected peaks but only
4 of them clear the 99th percentile of uniform random sampling, so a high
``peak_ratio`` there means much less than the same number on a sharp synthetic
landscape. Reporting it stops the suite from over-claiming.

A note on comparing across dimensions, which the team has flagged twice (Brianna:
"any distance based metrics can't be compared directly ... the metrics are more
forgiving in higher dimensions"; Aleks: "fix the dimensionality scaling of distance
metrics"). A fixed match radius r is an ABSOLUTE, physically meaningful tolerance
(0.05 in composition L2 is about the printer's resolution), so it is the right
primary. But the fraction of the simplex inside radius r collapses as d grows,
which is why ``pct_matched`` went to zero at 10-D. The cross-dimension quantity to
read is therefore :func:`lift_over_random` -- performance relative to a
matched-budget random search on the SAME landscape -- which is dimensionless and
comparable up the ladder.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

# Composition-L2 match radius. Imported from the core so the benchmark and the
# tuner agree by construction (optimize/eval_metrics.MATCH_RADIUS == 0.05).
_BIG = 1e6


def match_radius() -> float:
    from ._repo import eval_metrics
    return float(eval_metrics().MATCH_RADIUS)


# --- helpers -----------------------------------------------------------------

def _as2d(X, dim: int | None = None) -> np.ndarray:
    if X is None:
        return np.empty((0, dim or 0))
    A = np.asarray(X, dtype=float)
    if A.size == 0:
        return np.empty((0, dim or (A.shape[-1] if A.ndim > 1 else 0)))
    return np.atleast_2d(A)


def pairwise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    if A.size == 0 or B.size == 0:
        return np.empty((A.shape[0], B.shape[0]))
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)


def merge_true_optima(optima, values=None, min_sep: float | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Collapse reference optima that sit closer together than ``min_sep``.

    Necessary because ``peak_ratio`` is otherwise ill-posed: the real 3-D campaign
    GP has two detected peaks 0.067 apart, and with r = 0.05 a single declared
    needle sits within r of both. Keeps the highest-valued member of each cluster.
    ``min_sep`` defaults to 2r, the separation at which one point can no longer
    satisfy two optima.
    """
    P = _as2d(optima)
    if P.shape[0] == 0:
        return P, np.empty(0)
    v = (np.asarray(values, dtype=float).ravel() if values is not None
         else np.zeros(P.shape[0]))
    sep = 2.0 * match_radius() if min_sep is None else float(min_sep)
    order = np.argsort(-v) if values is not None else np.arange(P.shape[0])
    kept: list[int] = []
    for i in order:
        if all(np.linalg.norm(P[i] - P[j]) >= sep for j in kept):
            kept.append(int(i))
    idx = np.asarray(kept, dtype=int)
    return P[idx], v[idx]


def _threshold_matching(S: np.ndarray, T: np.ndarray, r: float) -> int:
    """Maximum number of (declared, true) pairs that can be matched one-to-one
    with distance <= r.

    One declared point may credit at most one true optimum, and vice versa. Without
    this, a single needle dropped between two nearby optima would "find" both, and
    a cluster of 50 needles on one optimum would look as good as 50 spread out.
    """
    if S.shape[0] == 0 or T.shape[0] == 0:
        return 0
    D = pairwise(S, T)
    cost = np.where(D <= r, D, _BIG)
    rows, cols = linear_sum_assignment(cost)
    return int((cost[rows, cols] < _BIG).sum())


# --- A. declared-optima quality ----------------------------------------------

def solution_set_scores(S, true_optima, true_values=None, r: float | None = None
                        ) -> dict:
    """Recall / precision / F1 / Hungarian distance for a declared solution set."""
    r = match_radius() if r is None else float(r)
    T, _tv = merge_true_optima(true_optima, true_values)
    S = _as2d(S, dim=T.shape[1] if T.size else None)
    n_true, n_dec = T.shape[0], S.shape[0]
    n_match = _threshold_matching(S, T, r)
    peak_ratio = n_match / n_true if n_true else float("nan")
    precision = n_match / n_dec if n_dec else 0.0
    f1 = (0.0 if (peak_ratio + precision) == 0 or not np.isfinite(peak_ratio)
          else 2 * peak_ratio * precision / (peak_ratio + precision))

    from ._repo import eval_metrics
    dist = float(eval_metrics().metric_dist_to_needles(S, list(T))) if n_true else float("nan")

    return {
        "peak_ratio": float(peak_ratio),
        "precision": float(precision),
        "f1": float(f1),
        "dist_to_needles": dist,
        "n_declared": int(n_dec),
        "n_true_optima": int(n_true),
        "n_matched": int(n_match),
        "match_radius": r,
    }


def posthoc_solution_set(X, y, k: int, min_sep: float | None = None,
                         top_frac: float = 1.0) -> np.ndarray:
    """Extract a declared solution set from samples, for methods that declare none.

    Greedy value-ordered selection with an exclusion radius: walk the samples from
    best to worst and accept one whenever it is at least ``min_sep`` from every
    point already accepted. This is the standard way to pull distinct niches out of
    a sample cloud, and it is the most generous procedure available to a
    single-solution method -- it uses only information the method already had.

    ``min_sep`` defaults to 2r so two accepted points can never match one optimum.
    Returns up to ``k`` points, best first.
    """
    X = _as2d(X)
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] == 0 or k <= 0:
        return np.empty((0, X.shape[1] if X.ndim > 1 else 0))
    sep = 2.0 * match_radius() if min_sep is None else float(min_sep)
    order = np.argsort(-y)
    if 0.0 < top_frac < 1.0:
        order = order[: max(k, int(np.ceil(top_frac * order.size)))]
    chosen: list[int] = []
    for i in order:
        if len(chosen) >= k:
            break
        if all(np.linalg.norm(X[i] - X[j]) >= sep for j in chosen):
            chosen.append(int(i))
    return X[np.asarray(chosen, dtype=int)] if chosen else np.empty((0, X.shape[1]))


def peak_ratio_curve(X, y, true_optima, true_values=None, k_max: int | None = None,
                     r: float | None = None) -> dict:
    """peak_ratio and precision as a function of how many solutions are declared.

    Reporting a single number at |S| = n_true makes the choice of |S| load-bearing
    and quietly tells every method how many optima exist. This curve removes that:
    it is the precision-recall trade-off of the post-hoc extractor, so no single
    budget of declarations has to be defended.
    """
    r = match_radius() if r is None else float(r)
    T, tv = merge_true_optima(true_optima, true_values)
    n_true = T.shape[0]
    k_max = int(k_max if k_max is not None else max(2 * n_true, 10))
    S = posthoc_solution_set(X, y, k=k_max, min_sep=2.0 * r)
    ks, recalls, precisions = [], [], []
    for k in range(1, S.shape[0] + 1):
        m = _threshold_matching(S[:k], T, r)
        ks.append(k)
        recalls.append(m / n_true if n_true else float("nan"))
        precisions.append(m / k)
    return {"k": ks, "peak_ratio": recalls, "precision": precisions}


# --- B. sample efficiency ----------------------------------------------------

def reached_flags(X, y_true, true_optima, true_values, r: float | None = None,
                  value_tol: float = 0.25, background: float | None = None
                  ) -> np.ndarray:
    """First sample index at which each true optimum is reached (inf if never).

    A true optimum j is reached at sample i when

        ||x_i - x*_j|| <= r   AND   y_i >= y*_j - value_tol * (y*_j - background)

    The value condition is the part that matters. Without it, a method that simply
    scatters points over the simplex "finds" every optimum it happens to land near,
    regardless of whether the sample was any good -- which is exactly how random
    search won the old coverage metric.

    The tolerance is expressed as a fraction of the peak's PROMINENCE above the
    landscape background, not as a fraction of y. That keeps it meaningful on
    landscapes whose values live in a narrow band: the real 4-D campaign GP spans
    0.48-0.88, so a 4.5%-of-y tolerance would be 40% of the entire range, and the
    condition would do nothing.
    """
    r = match_radius() if r is None else float(r)
    X = _as2d(X)
    y_true = np.asarray(y_true, dtype=float).ravel()
    T, tv = merge_true_optima(true_optima, true_values)
    n_true = T.shape[0]
    out = np.full(n_true, np.inf)
    if X.shape[0] == 0 or n_true == 0:
        return out
    bg = float(np.median(y_true)) if background is None else float(background)
    D = pairwise(X, T)                      # (n_samples, n_true)
    for j in range(n_true):
        thresh = tv[j] - value_tol * max(tv[j] - bg, 0.0)
        hit = np.nonzero((D[:, j] <= r) & (y_true >= thresh))[0]
        if hit.size:
            out[j] = int(hit[0])
    return out


def reached_ratio_curve(first_reached: np.ndarray, n_samples: int,
                        step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Fraction of true optima reached, as a function of samples spent."""
    n_true = first_reached.size
    grid = np.arange(0, n_samples + 1, step)
    if n_true == 0:
        return grid, np.full(grid.size, np.nan)
    ratio = np.array([(first_reached < t).sum() / n_true for t in grid], dtype=float)
    return grid, ratio


def time_to_k(first_reached: np.ndarray) -> dict[int, float]:
    """Samples needed to reach the 1st, 2nd, ... k-th distinct optimum."""
    order = np.sort(first_reached)
    return {k + 1: (float(order[k]) if np.isfinite(order[k]) else float("inf"))
            for k in range(order.size)}


# --- C. cost and hygiene -----------------------------------------------------

def input_cost(X) -> float:
    """SnAKe's input-change cost: total composition distance travelled.

    A printed line of q compositions is a single contiguous sweep, so it costs
    roughly its own length. A scattered batch of q points costs a full tour. This
    is the physical price a batch baseline would pay in the lab but does not pay in
    this benchmark, so measuring it is how we stay honest about the advantage the
    baselines are being given.
    """
    X = _as2d(X)
    if X.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(X, axis=0), axis=1).sum())


def dup_fraction(X, dim: int | None = None) -> float:
    from ._repo import eval_metrics
    return float(eval_metrics().metric_dup_fraction(_as2d(X), dim=dim))


def best_y_curve(y) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    return np.maximum.accumulate(y) if y.size else y


# --- D. how discriminative is this objective? --------------------------------

def landscape_contrast(fn, true_optima, true_values=None, dim: int | None = None,
                       n_probe: int = 4000, seed: int = 0,
                       domain: str = "simplex") -> dict:
    """How far the reference optima stand above a uniform random sample.

    Returns the fraction of reference optima that exceed the 99th percentile of
    uniformly sampled objective values. Near 1.0 the optima are genuinely sharp and
    a peak_ratio is meaningful; near 0 they are shallow bumps on a smooth surface
    and any method -- including random -- will look good.
    """
    T = _as2d(true_optima)
    d = dim or (T.shape[1] if T.size else None)
    if d is None:
        return {}
    rng = np.random.default_rng(seed)
    Xr = (rng.random((n_probe, d)) if domain == "cube"
          else rng.dirichlet(np.ones(d), size=n_probe))
    vr = np.asarray([float(fn(x)) for x in Xr], dtype=float)
    tv = (np.asarray(true_values, dtype=float).ravel() if true_values is not None
          else np.asarray([float(fn(p)) for p in T], dtype=float))
    p99 = float(np.percentile(vr, 99))
    rng_span = float(vr.max() - np.median(vr)) or 1.0
    return {
        "random_median": float(np.median(vr)),
        "random_p99": p99,
        "random_max": float(vr.max()),
        "peak_value_min": float(tv.min()) if tv.size else float("nan"),
        "peak_value_max": float(tv.max()) if tv.size else float("nan"),
        "frac_peaks_above_random_p99": float((tv > p99).mean()) if tv.size else float("nan"),
        "mean_peak_prominence": float(((tv - np.median(vr)) / rng_span).mean())
        if tv.size else float("nan"),
    }


def lift_over_random(value: float, random_value: float) -> float:
    """Performance relative to matched-budget random search on the same landscape.

    The cross-dimension quantity. A fixed match radius makes raw ``peak_ratio``
    incomparable between d=3 and d=12 because the fraction of the simplex within r
    of a point collapses with d; dividing by what random achieves under exactly the
    same geometry cancels that.
    """
    if not np.isfinite(random_value) or random_value <= 0:
        return float("inf") if value > 0 else float("nan")
    return float(value / random_value)


# --- top-level assembly ------------------------------------------------------

def compute_all(run, objective, declared=None, wall_s: float | None = None,
                value_tol: float = 0.25) -> dict:
    """Every metric for one finished run.

    ``run`` is a :class:`~.protocol.ObjectiveRun`; ``objective`` is a
    :class:`~.objectives.Objective`; ``declared`` is the method's own solution set
    or None, in which case one is extracted post hoc.
    """
    h = run.stacked()
    X, y_obs, y_true = h["X_actual"], h["y_observed"], h["y_true"]
    T, tv = merge_true_optima(objective.true_optima, objective.true_values)
    r = match_radius()

    declared_is_own = declared is not None and len(declared) > 0
    S_posthoc = posthoc_solution_set(X, y_obs, k=max(T.shape[0], 1), min_sep=2.0 * r)
    S = _as2d(declared) if declared_is_own else S_posthoc

    out: dict = {"declared_source": "method" if declared_is_own else "posthoc"}
    out.update(solution_set_scores(S, T, tv, r=r))
    # Always report the post-hoc set too, so ZoMBI-Hop's declarations can be
    # compared against what its own samples would have supported.
    ph = solution_set_scores(S_posthoc, T, tv, r=r)
    out.update({f"posthoc_{k}": v for k, v in ph.items()
                if k in ("peak_ratio", "precision", "f1", "dist_to_needles", "n_declared")})

    first = reached_flags(X, y_true, T, tv, r=r, value_tol=value_tol)
    grid, ratio = reached_ratio_curve(first, run.protocol.n_samples, step=run.protocol.batch_size)
    tk = time_to_k(first)
    out.update({
        "reached_ratio_final": float(ratio[-1]) if ratio.size else float("nan"),
        "n_reached": int(np.isfinite(first).sum()),
        "t_first_optimum": tk.get(1, float("inf")),
        "t_half_optima": tk.get(max(1, T.shape[0] // 2), float("inf")),
        "reached_curve_t": grid.tolist(),
        "reached_curve_ratio": ratio.tolist(),
    })

    out.update({
        "best_y": float(y_true.max()) if y_true.size else float("nan"),
        "input_cost": input_cost(X),
        "dup_fraction": dup_fraction(X, dim=objective.dim),
        "n_samples": int(run.n_samples),
        "n_truncated": int(run.n_truncated),
        "n_batches": int(run.batch_idx),
    })
    if wall_s is not None:
        out["wall_s"] = float(wall_s)
        out["s_per_decision"] = float(wall_s / max(1, run.batch_idx))
    return out
