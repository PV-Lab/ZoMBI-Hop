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
actually is. This matters: only 0.21 of the real 3-D campaign's reference peaks
clear the 99th percentile of uniform random sampling, against 0.52 at 4-D and 0.75
at 6-D, so the same ``peak_ratio`` means very different things on the three
campaigns. Reporting it stops the suite from over-claiming.

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

    Pass **requested** compositions, not realized ones. The machine executes what it
    was commanded to; the realization error is a deposition artifact that costs no
    extra syringe travel. Scoring the realized path instead makes hardware-level
    noise look like movement cost and erases most of the line-vs-batch difference
    -- which is exactly what it did before this was fixed.
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
    """How rare a reference optimum's VALUE is under uniform random sampling.

    The headline is ``peak_rarity_median``: for each reference optimum, the
    fraction of uniform probes scoring at least as well. Small means the value
    signal is informative -- a random sampler rarely stumbles onto something that
    good. Large means it does, and a strong ``peak_ratio`` on that objective says
    less than it appears to.

    This used to be "fraction of peaks above the 99th percentile", which is
    unusable on any objective with a ceiling. ``synthetic_data.ensemble`` saturates
    at 1.0, and once more than 1% of the domain sits at the ceiling the 99th
    percentile IS 1.0, so a strict ``peak > p99`` test reports 0.0 for peaks that
    are themselves at 1.0 -- and flips back to 1.0 if you change the probe count.
    Measured on `ensemble(3, n_optima=10)`: 0.00 at 800 probes, 1.00 at 3000. A
    rarity is a rank, so ties at the ceiling are counted rather than mis-signed.

    ``frac_domain_at_max`` reports the saturation directly, because on a saturating
    landscape "the optimum's value is not rare" and "the optimum is easy to locate"
    are different statements -- ``peak_ratio`` is distance-based and still measures
    the second.
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
    med = float(np.median(vr))
    vmax = float(vr.max())
    rng_span = (vmax - med) or 1.0
    # Rarity of each peak's value: P(a uniform draw scores at least as well).
    rarity = (np.asarray([(vr >= t).mean() for t in tv], dtype=float)
              if tv.size else np.empty(0))
    return {
        "random_median": med,
        "random_p99": float(np.percentile(vr, 99)),
        "random_max": vmax,
        "peak_value_min": float(tv.min()) if tv.size else float("nan"),
        "peak_value_max": float(tv.max()) if tv.size else float("nan"),
        "peak_rarity_median": float(np.median(rarity)) if rarity.size else float("nan"),
        "frac_peaks_in_top_1pct": float((rarity <= 0.01).mean()) if rarity.size else float("nan"),
        "frac_domain_at_max": float((vr >= vmax - 1e-9).mean()),
        "mean_peak_prominence": float(((tv - med) / rng_span).mean())
        if tv.size else float("nan"),
        "n_probe": int(vr.size),
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

def needles_declared_curve(declared_at, n_samples: int, step: int = 24
                           ) -> tuple[list, list]:
    """Cumulative count of declared optima against samples spent.

    Worth plotting for ZoMBI-Hop because its needle count has a structural ceiling
    that has nothing to do with the landscape: with ``max_zooms=3, max_iterations=2,
    min_zoom_for_needle=1, min_iters_per_zoom=2`` an activation must spend 4-6 lines
    before it is allowed to declare anything, so ~42 lines (N=1000) buys at most
    7-10 needles no matter how many optima exist. Reading a low ``peak_ratio`` as a
    search failure, when it is really a declaration budget, is the single easiest
    mistake to make with this benchmark.
    """
    at = np.asarray(sorted(declared_at), dtype=float) if len(declared_at) else np.empty(0)
    grid = list(range(0, int(n_samples) + 1, int(step)))
    return grid, [int((at <= t).sum()) for t in grid]


def _score_prefix(X, X_req, y_obs, y_true, T, tv, S, r, value_tol) -> dict:
    """The metric block for one prefix of a run."""
    out = solution_set_scores(S, T, tv, r=r)
    first = reached_flags(X, y_true, T, tv, r=r, value_tol=value_tol)
    out["reached_ratio"] = float(np.isfinite(first).sum() / T.shape[0]) if T.shape[0] else float("nan")
    out["n_reached"] = int(np.isfinite(first).sum())
    out["best_y"] = float(y_true.max()) if y_true.size else float("nan")
    out["input_cost"] = input_cost(X_req)
    return out


def compute_all(run, objective, declared=None, declared_at=None,
                wall_s: float | None = None, value_tol: float = 0.25) -> dict:
    """Every metric for one finished run, at the endpoint and at each prefix.

    ``run`` is a :class:`~.protocol.ObjectiveRun`; ``objective`` is a
    :class:`~.objectives.Objective`; ``declared`` is the method's own solution set
    or None, in which case one is extracted post hoc; ``declared_at`` is the sample
    index at which each declared optimum was declared, so prefixes can be scored
    with only the optima the method had actually committed to by then.
    """
    h = run.stacked()
    X, y_obs, y_true = h["X_actual"], h["y_observed"], h["y_true"]
    X_req = h["X_requested"]
    T, tv = merge_true_optima(objective.true_optima, objective.true_values)
    r = match_radius()

    declared_is_own = declared is not None and len(declared) > 0
    S_posthoc = posthoc_solution_set(X, y_obs, k=max(T.shape[0], 1), min_sep=2.0 * r)
    S = _as2d(declared) if declared_is_own else S_posthoc

    out: dict = {"declared_source": "method" if declared_is_own else "posthoc"}

    # -- the same metrics at each budget checkpoint ---------------------------
    # One 2000-sample run answers "what would N=250 have looked like", because
    # every method is fed the same stream and nothing about a prefix depends on
    # what came after it. 3-D is saturated by uniform sampling long before
    # N=1000 -- ~9 random samples land within r of any given point -- so the small
    # -N columns are the ones that discriminate there.
    by_n: dict[str, dict] = {}
    at_arr = (np.asarray(declared_at, dtype=float)
              if declared_at is not None and len(declared_at) else None)
    for n in run.protocol.eval_at:
        if n > run.n_samples:
            continue
        m = int(n)
        Xp, Xrp, yop, ytp = X[:m], X_req[:m], y_obs[:m], y_true[:m]
        if declared_is_own:
            keep = (int((at_arr <= m).sum()) if at_arr is not None
                    else S.shape[0])
            Sp = S[:keep]
        else:
            Sp = posthoc_solution_set(Xp, yop, k=max(T.shape[0], 1), min_sep=2.0 * r)
        by_n[str(m)] = _score_prefix(Xp, Xrp, yop, ytp, T, tv, Sp, r, value_tol)
    out["by_n"] = by_n
    for m, blk in by_n.items():
        for k in ("peak_ratio", "precision", "f1", "reached_ratio", "best_y",
                  "n_declared"):
            out[f"{k}@{m}"] = blk[k]
    out.update(solution_set_scores(S, T, tv, r=r))
    # Always report the post-hoc set too, so ZoMBI-Hop's declarations can be
    # compared against what its own samples would have supported.
    ph = solution_set_scores(S_posthoc, T, tv, r=r)
    out.update({f"posthoc_{k}": v for k, v in ph.items()
                if k in ("peak_ratio", "precision", "f1", "dist_to_needles", "n_declared")})

    # The precision-recall curve over |S|, so the suite can compare methods at
    # MATCHED numbers of declared optima. Without this the comparison is unfair in a
    # way that is easy to miss: ZoMBI-Hop volunteers however many needles it is
    # confident in (7 on real4d), while a post-hoc set is free to declare n_true
    # (24) guesses. Recall at |S| = n_true therefore caps ZoMBI-Hop at 7/24 for a
    # purely structural reason. Read `peak_ratio` next to `pr_curve`.
    curve = peak_ratio_curve(X, y_obs, T, tv, k_max=max(2 * T.shape[0], 10), r=r)
    out["pr_curve_k"] = curve["k"]
    out["pr_curve_peak_ratio"] = curve["peak_ratio"]
    out["pr_curve_precision"] = curve["precision"]
    if declared_is_own and S.shape[0] > 0:
        k = int(S.shape[0])
        idx = [i for i, kk in enumerate(curve["k"]) if kk == k]
        out["posthoc_peak_ratio_at_n_declared"] = (curve["peak_ratio"][idx[0]]
                                                   if idx else float("nan"))

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

    nd_t, nd_n = needles_declared_curve(
        declared_at if declared_at is not None else [], run.protocol.n_samples,
        step=run.protocol.batch_size)
    out.update({
        "best_y": float(y_true.max()) if y_true.size else float("nan"),
        "input_cost": input_cost(h["X_requested"]),
        "dup_fraction": dup_fraction(X, dim=objective.dim),
        "n_samples": int(run.n_samples),
        "n_truncated": int(run.n_truncated),
        "n_batches": int(run.batch_idx),
        "realized_noise_std": run.realized_noise_std(),
        "needles_curve_t": nd_t,
        "needles_curve_n": nd_n,
    })
    if wall_s is not None:
        out["wall_s"] = float(wall_s)
        out["s_per_decision"] = float(wall_s / max(1, run.batch_idx))
    return out
