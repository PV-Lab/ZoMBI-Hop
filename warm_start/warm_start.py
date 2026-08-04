"""
Warm-start scheme for ZoMBI-Hop
===============================

A *warm start* spends 15% of a run's total iteration budget on a space-filling
design of the (d-dimensional) probability simplex, then uses those points to
seed the search / locate candidate optima before the adaptive Bayesian phase
begins.

Budgets (total iterations, from the run configs):
    3d  ->   600 pts   (15% ->   90 pts)
    4d  ->  1300 pts   (15% ->  195 pts)
    10d -> 20000 pts   (15% -> 3000 pts)

Hardware constraint: **lines of 24**
------------------------------------
Measurements are not collected one composition at a time — the instrument runs a
*line*, and a line is 24 points that must be **evenly spaced** between a start
and an end composition.  The two endpoints are free (anywhere on the simplex),
but everything in between is pinned to the segment at a fixed pitch.  A warm
start is therefore a union of ``L`` segments, not ``n`` free points, and the
question is *where to lay the lines* so the segments together cover the simplex.

    3d  ->   90 pts  ->   4 lines (96 pts)
    4d  ->  195 pts  ->   8 lines (192 pts)
    10d -> 3000 pts  -> 125 lines (3000 pts)

:func:`greedy_lines` is the sampler: greedy line-at-a-time placement against a
coverage objective, then coordinate-descent refinement sweeps over whole lines.
See the section comment above it for the objective and the algorithm.

:func:`maximin_simplex` is the older *free-point* sampler, kept because
:mod:`warm_start.greedy_optima` and other callers still use it, and because it
is what ZoMBI-Hop's own ``_space_filling_measurement`` fallback does
(src/core/zombihop.py).  It ignores the line constraint, so it does not describe
a design the hardware can actually measure.

Metrics.  Nearest-neighbour spacing does not describe a line design — points
inside one line sit a fixed ``|p-q|/23`` apart by construction — so a line
design is scored by :func:`coverage_stats`: the distance from a uniform simplex
probe to the nearest measured composition, whose max is the coverage radius.

Noise constants (optimize/run_mobo.py):
    NOISE_LEVEL       = 0.064   # input / SPATIAL per-component composition std
                                #  == zombihop.py input_noise default (0.064)
    OUTPUT_NOISE_FRAC = 0.045   # multiplicative OUTPUT (y) noise fraction
The physically meaningful "noise radius" in composition space is 0.064; 0.045 is
the y-noise fraction.  Both are reported.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARM_START_FRACTION = 0.15

BUDGETS = {           # dim : total iteration budget
    3:   600,
    4:   1300,
    10:  20000,
}

INPUT_NOISE = 0.064    # spatial composition-std noise radius (run_mobo NOISE_LEVEL)
OUTPUT_NOISE_FRAC = 0.045  # multiplicative y-noise fraction (run_mobo OUTPUT_NOISE_FRAC)

# Size of the uniform candidate pool the maximin selector greedily thins.  A
# larger pool gives a finer maximin design at O(pool * n) cost; ~10x the number
# of selected points keeps the min-gap within a few percent of optimal.  The cap
# bounds cost for the big 10d warm start (n=3000) so it stays HPC-friendly.
_POOL_MULTIPLIER = 10
_POOL_MIN = 4000
_POOL_MAX = 30000


def n_warm_start(dim: int) -> int:
    """Number of warm-start points = 15% of the run's total budget."""
    return int(round(WARM_START_FRACTION * BUDGETS[dim]))


# ---------------------------------------------------------------------------
# Maximin (best-candidate) design on the simplex
# ---------------------------------------------------------------------------

def maximin_subset(cand: np.ndarray, n: int, start: int = 0) -> np.ndarray:
    """Greedily pick `n` maximally-separated rows of `cand`; returns their indices.

    Starting from row `start`, repeatedly add the candidate whose *nearest
    already-chosen row is farthest away* — the emptiest spot on the map.
    Distances are tracked incrementally (one elementwise min per pick), so the
    whole selection is O(len(cand) * n), not O(len(cand) * n^2).  Ranking uses
    *squared* distance: sqrt is monotonic, so the argmax is identical while
    avoiding a sqrt over the whole pool every pick.

    The result maximises (greedily) the minimum pairwise Euclidean distance, so
    the selection is spread out and clump-free.  This is the shared core of
    :func:`maximin_simplex` (which applies it to a uniform pool) and of
    :func:`warm_start.greedy_optima.find_optima` (which applies it to the
    high-value candidates), so both stay in step.

    Indices are returned rather than rows so callers can carry along whatever
    they have aligned with `cand` (objective values, line membership, ...).
    """
    cand = np.asarray(cand, float)
    if n > cand.shape[0]:
        raise ValueError(f"cannot pick {n} of {cand.shape[0]} candidates")

    chosen = np.empty(n, dtype=np.intp)
    chosen[0] = start
    # min_d2[i] = squared distance from candidate i to its nearest chosen row.
    min_d2 = ((cand - cand[start]) ** 2).sum(1)
    for k in range(1, n):
        pick = int(np.argmax(min_d2))
        chosen[k] = pick
        np.minimum(min_d2, ((cand - cand[pick]) ** 2).sum(1), out=min_d2)
    return chosen


def maximin_simplex(n: int, dim: int, seed: int | None = None,
                    pool: int | None = None) -> np.ndarray:
    """Draw `n` greedy-maximin (best-candidate) points on the `dim`-simplex.

    Draws a large pool of candidates uniformly on the simplex
    (Dirichlet(1,...,1) == uniform on the probability simplex) and thins it with
    :func:`maximin_subset`.  Returns an (n, dim) array whose rows sum to 1.

    Note this is the *unconstrained* design: it ignores the hardware's
    24-points-per-line constraint, so it is a reference/benchmark rather than a
    measurable plan.  Use :func:`greedy_lines` for a design the instrument can
    actually run.
    """
    if dim < 2:
        raise ValueError("dim must be >= 2")
    if pool is None:
        pool = min(_POOL_MAX, max(_POOL_MIN, _POOL_MULTIPLIER * n))
    pool = max(pool, n)

    rng = np.random.default_rng(seed)
    cand = rng.dirichlet(np.ones(dim), size=pool)          # (pool, dim), uniform
    return cand[maximin_subset(cand, n, start=int(rng.integers(pool)))]


def mean_nn_distance(x: np.ndarray) -> tuple[float, float, float]:
    """Mean / median / min of the Euclidean nearest-neighbour distance.

    Distances are Euclidean in the ambient composition coordinates R^dim — the
    same metric ZoMBI-Hop uses when it compares spatial separations to
    `input_noise` (e.g. `torch.norm(X_a - X_b) <= input_noise` in zombihop.py).
    Uses a KD-tree so the 10d warm start (n=3000) stays O(n log n) in time and
    O(n) in memory rather than materialising an n*n*dim distance tensor.
    """
    from scipy.spatial import cKDTree

    # k=2: the first neighbour is the point itself (distance 0), the second is
    # its true nearest neighbour.
    nn = cKDTree(x).query(x, k=2)[0][:, 1]
    return float(nn.mean()), float(np.median(nn)), float(nn.min())


# ---------------------------------------------------------------------------
# Line-constrained design: greedy coverage with segments of 24 evenly spaced pts
# ---------------------------------------------------------------------------
#
# Why not just run maximin and snap it to lines?  Maximin picks free points; any
# 24 of them are almost never collinear, so a snapped design throws away most of
# the spacing it optimised for.  Instead the *line* is the decision variable: we
# choose whole segments greedily, scoring each one by how much of the simplex it
# actually leaves uncovered.
#
# Coverage is measured against a fixed cloud of uniform "probe" points standing
# in for the continuum of compositions we might care about.  For a design D the
# cost is the mean squared distance from a probe to its nearest design point
#
#     cost(D) = mean_p  min_{x in D} ||p - x||^2
#
# — the standard quantization / coverage error.  Driving it down is exactly
# "leave no region far from a sample"; unlike a pure maximin gap it is smooth in
# the endpoints, so greedy line-by-line descent behaves well.
#
# Each greedy round:
#   1. Take the *emptiest* probe (largest current distance to the design) as an
#      anchor — the region most in need of a line.
#   2. Propose candidate segments: anchor -> an endpoint drawn with probability
#      proportional to its uncovered-ness (``d ** _LINE_BIAS``), plus free
#      pairs drawn the same way, so a line can also be laid across a whole
#      under-covered band rather than always radiating from one hole.
#   3. Materialise each candidate's 24 evenly spaced points, score the resulting
#      cost, and keep the best.
# The endpoints are simplex points and the simplex is convex, so every
# intermediate point of a segment is a valid composition — no projection needed.
#
# A purely greedy pass is myopic: line 1 is placed as if it were the only line,
# and by the time the last one is placed the early lines are in the wrong place.
# So the greedy pass is followed by **refinement sweeps** — coordinate descent on
# whole lines.  Each line in turn is lifted out of the design and re-chosen
# against the coverage its neighbours already provide, with its current position
# among the candidates so a sweep can never make the design worse.  Lifting a
# line out is exact and cheap because we cache ``D[l]``, the per-line squared
# distance from every probe to line ``l``: the design's coverage is ``D.min(0)``
# and the leave-one-out coverage is the second-smallest entry wherever line ``l``
# was the winner (see :func:`_leave_one_out`).

POINTS_PER_LINE = 24   # hardware: one line == 24 evenly spaced compositions

# Candidate segments proposed per line, and the exponent biasing endpoint draws
# toward under-covered regions (higher = greedier about holes).
_LINE_CANDIDATES = 128
_LINE_BIAS = 4.0
# Endpoint pool.  Uniform (Dirichlet(1)) draws concentrate near the centroid, so
# segments built only from them are short and never reach the corners; the
# ``alpha < 1`` half piles mass on the simplex boundary and supplies long,
# space-spanning chords.  Measured effect of the mix is small (the refinement
# sweeps recover most of the difference), so the split is not finely tuned.
_ENDPOINT_POOL = 4000
_ENDPOINT_ALPHA = 0.6
_ENDPOINT_BOUNDARY_FRAC = 0.5
# Refinement sweeps of whole-line coordinate descent after the greedy pass.
_REFINE_SWEEPS = 4
# Exponent p in the coverage cost ``mean_p d^p``.  p=2 is the plain quantization
# error; larger p weights the worst-covered probes more heavily and so trades
# average spacing for a smaller coverage *radius*.  Measured over 3d/4d designs,
# p=4 beats both p=2 and p=8 on mean, p95 *and* max coverage.
_SCORE_POWER = 4.0
# Fraction of the way an endpoint is dragged toward a random simplex point when
# proposing a *local* tweak of an existing line during refinement.
_JITTER = 0.15
# Probe cloud measuring coverage.  Scales with the design size but is capped so
# the 10d warm start (125 lines) stays minutes rather than hours.
_PROBE_MULTIPLIER = 4
_PROBE_MIN = 3000
_PROBE_MAX = 6000
# Cap on the (probes x candidate-points) distance block, so scoring chunks its
# candidates instead of allocating one huge matrix.
_SCORE_BLOCK = 4_000_000


def n_lines(dim: int) -> int:
    """Number of 24-point lines in the warm start = 15% of the budget, rounded.

    The budget is not generally a multiple of 24 (3d: 90 -> 4 lines = 96 pts),
    so the realisable warm start is the nearest whole number of lines.
    """
    return max(1, int(round(n_warm_start(dim) / POINTS_PER_LINE)))


def line_points(p: np.ndarray, q: np.ndarray,
                k: int = POINTS_PER_LINE) -> np.ndarray:
    """The `k` evenly spaced compositions of the line from `p` to `q` (inclusive).

    Both endpoints lie on the simplex and the simplex is convex, so every
    returned row is a valid composition summing to 1.
    """
    t = np.linspace(0.0, 1.0, k).reshape(-1, 1)
    return (1.0 - t) * np.asarray(p, float) + t * np.asarray(q, float)


def _probe_cloud(dim: int, n_probes: int, rng) -> np.ndarray:
    """Uniform simplex points standing in for the compositions we want covered."""
    return rng.dirichlet(np.ones(dim), size=n_probes)


def _endpoint_pool(dim: int, n: int, rng) -> np.ndarray:
    """Candidate line endpoints: half uniform (interior), half boundary-heavy.

    A line's value comes mostly from its *reach*, and a segment between two
    uniform draws is short — uniform Dirichlet mass sits near the centroid.  The
    ``alpha < 1`` half of the pool concentrates on edges and vertices, so the
    proposal set contains chords that actually span the simplex.
    """
    n_edge = int(round(_ENDPOINT_BOUNDARY_FRAC * n))
    return np.vstack([
        rng.dirichlet(np.ones(dim), size=n - n_edge),
        rng.dirichlet(np.full(dim, _ENDPOINT_ALPHA), size=n_edge),
    ])


def _min_sq_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """For each row of `a`, the squared distance to the nearest row of `b`."""
    return ((a ** 2).sum(1)[:, None]
            + (b ** 2).sum(1)[None, :]
            - 2.0 * (a @ b.T)).min(axis=1)


def _uncovered_weights(d2: np.ndarray) -> np.ndarray:
    """Sampling weights ∝ ``d ** _LINE_BIAS`` over points with squared distance `d2`."""
    w = d2 ** (0.5 * _LINE_BIAS)
    tot = w.sum()
    if not np.isfinite(tot) or tot <= 0:
        return np.full(d2.shape[0], 1.0 / d2.shape[0])
    return w / tot


def _leave_one_out(D: np.ndarray, i: int) -> np.ndarray:
    """Coverage the design would have with line `i` removed.

    ``D`` is ``(L, P)``: ``D[l, p]`` is the squared distance from probe ``p`` to
    the nearest point of line ``l``.  Where line ``i`` is not the winner the
    answer is unchanged, and where it is, the runner-up takes over — so the
    two smallest entries of each column are all that is needed.
    """
    if D.shape[0] == 1:
        return np.full(D.shape[1], np.inf)
    two = np.partition(D, 1, axis=0)[:2]        # (2, P): smallest, second
    return np.where(D[i] > two[0], two[0], two[1])


def _jitter_endpoints(ep: np.ndarray, rng, dim: int) -> np.ndarray:
    """Drag each endpoint a short way toward a random simplex point.

    Interpolating toward a point of the simplex keeps the result on the simplex,
    so a jittered endpoint is always a valid composition.
    """
    target = rng.dirichlet(np.ones(dim), size=ep.shape[0])
    w = _JITTER * rng.random((ep.shape[0], 1))
    return (1.0 - w) * ep + w * target


def _score_candidates(probes: np.ndarray, d2: np.ndarray,
                      cand: np.ndarray) -> np.ndarray:
    """Coverage cost of each candidate line, given the current squared distances.

    ``cand`` is ``(C, k, dim)``.  Returns ``(C,)`` mean squared probe-to-design
    distance if that candidate were added.  Candidates are scored in blocks so
    the probe x point distance matrix stays bounded.
    """
    C, k, dim = cand.shape
    P = probes.shape[0]
    block = max(1, _SCORE_BLOCK // (P * k))
    out = np.empty(C)
    for s in range(0, C, block):
        chunk = cand[s:s + block]                       # (c, k, dim)
        pts = chunk.reshape(-1, dim)                    # (c*k, dim)
        # ||p - x||^2 expanded so this is one GEMM instead of a (P, c*k, dim) tensor.
        d2_new = ((probes ** 2).sum(1)[:, None]
                  + (pts ** 2).sum(1)[None, :]
                  - 2.0 * (probes @ pts.T))             # (P, c*k)
        d2_new = d2_new.reshape(P, chunk.shape[0], k).min(axis=2)   # (P, c)
        np.minimum(d2_new, d2[:, None], out=d2_new)
        # d2_new holds *squared* distances, so d^p is d2^(p/2); the p=4 case is
        # a plain square, which is much cheaper than a general power.
        cost = np.square(d2_new) if _SCORE_POWER == 4.0 else (
            d2_new if _SCORE_POWER == 2.0 else d2_new ** (0.5 * _SCORE_POWER))
        out[s:s + chunk.shape[0]] = cost.mean(axis=0)
    return out


def greedy_lines(n: int, dim: int, seed: int | None = None, *,
                 points_per_line: int = POINTS_PER_LINE,
                 n_candidates: int = _LINE_CANDIDATES,
                 n_probes: int | None = None,
                 refine_sweeps: int = _REFINE_SWEEPS,
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Lay `n` lines of `points_per_line` evenly spaced points to cover the simplex.

    A greedy pass places the lines one at a time, each minimising the mean
    squared distance from a uniform probe cloud to the design; `refine_sweeps`
    rounds of whole-line coordinate descent then re-choose each line against the
    coverage the others already provide.  See the section comment above for the
    criterion, the candidate proposals and the leave-one-out trick.

    Returns ``(lines, X)`` where ``lines`` is ``(n, 2, dim)`` of segment
    endpoints and ``X`` is ``(n * points_per_line, dim)`` of the measured
    compositions, ordered line by line.
    """
    if dim < 2:
        raise ValueError("dim must be >= 2")
    if n < 1:
        raise ValueError("n (number of lines) must be >= 1")
    k = int(points_per_line)
    if n_probes is None:
        n_probes = min(_PROBE_MAX,
                       max(_PROBE_MIN, _PROBE_MULTIPLIER * n * k))

    rng = np.random.default_rng(seed)
    probes = _probe_cloud(dim, int(n_probes), rng)
    ends = _endpoint_pool(dim, _ENDPOINT_POOL, rng)
    P, E = probes.shape[0], ends.shape[0]
    e2 = np.full(E, np.inf)      # endpoint -> nearest design point, squared
    D = np.empty((n, P))         # D[l] = probe -> nearest point of line l, squared

    def _propose(base: np.ndarray, current: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Candidate endpoint pairs against a background coverage `base`.

        `current` (the line being re-chosen, or None during the greedy pass) is
        always included, along with local jitters of it, so refinement can make
        small adjustments and can never pick something worse than what it had.
        """
        w = _uncovered_weights(e2)
        A = ends[rng.choice(E, size=n_candidates, p=w)]
        B = ends[rng.choice(E, size=n_candidates, p=w)]
        # The emptiest probe is a free endpoint in its own right, not just a
        # direction to aim at: anchor a share of the proposals exactly on it.
        if np.isfinite(base).any():
            A[:n_candidates // 2] = probes[int(np.argmax(base))]
        if current is not None:
            n_loc = max(1, n_candidates // 4)
            A[-n_loc:] = _jitter_endpoints(np.repeat(current[0:1], n_loc, 0), rng, dim)
            B[-n_loc:] = _jitter_endpoints(np.repeat(current[1:2], n_loc, 0), rng, dim)
            A[-1], B[-1] = current[0], current[1]      # keep-what-we-have
        return A, B

    def _choose(base: np.ndarray, current: np.ndarray | None
                ) -> tuple[np.ndarray, np.ndarray]:
        """Best candidate line given background coverage `base`; returns (endpoints, pts)."""
        A, B = _propose(base, current)
        cand = np.stack([line_points(a, b, k) for a, b in zip(A, B)])   # (C,k,dim)
        best = int(np.argmin(_score_candidates(probes, base, cand)))
        return np.stack([A[best], B[best]]), cand[best]

    lines = np.empty((n, 2, dim))

    # ── greedy pass ─────────────────────────────────────────────────────────
    # Nothing is placed for line 0, so its "background" is infinite distance and
    # the cost reduces to mean distance to this one line — already a sensible
    # objective (a long, well-centred segment).
    d2 = np.full(P, np.inf)
    for i in range(n):
        lines[i], pts = _choose(d2, None)
        D[i] = _min_sq_dist(probes, pts)
        np.minimum(d2, D[i], out=d2)
        np.minimum(e2, _min_sq_dist(ends, pts), out=e2)

    # ── refinement sweeps ───────────────────────────────────────────────────
    for _ in range(int(refine_sweeps)):
        for i in range(n):
            base = _leave_one_out(D, i)
            lines[i], pts = _choose(base, lines[i])
            D[i] = _min_sq_dist(probes, pts)
        d2 = D.min(axis=0)
        e2 = np.minimum.reduce([_min_sq_dist(ends, line_points(p, q, k))
                                for p, q in lines])

    X = np.concatenate([line_points(p, q, k) for p, q in lines])
    # Renormalise away the float drift the interpolation introduces.
    return lines, X / X.sum(1, keepdims=True)


def coverage_stats(x: np.ndarray, seed: int = 0,
                   n_probes: int = 20000) -> dict:
    """How well design `x` covers the simplex: probe-to-nearest-design distances.

    This is the metric that matters for a *line* design.  Nearest-neighbour
    spacing is meaningless here — points inside one line are only ``|p-q|/23``
    apart by construction — so coverage is measured from the space's point of
    view: draw uniform probes and ask how far each is from the nearest measured
    composition.  ``max`` is the coverage radius (the worst-covered spot);
    ``p95``/``mean`` describe the typical gap.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    probes = rng.dirichlet(np.ones(x.shape[1]), size=n_probes)
    d = cKDTree(x).query(probes, k=1)[0]
    return {
        "mean": float(d.mean()),
        "p95": float(np.percentile(d, 95)),
        "max": float(d.max()),
    }


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(seed: int = 0, n_repeats: int = 5) -> None:
    print(f"{'dim':>4} {'budget':>7} {'pts':>6} "
          f"{'mean NN':>9} {'min NN':>8} "
          f"{'NN/0.064':>9} {'NN/0.045':>9}")
    print("-" * 60)
    for dim in sorted(BUDGETS):
        n = n_warm_start(dim)
        # The 10d selection (n=3000) is the expensive one; average fewer repeats.
        reps = 1 if dim >= 10 else n_repeats
        means, mins = [], []
        for r in range(reps):
            x = maximin_simplex(n, dim, seed=seed + r)
            m, _, mn = mean_nn_distance(x)
            means.append(m)
            mins.append(mn)
        mean_nn = float(np.mean(means))
        min_nn = float(np.mean(mins))
        print(f"{dim:>4d} {BUDGETS[dim]:>7d} {n:>6d} "
              f"{mean_nn:>9.4f} {min_nn:>8.4f} "
              f"{mean_nn / INPUT_NOISE:>9.2f} {mean_nn / OUTPUT_NOISE_FRAC:>9.2f}")
    print("-" * 60)
    print(f"input (spatial) noise radius NOISE_LEVEL       = {INPUT_NOISE}")
    print(f"output (y) noise fraction    OUTPUT_NOISE_FRAC = {OUTPUT_NOISE_FRAC}")

    # How many maximin points would be needed to drive the mean NN spacing down
    # to the noise radius?  On a (d-1)-dim manifold the NN spacing scales as
    #   s(N) ~ N**(-1/(d-1)),  so  N_target = N * (s_now / s_target)**(d-1).
    print("\nPoints needed for mean NN spacing == a target radius "
          "(scaling s ~ N**(-1/(d-1))):")
    print(f"{'dim':>4} {'N now':>8} {'s now':>8} "
          f"{'N->0.064':>12} {'N->0.045':>12}")
    print("-" * 50)
    for dim in sorted(BUDGETS):
        n = n_warm_start(dim)
        x = maximin_simplex(n, dim, seed=seed)
        s_now, _, _ = mean_nn_distance(x)
        p = dim - 1
        n_in = n * (s_now / INPUT_NOISE) ** p
        n_out = n * (s_now / OUTPUT_NOISE_FRAC) ** p
        print(f"{dim:>4d} {n:>8d} {s_now:>8.4f} {n_in:>12,.0f} {n_out:>12,.0f}")


def analyze_lines(seed: int = 0) -> None:
    """Coverage of the line-constrained warm start at each benchmark dimension."""
    print("=" * 62)
    print("Line-constrained warm start (24 evenly spaced points per line)")
    print("=" * 62)
    print(f"{'dim':>4} {'lines':>6} {'pts':>6} "
          f"{'mean':>8} {'p95':>8} {'max':>8}")
    print("-" * 62)
    for dim in sorted(BUDGETS):
        L = n_lines(dim)
        _, x = greedy_lines(L, dim, seed=seed)
        c = coverage_stats(x, seed=seed + 1)
        print(f"{dim:>4d} {L:>6d} {L * POINTS_PER_LINE:>6d} "
              f"{c['mean']:>8.4f} {c['p95']:>8.4f} {c['max']:>8.4f}")
    print("-" * 62)
    print("coverage = distance from a uniform simplex probe to the nearest")
    print(f"measured composition; 'max' is the coverage radius.  Compare")
    print(f"against the input noise radius {INPUT_NOISE}.")


if __name__ == "__main__":
    analyze_lines()
