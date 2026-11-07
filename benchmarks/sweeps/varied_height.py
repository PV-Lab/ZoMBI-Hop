"""
benchmarks/sweeps/varied_height.py
==================================
The ``varied_height`` landscape: the needles sweep (``needles.py``) with optima of
**different heights**, on an objective that runs from 0 to 1.

Everything else is the needles landscape. Its objective,
``max(0.5 + 0.5*E, 0.75)``, is the affine image of ``max(2E - 1, 0)`` (plain 0,
peak 1), so this one keeps exactly that shape and gives each optimum ``c`` its own
height ``h_c``:

    y(x) = max_c  h_c * max( 2*E_c(x) - 1, 0 ),   E_c(x) = exp(-b * ||x - c|| / sqrt(d))

* the plain is **0**, flat, exactly where it was (a basin still meets it at radius
  ``sqrt(d) * ln 2 / b``, ``needles.basin_plain_radius``);
* optimum ``c`` peaks at exactly ``h_c``; heights are drawn ``U(0.5, 1.0)`` and the
  tallest is set to 1.0, so every landscape's global maximum is 1.0 and the
  objective spans [0, 1];
* the same grid (dim x n x b), budgets, methods and noise model.

What the noise model does here
------------------------------
Output noise is multiplicative (``y * (1 + N(0, 0.045^2))``, ``Problem``), as in
the needles sweep. On this objective that means **the plain reads exactly 0**, and
the noise at a peak is 4.5% of that peak's height. On the needles landscape the
plain read 0.75 +/- 0.034 against a 0.25 rise, so relative to the bump the noise
there was four times larger. This sweep is the cleaner measurement problem; that is
a property of the spec, not an accident, and the summary says so.

Resolvability with unequal heights
----------------------------------
A short optimum next to a tall one can vanish into the tall one's slope. Along the
segment between optima ``i`` (taller) and ``j``, the objective is the max of
``h_i*g(t)`` and ``h_j*g(s - t)`` with ``g(t) = max(2*exp(-b*t/sqrt(d)) - 1, 0)``.
Optimum ``j`` is resolvable when the objective dips at least one output-noise sd
(``sigma_y * h_j``) below its peak somewhere on that segment, i.e. both terms fall
under ``L = (1 - sigma_y) * h_j`` at once. That holds exactly when

    s >= rho(1 - sigma_y) + rho((1 - sigma_y) * h_j / h_i),
    rho(v) = -ln((1 + v) / 2) * sqrt(d) / b         (g(rho(v)) = v)

so the required separation is **per pair** and grows with the height ratio. As in
``needles.py`` the input-noise floor ``sigma_x`` applies too, and the pairwise rule
is necessary but not sufficient, so :func:`prominence_report` measures every built
landscape.
"""

from __future__ import annotations

import math

import numpy as np

from . import needles as nd

KIND = "varied_height"
PLAIN_Y = 0.0
PEAK_Y = 1.0                      # the tallest optimum, exactly
HEIGHT_RANGE = (0.5, 1.0)         # every optimum's height lies in here


# ─── The objective ───────────────────────────────────────────────────────────────

class VariedHeightNeedles:
    """``y(x) = max_c h_c * max(2*exp(-b*||x - c||/sqrt(d)) - 1, 0)`` on ``[0, 1]^d``.

    Exposes what the sweep reads off an ``Ensemble``: ``predict``, ``centers`` and
    ``known_maxima``.
    """

    def __init__(self, dim: int, centers, heights, basin_width: float) -> None:
        self.dim = int(dim)
        self.centers = np.asarray(centers, dtype=float).reshape(-1, self.dim)
        self.heights = np.asarray(heights, dtype=float).ravel()
        if len(self.heights) != len(self.centers):
            raise ValueError(f"{len(self.centers)} centers but {len(self.heights)} heights")
        self.basin_width = float(basin_width)
        self.known_maxima = self.centers.copy()

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(-1, self.dim)
        out = np.zeros(len(X))
        scale = self.basin_width / math.sqrt(self.dim)
        for lo in range(0, len(X), 4096):       # bounds the (chunk, n) distance block
            D = np.linalg.norm(X[lo:lo + 4096, None, :] - self.centers[None, :, :], axis=2)
            g = np.maximum(2.0 * np.exp(-scale * D) - 1.0, 0.0)
            out[lo:lo + 4096] = (g * self.heights[None, :]).max(axis=1)
        return out

    def __call__(self, X) -> np.ndarray:
        return self.predict(X)


def landscape_config(dim: int, basin_width: float, centers: np.ndarray,
                     heights: np.ndarray, seed: int) -> dict:
    """JSON-serialisable, and what :func:`from_config` rebuilds the objective from.
    ``pinned_optima`` keeps the key the needles cells use, so readers of
    ``ensemble_config.json`` find the optima in the same place."""
    return {
        "kind": KIND, "dim": int(dim), "domain": nd.DOMAIN,
        "pinned_optima": np.asarray(centers, dtype=float).tolist(),
        "heights": np.asarray(heights, dtype=float).tolist(),
        "basin_width": float(basin_width),
        "plain_y": PLAIN_Y,
        "seed": int(seed),
    }


def from_config(cfg: dict) -> VariedHeightNeedles:
    return VariedHeightNeedles(cfg["dim"], cfg["pinned_optima"], cfg["heights"],
                               cfg["basin_width"])


# ─── Resolvability ───────────────────────────────────────────────────────────────

def _rho(v: float, basin_width: float, dim: int) -> float:
    """Distance at which ``g`` has fallen to ``v`` (``0 < v <= 1``)."""
    return -math.log((1.0 + float(v)) / 2.0) * math.sqrt(float(dim)) / float(basin_width)


def found_radius(basin_width: float, dim: int) -> float:
    """Radius within which a sample counts as having found an optimum: where the
    objective is one output-noise sd below that optimum's peak. The noise is
    multiplicative, so this is the same for every height:
    ``h*g(r) = (1 - sigma_y)*h``."""
    return _rho(1.0 - nd.SIGMA_Y_FRAC, basin_width, dim)


def pair_separation(h_a, h_b, basin_width: float, dim: int):
    """Minimum separation for two optima of heights ``h_a``, ``h_b`` (see the module
    docstring); vectorised over ``h_b``. Includes the input-noise floor."""
    h_a = np.asarray(h_a, dtype=float)
    h_b = np.asarray(h_b, dtype=float)
    ratio = np.minimum(h_a, h_b) / np.maximum(h_a, h_b)
    scale = math.sqrt(float(dim)) / float(basin_width)
    v = 1.0 - nd.SIGMA_Y_FRAC
    s_prom = (-math.log((1.0 + v) / 2.0) - np.log((1.0 + v * ratio) / 2.0)) * scale
    return np.maximum(s_prom, float(nd.SIGMA_X))


def worst_case_separation(basin_width: float, dim: int) -> float:
    """The separation the most unequal pair (heights 0.5 and 1.0) needs."""
    lo, hi = HEIGHT_RANGE
    return float(pair_separation(lo, hi, basin_width, dim))


def plan_feasibility(dims=nd.GRID_DIM, needles=nd.GRID_N_NEEDLES,
                     widths=nd.GRID_BASIN_WIDTH) -> list[dict]:
    """``needles.plan_feasibility`` at the worst-case pair separation. Optimistic in
    the other direction too: most pairs need less, so a configuration over this
    bound may still place (``build_landscape`` is the real test)."""
    rows = []
    for dim in dims:
        for n in needles:
            for b in widths:
                s = worst_case_separation(b, dim)
                cap = nd.cube_capacity(dim, s)
                rows.append({
                    "dim": int(dim), "n_needles": int(n), "basin_width": float(b),
                    "separation_target": round(s, 6),
                    "separation_equal_heights": round(float(pair_separation(1, 1, b, dim)), 6),
                    "basin_plain_radius": round(nd.basin_plain_radius(b, dim), 6),
                    "found_radius": round(found_radius(b, dim), 6),
                    "capacity_estimate": round(min(cap, 1e12), 1),
                    "feasible": bool(n <= cap),
                })
    return rows


# ─── Placement ───────────────────────────────────────────────────────────────────

def draw_heights(n: int, rng) -> np.ndarray:
    lo, hi = HEIGHT_RANGE
    h = rng.uniform(lo, hi, size=int(n))
    h[int(np.argmax(h))] = PEAK_Y
    return h


def _need_matrix(heights: np.ndarray, basin_width: float, dim: int) -> np.ndarray:
    """Required separation for every pair, with the placement margin."""
    need = np.vstack([pair_separation(h, heights, basin_width, dim) for h in heights])
    return need * nd.SEPARATION_MARGIN


def _dart_throw(dim: int, heights: np.ndarray, basin_width: float, rng,
                pool: int = 200_000, chunk: int = 4096) -> np.ndarray:
    """Place optima one at a time, uniformly, each clearing its pair separation from
    every one already placed. Returns as many as found room in ``pool`` draws each
    (in ``heights`` order, so a short result is a prefix)."""
    kept = np.zeros((0, dim))
    for k, h in enumerate(heights):
        need = pair_separation(h, heights[:k], basin_width, dim) * nd.SEPARATION_MARGIN
        tried, spot = 0, None
        while tried < pool and spot is None:
            cand = rng.random((chunk, dim))
            tried += chunk
            if not len(kept):
                spot = cand[0]
                break
            D = np.linalg.norm(cand[:, None, :] - kept[None, :, :], axis=2)
            ok = np.flatnonzero((D >= need[None, :]).all(axis=1))
            if len(ok):
                spot = cand[ok[0]]
        if spot is None:
            break
        kept = np.vstack([kept, spot])
    return kept


def _relax(X: np.ndarray, need: np.ndarray, rng, iters: int = 4000) -> np.ndarray:
    """``needles._relax`` with a per-pair separation: push violating pairs apart,
    clipping to the cube, until every pair clears ``need`` (or ``iters`` run out)."""
    X = np.clip(X.copy(), 0.0, 1.0)
    for _ in range(iters):
        V = X[:, None, :] - X[None, :, :]
        D = np.linalg.norm(V, axis=2)
        np.fill_diagonal(D, np.inf)
        short = np.clip(need - D, 0.0, None)
        np.fill_diagonal(short, 0.0)
        if not short.any():
            return X
        tiny = D < 1e-12
        if tiny.any():
            V[tiny] = rng.normal(size=(int(tiny.sum()), X.shape[1]))
            D = np.where(tiny, np.linalg.norm(V, axis=2), D)
        step = (0.5 * short / D)[:, :, None] * V
        X = np.clip(X + step.sum(axis=1), 0.0, 1.0)
    return X


def place_optima(dim: int, n: int, basin_width: float, seed: int,
                 attempts: int = 20) -> dict:
    """Heights first, then positions that keep every optimum resolvable.

    Uniform dart-throwing first (an honest uniform sample conditioned on the
    separations); when it saturates short of ``n``, the remainder is placed at
    random and the set relaxed apart, as ``needles.place_optima`` does. Raises if
    ``attempts`` of those all fail: unlike ``needles.py`` there is no smaller
    fall-back separation that keeps the count honest, because a short optimum too
    close to a tall one is not an optimum at all."""
    dim, n = int(dim), int(n)
    rng = np.random.default_rng(int(seed))
    heights = draw_heights(n, rng)
    need = _need_matrix(heights, basin_width, dim)
    relaxed = False
    for attempt in range(1, attempts + 1):
        X = _dart_throw(dim, heights, basin_width, rng)
        if len(X) < n:
            X = _relax(np.vstack([X, rng.random((n - len(X), dim))]), need, rng)
            relaxed = True
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        if (D >= need * 0.999).all():
            break
    else:
        raise RuntimeError(
            f"could not place {n} resolvable optima of heights in {HEIGHT_RANGE} in the "
            f"{dim}-cube at b = {basin_width:g} in {attempts} attempts")
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    return {
        "centers": X, "heights": heights,
        "separation_achieved": float(D.min()),
        "placement_attempts": attempt,
        "placement_relaxed": relaxed,
        "prominence_target_met": True,
        "placement_seed": int(seed),
    }


def prominence_report(fn: VariedHeightNeedles, n_probe: int = 65) -> dict:
    """Measure, don't assume: for every optimum, the smallest dip toward any optimum
    at least as tall (or toward its nearest neighbour, for the tallest), against
    one output-noise sd of its own height."""
    C, H = fn.centers, fn.heights
    if len(C) < 2:
        return {"n_prominence_resolved": int(len(C)), "min_prominence_ratio": None}
    t = np.linspace(0.0, 1.0, int(n_probe)).reshape(-1, 1)
    D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    ratios = []
    for j in range(len(C)):
        others = [i for i in range(len(C)) if i != j and H[i] >= H[j]] \
            or [int(np.argmin(D[j]))]
        dip = min(H[j] - float(fn.predict(C[j] + t * (C[i] - C[j])).min()) for i in others)
        ratios.append(dip / (nd.SIGMA_Y_FRAC * H[j]))
    ratios = np.asarray(ratios)
    return {"n_prominence_resolved": int((ratios >= 1.0 - 1e-9).sum()),
            "min_prominence_ratio": round(float(ratios.min()), 4)}


def build_landscape(dim: int, n: int, basin_width: float, seed: int) -> dict:
    """Place, build and VERIFY one landscape; same return shape as
    ``needles.build_landscape``."""
    placed = place_optima(dim, n, basin_width, seed)
    cfg = landscape_config(dim, basin_width, placed["centers"], placed["heights"], seed)
    fn = from_config(cfg)
    peaks = fn.predict(fn.centers)
    if not np.allclose(peaks, fn.heights, atol=1e-9):
        raise AssertionError("an optimum does not peak at its own height: a taller "
                             "optimum's slope covers it")
    record = {
        "domain": nd.DOMAIN, "kind": KIND,
        "dim": int(dim), "n_needles": int(n), "basin_width": float(basin_width),
        "basin_plain_radius": round(nd.basin_plain_radius(basin_width, dim), 6),
        "found_radius": round(found_radius(basin_width, dim), 6),
        "heights": [round(float(h), 6) for h in placed["heights"]],
        **{k: v for k, v in placed.items() if k not in ("centers", "heights")},
        **prominence_report(fn),
    }
    return {"config": cfg, "fn": fn, "record": record, "centers": fn.centers.copy()}


# ─── Self-test ───────────────────────────────────────────────────────────────────

def selftest(verbose: bool = True) -> None:
    """Closed form, range, heights, and the pair-separation identity, numerically."""
    rng = np.random.default_rng(0)
    for dim in nd.GRID_DIM:
        built = build_landscape(dim, 10, 6.0, seed=dim)
        fn = built["fn"]
        probe = rng.random((20_000, dim))
        y = fn.predict(probe)
        assert y.min() >= 0.0 and y.max() <= 1.0 + 1e-12, f"dim {dim}: y outside [0, 1]"
        assert abs(fn.predict(fn.centers).max() - 1.0) < 1e-12, f"dim {dim}: max != 1"
        h = fn.heights
        assert h.min() >= HEIGHT_RANGE[0] and h.max() == PEAK_Y
        far = fn.predict(np.full((1, dim), 0.5) + 10.0)   # outside every basin
        assert far[0] == 0.0, f"dim {dim}: plain is not 0"
        if verbose:
            r = built["record"]
            print(f"  [selftest:{KIND}] dim {dim}: range, heights, plain OK "
                  f"({r['n_prominence_resolved']}/10 prominence-resolved)")
    # At exactly the pair separation (above the sigma_x floor), the shorter peak dips
    # by exactly sigma_y * h_short.
    for dim, b, h_lo in ((2, 2.2, 0.5), (9, 2.2, 0.7), (3, 2.2, 0.6)):
        s = float(pair_separation(1.0, h_lo, b, dim))
        assert s > nd.SIGMA_X, "pick a case above the input-noise floor"
        c0 = np.full(dim, 0.5)
        c1 = c0.copy()
        c1[0] += s
        fn = VariedHeightNeedles(dim, np.vstack([c0, c1]), [1.0, h_lo], b)
        t = np.linspace(0, 1, 20001).reshape(-1, 1)
        dip = h_lo - float(fn.predict(c1 + t * (c0 - c1)).min())
        assert abs(dip - nd.SIGMA_Y_FRAC * h_lo) < 1e-4, \
            f"dim {dim} b {b}: dip {dip}, want {nd.SIGMA_Y_FRAC * h_lo}"
    if verbose:
        print(f"  [selftest:{KIND}] pair separation reproduces sigma_y * h exactly")
    print(f"  [selftest:{KIND}] OK")
