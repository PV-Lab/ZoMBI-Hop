"""
benchmarks/sweeps/needles.py
============================
The landscape this sweep varies: **Ackley bumps on the unit cube, and nothing else**.

``synthetic_data.ensemble.CartesianEnsemble`` is the layered ``Ensemble`` objective
on ``[0, 1]^dim`` instead of the simplex (``Ensemble(domain="cube", ...)``). This
sweep turns off every feature family but the true optima — no roughness, ridges,
plateaus, distractors, anisotropy or edge bias — so the surface is exactly ``n``
negated-Ackley basins of sharpness ``b`` on a flat plain. With the background field
``G(x) == 0`` the objective has a closed form:

    y(x) = max( 0.5 + 0.5 * E(x),  0.75 ),   E(x) = max_c exp(-b * ||x - c|| / sqrt(d))

so the plain sits at **0.75**, every optimum peaks at exactly **1.0**, and a basin
meets the plain at radius ``sqrt(d) * ln 2 / b``. :func:`selftest` checks all of
that numerically against constructed objects. (Nothing in the closed form depends
on the domain; only where the optima may be placed does.)

Why the sweep is Cartesian
--------------------------
The sweep compares ZoMBI-Hop with general-purpose black-box optimisers — random
search, GP-BO, TuRBO, HEBO — which are defined on boxes. Running them on the
simplex would mean reparameterising each one, and the comparison would then partly
measure the reparameterisation. On the cube every method works in its native
domain, and ZoMBI-Hop runs through ``BoxDomain`` (``src/utils/domain.py``).

Why the optima are placed instead of drawn
------------------------------------------
The sweep wants a clean "number of needles" axis, and stock uniform placement does
not give one: ``Ensemble`` pares its optima, so a basin landing within
``input_noise`` of an already-tagged one is drawn on the surface but NOT
advertised in ``centers`` / ``known_maxima``. Uniform draws collide, so a request
for ``n`` would advertise fewer, and a different shortfall per dimension. So the
centers are placed here, subject to a separation that makes all ``n`` of them
resolvable, and handed to ``Ensemble(pinned_optima=...)`` with ``n_optima=0``;
``known_maxima`` is then exactly the placed set.

What "resolvable" means
-----------------------
``METHODS.md`` section 1: the target set is the local maximisers "whose basins are
resolvable above the noise — wider than sigma_x and more prominent than sigma_y".
Both are enforced as a minimum pairwise separation ``s*``:

1. **Wider than the input noise** — ``s >= sigma_x = 0.128`` (``run_mobo.NOISE_LEVEL``,
   the measured deposition noise; kept as the needle-separation floor so the
   landscapes stay comparable to the simplex campaigns). It is also the exact test
   ``Ensemble._tag_true_optima`` applies, so meeting it makes the paring a no-op and
   ``len(fn.centers) == n`` by construction (asserted in :func:`build_landscape`).

2. **More prominent than the output noise** — the saddle between two adjacent peaks
   must dip more than ``sigma_y`` below them. Two equal peaks at separation ``s``
   put their saddle at the midpoint, where ``E = exp(-b*s/(2*sqrt(d)))``, so with
   ``sigma_y = 0.045 * 1.0`` at a peak (multiplicative noise):

       s >= s_prom(b, d) = -2 * ln(1 - 2*sigma_y) * sqrt(d) / b  ~=  0.1886 * sqrt(d) / b

The target is ``s* = max(sigma_x, s_prom(b, d))``; condition 2 binds only at the
broadest sharpness in the default grid (``b = 2.2``). The pairwise rule is necessary
but not sufficient (a third peak can lift a pair's saddle), so the prominence is
also **measured** on every built landscape (:func:`prominence_report`) and recorded.

The unit cube has far more room than the simplex (a 3-cube holds ~900 points at
0.128 where the 3-simplex held ~60), so every cell of the default grid fits at its
prominence target and the dim-3 / n=50 lattice-packing corner of the simplex sweep
does not arise. :func:`plan_feasibility` still checks, for custom grids.
"""

from __future__ import annotations

import math

import numpy as np

from ._paths import ensure_paths

ensure_paths()

from eval_metrics import NOISE_LEVEL as SIGMA_X  # noqa: E402  input noise, 0.128

from benchmarks.methods.base import (  # noqa: E402
    DEFAULT_OUTPUT_NOISE_FRAC as SIGMA_Y_FRAC,  # metrology noise, 0.045
)

DOMAIN = "cube"

# Output-map constants of the bumps-only landscape (see the module docstring).
PLAIN_Y = 0.75   # the flat background
PEAK_Y = 1.0     # every optimum, exactly

# Placed separations are set this much above ``s*``. ``_tag_true_optima`` keeps a
# basin when it is ``>=`` min_dist from every kept one, and a relaxation that
# converges *to contact* lands exactly on the boundary, where float rounding decides
# whether an optimum is advertised. 2% of clearance removes the coin flip.
SEPARATION_MARGIN = 1.02


# ─── The sweep grid ──────────────────────────────────────────────────────────────

#: The three landscape axes. Full-factorial: 4 x 4 x 4 = 64 configurations.
GRID_N_NEEDLES: tuple[int, ...] = (2, 10, 30, 50)
GRID_BASIN_WIDTH: tuple[float, ...] = (2.2, 6.0, 10.0, 15.0)
#: Cube dims = the simplex grid (3, 4, 6, 10) in free dimensions; see hparams.py.
GRID_DIM: tuple[int, ...] = (2, 3, 5, 9)


# ─── Resolvability ───────────────────────────────────────────────────────────────

def sigma_y_at_peak() -> float:
    """Output-noise sd at an optimum: the noise is multiplicative and peaks are 1.0."""
    return SIGMA_Y_FRAC * PEAK_Y


def prominence_separation(basin_width: float, dim: int) -> float:
    """Smallest separation at which two adjacent peaks stay distinguishable.

    Solves ``1.0 - (0.5 + 0.5*exp(-b*s/(2*sqrt(d)))) >= sigma_y`` for ``s``.
    """
    b = float(basin_width)
    if b <= 0:
        raise ValueError(f"basin_width must be positive, got {basin_width}")
    e_max = 1.0 - 2.0 * sigma_y_at_peak()   # saddle envelope value at the threshold
    return -2.0 * math.log(e_max) * math.sqrt(float(dim)) / b


def found_radius(basin_width: float, dim: int) -> float:
    """Radius within which a sample counts as having FOUND an optimum.

    The distance at which the objective has fallen one output-noise sd below the
    peak: solves ``1.0 - (0.5 + 0.5*exp(-b*r/sqrt(d))) = sigma_y`` for ``r``.
    Inside it a measurement cannot be told apart from one at the optimum, so it is
    the closest a sample-based search can be asked to localise a needle. It is a
    property of the landscape alone (no method, budget or result enters), and it
    is exactly half of :func:`prominence_separation`, so optima placed at their
    prominence target never have overlapping found-balls and one sample can find
    at most one optimum.
    """
    return 0.5 * prominence_separation(basin_width, dim)


def target_separation(basin_width: float, dim: int) -> float:
    """``s* = max(sigma_x, s_prom(b, d))`` — both of METHODS' resolvability tests."""
    return max(float(SIGMA_X), prominence_separation(basin_width, dim))


def basin_plain_radius(basin_width: float, dim: int) -> float:
    """Distance from an optimum at which its basin meets the plain (``E = 0.5``).

    Diagnostic: against the separation it says whether a cell's basins overlap at
    all — the difference between "n needles" and "one ridged mesa with n tips".
    """
    return math.sqrt(float(dim)) * math.log(2.0) / float(basin_width)


def cube_capacity(dim: int, separation: float) -> float:
    """Roughly how many points at ``separation`` fit in the unit ``dim``-cube.

    Volume 1 over the volume of a ``dim``-ball of radius ``s/2``, with packing
    density taken as 1 and boundary effects ignored. Optimistic, so a cell this
    rejects certainly does not fit; used only to warn at plan time.
    """
    d = int(dim)
    r = float(separation) / 2.0
    ball = (math.pi ** (d / 2.0)) * (r ** d) / math.gamma(d / 2.0 + 1.0)
    return 1.0 / ball if ball > 0 else float("inf")


def plan_feasibility(dims=GRID_DIM, needles=GRID_N_NEEDLES,
                     widths=GRID_BASIN_WIDTH) -> list[dict]:
    """One row per landscape configuration: the separation it needs and whether it fits."""
    rows = []
    for dim in dims:
        for n in needles:
            for b in widths:
                s = target_separation(b, dim)
                cap = cube_capacity(dim, s)
                rows.append({
                    "dim": int(dim), "n_needles": int(n), "basin_width": float(b),
                    "separation_target": round(s, 6),
                    "prominence_binds": bool(prominence_separation(b, dim) > SIGMA_X),
                    "basin_plain_radius": round(basin_plain_radius(b, dim), 6),
                    "capacity_estimate": round(min(cap, 1e12), 1),
                    "feasible": bool(n <= cap),
                })
    return rows


# ─── Placement ───────────────────────────────────────────────────────────────────

def _min_pairwise(X: np.ndarray) -> float:
    if len(X) < 2:
        return float("inf")
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    return float(D.min())


def _dart_throw(dim: int, n: int, sep: float, rng, pool: int = 200_000) -> np.ndarray:
    """Uniform rejection sampling: keep a draw only if it clears every kept one.

    Tried first because it is the least structured way to meet the separation —
    the points stay an honest uniform sample conditioned on the constraint.
    """
    kept: list[np.ndarray] = []
    for p in rng.random((pool, dim)):
        if len(kept) >= n:
            break
        if not kept or np.min(np.linalg.norm(np.asarray(kept) - p, axis=1)) >= sep:
            kept.append(p)
    return np.asarray(kept, dtype=float).reshape(-1, dim)


def _relax(X: np.ndarray, sep: float, rng, iters: int = 4000) -> np.ndarray:
    """Push overlapping optima apart until every pair clears ``sep``, clipping to
    the cube each step. Only reached when dart-throwing saturates short of ``n``,
    which the default grid never does on the cube."""
    X = np.clip(X.copy(), 0.0, 1.0)
    for _ in range(iters):
        D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        if D.min() >= sep:
            return X
        step = np.zeros_like(X)
        ii, jj = np.where(D < sep)
        for a, b in zip(ii, jj):
            v = X[a] - X[b]
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                v = rng.normal(size=X.shape[1])
                nv = float(np.linalg.norm(v)) or 1.0
            step[a] += 0.5 * (sep - D[a, b]) * v / nv
        X = np.clip(X + step, 0.0, 1.0)
    return X


def place_optima(dim: int, n: int, basin_width: float, seed: int) -> dict:
    """``n`` mutually resolvable optima in the unit ``dim``-cube.

    The hard floor is ``sigma_x`` (the count must come out exactly ``n``); the
    prominence target is preferred and is the one abandoned first if both cannot
    be had. Returns the centers plus the placement record for the cell file.
    """
    dim, n = int(dim), int(n)
    rng = np.random.default_rng(int(seed))
    s_target = target_separation(basin_width, dim) * SEPARATION_MARGIN
    s_floor = float(SIGMA_X) * SEPARATION_MARGIN

    def _attempt(sep: float) -> np.ndarray | None:
        X = _dart_throw(dim, n, sep, rng)
        if len(X) < n:
            extra = rng.random((n - len(X), dim))
            X = np.vstack([X, extra]) if len(X) else extra
            X = _relax(X, sep, rng)
        return X if _min_pairwise(X) >= sep * 0.999 else None

    prominence_met = True
    X = _attempt(s_target)
    if X is None:
        prominence_met = False
        X = _attempt(s_floor)
        if X is None:
            raise RuntimeError(
                f"could not place {n} optima in the {dim}-cube even at the input-noise "
                f"floor {s_floor:.4f}")
    return {
        "centers": X,
        "separation_target": float(s_target / SEPARATION_MARGIN),
        "separation_floor": float(SIGMA_X),
        "separation_achieved": float(_min_pairwise(X)),
        "prominence_target_met": bool(prominence_met),
        "placement_seed": int(seed),
    }


# ─── Landscape construction ──────────────────────────────────────────────────────

def ensemble_config(dim: int, basin_width: float, centers: np.ndarray,
                    seed: int) -> dict:
    """``Ensemble`` kwargs for a bumps-only cube landscape with exactly these optima.

    ``"domain": "cube"`` makes ``Ensemble(**config)`` return a
    :class:`~synthetic_data.ensemble.CartesianEnsemble`. Every background family is
    zero; ``n_optima=0`` with ``pinned_optima`` means nothing is drawn at random.
    JSON-serialisable, so a cell can be rebuilt from its own ``ensemble_config.json``.
    """
    return {
        "dim": int(dim),
        "domain": DOMAIN,
        "n_optima": 0,
        "pinned_optima": np.asarray(centers, dtype=float).tolist(),
        "basin_width": float(basin_width),
        "basin_smoothing": 0.0,
        "n_weak": 0, "weak_amp": 0.0,
        "n_ridges": 0, "ridge_amp": 0.0,
        "noise_amp": 0.0,
        "aniso_strength": 0.0,
        "n_plateaus": 0, "plateau_amp": 0.0,
        "edge_region": None, "edge_amp": 0.0,
        # paring distance = the separation floor placement already guarantees
        "input_noise": float(SIGMA_X),
        "seed": int(seed),
    }


def prominence_report(fn, centers: np.ndarray, n_probe: int = 65) -> dict:
    """Measure, don't assume: the dip from each optimum toward its nearest neighbour.

    Walks the segment to the nearest neighbour, takes the true minimum of the
    objective along it, and calls the optimum resolved when the dip clears one
    output-noise sd.
    """
    C = np.asarray(centers, dtype=float)
    thresh = sigma_y_at_peak()
    if len(C) < 2:
        return {"n_prominence_resolved": int(len(C)), "min_prominence": None,
                "median_prominence": None, "prominence_threshold": round(thresh, 6)}
    D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = np.argmin(D, axis=1)
    t = np.linspace(0.0, 1.0, int(n_probe)).reshape(-1, 1)
    proms = np.asarray([
        float((y := np.asarray(fn.predict(C[i] + t * (C[j] - C[i])), dtype=float))[0]
              - y.min())
        for i, j in enumerate(nn)])
    return {
        "n_prominence_resolved": int((proms >= thresh).sum()),
        "min_prominence": round(float(proms.min()), 6),
        "median_prominence": round(float(np.median(proms)), 6),
        "prominence_threshold": round(float(thresh), 6),
    }


def placement_seed(seed_base: int, dim: int, n_needles: int, basin_width: float,
                   draw: int) -> int:
    """Deterministic in ``(configuration, draw)`` and independent of the METHOD, so
    every method in a campaign is handed the identical landscape for a given cell,
    and distinct across configurations so neighbouring cells are not correlated."""
    h = (int(seed_base) * 1_000_003
         ^ int(dim) * 2_654_435_761
         ^ int(n_needles) * 40_503
         ^ int(round(float(basin_width) * 10)) * 97_499
         ^ int(draw) * 15_485_863)
    return int(abs(h) % 1_000_000)


def build_landscape(dim: int, n: int, basin_width: float, seed: int) -> dict:
    """Place, build and VERIFY one landscape.

    Returns ``{"config", "fn", "centers", "record"}``. Raises if the built
    objective does not advertise exactly ``n`` optima, or is not on the cube.
    """
    from synthetic_data.ensemble import CartesianEnsemble, Ensemble

    placed = place_optima(dim, n, basin_width, seed)
    cfg = ensemble_config(dim, basin_width, placed["centers"], seed)
    fn = Ensemble(**cfg)
    if not isinstance(fn, CartesianEnsemble):
        raise AssertionError(f"expected a CartesianEnsemble, built {type(fn).__name__}")
    n_true = len(fn.centers)
    if n_true != n:
        raise AssertionError(
            f"placed {n} optima at separation {placed['separation_achieved']:.4f} "
            f"but Ensemble advertises {n_true}; the paring distance ({SIGMA_X}) and "
            "the placement floor have drifted apart")
    record = {
        "domain": DOMAIN,
        "dim": int(dim), "n_needles": int(n), "basin_width": float(basin_width),
        "basin_plain_radius": round(basin_plain_radius(basin_width, dim), 6),
        **{k: v for k, v in placed.items() if k != "centers"},
        **prominence_report(fn, placed["centers"]),
    }
    return {"config": cfg, "fn": fn, "record": record,
            "centers": np.asarray(fn.centers, dtype=float)}


# ─── Landscape kinds ─────────────────────────────────────────────────────────────

#: ``plan --landscape`` choices. "needles" is this module; the others are modules
#: with the same interface (build_landscape, found_radius, plan_feasibility,
#: selftest, PLAIN_Y, PEAK_Y).
LANDSCAPE_KINDS = ("needles", "varied_height")


def landscape_module(kind: str | None = None):
    """The module that builds landscapes of ``kind`` (None -> "needles", which is
    what manifests written before kinds existed hold)."""
    kind = kind or "needles"
    if kind == "needles":
        import sys
        return sys.modules[__name__]
    if kind == "varied_height":
        from . import varied_height
        return varied_height
    raise ValueError(f"unknown landscape kind {kind!r}; known: {LANDSCAPE_KINDS}")


def fn_from_config(cfg: dict):
    """Rebuild a cell's objective from its ``ensemble_config.json``."""
    if cfg.get("kind", "needles") == "needles":
        from synthetic_data.ensemble import Ensemble
        return Ensemble(**cfg)
    return landscape_module(cfg["kind"]).from_config(cfg)


# ─── Self-test ───────────────────────────────────────────────────────────────────

def selftest(verbose: bool = True) -> None:
    """Check the closed-form identities this module reasons from, on real objects.

    ``python -m benchmarks.sweeps selftest``: plain at 0.75, peaks at exactly 1.0,
    background identically zero, the objective equal to its closed form, paring a
    no-op, and the prominence separation reproducing sigma_y exactly.
    """
    from synthetic_data.ensemble import Ensemble

    rng = np.random.default_rng(0)
    for dim in GRID_DIM:
        built = build_landscape(dim, 10, 6.0, seed=dim)
        fn, C = built["fn"], built["centers"]
        peak = float(np.max(fn.predict(C)))
        assert abs(peak - PEAK_Y) < 1e-9, f"dim {dim}: peak {peak} != {PEAK_Y}"
        probe = rng.random((20_000, dim))
        y = fn.predict(probe)
        assert y.min() >= PLAIN_Y - 1e-9, f"dim {dim}: y dipped to {y.min()}"
        assert y.max() <= PEAK_Y + 1e-9, f"dim {dim}: y rose to {y.max()}"
        bg = fn._background_field(probe)
        assert np.abs(bg).max() < 1e-12, f"dim {dim}: background is not zero"
        E = np.exp(-6.0 * np.linalg.norm(probe[:, None, :] - C[None, :, :], axis=2)
                   / math.sqrt(dim)).max(axis=1)
        assert np.abs(y - np.maximum(0.5 + 0.5 * E, PLAIN_Y)).max() < 1e-9, \
            f"dim {dim}: y != max(0.5 + 0.5E, 0.75)"
        if verbose:
            print(f"  [selftest] dim {dim}: cube, closed form, plain, peaks, paring OK "
                  f"({built['record']['n_prominence_resolved']}/10 prominence-resolved)")

    for dim in (3, 10):
        for b in (2.2, 15.0):
            s = prominence_separation(b, dim)
            c0 = np.full(dim, 0.5)
            v = np.zeros(dim)
            v[0], v[1] = 1.0, -1.0
            c1 = c0 + s * v / np.linalg.norm(v)
            fn = Ensemble(**ensemble_config(dim, b, np.vstack([c0, c1]), 0))
            t = np.linspace(0, 1, 401).reshape(-1, 1)
            y = fn.predict(c0 + t * (c1 - c0))
            prom = float(y[0] - y.min())
            assert abs(prom - sigma_y_at_peak()) < 1e-6, \
                f"dim {dim} b {b}: prominence at s_prom is {prom}, want {sigma_y_at_peak()}"
    if verbose:
        print("  [selftest] prominence separation reproduces sigma_y exactly")

    infeasible = [r for r in plan_feasibility() if not r["feasible"]]
    if verbose:
        print(f"  [selftest] {len(infeasible)} of {len(plan_feasibility())} default "
              "configuration(s) over the packing bound")
    print("  [selftest] OK")
