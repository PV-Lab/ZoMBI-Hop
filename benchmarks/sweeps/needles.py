"""
benchmarks/sweeps/needles.py
============================
The landscape this sweep varies: **Ackley bumps and nothing else**.

``synthetic_data.ensemble.Ensemble`` stacks seven feature families (true optima,
weak distractors, ridges, roughness, anisotropy, plateaus, an edge bias) into one
objective. This sweep turns every one of them off but the first, so the surface is
exactly ``n`` negated-Ackley basins of sharpness ``b`` on a flat plain — the
needle-in-a-haystack picture in its purest form. With the whole background field
``G(x) == 0`` the raw objective collapses to ``F = max(T, 0)`` and the output map
(see ``synthetic_data/ENSEMBLE.md``) reduces to a closed form worth having:

    y(x) = max( 0.5 + 0.5 * E(x),  0.75 ),   E(x) = max_c exp(-b * ||x - c|| / sqrt(d))

so the plain sits at **0.75**, every optimum peaks at exactly **1.0**, and a basin
meets the plain at radius ``sqrt(d) * ln 2 / b``. Everything below is derived from
that identity, which :func:`selftest` checks numerically rather than trusting.

Why the optima are placed here instead of drawn
-----------------------------------------------
The sweep's whole point is a clean "number of needles" axis, and stock uniform
placement does not give one. ``Ensemble`` pares its own optima: a basin landing
within ``input_noise`` of an already-tagged one is still drawn on the surface but
is NOT advertised in ``centers`` / ``known_maxima``, because two peaks that close
are one peak as far as the apparatus is concerned. Uniform draws collide often in
a small simplex, so a request for 50 optima advertises ~21 in dim 3, ~33 in dim 4
and 50 in dim 10 — the count axis would collapse, and collapse *differently per
dimension*, confounding the two axes the sweep is trying to separate.

So this module places the centers itself, subject to a separation that makes all
``n`` of them resolvable, and hands them to ``Ensemble(pinned_optima=...)`` with
``n_optima=0``. Nothing is drawn at random on top, so ``known_maxima`` is exactly
the set placed here.

What "resolvable" means
-----------------------
``METHODS.md`` section 1 defines the target set as the local maximisers "whose
basins are resolvable above the noise — wider than sigma_x and more prominent than
sigma_y". That is two conditions, and both are enforced as a minimum pairwise
separation ``s*`` between optima:

1. **Wider than the input noise** — ``s >= sigma_x = 0.128``
   (``run_mobo.NOISE_LEVEL``, measured on the deposition system). Below this the
   apparatus cannot be *asked* for one optimum rather than its neighbour, and it is
   the exact test ``Ensemble._tag_true_optima`` applies. Meeting it is what makes
   the paring a no-op, so ``len(fn.centers) == n`` by construction (asserted in
   :func:`build_landscape`).

2. **More prominent than the output noise** — the saddle between two adjacent
   peaks has to dip more than ``sigma_y`` below them, or the pair reads as one
   broad hill under metrology noise no matter how far apart the tips are. With
   ``G == 0`` this is solvable in closed form. Two equal peaks at separation ``s``
   put their saddle at the midpoint, where ``E = exp(-b*s/(2*sqrt(d)))``, so

       prominence = 1.0 - max(0.5 + 0.5*E, 0.75)

   and the simulator's noise is multiplicative, ``sigma_y = 0.045 * |y|``
   (``run_mobo.OUTPUT_NOISE_FRAC``), i.e. ``0.045`` at a peak. Requiring
   ``prominence >= sigma_y`` gives ``E <= 1 - 2*sigma_y`` and therefore

       s >= s_prom(b, d) = -2 * ln(1 - 2*sigma_y) * sqrt(d) / b
                        ~=  0.1886 * sqrt(d) / b

The target is ``s* = max(sigma_x, s_prom(b, d))``. Condition 2 binds only at the
broadest sharpness in the sweep (``b = 2.2``): at ``b >= 6`` a basin is narrow
enough that ``sigma_x`` is the stricter of the two at every dimension.

One separation per row, not per cell
------------------------------------
Because ``s_prom`` goes as ``1/b``, letting each cell place at its *own* ``s*``
makes the sharpness axis vary two things at once — the basin shape and the layout
the basin shape forced. On this grid that splits the axis in half: ``b = 2.2``
places at ``s_prom`` (0.1485 at dim 3, rising to 0.2711 at dim 10) while ``b >= 6``
all place at the plain 0.128 floor, so the broadest column — the one the sharp
columns are read against — is the only one laid out differently.

So the whole ``(dim, n_needles, draw)`` row is placed **once**, at the strictest
width's target (:func:`placement_width`), and every sharpness reuses those exact
centers: :meth:`NeedleFactory.placement_seed` leaves ``basin_width`` out of the
hash, so the four cells share a seed as well as a separation. Sharpness then varies
sharpness alone, and a comparison along the axis is paired — the layout cancels
instead of being one more thing the draws have to average over. The strictest
width is the safe standard because it is the only one no column has to be relaxed
below, so no cell is placed tighter than its own prominence rule wanted; each cell
still records ``separation_own_target`` next to the shared ``separation_target``.

The pairwise form is *necessary* but not quite sufficient — the field is a max over
all basins, so a third peak near a pair can only lift their saddle. The prominence
is therefore also **measured** on the built landscape (:func:`prominence_report`),
by probing the true saddle along the segment to each optimum's nearest neighbour,
and the measured count is recorded per cell rather than assumed.

Where the geometry runs out
---------------------------
``s*`` and ``n`` can be jointly impossible: a 2-simplex is a triangle of area
0.866, so it holds about ``1/s*^2`` points at separation ``s*`` — 61 at 0.128 but
only ~45 at the 0.1485 that ``b = 2.2`` demands. Since the whole row now places at
that separation, this is a property of the row rather than of one cell.
:func:`plan_feasibility` reports it at *plan* time, before any GPU time is spent,
and :func:`place_optima` falls
back to the largest separation it can actually achieve (never below ``sigma_x``, so
the count stays exact) while recording ``separation_target`` alongside
``separation_achieved``, so such a cell is analysed knowing its optima are packed
tighter than the prominence rule wanted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ._paths import ensure_paths

ensure_paths()

from mobo_landscapes import LandscapeSpec  # noqa: E402

# Noise constants, imported by value from the shared code rather than re-typed:
# these are physical measurements (see src/default_hparams.DEFAULT_INPUT_NOISE and
# run_mobo's own comments) and a second literal here would be a second thing to
# update when they are re-measured.
from run_mobo import NOISE_LEVEL as SIGMA_X  # noqa: E402  input/actuation noise, 0.128
from run_mobo import OUTPUT_NOISE_FRAC as SIGMA_Y_FRAC  # noqa: E402  metrology noise, 0.045

# Output-map constants of the bumps-only landscape (see the module docstring).
PLAIN_Y = 0.75   # the flat background
PEAK_Y = 1.0     # every optimum, exactly

# Placed separations are set this much above ``s*``. ``_tag_true_optima`` keeps a
# basin when it is ``>=`` min_dist from every kept one, and a relaxation that
# converges *to contact* lands exactly on the boundary, where a float rounding
# either way decides whether an optimum is advertised. 2% of clearance costs
# nothing and removes the coin flip.
SEPARATION_MARGIN = 1.02


# ─── The sweep grid ──────────────────────────────────────────────────────────────

#: The three swept axes. Full-factorial: 4 x 4 x 4 = 64 landscape configurations.
GRID_N_NEEDLES: tuple[int, ...] = (2, 10, 30, 50)
GRID_BASIN_WIDTH: tuple[float, ...] = (2.2, 6.0, 10.0, 15.0)
GRID_DIM: tuple[int, ...] = (3, 4, 6, 10)


# ─── Resolvability ───────────────────────────────────────────────────────────────

def sigma_y_at_peak() -> float:
    """Output-noise sd at an optimum: the noise is multiplicative and peaks are 1.0."""
    return SIGMA_Y_FRAC * PEAK_Y


def prominence_separation(basin_width: float, dim: int) -> float:
    """Smallest separation at which two adjacent peaks stay distinguishable.

    Solves ``1.0 - (0.5 + 0.5*exp(-b*s/(2*sqrt(d)))) >= sigma_y`` for ``s`` — the
    saddle between the pair must dip at least one output-noise sd below the peaks.
    See "What resolvable means" in the module docstring for the derivation.
    """
    b = float(basin_width)
    if b <= 0:
        raise ValueError(f"basin_width must be positive, got {basin_width}")
    e_max = 1.0 - 2.0 * sigma_y_at_peak()   # saddle envelope value at the threshold
    return -2.0 * math.log(e_max) * math.sqrt(float(dim)) / b


def target_separation(basin_width: float, dim: int) -> float:
    """``s* = max(sigma_x, s_prom(b, d))`` — both of METHODS' resolvability tests."""
    return max(float(SIGMA_X), prominence_separation(basin_width, dim))


def placement_width(dim: int, widths=GRID_BASIN_WIDTH) -> float:
    """The width in ``widths`` that demands the LARGEST separation at ``dim``.

    Every cell in a ``(dim, n_needles, draw)`` row is placed at this one width's
    ``target_separation``, so the sharpness axis varies sharpness ALONE. Placing
    each cell at its own ``target_separation`` — which is what this sweep did
    before — makes the axis change two things at once: ``prominence_separation``
    goes as ``1/b``, so the broadest column is placed under a materially stricter
    spacing rule than the rest, and a difference read along the axis cannot be
    attributed to the basin shape rather than to the layout it forced. The effect
    is confined to the columns where the prominence rule actually beats the
    ``sigma_x`` floor (on the shipped grid: ``b = 2.2`` only, by 16% at dim 3
    rising to 112% at dim 10), which is precisely the column the sharp ones are
    compared against.

    The strictest width is the safe one to standardise on: it is the only choice
    that no column has to be relaxed below, so every cell keeps peaks its own
    prominence rule would call resolvable. Since ``prominence_separation`` is
    monotone decreasing in ``b`` this is the smallest width on the grid, but it is
    computed rather than assumed so a custom ``--basin-widths`` list still gets the
    right answer, floor clamping included.
    """
    return max((float(b) for b in widths),
               key=lambda b: target_separation(b, dim))


def basin_plain_radius(basin_width: float, dim: int) -> float:
    """Distance from an optimum at which its basin meets the plain (``E = 0.5``).

    Purely diagnostic: compared against the separation it says whether a cell's
    basins overlap at all, which is the difference between "50 needles" and "one
    ridged mesa with 50 tips".
    """
    return math.sqrt(float(dim)) * math.log(2.0) / float(basin_width)


def simplex_capacity(dim: int, separation: float) -> float:
    """Roughly how many points at ``separation`` fit in the ``dim``-simplex.

    The unit simplex embedded in R^dim is a regular ``(dim-1)``-simplex of edge
    ``sqrt(2)``; volume per point is the ``(dim-1)``-ball of radius ``s/2`` divided
    by the packing density. Only ever used to *warn* at plan time, so the density is
    taken as 1 — an optimistic bound, so a cell this call rejects certainly does not
    fit — and boundary effects are ignored.
    """
    p = int(dim) - 1                       # intrinsic dimension of the simplex
    edge = math.sqrt(2.0)
    # Volume of a regular p-simplex of edge a, embedded in R^(p+1).
    vol = (edge ** p) * math.sqrt(p + 1.0) / (math.factorial(p) * (2.0 ** (p / 2.0)))
    r = float(separation) / 2.0
    ball = (math.pi ** (p / 2.0)) * (r ** p) / math.gamma(p / 2.0 + 1.0)
    return vol / ball if ball > 0 else float("inf")


def plan_feasibility(dims=GRID_DIM, needles=GRID_N_NEEDLES,
                     widths=GRID_BASIN_WIDTH) -> list[dict]:
    """One row per grid cell: the separation it needs and whether it can fit.

    Called by ``plan`` so a campaign says up front which cells are geometrically
    over-subscribed, instead of discovering it hours in on a worker.
    """
    rows = []
    for dim in dims:
        # Every width at this dim is placed at the strictest one's separation, so
        # feasibility is a property of the row, not of the individual cell.
        pw = placement_width(dim, widths)
        s = target_separation(pw, dim)
        cap = simplex_capacity(dim, s)
        for n in needles:
            for b in widths:
                rows.append({
                    "dim": int(dim), "n_needles": int(n), "basin_width": float(b),
                    "separation_width": float(pw),
                    "separation_target": round(s, 6),
                    "separation_own_target": round(target_separation(b, dim), 6),
                    "prominence_binds": bool(prominence_separation(b, dim) > SIGMA_X),
                    "basin_plain_radius": round(basin_plain_radius(b, dim), 6),
                    "capacity_estimate": round(cap, 1),
                    "feasible": bool(n <= cap),
                })
    return rows


# ─── Placement ───────────────────────────────────────────────────────────────────

def _project_simplex(X: np.ndarray) -> np.ndarray:
    """Exact Euclidean projection of each row of ``X`` onto the probability simplex.

    ``ensemble._project_simplex`` clips-and-renormalises, which is cheap and fine
    for scattering a ridge endpoint but is NOT the nearest point on the simplex —
    it drags a point sideways. The relaxation below moves optima a fraction of the
    separation at a time and pushes them against the boundary, where that sideways
    drag would fight the repulsion, so the real projection (Duchi et al. 2008) is
    used here.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n, d = X.shape
    U = -np.sort(-X, axis=1)
    css = np.cumsum(U, axis=1)
    idx = np.arange(1, d + 1)
    cond = U - (css - 1.0) / idx > 0
    rho = d - 1 - np.argmax(cond[:, ::-1], axis=1)
    theta = (css[np.arange(n), rho] - 1.0) / (rho + 1.0)
    return np.maximum(X - theta[:, None], 0.0)


def _min_pairwise(X: np.ndarray) -> float:
    if len(X) < 2:
        return float("inf")
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    return float(D.min())


def _dart_throw(dim: int, n: int, sep: float, rng, pool: int = 200_000) -> np.ndarray:
    """Uniform rejection sampling: keep a draw only if it clears every kept one.

    Tried first because it is the *least structured* way to hit the separation —
    the points stay an honest uniform sample conditioned on the constraint. It
    saturates well below the packing limit (~45 of a possible ~61 at dim 3 /
    0.128), so it hands off to the relaxation whenever it comes up short.
    """
    kept: list[np.ndarray] = []
    for p in rng.dirichlet(np.ones(dim), size=pool):
        if len(kept) >= n:
            break
        if not kept or np.min(np.linalg.norm(np.asarray(kept) - p, axis=1)) >= sep:
            kept.append(p)
    return np.asarray(kept, dtype=float).reshape(-1, dim)


def _relax(X: np.ndarray, sep: float, rng, iters: int = 4000) -> np.ndarray:
    """Push overlapping optima apart until every pair clears ``sep``.

    Pairs closer than ``sep`` shove each other along their connecting axis by half
    the shortfall each, and the whole set is re-projected onto the simplex every
    step. This is what reaches counts dart-throwing cannot (50 in dim 3 at 0.128,
    which random sequential adsorption saturates short of), at the cost of some
    regularity: near the packing limit the result approaches a lattice, which is
    recorded via ``separation_achieved`` rather than hidden.
    """
    X = _project_simplex(X.copy())
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
                # Coincident points have no separating axis; nudge along a random
                # tangent (sum-zero, so the step stays in the simplex's plane).
                v = rng.normal(size=X.shape[1])
                v -= v.mean()
                nv = float(np.linalg.norm(v)) or 1.0
            step[a] += 0.5 * (sep - D[a, b]) * v / nv
        X = _project_simplex(X + step)
    return X


def place_optima(dim: int, n: int, basin_width: float, seed: int,
                 *, separation_width: float | None = None) -> dict:
    """``n`` mutually resolvable optima on the ``dim``-simplex.

    Returns the centers plus the placement record that goes into the cell's
    landscape file: what separation was asked for, what was achieved, and whether
    the geometry allowed the prominence rule to be met.

    The hard floor is ``sigma_x``: the count must come out exactly ``n``, and that
    is only guaranteed while every pair clears the distance ``Ensemble`` pares on.
    The prominence target is the *preferred* separation and is the one abandoned
    first if both cannot be had.

    ``basin_width`` enters ONLY through the separation, so ``separation_width``
    overrides it there and leaves the rest of the call unchanged: the sweep passes
    :func:`placement_width` so every cell in a ``(dim, n, draw)`` row is laid out
    identically and the sharpness axis is sharpness alone. ``basin_width`` is still
    recorded, as ``separation_own_target``, so a row says what this cell would have
    asked for on its own.
    """
    dim, n = int(dim), int(n)
    rng = np.random.default_rng(int(seed))
    s_width = float(basin_width if separation_width is None else separation_width)
    s_target = target_separation(s_width, dim) * SEPARATION_MARGIN
    s_floor = float(SIGMA_X) * SEPARATION_MARGIN

    def _attempt(sep: float) -> np.ndarray | None:
        X = _dart_throw(dim, n, sep, rng)
        if len(X) < n:
            # Top up to n from a fresh uniform draw and let the relaxation sort out
            # the overlaps; starting the relaxation from a partly-valid set beats
            # starting it from scratch.
            extra = rng.dirichlet(np.ones(dim), size=n - len(X))
            X = np.vstack([X, extra]) if len(X) else extra
        X = _relax(X, sep, rng)
        return X if _min_pairwise(X) >= sep * 0.999 else None

    prominence_met = True
    X = _attempt(s_target)
    if X is None:
        # The prominence separation does not fit. Fall back to the input-noise
        # floor, which the packing bound says always does, and flag the cell.
        prominence_met = False
        X = _attempt(s_floor)
        if X is None:
            raise RuntimeError(
                f"could not place {n} optima on the {dim}-simplex even at the "
                f"input-noise floor {s_floor:.4f}: the grid asks for more optima "
                "than the simplex holds")

    return {
        "centers": X,
        "separation_target": float(s_target / SEPARATION_MARGIN),
        "separation_width": s_width,
        "separation_own_target": float(target_separation(basin_width, dim)),
        "separation_floor": float(SIGMA_X),
        "separation_achieved": float(_min_pairwise(X)),
        "prominence_target_met": bool(prominence_met),
        "placement_seed": int(seed),
    }


# ─── Landscape construction ──────────────────────────────────────────────────────

def ensemble_config(dim: int, basin_width: float, centers: np.ndarray,
                    seed: int) -> dict:
    """``Ensemble`` kwargs for a bumps-only landscape with exactly these optima.

    Every background family is passed at zero so ``_background_field`` returns
    identically 0 and the raw objective is ``max(T, 0)``. ``n_optima=0`` with
    ``pinned_optima`` set means nothing is drawn at random: ``peak_centers`` is the
    array handed in, in that order, and the greedy tagging walks it from the front.

    The dict is JSON-serialisable because ``run_single_trial`` writes it to
    ``ensemble_config.json`` and ``reseed_ensemble`` rebuilds the objective from
    that file — the cell has to be reproducible from its own artifacts.
    """
    return {
        "dim": int(dim),
        # true optima: placed explicitly, none sampled
        "n_optima": 0,
        "pinned_optima": np.asarray(centers, dtype=float).tolist(),
        "basin_width": float(basin_width),
        "basin_smoothing": 0.0,
        # every other feature family off
        "n_weak": 0, "weak_amp": 0.0,
        "n_ridges": 0, "ridge_amp": 0.0,
        "noise_amp": 0.0,
        "aniso_strength": 0.0,
        "n_plateaus": 0, "plateau_amp": 0.0,
        "edge_region": None, "edge_amp": 0.0,
        # paring distance: the separation the placement already guarantees, so the
        # tagging is a no-op and centers == pinned_optima
        "input_noise": float(SIGMA_X),
        "seed": int(seed),
    }


def prominence_report(fn, centers: np.ndarray, n_probe: int = 65) -> dict:
    """Measure, don't assume: how prominent each optimum actually is.

    The pairwise separation rule is derived for an isolated pair, but the field is a
    max over every basin, so a third peak sitting near a pair can only lift their
    saddle. This walks the segment from each optimum to its nearest neighbour, takes
    the true minimum of the objective along it, and calls the optimum resolved when
    the dip clears one output-noise sd. What comes back is the honest count for the
    landscape as built.
    """
    C = np.asarray(centers, dtype=float)
    if len(C) < 2:
        return {"n_prominence_resolved": int(len(C)),
                "min_prominence": None, "median_prominence": None,
                "prominence_threshold": round(float(sigma_y_at_peak()), 6)}
    D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    nn = np.argmin(D, axis=1)
    t = np.linspace(0.0, 1.0, int(n_probe)).reshape(-1, 1)
    proms = []
    for i, j in enumerate(nn):
        seg = C[i] + t * (C[j] - C[i])
        y = np.asarray(fn.predict(seg), dtype=float)
        proms.append(float(y[0] - y.min()))
    proms = np.asarray(proms)
    thresh = sigma_y_at_peak()
    return {
        "n_prominence_resolved": int((proms >= thresh).sum()),
        "min_prominence": round(float(proms.min()), 6),
        "median_prominence": round(float(np.median(proms)), 6),
        "prominence_threshold": round(float(thresh), 6),
    }


@dataclass
class NeedleFactory:
    """One grid cell's landscape source: fixed ``(dim, n_needles, basin_width)``.

    Satisfies ``benchmarks.ablations.landscapes.LandscapeFactory``, so a cell runs
    through the same ``run_ablation_trial`` path — and writes the same artifact set —
    as an ablation cell. ``build(index)`` varies only the *placement* of the optima,
    so the draws within a cell are repeats of one landscape TYPE, which is the unit
    this sweep averages over.

    ``time_limit_hours`` is a safety ceiling, not the budget: the budget is measured
    in LineBO lines and enforced by :mod:`benchmarks.sweeps.budget`. It is here only
    so a pathological cell cannot hold a worker forever.
    """

    dim: int
    n_needles: int
    basin_width: float
    seed: int = 0
    time_limit_hours: float | None = None
    kind: str = "needles"
    n_available: int | None = None      # placement seeds: effectively unbounded
    # The campaign's full sharpness axis, which sets the shared placement
    # separation via ``placement_width``. Defaults to the shipped grid so a
    # factory built by hand still lands on the same layout as the sweep's.
    placement_widths: tuple[float, ...] = GRID_BASIN_WIDTH
    _base: LandscapeSpec | None = field(default=None, repr=False, compare=False)

    @property
    def cell_name(self) -> str:
        return f"d{self.dim:02d}_n{self.n_needles:02d}_b{self.basin_width:g}"

    def spec(self) -> dict:
        return {
            "kind": self.kind, "dim": self.dim, "n_needles": self.n_needles,
            "basin_width": self.basin_width, "seed": self.seed,
            "separation_width": self.placement_width,
            "separation_target": round(
                target_separation(self.placement_width, self.dim), 6),
            "separation_own_target": round(
                target_separation(self.basin_width, self.dim), 6),
            "basin_plain_radius": round(basin_plain_radius(self.basin_width, self.dim), 6),
            "time_limit_hours": self.time_limit_hours,
        }

    def placement_seed(self, index: int) -> int:
        """Deterministic in ``(dim, n_needles, draw)`` so a landscape is reproducible
        from the manifest alone.

        ``basin_width`` is deliberately NOT mixed in. Together with the shared
        :func:`placement_width` this is what pairs the sharpness axis: all four
        widths at one ``(dim, n, draw)`` get the same seed AND the same separation,
        so they are the same centers under different basin shapes, and a paired
        comparison across the axis cancels the layout entirely. It used to be mixed
        in, so no two cells shared a layout and every width comparison also
        compared terrain — re-running an older campaign against this code will
        therefore build different landscapes than it did originally.

        Rows still differ from one another: ``dim`` and ``n_needles`` are mixed in,
        so two grid rows at the same draw index do not inherit correlated layouts.
        """
        h = (int(self.seed) * 1_000_003
             ^ int(self.dim) * 2_654_435_761
             ^ int(self.n_needles) * 40_503
             ^ int(index) * 15_485_863)
        return int(abs(h) % 1_000_000)

    @property
    def placement_width(self) -> float:
        """The separation-setting width shared by this cell's whole sharpness row."""
        return placement_width(self.dim, self.placement_widths)

    def _base_spec(self) -> LandscapeSpec:
        """The heavy ``LandscapeSpec`` (dim-3 render grid included), built once.

        Only the cheap per-draw config varies; ``run_single_trial`` reseeds the
        objective from it. Mirrors ``ablations.landscapes.EnsembleFactory``.
        """
        if self._base is None:
            import run_mobo as rm

            self._base = rm.build_ensemble_landscape(
                self.dim, optima_margin=0.2, seed=self.seed,
                time_limit_hours=self.time_limit_hours)
        return self._base

    def build(self, index: int) -> tuple[LandscapeSpec, dict]:
        seed = self.placement_seed(index)
        placed = place_optima(self.dim, self.n_needles, self.basin_width, seed,
                              separation_width=self.placement_width)
        cfg = ensemble_config(self.dim, self.basin_width, placed["centers"], seed)
        return self._base_spec(), cfg


def build_landscape(dim: int, n: int, basin_width: float, seed: int,
                    *, separation_width: float | None = None) -> dict:
    """Place, build and VERIFY one landscape. Returns the record for the cell file.

    The verification is the point: it asserts the built ``Ensemble`` advertises
    exactly ``n`` optima — the promise the whole placement scheme exists to keep —
    and it measures the prominence actually achieved rather than the prominence the
    pairwise formula predicted.
    """
    from synthetic_data.ensemble import Ensemble

    placed = place_optima(dim, n, basin_width, seed,
                          separation_width=separation_width)
    cfg = ensemble_config(dim, basin_width, placed["centers"], seed)
    fn = Ensemble(**cfg)
    n_true = len(fn.centers)
    if n_true != n:
        raise AssertionError(
            f"placed {n} optima at separation {placed['separation_achieved']:.4f} "
            f"but Ensemble advertises {n_true}; the paring distance ({SIGMA_X}) and "
            "the placement floor have drifted apart")
    record = {
        "dim": int(dim), "n_needles": int(n), "basin_width": float(basin_width),
        **{k: v for k, v in placed.items() if k != "centers"},
        **prominence_report(fn, placed["centers"]),
    }
    return {"config": cfg, "fn": fn, "record": record, "centers": placed["centers"]}


# ─── Self-test ───────────────────────────────────────────────────────────────────

def selftest(verbose: bool = True) -> None:
    """Check the closed-form identities this module reasons from, on real objects.

    Run by ``python -m benchmarks.sweeps selftest``. Every claim in the docstring
    that a later reader would otherwise have to take on faith — the plain at 0.75,
    peaks at exactly 1.0, the background vanishing, the paring being a no-op, the
    prominence threshold being the separation that produces it — is checked here
    against a constructed ``Ensemble`` rather than re-derived on paper.
    """
    from synthetic_data.ensemble import Ensemble

    rng = np.random.default_rng(0)
    for dim in GRID_DIM:
        built = build_landscape(dim, 10, 6.0, seed=dim)
        fn, C = built["fn"], built["centers"]
        # Peaks are exactly 1.0 and the plain is exactly 0.75.
        peak = float(np.max(fn.predict(C)))
        assert abs(peak - PEAK_Y) < 1e-9, f"dim {dim}: peak {peak} != {PEAK_Y}"
        probe = rng.dirichlet(np.ones(dim), size=20_000)
        y = fn.predict(probe)
        assert y.min() >= PLAIN_Y - 1e-9, f"dim {dim}: y dipped to {y.min()}"
        assert y.max() <= PEAK_Y + 1e-9, f"dim {dim}: y rose to {y.max()}"
        # The background field really is identically zero.
        bg = fn._background_field(probe)
        assert np.abs(bg).max() < 1e-12, f"dim {dim}: background is not zero"
        # The closed form the separation rule is derived from.
        E = np.exp(-6.0 * np.linalg.norm(probe[:, None, :] - C[None, :, :], axis=2)
                   / math.sqrt(dim)).max(axis=1)
        assert np.abs(y - np.maximum(0.5 + 0.5 * E, PLAIN_Y)).max() < 1e-9, \
            f"dim {dim}: y != max(0.5 + 0.5E, 0.75)"
        if verbose:
            print(f"  [selftest] dim {dim}: closed form, plain, peaks, paring OK "
                  f"({built['record']['n_prominence_resolved']}/10 prominence-resolved)")

    # A pair placed exactly at s_prom must land exactly on the prominence threshold.
    for dim in (3, 10):
        for b in (2.2, 15.0):
            s = prominence_separation(b, dim)
            c0 = np.full(dim, 1.0 / dim)
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
        print(f"  [selftest] {len(infeasible)} of 64 grid cell(s) over the packing "
              "bound: " + (", ".join(f"dim{r['dim']}/n{r['n_needles']}/b{r['basin_width']:g}"
                                     for r in infeasible) or "none"))
    print("  [selftest] OK")
