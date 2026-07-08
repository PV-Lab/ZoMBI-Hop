"""
ensemble.py
===========
A layered, "realistic" synthetic objective on the probability simplex.

Where ``ackley.py`` gives a (negated) Ackley field plus optional background
noise, this module stacks several independent landscape *features* into a
single objective so the benchmark looks more like a real materials-composition
response surface:

    * **Optima**          — strong negated-Ackley basins; the *true* maxima.
                            Their placement can be uniform ("scatter") or
                            **clustered** around a chosen simplex region, with an
                            adjustable *spread* controlling how loosely each
                            cluster hugs that corner / edge / face / centroid.
    * **Roughness**       — coherent Perlin-style simplex noise (fine ripple).
    * **Weak optima**     — extra, shorter negated-Ackley basins acting as
                            distractors / fake local optima.
    * **Ridges**          — high-value tubes around line segments of an
                            **adjustable length**.
    * **Edge bias**       — a smooth structural trend that raises (or lowers)
                            the whole surface toward a chosen simplex region
                            (corners / edges / faces / middle).
    * **Anisotropy**      — per-axis stretching of the distance metric, so
                            basins/ridges are elongated along some axes.
    * **Plateaus**        — regions flattened to a constant mesa (or basin) height.

Signed features (centered output)
---------------------------------
Unlike the earlier additive-only design, the background is built around a
**neutral level of 0** (raw units), and every background feature can add *or*
subtract mass: weak optima can be bumps or pits, ridges can be ridges or
trenches, plateaus can be mesas or basins, and the edge bias can push the
chosen region up or down.  The per-instance sign is drawn randomly (a fraction
``neg_frac`` of instances subtract).  This lets the *raw* surface centre around
its neutral level and span symmetrically in both directions; the final output
map (see "Output normalization" below) then squishes that symmetric span into
the upper half ``[0.5, 1]``.

Unlike ``ackley.py`` this module deliberately avoids any hardcoded
"parameter-by-dimension" tables: every knob is a plain scalar (or a count) that
means the same thing at any ``dim`` (the only exception is the *count* of true
optima, which scales with simplex room).  Random placement of every feature is
driven by a single ``seed``.

Dominance of the true optima
----------------------------
The true optima are always the global maxima: let ``T(x)`` be the clean
true-optima field (peaking at ``_PEAK`` at each optimum) and ``G(x)`` the
combined signed *background* (weak optima + ridges + edge bias + roughness +
plateaus).  The raw objective is

    F(x) = max( T(x),  min( G(x), cap_high ) )

The background's *upward* excursions are capped at
``cap_high = _PEAK * (1 - 2 * optima_margin)`` *before* the true-optima field is
laid on top, so no amount of stretching, ridging or noise can push a distractor
up to a real optimum.  (Downward excursions — valleys/pits — are never capped;
they are what let the surface dip toward 0 and "float" the optima below 1.)

Output normalization ("optima float")
--------------------------------------
The raw field is mapped to output with a single symmetric scale centred on the
neutral level, then the result is squished into the **upper half** ``[0.5, 1]``:

    u = clip( 0.5 + 0.5 * F / scale,  0, 1 ),   scale = max(|raw_min|, |raw_max|)
    y = 0.5 + 0.5 * u

so a symmetric ``[0, 1]`` span (neutral at 0.5) becomes a ``[0.5, 1]`` span with
the neutral level at 0.75.  The true optima are still the maxima, but they are
**not pinned to exactly 1.0**: if the deepest valley is deeper than the optima
are tall, ``scale`` is set by the valley and the optima land below 1.0.  The
final squish halves every distance from the neutral level, so it **deliberately
deflates the optima margin** in exchange for keeping the whole surface in the
top half of the output range.

The ``Ensemble`` class exposes ``predict(X)`` with the scikit-learn-style
``(N, d) -> (N,)`` signature, plus ``known_maxima`` and ``centers`` like
``Ackley``.

Example
-------
    from synthetic_data.ensemble import Ensemble

    fn = Ensemble(dim=4, n_optima=4, n_weak=8, n_ridges=2, n_plateaus=2)
    y = fn.predict(X)                       # (N,) in [0.5, 1]
    for c, v in fn.known_maxima:            # each v is the global max
        ...
"""

from __future__ import annotations

import numpy as np

from synthetic_data.ackley import (
    ACKLEY_A,
    ACKLEY_C,
    ACKLEY_SCALE,
    simplex_noise,
)

# Raw amplitude unit: the span of a single negated-Ackley basin (``scale * a``).
# The background is centred on a neutral level of 0; the true optima rise one
# full span (``_PEAK``) above that level.
_BASE = -ACKLEY_SCALE * ACKLEY_A
_SPAN = 0.0 - _BASE  # == ACKLEY_SCALE * ACKLEY_A
_PEAK = _SPAN        # raw value at a true optimum (one span above neutral 0)
_RANGE_SAMPLES = 100_000  # simplex samples used to estimate raw min/max for the output map

# Default minimum separation (composition L2) below which two true optima are
# considered indistinguishable: a basin placed this close to an already-tagged
# optimum still appears on the landscape but is *not* advertised as a true
# optimum.  Matches ``DataHandler``'s default input noise.
_DEFAULT_INPUT_NOISE = 0.064

# Anisotropy: max log-stretch grows as ``log1p(ANISO_RATE * strength)`` so the
# stretched axes widen by ~4x at strength 10 and ~16x at strength 50.
ANISO_RATE = 0.3

# Per-feature seed offsets so one master ``seed`` deterministically drives every
# random placement without the features sharing a stream.
_SEED_OPTIMA = 0
_SEED_WEAK = 101
_SEED_RIDGES = 202
_SEED_PLATEAUS = 303
_SEED_ANISO = 404
_SEED_NOISE = 505
_SEED_SIGNS = 606
_SEED_EDGE = 707
_SEED_CLUSTERS = 808

# Region targets selectable by the location-based features (clustering of optima
# and the edge bias).  ``"scatter"`` (optima only) means uniform placement / no
# region preference.  ``"faces"`` only differs from ``"edges"`` for dim >= 4.
REGION_MODES = ("corners", "edges", "faces", "middle")


def region_modes(dim: int) -> tuple[str, ...]:
    """Region targets meaningful at simplex ``dim`` (drops ``"faces"`` for dim 3,
    where a facet coincides with an edge)."""
    if dim <= 3:
        return tuple(m for m in REGION_MODES if m != "faces")
    return REGION_MODES


# ── Per-run randomization ─────────────────────────────────────────────────────
# Hardcoded range of *true* optima to draw for each simplex dimension.  These are
# the only feature counts that scale with ``dim`` (every other Ensemble knob means
# the same thing at any dimension); higher-dimensional simplices have far more room
# so they get many more basins.  ``random_ensemble_config`` is the single source of
# truth shared by optimize/run_mobo.py, optimize/evaluate.py and plot_ensemble.py.
OPTIMA_COUNT_RANGES: dict[int, tuple[int, int]] = {
    3: (5, 30),
    4: (20, 50),
    10: (50, 150),
}


def optima_count_range(dim: int) -> tuple[int, int]:
    """Inclusive ``(lo, hi)`` range of true optima to draw at simplex ``dim``.

    Uses the hardcoded :data:`OPTIMA_COUNT_RANGES` table for the benchmarked
    dimensions (3/4/10); other dimensions fall back to ``(5*dim, 15*dim)`` — the
    same slope the 10-simplex anchor sits on.
    """
    if dim in OPTIMA_COUNT_RANGES:
        return OPTIMA_COUNT_RANGES[dim]
    return (5 * dim, 15 * dim)


# Range of basin sharpness ``b`` to draw, fixed for every simplex dimension.
BASIN_WIDTH_RANGE: tuple[float, float] = (2.2, 15.0)


def basin_width_range(dim: int) -> tuple[float, float]:
    """``(lo, hi)`` range of basin sharpness ``b`` to draw.

    Fixed at :data:`BASIN_WIDTH_RANGE` for all dimensions: a broad basin floor
    (``2.2``) up to a moderately sharp basin (``15``).
    """
    return BASIN_WIDTH_RANGE


def _sobol_point(d: int, index: int, seed: int) -> np.ndarray:
    """The ``index``-th point of a scrambled ``d``-dimensional Sobol' sequence.

    Quasi-random (low-discrepancy) sampling spreads the drawn configurations far
    more evenly across the parameter hypercube than independent uniform draws, so
    a sweep of trials covers the feature space with fewer gaps/clusters.  The
    ``seed`` selects the scramble; any two calls with the same ``(d, index, seed)``
    return the same point, so a landscape is reproducible from ``(seed, index)``.
    """
    from scipy.stats import qmc

    sob = qmc.Sobol(d=d, scramble=True, seed=int(seed) & 0xFFFFFFFF)
    if index > 0:
        sob.fast_forward(int(index))
    return sob.random(1)[0]


def random_ensemble_config(
    dim: int,
    index: int = 0,
    total: int | None = None,
    *,
    seed: int = 0,
    optima_margin: float = 0.2,
    input_noise: float = _DEFAULT_INPUT_NOISE,
) -> dict:
    """Draw the ``index``-th :class:`Ensemble` configuration from a Sobol' sweep.

    Every continuous knob, on/off toggle and categorical choice (region modes,
    optima layout) is mapped from one coordinate of a scrambled Sobol' point, so
    a run of ``index = 0, 1, 2, …`` walks a low-discrepancy path through the whole
    configuration space (mirrors the "Randomize" button in
    ``synthetic_data/plot_ensemble.py``).  A disabled feature passes its
    count/amplitude as 0.  ``n_optima`` is drawn from the dimension-specific
    :func:`optima_count_range`; ``optima_margin`` and ``input_noise`` are held
    fixed.  Returns a kwargs
    dict accepted directly by ``Ensemble(**config)``, so a saved config exactly
    recreates the landscape.

    Parameters
    ----------
    dim : simplex dimensionality.
    index : position in the Sobol' sequence (stable, reproducible per landscape).
    total : advisory total number of configs in the sweep (unused by the Sobol'
        sequence itself; accepted for API symmetry / future LHS batching).
    seed : selects the Sobol' scramble *and* the per-landscape feature placement
        seed; same ``(seed, index)`` -> identical landscape.
    """
    dim = int(dim)
    opt_lo, opt_hi = optima_count_range(dim)
    bw_lo, bw_hi = basin_width_range(dim)
    regions = region_modes(dim)
    layout_opts = ("scatter",) + regions

    # One Sobol' coordinate per knob, consumed in order.
    n_coords = 30
    pt = _sobol_point(n_coords, int(index), int(seed))
    _i = 0

    def u() -> float:
        nonlocal _i
        v = float(pt[_i])
        _i += 1
        return v

    def f_range(lo: float, hi: float) -> float:
        return lo + u() * (hi - lo)

    def i_range(lo: int, hi: int) -> int:
        return int(round(lo + u() * (hi - lo)))

    def on(p: float = 0.5) -> bool:
        return u() < p

    def pick(opts):
        return opts[min(len(opts) - 1, int(u() * len(opts)))]

    # Feature on/off toggles.
    weak_on, ridges_on, rough_on, aniso_on, plateaus_on, edge_on = (
        on(), on(), on(), on(), on(), on())

    # Per-landscape placement seed, deterministic in (seed, index).
    feature_seed = (int(seed) * 100_003 + int(index)) % 1_000_000

    return {
        "dim": dim,
        # true optima
        "n_optima": i_range(opt_lo, opt_hi),
        "basin_width": round(f_range(bw_lo, bw_hi), 1),
        "optima_margin": float(optima_margin),
        # optima placement / clustering
        "optima_layout": pick(layout_opts),
        "n_optima_clusters": i_range(1, 6),
        "optima_cluster_conc": round(f_range(20.0, 250.0), 1),
        "optima_cluster_spread": round(f_range(0.0, 1.0), 2),
        # weak optima (distractors)
        "n_weak": i_range(0, 30) if weak_on else 0,
        "weak_width": float(i_range(5, 300)),
        "weak_amp": round(f_range(0.0, 1.0), 2),
        # ridges
        "n_ridges": i_range(0, 8) if ridges_on else 0,
        "ridge_width": round(f_range(0.01, 0.25), 3),
        "ridge_amp": round(f_range(0.0, 1.0), 2),
        "ridge_length": round(f_range(0.1, 1.0), 2),
        # roughness
        "noise_freq": round(f_range(0.0, 40.0), 1),
        "noise_amp": float(i_range(0, 2000)) if rough_on else 0.0,
        "noise_octaves": i_range(1, 6),
        # anisotropy
        "aniso_strength": round(f_range(0.0, 50.0), 1) if aniso_on else 0.0,
        # plateaus
        "n_plateaus": i_range(0, 8) if plateaus_on else 0,
        "plateau_radius": round(f_range(0.02, 0.40), 2),
        "plateau_amp": round(f_range(0.0, 1.0), 2),
        # edge / structural bias
        "edge_region": pick(regions) if edge_on else None,
        "edge_amp": round(f_range(0.0, 1.0), 2),
        "edge_reach": round(f_range(0.10, 0.80), 2),
        # signed-feature mix (fraction of instances that subtract mass)
        "neg_frac": round(f_range(0.3, 0.6), 2),
        # optima paring (held fixed, like optima_margin)
        "input_noise": float(input_noise),
        # global
        "seed": int(feature_seed),
    }


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _anisotropy_scale(dim: int, strength: float, seed: int) -> np.ndarray:
    """Per-axis weights in ``(0, 1]`` that *widen* (stretch) some axes more
    than others, with the most-native axis anchored at 1.

    These weights multiply the per-axis deltas before any distance is taken, so
    a smaller weight makes the objective *less* sensitive along that axis and
    therefore elongates (stretches) basins/ridges along it.

    The scaling is deliberately one-sided: every weight is ``<= 1``, so
    increasing ``strength`` only ever widens basins along the stretched axes and
    never squeezes another axis into a sub-grid sliver that visually vanishes.
    The amount of stretch grows *logarithmically* with ``strength`` (and so
    saturates gently), giving a usable response across the whole slider range:
    the most-stretched axis is widened by up to ``exp(log1p(ANISO_RATE * strength))``
    relative to the most-native axis.

    ``strength == 0`` returns all ones (isotropic).
    """
    if strength <= 0:
        return np.ones(dim)
    rng = np.random.default_rng(seed)
    w = rng.random(dim)
    span = float(w.max() - w.min())
    if span < 1e-12:
        return np.ones(dim)
    # Rank axes in [0, 1]: one axis ends up native (0 -> scale 1), one maximally
    # stretched (1 -> scale exp(-max_log)); the rest spread between.
    w = (w - w.min()) / span
    max_log = float(np.log1p(ANISO_RATE * float(strength)))
    return np.exp(-max_log * w)


def _negated_ackley_env01(X: np.ndarray, center: np.ndarray, b: float,
                          axis_scale: np.ndarray) -> np.ndarray:
    """Anisotropic negated-Ackley envelope rescaled to ``[0, 1]`` (1 at ``center``)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    center = np.asarray(center, dtype=float).reshape(1, -1)
    delta = (X - center) * axis_scale.reshape(1, -1)
    d_eff = delta.shape[1]
    # ``exp(-b * rms_delta)`` is 1 at the center and decays to 0 far away.
    return np.exp(-b * np.sqrt(np.sum(delta ** 2, axis=1) / d_eff))


def _point_segment_dist(X: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Euclidean distance from each row of ``X`` to the segment ``p``–``q``."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    p = np.asarray(p, dtype=float).reshape(1, -1)
    q = np.asarray(q, dtype=float).reshape(1, -1)
    pq = q - p
    l2 = float((pq * pq).sum())
    if l2 < 1e-18:
        return np.linalg.norm(X - p, axis=1)
    t = np.clip(((X - p) @ pq.T).ravel() / l2, 0.0, 1.0)
    proj = p + t.reshape(-1, 1) * pq
    return np.linalg.norm(X - proj, axis=1)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _project_simplex(x: np.ndarray) -> np.ndarray:
    """Cheap projection of a point onto the probability simplex: clip the
    negatives and renormalize to sum 1 (used to place ridge endpoints)."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, None)
    s = x.sum()
    if s < 1e-12:
        return np.full_like(x, 1.0 / x.size)
    return x / s


# ── Ensemble objective ───────────────────────────────────────────────────────

class Ensemble:
    """A layered synthetic objective on the ``dim``-element probability simplex.

    Parameters
    ----------
    dim : int
        Simplex dimensionality (>= 2).
    n_optima, basin_width : true (strong) optima count and Ackley sharpness ``b``.
    optima_margin : float
        Normalized gap in ``[0, 0.5]`` guaranteeing the background's upward
        excursions stay at or below ``_PEAK * (1 - 2*optima_margin)`` while the
        optima sit at ``_PEAK`` (see module docstring).
    optima_layout : str
        ``"scatter"`` (uniform Dirichlet placement) or one of
        :data:`REGION_MODES` (``"corners"``/``"edges"``/``"faces"``/``"middle"``)
        to **cluster** the optima around that simplex region.
    n_optima_clusters, optima_cluster_conc : number of clusters and Dirichlet
        concentration (higher = tighter clusters) used when clustering.
    optima_cluster_spread : float
        How loosely the clusters hug the target region, in ``[0, 1]``.  Each
        cluster seed is pulled toward the simplex centroid by a random fraction
        in ``[0, optima_cluster_spread]`` before optima are drawn around it, so
        ``0`` reproduces the tight on-feature hug and higher values let optima
        sit progressively further off the corner/edge/face toward the interior.
    n_weak, weak_width, weak_amp : weak-optima (distractor) count, sharpness, and
        prominence in ``[0, 1]`` (fraction of the full basin height).  Each weak
        optimum is randomly a bump or a pit (see ``neg_frac``).
    n_ridges, ridge_width, ridge_amp, ridge_length : ridge count, tube width
        (simplex units), prominence in ``[0, 1]``, and segment length as a
        fraction of the simplex extent.  Each ridge is randomly raised or sunk.
    noise_freq, noise_amp, noise_octaves : background Perlin-noise controls.
    aniso_strength : per-axis stretching of the distance metric (0 = isotropic).
    n_plateaus, plateau_radius, plateau_amp : plateau count, radius (simplex
        units) and mesa height in ``[0, 1]``.  Each plateau is randomly a mesa or
        a basin.
    edge_region, edge_amp, edge_reach : structural-bias target region (one of
        :data:`REGION_MODES`, or ``None`` to disable), prominence in ``[0, 1]``,
        and reach (simplex units) over which the bias falls off.  The bias is
        randomly positive or negative.
    neg_frac : float
        Fraction of signed-feature instances that *subtract* mass instead of
        adding it (in ``[0, 1]``; ~0.5 keeps the surface centred).
    input_noise : float
        Minimum separation (composition L2) below which two optima are treated
        as the same peak.  Optima are tagged as *true* optima greedily in
        placement order: a basin placed within ``input_noise`` of an
        already-tagged optimum is still drawn on the landscape (its peak is
        real) but is left **untagged**, so ``centers`` / ``known_maxima`` only
        ever advertise mutually-separated optima.  ``0`` disables paring (every
        basin is a true optimum).  Defaults to :data:`_DEFAULT_INPUT_NOISE`.
    seed : single master seed driving every random placement and sign.
    """

    def __init__(
        self,
        dim: int = 3,
        *,
        # true optima
        n_optima: int = 4,
        basin_width: float = 65.0,
        optima_margin: float = 0.2,
        # optima placement / clustering
        optima_layout: str = "scatter",
        n_optima_clusters: int = 3,
        optima_cluster_conc: float = 80.0,
        optima_cluster_spread: float = 0.0,
        # weak optima (distractors)
        n_weak: int = 6,
        weak_width: float = 120.0,
        weak_amp: float = 0.6,
        # ridges
        n_ridges: int = 2,
        ridge_width: float = 0.06,
        ridge_amp: float = 0.6,
        ridge_length: float = 1.0,
        # roughness
        noise_freq: float = 8.0,
        noise_amp: float = 120.0,
        noise_octaves: int = 4,
        # anisotropy
        aniso_strength: float = 0.0,
        # plateaus
        n_plateaus: int = 2,
        plateau_radius: float = 0.12,
        plateau_amp: float = 0.7,
        # edge / structural bias
        edge_region: str | None = None,
        edge_amp: float = 0.0,
        edge_reach: float = 0.3,
        # signed-feature mix
        neg_frac: float = 0.5,
        # optima paring
        input_noise: float = _DEFAULT_INPUT_NOISE,
        # global
        seed: int = 0,
    ) -> None:
        if dim < 2:
            raise ValueError(f"dim must be >= 2 (got {dim}).")
        self.dim = dim
        self.seed = int(seed)

        self.optima_margin = float(np.clip(optima_margin, 0.0, 0.5))
        self.basin_width = float(basin_width)
        self.optima_layout = str(optima_layout)
        self.n_optima_clusters = max(1, int(n_optima_clusters))
        self.optima_cluster_conc = max(1e-3, float(optima_cluster_conc))
        self.optima_cluster_spread = float(np.clip(optima_cluster_spread, 0.0, 1.0))
        self.weak_width = float(weak_width)
        self.weak_amp = float(np.clip(weak_amp, 0.0, 1.0))
        self.ridge_width = max(1e-4, float(ridge_width))
        self.ridge_amp = float(np.clip(ridge_amp, 0.0, 1.0))
        self.ridge_length = float(np.clip(ridge_length, 1e-3, 2.0))
        self.noise_freq = float(noise_freq)
        self.noise_amp = max(0.0, float(noise_amp))
        self.noise_octaves = int(noise_octaves)
        self.plateau_radius = max(1e-4, float(plateau_radius))
        self.plateau_amp = float(np.clip(plateau_amp, 0.0, 1.0))
        self.edge_region = edge_region if edge_region in REGION_MODES else None
        self.edge_amp = float(np.clip(edge_amp, 0.0, 1.0))
        self.edge_reach = max(1e-4, float(edge_reach))
        self.neg_frac = float(np.clip(neg_frac, 0.0, 1.0))
        self.input_noise = max(0.0, float(input_noise))

        # Per-axis stretch shared by every distance-based feature.
        self.axis_scale = _anisotropy_scale(dim, float(aniso_strength), self.seed + _SEED_ANISO)

        # Random placement of every feature.  ``peak_centers`` holds *every*
        # placed basin (all of them shape the landscape); ``centers`` is the
        # subset advertised as true optima after paring out near-duplicates.
        self.peak_centers = self._sample_optima(int(n_optima))
        self._true_mask = self._tag_true_optima(self.peak_centers, self.input_noise)
        self.centers = self.peak_centers[self._true_mask]
        self.weak_centers = self._sample_simplex(int(n_weak), _SEED_WEAK)
        self.plateau_centers = self._sample_simplex(int(n_plateaus), _SEED_PLATEAUS)
        self.ridges = self._sample_ridges(int(n_ridges))

        # Per-instance signs (+1 raises, -1 lowers) for the signed features.
        sign_rng = np.random.default_rng(self.seed + _SEED_SIGNS)

        def _signs(n: int) -> np.ndarray:
            if n <= 0:
                return np.empty(0)
            return np.where(sign_rng.random(n) < self.neg_frac, -1.0, 1.0)

        self.weak_signs = _signs(int(n_weak))
        self.ridge_signs = _signs(int(n_ridges))
        self.plateau_signs = _signs(int(n_plateaus))
        self.edge_sign = -1.0 if sign_rng.random() < self.neg_frac else 1.0

        # Upward background excursions are capped this far below the optima peak.
        self._cap_high = _PEAK * (1.0 - 2.0 * self.optima_margin)

        # Estimate the raw range for the symmetric [0, 1] map.  Include the true
        # centers so the analytic peak anchors the top of the range.
        rng = np.random.default_rng(12345)
        samples = rng.dirichlet(np.ones(dim), size=_RANGE_SAMPLES)
        if len(self.peak_centers):
            samples = np.vstack([samples, self.peak_centers])
        final = self._predict_raw(samples)
        self._raw_min = float(final.min())
        self._raw_max = float(final.max())
        # Symmetric scale around the neutral level (raw 0 -> output 0.5).
        self._scale = max(abs(self._raw_min), abs(self._raw_max), 1e-12)

    # ── Placement helpers ───────────────────────────────────────────────────

    def _sample_simplex(self, n: int, seed_offset: int) -> np.ndarray:
        if n <= 0:
            return np.empty((0, self.dim), dtype=float)
        rng = np.random.default_rng(self.seed + seed_offset)
        return rng.dirichlet(np.ones(self.dim), size=n)

    def _region_targets(self, n: int, region: str, rng, spread: float = 0.0) -> np.ndarray:
        """``n`` points on/near ``region`` of the simplex (used as cluster seeds).

        ``spread`` in ``[0, 1]`` controls how tightly each seed hugs the region:
        every seed is pulled toward the simplex centroid by a random fraction drawn
        in ``[0, spread]`` (0 = sit exactly on the corner/edge/face; 1 = may drift
        all the way in to the centroid).  Each seed draws its own pull, so within
        one cluster layout some optima can hug the feature while others sit looser.
        """
        d = self.dim
        eps = 1e-3
        centroid = np.full(d, 1.0 / d)
        spread = float(np.clip(spread, 0.0, 1.0))
        out = np.empty((n, d), dtype=float)
        for i in range(n):
            if region == "middle":
                t = np.full(d, 1.0 / d)
            elif region == "corners":
                t = np.zeros(d)
                t[rng.integers(d)] = 1.0
            elif region == "edges":
                a, b = rng.choice(d, size=2, replace=False)
                w = rng.random()
                t = np.zeros(d)
                t[a], t[b] = w, 1.0 - w
            elif region == "faces":
                # A point on a random facet {x_j = 0}: Dirichlet on the rest.
                j = rng.integers(d)
                t = rng.dirichlet(np.ones(d))
                t[j] = 0.0
                s = t.sum()
                t = t / s if s > 1e-12 else np.full(d, 1.0 / d)
            else:  # scatter / unknown
                t = rng.dirichlet(np.ones(d))
            if spread > 0.0:
                pull = float(rng.random()) * spread
                t = (1.0 - pull) * t + pull * centroid
            out[i] = t + eps
        return out

    def _sample_optima(self, n: int) -> np.ndarray:
        """True-optima centers: uniform ("scatter") or clustered around a region."""
        if n <= 0:
            return np.empty((0, self.dim), dtype=float)
        rng = np.random.default_rng(self.seed + _SEED_OPTIMA)
        if self.optima_layout not in REGION_MODES:
            return rng.dirichlet(np.ones(self.dim), size=n)
        # Clustered placement: seed a few cluster centers in the chosen region,
        # then draw each optimum tightly around a randomly assigned cluster.
        crng = np.random.default_rng(self.seed + _SEED_CLUSTERS)
        n_clusters = min(self.n_optima_clusters, n)
        seeds = self._region_targets(
            n_clusters, self.optima_layout, crng, self.optima_cluster_spread)
        assign = rng.integers(n_clusters, size=n)
        centers = np.empty((n, self.dim), dtype=float)
        for k in range(n):
            alpha = self.optima_cluster_conc * seeds[assign[k]]
            centers[k] = rng.dirichlet(alpha)
        return centers

    def _tag_true_optima(self, centers: np.ndarray, min_dist: float) -> np.ndarray:
        """Boolean mask over ``centers`` marking which count as *true* optima.

        Walks the basins in placement order and tags one greedily as a true
        optimum only if it sits at least ``min_dist`` (composition L2) from every
        already-tagged optimum.  Basins closer than that to an existing optimum
        stay on the landscape (they are still real peaks) but are left untagged,
        so the advertised optima are guaranteed mutually >= ``min_dist`` apart.
        ``min_dist <= 0`` tags every basin.
        """
        n = centers.shape[0]
        if n == 0:
            return np.zeros(0, dtype=bool)
        if min_dist <= 0.0:
            return np.ones(n, dtype=bool)
        keep = np.zeros(n, dtype=bool)
        kept: list[np.ndarray] = []
        for i in range(n):
            c = centers[i]
            if all(np.linalg.norm(c - k) >= min_dist for k in kept):
                keep[i] = True
                kept.append(c)
        return keep

    def _sample_ridges(self, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """``n`` segments of controlled length: a random midpoint plus a random
        tangent direction, endpoints offset by ``ridge_length`` and reprojected
        onto the simplex."""
        if n <= 0:
            return []
        rng = np.random.default_rng(self.seed + _SEED_RIDGES)
        half = 0.5 * self.ridge_length
        ridges = []
        for _ in range(n):
            m = rng.dirichlet(np.ones(self.dim))
            d = rng.standard_normal(self.dim)
            d -= d.mean()  # tangent to the simplex (sum-zero)
            nrm = float(np.linalg.norm(d))
            if nrm < 1e-9:
                p = q = m
            else:
                d /= nrm
                p = _project_simplex(m - half * d)
                q = _project_simplex(m + half * d)
            ridges.append((p, q))
        return ridges

    # ── Field components (raw units) ────────────────────────────────────────

    def _true_field(self, X: np.ndarray) -> np.ndarray:
        """Clean true-optima field: ``_PEAK`` at each optimum, ``_BASE`` far away."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if not len(self.peak_centers):
            return np.full(X.shape[0], _BASE)
        env = np.stack(
            [_negated_ackley_env01(X, c, self.basin_width, self.axis_scale)
             for c in self.peak_centers],
            axis=0,
        ).max(axis=0)
        return _BASE + (_PEAK - _BASE) * env

    def _region_proximity(self, X: np.ndarray, region: str, reach: float) -> np.ndarray:
        """Smooth ``[0, 1]`` proximity to ``region`` (1 at the region, 0 beyond
        ``reach``)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if region == "faces":
            # Distance to the simplex boundary = smallest coordinate.
            d = X.min(axis=1)
            return _smoothstep(1.0 - d / reach)
        Xs = X * self.axis_scale.reshape(1, -1)
        if region == "middle":
            c = (np.full(self.dim, 1.0 / self.dim) * self.axis_scale).reshape(1, -1)
            d = np.linalg.norm(Xs - c, axis=1)
        elif region == "corners":
            V = np.eye(self.dim) * self.axis_scale.reshape(1, -1)
            d = np.min([np.linalg.norm(Xs - v.reshape(1, -1), axis=1) for v in V], axis=0)
        elif region == "edges":
            verts = np.eye(self.dim) * self.axis_scale.reshape(1, -1)
            dists = []
            for a in range(self.dim):
                for b in range(a + 1, self.dim):
                    dists.append(_point_segment_dist(Xs, verts[a], verts[b]))
            d = np.min(dists, axis=0)
        else:
            return np.zeros(X.shape[0])
        return _smoothstep(1.0 - d / reach)

    def _background_field(self, X: np.ndarray) -> np.ndarray:
        """Combined *signed* distractor field, centred on the neutral level 0
        (weak optima, ridges, edge bias, roughness, plateaus)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        g = np.zeros(X.shape[0])

        # Weak optima — shorter negated-Ackley bumps/pits.  Combine the raises by
        # max and the digs by min, then add, so each stays bounded by its amp.
        if self.weak_amp > 0 and len(self.weak_centers):
            up = np.zeros(X.shape[0])
            down = np.zeros(X.shape[0])
            for c, s in zip(self.weak_centers, self.weak_signs):
                env = _negated_ackley_env01(X, c, self.weak_width, self.axis_scale)
                contrib = self.weak_amp * _SPAN * env
                if s > 0:
                    up = np.maximum(up, contrib)
                else:
                    down = np.minimum(down, -contrib)
            g = g + up + down

        # Ridges — Gaussian tubes/trenches around segments.
        if self.ridge_amp > 0 and self.ridges:
            Xs = X * self.axis_scale.reshape(1, -1)
            up = np.zeros(X.shape[0])
            down = np.zeros(X.shape[0])
            for (p, q), s in zip(self.ridges, self.ridge_signs):
                dseg = _point_segment_dist(Xs, p * self.axis_scale, q * self.axis_scale)
                gauss = np.exp(-(dseg / self.ridge_width) ** 2)
                contrib = self.ridge_amp * _SPAN * gauss
                if s > 0:
                    up = np.maximum(up, contrib)
                else:
                    down = np.minimum(down, -contrib)
            g = g + up + down

        # Edge / structural bias — smooth signed trend toward a region.
        if self.edge_region is not None and self.edge_amp > 0:
            prox = self._region_proximity(X, self.edge_region, self.edge_reach)
            g = g + self.edge_sign * self.edge_amp * _SPAN * prox

        # Roughness — coherent simplex noise (already ~zero-mean) added on top.
        if self.noise_amp > 0:
            g = g + simplex_noise(
                X, frequency=self.noise_freq, amplitude=self.noise_amp,
                octaves=self.noise_octaves, seed=self.seed + _SEED_NOISE,
            )

        # Plateaus — flatten regions toward a constant (signed) mesa/basin level.
        if len(self.plateau_centers):
            Xs = X * self.axis_scale.reshape(1, -1)
            for c, s in zip(self.plateau_centers, self.plateau_signs):
                level = s * self.plateau_amp * _SPAN
                d = np.linalg.norm(Xs - (c * self.axis_scale).reshape(1, -1), axis=1)
                w = _smoothstep(1.0 - d / self.plateau_radius)
                g = (1.0 - w) * g + w * level

        return g

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Raw objective: background's upward excursions capped below the optima
        peak, then the clean true-optima field laid on top."""
        true = self._true_field(X)
        bg = np.minimum(self._background_field(X), self._cap_high)
        return np.maximum(true, bg)

    def components(self, X: np.ndarray) -> dict:
        """Raw field pieces, for inspection/plotting (not used by ``predict``)."""
        return {
            "true_optima": self._true_field(X),
            "background": np.minimum(self._background_field(X), self._cap_high),
            "combined": self._predict_raw(X),
        }

    # ── Public API (matches Ackley) ─────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self._predict_raw(X)
        # Symmetric map to [0, 1] centred on the neutral level (raw 0 -> 0.5).
        y = np.clip(0.5 + 0.5 * raw / self._scale, 0.0, 1.0)
        # Squish the full [0, 1] span into the upper half [0.5, 1].  This
        # deliberately deflates the optima margin (everything is compressed by
        # 2x) so the surface only ever ranges across the top half of the output.
        return 0.5 + 0.5 * y

    def __call__(self, x: np.ndarray) -> float:
        return float(self.predict(np.asarray(x, dtype=float).reshape(1, -1))[0])

    @property
    def known_maxima(self) -> list[tuple[np.ndarray, float]]:
        return [(c.copy(), float(self.predict(c.reshape(1, -1))[0])) for c in self.centers]

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Ensemble(dim={self.dim}, n_optima={len(self.centers)}"
                f"/{len(self.peak_centers)} true/peaks, "
                f"layout={self.optima_layout!r}, n_weak={len(self.weak_centers)}, "
                f"n_ridges={len(self.ridges)}, n_plateaus={len(self.plateau_centers)}, "
                f"edge_region={self.edge_region!r}, seed={self.seed})")
