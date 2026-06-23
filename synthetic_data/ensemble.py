"""
ensemble.py
===========
A layered, "realistic" synthetic objective on the probability simplex.

Where ``ackley.py`` gives a (negated) Ackley field plus optional background
noise, this module stacks several independent landscape *features* into a
single objective so the benchmark looks more like a real materials-composition
response surface:

    * **Optima**          — strong negated-Ackley basins; the *true* maxima.
    * **Roughness**       — coherent Perlin-style simplex noise (fine ripple).
    * **Weak optima**     — extra, shorter negated-Ackley basins acting as
                            distractors / fake local optima.
    * **Ridges**          — high-value tubes around random line segments.
    * **Anisotropy**      — per-axis stretching of the distance metric, so
                            basins/ridges are elongated along some axes.
    * **Plateaus**        — regions flattened to a constant mesa height.

Unlike ``ackley.py`` this module deliberately avoids any hardcoded
"parameter-by-dimension" tables: every knob is a plain scalar (or a count) that
means the same thing at any ``dim``.  Random placement of every feature is
driven by a single ``seed``.

Guaranteed dominance of the true optima
---------------------------------------
The defining contract of this objective is that **the true optima are always
the strict global maxima**, by a tunable margin, *no matter* how the other
features are configured.  Concretely, let ``T(x)`` be the clean true-optima
field (value 0 at each optimum, negative elsewhere) and ``G(x)`` the combined
*background* (weak optima + ridges + roughness + plateaus).  The raw objective is

    F(x) = max( T(x),  min( G(x), cap_raw ) )

The background is capped *before* the true-optima field is laid on top, so no
amount of stretching, plateau insertion or noise can push a distractor up to or
past a real optimum.

``optima_margin`` is expressed in the **normalized output units** (the same
``[0.5, 1]`` scale ``predict`` returns), in ``[0, 0.5]``: the optima sit at 1.0
and the entire background is guaranteed to sit at or below ``1 - optima_margin``.
Because the ``[0.5, 1]`` map is monotonic in the raw signal with the peak pinned
at raw 0, that normalized cap corresponds to the raw cap
``cap_raw = 2 * optima_margin * raw_min`` (and ``raw_min`` is unaffected by the
cap, since capping only lowers the *top* of the background, never the landscape
floor — so there is no circularity).

(The guarantee is about the background / distractor field.  The true optima's
*own* basins of course rise continuously to the peak, so points inside a real
basin are legitimately close to it — those are the optimum, not competitors.)

The ``Ensemble`` class exposes ``predict(X)`` with the scikit-learn-style
``(N, d) -> (N,)`` signature, plus ``known_maxima`` and ``centers`` like
``Ackley``.

Example
-------
    from synthetic_data.ensemble import Ensemble

    fn = Ensemble(dim=4, n_optima=4, n_weak=8, n_ridges=2, n_plateaus=2)
    y = fn.predict(X)                       # (N,) in [0.5, 1]
    for c, v in fn.known_maxima:            # each v == 1.0 (the global max)
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

# Raw "far-field floor": the value a negated-Ackley basin asymptotes to far from
# its center (``-scale * a``).  Background features are expressed as a fraction
# of the span between this floor and the peak value of 0.
_BASE = -ACKLEY_SCALE * ACKLEY_A
_SPAN = 0.0 - _BASE  # == ACKLEY_SCALE * ACKLEY_A
_RANGE_SAMPLES = 100_000  # simplex samples used to estimate raw min/max for the [0.5, 1] map

# Per-feature seed offsets so one master ``seed`` deterministically drives every
# random placement without the features sharing a stream.
# Anisotropy: max log-stretch grows as ``log1p(ANISO_RATE * strength)`` so the
# stretched axes widen by ~4x at strength 10 and ~16x at strength 50.
ANISO_RATE = 0.3

_SEED_OPTIMA = 0
_SEED_WEAK = 101
_SEED_RIDGES = 202
_SEED_PLATEAUS = 303
_SEED_ANISO = 404
_SEED_NOISE = 505


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


def random_ensemble_config(dim: int, rng, *, optima_margin: float = 0.1) -> dict:
    """Draw a random :class:`Ensemble` configuration for one run.

    Mirrors the "Randomize" button in ``synthetic_data/plot_ensemble.py`` — the
    same per-feature ranges and the same on/off toggles (a disabled feature passes
    its count/amplitude as 0) — and additionally draws a random master ``seed`` so
    each call also relocates every feature.  ``n_optima`` is drawn from the
    dimension-specific :func:`optima_count_range`; ``optima_margin`` is held fixed
    (the viewer does not randomize it either).  Returns a kwargs dict accepted
    directly by ``Ensemble(**config)``, so a saved config exactly recreates the
    landscape.

    ``rng`` is any object with the ``random.Random`` interface; seed it
    deterministically (e.g. ``random.Random(f"{master_seed}-{trial}")``) for a
    reproducible per-trial landscape sequence.
    """
    opt_lo, opt_hi = optima_count_range(int(dim))
    toggle = lambda: rng.random() > 0.5  # noqa: E731
    weak_on, ridges_on, rough_on, aniso_on, plateaus_on = (
        toggle(), toggle(), toggle(), toggle(), toggle())
    return {
        "dim": int(dim),
        "n_optima": rng.randint(opt_lo, opt_hi),
        "basin_width": float(rng.randint(40, 90)),
        "optima_margin": float(optima_margin),
        "n_weak": rng.randint(0, 30) if weak_on else 0,
        "weak_width": float(rng.randint(5, 300)),
        "weak_amp": round(rng.uniform(0.0, 1.0), 2),
        "n_ridges": rng.randint(0, 8) if ridges_on else 0,
        "ridge_width": round(rng.uniform(0.01, 0.25), 3),
        "ridge_amp": round(rng.uniform(0.0, 1.0), 2),
        "noise_freq": round(rng.uniform(0.0, 40.0), 1),
        "noise_amp": float(rng.randint(0, 2000)) if rough_on else 0.0,
        "noise_octaves": rng.randint(1, 6),
        "aniso_strength": round(rng.uniform(0.0, 50.0), 1) if aniso_on else 0.0,
        "n_plateaus": rng.randint(0, 8) if plateaus_on else 0,
        "plateau_radius": round(rng.uniform(0.02, 0.40), 2),
        "plateau_amp": round(rng.uniform(0.0, 1.0), 2),
        "seed": rng.randint(0, 10_000),
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


def _negated_ackley_env(X: np.ndarray, center: np.ndarray, b: float,
                        axis_scale: np.ndarray) -> np.ndarray:
    """Anisotropic negated-Ackley envelope in ``[_BASE, 0]`` (0 at ``center``)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    center = np.asarray(center, dtype=float).reshape(1, -1)
    delta = (X - center) * axis_scale.reshape(1, -1)
    d_eff = delta.shape[1]
    t1 = -ACKLEY_A * np.exp(-b * np.sqrt(np.sum(delta ** 2, axis=1) / d_eff))
    return -ACKLEY_SCALE * (t1 + ACKLEY_A)


def _point_segment_dist(X: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Euclidean distance from each row of ``X`` to the segment ``p``–``q``."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    p = np.asarray(p, dtype=float).reshape(1, -1)
    q = np.asarray(q, dtype=float).reshape(1, -1)
    pq = q - p
    l2 = float(pq @ pq.T)
    if l2 < 1e-18:
        return np.linalg.norm(X - p, axis=1)
    t = np.clip(((X - p) @ pq.T).ravel() / l2, 0.0, 1.0)
    proj = p + t.reshape(-1, 1) * pq
    return np.linalg.norm(X - proj, axis=1)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ── Ensemble objective ───────────────────────────────────────────────────────

class Ensemble:
    """A layered synthetic objective on the ``dim``-element probability simplex.

    Parameters
    ----------
    dim : int
        Simplex dimensionality (>= 2).
    n_optima, basin_width : true (strong) optima count and Ackley sharpness ``b``.
    optima_margin : float
        Normalized gap in ``[0, 0.5]`` guaranteeing the whole background stays at
        or below ``1 - optima_margin`` while the optima sit at 1.0 (see module
        docstring).
    n_weak, weak_width, weak_amp : weak-optima (distractor) count, sharpness, and
        prominence in ``[0, 1]`` (fraction of the full basin height).
    n_ridges, ridge_width, ridge_amp : ridge count, tube width (simplex units),
        and prominence in ``[0, 1]``.
    noise_freq, noise_amp, noise_octaves : background Perlin-noise controls.
    aniso_strength : per-axis stretching of the distance metric (0 = isotropic).
    n_plateaus, plateau_radius, plateau_amp : plateau count, radius (simplex
        units) and mesa height in ``[0, 1]``.
    seed : single master seed driving every random placement.
    """

    def __init__(
        self,
        dim: int = 3,
        *,
        # true optima
        n_optima: int = 4,
        basin_width: float = 65.0,
        optima_margin: float = 0.1,
        # weak optima (distractors)
        n_weak: int = 6,
        weak_width: float = 120.0,
        weak_amp: float = 0.6,
        # ridges
        n_ridges: int = 2,
        ridge_width: float = 0.06,
        ridge_amp: float = 0.6,
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
        # global
        seed: int = 0,
    ) -> None:
        if dim < 2:
            raise ValueError(f"dim must be >= 2 (got {dim}).")
        self.dim = dim
        self.seed = int(seed)

        self.optima_margin = float(np.clip(optima_margin, 0.0, 0.5))
        self.basin_width = float(basin_width)
        self.weak_width = float(weak_width)
        self.weak_amp = float(np.clip(weak_amp, 0.0, 1.0))
        self.ridge_width = max(1e-4, float(ridge_width))
        self.ridge_amp = float(np.clip(ridge_amp, 0.0, 1.0))
        self.noise_freq = float(noise_freq)
        self.noise_amp = max(0.0, float(noise_amp))
        self.noise_octaves = int(noise_octaves)
        self.plateau_radius = max(1e-4, float(plateau_radius))
        self.plateau_amp = float(np.clip(plateau_amp, 0.0, 1.0))

        # Per-axis stretch shared by every distance-based feature.
        self.axis_scale = _anisotropy_scale(dim, float(aniso_strength), self.seed + _SEED_ANISO)

        # Random placement of every feature (Dirichlet on the simplex).
        self.centers = self._sample_simplex(int(n_optima), _SEED_OPTIMA)
        self.weak_centers = self._sample_simplex(int(n_weak), _SEED_WEAK)
        self.plateau_centers = self._sample_simplex(int(n_plateaus), _SEED_PLATEAUS)
        # Ridges: each is a segment between two distinct random simplex points.
        ridge_pts = self._sample_simplex(2 * int(n_ridges), _SEED_RIDGES)
        self.ridges = [(ridge_pts[2 * i], ridge_pts[2 * i + 1]) for i in range(int(n_ridges))]

        # Estimate the raw range for the [0.5, 1] map.  Include the true centers
        # so the analytic peak (raw 0) anchors the top of the range exactly.
        rng = np.random.default_rng(12345)
        samples = rng.dirichlet(np.ones(dim), size=_RANGE_SAMPLES)
        if len(self.centers):
            samples = np.vstack([samples, self.centers])
        true = self._true_field(samples)
        bg = self._background_field(samples)
        # raw_min comes from the *uncapped* landscape (the cap only lowers the top
        # of the background, never the floor), so it is consistent to derive the
        # raw cap from it.  ``optima_margin`` is normalized: a value m means the
        # background top maps to (1 - m) on the final [0.5, 1] scale, i.e. the raw
        # cap is ``2 * m * raw_min`` (raw peak is pinned at 0).
        self._raw_min = float(np.maximum(true, bg).min())
        self._cap_raw = 2.0 * self.optima_margin * self._raw_min
        final = np.maximum(true, np.minimum(bg, self._cap_raw))
        self._raw_max = float(final.max())

    def _sample_simplex(self, n: int, seed_offset: int) -> np.ndarray:
        if n <= 0:
            return np.empty((0, self.dim), dtype=float)
        rng = np.random.default_rng(self.seed + seed_offset)
        return rng.dirichlet(np.ones(self.dim), size=n)

    # ── Field components (raw units) ────────────────────────────────────────

    def _true_field(self, X: np.ndarray) -> np.ndarray:
        """Clean true-optima field: 0 at each optimum, ``_BASE`` far away."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if not len(self.centers):
            return np.full(X.shape[0], _BASE)
        terms = np.stack(
            [_negated_ackley_env(X, c, self.basin_width, self.axis_scale)
             for c in self.centers],
            axis=0,
        )
        return terms.max(axis=0)

    def _background_field(self, X: np.ndarray) -> np.ndarray:
        """Combined distractor field (weak optima, ridges, roughness, plateaus)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        g = np.full(X.shape[0], _BASE)

        # Weak optima — shorter negated-Ackley basins, max-combined.
        for c in self.weak_centers:
            env = (_negated_ackley_env(X, c, self.weak_width, self.axis_scale) - _BASE) / _SPAN
            g = np.maximum(g, _BASE + self.weak_amp * _SPAN * env)

        # Ridges — Gaussian tubes around segments, max-combined.
        if self.ridge_amp > 0:
            Xs = X * self.axis_scale.reshape(1, -1)
            for p, q in self.ridges:
                d = _point_segment_dist(Xs, p * self.axis_scale, q * self.axis_scale)
                gauss = np.exp(-(d / self.ridge_width) ** 2)
                g = np.maximum(g, _BASE + self.ridge_amp * _SPAN * gauss)

        # Roughness — coherent simplex noise added on top.
        if self.noise_amp > 0:
            g = g + simplex_noise(
                X, frequency=self.noise_freq, amplitude=self.noise_amp,
                octaves=self.noise_octaves, seed=self.seed + _SEED_NOISE,
            )

        # Plateaus — flatten regions toward a constant mesa level.
        if len(self.plateau_centers):
            Xs = X * self.axis_scale.reshape(1, -1)
            level = _BASE + self.plateau_amp * _SPAN
            for c in self.plateau_centers:
                d = np.linalg.norm(Xs - (c * self.axis_scale).reshape(1, -1), axis=1)
                w = _smoothstep(1.0 - d / self.plateau_radius)
                g = (1.0 - w) * g + w * level

        return g

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Raw objective: background capped ``optima_margin`` (normalized) below
        the peak, then the clean true-optima field laid on top."""
        true = self._true_field(X)
        bg = np.minimum(self._background_field(X), self._cap_raw)
        return np.maximum(true, bg)

    def components(self, X: np.ndarray) -> dict:
        """Raw field pieces, for inspection/plotting (not used by ``predict``)."""
        return {
            "true_optima": self._true_field(X),
            "background": np.minimum(self._background_field(X), self._cap_raw),
            "combined": self._predict_raw(X),
        }

    # ── Public API (matches Ackley) ─────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self._predict_raw(X)
        span = self._raw_max - self._raw_min
        if span < 1e-12:
            return np.full(raw.shape, 0.75)
        y = 0.5 + 0.5 * (raw - self._raw_min) / span
        return np.clip(y, 0.5, 1.0)

    def __call__(self, x: np.ndarray) -> float:
        return float(self.predict(np.asarray(x, dtype=float).reshape(1, -1))[0])

    @property
    def known_maxima(self) -> list[tuple[np.ndarray, float]]:
        return [(c.copy(), float(self.predict(c.reshape(1, -1))[0])) for c in self.centers]

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Ensemble(dim={self.dim}, n_optima={len(self.centers)}, "
                f"n_weak={len(self.weak_centers)}, n_ridges={len(self.ridges)}, "
                f"n_plateaus={len(self.plateau_centers)}, seed={self.seed})")
