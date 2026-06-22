"""
ackley.py
=========
Negated Ackley test functions on the probability simplex, with optional
Perlin-style background noise for the "realistic" variant.

Five variants are provided, each a *negated* Ackley function whose global
maximum (value ~ 0) sits at a canonical simplex location:

    "centroid"    peak at [1/3, 1/3, 1/3]   (simplex centroid)
    "edge"        peak at [0.5, 0.5, 0]     (edge midpoint)
    "vertex"      peak at [1,   0,   0]     (simplex vertex)
    "multimodal"  sum of three skinnier-peaked Ackleys
    "realistic"   Dirichlet-sampled peaks + background simplex noise,
                  fully configurable via ``synthetic_data/defaults/ackley.json``

The ``Ackley`` class exposes a ``predict(X)`` method matching scikit-learn's
``RandomForestRegressor.predict`` signature ``(N, d) -> (N,)``.

For the "realistic" variant, ``Ackley("realistic")`` reads its parameters
(``n_optima``, ``basin_width``, ``noise_freq``, ``noise_amp``) from
``synthetic_data/defaults/ackley.json``.  You can override any of them via
keyword arguments::

    fn = Ackley("realistic", dim=3, n_optima=5, noise_amp=10.0)

The configured ``n_optima`` describes the **3-simplex (d=3)**; when it is *not*
overridden, the number of optima is scaled with dimension by ``(d-1)/2`` (see
``scaled_n_optima``) so higher-d benchmarks have proportionally more optima.
Passing ``n_optima`` explicitly disables this scaling and uses that exact count.

All other variants ignore these keyword arguments and behave exactly as
before.

Example
-------
    from synthetic_data.ackley import Ackley

    fn = Ackley("centroid")
    y = fn.predict(np.array([[1/3, 1/3, 1/3]]))   # ~ [0.0]

    fn = Ackley("realistic")             # reads from synthetic_data/defaults/ackley.json
    fn = Ackley("realistic", n_optima=5) # override one param
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ── Config file for tunable defaults ─────────────────────────────────────────
_CONFIGS_DIR = Path(__file__).resolve().parent / "defaults"
_DEFAULTS_PATH = _CONFIGS_DIR / "ackley.json"

SCALE_OPTIMA_WITH_DIM = True

# The classic Ackley function has two parts: a smooth Gaussian-like envelope
# (``t1``) that forms the basin, and an oscillatory cosine term (``t2``) that
# studs that basin with fine local ripples / minima.  When this is False the
# cosine term is dropped entirely, so each peak is a single smooth, monotonic
# basin with no ripples — the envelope alone.
USE_OSCILLATION = False

# Hardcoded Ackley sharpness ``b`` (basin width) per simplex dimensionality.
# When the "realistic" variant is built without an explicit ``basin_width``
# override, the value for its ``dim`` is taken from here; dims not listed fall
# back to the config ``basin_width``.
BASIN_WIDTH_BY_DIM = {
    3: 86.0,
    4: 65.0,
    10: 20,
}

# Hardcoded background-noise amplitude per simplex dimensionality, tuned by hand.
# When the "realistic" variant is built without an explicit ``noise_amp``
# override, the value for its ``dim`` is taken from here; dims not listed fall
# back to the config ``noise_amp``.  (Interactive sliders pass noise_amp
# explicitly, so they still override this.)
NOISE_AMP_BY_DIM = {
    3: 400.0,
    4: 300.0,
    10: 20.0,
}

_HARDCODED_DEFAULTS = {
    "n_optima": 10,
    "basin_width": 50,
    "intensity_mean": 0.0,
    "intensity_var": 0.0,
    "noise_freq": 8.0,
    "noise_amp": 5.0,
}


def load_config() -> dict:
    """Load tunable defaults from ``synthetic_data/defaults/ackley.json``."""
    if _DEFAULTS_PATH.exists():
        with open(_DEFAULTS_PATH) as f:
            return json.load(f)
    return dict(_HARDCODED_DEFAULTS)


def save_config(cfg: dict) -> None:
    """Write tunable defaults to ``synthetic_data/defaults/ackley.json``."""
    _CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_DEFAULTS_PATH, "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")


def scaled_n_optima(n_base: int, dim: int) -> int:
    """Number of "realistic" optima for ``dim``, scaling a 3-simplex baseline.

    The configured ``n_optima`` describes the **3-simplex (d=3)**.  The count of
    distinct optima then grows **linearly** with the tangent dimension by the
    factor ``(d-1)/2`` — exactly the baseline at d=3, ~1.5× at d=4, 4.5× at d=10.

    Why linear: the room for distinct basins lives in the ``(d-1)``-dimensional
    simplex, so it expands with dimension; true geometric growth is exponential,
    but that is both unrealistic for materials composition spaces — which are
    mostly phase-separated, not densely packed with stable phases — and
    undiscoverable as a benchmark.  The linear ``(d-1)/2`` factor keeps the
    benchmark difficulty growing gently and predictably with dimension.

    This is intentionally a *plain* function of ``dim`` (the benchmark's optima
    count), so a given configuration yields the identical objective every run.
    """
    return max(1, int(round(n_base * (dim - 1) / 2.0)))


def scaled_n_optima_multiplicative(n_base: int, dim: int, *, ref_dim: int = 3) -> int:
    """Peak count scaling proportional to ``dim / ref_dim`` (multiplicative in d).

    Alternative to ``scaled_n_optima`` (linear in ``(d-1)/2``).  At the default
    ``ref_dim=3`` baseline this gives 20 → 27 → 67 peaks for d = 3, 4, 10 when
    ``n_base=20``, vs 20 → 30 → 90 under linear scaling.
    """
    if ref_dim < 1:
        raise ValueError("ref_dim must be >= 1")
    return max(1, int(round(n_base * dim / ref_dim)))


def resolve_scaled_n_optima(
    n_base: int,
    dim: int,
    *,
    mode: str = "linear",
    ref_dim: int = 3,
) -> int:
    if mode == "linear":
        return scaled_n_optima(n_base, dim)
    if mode == "multiplicative":
        return scaled_n_optima_multiplicative(n_base, dim, ref_dim=ref_dim)
    raise ValueError(f"Unknown peak scaling mode {mode!r}; use 'linear' or 'multiplicative'.")

# ── Ackley constants ─────────────────────────────────────────────────────────
ACKLEY_A = 20.0
ACKLEY_B = 0.2
ACKLEY_B_SKINNY = 1.2
ACKLEY_C = 2.0 * np.pi
ACKLEY_SCALE = 30.0
_RANGE_SAMPLES = 100_000  # simplex samples used to estimate raw min/max for [0.5, 1] scaling


# ── Dimension-general variant helpers ────────────────────────────────────────

def _uniform_center(dim: int) -> np.ndarray:
    return np.full(dim, 1.0 / dim)


def _edge_center(dim: int) -> np.ndarray:
    c = np.zeros(dim)
    c[0] = c[1] = 0.5
    return c


def _vertex_center(dim: int) -> np.ndarray:
    c = np.zeros(dim)
    c[0] = 1.0
    return c


_VARIANT_SPECS = {
    "centroid":   (lambda dim: [(_uniform_center(dim), ACKLEY_B)], "sum"),
    "edge":       (lambda dim: [(_edge_center(dim), ACKLEY_B)], "sum"),
    "vertex":     (lambda dim: [(_vertex_center(dim), ACKLEY_B)], "sum"),
    "multimodal": (lambda dim: [(_uniform_center(dim), ACKLEY_B_SKINNY),
                                (_edge_center(dim), ACKLEY_B_SKINNY),
                                (_vertex_center(dim), ACKLEY_B_SKINNY)], "sum"),
    "realistic":  None,  # handled specially inside Ackley.__init__
}

VARIANTS = tuple(_VARIANT_SPECS)

# ── Backwards-compatible 3-simplex names (re-exported by run_zombi_test.py) ──
CENTER_CENTROID = _uniform_center(3)
CENTER_EDGE     = _edge_center(3)
CENTER_VERTEX   = _vertex_center(3)
MULTIMODAL_CENTERS = [CENTER_CENTROID, CENTER_EDGE, CENTER_VERTEX]


def _realistic_peaks(dim: int) -> list[tuple[np.ndarray, float]]:
    """Legacy helper kept for REALISTIC_PEAKS constant."""
    from synthetic_data.ackley import load_config as _lc
    cfg = _lc()
    n_optima = int(cfg["n_optima"])
    basin_width = float(cfg["basin_width"])
    rng = np.random.default_rng(0)
    centers = rng.dirichlet(np.ones(dim), size=n_optima)
    return [(np.asarray(c, dtype=float).copy(), basin_width) for c in centers]


REALISTIC_PEAKS = _realistic_peaks(3)


def _negated_ackley(
    X: np.ndarray,
    center: np.ndarray,
    *,
    a: float = ACKLEY_A,
    b: float = ACKLEY_B,
    c: float = ACKLEY_C,
    scale: float = ACKLEY_SCALE,
) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    center = np.asarray(center, dtype=float).reshape(1, -1)
    delta = X - center
    d_eff = delta.shape[1]
    t1 = -a * np.exp(-b * np.sqrt(np.sum(delta ** 2, axis=1) / d_eff))
    if USE_OSCILLATION:
        t2 = -np.exp(np.sum(np.cos(c * delta), axis=1) / d_eff)
        ackley = t1 + t2 + a + np.e
    else:
        # Drop the cosine ripple: pure smooth envelope, 0 at the center rising
        # to ``a`` far away (same peak value as the full form, no local minima).
        ackley = t1 + a
    return -scale * ackley


# ── Simplex noise ────────────────────────────────────────────────────────────

def simplex_noise(
    X: np.ndarray,
    *,
    frequency: float = 8.0,
    amplitude: float = 5.0,
    octaves: int = 4,
    persistence: float = 0.5,
    n_waves: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """Smooth coherent noise evaluated at simplex compositions."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    result = np.zeros(X.shape[0])
    amp = amplitude
    freq = frequency
    for _ in range(octaves):
        directions = rng.standard_normal((n_waves, d))
        phases = rng.uniform(0, 2.0 * np.pi, n_waves)
        proj = X @ directions.T * freq
        result += amp * np.sin(proj + phases).mean(axis=1)
        freq *= 2.0
        amp *= persistence
    return result


# ── Unified Ackley class ─────────────────────────────────────────────────────

class Ackley:
    """A negated Ackley objective on the ``dim``-element probability simplex.

    For the ``"realistic"`` variant, parameters are read from
    ``synthetic_data/defaults/ackley.json`` and can be overridden via kwargs.
    All other variants ignore the extra kwargs.

    Parameters
    ----------
    variant : str
        One of ``Ackley.VARIANTS``.
    dim : int
        Simplex dimensionality.
    n_optima, basin_width, noise_freq, noise_amp : optional
        Overrides for the "realistic" variant (ignored by others).
        Unspecified values are read from the config file.  When ``n_optima`` is
        left unspecified it is scaled with ``dim`` by ``(d-1)/2`` relative to the
        configured (d=3) value; passing it explicitly uses that exact count.
    intensity_mean, intensity_var : optional
        Mean and variance of a nonneg offset added to each optimum's basin
        width, producing varying peak intensities.  Sampled from a Gamma
        distribution (constant when ``intensity_var`` is 0).
    noise_octaves, noise_seed, peak_seed : optional
        Additional "realistic" controls with sensible defaults.
    """

    VARIANTS = VARIANTS

    def __init__(
        self,
        variant: str = "centroid",
        dim: int = 3,
        *,
        n_optima: int | None = None,
        basin_width: float | None = None,
        intensity_mean: float | None = None,
        intensity_var: float | None = None,
        noise_freq: float | None = None,
        noise_amp: float | None = None,
        noise_octaves: int = 4,
        noise_seed: int = 42,
        peak_seed: int = 0,
    ) -> None:
        variant = str(variant).strip().lower()
        if variant not in _VARIANT_SPECS:
            raise ValueError(
                f"Unknown Ackley variant {variant!r}; expected one of {VARIANTS}."
            )
        if dim < 2:
            raise ValueError(f"dim must be >= 2 (got {dim}).")
        self.variant = variant
        self.dim = dim

        if variant == "realistic":
            cfg = load_config()
            # An explicit n_optima is honoured exactly (e.g. the plot.py
            # sliders); otherwise the configured value is the d=3 baseline and is
            # scaled with dimension so higher-d benchmarks have proportionally
            # more optima (see ``scaled_n_optima``).
            if n_optima is not None:
                _n = int(n_optima)
            elif SCALE_OPTIMA_WITH_DIM:
                _n = scaled_n_optima(int(cfg["n_optima"]), dim)
            else:
                _n = int(cfg["n_optima"])
            # An explicit basin_width override wins; otherwise use the hardcoded
            # per-dim value, falling back to the config for dims not listed.
            if basin_width is not None:
                _b = float(basin_width)
            else:
                _b = float(BASIN_WIDTH_BY_DIM.get(dim, cfg["basin_width"]))
            _im = float(intensity_mean if intensity_mean is not None else cfg.get("intensity_mean", 0.0))
            _iv = float(intensity_var if intensity_var is not None else cfg.get("intensity_var", 0.0))
            _nf = float(noise_freq if noise_freq is not None else cfg["noise_freq"])
            # An explicit noise_amp override wins; otherwise use the hardcoded
            # per-dim value, falling back to the config for dims not listed.
            if noise_amp is not None:
                _na = float(noise_amp)
            else:
                _na = float(NOISE_AMP_BY_DIM.get(dim, cfg["noise_amp"]))

            rng = np.random.default_rng(peak_seed)
            self.centers = [c.copy() for c in rng.dirichlet(np.ones(dim), size=_n)]
            if _iv > 0 and _im > 0:
                shape = _im ** 2 / _iv
                scale = _iv / _im
                offsets = rng.gamma(shape, scale, size=_n)
            else:
                offsets = np.full(_n, _im)
            self.basin_widths = [max(1.0, _b - float(o)) for o in offsets]
            self.combine = "max"

            self._noise_freq = _nf
            self._noise_amp = _na
            self._noise_octaves = noise_octaves
            self._noise_seed = noise_seed
        else:
            peaks_builder, combine = _VARIANT_SPECS[variant]
            peaks = peaks_builder(dim)
            self.centers = [c.copy() for c, _ in peaks]
            self.basin_widths = [b for _, b in peaks]
            self.combine = combine

            self._noise_freq = 0.0
            self._noise_amp = 0.0
            self._noise_octaves = 0
            self._noise_seed = 0

        # For dims above 3 the [0.5, 1] map is fixed from the *noise-free* Ackley
        # signal (peak -> 1, far-field floor -> 0.5); noise (level set per-dim via
        # NOISE_AMP_BY_DIM) is then added through that same map and the result is
        # clipped to [0.5, 1].  At dim <= 3 the Ackley+noise signal is normalized
        # together and left unclamped, as before.
        self._clip_to_unit = self.dim > 3

        _est_rng = np.random.default_rng(12345)
        _samples = _est_rng.dirichlet(np.ones(dim), size=_RANGE_SAMPLES)
        # Include the peak centers (the analytic maxima) so a sample-only max
        # doesn't underestimate the peak in high dim.
        if self.centers:
            _samples = np.vstack([_samples, np.asarray(self.centers, dtype=float)])
        # dim > 3 normalizes on the noise-free Ackley span; dim <= 3 on Ackley+noise.
        _raw = self._ackley_raw(_samples) if self._clip_to_unit else self._predict_raw(_samples)
        self._raw_min = float(_raw.min())
        self._raw_max = float(_raw.max())

    def _ackley_raw(self, X: np.ndarray) -> np.ndarray:
        """Noise-free negated-Ackley signal, combined over the peaks."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        terms = np.stack(
            [_negated_ackley(X, center, b=b)
             for center, b in zip(self.centers, self.basin_widths)],
            axis=0,
        )
        return terms.max(axis=0) if self.combine == "max" else terms.sum(axis=0)

    def _noise_raw(self, X: np.ndarray) -> np.ndarray:
        """Background simplex-noise field in raw units (0 when noise is off)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self._noise_amp <= 0:
            return np.zeros(X.shape[0])
        return simplex_noise(
            X,
            frequency=self._noise_freq,
            amplitude=self._noise_amp,
            octaves=self._noise_octaves,
            seed=self._noise_seed,
        )

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        return self._ackley_raw(X) + self._noise_raw(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self._predict_raw(X)
        span = self._raw_max - self._raw_min
        if span < 1e-12:
            return np.full(raw.shape, 0.75)
        y = 0.5 + 0.5 * (raw - self._raw_min) / span
        if self._clip_to_unit:
            # dim > 3: keep the (noisy) objective within [0.5, 1] -- the span is
            # the noise-free Ackley range, so noise can push a value past it.
            y = np.clip(y, 0.5, 1.0)
        return y

    def __call__(self, x: np.ndarray) -> float:
        return float(self.predict(np.asarray(x, dtype=float).reshape(1, -1))[0])

    @property
    def known_maxima(self) -> list[tuple[np.ndarray, float]]:
        return [(c.copy(), float(self.predict(c.reshape(1, -1))[0])) for c in self.centers]

    def __repr__(self) -> str:  # pragma: no cover
        return f"Ackley(variant={self.variant!r})"


# Backwards-compatible alias
TunableRealistic = Ackley
