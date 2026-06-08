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
                  fully configurable via ``configs/defaults.json``

The ``Ackley`` class exposes a ``predict(X)`` method matching scikit-learn's
``RandomForestRegressor.predict`` signature ``(N, d) -> (N,)``.

For the "realistic" variant, ``Ackley("realistic")`` reads its parameters
(``n_optima``, ``basin_width``, ``noise_freq``, ``noise_amp``) from
``synthetic_data/configs/defaults.json``.  You can override any of them via
keyword arguments::

    fn = Ackley("realistic", dim=3, n_optima=5, noise_amp=10.0)

All other variants ignore these keyword arguments and behave exactly as
before.

Example
-------
    from synthetic_data.ackley import Ackley

    fn = Ackley("centroid")
    y = fn.predict(np.array([[1/3, 1/3, 1/3]]))   # ~ [0.0]

    fn = Ackley("realistic")             # reads from configs/defaults.json
    fn = Ackley("realistic", n_optima=5) # override one param
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ── Config file for tunable defaults ─────────────────────────────────────────
_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
_DEFAULTS_PATH = _CONFIGS_DIR / "defaults.json"

_HARDCODED_DEFAULTS = {
    "n_optima": 10,
    "basin_width": 50,
    "noise_freq": 8.0,
    "noise_amp": 5.0,
}


def load_config() -> dict:
    """Load tunable defaults from ``synthetic_data/configs/defaults.json``."""
    if _DEFAULTS_PATH.exists():
        with open(_DEFAULTS_PATH) as f:
            return json.load(f)
    return dict(_HARDCODED_DEFAULTS)


def save_config(cfg: dict) -> None:
    """Write tunable defaults to ``synthetic_data/configs/defaults.json``."""
    _CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_DEFAULTS_PATH, "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")

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
    d = X.shape[1]
    delta = X - np.asarray(center, dtype=float)
    t1 = -a * np.exp(-b * np.sqrt(np.sum(delta ** 2, axis=1) / d))
    t2 = -np.exp(np.sum(np.cos(c * delta), axis=1) / d)
    ackley = t1 + t2 + a + np.e
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
    ``synthetic_data/configs/defaults.json`` and can be overridden via kwargs.
    All other variants ignore the extra kwargs.

    Parameters
    ----------
    variant : str
        One of ``Ackley.VARIANTS``.
    dim : int
        Simplex dimensionality.
    n_optima, basin_width, noise_freq, noise_amp : optional
        Overrides for the "realistic" variant (ignored by others).
        Unspecified values are read from the config file.
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
            _n = int(n_optima if n_optima is not None else cfg["n_optima"])
            _b = float(basin_width if basin_width is not None else cfg["basin_width"])
            _nf = float(noise_freq if noise_freq is not None else cfg["noise_freq"])
            _na = float(noise_amp if noise_amp is not None else cfg["noise_amp"])

            rng = np.random.default_rng(peak_seed)
            self.centers = [c.copy() for c in rng.dirichlet(np.ones(dim), size=_n)]
            self.basin_widths = [_b] * _n
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

        # Estimate raw range by sampling the simplex, then scale to [0.5, 1].
        _est_rng = np.random.default_rng(12345)
        _samples = _est_rng.dirichlet(np.ones(dim), size=_RANGE_SAMPLES)
        _raw = self._predict_raw(_samples)
        self._raw_min = float(_raw.min())
        self._raw_max = float(_raw.max())

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        terms = np.stack(
            [_negated_ackley(X, center, b=b)
             for center, b in zip(self.centers, self.basin_widths)],
            axis=0,
        )
        result = terms.max(axis=0) if self.combine == "max" else terms.sum(axis=0)
        if self._noise_amp > 0:
            result = result + simplex_noise(
                X,
                frequency=self._noise_freq,
                amplitude=self._noise_amp,
                octaves=self._noise_octaves,
                seed=self._noise_seed,
            )
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self._predict_raw(X)
        span = self._raw_max - self._raw_min
        if span < 1e-12:
            return np.full(raw.shape, 0.75)
        return 0.5 + 0.5 * (raw - self._raw_min) / span

    def __call__(self, x: np.ndarray) -> float:
        return float(self.predict(np.asarray(x, dtype=float).reshape(1, -1))[0])

    @property
    def known_maxima(self) -> list[tuple[np.ndarray, float]]:
        return [(c.copy(), float(self.predict(c.reshape(1, -1))[0])) for c in self.centers]

    def __repr__(self) -> str:  # pragma: no cover
        return f"Ackley(variant={self.variant!r})"


# Backwards-compatible alias
TunableRealistic = Ackley
