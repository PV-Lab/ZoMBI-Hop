"""
Realistic Gaussian mixture on the probability simplex.

Mirrors ``Ackley("realistic")``: Dirichlet-sampled peak locations (count scaled
with dimension), optional per-peak σ jitter, and Perlin-style background noise.
Configurable via ``synthetic_data/gaussian/defaults.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from synthetic_data.ackley import SCALE_OPTIMA_WITH_DIM, scaled_n_optima, simplex_noise

_CONFIGS_DIR = Path(__file__).resolve().parent / "gaussian"
_DEFAULTS_PATH = _CONFIGS_DIR / "defaults.json"

_HARDCODED_DEFAULTS = {
    "n_peaks": 20,
    "sigma": 0.07,
    "sigma_var": 0.0,
    "noise_freq": 9.0,
    "noise_amp": 0.15,
}

SIGMA_BY_DIM = {
    3: 0.07,
    4: 0.06,
    10: 0.04,
}

NOISE_AMP_BY_DIM = {
    3: 0.15,
    4: 0.12,
    10: 0.08,
}

_RANGE_SAMPLES = 100_000


def load_config() -> dict:
    if _DEFAULTS_PATH.is_file():
        with open(_DEFAULTS_PATH) as f:
            return json.load(f)
    return dict(_HARDCODED_DEFAULTS)


def save_config(cfg: dict) -> None:
    _CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_DEFAULTS_PATH, "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")


class RealisticGaussian:
    """Sum-of-Gaussians objective with random simplex peaks and optional noise."""

    maximize = True

    def __init__(
        self,
        dim: int = 3,
        *,
        n_peaks: int | None = None,
        sigma: float | None = None,
        sigma_var: float | None = None,
        noise_freq: float | None = None,
        noise_amp: float | None = None,
        noise_octaves: int = 4,
        noise_seed: int = 42,
        peak_seed: int = 0,
    ) -> None:
        if dim < 2:
            raise ValueError(f"dim must be >= 2 (got {dim}).")
        self.dim = dim
        cfg = load_config()

        if n_peaks is not None:
            _n = int(n_peaks)
        elif SCALE_OPTIMA_WITH_DIM:
            _n = scaled_n_optima(int(cfg["n_peaks"]), dim)
        else:
            _n = int(cfg["n_peaks"])

        if sigma is not None:
            _sigma = float(sigma)
        else:
            _sigma = float(SIGMA_BY_DIM.get(dim, cfg["sigma"]))

        _sv = float(sigma_var if sigma_var is not None else cfg.get("sigma_var", 0.0))
        _nf = float(noise_freq if noise_freq is not None else cfg["noise_freq"])
        if noise_amp is not None:
            _na = float(noise_amp)
        else:
            _na = float(NOISE_AMP_BY_DIM.get(dim, cfg["noise_amp"]))

        rng = np.random.default_rng(peak_seed)
        self.centers = [c.copy() for c in rng.dirichlet(np.ones(dim), size=_n)]
        if _sv > 0:
            self.sigmas = [max(0.01, float(rng.normal(_sigma, _sv))) for _ in range(_n)]
        else:
            self.sigmas = [_sigma] * _n

        self._noise_freq = _nf
        self._noise_amp = _na
        self._noise_octaves = noise_octaves
        self._noise_seed = noise_seed

        est_rng = np.random.default_rng(12345)
        samples = est_rng.dirichlet(np.ones(dim), size=_RANGE_SAMPLES)
        if self.centers:
            samples = np.vstack([samples, np.asarray(self.centers, dtype=float)])
        raw = self._peaks_raw(samples) + self._noise_raw(samples)
        self._raw_min = float(raw.min())
        self._raw_max = float(raw.max())

    def _peaks_raw(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.zeros(X.shape[0], dtype=float)
        for center, sigma in zip(self.centers, self.sigmas):
            out += np.exp(-np.sum((X - center) ** 2, axis=1) / (2.0 * sigma ** 2))
        return out

    def _noise_raw(self, X: np.ndarray) -> np.ndarray:
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
        return self._peaks_raw(X) + self._noise_raw(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self._predict_raw(X)
        span = self._raw_max - self._raw_min
        if span < 1e-12:
            return np.full(raw.shape, 0.75)
        return 0.5 + 0.5 * (raw - self._raw_min) / span

    def __call__(self, x: np.ndarray) -> float:
        return float(self.predict(np.asarray(x, dtype=float).reshape(1, -1))[0])
