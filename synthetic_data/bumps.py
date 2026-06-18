"""
bumps.py
========
A "planted bump field" objective on the probability simplex: a handful of wide
*major* Gaussian bumps (the true optima) sitting in a sea of many narrow
*micro* bumps that stud the landscape with local structure.

Both the major and the micro bumps are randomly placed (Dirichlet-sampled on
the simplex) and fully customizable, in the same spirit as the
``Ackley("realistic")`` optima:

    * the **number** of major / micro bumps scales with dimension by ``(d-1)/2``
      relative to a 3-simplex baseline (see ``scaled_n_bumps``); passing
      ``n_major`` / ``n_micro`` explicitly disables that scaling and uses the
      exact count given;
    * **placement** is reproducible via ``seed`` (Dirichlet draws);
    * bump **widths** and micro-bump **weights** are tunable via keyword.

The raw bump field is rescaled onto ``[0.5, 1]`` (peak -> 1, far-field floor ->
0.5), exactly like the ``Ackley("realistic")`` variant, so the two objectives
share a value range.

The ``Bumps`` class exposes a ``predict(X)`` method matching scikit-learn's
``RandomForestRegressor.predict`` signature ``(N, d) -> (N,)``, an
``__call__(x) -> float`` for single points, and a ``known_maxima`` property
listing each major bump center with its objective value.

Example
-------
    from synthetic_data.bumps import Bumps

    fn = Bumps(dim=3)                       # default scaled counts
    y = fn.predict(np.array([[1/3, 1/3, 1/3]]))

    fn = Bumps(dim=4, n_major=5)            # exactly 5 major bumps
    fn = Bumps(dim=3, signed_micro=True)    # micro bumps may dent the surface
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# ── Config file for tunable defaults ─────────────────────────────────────────
_CONFIGS_DIR = Path(__file__).resolve().parent / "defaults"
_DEFAULTS_PATH = _CONFIGS_DIR / "bumps.json"

# When True (the default) the major/micro bump counts grow with dimension by
# ``(d-1)/2`` relative to the configured 3-simplex baseline; an explicit
# ``n_major`` / ``n_micro`` override always disables this for that count.
SCALE_COUNTS_WITH_DIM = True

# 3-simplex (d=3) baselines for the bump counts, scaled with dimension by
# ``scaled_n_bumps`` unless overridden.
N_MAJOR_BASE = 3
N_MICRO_BASE = 40

# Default bump widths (Gaussian sigmas, in composition units).
MAJOR_SIGMA = 0.09
MICRO_SIGMA_RANGE = (0.015, 0.05)

# Default micro-bump weight ranges.  Unsigned micro bumps only add to the
# surface; signed micro bumps may also subtract from it.
MICRO_WEIGHT_RANGE = (0.08, 0.35)
MICRO_WEIGHT_RANGE_SIGNED = (-0.50, 0.50)

# Simplex samples used to estimate the raw bump-field min/max for the [0.5, 1]
# rescaling (mirrors ``synthetic_data.ackley._RANGE_SAMPLES``).
_RANGE_SAMPLES = 100_000

_HARDCODED_DEFAULTS = {
    "n_major": N_MAJOR_BASE,
    "n_micro": N_MICRO_BASE,
    "major_sigma": MAJOR_SIGMA,
    "micro_sigma_min": MICRO_SIGMA_RANGE[0],
    "micro_sigma_max": MICRO_SIGMA_RANGE[1],
    "micro_weight_min": MICRO_WEIGHT_RANGE[0],
    "micro_weight_max": MICRO_WEIGHT_RANGE[1],
    "seed": 42,
}


def load_config() -> dict:
    """Load tunable defaults from ``synthetic_data/defaults/bumps.json``."""
    if _DEFAULTS_PATH.exists():
        with open(_DEFAULTS_PATH) as f:
            return json.load(f)
    return dict(_HARDCODED_DEFAULTS)


def save_config(cfg: dict) -> None:
    """Write tunable defaults to ``synthetic_data/defaults/bumps.json``."""
    _CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_DEFAULTS_PATH, "w") as f:
        json.dump(cfg, f, indent=4)
        f.write("\n")


def scaled_n_bumps(n_base: int, dim: int) -> int:
    """Number of bumps for ``dim``, scaling a 3-simplex baseline.

    The configured count describes the **3-simplex (d=3)**.  It then grows
    **linearly** with the tangent dimension by ``(d-1)/2`` -- exactly the
    baseline at d=3, ~1.5x at d=4, 4.5x at d=10 -- matching the optima scaling
    used by ``synthetic_data.ackley.scaled_n_optima`` so the two benchmarks stay
    comparable per unit of sampling budget across dimensions.
    """
    return max(1, int(round(n_base * (dim - 1) / 2.0)))


class Bumps:
    """A planted bump field on the ``dim``-element probability simplex.

    The objective is the sum of ``n_major`` wide Gaussian bumps (weight 1, the
    true optima) plus ``n_micro`` narrow Gaussian bumps of random width and
    weight.  All centers are Dirichlet-sampled on the simplex.

    Parameters
    ----------
    dim : int
        Simplex dimensionality (>= 2).
    n_major, n_micro : optional
        Number of major / micro bumps.  When left unspecified they are scaled
        with ``dim`` by ``(d-1)/2`` relative to the ``N_MAJOR_BASE`` /
        ``N_MICRO_BASE`` (d=3) baselines; passing a value uses that exact count.
    major_sigma : float
        Gaussian width of the major bumps.
    micro_sigma_range, micro_weight_range : (low, high), optional
        Uniform ranges the micro bumps' widths / weights are drawn from.  When
        ``micro_weight_range`` is unset it defaults to ``MICRO_WEIGHT_RANGE``
        (or ``MICRO_WEIGHT_RANGE_SIGNED`` when ``signed_micro`` is True).
    signed_micro : bool
        If True, micro bumps may have negative weight (dents as well as bumps).
    seed : int
        Seed for the Dirichlet placement and the micro width/weight draws.
    """

    maximize = True

    def __init__(
        self,
        dim: int = 3,
        *,
        n_major: int | None = None,
        n_micro: int | None = None,
        major_sigma: float = MAJOR_SIGMA,
        micro_sigma_range: tuple[float, float] = MICRO_SIGMA_RANGE,
        micro_weight_range: tuple[float, float] | None = None,
        signed_micro: bool = False,
        seed: int = 42,
    ) -> None:
        if dim < 2:
            raise ValueError(f"dim must be >= 2 (got {dim}).")
        self.dim = dim
        self.major_sigma = float(major_sigma)
        self.signed_micro = signed_micro
        self.seed = seed

        # Explicit counts are honoured exactly; otherwise the (d=3) baselines are
        # scaled with dimension so higher-d benchmarks have proportionally more
        # bumps (see ``scaled_n_bumps``).
        if n_major is not None:
            _n_major = int(n_major)
        elif SCALE_COUNTS_WITH_DIM:
            _n_major = scaled_n_bumps(N_MAJOR_BASE, dim)
        else:
            _n_major = N_MAJOR_BASE
        if n_micro is not None:
            _n_micro = int(n_micro)
        elif SCALE_COUNTS_WITH_DIM:
            _n_micro = scaled_n_bumps(N_MICRO_BASE, dim)
        else:
            _n_micro = N_MICRO_BASE

        if micro_weight_range is None:
            micro_weight_range = (
                MICRO_WEIGHT_RANGE_SIGNED if signed_micro else MICRO_WEIGHT_RANGE
            )
        s_lo, s_hi = micro_sigma_range
        w_lo, w_hi = micro_weight_range

        rng = np.random.default_rng(seed)
        # Major bumps: the true optima, Dirichlet-placed, unit weight.
        self.major_centers = [c.copy() for c in rng.dirichlet(np.ones(dim), size=_n_major)]
        # Micro bumps: Dirichlet-placed with random width and weight.
        self._micro: list[tuple[np.ndarray, float, float]] = []
        for _ in range(_n_micro):
            center = rng.dirichlet(np.ones(dim))
            sigma = float(rng.uniform(s_lo, s_hi))
            weight = float(rng.uniform(w_lo, w_hi))
            self._micro.append((center, sigma, weight))

        # Rescale the raw bump field onto [0.5, 1] (peak -> 1, far-field floor ->
        # 0.5), matching the ``Ackley`` "realistic" variant.  The major centers
        # (the analytic maxima) are included so a sample-only max doesn't
        # underestimate the peak in high dim; for dim > 3 the mapped value is
        # clipped to [0.5, 1] (as in Ackley), since micro bumps between samples
        # can poke past the estimated span.
        self._clip_to_unit = dim > 3
        _est_rng = np.random.default_rng(12345)
        _samples = _est_rng.dirichlet(np.ones(dim), size=_RANGE_SAMPLES)
        if self.major_centers:
            _samples = np.vstack([_samples, np.asarray(self.major_centers, dtype=float)])
        _raw = self._predict_raw(_samples)
        self._raw_min = float(_raw.min())
        self._raw_max = float(_raw.max())

    @staticmethod
    def _bump(X: np.ndarray, center: np.ndarray, sigma: float, weight: float) -> np.ndarray:
        delta = X - center.reshape(1, -1)
        return weight * np.exp(-np.sum(delta ** 2, axis=1) / (2.0 * sigma ** 2))

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.zeros(X.shape[0])
        for c in self.major_centers:
            out += self._bump(X, c, self.major_sigma, 1.0)
        for c, s, w in self._micro:
            out += self._bump(X, c, s, w)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self._predict_raw(X)
        span = self._raw_max - self._raw_min
        if span < 1e-12:
            return np.full(raw.shape, 0.75)
        y = 0.5 + 0.5 * (raw - self._raw_min) / span
        if self._clip_to_unit:
            y = np.clip(y, 0.5, 1.0)
        return y

    def __call__(self, x: np.ndarray) -> float:
        return float(self.predict(np.asarray(x, dtype=float).reshape(1, -1))[0])

    @property
    def true_optima(self) -> list[np.ndarray]:
        return [c.copy() for c in self.major_centers]

    @property
    def known_maxima(self) -> list[tuple[np.ndarray, float]]:
        return [(c.copy(), float(self.predict(c.reshape(1, -1))[0])) for c in self.major_centers]

    def __repr__(self) -> str:  # pragma: no cover
        return f"Bumps(dim={self.dim}, n_major={len(self.major_centers)}, n_micro={len(self._micro)})"
