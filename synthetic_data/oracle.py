"""
Dimension-general synthetic oracles for campaign-style benchmarking.

Used by ``generate_synthetic_campaign.py`` (dataset generation) and scalable
to 10D+ for MOBO hyperparameter transfer experiments.
"""

from __future__ import annotations

import math

import numpy as np

ORACLE_CHOICES = (
    "messy",
    "ackley",
    "gaussian",
    "rastrigin_ilr",
    "rastrigin_direct",
    "planted_bumps",
)


def simplex_vertex(d: int, i: int) -> np.ndarray:
    p = np.zeros(d, dtype=float)
    p[i] = 1.0
    return p


def centroid_composition(d: int) -> np.ndarray:
    return np.ones(d, dtype=float) / d


def edge_midpoint(d: int, i: int, j: int) -> np.ndarray:
    p = np.zeros(d, dtype=float)
    p[i] = 0.5
    p[j] = 0.5
    return p


def ackley_centers_for_layout(d: int, layout: str) -> list[np.ndarray]:
    if d < 2:
        raise ValueError("Multi-Ackley requires d >= 2")
    centers: list[np.ndarray] = [
        centroid_composition(d),
        simplex_vertex(d, 0),
        edge_midpoint(d, 0, 1),
    ]
    if layout in ("2", "3"):
        if d < 3:
            raise ValueError(f"Layout {layout} requires d >= 3")
        centers.append(simplex_vertex(d, 1))
        centers.append(simplex_vertex(d, 2))
    if layout == "3":
        if d < 5:
            raise ValueError("Layout 3 requires d >= 5")
        centers.append(simplex_vertex(d, 3))
        centers.append(simplex_vertex(d, 4))
    if layout not in ("1", "2", "3"):
        raise ValueError(f"Unknown layout {layout!r}; use '1', '2', or '3'.")
    return centers


def composition_to_ilr_np(x: np.ndarray) -> np.ndarray:
    """Helmert ILR (numpy), matching ``src.utils.simplex.composition_to_ilr``."""
    x = np.asarray(x, dtype=float)
    eps = 1e-10
    log_x = np.log(x + eps)
    d = x.shape[-1]
    ilr = np.empty(d - 1, dtype=float)
    for i in range(d - 1):
        coef = math.sqrt((i + 1) / (i + 2))
        ilr[i] = coef * (log_x[: i + 1].sum() / (i + 1) - log_x[i + 1])
    return ilr


def ilr_to_composition_np(z: np.ndarray, d: int) -> np.ndarray:
    """Inverse Helmert ILR: map ILR coordinates back to a composition."""
    z = np.asarray(z, dtype=float)
    # Accumulate clr = H^T @ z using the same Helmert contrast H as composition_to_ilr_np
    clr = np.zeros(d, dtype=float)
    for i in range(d - 1):
        coef = math.sqrt((i + 1) / (i + 2))
        clr[: i + 1] += coef / (i + 1) * z[i]
        clr[i + 1] -= coef * z[i]
    clr -= clr.max()
    x = np.exp(clr)
    return x / x.sum()


def normalize_rows(X: np.ndarray) -> np.ndarray:
    s = X.sum(axis=1, keepdims=True)
    return X / np.where(s == 0, 1.0, s)


class GaussianMixtureOracle:
    maximize = True

    def __init__(self, peaks: list[np.ndarray], *, sigma: float = 0.07):
        self.peaks = [normalize_rows(np.asarray(p, dtype=float).reshape(1, -1))[0] for p in peaks]
        self.sigma = sigma

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(sum(
            math.exp(-np.sum((x - p) ** 2) / (2.0 * self.sigma ** 2))
            for p in self.peaks
        ))

    @property
    def true_optima(self) -> list[np.ndarray]:
        return [p.copy() for p in self.peaks]


class RastriginILROracle:
    maximize = True

    def __init__(self, d: int = 3, *, amplitude: float = 10.0):
        self.d = d
        self.amplitude = amplitude
        self._centroid = centroid_composition(d)

    def __call__(self, x: np.ndarray) -> float:
        z = composition_to_ilr_np(np.asarray(x, dtype=float))
        n = z.shape[0]
        rastrigin = self.amplitude * n + float(np.sum(z ** 2 - self.amplitude * np.cos(2.0 * math.pi * z)))
        return -rastrigin

    @property
    def true_optima(self) -> list[np.ndarray]:
        return [self._centroid.copy()]

    @property
    def local_optima(self) -> list[np.ndarray]:
        """All integer ILR lattice points that map to interior simplex points.

        Rastrigin's local minima (our local maxima) occur where every ILR
        coordinate is an integer.  We enumerate integer combinations in [-k, k]
        and invert back to composition; all-positive compositions are kept.
        """
        k = 5  # search radius; ILR values beyond ~4 hit simplex boundary
        n_ilr = self.d - 1
        ranges = [range(-k, k + 1)] * n_ilr
        optima: list[np.ndarray] = []
        eps = 1e-3
        for combo in np.ndindex(*[2 * k + 1] * n_ilr):
            z = np.array([c - k for c in combo], dtype=float)
            x = ilr_to_composition_np(z, self.d)
            if np.all(x > eps):
                optima.append(x)
        return optima


class RastriginDirectOracle:
    """Rastrigin applied directly to composition coordinates (no ILR transform).

    Local optima occur where ``frequency * x_i`` is integer for every component.
    """

    maximize = True

    def __init__(self, d: int = 3, *, amplitude: float = 10.0, frequency: int = 5):
        self.d = d
        self.amplitude = amplitude
        self.frequency = frequency

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        z = self.frequency * x
        n = z.shape[0]
        rastrigin = self.amplitude * n + float(
            np.sum(z ** 2 - self.amplitude * np.cos(2.0 * math.pi * z))
        )
        return -rastrigin

    @property
    def local_optima(self) -> list[np.ndarray]:
        """All compositions where ``frequency * x_i`` is a non-negative integer."""
        optima: list[np.ndarray] = []
        freq = self.frequency

        def _enumerate(dims_left: int, total: int, prefix: list[int]) -> None:
            if dims_left == 1:
                optima.append(np.array(prefix + [total], dtype=float) / freq)
                return
            for k in range(total + 1):
                _enumerate(dims_left - 1, total - k, prefix + [k])

        _enumerate(self.d, freq, [])
        return optima


class PlantedBumpField:
    maximize = True

    def __init__(
        self,
        major_centers: list[np.ndarray],
        *,
        n_micro: int = 40,
        major_sigma: float = 0.09,
        signed_micro: bool = False,
        seed: int = 42,
    ):
        self.major_centers = [np.asarray(c, dtype=float).copy() for c in major_centers]
        self.major_sigma = major_sigma
        rng = np.random.default_rng(seed)
        d = major_centers[0].shape[0]
        self._micro: list[tuple[np.ndarray, float, float]] = []
        for _ in range(n_micro):
            center = rng.dirichlet(np.ones(d))
            sigma = float(rng.uniform(0.015, 0.05))
            weight = float(rng.uniform(-0.50, 0.50)) if signed_micro else float(rng.uniform(0.08, 0.35))
            self._micro.append((center, sigma, weight))

    def _bump(self, x: np.ndarray, center: np.ndarray, sigma: float, weight: float) -> float:
        return weight * math.exp(-np.sum((x - center) ** 2) / (2.0 * sigma ** 2))

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        val = sum(self._bump(x, c, self.major_sigma, 1.0) for c in self.major_centers)
        val += sum(self._bump(x, c, s, w) for c, s, w in self._micro)
        return float(val)

    @property
    def true_optima(self) -> list[np.ndarray]:
        return [c.copy() for c in self.major_centers]


class MessyCampaignOracle:
    maximize = True

    def __init__(
        self,
        major_centers: list[np.ndarray],
        *,
        n_micro: int = 150,
        n_ripples: int = 30,
        major_sigma: float = 0.055,
        seed: int = 42,
    ):
        self._bumps = PlantedBumpField(
            major_centers,
            n_micro=n_micro,
            major_sigma=major_sigma,
            signed_micro=True,
            seed=seed,
        )
        rng = np.random.default_rng(seed + 1)
        d = major_centers[0].shape[0]
        self._ripples: list[tuple[np.ndarray, float, float]] = []
        for _ in range(n_ripples):
            freq = rng.normal(0.0, 1.0, size=d - 1)
            freq /= np.linalg.norm(freq) + 1e-12
            amp = float(rng.uniform(0.06, 0.18))
            phase = float(rng.uniform(0.0, 2.0 * math.pi))
            self._ripples.append((freq, amp, phase))
        self.major_centers = self._bumps.major_centers

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        val = self._bumps(x)
        z = composition_to_ilr_np(x)
        for freq, amp, phase in self._ripples:
            val += amp * math.sin(float(np.dot(freq, z)) + phase)
        return float(val)

    @property
    def true_optima(self) -> list[np.ndarray]:
        return [c.copy() for c in self.major_centers]


def build_oracle(
    name: str,
    d: int,
    layout: str,
    *,
    seed: int,
    ackley_b: float | None = None,
) -> tuple[object, list[np.ndarray], str]:
    """Return (callable oracle, reference optima, display label)."""
    if name == "ackley":
        from synthetic_data.ackley import Ackley

        obj = Ackley("realistic", dim=d, peak_seed=seed)
        optima = [c.copy() for c in obj.centers]
        label = f"Realistic Ackley ({len(optima)} peaks, dim={d}, seed={seed})"
        return obj, optima, label

    centers = ackley_centers_for_layout(d, layout)
    if name == "messy":
        obj = MessyCampaignOracle(centers, n_micro=150, n_ripples=30, seed=seed)
        label = f"Messy campaign ({len(centers)} major + 150 signed micro + 30 ILR ripples)"
        return obj, obj.true_optima, label
    if name == "gaussian":
        obj = GaussianMixtureOracle(centers, sigma=0.07)
        label = f"Gaussian mixture ({len(centers)} peaks, σ=0.07)"
        return obj, obj.true_optima, label
    if name == "rastrigin_ilr":
        obj = RastriginILROracle(d)
        label = "Rastrigin in ILR (cosine ripples)"
        return obj, obj.true_optima, label
    if name == "rastrigin_direct":
        obj = RastriginDirectOracle(d)
        label = "Rastrigin direct (no ILR)"
        return obj, obj.local_optima, label
    if name == "planted_bumps":
        obj = PlantedBumpField(centers, n_micro=40, seed=seed)
        label = f"Planted bumps ({len(centers)} major + 40 micro)"
        return obj, obj.true_optima, label
    raise ValueError(f"Unknown oracle {name!r}; choose from {ORACLE_CHOICES}")
