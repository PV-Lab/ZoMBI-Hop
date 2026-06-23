"""
Dimension-general synthetic oracles for campaign-style benchmarking.

Used by ``generate_synthetic_campaign.py`` (dataset generation) and scalable
to 10D+ for MOBO hyperparameter transfer experiments.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

from synthetic_data.ackley import SCALE_OPTIMA_WITH_DIM, scaled_n_optima

ORACLE_CHOICES = (
    "messy",
    "ackley",
    "gaussian",
    "rastrigin_ilr",
    "planted_bumps",
    "aitchison_gaussian",
    "quadratic_ilr",
    "simplex_linear",
    "vertex_max",
    "rastrigin_direct",
    "bumps",
)

# Short mathematical description for metadata / dashboards.
ORACLE_EXPRESSIONS: dict[str, str] = {
    "messy": "Σ exp(-||x-c||²/2σ²) major + signed micro + Σ A·sin(ω·ilr(x)+φ)",
    "ackley": "negated Ackley basins at Dirichlet peaks + Perlin noise",
    "gaussian": "Σ exp(-||x-p||²/2σ²)  (Euclidean composition distance)",
    "rastrigin_ilr": "-[An + Σ(zᵢ² - A cos(2πzᵢ))],  z = ilr(x)",
    "planted_bumps": "Σ exp(-||x-c||²/2σ²) major + unsigned micro bumps",
    "aitchison_gaussian": "Σ exp(-d_A(x,p)²/2σ²)  (Aitchison / CLR distance)",
    "quadratic_ilr": "-||ilr(x) - ilr(c*)||²  (single ILR bowl)",
    "simplex_linear": "w·x  (dot product; peak at one vertex)",
    "vertex_max": "maxᵢ xᵢ  (facet expression; peaks at all vertices)",
    "rastrigin_direct": "-[An + Σ((fxᵢ)² - A cos(2πfxᵢ))],  f·x ∈ ℤ₊",
    "bumps": "Σ exp(-||x-c||²/2σ²) random Dirichlet majors + micro ([0.5,1])",
}

_RASTRIGIN_CONFIGS_DIR = Path(__file__).resolve().parent / "rastrigin_ilr"
_RASTRIGIN_DEFAULTS_PATH = _RASTRIGIN_CONFIGS_DIR / "defaults.json"
_RASTRIGIN_HARDCODED_DEFAULTS = {
    "n_optima": 20,
    "amplitude": 30.0,
}


def load_rastrigin_config() -> dict:
    if _RASTRIGIN_DEFAULTS_PATH.is_file():
        with open(_RASTRIGIN_DEFAULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return dict(_RASTRIGIN_HARDCODED_DEFAULTS)


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
    """Fixed symmetric peak geometry for messy / gaussian / planted_bumps oracles."""
    if d < 2:
        raise ValueError("layout-based peak geometry requires d >= 2")
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


def ilr_to_composition_np(ilr: np.ndarray, d: int) -> np.ndarray:
    """Inverse Helmert ILR (numpy), matching ``src.utils.simplex.ilr_to_composition``."""
    ilr = np.asarray(ilr, dtype=float)
    log_x = np.zeros(d, dtype=float)
    for i in range(d - 1):
        coef = math.sqrt((i + 1) / (i + 2))
        contribution = ilr[i] * coef
        log_x[: i + 1] += contribution / (i + 1)
        log_x[i + 1] -= contribution
    x = np.exp(log_x)
    return x / x.sum()


def rastrigin_ilr_optima(
    d: int,
    *,
    n_optima: int = 20,
    amplitude: float = 30.0,
    max_search_radius: int = 12,
) -> list[np.ndarray]:
    """Top ``n_optima`` simplex compositions at integer ILR lattice points."""
    if n_optima < 1:
        raise ValueError("n_optima must be >= 1")

    def _eval(x: np.ndarray) -> float:
        z = composition_to_ilr_np(x)
        n = z.shape[0]
        rastrigin = amplitude * n + float(np.sum(z ** 2 - amplitude * np.cos(2.0 * math.pi * z)))
        return -rastrigin

    optima: list[np.ndarray] = []
    for radius in range(1, max_search_radius + 1):
        ranges = [range(-radius, radius + 1) for _ in range(d - 1)]
        candidates: list[tuple[float, np.ndarray]] = []
        for z_tuple in itertools.product(*ranges):
            x = ilr_to_composition_np(np.asarray(z_tuple, dtype=float), d)
            if np.all(x >= -1e-12) and abs(float(x.sum()) - 1.0) < 1e-9:
                candidates.append((_eval(x), x.copy()))
        candidates.sort(key=lambda item: -item[0])
        optima = []
        seen: set[tuple[float, ...]] = set()
        for _, x in candidates:
            key = tuple(np.round(x, 8))
            if key in seen:
                continue
            seen.add(key)
            optima.append(x)
            if len(optima) >= n_optima:
                return optima

    raise ValueError(
        f"Found only {len(optima)} rastrigin_ilr optima for d={d} "
        f"within search radius {max_search_radius}; need {n_optima}."
    )


def normalize_rows(X: np.ndarray) -> np.ndarray:
    s = X.sum(axis=1, keepdims=True)
    return X / np.where(s == 0, 1.0, s)


_EPS = 1e-10


def aitchison_distance_np(x: np.ndarray, y: np.ndarray) -> float:
    """Aitchison distance between two compositions (CLR Euclidean norm)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    log_x = np.log(x + _EPS)
    log_y = np.log(y + _EPS)
    clr_x = log_x - log_x.mean()
    clr_y = log_y - log_y.mean()
    return float(np.linalg.norm(clr_x - clr_y))


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

    def __init__(
        self,
        d: int = 3,
        *,
        amplitude: float | None = None,
        n_optima: int | None = None,
    ):
        cfg = load_rastrigin_config()
        self.d = d
        self.amplitude = float(amplitude if amplitude is not None else cfg["amplitude"])
        if n_optima is not None:
            _n = int(n_optima)
        elif SCALE_OPTIMA_WITH_DIM:
            _n = scaled_n_optima(int(cfg["n_optima"]), d)
        else:
            _n = int(cfg["n_optima"])
        self.n_optima = _n
        self._optima = rastrigin_ilr_optima(d, n_optima=_n, amplitude=self.amplitude)

    def __call__(self, x: np.ndarray) -> float:
        z = composition_to_ilr_np(np.asarray(x, dtype=float))
        n = z.shape[0]
        rastrigin = self.amplitude * n + float(np.sum(z ** 2 - self.amplitude * np.cos(2.0 * math.pi * z)))
        return -rastrigin

    @property
    def true_optima(self) -> list[np.ndarray]:
        return [p.copy() for p in self._optima]


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


class AitchisonGaussianOracle:
    """Gaussian mixture on Aitchison distance — compositional-native smooth peaks."""

    maximize = True

    def __init__(self, peaks: list[np.ndarray], *, sigma: float = 0.35):
        self.peaks = [normalize_rows(np.asarray(p, dtype=float).reshape(1, -1))[0] for p in peaks]
        self.sigma = float(sigma)

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(sum(
            math.exp(-(aitchison_distance_np(x, p) / self.sigma) ** 2 / 2.0)
            for p in self.peaks
        ))

    @property
    def true_optima(self) -> list[np.ndarray]:
        return [p.copy() for p in self.peaks]


class QuadraticILROracle:
    """Single smooth bowl: negative squared ILR distance from a target composition."""

    maximize = True

    def __init__(self, center: np.ndarray, *, scale: float = 1.0):
        self.center = normalize_rows(np.asarray(center, dtype=float).reshape(1, -1))[0]
        self.z_star = composition_to_ilr_np(self.center)
        self.scale = float(scale)

    def __call__(self, x: np.ndarray) -> float:
        z = composition_to_ilr_np(np.asarray(x, dtype=float))
        return -self.scale * float(np.sum((z - self.z_star) ** 2))

    @property
    def true_optima(self) -> list[np.ndarray]:
        return [self.center.copy()]


class SimplexLinearOracle:
    """Linear expression on composition coordinates; optimum at one vertex."""

    maximize = True

    def __init__(self, d: int, *, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.weights = rng.uniform(0.5, 2.0, d)
        self._optimum_vertex = simplex_vertex(d, int(np.argmax(self.weights)))

    def __call__(self, x: np.ndarray) -> float:
        return float(np.dot(self.weights, np.asarray(x, dtype=float)))

    @property
    def true_optima(self) -> list[np.ndarray]:
        return [self._optimum_vertex.copy()]


class VertexMaxOracle:
    """Facet / vertex expression: max component (all pure compositions tie)."""

    maximize = True

    def __init__(self, d: int) -> None:
        if d < 2:
            raise ValueError("vertex_max requires d >= 2")
        self.d = d

    def __call__(self, x: np.ndarray) -> float:
        return float(np.max(np.asarray(x, dtype=float)))

    @property
    def true_optima(self) -> list[np.ndarray]:
        return [simplex_vertex(self.d, i) for i in range(self.d)]


class RastriginDirectOracle:
    """Rastrigin on raw composition coordinates (rational lattice optima)."""

    maximize = True

    def __init__(self, d: int = 3, *, amplitude: float = 10.0, frequency: int = 5):
        self.d = d
        self.amplitude = float(amplitude)
        self.frequency = int(frequency)

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
        """Compositions where frequency * x_i is a non-negative integer summing to frequency."""
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

    @property
    def true_optima(self) -> list[np.ndarray]:
        return self.local_optima


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
    variant: str = "layout",
    n_peaks: int | None = None,
    sigma: float | None = None,
    sigma_var: float | None = None,
    noise_freq: float | None = None,
    noise_amp: float | None = None,
) -> tuple[object, list[np.ndarray], str]:
    """Return (callable oracle, reference optima, display label)."""
    if name == "gaussian" and variant == "realistic":
        from synthetic_data.gaussian_landscape import RealisticGaussian
        obj = RealisticGaussian(
            d,
            n_peaks=n_peaks,
            sigma=sigma,
            sigma_var=sigma_var,
            noise_freq=noise_freq,
            noise_amp=noise_amp,
            peak_seed=seed,
        )
        label = (f"Realistic Gaussian ({len(obj.centers)} random peaks, "
                 f"σ≈{obj.sigmas[0]:.3g}, noise_amp={obj._noise_amp})")
        return obj, [c.copy() for c in obj.centers], label

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
        cfg = load_rastrigin_config()
        obj = RastriginILROracle(d, amplitude=float(cfg["amplitude"]))
        label = f"Rastrigin in ILR ({len(obj.true_optima)} lattice optima)"
        return obj, obj.true_optima, label
    if name == "planted_bumps":
        obj = PlantedBumpField(centers, n_micro=40, seed=seed)
        label = f"Planted bumps ({len(centers)} major + 40 micro)"
        return obj, obj.true_optima, label
    if name == "aitchison_gaussian":
        obj = AitchisonGaussianOracle(centers, sigma=0.35)
        label = f"Aitchison Gaussian ({len(centers)} peaks, σ=0.35)"
        return obj, obj.true_optima, label
    if name == "quadratic_ilr":
        obj = QuadraticILROracle(centers[0], scale=1.0)
        label = f"Quadratic ILR bowl (center=layout peak 0)"
        return obj, obj.true_optima, label
    if name == "simplex_linear":
        obj = SimplexLinearOracle(d, seed=seed)
        label = f"Simplex linear w·x (seed={seed})"
        return obj, obj.true_optima, label
    if name == "vertex_max":
        obj = VertexMaxOracle(d)
        label = f"Vertex max max(x) ({d} global optima at vertices)"
        return obj, obj.true_optima, label
    if name == "rastrigin_direct":
        obj = RastriginDirectOracle(d, amplitude=10.0, frequency=5)
        label = f"Rastrigin direct ({len(obj.true_optima)} rational lattice optima)"
        return obj, obj.true_optima, label
    if name == "bumps":
        from synthetic_data.bumps import Bumps

        obj = Bumps(dim=d, seed=seed)
        optima = [c.copy() for c in obj.major_centers]
        label = f"Random bumps ({len(optima)} major + micro, seed={seed})"
        return obj, optima, label
    raise ValueError(f"Unknown oracle {name!r}; choose from {ORACLE_CHOICES}")
