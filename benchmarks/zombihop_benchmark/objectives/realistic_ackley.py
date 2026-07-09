from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from ..spaces import project_simplex, sample_simplex, validate_simplex
from ..types import BatchObservation, ObjectiveInfo


@dataclass
class RealisticAckleySimplexObjective:
    name: str = "realistic_ackley_3d"
    n_components: int = 3
    maximize: bool = True
    n_optima: int = 20
    basin_width: float = 86.0
    noise_freq: float = 9.0
    noise_amp: float = 400.0
    intensity_mean: float = 0.0
    intensity_var: float = 0.0
    noise_octaves: int = 4
    noise_seed: int = 42
    peak_seed: int = 0
    match_radius_ilr: float | None = 0.25
    match_radius_comp: float | None = 0.05

    def __post_init__(self) -> None:
        if self.n_components < 2:
            raise ValueError("RealisticAckleySimplexObjective requires at least 2 components")
        self._model = _cached_ackley(
            self.n_components,
            int(self.n_optima),
            float(self.basin_width),
            float(self.noise_freq),
            float(self.noise_amp),
            float(self.intensity_mean),
            float(self.intensity_var),
            int(self.noise_octaves),
            int(self.noise_seed),
            int(self.peak_seed),
        )
        maxima = getattr(self._model, "known_maxima", [])
        if maxima:
            self.true_needles = project_simplex(np.asarray([point for point, _ in maxima], dtype=float))
            self._true_needle_y = np.asarray([float(value) for _, value in maxima], dtype=float)
        else:
            centers = getattr(self._model, "centers", [])
            self.true_needles = project_simplex(np.asarray(centers, dtype=float))
            self._true_needle_y = self._values(self.true_needles, seed=None)
        self.info = ObjectiveInfo(
            name=self.name,
            n_components=self.n_components,
            maximize=self.maximize,
            true_needles=self.true_needles.copy(),
            y_star=float(np.max(self._true_needle_y)) if len(self._true_needle_y) else None,
            match_radius_ilr=self.match_radius_ilr,
            match_radius_comp=self.match_radius_comp,
        )

    def initial_design(self, n: int, seed: int) -> np.ndarray:
        return sample_simplex(n, self.n_components, seed)

    def evaluate_points(self, X_expected: np.ndarray, seed: int | None = None) -> BatchObservation:
        X_expected = project_simplex(X_expected)
        if X_expected.ndim == 1:
            X_expected = X_expected.reshape(1, -1)
        validate_simplex(X_expected, atol=1e-5)
        X_actual = X_expected.copy()
        y = self._values(X_actual, seed)
        return BatchObservation(
            X_expected=X_expected,
            X_actual=X_actual,
            y=y.reshape(-1),
            metadata={
                "kind": "realistic_ackley_simplex",
                "variant": "realistic",
                "seed": seed,
                "n_optima": int(self.n_optima),
                "basin_width": float(self.basin_width),
                "noise_freq": float(self.noise_freq),
                "noise_amp": float(self.noise_amp),
                "peak_seed": int(self.peak_seed),
                "noise_seed": int(self.noise_seed),
            },
        )

    def evaluate_line(
        self,
        endpoints: np.ndarray,
        n_points: int,
        seed: int | None = None,
    ) -> BatchObservation:
        arr = np.asarray(endpoints, dtype=float)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.shape != (2, self.n_components):
            raise ValueError(f"endpoints must have shape (2, {self.n_components})")
        left = project_simplex(arr[0])
        right = project_simplex(arr[1])
        ts = np.linspace(0.0, 1.0, int(n_points))
        X_expected = (1.0 - ts[:, None]) * left[None, :] + ts[:, None] * right[None, :]
        obs = self.evaluate_points(X_expected, seed=seed)
        return BatchObservation(
            X_expected=obs.X_expected,
            X_actual=obs.X_actual,
            y=obs.y,
            metadata={**dict(obs.metadata), "left": left.tolist(), "right": right.tolist()},
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "objective_kind": "realistic_ackley_simplex",
            "objective_source": "synthetic_data.ackley.Ackley('realistic')",
            "objective_variant": "realistic",
            "n_components": int(self.n_components),
            "n_optima": int(self.n_optima),
            "basin_width": float(self.basin_width),
            "noise_freq": float(self.noise_freq),
            "noise_amp": float(self.noise_amp),
            "intensity_mean": float(self.intensity_mean),
            "intensity_var": float(self.intensity_var),
            "noise_octaves": int(self.noise_octaves),
            "noise_seed": int(self.noise_seed),
            "peak_seed": int(self.peak_seed),
            "match_radius_ilr": self.match_radius_ilr,
            "match_radius_comp": self.match_radius_comp,
            "num_true_needles": int(len(self.true_needles)),
            "y_star": self.info.y_star,
            "true_needle_best_y": float(np.max(self._true_needle_y)) if len(self._true_needle_y) else None,
            "true_needle_worst_y": float(np.min(self._true_needle_y)) if len(self._true_needle_y) else None,
            "synthetic_role": f"headline_brianna_realistic_{int(self.n_components)}d",
        }

    def true_needle_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        order = np.argsort(self._true_needle_y)[::-1] if len(self._true_needle_y) else []
        for rank, idx in enumerate(order):
            x = self.true_needles[int(idx)]
            row = {
                "rank": int(rank),
                "y": float(self._true_needle_y[int(idx)]),
                "source": "ackley_known_maxima",
            }
            for i, value in enumerate(x):
                row[f"x_{i}"] = float(value)
            rows.append(row)
        return rows

    def objective_distribution_rows(self, n_samples: int = 4096, seed: int | None = None) -> list[dict[str, Any]]:
        n = int(n_samples)
        sample_seed = int(seed if seed is not None else 97_003 + self.n_components * 1_009 + self.n_optima)
        X = sample_simplex(n, self.n_components, sample_seed)
        y = self._values(X, seed=None)
        return [
            {
                "n_samples": n,
                "seed": sample_seed,
                "y_min": float(np.min(y)),
                "y_p01": float(np.quantile(y, 0.01)),
                "y_p05": float(np.quantile(y, 0.05)),
                "y_p25": float(np.quantile(y, 0.25)),
                "y_median": float(np.median(y)),
                "y_p75": float(np.quantile(y, 0.75)),
                "y_p95": float(np.quantile(y, 0.95)),
                "y_p99": float(np.quantile(y, 0.99)),
                "y_max": float(np.max(y)),
                "fraction_above_0.9": float(np.mean(y >= 0.9)),
                "fraction_above_0.95": float(np.mean(y >= 0.95)),
            }
        ]

    def _values(self, X: np.ndarray, seed: int | None = None) -> np.ndarray:
        del seed
        y = np.asarray(self._model.predict(project_simplex(X)), dtype=float).reshape(-1)
        return y if self.maximize else -y


@lru_cache(maxsize=16)
def _cached_ackley(
    n_components: int,
    n_optima: int,
    basin_width: float,
    noise_freq: float,
    noise_amp: float,
    intensity_mean: float,
    intensity_var: float,
    noise_octaves: int,
    noise_seed: int,
    peak_seed: int,
):
    from synthetic_data.ackley import Ackley

    return Ackley(
        "realistic",
        dim=int(n_components),
        n_optima=int(n_optima),
        basin_width=float(basin_width),
        intensity_mean=float(intensity_mean),
        intensity_var=float(intensity_var),
        noise_freq=float(noise_freq),
        noise_amp=float(noise_amp),
        noise_octaves=int(noise_octaves),
        noise_seed=int(noise_seed),
        peak_seed=int(peak_seed),
    )
