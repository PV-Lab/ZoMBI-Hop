from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..spaces import composition_to_ilr_np, project_simplex, sample_simplex, validate_simplex
from ..types import BatchObservation, ObjectiveInfo


@dataclass
class SyntheticSimplexObjective:
    name: str = "synthetic_simplex"
    n_components: int = 3
    maximize: bool = True
    n_needles: int = 4
    basin_width: float = 20.0
    noise_std: float = 0.0
    seed: int = 123
    match_radius_ilr: float | None = 0.25
    match_radius_comp: float | None = 0.05

    def __post_init__(self) -> None:
        if self.n_components < 3:
            raise ValueError("SyntheticSimplexObjective requires at least 3 components")
        self.true_needles = sample_simplex(self.n_needles, self.n_components, self.seed)
        self._needle_ilr = composition_to_ilr_np(self.true_needles)
        self._weights = np.linspace(1.0, 0.8, self.n_needles)
        self.info = ObjectiveInfo(
            name=self.name,
            n_components=self.n_components,
            maximize=self.maximize,
            true_needles=self.true_needles.copy(),
            y_star=1.0,
            match_radius_ilr=self.match_radius_ilr,
            match_radius_comp=self.match_radius_comp,
        )

    def initial_design(self, n: int, seed: int) -> np.ndarray:
        return sample_simplex(n, self.n_components, seed)

    def _values(self, X: np.ndarray, seed: int | None = None) -> np.ndarray:
        X = project_simplex(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        z = composition_to_ilr_np(X)
        diff = z[:, None, :] - self._needle_ilr[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        bumps = self._weights[None, :] * np.exp(-self.basin_width * dist2)
        best_bump = bumps.max(axis=1)
        rugged = 0.5 + 0.25 * np.sin(3.7 * z[:, 0])
        if z.shape[1] > 1:
            rugged += 0.25 * np.cos(2.9 * z[:, 1])
        y = 0.9 * best_bump + 0.1 * np.clip(rugged, 0.0, 1.0)
        if self.noise_std > 0:
            rng = np.random.default_rng(seed)
            y = y + rng.normal(0.0, self.noise_std, size=y.shape)
        y = np.clip(y, 0.0, 1.0)
        return y if self.maximize else -y

    def evaluate_points(self, X_expected: np.ndarray, seed: int | None = None) -> BatchObservation:
        X_expected = project_simplex(X_expected)
        if X_expected.ndim == 1:
            X_expected = X_expected.reshape(1, -1)
        validate_simplex(X_expected)
        X_actual = X_expected.copy()
        y = self._values(X_actual, seed)
        return BatchObservation(
            X_expected=X_expected,
            X_actual=X_actual,
            y=y.reshape(-1),
            metadata={"kind": "synthetic_simplex", "seed": seed, "maximize": self.maximize},
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
            raise ValueError(f"endpoints must have shape (2, {self.n_components}) or (n_lines, 2, {self.n_components})")
        left = project_simplex(arr[0])
        right = project_simplex(arr[1])
        ts = np.linspace(0.0, 1.0, n_points)
        X_expected = (1.0 - ts[:, None]) * left[None, :] + ts[:, None] * right[None, :]
        X_expected = project_simplex(X_expected)
        X_actual = X_expected.copy()
        y = self._values(X_actual, seed)
        return BatchObservation(
            X_expected=X_expected,
            X_actual=X_actual,
            y=y.reshape(-1),
            metadata={
                "kind": "synthetic_simplex_line",
                "seed": seed,
                "left": left.tolist(),
                "right": right.tolist(),
            },
        )

