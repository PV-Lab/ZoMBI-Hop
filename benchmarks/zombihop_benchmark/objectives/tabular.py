from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..spaces import ilr_distance, project_simplex, sample_simplex, validate_simplex
from ..types import BatchObservation, ObjectiveInfo


@dataclass
class TabularObjective:
    name: str
    n_components: int
    csv_path: str
    composition_columns: list[str]
    target_column: str
    maximize: bool = True
    match_radius_ilr: float | None = None
    match_radius_comp: float | None = None

    def __post_init__(self) -> None:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("TabularObjective requires pandas to load CSV datasets") from exc
        df = pd.read_csv(self.csv_path)
        missing = [c for c in [*self.composition_columns, self.target_column] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {self.csv_path}: {missing}")
        self.X = project_simplex(df[self.composition_columns].to_numpy(dtype=float))
        validate_simplex(self.X, atol=1e-5)
        y = df[self.target_column].to_numpy(dtype=float).reshape(-1)
        self.y = y if self.maximize else -y
        self.info = ObjectiveInfo(
            name=self.name,
            n_components=self.n_components,
            maximize=True,
            true_needles=None,
            y_star=float(np.max(self.y)),
            match_radius_ilr=self.match_radius_ilr,
            match_radius_comp=self.match_radius_comp,
        )

    def initial_design(self, n: int, seed: int) -> np.ndarray:
        return sample_simplex(n, self.n_components, seed)

    def evaluate_points(self, X_expected: np.ndarray, seed: int | None = None) -> BatchObservation:
        X_expected = project_simplex(X_expected)
        if X_expected.ndim == 1:
            X_expected = X_expected.reshape(1, -1)
        dists = ilr_distance(X_expected, self.X)
        idx = np.argmin(dists, axis=1)
        X_actual = self.X[idx]
        return BatchObservation(
            X_expected=X_expected,
            X_actual=X_actual,
            y=self.y[idx],
            metadata={"kind": "tabular_nearest", "indices": idx.tolist()},
        )

    def evaluate_line(self, endpoints: np.ndarray, n_points: int, seed: int | None = None) -> BatchObservation:
        arr = np.asarray(endpoints, dtype=float)
        if arr.ndim == 3:
            arr = arr[0]
        ts = np.linspace(0.0, 1.0, n_points)
        X_line = (1.0 - ts[:, None]) * project_simplex(arr[0])[None, :] + ts[:, None] * project_simplex(arr[1])[None, :]
        return self.evaluate_points(X_line, seed=seed)

