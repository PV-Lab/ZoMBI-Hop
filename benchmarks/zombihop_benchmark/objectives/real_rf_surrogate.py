from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..spaces import composition_to_ilr_np, ilr_distance, project_simplex, sample_simplex, validate_simplex
from ..types import BatchObservation, ObjectiveInfo


@dataclass
class RealRFSurrogateObjective:
    name: str = "real_3d_perovskite_rf"
    n_components: int = 3
    maximize: bool = True
    model_path: str | None = None
    model_format: str = "auto"
    data_path: str | None = None
    grid_path: str | None = None
    component_labels: list[str] | None = None
    composition_columns: list[str] | None = None
    target_column: str = "Objective"
    feature_transform: str = "raw_composition"
    train_surrogate: dict[str, Any] = field(default_factory=dict)
    needle_detection: dict[str, Any] = field(default_factory=dict)
    true_needles_path: str | None = None

    def __post_init__(self) -> None:
        if self.n_components != 3:
            raise ValueError("RealRFSurrogateObjective currently supports 3-component simplex inputs")
        self.component_labels = list(self.component_labels or ["FAPbI3", "MAPbI3", "MAPbBr3"])
        self.composition_columns = list(self.composition_columns or self.component_labels)
        if len(self.component_labels) != self.n_components:
            raise ValueError("component_labels length must match n_components")
        if len(self.composition_columns) != self.n_components:
            raise ValueError("composition_columns length must match n_components")
        if self.feature_transform not in {"raw_composition", "ilr"}:
            raise ValueError("feature_transform must be 'raw_composition' or 'ilr'")

        self._source_path = self.model_path or self.data_path or self.grid_path
        self._source_kind = ""
        self._model = None
        self._grid_X: np.ndarray | None = None
        self._grid_y: np.ndarray | None = None
        self._load_or_train()

        true_needles, needle_source = self._load_or_detect_needles()
        self.true_needles = true_needles
        self._needle_source = needle_source
        self._true_needle_y = self._predict_internal(self.true_needles) if len(self.true_needles) else np.array([])
        self.info = ObjectiveInfo(
            name=self.name,
            n_components=self.n_components,
            maximize=True,
            true_needles=self.true_needles.copy(),
            y_star=float(np.max(self._true_needle_y)) if len(self._true_needle_y) else None,
            match_radius_ilr=self.needle_detection.get("match_radius_ilr", 0.25),
            match_radius_comp=self.needle_detection.get("match_radius_comp", 0.05),
        )

    def initial_design(self, n: int, seed: int) -> np.ndarray:
        return sample_simplex(n, self.n_components, seed)

    def evaluate_points(self, X_expected: np.ndarray, seed: int | None = None) -> BatchObservation:
        X_expected = project_simplex(X_expected)
        if X_expected.ndim == 1:
            X_expected = X_expected.reshape(1, -1)
        validate_simplex(X_expected, atol=1e-5)
        y = self._predict_internal(X_expected)
        return BatchObservation(
            X_expected=X_expected,
            X_actual=X_expected.copy(),
            y=y.reshape(-1),
            metadata={
                "kind": "real_rf_surrogate",
                "seed": seed,
                "source_kind": self._source_kind,
                "objective_source_path": self._source_path,
                "feature_transform": self.feature_transform,
                "component_labels": self.component_labels,
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
        X_line = (1.0 - ts[:, None]) * left[None, :] + ts[:, None] * right[None, :]
        obs = self.evaluate_points(X_line, seed=seed)
        return BatchObservation(
            X_expected=obs.X_expected,
            X_actual=obs.X_actual,
            y=obs.y,
            metadata={**dict(obs.metadata), "left": left.tolist(), "right": right.tolist()},
        )

    def get_metadata(self) -> dict[str, Any]:
        return {
            "objective_kind": "real_rf_surrogate",
            "objective_source_path": self._source_path,
            "source_kind": self._source_kind,
            "model_path": self.model_path,
            "data_path": self.data_path,
            "grid_path": self.grid_path,
            "component_labels": self.component_labels,
            "composition_columns": self.composition_columns,
            "target_column": self.target_column,
            "feature_transform": self.feature_transform,
            "maximize_original_target": self.maximize,
            "needle_detection_method": self.needle_detection.get("method", "grid_local_maxima"),
            "needle_source": self._needle_source,
            "grid_resolution": self.needle_detection.get("grid_resolution", 151),
            "num_true_needles": int(len(self.true_needles)),
            "true_needle_best_y": float(np.max(self._true_needle_y)) if len(self._true_needle_y) else None,
            "true_needle_worst_y": float(np.min(self._true_needle_y)) if len(self._true_needle_y) else None,
        }

    def true_needle_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        order = np.argsort(self._true_needle_y)[::-1] if len(self._true_needle_y) else []
        for rank, idx in enumerate(order):
            x = self.true_needles[int(idx)]
            row = {
                "rank": int(rank),
                "y": float(self._true_needle_y[int(idx)]),
                "source": self._needle_source,
            }
            for i, value in enumerate(x):
                row[f"x_{i}"] = float(value)
                row[f"component_{self.component_labels[i]}"] = float(value)
            rows.append(row)
        return rows

    def _load_or_train(self) -> None:
        if self.model_path:
            self._model = _load_model(Path(self.model_path), self.model_format)
            if not hasattr(self._model, "predict"):
                raise ValueError(f"Loaded model from {self.model_path!r} does not provide predict(...)")
            self._source_kind = "model"
            return

        table_path = self.data_path or self.grid_path
        if not table_path:
            raise ValueError("real_rf_surrogate requires model_path, data_path, or grid_path")
        X, y = self._load_table(Path(table_path))
        self._grid_X = X
        self._grid_y = y
        train_cfg = dict(self.train_surrogate or {})
        train_enabled = bool(train_cfg.get("enabled", self.data_path is not None))
        if train_enabled:
            try:
                from sklearn.ensemble import RandomForestRegressor
            except ImportError as exc:
                raise ImportError("RealRFSurrogateObjective requires scikit-learn to train a table RF") from exc
            n_estimators = int(train_cfg.get("n_estimators", 500))
            random_state = int(train_cfg.get("random_state", 42))
            min_samples_leaf = int(train_cfg.get("min_samples_leaf", 1))
            n_jobs = int(train_cfg.get("n_jobs", -1))
            self._model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                min_samples_leaf=min_samples_leaf,
                n_jobs=n_jobs,
            )
            self._model.fit(self._features(X), y)
            self._source_kind = "table_random_forest" if self.data_path else "grid_random_forest"
        else:
            self._source_kind = "grid_nearest"

    def _load_table(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("RealRFSurrogateObjective requires pandas to load table/grid data") from exc
        df = pd.read_csv(path)
        missing = [c for c in [*self.composition_columns, self.target_column] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")
        sub = df[[*self.composition_columns, self.target_column]].dropna()
        X = project_simplex(sub[self.composition_columns].to_numpy(dtype=float))
        validate_simplex(X, atol=1e-4)
        y = sub[self.target_column].to_numpy(dtype=float).reshape(-1)
        return X, y

    def _load_or_detect_needles(self) -> tuple[np.ndarray, str]:
        if self.true_needles_path:
            needles = _load_true_needles(Path(self.true_needles_path), self.n_components)
            if len(needles):
                return needles, str(self.true_needles_path)
        return self._detect_needles(), "grid_local_maxima"

    def _detect_needles(self) -> np.ndarray:
        cfg = dict(self.needle_detection or {})
        resolution = int(cfg.get("grid_resolution", 151))
        top_k = int(cfg.get("top_k", 20))
        min_sep = float(cfg.get("min_separation_ilr", 0.08))
        threshold_quantile = cfg.get("threshold_quantile")
        X_grid = simplex_grid_3d(resolution)
        y_grid = self._predict_internal(X_grid)
        order = np.argsort(y_grid)[::-1]
        if threshold_quantile is not None:
            threshold = float(np.quantile(y_grid, float(threshold_quantile)))
            order = np.asarray([idx for idx in order if y_grid[idx] >= threshold], dtype=int)
        selected: list[np.ndarray] = []
        for idx in order:
            candidate = X_grid[int(idx)]
            if not selected:
                selected.append(candidate)
            else:
                dists = ilr_distance(candidate.reshape(1, -1), np.vstack(selected))
                if float(np.min(dists)) >= min_sep:
                    selected.append(candidate)
            if len(selected) >= top_k:
                break
        return np.vstack(selected) if selected else np.empty((0, self.n_components), dtype=float)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        X = project_simplex(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self._model is not None:
            y = np.asarray(self._model.predict(self._features(X)), dtype=float).reshape(-1)
            return y if self.maximize else -y
        if self._grid_X is None or self._grid_y is None:
            raise RuntimeError("Real RF objective has neither a model nor grid values")
        dists = ilr_distance(X, self._grid_X)
        idx = np.argmin(dists, axis=1)
        y = self._grid_y[idx].reshape(-1)
        return y if self.maximize else -y

    def _features(self, X: np.ndarray) -> np.ndarray:
        if self.feature_transform == "raw_composition":
            return np.asarray(X, dtype=float)
        return composition_to_ilr_np(X)


def simplex_grid_3d(resolution: int) -> np.ndarray:
    resolution = int(resolution)
    if resolution < 1:
        raise ValueError("grid resolution must be at least 1")
    pts = []
    for i in range(resolution + 1):
        for j in range(resolution + 1 - i):
            k = resolution - i - j
            pts.append([i / resolution, j / resolution, k / resolution])
    return np.asarray(pts, dtype=float)


def _load_model(path: Path, model_format: str):
    fmt = model_format.lower()
    if fmt == "auto":
        fmt = "joblib" if path.suffix.lower() in {".joblib", ".pkl", ".pickle"} else "pickle"
    if fmt == "joblib":
        try:
            import joblib
            return joblib.load(path)
        except Exception:
            pass
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_true_needles(path: Path, n_components: int) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    values = payload.get("true_optima") or payload.get("true_needles") or payload.get("needles") or []
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.empty((0, n_components), dtype=float)
    arr = arr.reshape(-1, n_components)
    return project_simplex(arr)
