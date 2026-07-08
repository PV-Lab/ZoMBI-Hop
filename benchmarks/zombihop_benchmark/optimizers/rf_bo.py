from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..spaces import composition_to_ilr_np, sample_simplex, validate_simplex
from ..types import BatchObservation, ObjectiveInfo


class RFBOOptimizer:
    name = "rf_bo"
    supports_point = True
    supports_line = False

    def __init__(
        self,
        candidate_pool_size: int = 4096,
        n_estimators: int = 200,
        max_features: str | float | None = "sqrt",
        min_samples_leaf: int = 1,
        acquisition: str = "ucb",
        ucb_beta: float = 0.2,
        xi: float = 0.01,
        random_state: int | None = None,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")
        if acquisition not in {"ucb", "ei"}:
            raise ValueError("RFBOOptimizer acquisition must be 'ucb' or 'ei'")
        self.candidate_pool_size = int(candidate_pool_size)
        self.n_estimators = int(n_estimators)
        self.max_features = max_features
        self.min_samples_leaf = int(min_samples_leaf)
        self.acquisition = acquisition
        self.ucb_beta = float(ucb_beta)
        self.xi = float(xi)
        self.random_state = random_state
        self.n_jobs = int(n_jobs)
        self.kwargs = kwargs
        self.n_components: int | None = None
        self.seed: int | None = None
        self._suggest_calls = 0
        self._score_calls = 0
        self._fit_calls = 0
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def initialize(self, X: np.ndarray, y: np.ndarray, objective_info: ObjectiveInfo, seed: int) -> None:
        X_arr = np.asarray(X, dtype=float)
        validate_simplex(X_arr)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        self.n_components = objective_info.n_components
        self.seed = int(seed)
        self._suggest_calls = 0
        self._score_calls = 0
        self._fit_calls = 0
        self.X = X_arr.copy()
        self.y = y_arr.copy()

    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        if self.X is None or self.y is None or self.n_components is None or self.seed is None:
            raise RuntimeError("RFBOOptimizer must be initialized before suggest()")
        if n_suggestions <= 0:
            raise ValueError("n_suggestions must be positive")
        model = self._fit_model()
        pool = sample_simplex(
            self.candidate_pool_size,
            self.n_components,
            seed=self.seed + 3_000_007 + self._suggest_calls,
        )
        values = self._score_candidates_with_model(model, pool)
        order = np.argsort(values)[::-1]
        self._suggest_calls += 1
        return pool[order[:n_suggestions]]

    def score_candidates(self, X_candidates: np.ndarray) -> np.ndarray:
        if self.X is None or self.y is None:
            raise RuntimeError("RFBOOptimizer must be initialized before score_candidates()")
        model = self._fit_model()
        values = self._score_candidates_with_model(model, X_candidates)
        self._score_calls += 1
        return values

    def observe(self, obs: BatchObservation) -> None:
        if self.X is None or self.y is None:
            raise RuntimeError("RFBOOptimizer must be initialized before observe()")
        validate_simplex(obs.X_actual)
        self.X = np.vstack([self.X, np.asarray(obs.X_actual, dtype=float)])
        self.y = np.concatenate([self.y, np.asarray(obs.y, dtype=float).reshape(-1)])

    def get_state(self) -> dict[str, object]:
        return {
            "name": self.name,
            "implemented": True,
            "candidate_pool_size": self.candidate_pool_size,
            "n_estimators": self.n_estimators,
            "acquisition": self.acquisition,
            "n_observations": 0 if self.y is None else int(self.y.shape[0]),
            "suggest_calls": self._suggest_calls,
            "score_calls": self._score_calls,
            "fit_calls": self._fit_calls,
        }

    def _fit_model(self):
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError as exc:
            raise ImportError("RFBOOptimizer requires scikit-learn") from exc
        assert self.X is not None and self.y is not None
        random_state = self.random_state if self.random_state is not None else self.seed
        model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf,
            random_state=random_state,
            n_jobs=self.n_jobs,
            **self.kwargs,
        )
        model.fit(composition_to_ilr_np(self.X), self.y)
        self._fit_calls += 1
        return model

    def _score_candidates_with_model(self, model, X_candidates: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X_candidates, dtype=float)
        validate_simplex(X_arr)
        pool_ilr = composition_to_ilr_np(X_arr)
        tree_preds = np.vstack([tree.predict(pool_ilr) for tree in model.estimators_])
        mean = tree_preds.mean(axis=0)
        std = tree_preds.std(axis=0)
        if self.acquisition == "ucb":
            values = mean + self.ucb_beta * std
        else:
            assert self.y is not None
            values = _expected_improvement(mean, std, best_f=float(np.max(self.y)), xi=self.xi)
        return np.asarray(values, dtype=float)


def _expected_improvement(mean: np.ndarray, std: np.ndarray, best_f: float, xi: float) -> np.ndarray:
    sigma = np.maximum(std, 1e-12)
    improvement = mean - best_f - xi
    z = improvement / sigma
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    ei = improvement * cdf + sigma * pdf
    return np.where(std <= 1e-12, np.maximum(improvement, 0.0), ei)
