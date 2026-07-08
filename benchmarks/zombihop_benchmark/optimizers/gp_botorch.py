from __future__ import annotations

from typing import Any
import warnings

import numpy as np

from ..spaces import composition_to_ilr_np, sample_simplex, validate_simplex
from ..types import BatchObservation, ObjectiveInfo


class GPBoTorchOptimizer:
    supports_point = True
    supports_line = False

    def __init__(
        self,
        kind: str = "ei",
        candidate_pool_size: int = 2048,
        ucb_beta: float = 0.2,
        xi: float = 0.01,
        device: str = "cpu",
        dtype: str = "float64",
        max_train_points: int | None = None,
        **kwargs: Any,
    ) -> None:
        if kind not in {"ei", "ucb"}:
            raise ValueError("GPBoTorchOptimizer kind must be 'ei' or 'ucb'")
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")
        self.kind = kind
        self.name = f"gp_ard_{kind}"
        self.candidate_pool_size = int(candidate_pool_size)
        self.ucb_beta = float(ucb_beta)
        self.xi = float(xi)
        self.device_name = device
        self.dtype_name = dtype
        self.max_train_points = max_train_points
        self.kwargs = kwargs
        self.n_components: int | None = None
        self.seed: int | None = None
        self._suggest_calls = 0
        self._score_calls = 0
        self._fit_calls = 0
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self._torch = None

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
            raise RuntimeError("GPBoTorchOptimizer must be initialized before suggest()")
        if n_suggestions <= 0:
            raise ValueError("n_suggestions must be positive")
        torch = self._import_torch_stack()
        model = self._fit_model(torch)
        pool = sample_simplex(
            self.candidate_pool_size,
            self.n_components,
            seed=self.seed + 2_000_003 + self._suggest_calls,
        )
        values = self._score_candidates_with_model(torch, model, pool)
        order = np.argsort(values)[::-1]
        self._suggest_calls += 1
        return pool[order[:n_suggestions]]

    def score_candidates(self, X_candidates: np.ndarray) -> np.ndarray:
        if self.X is None or self.y is None:
            raise RuntimeError("GPBoTorchOptimizer must be initialized before score_candidates()")
        torch = self._import_torch_stack()
        model = self._fit_model(torch)
        values = self._score_candidates_with_model(torch, model, X_candidates)
        self._score_calls += 1
        return values

    def observe(self, obs: BatchObservation) -> None:
        if self.X is None or self.y is None:
            raise RuntimeError("GPBoTorchOptimizer must be initialized before observe()")
        validate_simplex(obs.X_actual)
        self.X = np.vstack([self.X, np.asarray(obs.X_actual, dtype=float)])
        self.y = np.concatenate([self.y, np.asarray(obs.y, dtype=float).reshape(-1)])

    def get_state(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "implemented": True,
            "candidate_pool_size": self.candidate_pool_size,
            "n_observations": 0 if self.y is None else int(self.y.shape[0]),
            "suggest_calls": self._suggest_calls,
            "score_calls": self._score_calls,
            "fit_calls": self._fit_calls,
        }

    def _training_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.X is not None and self.y is not None
        X = self.X
        y = self.y
        if self.max_train_points is not None and X.shape[0] > self.max_train_points:
            X = X[-self.max_train_points :]
            y = y[-self.max_train_points :]
        return X, y

    def _fit_model(self, torch):
        try:
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import SingleTaskGP
            from gpytorch.mlls import ExactMarginalLogLikelihood
        except ImportError as exc:
            raise ImportError("GPBoTorchOptimizer requires botorch and gpytorch") from exc

        X_np, y_np = self._training_arrays()
        X_ilr = composition_to_ilr_np(X_np)
        train_X = torch.as_tensor(X_ilr, device=self._device(torch), dtype=self._dtype(torch))
        train_Y = torch.as_tensor(y_np.reshape(-1, 1), device=self._device(torch), dtype=self._dtype(torch))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
        self._fit_calls += 1
        return model

    def _build_acquisition(self, torch, model):
        try:
            from botorch.acquisition.analytic import UpperConfidenceBound
        except ImportError as exc:
            raise ImportError("GPBoTorchOptimizer requires BoTorch analytic acquisitions") from exc
        assert self.y is not None
        if self.kind == "ucb":
            return UpperConfidenceBound(model, beta=self.ucb_beta)
        best_f = float(np.max(self.y)) + self.xi
        try:
            from botorch.acquisition.analytic import LogExpectedImprovement

            return LogExpectedImprovement(model, best_f=best_f, maximize=True)
        except ImportError:
            from botorch.acquisition.analytic import ExpectedImprovement

            return ExpectedImprovement(model, best_f=best_f, maximize=True)

    def _score_candidates_with_model(self, torch, model, X_candidates: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X_candidates, dtype=float)
        validate_simplex(X_arr)
        pool_ilr = composition_to_ilr_np(X_arr)
        Xq = torch.as_tensor(pool_ilr, device=self._device(torch), dtype=self._dtype(torch)).unsqueeze(1)
        acq = self._build_acquisition(torch, model)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            values = acq(Xq).detach().reshape(-1).cpu().numpy()
        return np.asarray(values, dtype=float)

    def _import_torch_stack(self):
        if self._torch is not None:
            return self._torch
        try:
            import torch
            import botorch  # noqa: F401
            import gpytorch  # noqa: F401
        except ImportError as exc:
            raise ImportError("GPBoTorchOptimizer requires torch, botorch, and gpytorch") from exc
        self._torch = torch
        return torch

    def _device(self, torch):
        return torch.device(self.device_name)

    def _dtype(self, torch):
        try:
            return getattr(torch, self.dtype_name)
        except AttributeError as exc:
            raise ValueError(f"Unknown torch dtype {self.dtype_name!r}") from exc
