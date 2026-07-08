from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np

from ..spaces import composition_to_ilr_np, sample_simplex, validate_simplex
from ..types import BatchObservation, ObjectiveInfo


@dataclass
class TurboTrustRegionState:
    length: float = 0.8
    length_initial: float = 0.8
    length_min: float = 0.01
    length_max: float = 1.6
    success_counter: int = 0
    failure_counter: int = 0
    success_tolerance: int = 3
    failure_tolerance: int = 3
    best_value: float | None = None
    restart_count: int = 0
    restart_on_min_length: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "TurboTrustRegionState":
        cfg = dict(config or {})
        length_initial = float(cfg.get("length_initial", cfg.get("initial_length", 0.8)))
        return cls(
            length=length_initial,
            length_initial=length_initial,
            length_min=float(cfg.get("length_min", cfg.get("min_length", 0.01))),
            length_max=float(cfg.get("length_max", cfg.get("max_length", 1.6))),
            success_tolerance=int(cfg.get("success_tolerance", 3)),
            failure_tolerance=int(cfg.get("failure_tolerance", 3)),
            restart_on_min_length=bool(cfg.get("restart_on_min_length", True)),
        )

    def initialize_best(self, values: np.ndarray, maximize: bool = True) -> None:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            self.best_value = None
            return
        self.best_value = float(np.max(arr) if maximize else np.min(arr))

    def update(self, new_values: np.ndarray, maximize: bool = True, improvement_tolerance: float = 1e-8) -> bool:
        arr = np.asarray(new_values, dtype=float).reshape(-1)
        if arr.size == 0:
            return False
        new_best = float(np.max(arr) if maximize else np.min(arr))
        if self.best_value is None:
            self.best_value = new_best
            return True

        scale = max(1.0, abs(float(self.best_value)))
        tol = float(improvement_tolerance) * scale
        improved = new_best > self.best_value + tol if maximize else new_best < self.best_value - tol
        if improved:
            self.best_value = new_best
            self.success_counter += 1
            self.failure_counter = 0
            if self.success_counter >= self.success_tolerance:
                self.length = min(2.0 * self.length, self.length_max)
                self.success_counter = 0
        else:
            self.failure_counter += 1
            self.success_counter = 0
            if self.failure_counter >= self.failure_tolerance:
                self.length *= 0.5
                self.failure_counter = 0
                if self.length < self.length_min:
                    if self.restart_on_min_length:
                        self.restart_count += 1
                        self.length = self.length_initial
                        self.success_counter = 0
                        self.failure_counter = 0
                    else:
                        self.length = self.length_min
        return improved

    def as_dict(self) -> dict[str, Any]:
        return {
            "length": float(self.length),
            "length_initial": float(self.length_initial),
            "length_min": float(self.length_min),
            "length_max": float(self.length_max),
            "success_counter": int(self.success_counter),
            "failure_counter": int(self.failure_counter),
            "success_tolerance": int(self.success_tolerance),
            "failure_tolerance": int(self.failure_tolerance),
            "best_value": self.best_value,
            "restart_count": int(self.restart_count),
            "restart_on_min_length": bool(self.restart_on_min_length),
        }


class TuRBOOptimizer:
    name = "turbo"
    supports_point = True
    supports_line = False

    def __init__(
        self,
        internal_space: str = "ilr",
        acquisition: str = "log_ei",
        candidate_pool_size: int = 2048,
        min_tr_candidates: int = 64,
        trust_region: dict[str, Any] | None = None,
        ilr_bounds: dict[str, Any] | None = None,
        xi: float = 0.01,
        ucb_beta: float = 0.2,
        device: str = "cpu",
        dtype: str = "float64",
        max_train_points: int | None = None,
        improvement_tolerance: float = 1e-8,
        line_adapter_label: str = "turbo_acq_line",
        **kwargs: Any,
    ) -> None:
        if internal_space != "ilr":
            raise ValueError("TuRBOOptimizer currently supports internal_space='ilr' only")
        if acquisition not in {"log_ei", "ei", "ucb", "posterior_mean", "thompson_sampling"}:
            raise ValueError(
                "TuRBOOptimizer acquisition must be one of: "
                "'log_ei', 'ei', 'ucb', 'posterior_mean', 'thompson_sampling'"
            )
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")
        if min_tr_candidates <= 0:
            raise ValueError("min_tr_candidates must be positive")

        self.internal_space = internal_space
        self.acquisition = acquisition
        self.candidate_pool_size = int(candidate_pool_size)
        self.min_tr_candidates = int(min_tr_candidates)
        self.ilr_bounds_cfg = dict(ilr_bounds or {})
        self.xi = float(xi)
        self.ucb_beta = float(ucb_beta)
        self.device_name = device
        self.dtype_name = dtype
        self.max_train_points = max_train_points
        self.improvement_tolerance = float(improvement_tolerance)
        self.line_adapter_label = line_adapter_label
        self.extra_kwargs = kwargs
        self.state = TurboTrustRegionState.from_config(trust_region)

        self.n_components: int | None = None
        self.seed: int | None = None
        self.maximize = True
        self.ilr_lower: np.ndarray | None = None
        self.ilr_upper: np.ndarray | None = None
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self._suggest_calls = 0
        self._score_calls = 0
        self._fit_calls = 0
        self._last_candidates_in_tr = 0
        self._last_candidate_pool_size = 0
        self._last_tr_fallback = False
        self._torch = None

    def initialize(self, X: np.ndarray, y: np.ndarray, objective_info: ObjectiveInfo, seed: int) -> None:
        X_arr = np.asarray(X, dtype=float)
        validate_simplex(X_arr)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        self.n_components = int(objective_info.n_components)
        self.seed = int(seed)
        self.maximize = bool(objective_info.maximize)
        self.X = X_arr.copy()
        self.y = y_arr.copy()
        self._suggest_calls = 0
        self._score_calls = 0
        self._fit_calls = 0
        self._last_candidates_in_tr = 0
        self._last_candidate_pool_size = 0
        self._last_tr_fallback = False
        self.ilr_lower, self.ilr_upper = self._derive_ilr_bounds(self.X)
        self.state.initialize_best(self.y, maximize=self.maximize)

    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        if self.X is None or self.y is None or self.n_components is None or self.seed is None:
            raise RuntimeError("TuRBOOptimizer must be initialized before suggest()")
        if n_suggestions <= 0:
            raise ValueError("n_suggestions must be positive")

        torch = self._import_torch_stack()
        model = self._fit_model(torch)
        pool = self._trust_region_candidate_pool(n_suggestions)
        values = self._score_candidates_with_model(torch, model, pool, penalize_outside=True)
        order = np.argsort(values)[::-1]
        self._suggest_calls += 1
        return pool[order[:n_suggestions]]

    def score_candidates(self, X_candidates: np.ndarray) -> np.ndarray:
        if self.X is None or self.y is None:
            raise RuntimeError("TuRBOOptimizer must be initialized before score_candidates()")
        torch = self._import_torch_stack()
        model = self._fit_model(torch)
        values = self._score_candidates_with_model(torch, model, X_candidates, penalize_outside=True)
        self._score_calls += 1
        return values

    def observe(self, obs: BatchObservation) -> None:
        if self.X is None or self.y is None:
            raise RuntimeError("TuRBOOptimizer must be initialized before observe()")
        validate_simplex(obs.X_actual)
        X_new = np.asarray(obs.X_actual, dtype=float)
        y_new = np.asarray(obs.y, dtype=float).reshape(-1)
        self.state.update(y_new, maximize=self.maximize, improvement_tolerance=self.improvement_tolerance)
        self.X = np.vstack([self.X, X_new])
        self.y = np.concatenate([self.y, y_new])

    def line_metadata(self) -> dict[str, Any]:
        return {
            "line_adapter": self.line_adapter_label,
            "line_adapter_caveat": _TURBO_LINE_CAVEAT,
            "turbo_line_score_coordinate_system": "ilr_trust_region_acquisition_on_raw_simplex_candidates",
        }

    def get_state(self) -> dict[str, Any]:
        center_raw = self._center_raw()
        center_ilr = None if center_raw is None else composition_to_ilr_np(center_raw).tolist()
        return {
            "name": self.name,
            "implemented": True,
            "algorithm": "benchmark_local_finite_pool_turbo_1",
            "dependency": "torch+botorch+gpytorch",
            "dependency_available": self._torch is not None,
            "internal_space": self.internal_space,
            "objective_space": "raw_simplex",
            "acquisition": self.acquisition,
            "candidate_pool_size": self.candidate_pool_size,
            "min_tr_candidates": self.min_tr_candidates,
            "n_components": self.n_components,
            "n_observations": 0 if self.y is None else int(self.y.shape[0]),
            "suggest_calls": self._suggest_calls,
            "score_calls": self._score_calls,
            "fit_calls": self._fit_calls,
            "last_candidate_pool_size": int(self._last_candidate_pool_size),
            "last_candidates_in_tr": int(self._last_candidates_in_tr),
            "last_tr_fallback": bool(self._last_tr_fallback),
            "turbo_state": self.state.as_dict(),
            "turbo_length": float(self.state.length),
            "turbo_length_min": float(self.state.length_min),
            "turbo_length_max": float(self.state.length_max),
            "turbo_success_counter": int(self.state.success_counter),
            "turbo_failure_counter": int(self.state.failure_counter),
            "turbo_restart_count": int(self.state.restart_count),
            "turbo_center_raw": None if center_raw is None else center_raw.tolist(),
            "turbo_center_ilr": center_ilr,
            "ilr_lower": None if self.ilr_lower is None else self.ilr_lower.tolist(),
            "ilr_upper": None if self.ilr_upper is None else self.ilr_upper.tolist(),
            "line_adapter": self.line_adapter_label,
            "line_adapter_caveat": _TURBO_LINE_CAVEAT,
        }

    def _training_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.X is not None and self.y is not None
        X = self.X
        y = self._model_y()
        if self.max_train_points is not None and X.shape[0] > self.max_train_points:
            X = X[-self.max_train_points :]
            y = y[-self.max_train_points :]
        return X, y

    def _model_y(self) -> np.ndarray:
        assert self.y is not None
        return self.y if self.maximize else -self.y

    def _fit_model(self, torch):
        try:
            from botorch.fit import fit_gpytorch_mll
            from botorch.models import SingleTaskGP
            from gpytorch.mlls import ExactMarginalLogLikelihood
        except ImportError as exc:
            raise ImportError("TuRBOOptimizer requires botorch and gpytorch") from exc

        X_np, y_np = self._training_arrays()
        train_X_np = self._composition_to_normalized_ilr(X_np, clip=True)
        train_X = torch.as_tensor(train_X_np, device=self._device(torch), dtype=self._dtype(torch))
        train_Y = torch.as_tensor(y_np.reshape(-1, 1), device=self._device(torch), dtype=self._dtype(torch))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
        self._fit_calls += 1
        return model

    def _score_candidates_with_model(
        self,
        torch,
        model,
        X_candidates: np.ndarray,
        penalize_outside: bool,
    ) -> np.ndarray:
        X_arr = np.asarray(X_candidates, dtype=float)
        validate_simplex(X_arr)
        norm_unclipped = self._composition_to_normalized_ilr(X_arr, clip=False)
        norm_clipped = np.clip(norm_unclipped, 0.0, 1.0)

        if self.acquisition == "posterior_mean":
            X_eval = torch.as_tensor(norm_clipped, device=self._device(torch), dtype=self._dtype(torch))
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                values = model.posterior(X_eval).mean.detach().reshape(-1).cpu().numpy()
        elif self.acquisition == "thompson_sampling":
            X_eval = torch.as_tensor(norm_clipped, device=self._device(torch), dtype=self._dtype(torch))
            seed = int((self.seed or 0) + 7_000_019 + 53 * self._suggest_calls + 101 * self._score_calls)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                with torch.no_grad(), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    values = model.posterior(X_eval).rsample(sample_shape=torch.Size([1])).detach()
            values = values.reshape(-1).cpu().numpy()
        else:
            Xq = torch.as_tensor(norm_clipped, device=self._device(torch), dtype=self._dtype(torch)).unsqueeze(1)
            acq = self._build_acquisition(torch, model)
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                values = acq(Xq).detach().reshape(-1).cpu().numpy()

        out = np.asarray(values, dtype=float)
        if penalize_outside:
            inside = self._inside_tr_from_normalized(norm_unclipped)
            if not bool(np.all(inside)):
                out = out.copy()
                finite = out[np.isfinite(out)]
                if finite.size:
                    span = max(1.0, float(np.max(finite) - np.min(finite)))
                    floor = float(np.min(finite) - span)
                else:
                    floor = -1.0e12
                out[~inside] = floor
        return out

    def _build_acquisition(self, torch, model):
        if self.acquisition == "ucb":
            from botorch.acquisition.analytic import UpperConfidenceBound

            return UpperConfidenceBound(model, beta=self.ucb_beta)

        assert self.y is not None
        best_f = float(np.max(self._model_y())) + self.xi
        if self.acquisition == "ei":
            from botorch.acquisition.analytic import ExpectedImprovement

            return ExpectedImprovement(model, best_f=best_f, maximize=True)

        try:
            from botorch.acquisition.analytic import LogExpectedImprovement

            return LogExpectedImprovement(model, best_f=best_f, maximize=True)
        except ImportError:
            from botorch.acquisition.analytic import ExpectedImprovement

            return ExpectedImprovement(model, best_f=best_f, maximize=True)

    def _trust_region_candidate_pool(self, n_suggestions: int) -> np.ndarray:
        assert self.n_components is not None and self.seed is not None
        n_draws = max(self.candidate_pool_size, self.min_tr_candidates, n_suggestions)
        pool = sample_simplex(
            n_draws,
            self.n_components,
            seed=self.seed + 5_000_011 + self._suggest_calls,
        )
        norm = self._composition_to_normalized_ilr(pool, clip=False)
        inside = self._inside_tr_from_normalized(norm)
        self._last_candidate_pool_size = int(pool.shape[0])
        self._last_candidates_in_tr = int(np.sum(inside))
        self._last_tr_fallback = False
        if self._last_candidates_in_tr >= max(self.min_tr_candidates, n_suggestions):
            return pool[inside]

        center = self._center_normalized()
        distances = np.linalg.norm(norm - center[None, :], axis=1)
        order = np.argsort(distances)
        n_take = min(pool.shape[0], max(self.min_tr_candidates, n_suggestions))
        self._last_tr_fallback = True
        return pool[order[:n_take]]

    def _inside_tr_from_normalized(self, normalized_ilr: np.ndarray) -> np.ndarray:
        arr = np.asarray(normalized_ilr, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        center = self._center_normalized()
        finite = np.isfinite(arr).all(axis=1)
        return finite & np.all(np.abs(arr - center[None, :]) <= 0.5 * self.state.length, axis=1)

    def _center_raw(self) -> np.ndarray | None:
        if self.X is None or self.y is None or self.y.size == 0:
            return None
        idx = int(np.argmax(self.y) if self.maximize else np.argmin(self.y))
        return np.asarray(self.X[idx], dtype=float)

    def _center_normalized(self) -> np.ndarray:
        center_raw = self._center_raw()
        if center_raw is None:
            assert self.n_components is not None
            return np.full(self.n_components - 1, 0.5, dtype=float)
        return self._composition_to_normalized_ilr(center_raw.reshape(1, -1), clip=True).reshape(-1)

    def _derive_ilr_bounds(self, observed_X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.n_components is not None and self.seed is not None
        cfg = self.ilr_bounds_cfg
        if "lower" in cfg and "upper" in cfg:
            lower = np.asarray(cfg["lower"], dtype=float)
            upper = np.asarray(cfg["upper"], dtype=float)
        else:
            n_candidates = int(cfg.get("n_candidates", 4096))
            lower_q = float(cfg.get("lower_quantile", 0.005))
            upper_q = float(cfg.get("upper_quantile", 0.995))
            pool = sample_simplex(n_candidates, self.n_components, seed=self.seed + 8_500_037)
            pool_ilr = composition_to_ilr_np(pool)
            lower = np.quantile(pool_ilr, lower_q, axis=0)
            upper = np.quantile(pool_ilr, upper_q, axis=0)

        observed_ilr = composition_to_ilr_np(observed_X)
        observed_min = np.min(observed_ilr, axis=0)
        observed_max = np.max(observed_ilr, axis=0)
        span = np.maximum(upper - lower, 1.0)
        lower = np.minimum(lower, observed_min - 0.05 * span)
        upper = np.maximum(upper, observed_max + 0.05 * span)

        if lower.shape != (self.n_components - 1,) or upper.shape != (self.n_components - 1,):
            raise ValueError("TuRBO ILR bounds must have shape (n_components - 1,)")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or not np.all(lower < upper):
            raise ValueError("TuRBO ILR bounds must be finite and strictly ordered")
        return lower, upper

    def _composition_to_normalized_ilr(self, X: np.ndarray, clip: bool) -> np.ndarray:
        if self.ilr_lower is None or self.ilr_upper is None:
            raise RuntimeError("TuRBO ILR bounds are not initialized")
        z = composition_to_ilr_np(X)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        span = self.ilr_upper - self.ilr_lower
        normalized = (z - self.ilr_lower[None, :]) / span[None, :]
        if clip:
            normalized = np.clip(normalized, 0.0, 1.0)
        return normalized

    def _import_torch_stack(self):
        if self._torch is not None:
            return self._torch
        try:
            import torch
            import botorch  # noqa: F401
            import gpytorch  # noqa: F401
        except ImportError as exc:
            raise ImportError("TuRBOOptimizer requires torch, botorch, and gpytorch") from exc
        self._torch = torch
        return torch

    def _device(self, torch):
        return torch.device(self.device_name)

    def _dtype(self, torch):
        try:
            return getattr(torch, self.dtype_name)
        except AttributeError as exc:
            raise ValueError(f"Unknown torch dtype {self.dtype_name!r}") from exc


_TURBO_LINE_CAVEAT = (
    "TuRBO is point-native here; line mode uses the benchmark line wrapper to score deterministic "
    "simplex candidate lines by mean TuRBO acquisition, with points outside the current ILR trust "
    "region downweighted before the full line is batch-observed."
)
