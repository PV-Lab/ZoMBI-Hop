from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import time
import warnings
from typing import Any, Callable

import numpy as np

from ..spaces import composition_to_ilr_np, sample_simplex, validate_simplex
from ..types import BatchObservation, ObjectiveInfo


_REQUIRED_MODULES = ["torch", "botorch", "gpytorch", "jax", "jaxlib", "numpyro"]


def get_saasbo_dependency_status() -> dict[str, Any]:
    modules: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for module_name in _REQUIRED_MODULES:
        found = importlib.util.find_spec(module_name) is not None
        version = None
        if found:
            version = _module_version(module_name)
        else:
            missing.append(module_name)
        modules[module_name] = {"available": found, "version": version}

    symbols: dict[str, bool] = {}
    try:
        from botorch.fit import fit_fully_bayesian_model_nuts  # noqa: F401

        symbols["fit_fully_bayesian_model_nuts"] = True
    except Exception:
        symbols["fit_fully_bayesian_model_nuts"] = False
    try:
        from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP  # noqa: F401

        symbols["SaasFullyBayesianSingleTaskGP"] = True
    except Exception:
        symbols["SaasFullyBayesianSingleTaskGP"] = False

    available = not missing and all(symbols.values())
    reason = "" if available else "missing required modules or symbols: " + ", ".join(
        [*missing, *[key for key, value in symbols.items() if not value]]
    )
    return {
        "available": bool(available),
        "backend": "botorch_saas_fully_bayesian",
        "required_modules": list(_REQUIRED_MODULES),
        "modules": modules,
        "symbols": symbols,
        "missing": missing,
        "reason": reason,
        "install_hint": '.\\.venv\\Scripts\\python -m pip install "botorch[fully_bayesian]"',
    }


def is_saasbo_available() -> bool:
    return bool(get_saasbo_dependency_status()["available"])


class SAASBOOptimizer:
    name = "saasbo"
    supports_point = True
    supports_line = False

    def __init__(
        self,
        internal_space: str = "ilr",
        acquisition: str = "log_ei",
        candidate_pool_size: int = 512,
        ilr_bounds: dict[str, Any] | None = None,
        warmup_steps: int = 32,
        num_samples: int = 16,
        thinning: int = 8,
        max_tree_depth: int = 6,
        train_yvar: float = 1.0e-6,
        max_train_points: int | None = 120,
        refit_every: int = 1,
        mc_samples: int = 128,
        xi: float = 0.01,
        ucb_beta: float = 0.2,
        device: str = "cpu",
        dtype: str = "float64",
        disable_progbar: bool = True,
        jit_compile: bool = False,
        standardize_y: bool = True,
        line_adapter_label: str = "saasbo_acq_line",
        model_cls: Any | None = None,
        fit_func: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if internal_space != "ilr":
            raise ValueError("SAASBOOptimizer currently supports internal_space='ilr' only")
        if acquisition not in {"log_ei", "ei", "ucb", "posterior_mean"}:
            raise ValueError("SAASBO acquisition must be one of 'log_ei', 'ei', 'ucb', or 'posterior_mean'")
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")
        if warmup_steps <= 0 or num_samples <= 0 or thinning <= 0:
            raise ValueError("warmup_steps, num_samples, and thinning must be positive")
        if refit_every <= 0:
            raise ValueError("refit_every must be positive")

        self.internal_space = internal_space
        self.acquisition = acquisition
        self.candidate_pool_size = int(candidate_pool_size)
        self.ilr_bounds_cfg = dict(ilr_bounds or {})
        self.warmup_steps = int(warmup_steps)
        self.num_samples = int(num_samples)
        self.thinning = int(thinning)
        self.max_tree_depth = int(max_tree_depth)
        self.train_yvar = float(train_yvar)
        self.max_train_points = max_train_points
        self.refit_every = int(refit_every)
        self.mc_samples = int(mc_samples)
        self.xi = float(xi)
        self.ucb_beta = float(ucb_beta)
        self.device_name = device
        self.dtype_name = dtype
        self.disable_progbar = bool(disable_progbar)
        self.jit_compile = bool(jit_compile)
        self.standardize_y = bool(standardize_y)
        self.line_adapter_label = line_adapter_label
        self.extra_kwargs = kwargs
        self._model_cls = model_cls
        self._fit_func = fit_func

        self.n_components: int | None = None
        self.seed: int | None = None
        self.maximize = True
        self.ilr_lower: np.ndarray | None = None
        self.ilr_upper: np.ndarray | None = None
        self.normalized_ilr_bounds_source = "uninitialized"
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self._torch = None
        self._model = None
        self._model_observe_version = -1
        self._observe_version = 0
        self._suggest_calls = 0
        self._score_calls = 0
        self._fit_calls = 0
        self._last_candidate_pool_size = 0
        self._last_train_points = 0
        self._fit_time_s_last = 0.0
        self._fit_time_s_total = 0.0
        self._acq_time_s_total = 0.0
        self._last_acq_time_s = 0.0
        self._last_fit_warnings: list[str] = []
        self._median_lengthscale_values: list[float] | None = None

    def initialize(self, X: np.ndarray, y: np.ndarray, objective_info: ObjectiveInfo, seed: int) -> None:
        X_arr = np.asarray(X, dtype=float)
        validate_simplex(X_arr)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        self._ensure_dependencies()
        self.n_components = int(objective_info.n_components)
        self.seed = int(seed)
        self.maximize = bool(objective_info.maximize)
        self.X = X_arr.copy()
        self.y = y_arr.copy()
        self._suggest_calls = 0
        self._score_calls = 0
        self._fit_calls = 0
        self._observe_version = 0
        self._model = None
        self._model_observe_version = -1
        self._last_candidate_pool_size = 0
        self._last_train_points = 0
        self._fit_time_s_last = 0.0
        self._fit_time_s_total = 0.0
        self._acq_time_s_total = 0.0
        self._last_acq_time_s = 0.0
        self._last_fit_warnings = []
        self._median_lengthscale_values = None
        self.ilr_lower, self.ilr_upper = self._derive_ilr_bounds(self.X)

    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        if self.X is None or self.y is None or self.n_components is None or self.seed is None:
            raise RuntimeError("SAASBOOptimizer must be initialized before suggest()")
        if n_suggestions <= 0:
            raise ValueError("n_suggestions must be positive")

        torch = self._import_torch_stack()
        model = self._get_model(torch)
        pool = sample_simplex(
            self.candidate_pool_size,
            self.n_components,
            seed=self.seed + 6_000_031 + self._suggest_calls,
        )
        values = self._score_candidates_with_model(torch, model, pool)
        order = np.argsort(values)[::-1]
        self._suggest_calls += 1
        self._last_candidate_pool_size = int(pool.shape[0])
        return pool[order[:n_suggestions]]

    def score_candidates(self, X_candidates: np.ndarray) -> np.ndarray:
        if self.X is None or self.y is None:
            raise RuntimeError("SAASBOOptimizer must be initialized before score_candidates()")
        torch = self._import_torch_stack()
        model = self._get_model(torch)
        values = self._score_candidates_with_model(torch, model, X_candidates)
        self._score_calls += 1
        return values

    def observe(self, obs: BatchObservation) -> None:
        if self.X is None or self.y is None:
            raise RuntimeError("SAASBOOptimizer must be initialized before observe()")
        validate_simplex(obs.X_actual)
        self.X = np.vstack([self.X, np.asarray(obs.X_actual, dtype=float)])
        self.y = np.concatenate([self.y, np.asarray(obs.y, dtype=float).reshape(-1)])
        self._observe_version += 1

    def line_metadata(self) -> dict[str, Any]:
        return {
            "line_adapter": self.line_adapter_label,
            "line_adapter_caveat": _SAASBO_LINE_CAVEAT,
            "saasbo_line_score_coordinate_system": "normalized_ilr_acquisition_on_raw_simplex_candidates",
        }

    def get_state(self) -> dict[str, Any]:
        lengthscales = self._median_lengthscale_values
        return {
            "name": self.name,
            "implemented": True,
            "backend": "botorch_saas_fully_bayesian",
            "dependency_available": self._dependencies_available_or_injected(),
            "dependency_status": get_saasbo_dependency_status() if self._model_cls is None else {"available": True, "backend": "injected_test_backend"},
            "internal_space": self.internal_space,
            "objective_space": "raw_simplex",
            "acquisition": self.acquisition,
            "candidate_pool_size": self.candidate_pool_size,
            "warmup_steps": self.warmup_steps,
            "num_samples": self.num_samples,
            "thinning": self.thinning,
            "max_tree_depth": self.max_tree_depth,
            "train_yvar": self.train_yvar,
            "max_train_points": self.max_train_points,
            "refit_every": self.refit_every,
            "mc_samples": self.mc_samples,
            "device": self.device_name,
            "dtype": self.dtype_name,
            "n_components": self.n_components,
            "n_observations": 0 if self.y is None else int(self.y.shape[0]),
            "n_train_points_last": int(self._last_train_points),
            "suggest_calls": self._suggest_calls,
            "score_calls": self._score_calls,
            "fit_calls": self._fit_calls,
            "last_candidate_pool_size": int(self._last_candidate_pool_size),
            "normalized_ilr_bounds_source": self.normalized_ilr_bounds_source,
            "ilr_lower": None if self.ilr_lower is None else self.ilr_lower.tolist(),
            "ilr_upper": None if self.ilr_upper is None else self.ilr_upper.tolist(),
            "fit_time_s_last": float(self._fit_time_s_last),
            "fit_time_s_total": float(self._fit_time_s_total),
            "acq_time_s_last": float(self._last_acq_time_s),
            "acq_time_s_total": float(self._acq_time_s_total),
            "median_lengthscale_min": None if not lengthscales else float(np.min(lengthscales)),
            "median_lengthscale_max": None if not lengthscales else float(np.max(lengthscales)),
            "median_lengthscale_values": lengthscales,
            "last_fit_warnings": list(self._last_fit_warnings),
            "line_adapter": self.line_adapter_label,
            "line_adapter_caveat": _SAASBO_LINE_CAVEAT,
        }

    def _get_model(self, torch):
        if (
            self._model is not None
            and self._model_observe_version >= 0
            and (self._observe_version - self._model_observe_version) < self.refit_every
        ):
            return self._model
        self._model = self._fit_model(torch)
        self._model_observe_version = self._observe_version
        return self._model

    def _training_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.X is not None and self.y is not None
        X = self.X
        y = self._model_y()
        if self.max_train_points is not None and X.shape[0] > self.max_train_points:
            X = X[-self.max_train_points :]
            y = y[-self.max_train_points :]
        self._last_train_points = int(X.shape[0])
        return X, y

    def _model_y(self) -> np.ndarray:
        assert self.y is not None
        return self.y if self.maximize else -self.y

    def _fit_model(self, torch):
        model_cls, fit_func = self._load_backend()
        X_np, y_np = self._training_arrays()
        train_X_np = self._composition_to_normalized_ilr(X_np, clip=True)
        train_X = torch.as_tensor(train_X_np, device=self._device(torch), dtype=self._dtype(torch))
        train_Y = torch.as_tensor(y_np.reshape(-1, 1), device=self._device(torch), dtype=self._dtype(torch))
        train_Yvar = torch.full_like(train_Y, self.train_yvar)

        model_kwargs: dict[str, Any] = {"train_Yvar": train_Yvar}
        if self.standardize_y and self._model_cls is None:
            from botorch.models.transforms.outcome import Standardize

            model_kwargs["outcome_transform"] = Standardize(m=1)

        start = time.time()
        fit_warnings: list[str] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = model_cls(train_X, train_Y, **model_kwargs)
            fit_func(
                model,
                warmup_steps=self.warmup_steps,
                num_samples=self.num_samples,
                thinning=self.thinning,
                max_tree_depth=self.max_tree_depth,
                disable_progbar=self.disable_progbar,
                jit_compile=self.jit_compile,
                seed=int((self.seed or 0) + 11_000_039 + self._fit_calls),
            )
            fit_warnings = [str(item.message) for item in caught]
        self._fit_time_s_last = float(time.time() - start)
        self._fit_time_s_total += self._fit_time_s_last
        self._fit_calls += 1
        self._last_fit_warnings = fit_warnings[-10:]
        self._median_lengthscale_values = self._extract_median_lengthscales(model)
        return model

    def _score_candidates_with_model(self, torch, model, X_candidates: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X_candidates, dtype=float)
        validate_simplex(X_arr)
        norm = self._composition_to_normalized_ilr(X_arr, clip=True)
        start = time.time()
        if self.acquisition == "posterior_mean":
            X_eval = torch.as_tensor(norm, device=self._device(torch), dtype=self._dtype(torch))
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                values = model.posterior(X_eval).mean.detach().reshape(-1).cpu().numpy()
        else:
            Xq = torch.as_tensor(norm, device=self._device(torch), dtype=self._dtype(torch)).unsqueeze(1)
            acq = self._build_acquisition(torch, model)
            with torch.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                values = acq(Xq).detach().reshape(-1).cpu().numpy()
        elapsed = time.time() - start
        self._last_acq_time_s = float(elapsed)
        self._acq_time_s_total += float(elapsed)
        return np.asarray(values, dtype=float)

    def _build_acquisition(self, torch, model):
        assert self.y is not None
        from botorch.sampling.normal import IIDNormalSampler

        sampler = IIDNormalSampler(sample_shape=torch.Size([self.mc_samples]), seed=int((self.seed or 0) + 13_000_057))
        if self.acquisition == "ucb":
            from botorch.acquisition.monte_carlo import qUpperConfidenceBound

            return qUpperConfidenceBound(model, beta=self.ucb_beta, sampler=sampler)

        best_f = float(np.max(self._model_y())) + self.xi
        if self.acquisition == "log_ei":
            try:
                from botorch.acquisition.logei import qLogExpectedImprovement

                return qLogExpectedImprovement(model, best_f=best_f, sampler=sampler)
            except ImportError:
                pass
        from botorch.acquisition.monte_carlo import qExpectedImprovement

        return qExpectedImprovement(model, best_f=best_f, sampler=sampler)

    def _derive_ilr_bounds(self, observed_X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.n_components is not None and self.seed is not None
        cfg = self.ilr_bounds_cfg
        if "lower" in cfg and "upper" in cfg:
            lower = np.asarray(cfg["lower"], dtype=float)
            upper = np.asarray(cfg["upper"], dtype=float)
            self.normalized_ilr_bounds_source = "config_lower_upper"
        else:
            n_candidates = int(cfg.get("n_candidates", 4096))
            lower_q = float(cfg.get("lower_quantile", 0.005))
            upper_q = float(cfg.get("upper_quantile", 0.995))
            pool = sample_simplex(n_candidates, self.n_components, seed=self.seed + 8_750_043)
            pool_ilr = composition_to_ilr_np(pool)
            lower = np.quantile(pool_ilr, lower_q, axis=0)
            upper = np.quantile(pool_ilr, upper_q, axis=0)
            self.normalized_ilr_bounds_source = (
                f"candidate_pool_quantile:n={n_candidates}:q={lower_q:g},{upper_q:g}"
            )

        observed_ilr = composition_to_ilr_np(observed_X)
        observed_min = np.min(observed_ilr, axis=0)
        observed_max = np.max(observed_ilr, axis=0)
        span = np.maximum(upper - lower, 1.0)
        lower = np.minimum(lower, observed_min - 0.05 * span)
        upper = np.maximum(upper, observed_max + 0.05 * span)

        if lower.shape != (self.n_components - 1,) or upper.shape != (self.n_components - 1,):
            raise ValueError("SAASBO ILR bounds must have shape (n_components - 1,)")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or not np.all(lower < upper):
            raise ValueError("SAASBO ILR bounds must be finite and strictly ordered")
        return lower, upper

    def _composition_to_normalized_ilr(self, X: np.ndarray, clip: bool) -> np.ndarray:
        if self.ilr_lower is None or self.ilr_upper is None:
            raise RuntimeError("SAASBO ILR bounds are not initialized")
        z = composition_to_ilr_np(X)
        if z.ndim == 1:
            z = z.reshape(1, -1)
        span = self.ilr_upper - self.ilr_lower
        normalized = (z - self.ilr_lower[None, :]) / span[None, :]
        if clip:
            normalized = np.clip(normalized, 0.0, 1.0)
        return normalized

    def _load_backend(self):
        if self._model_cls is not None and self._fit_func is not None:
            return self._model_cls, self._fit_func
        self._ensure_dependencies()
        from botorch.fit import fit_fully_bayesian_model_nuts
        from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

        return SaasFullyBayesianSingleTaskGP, fit_fully_bayesian_model_nuts

    def _ensure_dependencies(self) -> None:
        if self._model_cls is not None and self._fit_func is not None:
            return
        status = get_saasbo_dependency_status()
        if not status["available"]:
            raise ImportError(
                "SAASBOOptimizer requires BoTorch's fully Bayesian optional dependencies "
                "(JAX, jaxlib, and NumPyro in this BoTorch version). "
                f"{status['reason']}. Install with: {status['install_hint']}"
            )

    def _dependencies_available_or_injected(self) -> bool:
        return bool((self._model_cls is not None and self._fit_func is not None) or is_saasbo_available())

    def _import_torch_stack(self):
        if self._torch is not None:
            return self._torch
        try:
            import torch
            import botorch  # noqa: F401
            import gpytorch  # noqa: F401
        except ImportError as exc:
            raise ImportError("SAASBOOptimizer requires torch, botorch, and gpytorch") from exc
        self._torch = torch
        return torch

    def _device(self, torch):
        return torch.device(self.device_name)

    def _dtype(self, torch):
        try:
            return getattr(torch, self.dtype_name)
        except AttributeError as exc:
            raise ValueError(f"Unknown torch dtype {self.dtype_name!r}") from exc

    def _extract_median_lengthscales(self, model) -> list[float] | None:
        value = None
        if hasattr(model, "median_lengthscale"):
            try:
                value = model.median_lengthscale
            except Exception:
                value = None
        if value is None:
            for attr_path in [("covar_module", "base_kernel", "lengthscale"), ("covar_module", "lengthscale")]:
                current = model
                try:
                    for attr in attr_path:
                        current = getattr(current, attr)
                    value = current
                    break
                except Exception:
                    continue
        if value is None:
            return None
        try:
            if hasattr(value, "detach"):
                arr = value.detach().cpu().numpy()
            else:
                arr = np.asarray(value, dtype=float)
            arr = np.asarray(arr, dtype=float).reshape(-1)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                return None
            return [float(x) for x in arr]
        except Exception:
            return None


def _module_version(module_name: str) -> str | None:
    try:
        dist_name = {"pyro": "pyro-ppl"}.get(module_name, module_name)
        return importlib.metadata.version(dist_name)
    except Exception:
        try:
            module = importlib.import_module(module_name)
            return str(getattr(module, "__version__", "unknown"))
        except Exception:
            return None


_SAASBO_LINE_CAVEAT = (
    "SAASBO is point-native here; line mode uses the benchmark line wrapper to score deterministic "
    "simplex candidate lines by mean SAASBO acquisition on normalized ILR coordinates, then batch-observes the full line."
)
