from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..line_audit import audit_line_endpoints
from ..line_mode import SimplexLineCandidate
from ..spaces import (
    composition_to_ilr_np,
    ilr_to_composition_np,
    sample_simplex,
    validate_simplex,
)
from ..types import BatchObservation, ObjectiveInfo


class HEBOOptimizer:
    name = "hebo"
    supports_point = True
    supports_line = True

    def __init__(
        self,
        internal_space: str = "ilr",
        ilr_bounds: dict[str, Any] | None = None,
        candidate_repair: dict[str, Any] | None = None,
        objective_sign: dict[str, Any] | None = None,
        points_per_line: int = 24,
        n_line_candidates: int = 64,
        line_adapter: str = "anchor_chord",
        line_max_radius_l2: float | None = None,
        hebo_kwargs: dict[str, Any] | None = None,
        hebo_cls=None,
        design_space_cls=None,
        **kwargs: Any,
    ) -> None:
        if internal_space != "ilr":
            raise ValueError("HEBOOptimizer currently supports internal_space='ilr' only")
        if line_adapter not in {"anchor_chord", "disabled"}:
            raise ValueError("HEBO line_adapter must be 'anchor_chord' or 'disabled'")
        self.internal_space = internal_space
        self.ilr_bounds_cfg = ilr_bounds or {}
        self.candidate_repair = candidate_repair or {}
        self.objective_sign = objective_sign or {"hebo_internal": "minimize", "benchmark_external": "maximize"}
        self.points_per_line = int(points_per_line)
        self.n_line_candidates = int(n_line_candidates)
        self.line_adapter = line_adapter
        self.line_max_radius_l2 = None if line_max_radius_l2 is None else float(line_max_radius_l2)
        self.hebo_kwargs = dict(hebo_kwargs or {})
        self.extra_kwargs = kwargs
        self._hebo_cls = hebo_cls
        self._design_space_cls = design_space_cls

        self.n_components: int | None = None
        self.seed: int | None = None
        self.maximize: bool = True
        self.var_names: list[str] = []
        self.ilr_lower: np.ndarray | None = None
        self.ilr_upper: np.ndarray | None = None
        self._hebo = None
        self._design_space = None
        self._suggest_calls = 0
        self._observe_calls = 0
        self._line_calls = 0
        self._repair_clip_count = 0
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None

    def initialize(self, X: np.ndarray, y: np.ndarray, objective_info: ObjectiveInfo, seed: int) -> None:
        X_arr = np.asarray(X, dtype=float)
        validate_simplex(X_arr)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        self.n_components = int(objective_info.n_components)
        self.seed = int(seed)
        self.maximize = bool(objective_info.maximize)
        self.var_names = [f"z{i}" for i in range(self.n_components - 1)]
        self.ilr_lower, self.ilr_upper = self._derive_ilr_bounds()
        self._suggest_calls = 0
        self._observe_calls = 0
        self._line_calls = 0
        self._repair_clip_count = 0
        self.X = X_arr.copy()
        self.y = y_arr.copy()

        HEBO, DesignSpace = self._load_hebo_classes()
        specs = [
            {"name": name, "type": "num", "lb": float(lb), "ub": float(ub)}
            for name, lb, ub in zip(self.var_names, self.ilr_lower, self.ilr_upper)
        ]
        self._design_space = DesignSpace().parse(specs)
        self._hebo = HEBO(self._design_space, **self.hebo_kwargs)
        self._observe_hebo(self.X, self.y)

    def configure_line_mode(
        self,
        points_per_line: int | None = None,
        n_line_candidates: int | None = None,
        line_score: str | None = None,
        **_: Any,
    ) -> None:
        if points_per_line is not None:
            self.points_per_line = int(points_per_line)
        if n_line_candidates is not None:
            self.n_line_candidates = int(n_line_candidates)

    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        if self._hebo is None or self.n_components is None:
            raise RuntimeError("HEBOOptimizer must be initialized before suggest()")
        if n_suggestions <= 0:
            raise ValueError("n_suggestions must be positive")

        rec = self._hebo.suggest(n_suggestions=n_suggestions)
        z = self._records_to_ilr(rec)
        z = self._repair_ilr(z)
        X = ilr_to_composition_np(z, self.n_components, eps=float(self.candidate_repair.get("clip_min", 1e-12)))
        validate_simplex(X, atol=1e-6)
        self._suggest_calls += 1
        return X

    def suggest_line(self) -> SimplexLineCandidate:
        if self.line_adapter == "disabled":
            raise RuntimeError(
                "HEBO line mode is disabled. Use point mode, or set params.line_adapter='anchor_chord' "
                "to use the documented point-native HEBO anchor-to-line adapter."
            )
        if self.n_components is None or self.seed is None:
            raise RuntimeError("HEBOOptimizer must be initialized before suggest_line()")

        anchor = self.suggest(1)[0]
        candidates = self._anchor_chord_candidates(anchor)
        if not candidates:
            raise RuntimeError("HEBO anchor-to-line adapter could not generate a valid simplex chord")
        lengths = np.asarray([candidate.length_l2 for candidate in candidates], dtype=float)
        chosen_idx = int(np.argmax(lengths))
        line_id = f"hebo_seed{self.seed}_line{self._line_calls + 1}_anchor_chord{chosen_idx}"
        chosen = candidates[chosen_idx].with_score(
            score=float(lengths[chosen_idx]),
            score_method="hebo_anchor_chord_longest_l2",
            line_id=line_id,
        )
        self._line_calls += 1
        return chosen

    def observe(self, obs: BatchObservation) -> None:
        if self.X is None or self.y is None:
            raise RuntimeError("HEBOOptimizer must be initialized before observe()")
        validate_simplex(obs.X_actual)
        X_new = np.asarray(obs.X_actual, dtype=float)
        y_new = np.asarray(obs.y, dtype=float).reshape(-1)
        self.X = np.vstack([self.X, X_new])
        self.y = np.concatenate([self.y, y_new])
        self._observe_hebo(X_new, y_new)

    def score_candidates(self, X_candidates: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "The installed HEBO API is treated as point-native in this benchmark. "
            "Line mode uses the documented HEBO anchor-to-line adapter instead of acquisition scoring."
        )

    def get_state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "implemented": True,
            "dependency": "hebo",
            "dependency_available": self._hebo is not None,
            "internal_space": self.internal_space,
            "objective_sign": self.objective_sign,
            "n_components": self.n_components,
            "n_observations": 0 if self.y is None else int(self.y.shape[0]),
            "suggest_calls": self._suggest_calls,
            "observe_calls": self._observe_calls,
            "line_calls": self._line_calls,
            "points_per_line": self.points_per_line,
            "n_line_candidates": self.n_line_candidates,
            "line_adapter": self.line_adapter,
            "line_adapter_caveat": _HEBO_LINE_CAVEAT if self.line_adapter == "anchor_chord" else "",
            "repair_clip_count": self._repair_clip_count,
            "ilr_lower": None if self.ilr_lower is None else self.ilr_lower.tolist(),
            "ilr_upper": None if self.ilr_upper is None else self.ilr_upper.tolist(),
        }

    def _load_hebo_classes(self):
        if self._hebo_cls is not None and self._design_space_cls is not None:
            return self._hebo_cls, self._design_space_cls
        try:
            from hebo.design_space.design_space import DesignSpace
            from hebo.optimizers.hebo import HEBO
        except ImportError as exc:
            raise ImportError(
                "HEBOOptimizer requires the optional HEBO package. Install it in this environment, "
                "for example: .\\.venv\\Scripts\\python -m pip install HEBO"
            ) from exc
        return HEBO, DesignSpace

    def _derive_ilr_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.n_components is not None and self.seed is not None
        cfg = self.ilr_bounds_cfg
        if "lower" in cfg and "upper" in cfg:
            lower = np.asarray(cfg["lower"], dtype=float)
            upper = np.asarray(cfg["upper"], dtype=float)
        else:
            n_candidates = int(cfg.get("n_candidates", 4096))
            lower_q = float(cfg.get("lower_quantile", 0.005))
            upper_q = float(cfg.get("upper_quantile", 0.995))
            pool = sample_simplex(n_candidates, self.n_components, seed=self.seed + 8_000_021)
            pool_ilr = composition_to_ilr_np(pool)
            lower = np.quantile(pool_ilr, lower_q, axis=0)
            upper = np.quantile(pool_ilr, upper_q, axis=0)
        if lower.shape != (self.n_components - 1,) or upper.shape != (self.n_components - 1,):
            raise ValueError("HEBO ILR bounds must have shape (n_components - 1,)")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)) or not np.all(lower < upper):
            raise ValueError("HEBO ILR bounds must be finite and strictly ordered")
        return lower, upper

    def _records_to_ilr(self, records) -> np.ndarray:
        if isinstance(records, pd.DataFrame):
            missing = [name for name in self.var_names if name not in records.columns]
            if missing:
                raise ValueError(f"HEBO suggest() result missing ILR columns: {missing}")
            return records[self.var_names].to_numpy(dtype=float)
        arr = np.asarray(records, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _ilr_to_records(self, z: np.ndarray) -> pd.DataFrame:
        z_arr = np.asarray(z, dtype=float)
        if z_arr.ndim == 1:
            z_arr = z_arr.reshape(1, -1)
        return pd.DataFrame(z_arr, columns=self.var_names)

    def _observe_hebo(self, X: np.ndarray, y: np.ndarray) -> None:
        if self._hebo is None:
            return
        z = composition_to_ilr_np(X)
        z = self._repair_ilr(z)
        y_internal = self._to_hebo_y(y)
        self._hebo.observe(self._ilr_to_records(z), y_internal.reshape(-1, 1))
        self._observe_calls += 1

    def _to_hebo_y(self, y: np.ndarray) -> np.ndarray:
        arr = np.asarray(y, dtype=float).reshape(-1)
        hebo_internal = str(self.objective_sign.get("hebo_internal", "minimize"))
        if hebo_internal == "minimize":
            return -arr if self.maximize else arr
        if hebo_internal == "maximize":
            return arr if self.maximize else -arr
        raise ValueError("objective_sign.hebo_internal must be 'minimize' or 'maximize'")

    def _repair_ilr(self, z: np.ndarray) -> np.ndarray:
        z_arr = np.asarray(z, dtype=float)
        if z_arr.ndim == 1:
            z_arr = z_arr.reshape(1, -1)
        if self.ilr_lower is None or self.ilr_upper is None:
            return z_arr
        clipped = np.clip(z_arr, self.ilr_lower, self.ilr_upper)
        self._repair_clip_count += int(np.sum(~np.isclose(clipped, z_arr)))
        return clipped

    def _anchor_chord_candidates(self, anchor: np.ndarray) -> list[SimplexLineCandidate]:
        assert self.seed is not None and self.n_components is not None
        rng = np.random.default_rng(self.seed + 9_000_031 + self._line_calls)
        anchor = np.asarray(anchor, dtype=float).reshape(-1)
        validate_simplex(anchor)
        candidates: list[SimplexLineCandidate] = []
        ts = np.linspace(0.0, 1.0, self.points_per_line)
        attempts = max(self.n_line_candidates * 4, self.n_line_candidates)
        for idx in range(attempts):
            if len(candidates) >= self.n_line_candidates:
                break
            direction = rng.normal(size=self.n_components)
            direction -= direction.mean()
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                continue
            direction /= norm
            lower, upper = _simplex_chord_bounds(anchor, direction)
            if self.line_max_radius_l2 is not None:
                lower = max(lower, -self.line_max_radius_l2)
                upper = min(upper, self.line_max_radius_l2)
            if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
                continue
            endpoints = np.vstack([anchor + lower * direction, anchor + upper * direction])
            endpoints = np.clip(endpoints, 0.0, None)
            endpoints /= endpoints.sum(axis=1, keepdims=True)
            points = (1.0 - ts[:, None]) * endpoints[0][None, :] + ts[:, None] * endpoints[1][None, :]
            validate_simplex(points)
            endpoint_ilr = composition_to_ilr_np(endpoints)
            audit = audit_line_endpoints(endpoints)
            candidates.append(
                SimplexLineCandidate(
                    candidate_index=len(candidates),
                    endpoints=endpoints,
                    points=points,
                    length_l2=float(np.linalg.norm(endpoints[1] - endpoints[0])),
                    length_ilr=float(np.linalg.norm(endpoint_ilr[1] - endpoint_ilr[0])),
                    extra_metadata={
                        "line_adapter": "hebo_anchor_chord",
                        "line_adapter_caveat": _HEBO_LINE_CAVEAT,
                        "candidate_anchor": anchor.tolist(),
                        "n_ranked_candidate_lines": self.n_line_candidates,
                        **audit,
                    },
                )
            )
        return candidates


def _simplex_chord_bounds(anchor: np.ndarray, direction: np.ndarray) -> tuple[float, float]:
    lower = -math.inf
    upper = math.inf
    for x_i, v_i in zip(anchor, direction):
        if abs(v_i) <= 1e-15:
            continue
        bound = -float(x_i) / float(v_i)
        if v_i > 0:
            lower = max(lower, bound)
        else:
            upper = min(upper, bound)
    return lower, upper


_HEBO_LINE_CAVEAT = (
    "HEBO is point-native here; line mode converts each HEBO anchor suggestion into the longest "
    "deterministic simplex chord among sampled zero-sum directions, then batch-observes the full line."
)
