from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .line_audit import audit_line_endpoints
from .spaces import composition_to_ilr_np, validate_simplex
from .types import BatchObservation, ObjectiveInfo


@dataclass(frozen=True)
class SimplexLineCandidate:
    candidate_index: int
    endpoints: np.ndarray
    points: np.ndarray
    length_l2: float
    length_ilr: float
    score: float = math.nan
    score_method: str = "unscored"
    line_id: str = ""
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    def with_score(self, score: float, score_method: str, line_id: str) -> "SimplexLineCandidate":
        return replace(self, score=float(score), score_method=score_method, line_id=line_id)

    def metadata(self) -> dict[str, Any]:
        audit = audit_line_endpoints(self.endpoints)
        return {
            "line_id": self.line_id,
            "candidate_index": int(self.candidate_index),
            "endpoints": np.asarray(self.endpoints, dtype=float).tolist(),
            "line_num_points": self.n_points,
            "line_score": float(self.score),
            "line_score_method": self.score_method,
            "line_length_l2": float(self.length_l2),
            "line_length_ilr": float(self.length_ilr),
            "line_adapter": "benchmark_candidate_pool_line",
            "line_adapter_caveat": "",
            **audit,
            **self.extra_metadata,
        }


def generate_simplex_line_candidates(
    n_candidates: int,
    n_components: int,
    points_per_line: int,
    seed: int,
    include_endpoints: bool = True,
) -> list[SimplexLineCandidate]:
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")
    if n_components < 2:
        raise ValueError("n_components must be at least 2")
    if points_per_line <= 0:
        raise ValueError("points_per_line must be positive")

    rng = np.random.default_rng(seed)
    endpoint_pairs = rng.dirichlet(np.ones(n_components), size=(n_candidates, 2))
    if include_endpoints:
        ts = np.linspace(0.0, 1.0, points_per_line)
    else:
        ts = np.linspace(0.0, 1.0, points_per_line + 2)[1:-1]

    candidates: list[SimplexLineCandidate] = []
    for idx, endpoints in enumerate(endpoint_pairs):
        points = (1.0 - ts[:, None]) * endpoints[0][None, :] + ts[:, None] * endpoints[1][None, :]
        validate_simplex(points)
        validate_simplex(endpoints)
        endpoint_ilr = composition_to_ilr_np(endpoints)
        candidates.append(
            SimplexLineCandidate(
                candidate_index=idx,
                endpoints=endpoints.copy(),
                points=points.copy(),
                length_l2=float(np.linalg.norm(endpoints[1] - endpoints[0])),
                length_ilr=float(np.linalg.norm(endpoint_ilr[1] - endpoint_ilr[0])),
            )
        )
    return candidates


class LineModeOptimizerWrapper:
    supports_point = False
    supports_line = True

    def __init__(
        self,
        base_optimizer,
        points_per_line: int = 24,
        n_line_candidates: int = 256,
        line_score: str = "mean_acq",
        include_endpoints: bool = True,
    ) -> None:
        if line_score not in {"mean_acq", "max_acq", "random"}:
            raise ValueError("line_score must be one of 'mean_acq', 'max_acq', or 'random'")
        self.base_optimizer = base_optimizer
        self.name = base_optimizer.name
        self.points_per_line = int(points_per_line)
        self.n_line_candidates = int(n_line_candidates)
        self.line_score = line_score
        self.include_endpoints = bool(include_endpoints)
        self.n_components: int | None = None
        self.seed: int | None = None
        self._line_calls = 0
        self._last_line: SimplexLineCandidate | None = None

    def initialize(self, X: np.ndarray, y: np.ndarray, objective_info: ObjectiveInfo, seed: int) -> None:
        self.n_components = int(objective_info.n_components)
        self.seed = int(seed)
        self._line_calls = 0
        self._last_line = None
        self.base_optimizer.initialize(X, y, objective_info, seed)

    def suggest_line(self) -> SimplexLineCandidate:
        if self.n_components is None or self.seed is None:
            raise RuntimeError("LineModeOptimizerWrapper must be initialized before suggest_line()")
        call_seed = self.seed + 4_000_009 + self._line_calls
        candidates = generate_simplex_line_candidates(
            n_candidates=self.n_line_candidates,
            n_components=self.n_components,
            points_per_line=self.points_per_line,
            seed=call_seed,
            include_endpoints=self.include_endpoints,
        )
        scores, score_method = self._score_lines(candidates, call_seed)
        clean_scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(float).max, neginf=-np.inf)
        chosen_idx = int(np.argmax(clean_scores))
        line_id = f"{self.name}_seed{self.seed}_line{self._line_calls + 1}_cand{chosen_idx}"
        chosen = candidates[chosen_idx].with_score(scores[chosen_idx], score_method, line_id)
        extra_metadata = {
            "n_ranked_candidate_lines": self.n_line_candidates,
        }
        if hasattr(self.base_optimizer, "line_metadata"):
            extra_metadata.update(self.base_optimizer.line_metadata())
        chosen = replace(chosen, extra_metadata={**chosen.extra_metadata, **extra_metadata})
        self._line_calls += 1
        self._last_line = chosen
        return chosen

    def observe(self, obs: BatchObservation) -> None:
        self.base_optimizer.observe(obs)

    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        raise NotImplementedError("LineModeOptimizerWrapper exposes suggest_line() in line mode")

    def get_state(self) -> dict[str, Any]:
        state = {
            "name": self.name,
            "mode_wrapper": "line",
            "line_calls": self._line_calls,
            "points_per_line": self.points_per_line,
            "n_line_candidates": self.n_line_candidates,
            "line_score": self.line_score,
            "include_endpoints": self.include_endpoints,
            "base_optimizer_state": self.base_optimizer.get_state(),
        }
        if self._last_line is not None:
            state["last_line"] = self._last_line.metadata()
        return state

    def _score_lines(self, candidates: list[SimplexLineCandidate], seed: int) -> tuple[np.ndarray, str]:
        if self.line_score == "random" or not hasattr(self.base_optimizer, "score_candidates"):
            rng = np.random.default_rng(seed + 97)
            return rng.random(len(candidates)), "random"

        flat_points = np.vstack([candidate.points for candidate in candidates])
        point_scores = np.asarray(self.base_optimizer.score_candidates(flat_points), dtype=float)
        point_scores = point_scores.reshape(len(candidates), self.points_per_line)
        if self.line_score == "mean_acq":
            return point_scores.mean(axis=1), "mean_acq"
        return point_scores.max(axis=1), "max_acq"
