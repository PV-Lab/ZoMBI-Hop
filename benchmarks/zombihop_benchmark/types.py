from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class BatchObservation:
    X_expected: np.ndarray
    X_actual: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectiveInfo:
    name: str
    n_components: int
    maximize: bool = True
    true_needles: np.ndarray | None = None
    y_star: float | None = None
    match_radius_ilr: float | None = None


class BenchmarkObjective(Protocol):
    info: ObjectiveInfo

    def initial_design(self, n: int, seed: int) -> np.ndarray:
        ...

    def evaluate_points(self, X_expected: np.ndarray, seed: int | None = None) -> BatchObservation:
        ...

    def evaluate_line(
        self,
        endpoints: np.ndarray,
        n_points: int,
        seed: int | None = None,
    ) -> BatchObservation:
        ...


class BenchmarkOptimizer(Protocol):
    name: str
    supports_point: bool
    supports_line: bool

    def initialize(self, X: np.ndarray, y: np.ndarray, objective_info: ObjectiveInfo, seed: int) -> None:
        ...

    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        ...

    def observe(self, obs: BatchObservation) -> None:
        ...

    def get_state(self) -> dict[str, Any]:
        ...

