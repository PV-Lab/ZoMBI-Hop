from __future__ import annotations

import numpy as np

from ..spaces import sample_simplex
from ..types import BatchObservation, ObjectiveInfo


class RandomSimplexOptimizer:
    name = "random_simplex"
    supports_point = True
    supports_line = False

    def __init__(self) -> None:
        self.n_components: int | None = None
        self.seed: int | None = None
        self._calls = 0
        self._observations: list[BatchObservation] = []

    def initialize(self, X: np.ndarray, y: np.ndarray, objective_info: ObjectiveInfo, seed: int) -> None:
        self.n_components = objective_info.n_components
        self.seed = seed
        self._calls = 0
        self._observations = []

    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        if self.n_components is None or self.seed is None:
            raise RuntimeError("RandomSimplexOptimizer must be initialized before suggest()")
        X = sample_simplex(n_suggestions, self.n_components, self.seed + 1_000_003 + self._calls)
        self._calls += 1
        return X

    def observe(self, obs: BatchObservation) -> None:
        self._observations.append(obs)

    def get_state(self) -> dict[str, object]:
        return {
            "name": self.name,
            "n_components": self.n_components,
            "seed": self.seed,
            "suggest_calls": self._calls,
            "n_observation_batches": len(self._observations),
        }
