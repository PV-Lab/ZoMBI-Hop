from __future__ import annotations

import numpy as np

from ..types import BatchObservation, ObjectiveInfo


class ExternalOptimizer:
    supports_point = True
    supports_line = False

    def __init__(self, kind: str, **kwargs) -> None:
        if kind not in {"hebo", "turbo", "saasbo"}:
            raise ValueError("ExternalOptimizer kind must be one of: hebo, turbo, saasbo")
        self.kind = kind
        self.name = kind
        self.kwargs = kwargs

    def initialize(self, X: np.ndarray, y: np.ndarray, objective_info: ObjectiveInfo, seed: int) -> None:
        raise NotImplementedError(f"{self.kind} support is planned for a later benchmark step.")

    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        raise NotImplementedError(f"{self.kind} support is not implemented in step 1.")

    def observe(self, obs: BatchObservation) -> None:
        raise NotImplementedError(f"{self.kind} support is not implemented in step 1.")

    def get_state(self) -> dict[str, object]:
        return {"name": self.name, "kind": self.kind, "implemented": False}

