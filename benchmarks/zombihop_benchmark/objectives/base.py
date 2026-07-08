from __future__ import annotations

from typing import Any

from ..types import BenchmarkObjective
from .real_rf_surrogate import RealRFSurrogateObjective
from .synthetic import SyntheticSimplexObjective
from .tabular import TabularObjective


def build_objective(config: dict[str, Any]) -> BenchmarkObjective:
    kind = config.get("kind")
    params = dict(config.get("params") or {})
    name = config.get("name", kind)
    n_components = int(config.get("n_components", params.pop("n_components", 3)))
    maximize = bool(config.get("maximize", True))
    if kind == "synthetic_simplex":
        return SyntheticSimplexObjective(
            name=name,
            n_components=n_components,
            maximize=maximize,
            **params,
        )
    if kind == "tabular":
        return TabularObjective(
            name=name,
            n_components=n_components,
            maximize=maximize,
            **params,
        )
    if kind == "real_rf_surrogate":
        return RealRFSurrogateObjective(
            name=name,
            n_components=n_components,
            maximize=maximize,
            **params,
        )
    raise ValueError(f"Unknown objective kind: {kind!r}")

