from __future__ import annotations

from typing import Any

from ..types import BenchmarkOptimizer
from .external import ExternalOptimizer
from .gp_botorch import GPBoTorchOptimizer
from .hebo_optimizer import HEBOOptimizer
from .random_simplex import RandomSimplexOptimizer
from .rf_bo import RFBOOptimizer
from .turbo_optimizer import TuRBOOptimizer
from .zombihop_adapter import ZoMBIHopAdapter


def build_optimizer(config: dict[str, Any]) -> BenchmarkOptimizer:
    kind = config.get("kind")
    params = dict(config.get("params") or {})
    if kind == "random_simplex":
        return RandomSimplexOptimizer(**params)
    if kind == "zombihop":
        return ZoMBIHopAdapter(**params)
    if kind in {"gp_botorch", "gp_ard_ei", "gp_ard_ucb"}:
        if "kind" not in params:
            params["kind"] = "ei" if kind.endswith("ei") else "ucb"
        return GPBoTorchOptimizer(**params)
    if kind == "rf_bo":
        return RFBOOptimizer(**params)
    if kind == "hebo":
        return HEBOOptimizer(**params)
    if kind in {"turbo", "turbo_1"}:
        return TuRBOOptimizer(**params)
    if kind in {"saasbo"}:
        return ExternalOptimizer(kind=kind, **params)
    raise ValueError(f"Unknown optimizer kind: {kind!r}")

