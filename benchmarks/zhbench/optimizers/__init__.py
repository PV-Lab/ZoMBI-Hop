"""Optimizer registry.

Imports are lazy: pulling in torch/BoTorch costs seconds, and a random-search run
should not pay for it.
"""

from __future__ import annotations

from .base import BaseOptimizer, Optimizer

#: name -> default kwargs
_SPECS: dict[str, dict] = {
    "random": {},
    "gp_qucb": {"kind": "ucb"},
    "gp_qlogei": {"kind": "logei"},
    "zombihop": {},
}


def available() -> list[str]:
    return sorted(_SPECS)


def _resolve(name: str):
    if name == "random":
        from .random_simplex import RandomSearch
        return RandomSearch
    if name in ("gp_qucb", "gp_qlogei"):
        from .gp import GPBatch
        return GPBatch
    if name == "zombihop":
        from ..zombihop_runner import ZoMBIHopRunner
        return ZoMBIHopRunner
    raise ValueError(f"unknown optimizer {name!r}; available: {available()}")


def build(spec: dict):
    """Build an optimizer from ``{"name": ..., **kwargs}``."""
    spec = dict(spec)
    name = spec.pop("name")
    if name not in _SPECS:
        raise ValueError(f"unknown optimizer {name!r}; available: {available()}")
    cls = _resolve(name)
    obj = cls(**{**_SPECS[name], **spec})
    obj.name = name
    return obj


__all__ = ["BaseOptimizer", "Optimizer", "available", "build"]
