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
    "gp_ts": {},
    "zombihop": {},
    # n_consecutive_converged=5 sensitivity: the value src/default_hparams.py
    # still carries (verified unchanged on origin/brianna and origin/brianna-v2),
    # against whatever the tuned JSON for that dimension carries. That is NOT a
    # uniform 2: 3d.json has 1, 4d.json and 6d_ensemble.json have 2. So this arm is
    # "1 vs 5" at 3-D and "2 vs 5" at 4-D/6-D -- do not pool them into one claim.
    "zombihop_nc5": {"n_consecutive_converged": 5},
    # Tests OUR OWN published causal claim, in house, before someone else's branch
    # tests it for us. RESULTS.md attributes ZoMBI-Hop's low peak_ratio to the
    # declaration gate, and 496 of 563 logged needles were declared at exactly
    # zoom == 1 -- the min_zoom_for_needle floor -- with none at zoom 0. So we have
    # no observation at all of the regime where the gate is open. origin/brianna-v2
    # pins this to 0 and deletes the gate; if our claim is right, this arm should
    # declare materially MORE needles at some cost in precision. Everything else
    # stays per-dimension tuned, so the gate is the only thing that moves.
    "zombihop_mz0": {"min_zoom_for_needle": 0},
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
    if name == "gp_ts":
        from .gp import GPThompson
        return GPThompson
    if name.startswith("zombihop"):
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
