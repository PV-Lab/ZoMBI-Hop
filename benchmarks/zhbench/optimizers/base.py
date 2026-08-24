"""The optimizer contract.

One decision = one batch of ``q`` compositions. Every method gets the same number
of decisions and the same q, so "same number of candidates per iteration" holds by
construction (Aleks: batches for the comparison methods versus lines for
ZoMBI-Hop, "we don't really want to be modifying the benchmark approaches").

Baselines are run as their authors intended: a real joint q-batch acquisition, not
greedy top-k of a single-point acquisition. Greedy top-k over a pool returns q
near-identical points and would quietly cripple the baselines, which is the
mirror-image of the mistake the previous benchmark made with ZoMBI-Hop.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Optimizer(Protocol):
    name: str

    def initialize(self, X_actual: np.ndarray, y: np.ndarray, objective, seed: int) -> None:
        """Seed the method with the shared initial design."""

    def suggest(self, q: int) -> np.ndarray:
        """Return ``(q, d)`` requested compositions on the domain."""

    def observe(self, X_actual: np.ndarray, y: np.ndarray) -> None:
        """Take the realized compositions and measured values for the whole batch."""

    def declared_optima(self) -> np.ndarray | None:
        """The method's own solution set, or None if it does not declare one."""


class BaseOptimizer:
    """Shared bookkeeping: history, seed, domain."""

    name = "base"
    #: Methods that run their own closed loop against the objective (ZoMBI-Hop)
    #: rather than being driven by the suggest/observe loop.
    self_driving = False

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.X: np.ndarray | None = None
        self.y: np.ndarray | None = None
        self.dim: int | None = None
        self.seed: int = 0
        self.domain: str = "simplex"
        self._n_suggest = 0

    def initialize(self, X_actual, y, objective, seed: int) -> None:
        self.X = np.atleast_2d(np.asarray(X_actual, dtype=float)).copy()
        self.y = np.asarray(y, dtype=float).ravel().copy()
        self.dim = int(objective.dim)
        self.domain = objective.domain
        self.seed = int(seed)
        self._n_suggest = 0
        self._rng = np.random.default_rng(seed + 7919)

    def observe(self, X_actual, y) -> None:
        self.X = np.vstack([self.X, np.atleast_2d(np.asarray(X_actual, dtype=float))])
        self.y = np.concatenate([self.y, np.asarray(y, dtype=float).ravel()])

    def declared_optima(self):
        return None

    def state(self) -> dict:
        return {"name": self.name,
                "n_observations": 0 if self.y is None else int(self.y.size),
                "n_suggest_calls": int(self._n_suggest)}

    # -- helpers ---------------------------------------------------------------
    def _sample_domain(self, n: int, rng=None) -> np.ndarray:
        rng = rng or self._rng
        if self.domain == "cube":
            return rng.random((n, self.dim))
        return rng.dirichlet(np.ones(self.dim), size=n)
