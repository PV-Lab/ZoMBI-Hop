"""
benchmarks/methods/random_search.py
===================================
Random sampling: the floor every other method has to clear.

``sampler="uniform"`` (default) draws i.i.d. uniform points in the unit box — pure
random search. ``sampler="sobol"`` walks one scrambled Sobol' sequence instead,
which covers the box more evenly for the same budget; it is the stronger
space-filling baseline, and the gap between the two says how much of a method's
advantage is merely "not clumping".
"""

from __future__ import annotations

import numpy as np

from .base import AskTellMethod, Problem
from .registry import register


@register
class RandomSearch(AskTellMethod):
    name = "random"
    description = "i.i.d. uniform (or Sobol') samples in the unit box"
    defaults = {"sampler": "uniform"}

    def setup(self, problem: Problem) -> None:
        if self.config["sampler"] not in ("uniform", "sobol"):
            raise ValueError(f"random: sampler must be 'uniform' or 'sobol', "
                             f"got {self.config['sampler']!r}")
        self.dim = problem.dim
        self._sobol = None
        if self.config["sampler"] == "sobol":
            from scipy.stats import qmc

            self._sobol = qmc.Sobol(d=self.dim, scramble=True,
                                    seed=self.seed & 0xFFFFFFFF)

    def ask(self, n: int) -> np.ndarray:
        if self._sobol is not None:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return self._sobol.random(n)
        return self.rng.random((n, self.dim))

    def tell(self, X_requested, X_actual, Y) -> None:
        pass
