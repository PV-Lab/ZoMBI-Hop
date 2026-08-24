"""Uniform random search. The floor, and the yardstick.

Aleks: "given infinite time, random sampling will find all of the points in the
space. However, we should be able to identify local optima (needles) faster than
that using ZoMBI-Hop." So random is not just a weak baseline here -- it is the
denominator of :func:`metrics.lift_over_random`, the one quantity that stays
comparable as the dimension ladder makes every absolute score smaller.
"""

from __future__ import annotations

import numpy as np

from .base import BaseOptimizer


class RandomSearch(BaseOptimizer):
    name = "random"

    def suggest(self, q: int) -> np.ndarray:
        self._n_suggest += 1
        return self._sample_domain(int(q))
