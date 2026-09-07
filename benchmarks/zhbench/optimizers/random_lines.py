"""Uniform random *lines* -- the only floor the printer can actually execute.

``random`` scatters 24 unrelated compositions per decision. The printer cannot do
that: it deposits a chord between two compositions, and every point on that chord
shares the syringe ramp that produced it. So `random` is not a floor the lab could
run, and the ~10x `input_cost` gap between it and ZoMBI-Hop is partly just the
difference between a scattered batch and a printed line -- a difference of
instrument, not of algorithm.

This arm removes that confound. It draws two uniform Dirichlet endpoints and prints
the chord between them, through the *same* ``realize_line`` path ZoMBI-Hop uses:
deterministic ramp-lag / diffusion model plus the hardware residual, not the
point-wise perturbation the scattered baselines get. It is the honest question
"what does the printer get for free?", and the right denominator for any claim that
line-based search is efficient.

It is deliberately non-adaptive: ``observe`` is inherited and ignored. Any gap
between this and ZoMBI-Hop is attributable to ZoMBI-Hop's search, because the two
share the instrument exactly.

On a cube domain (dataset objectives) there is no printer, so the chord endpoints
are drawn uniformly in the cube and realization falls back to the point-wise path.
"""

from __future__ import annotations

import numpy as np

from ..protocol import realize_line
from .base import BaseOptimizer


class RandomLines(BaseOptimizer):
    name = "random_lines"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._left: np.ndarray | None = None
        self._right: np.ndarray | None = None

    def suggest(self, q: int) -> np.ndarray:
        self._n_suggest += 1
        q = int(q)
        if self.domain == "cube":
            a, b = self._rng.random((2, self.dim))
        else:
            a, b = self._rng.dirichlet(np.ones(self.dim), size=2)
        self._left, self._right = a, b
        t = np.linspace(0.0, 1.0, q)[:, None]
        return a[None, :] + t * (b - a)[None, :]

    def realize_request(self, X_req: np.ndarray, run) -> np.ndarray | None:
        """Print the chord, rather than perturbing each point independently.

        Returning ``None`` hands realization back to the protocol's point-wise
        path, which is what a cube domain wants -- there is no printer there.
        """
        if self.domain == "cube" or self._left is None:
            return None
        return realize_line(self._left, self._right, X_req.shape[0],
                            run.protocol, run.rng)
