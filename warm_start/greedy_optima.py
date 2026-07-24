"""
Greedy optima finder
====================

Locate a spread-out set of candidate optima on the probability simplex from a
warm-start budget of objective evaluations.

The procedure has three stages:

1. **Thin the space.**  Lay down the warm-start design on the ``dim``-simplex
   with :func:`warm_start.warm_start.greedy_lines` — the same sampler, and the
   same 24-points-per-line hardware constraint, as the warm start itself — so
   every region of the simplex has a sample near it and the design is one the
   instrument can actually measure.
2. **Keep the top 5%.**  Evaluate the objective on the design and retain only
   the highest-valued ``top_frac`` of the points.  These are the plausible
   basins; everything else is background.
3. **Greedy maximin over the survivors.**  The highest-valued survivor is the
   first optimum.  Each subsequent optimum is the survivor whose *nearest
   already-chosen optimum is farthest away* — :func:`maximin_subset`, the same
   routine that thins a uniform pool, restricted here to high-value points.
   This spreads the reported optima over distinct basins instead of returning
   ``n`` samples off one tall peak.

Everything dimension- or geometry-specific (budgets, line count, the sampler,
the maximin rule) lives in ``warm_start/warm_start.py`` and is imported here, so
changing the warm start changes this module with it.

Default optima counts (``n``) scale with simplex room:

    3d  ->  5      4d  -> 10      10d -> 20

Example
-------
    from warm_start.greedy_optima import find_optima
    from synthetic_data.ensemble import Ensemble

    fn = Ensemble(dim=3, n_optima=12, seed=0)
    X, y = find_optima(fn.predict, dim=3)      # (5, 3) compositions, (5,) values
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from warm_start.warm_start import (
    greedy_lines,
    maximin_subset,
    n_lines as default_n_lines,
)

# Number of optima to report, by simplex dimension.
N_OPTIMA = {
    3:   5,
    4:   10,
    10:  20,
}

TOP_FRACTION = 0.05  # keep the best 5% of the warm-start design as optima candidates


def n_optima(dim: int) -> int:
    """Default number of optima to report at simplex ``dim``.

    Uses the hardcoded :data:`N_OPTIMA` table for the benchmarked dimensions
    (3/4/10); other dimensions fall back to ``2 * dim``, the slope those anchors
    sit on.
    """
    return N_OPTIMA.get(int(dim), 2 * int(dim))


def find_optima(
    objective: Callable[[np.ndarray], np.ndarray],
    dim: int,
    n: int | None = None,
    *,
    seed: int | None = None,
    n_lines: int | None = None,
    top_frac: float = TOP_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Find ``n`` well-separated high-value points on the ``dim``-simplex.

    Parameters
    ----------
    objective : callable
        ``(N, dim) -> (N,)``, e.g. ``Ensemble.predict``.  Maximised.
    dim : simplex dimensionality (>= 2).
    n : number of optima to return; defaults to :func:`n_optima` for ``dim``.
    seed : seed for the warm-start design (``None`` = nondeterministic).
    n_lines : number of 24-point lines to measure; defaults to the warm start's
        own budget, :func:`warm_start.warm_start.n_lines`.
    top_frac : fraction of the design kept as optima candidates (default 5%).

    Returns
    -------
    ``(X, y)`` — an ``(n, dim)`` array of compositions and their ``(n,)``
    objective values, ordered as they were greedily selected (``X[0]`` is the
    single highest-valued sample found).
    """
    if dim < 2:
        raise ValueError("dim must be >= 2")
    if n is None:
        n = n_optima(dim)
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    if n_lines is None:
        n_lines = default_n_lines(dim)

    # Stage 1: the warm-start line design.
    _, X = greedy_lines(int(n_lines), dim, seed=seed)
    y = np.asarray(objective(X), dtype=float).ravel()
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"objective returned {y.shape[0]} values for {X.shape[0]} points")

    # Stage 2: keep the top `top_frac`, but never fewer candidates than optima
    # requested (5% of a small design can be shorter than n).
    k = max(n, int(np.ceil(top_frac * X.shape[0])))
    k = min(k, X.shape[0])
    top = np.argsort(y)[::-1][:k]     # descending value; top[0] is the global best
    Xc, yc = X[top], y[top]

    # Stage 3: greedy maximin over the survivors, seeded by the best point.
    # Candidates are value-sorted, so index 0 is the global max.
    chosen = maximin_subset(Xc, n, start=0)
    return Xc[chosen], yc[chosen]
