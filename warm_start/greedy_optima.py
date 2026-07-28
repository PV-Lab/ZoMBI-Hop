"""
Greedy optima finder
====================

Locate a spread-out set of candidate optima on the probability simplex by
evaluating an objective on a free design (no hardware line constraint).

Lines of 24 exist because the *printer* measures that way.  For reference
optima on a known oracle / RF(g) twin you can query anywhere, so the default
design is a large **Sobol** (or Dirichlet) cloud on the simplex — denser and
not forced onto chords.

The procedure has three stages:

1. **Sample the simplex.**  Draw ``n_design`` free compositions (default Sobol
   via :func:`ela.features.sample_simplex_sobol`, or uniform Dirichlet).  Size
   auto-grows with ``n`` so survivors ≫ optima (see :data:`CANDIDATE_MULTIPLIER`
   / :data:`TOP_FRACTION`).  Pass ``design="lines"`` only if you explicitly want
   a hardware-faithful warm-start line design.
2. **Keep the top survivors.**  Retain the highest-valued fraction
   (``top_frac``, default 25%), but never fewer than
   ``CANDIDATE_MULTIPLIER * n`` candidates when the design allows it.
3. **Greedy maximin over the survivors.**  First pick = global best among
   survivors; each next pick maximises distance to the nearest already-chosen
   point (:func:`maximin_subset`).

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

from typing import Callable, Literal

import numpy as np

from warm_start.warm_start import (
    POINTS_PER_LINE,
    greedy_lines,
    maximin_subset,
    n_lines as default_n_lines,
)

DesignKind = Literal["sobol", "dirichlet", "lines"]

# Number of optima to report, by simplex dimension.
N_OPTIMA = {
    3:   5,
    4:   10,
    10:  20,
}

# Keep enough of the design that maximin has room to spread.
TOP_FRACTION = 0.25
CANDIDATE_MULTIPLIER = 10  # aim for >= 10 survivors per requested optimum
# Floor on free-point design size (on top of the multiplier/top_frac requirement).
DESIGN_POOL_FLOOR = 8192
DEFAULT_DESIGN: DesignKind = "sobol"


def n_optima(dim: int) -> int:
    """Default number of optima to report at simplex ``dim``.

    Uses the hardcoded :data:`N_OPTIMA` table for the benchmarked dimensions
    (3/4/10); other dimensions fall back to ``2 * dim``, the slope those anchors
    sit on.
    """
    return N_OPTIMA.get(int(dim), 2 * int(dim))


def n_design_for_optima(
    n: int,
    *,
    top_frac: float = TOP_FRACTION,
    candidate_multiplier: int = CANDIDATE_MULTIPLIER,
    pool_floor: int = DESIGN_POOL_FLOOR,
) -> int:
    """Free-point design size so top-``top_frac`` can hold ``multiplier * n`` pts."""
    top_frac = float(top_frac)
    if not (0.0 < top_frac <= 1.0):
        raise ValueError("top_frac must be in (0, 1]")
    needed = int(np.ceil(candidate_multiplier * int(n) / top_frac))
    return max(int(pool_floor), needed)


def n_lines_for_optima(
    dim: int,
    n: int,
    *,
    top_frac: float = TOP_FRACTION,
    candidate_multiplier: int = CANDIDATE_MULTIPLIER,
) -> int:
    """Line count for ``design='lines'`` (hardware-faithful path).

    Never smaller than the warm-start :func:`warm_start.warm_start.n_lines`.
    """
    needed_pts = n_design_for_optima(
        n, top_frac=top_frac, candidate_multiplier=candidate_multiplier,
        pool_floor=0,
    )
    needed_lines = int(np.ceil(needed_pts / POINTS_PER_LINE))
    return max(default_n_lines(dim), needed_lines)


def _survivor_count(
    n_design: int,
    n: int,
    *,
    top_frac: float,
    candidate_multiplier: int,
) -> int:
    """How many top-valued design points to hand to maximin."""
    by_frac = int(np.ceil(top_frac * n_design))
    by_mult = int(candidate_multiplier) * int(n)
    return int(min(n_design, max(n, by_frac, by_mult)))


def _sample_design(
    n_design: int,
    dim: int,
    *,
    seed: int | None,
    design: DesignKind,
    n_lines: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Return ``(X, meta)`` for the stage-1 design."""
    if design == "dirichlet":
        rng = np.random.default_rng(seed)
        X = rng.dirichlet(np.ones(dim), size=int(n_design))
        return X, {"design": "dirichlet", "n_design": int(X.shape[0])}

    if design == "sobol":
        from ela.features import sample_simplex_sobol

        sobol_seed = 0 if seed is None else int(seed)
        X = sample_simplex_sobol(dim, int(n_design), seed=sobol_seed)
        return X, {"design": "sobol", "n_design": int(X.shape[0]), "sobol_seed": sobol_seed}

    if design == "lines":
        L = int(n_lines) if n_lines is not None else n_lines_for_optima(dim, max(1, n_design // POINTS_PER_LINE))
        # If caller passed n_design instead of n_lines, honour density via ceil.
        if n_lines is None and n_design is not None:
            L = max(L, int(np.ceil(int(n_design) / POINTS_PER_LINE)))
        _, X = greedy_lines(L, dim, seed=seed)
        return X, {"design": "lines", "n_lines": L, "n_design": int(X.shape[0])}

    raise ValueError(f"unknown design={design!r}; expected sobol|dirichlet|lines")


def find_optima(
    objective: Callable[[np.ndarray], np.ndarray],
    dim: int,
    n: int | None = None,
    *,
    seed: int | None = None,
    design: DesignKind = DEFAULT_DESIGN,
    n_design: int | None = None,
    n_lines: int | None = None,
    top_frac: float = TOP_FRACTION,
    candidate_multiplier: int = CANDIDATE_MULTIPLIER,
    edge_min: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Find ``n`` well-separated high-value points on the ``dim``-simplex.

    Parameters
    ----------
    objective : callable
        ``(N, dim) -> (N,)``, e.g. ``Ensemble.predict``.  Maximised.
    dim : simplex dimensionality (>= 2).
    n : number of optima to return; defaults to :func:`n_optima` for ``dim``.
    seed : RNG / Sobol seed (``None`` = nondeterministic Dirichlet; Sobol uses 0).
    design : ``"sobol"`` (default), ``"dirichlet"``, or ``"lines"`` (hardware).
    n_design : free-point design size; default :func:`n_design_for_optima`.
    n_lines : only for ``design="lines"``; default :func:`n_lines_for_optima`.
    top_frac : fraction of the design kept as optima candidates (default 25%).
    candidate_multiplier : floor on survivors as a multiple of ``n`` (default 10).
    edge_min : if > 0, drop compositions with any coordinate ``< edge_min``
        before ranking (ignores faces / vertices of the simplex).

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
    top_frac = float(top_frac)
    if not (0.0 < top_frac <= 1.0):
        raise ValueError("top_frac must be in (0, 1]")
    candidate_multiplier = int(candidate_multiplier)
    if candidate_multiplier < 1:
        raise ValueError("candidate_multiplier must be >= 1")
    edge_min = float(edge_min)
    if edge_min < 0.0:
        raise ValueError("edge_min must be >= 0")
    if edge_min * dim >= 1.0:
        raise ValueError(f"edge_min={edge_min} leaves an empty interior on dim={dim}")

    if n_design is None:
        n_design = n_design_for_optima(
            n, top_frac=top_frac, candidate_multiplier=candidate_multiplier,
        )
    if design == "lines" and n_lines is None:
        n_lines = n_lines_for_optima(
            dim, n, top_frac=top_frac, candidate_multiplier=candidate_multiplier,
        )

    # Stage 1: free (or line) design.
    X, _design_meta = _sample_design(
        int(n_design), dim, seed=seed, design=design, n_lines=n_lines,
    )
    y = np.asarray(objective(X), dtype=float).ravel()
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"objective returned {y.shape[0]} values for {X.shape[0]} points")

    # Optional: ignore the simplex boundary (faces / edges / vertices).
    if edge_min > 0.0:
        interior = np.all(X >= edge_min, axis=1)
        if not np.any(interior):
            raise ValueError(
                f"no design points with all coords >= edge_min={edge_min}"
            )
        X, y = X[interior], y[interior]

    # Stage 2: top survivors — prefer many more candidates than optima.
    k = _survivor_count(
        X.shape[0], n,
        top_frac=top_frac,
        candidate_multiplier=candidate_multiplier,
    )
    top = np.argsort(y)[::-1][:k]     # descending value; top[0] is the global best
    Xc, yc = X[top], y[top]

    # Stage 3: greedy maximin over the survivors, seeded by the best point.
    chosen = maximin_subset(Xc, n, start=0)
    return Xc[chosen], yc[chosen]
