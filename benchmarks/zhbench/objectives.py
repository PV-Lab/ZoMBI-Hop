"""Objective registry.

Every objective exposes the same surface:

    dim           number of composition components
    domain        "simplex" (default) or "cube"
    maximize      bool
    fn(x)         (d,) -> float, noiseless ground truth
    true_optima   (k, d) reference optima
    true_values   (k,)   their objective values

Nothing here re-implements a landscape. The synthetic ones come from
``synthetic_data.ensemble`` (Brianna's generator: Ackley basins on a Perlin
background, plus distractors, ridges, plateaus and anisotropy) and the real ones
from ``warm_start.warm_gp_landscape``, which fits a GP to an entire hardware
campaign and detects its peaks.

Two honesty notes that belong next to the data, not buried in a report:

  * ``real3d`` / ``real4d`` reference optima are peaks of a SURROGATE fit to the
    campaign, not hardware-validated optima. They are the same reference the team
    already tunes hyperparameters against, so they are the right target -- but a
    peak_ratio against them is a statement about the surrogate.
  * Those two landscapes are shallow. Measured with
    ``metrics.landscape_contrast``: only 4 of 24 (3-D) and 11 of 24 (4-D) detected
    peaks clear the 99th percentile of uniform random sampling. Read their scores
    with that in mind; the sharp multi-optimum test is the ensemble suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np


@dataclass
class Objective:
    name: str
    dim: int
    fn: Callable[[np.ndarray], float]
    true_optima: np.ndarray
    true_values: np.ndarray
    maximize: bool = True
    domain: str = "simplex"
    meta: dict = field(default_factory=dict)

    @property
    def n_true(self) -> int:
        return int(self.true_optima.shape[0])

    def __call__(self, x) -> float:
        return float(self.fn(np.asarray(x, dtype=float).ravel()))


# --- synthetic: the ensemble generator ---------------------------------------

def make_ensemble(dim: int = 3, n_optima: int | None = None, landscape: int = 0,
                  seed: int = 0, basin_width: float | None = None,
                  domain: str = "simplex", **overrides) -> Objective:
    """A random ensemble landscape, optionally with ``n_optima`` pinned.

    ``landscape`` indexes the Sobol' sweep of ensemble configurations, so
    ``landscape = 0, 1, 2, ...`` walks a low-discrepancy path through the whole
    configuration space and each index is a stable, reproducible landscape.
    Overriding ``n_optima`` while leaving everything else on that path is exactly
    the needle-count sweep: same family of landscapes, more needles.

    Note ``basin_width`` is an Ackley sharpness: LARGER means NARROWER basins.
    """
    from synthetic_data.ensemble import (CartesianEnsemble, Ensemble,
                                         random_ensemble_config)

    cfg = random_ensemble_config(int(dim), index=int(landscape), seed=int(seed))
    if n_optima is not None:
        cfg["n_optima"] = int(n_optima)
    if basin_width is not None:
        cfg["basin_width"] = float(basin_width)
    cfg.update(overrides)
    cfg["seed"] = int(seed) * 10_000 + int(landscape)

    cls = CartesianEnsemble if domain == "cube" else Ensemble
    land = cls(**cfg)

    # known_maxima is a property that re-predicts on every access; take it once.
    km = land.known_maxima
    P = (np.asarray([c for c, _ in km], dtype=float) if km
         else np.empty((0, int(dim))))
    V = np.asarray([v for _, v in km], dtype=float) if km else np.empty(0)

    name = f"ensemble{dim}d"
    if n_optima is not None:
        name += f"_n{n_optima}"
    return Objective(
        name=f"{name}_L{landscape}",
        dim=int(dim),
        fn=lambda x, _l=land: float(_l(np.asarray(x, dtype=float).ravel())),
        true_optima=P,
        true_values=V,
        maximize=True,
        domain=domain,
        meta={"kind": "ensemble", "landscape": int(landscape), "seed": int(seed),
              "n_optima_requested": n_optima,
              "n_optima_placed": int(land.peak_centers.shape[0]),
              "n_optima_advertised": int(P.shape[0]),
              "config": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in cfg.items()}},
    )


# --- real campaigns: full-campaign GP surrogates ------------------------------

@lru_cache(maxsize=4)
def _fullgp(dim: int) -> dict:
    from warm_start.warm_gp_landscape import fullgp_objective
    return fullgp_objective(int(dim))


def make_real_gp(dim: int) -> Objective:
    """GP fit to the whole real campaign; reference optima are its detected peaks.

    d=3 reads ``data/2nd_real_run.db`` (41 lines / 953 rows, 644 scored).
    d=4 reads ``data/3rd_real_run.db`` (61 lines / 1358 rows, 1224 scored).
    Both are gitignored; ``data/2nd_real_run.db`` is recoverable with
    ``git show origin/evelyn-compositional:data/2nd_real_run.db > data/2nd_real_run.db``.
    """
    L = _fullgp(int(dim))
    fn = L["fn"]
    P = np.asarray(L["peaks"], dtype=float)
    V = np.asarray([float(fn(p)) for p in P], dtype=float)
    return Objective(
        name=f"real{dim}d",
        dim=int(dim),
        fn=fn,
        true_optima=P,
        true_values=V,
        maximize=True,
        domain="simplex",
        meta={"kind": "real_gp", "n_campaign_points": int(L["n_points"]),
              "n_campaign_lines": int(L["n_lines"]),
              "reference": "surrogate peaks, not hardware-validated"},
    )


# --- registry ----------------------------------------------------------------

def build(spec: dict) -> Objective:
    """Build an objective from a config dict. ``kind`` selects the family."""
    spec = dict(spec)
    kind = spec.pop("kind")
    if kind == "ensemble":
        return make_ensemble(**spec)
    if kind == "real_gp":
        return make_real_gp(**spec)
    raise ValueError(f"unknown objective kind: {kind!r}")
