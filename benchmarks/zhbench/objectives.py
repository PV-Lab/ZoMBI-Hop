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
from ``.landscapes``, which fits the upstream fixed-length-scale GP to an entire
hardware campaign and detects the peaks that have measured support.

Two honesty notes that belong next to the data, not buried in a report:

  * Reference optima on the real campaigns are peaks of a SURROGATE fit to the
    campaign, not hardware-validated optima. They are the same kind of reference
    the team already tunes hyperparameters against, so they are the right target
    -- but a peak_ratio against them is a statement about the surrogate.
  * The three campaigns differ a lot in how discriminative they are. Fraction of
    reference peaks clearing the 99th percentile of uniform random sampling
    (``metrics.landscape_contrast``), at prominence 0.3 with measured support::

        real3d   n_true 14   contrast 0.21     weak -- barely separates methods
        real4d   n_true 27   contrast 0.52     usable but soft
        real6d   n_true 68   contrast 0.75     the strong real test

    Read real3d's scores with that firmly in mind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    #: Optional (n, d) -> (n,) evaluator. The surrogate objectives are sklearn GPs,
    #: where per-call overhead dominates: ~1 ms of Python and validation per point
    #: against microseconds of actual kernel algebra. A 2000-sample run makes 2000
    #: single-point calls, so batching is a large win -- and it keeps sklearn's
    #: threaded code out of the inner loop, which matters on Windows where sklearn
    #: (vcomp140) and torch (libiomp5md) each load a different OpenMP runtime.
    fn_batch: Callable[[np.ndarray], np.ndarray] | None = None

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

def make_real_gp(dim: int, prominence: float | None = None) -> Objective:
    """GP fit to the whole real campaign; reference optima are its supported peaks.

    d=3 ``data/2nd_real_run.db`` (41 lines / 953 rows, 644 scored)
    d=4 ``data/3rd_real_run.db`` (61 lines / 1358 rows, 1224 scored)
    d=6 ``data/4th_real_run.db`` (111 lines / 2423 rows, 2042 scored across 109 lines)

    All three are gitignored. ``data/2nd_real_run.db`` is recoverable with
    ``git show origin/evelyn-compositional:data/2nd_real_run.db > data/2nd_real_run.db``;
    the other two came from Colin.

    d=6 is built on the **full simplex**. The campaign's per-component maxima
    (MACl 0.412, FAPbI3 0.515) look like a search box but are not one: of 218 line
    endpoints only 7 sit within 0.02 of a component maximum -- exactly one per
    component, i.e. only the point that defines it -- while 92 endpoints have a
    component at zero. LineBO presses against the zero faces constantly and never
    against an upper face, which is what a real box would have produced. If Brianna
    confirms a box was used after all, restrict here.
    """
    from . import landscapes as LS

    prom = LS.DEFAULT_PROMINENCE if prominence is None else float(prominence)
    L = LS.campaign_landscape(int(dim), prominence_frac=prom)
    predict = L.predict

    def fn(x, _p=predict):
        return float(_p(np.asarray(x, dtype=float).reshape(1, -1))[0])

    def fn_batch(X, _p=predict):
        from .landscapes import predict_chunked
        return predict_chunked(_p, np.atleast_2d(np.asarray(X, dtype=float)))

    return Objective(
        name=f"real{dim}d",
        dim=int(dim),
        fn=fn,
        fn_batch=fn_batch,
        true_optima=L.peaks,
        true_values=L.peak_values,
        maximize=True,
        domain="simplex",
        meta={"kind": "real_gp",
              "reference": "supported surrogate peaks; NOT hardware-validated",
              **L.diagnostics},
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
