"""Full-campaign GP surrogates and their reference optima.

A benchmark-local copy of ``warm_start.warm_gp_landscape.fullgp_objective``, for
two reasons.

**Dimension.** Upstream detects peaks on a simplex lattice (``simplex_grid``),
which is O(n^(d-1)) and stops being usable past d=4. The 6-D campaign needs a
peak finder that works off a probe cloud instead. The GP itself is unchanged --
``build_gp_landscape`` with ``GP_LENGTH_SCALE`` is imported from upstream, so the
surface is identical where both apply.

**Honesty of the reference set.** Upstream keeps any local maximum rising 12% of
the way from the GP median to the max. On the real campaigns that admits a lot of
GP wiggle along measured lines: at ``prominence_frac=0.12`` only 15% (3-D) and 46%
(4-D) of the resulting "true optima" stand above the 99th percentile of uniform
random sampling, i.e. most of them are artifacts, and a ``peak_ratio`` against
them measures recovery of artifacts. Two changes fix that:

  * ``prominence_frac`` defaults to 0.3 here.
  * A peak must have **measured support**: at least one real campaign sample
    within ``r`` of it whose measured objective is within the same prominence
    fraction of the peak's predicted value. A GP bump with no measurement behind
    it is not a discovered material.

``n_true`` is reported at both prominence levels so the effect is visible rather
than assumed.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.spatial import cKDTree

from ._repo import REPO_ROOT

#: Per-dimension campaign source. d=3 and d=4 mirror
#: ``warm_gp_landscape.CAMPAIGNS`` exactly; d=6 is new here (see
#: ``benchmarks/UPSTREAM_REQUESTS.md``).
CAMPAIGNS: dict[int, dict] = {
    3: {"db": "data/2nd_real_run.db",
        "columns": ("FAPbI3", "MAPbI3", "MAPbBr3")},
    4: {"db": "data/3rd_real_run.db",
        "columns": ("FAPbI3", "MAPbI3", "MAPbBr3", "CsPbI3")},
    6: {"db": "data/4th_real_run.db",
        "columns": ("FAPbI3", "CsPbI3", "FAPbBr3", "MACl", "MAPbI3", "MAPbBr3")},
}

DEFAULT_PROMINENCE = 0.3
UPSTREAM_PROMINENCE = 0.12
DEFAULT_MIN_SEP = 0.06          # == warm_gp_landscape._PEAK_MIN_SEP


@dataclass
class Landscape:
    dim: int
    predict: object
    X: np.ndarray
    Y: np.ndarray
    line_id: np.ndarray
    peaks: np.ndarray
    peak_values: np.ndarray
    diagnostics: dict


def load_campaign(dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(X, Y, line_id)`` for one campaign, unscored rows dropped.

    Same query and same renormalisation as ``warm_gp_landscape.load_campaign``,
    extended to d=6.
    """
    spec = CAMPAIGNS[int(dim)]
    db = f"{REPO_ROOT}/{spec['db']}"
    cols = ", ".join(spec["columns"])
    with sqlite3.connect(db) as conn:
        rows = list(conn.execute(
            f"SELECT Iteration, {cols}, Objective FROM results ORDER BY rowid"))
    line_id = np.array([r[0] for r in rows], dtype=float)
    X = np.array([r[1:-1] for r in rows], dtype=float)
    Y = np.array([np.nan if r[-1] is None else r[-1] for r in rows], dtype=float)
    ok = np.isfinite(Y) & np.isfinite(X).all(axis=1)
    X, Y, line_id = X[ok], Y[ok], line_id[ok].astype(int)
    X = X / X.sum(axis=1, keepdims=True)
    return X, Y, line_id


def detect_supported_peaks(predict, X_data: np.ndarray, Y_data: np.ndarray,
                           dim: int, *, prominence_frac: float = DEFAULT_PROMINENCE,
                           min_sep: float = DEFAULT_MIN_SEP, r: float = 0.05,
                           n_probe: int = 60_000, seed: int = 0
                           ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Local maxima of the GP that are prominent AND backed by a measurement.

    The probe cloud is uniform Dirichlet plus the measured points themselves, so
    peaks sitting exactly on measured lines are not missed by a coarse sample. A
    probe is a local maximum when nothing within ``min_sep`` predicts higher;
    survivors are taken in descending value and accepted only if ``min_sep`` from
    every peak already kept, which is upstream's rule.

    The measured-support test is the addition: a kept peak needs a real campaign
    sample within ``r`` whose measured ``y`` is within ``prominence_frac`` of the
    peak's own prominence. Without it the reference set is dominated by GP
    excursions between measurements.
    """
    rng = np.random.default_rng(seed)
    probes = np.vstack([rng.dirichlet(np.ones(dim), size=n_probe), X_data])
    z = np.asarray(predict(probes), dtype=float).ravel()

    tree = cKDTree(probes)
    is_max = np.fromiter(
        (z[i] >= z[nb].max() for i, nb in
         enumerate(tree.query_ball_point(probes, r=min_sep))),
        dtype=bool, count=probes.shape[0])

    background = float(np.median(z))
    span = float(z.max()) - background
    floor = background + prominence_frac * span
    cand = np.where(is_max & (z >= floor))[0]
    cand = cand[np.argsort(z[cand])[::-1]]

    kept: list[int] = []
    for idx in cand:
        p = probes[idx]
        if all(np.linalg.norm(p - probes[k]) >= min_sep for k in kept):
            kept.append(int(idx))
    naive = np.asarray(kept, dtype=int)

    # Measured support.
    data_tree = cKDTree(X_data)
    supported: list[int] = []
    for idx in naive:
        near = data_tree.query_ball_point(probes[idx], r=r)
        if not near:
            continue
        thresh = z[idx] - prominence_frac * max(z[idx] - background, 0.0)
        if float(np.max(Y_data[near])) >= thresh:
            supported.append(int(idx))
    sup = np.asarray(supported, dtype=int)

    diag = {
        "prominence_frac": float(prominence_frac),
        "min_sep": float(min_sep),
        "support_radius": float(r),
        "n_probe": int(probes.shape[0]),
        "background_median": background,
        "n_peaks_naive": int(naive.size),
        "n_peaks_supported": int(sup.size),
        "n_peaks_dropped_unsupported": int(naive.size - sup.size),
    }
    if sup.size == 0:
        return np.empty((0, dim)), np.empty(0), diag
    return probes[sup], z[sup], diag


@lru_cache(maxsize=8)
def campaign_landscape(dim: int, prominence_frac: float = DEFAULT_PROMINENCE,
                       seed: int = 0) -> Landscape:
    """Fit the full-campaign GP and detect its supported reference optima."""
    from ._repo import _ensure_path
    _ensure_path()
    from warm_start.test_greedy_optima_gp import GP_LENGTH_SCALE, build_gp_landscape

    X, Y, line_id = load_campaign(dim)
    predict = build_gp_landscape(X, Y, GP_LENGTH_SCALE)
    peaks, vals, diag = detect_supported_peaks(
        predict, X, Y, dim, prominence_frac=prominence_frac, seed=seed)

    # Report the upstream setting alongside, so the effect of the change is
    # visible in every run's artifacts rather than argued about in a doc.
    if abs(prominence_frac - UPSTREAM_PROMINENCE) > 1e-12:
        _, _, d0 = detect_supported_peaks(
            predict, X, Y, dim, prominence_frac=UPSTREAM_PROMINENCE, seed=seed)
        diag["n_peaks_naive_at_0.12"] = d0["n_peaks_naive"]
        diag["n_peaks_supported_at_0.12"] = d0["n_peaks_supported"]

    diag.update({"n_campaign_points": int(X.shape[0]),
                 "n_campaign_lines": int(np.unique(line_id).size),
                 "gp_length_scale": float(GP_LENGTH_SCALE),
                 "db": CAMPAIGNS[int(dim)]["db"],
                 "columns": list(CAMPAIGNS[int(dim)]["columns"])})
    return Landscape(dim=int(dim), predict=predict, X=X, Y=Y, line_id=line_id,
                     peaks=peaks, peak_values=vals, diagnostics=diag)


def unsupported_mask(X_query: np.ndarray, X_data: np.ndarray,
                     radius: float = 0.12) -> np.ndarray:
    """True where a query point has no campaign measurement within ``radius``.

    The 6-D surrogate extrapolates in the high-MACl and high-FAPbI3 corners the
    campaign never reached. Optimisers are not forbidden from going there -- that
    would be a modification -- but runs report what fraction of their samples
    landed in unsupported territory, so a result driven by extrapolation is
    visible.
    """
    d, _ = cKDTree(X_data).query(np.atleast_2d(X_query), k=1)
    return np.asarray(d, dtype=float).ravel() > radius
