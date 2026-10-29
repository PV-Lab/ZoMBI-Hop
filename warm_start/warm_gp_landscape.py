"""
warm_start/warm_gp_landscape.py
===============================
The **warm-started landscape**: a GP fit to nothing but the warm-start lines of a
real campaign, plus its automatically detected peaks — every local maximum of the
GP surface, however many there are, wherever they sit.

This is the surrogate the three hyperparameter-tuning modes (dynamic / mixture /
static) train against, so its peaks are the ``dist_to_needles`` reference.  A
tuning run is only as meaningful as this landscape, which is why this module
exists on its own and renders a picture: the peaks below are what the tuner will
be scored against, and they are worth eyeballing before any GPU time is spent.

Where the lines come from
-------------------------
The real campaigns measured a *line* at a time — 24 evenly spaced compositions
between two free endpoints — and the DB's ``Iteration`` column is the line index:

    data/2nd_real_run.db   d=3   41 lines (~24 pts each, 953 rows)
    data/3rd_real_run.db   d=4   61 lines (~24 pts each, 1358 rows)

matching the ~40 and ~60 lines a campaign actually prints.  Some rows have no
``Objective`` (the assay failed), so a line is rarely a full 24 points; those rows
are dropped and the line is kept at whatever length survives.

A warm start is the first 15% of that budget, :func:`warm_start.warm_start.n_lines`
lines (4 at d=3, 8 at d=4).  Which 4 of the 41?  The campaign's own lines were
chosen *adaptively* — they crowd wherever ZoMBI-Hop was already interested — so
taking the first few, or a random few, gives a lopsided design that a real warm
start would never produce.  Instead :func:`select_warm_lines` picks the subset by
the same criterion :func:`~warm_start.warm_start.greedy_lines` optimises, the
coverage cost

    cost(D) = mean_p  min_{x in D} ||p - x||^p        (p = _SCORE_POWER = 4)

over a uniform simplex probe cloud — greedy line-at-a-time, then whole-line
coordinate-descent sweeps.  It is the discrete version of the same algorithm:
here the candidate set is the campaign's real measured lines rather than free
endpoint pairs, so the result is a design that both *covers the simplex* and
consists only of lines someone actually ran.

Why fit to only the warm start
------------------------------
Fitting the GP to the whole campaign would give a much better surrogate — and the
wrong one.  The question the warm start has to answer is "after 15% of the budget,
where do we think the optima are?", so the landscape must be built from exactly
the information a warm start has: those 4 (or 8) lines and nothing else.  The GP
uses a Matern(nu=2.5) kernel with the length scale **fixed at 0.05**
(:data:`GP_LENGTH_SCALE`), the same kernel and length scale
``warm_start/test_greedy_optima_gp.py`` and ``plot_run.fit_gp_background`` use, so
"the GP landscape" means one thing across the repo.

Consequence worth keeping in mind while reading the figures: away from the lines
the GP has no data, and a fixed 0.05 length scale is short relative to the gaps
between 4 lines on a triangle.  The landscape therefore relaxes to the prior mean
in the uncovered regions, so the auto-detected peaks cluster *on or near the
measured lines* — that is where the surface actually rises.  That is honest, not a
bug, but it is the main thing to judge when deciding whether this landscape is a
fair tuning target.

Output
------
    warm_start/warm_gp_landscape/warm_gp_landscape_3d.png
    warm_start/warm_gp_landscape/warm_gp_landscape_4d.png
    warm_start/warm_gp_landscape/warm_gp_landscape_summary.json

Run:
  uv run python -m warm_start.warm_gp_landscape
  uv run python -m warm_start.warm_gp_landscape --dims 3
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from visualization.plot_run import (  # noqa: E402
    TETRA_EDGES,
    TETRA_VERTS,
    comp_to_xy,
    comp_to_xyz,
    simplex_grid,
)
from warm_start.test_greedy_optima_gp import (  # noqa: E402
    GP_LENGTH_SCALE,
    build_gp_landscape,
)
from warm_start.warm_start import (  # noqa: E402
    _SCORE_POWER,
    coverage_stats,
    n_lines,
)

OUT_DIR = _HERE / "warm_gp_landscape"

#: Per-dimension campaign source.  ``columns`` is the composition basis in the
#: order the simplex plots expect (column i -> corner i); ``labels`` names those
#: corners.  ``n_campaign_lines`` is the campaign's own line count, recorded in
#: the figures so the warm start's 15% share is readable at a glance.
CAMPAIGNS: dict[int, dict] = {
    3: {
        "db": "data/2nd_real_run.db",
        "columns": ("FAPbI3", "MAPbI3", "MAPbBr3"),
        "n_campaign_lines": 40,
    },
    4: {
        "db": "data/3rd_real_run.db",
        "columns": ("FAPbI3", "MAPbI3", "MAPbBr3", "CsPbI3"),
        "n_campaign_lines": 60,
    },
}

GRID_3D = 220          # ternary contour resolution
GRID_4D = 42           # 3-simplex lattice resolution (O(n^3) points)
SEED = 0

# Probe cloud for the discrete line-subset coverage objective.  Fixed (not scaled
# with the design) because the candidate set here is small — a few dozen lines —
# so scoring is cheap and a larger cloud just buys a steadier ranking.
_SELECT_PROBES = 20000
# Whole-line coordinate-descent sweeps after the greedy subset pass, mirroring
# ``greedy_lines``' ``_REFINE_SWEEPS``.
_SELECT_SWEEPS = 4
# Auto peak detection on the GP surface.  A grid point is a peak candidate if it
# is a local maximum; ``_PEAK_MIN_SEP`` then merges candidates that sit inside one
# bump, and ``_PEAK_PROMINENCE_FRAC`` drops anything that fails to rise this
# fraction of the way from the background (median GP value) to the global max, so
# the flat prior region away from the data never spawns spurious peaks.
_PEAK_MIN_SEP = 0.06
_PEAK_PROMINENCE_FRAC = 0.12


# ── campaign loading ─────────────────────────────────────────────────────────

def load_campaign(dim: int, repo_root: Path = _REPO_ROOT
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one campaign as ``(X, Y, line_id)``, dropping unscored rows.

    ``X`` is ``(N, dim)`` compositions in :data:`CAMPAIGNS` column order, ``Y`` is
    the campaign ``Objective``, and ``line_id`` is the DB's ``Iteration`` — the
    index of the printed line each measurement belongs to.  Rows whose
    ``Objective`` is NULL or non-finite are dropped: the assay failed, so the
    composition was printed but never scored, and a GP cannot use it.
    """
    spec = CAMPAIGNS[int(dim)]
    db = repo_root / spec["db"]
    if not db.exists():
        raise FileNotFoundError(f"campaign DB not found: {db}")

    cols = ", ".join(spec["columns"])
    with sqlite3.connect(str(db)) as conn:
        rows = list(conn.execute(
            f"SELECT Iteration, {cols}, Objective FROM results ORDER BY rowid"))

    line_id = np.array([r[0] for r in rows], dtype=float)
    X = np.array([r[1:-1] for r in rows], dtype=float)
    Y = np.array([np.nan if r[-1] is None else r[-1] for r in rows], dtype=float)

    ok = np.isfinite(Y) & np.isfinite(X).all(axis=1)
    X, Y, line_id = X[ok], Y[ok], line_id[ok].astype(int)
    # Renormalise away the DB's rounding drift so every row is exactly a
    # composition (the plots and the GP both assume rows sum to 1).
    X = X / X.sum(axis=1, keepdims=True)
    return X, Y, line_id


# ── warm-start line selection ────────────────────────────────────────────────

def _coverage_cost(d2: np.ndarray) -> float:
    """Coverage cost ``mean_p d^p`` from *squared* probe distances ``d2``."""
    return float((d2 ** (0.5 * _SCORE_POWER)).mean())


def _min_sq_dist(probes: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """For each probe, the squared distance to the nearest row of ``pts``."""
    return ((probes ** 2).sum(1)[:, None]
            + (pts ** 2).sum(1)[None, :]
            - 2.0 * (probes @ pts.T)).min(axis=1)


def select_warm_lines(X: np.ndarray, line_id: np.ndarray, n_select: int,
                      *, seed: int = SEED,
                      n_probes: int = _SELECT_PROBES,
                      sweeps: int = _SELECT_SWEEPS) -> np.ndarray:
    """Choose the ``n_select`` campaign lines that best cover the simplex.

    The discrete counterpart of :func:`warm_start.warm_start.greedy_lines`: same
    coverage cost, same greedy-then-refine structure, but the candidates are the
    campaign's real measured lines instead of free endpoint pairs.  A greedy pass
    adds whichever line drops the cost most; then ``sweeps`` rounds of coordinate
    descent lift each chosen line out in turn and re-pick it against the coverage
    the others already provide, so an early greedy choice that a later one made
    redundant can be traded away.

    Returns the selected line ids, in the order they were chosen.
    """
    ids = np.unique(line_id)
    if n_select >= ids.size:
        return ids
    rng = np.random.default_rng(seed)
    probes = rng.dirichlet(np.ones(X.shape[1]), size=int(n_probes))

    # D[j] = squared distance from every probe to the nearest point of line ids[j].
    D = np.stack([_min_sq_dist(probes, X[line_id == i]) for i in ids])

    chosen: list[int] = []
    covered = np.full(probes.shape[0], np.inf)
    for _ in range(int(n_select)):
        best_j, best_cost = -1, np.inf
        for j in range(ids.size):
            if j in chosen:
                continue
            cost = _coverage_cost(np.minimum(covered, D[j]))
            if cost < best_cost:
                best_j, best_cost = j, cost
        chosen.append(best_j)
        covered = np.minimum(covered, D[best_j])

    # Coordinate descent: re-choose each slot against the others' coverage.
    for _ in range(int(sweeps)):
        for slot in range(len(chosen)):
            others = [chosen[t] for t in range(len(chosen)) if t != slot]
            base = (np.minimum.reduce([D[j] for j in others]) if others
                    else np.full(probes.shape[0], np.inf))
            best_j, best_cost = chosen[slot], _coverage_cost(
                np.minimum(base, D[chosen[slot]]))
            for j in range(ids.size):
                if j in others or j == chosen[slot]:
                    continue
                cost = _coverage_cost(np.minimum(base, D[j]))
                if cost < best_cost:
                    best_j, best_cost = j, cost
            chosen[slot] = best_j

    return ids[np.array(chosen, dtype=int)]


# ── the warm-started landscape ───────────────────────────────────────────────

def build_warm_landscape(dim: int, *, seed: int = SEED,
                         n_warm_lines: int | None = None) -> dict:
    """Fit the length-scale-0.05 GP to a campaign's warm-start lines.

    Returns a dict with the fitted ``predict`` callable, the warm-start subset
    ``(X_warm, Y_warm, warm_line_id)`` it was fit to, the selected line ids, and
    the full campaign for context.  Nothing outside the warm-start subset ever
    reaches the GP.
    """
    dim = int(dim)
    if n_warm_lines is None:
        n_warm_lines = n_lines(dim)

    X, Y, line_id = load_campaign(dim)
    picked = select_warm_lines(X, line_id, n_warm_lines, seed=seed)
    mask = np.isin(line_id, picked)
    X_warm, Y_warm, id_warm = X[mask], Y[mask], line_id[mask]

    predict = build_gp_landscape(X_warm, Y_warm, GP_LENGTH_SCALE)
    return {
        "dim": dim,
        "predict": predict,
        "X_warm": X_warm,
        "Y_warm": Y_warm,
        "warm_line_id": id_warm,
        "picked_lines": picked,
        "n_warm_lines": int(n_warm_lines),
        "X_all": X,
        "Y_all": Y,
        "line_id_all": line_id,
    }


# ── automatic peak detection ─────────────────────────────────────────────────

def detect_peaks(grid: np.ndarray, z: np.ndarray, *,
                 min_sep: float = _PEAK_MIN_SEP,
                 prominence_frac: float = _PEAK_PROMINENCE_FRAC
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Find every local maximum of the GP surface ``z`` over ``grid``.

    The count is not fixed: the landscape is scanned for grid points that are
    local maxima (no lattice neighbour within ``3 * spacing`` is higher), the
    background (median GP value over the grid — the level the fixed-length-scale GP
    relaxes to away from the data) is subtracted, and only candidates rising at
    least ``prominence_frac`` of the way from that background to the global max are
    kept.  Survivors are taken in descending value and each is accepted only if it
    is at least ``min_sep`` from every peak already kept, so one bump yields one
    marker.  Peaks are found wherever the surface is highest, on or off any line.

    Returns ``(peaks, values)`` sorted by descending GP value.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(grid)
    spacing = float(np.median(tree.query(grid[:200], k=2)[0][:, 1]))
    neighbours = tree.query_ball_point(grid, r=3.0 * spacing)
    is_max = np.array([z[i] >= z[nb].max() for i, nb in enumerate(neighbours)])

    background = float(np.median(z))
    floor = background + prominence_frac * (float(z.max()) - background)
    cand = np.where(is_max & (z >= floor))[0]
    cand = cand[np.argsort(z[cand])[::-1]]        # descending value

    kept: list[int] = []
    for idx in cand:
        p = grid[idx]
        if all(np.linalg.norm(p - grid[k]) >= min_sep for k in kept):
            kept.append(idx)
    kept_arr = np.array(kept, dtype=int)
    return grid[kept_arr], z[kept_arr]


# ── objective adapter (for MOBO hyperparameter tuning) ───────────────────────

def warmgp_objective(dim: int, *, seed: int = SEED) -> dict:
    """The warm-start GP landscape packaged as a tuning objective.

    Returns a dict carrying everything ``optimize.run_mobo`` needs to score
    hyperparameters against this landscape:

      ``fn``          scalar objective ``fn(x)`` for a single composition ``x``
                      (``(dim,)`` array) — a thin wrapper over the fitted GP so it
                      matches the ``fn_callable`` contract the MOBO harness expects.
      ``predict``     the underlying batched GP ``predict(X)`` callable.
      ``peaks``       the auto-detected GP peaks (``(k, dim)``) — the objective's
                      ``true_optima``, i.e. the ``dist_to_needles`` reference.
      ``grid_pts`` /  the render grid and GP values on it (``grid_vals`` is ``None``
      ``grid_vals``   for ``dim != 3``; used only for the ternary coverage plot).
      ``n_warm_lines`` / ``picked_lines`` — provenance for logging.

    The landscape is FIXED (deterministic in ``seed``): unlike the ensemble
    objective it is not re-randomized per trial, so the static / dynamic tuning
    runs all score against exactly this surface.
    """
    L = build_warm_landscape(dim, seed=seed)
    predict = L["predict"]
    grid = simplex_grid(GRID_3D if int(dim) == 3 else GRID_4D, int(dim))
    z = predict(grid)
    peaks, _ = detect_peaks(grid, z)

    def fn(x: np.ndarray) -> float:
        return float(predict(np.asarray(x, dtype=float).reshape(1, -1))[0])

    grid_vals = z if int(dim) == 3 else None
    return {
        "dim": int(dim),
        "fn": fn,
        "predict": predict,
        "peaks": peaks,
        "grid_pts": grid if int(dim) == 3 else None,
        "grid_vals": grid_vals,
        "n_warm_lines": L["n_warm_lines"],
        "picked_lines": [int(i) for i in L["picked_lines"]],
    }


# ── full-run landscape (evaluation ground truth) ─────────────────────────────

def build_full_landscape(dim: int, *, seed: int = SEED) -> dict:
    """Fit the length-scale-0.05 GP to a campaign's ENTIRE measured run.

    The evaluation counterpart of :func:`build_warm_landscape`: the same fixed
    Matern GP and the same campaign, but fit to *every* scored point of the real
    run rather than only the selected warm-start lines. This is the honest
    ground-truth surface — "what the campaign actually measured everywhere" — that
    deployed hyperparameters are scored against, whereas the warm-start GP (limited
    to the first 15% of lines) is only the surrogate the *tuner* sees. ``seed`` is
    accepted for signature parity with :func:`build_warm_landscape` but is unused:
    the full run has no warm-line subset to select, so the surface is deterministic.
    """
    dim = int(dim)
    X, Y, line_id = load_campaign(dim)
    predict = build_gp_landscape(X, Y, GP_LENGTH_SCALE)
    return {
        "dim": dim,
        "predict": predict,
        "X_all": X,
        "Y_all": Y,
        "line_id_all": line_id,
        "n_points": int(X.shape[0]),
        "n_lines": int(np.unique(line_id).size),
    }


def fullgp_objective(dim: int, *, seed: int = SEED) -> dict:
    """The FULL-run GP landscape packaged as an evaluation objective.

    Mirrors :func:`warmgp_objective` (same keys, same detected-peak reference) but
    the surface is fit to the whole real campaign via :func:`build_full_landscape`,
    so its auto-detected peaks are the campaign's true optima — the
    ``dist_to_needles`` reference deployed hyperparameters are evaluated against.
    Deterministic; ``seed`` is unused (no warm-line subset).
    """
    L = build_full_landscape(dim, seed=seed)
    predict = L["predict"]
    grid = simplex_grid(GRID_3D if int(dim) == 3 else GRID_4D, int(dim))
    z = predict(grid)
    peaks, _ = detect_peaks(grid, z)

    def fn(x: np.ndarray) -> float:
        return float(predict(np.asarray(x, dtype=float).reshape(1, -1))[0])

    return {
        "dim": int(dim),
        "fn": fn,
        "predict": predict,
        "peaks": peaks,
        "grid_pts": grid if int(dim) == 3 else None,
        "grid_vals": z if int(dim) == 3 else None,
        "n_points": L["n_points"],
        "n_lines": L["n_lines"],
    }


# ── GP fit to a ZoMBI-Hop run directory (any d) ──────────────────────────────
#
# The campaigns above are DBs at d=3/4, where a simplex lattice is dense enough for
# detect_peaks. A 6-d hardware run is not: a lattice fine enough for a 0.05 length
# scale has billions of points. Peaks are found instead by projected gradient
# ascent on the GP mean, started from every measured point. With a short fixed
# length scale every bump of the mean sits next to data, so those starts reach
# every peak, and the same min-sep / prominence rules as detect_peaks thin them.

_ASCENT_STEPS = 400
_ASCENT_STEP0 = 0.01
_ASCENT_CHUNK = 256          # query rows per kernel block (chunk x N x d floats)
_BACKGROUND_SAMPLES = 20000  # uniform simplex draws for the background level
# runrf: trees in the forest (visualization/plot_run.RF_N_ESTIMATORS) and the
# training-point neighbourhood a peak candidate must top (rf_peaks).
RF_N_ESTIMATORS = 500
_RF_PEAK_K = 10


def _project_simplex(V: np.ndarray) -> np.ndarray:
    """Row-wise Euclidean projection onto the probability simplex."""
    n, d = V.shape
    U = -np.sort(-V, axis=1)
    css = np.cumsum(U, axis=1) - 1.0
    k = np.arange(1, d + 1)
    rho = (U - css / k > 0).sum(axis=1)
    theta = css[np.arange(n), rho - 1] / rho
    return np.maximum(V - theta[:, None], 0.0)


def _gp_mean_grad(predict, Xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """GP mean and its gradient at ``Xq`` for a :func:`build_gp_landscape` predictor.

    Uses the fitted Constant x Matern(nu=2.5) part only; the White term adds nothing
    off the training points.
    """
    gp = predict.gp
    c = float(gp.kernel_.k1.k1.constant_value)
    ls = float(gp.kernel_.k1.k2.length_scale)
    Xt, alpha = gp.X_train_, gp.alpha_.ravel()
    mean = np.empty(len(Xq))
    grad = np.empty_like(Xq)
    for i in range(0, len(Xq), _ASCENT_CHUNK):
        diff = Xq[i:i + _ASCENT_CHUNK, None, :] - Xt[None, :, :]       # (b, N, d)
        s = np.sqrt(5.0) * np.sqrt((diff ** 2).sum(-1)) / ls           # (b, N)
        e = np.exp(-s)
        mean[i:i + _ASCENT_CHUNK] = (c * (1.0 + s + s ** 2 / 3.0) * e) @ alpha
        dk = -c * (5.0 / (3.0 * ls ** 2)) * (1.0 + s) * e              # dk/dx / diff
        grad[i:i + _ASCENT_CHUNK] = np.einsum("bn,bnd->bd", dk * alpha, diff)
    return mean * predict.y_std + predict.y_mean, grad * predict.y_std


def ascend_peaks(predict, starts: np.ndarray, *,
                 min_sep: float = _PEAK_MIN_SEP,
                 prominence_frac: float = _PEAK_PROMINENCE_FRAC,
                 seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of the GP mean on the simplex, by ascent from ``starts``.

    Each start takes projected gradient steps with its own step size (grown 1.2x on
    an improving step, halved and the step rejected otherwise). The converged points
    are then filtered exactly as :func:`detect_peaks` filters lattice maxima: the
    background is the median mean over uniform simplex draws, a peak must rise
    ``prominence_frac`` of the way from it to the global max, and peaks are kept in
    descending value at least ``min_sep`` apart.

    Returns ``(peaks, values)`` sorted by descending GP value.
    """
    X = _project_simplex(np.asarray(starts, dtype=float))
    f, g = _gp_mean_grad(predict, X)
    eta = np.full(len(X), _ASCENT_STEP0)
    for _ in range(_ASCENT_STEPS):
        g_t = g - g.mean(axis=1, keepdims=True)                 # tangent to the simplex
        X_new = _project_simplex(X + eta[:, None] * g_t)
        f_new, g_new = _gp_mean_grad(predict, X_new)
        up = f_new > f
        X[up], f[up], g[up] = X_new[up], f_new[up], g_new[up]
        eta = np.where(up, eta * 1.2, eta * 0.5)
        if eta.max() < 1e-7:
            break

    rng = np.random.default_rng(seed)
    background = float(np.median(predict(rng.dirichlet(np.ones(X.shape[1]),
                                                       _BACKGROUND_SAMPLES))))
    floor = background + prominence_frac * (float(f.max()) - background)
    kept: list[int] = []
    for idx in np.argsort(f)[::-1]:
        if f[idx] < floor:
            break
        if all(np.linalg.norm(X[idx] - X[k]) >= min_sep for k in kept):
            kept.append(idx)
    kept_arr = np.array(kept, dtype=int)
    return X[kept_arr], f[kept_arr]


def _load_points(run_dir: Path, snapshot: str | None):
    """``(X, Y, snapshot)``: the finite scored points of a run dir, or of a points file.

    ``run_dir`` may also be an ``.npz`` with ``X`` (N, d) and ``Y`` (N,) arrays — e.g.
    uniform samples of a synthetic landscape (optimize/scripts/needles_rf_points.py) —
    which the surrogate is then fit to exactly as to a run's measurements. Its
    "snapshot" is the file name.
    """
    run_dir = Path(run_dir)
    if run_dir.suffix == ".npz":
        z = np.load(run_dir)
        X, Y, snapshot = np.asarray(z["X"], float), np.asarray(z["Y"], float).ravel(), run_dir.name
    else:
        from visualization.plot_run import _default_snapshot, load_run_dataset

        snapshot = snapshot or _default_snapshot(run_dir)
        X, Y = load_run_dataset(run_dir, snapshot)
    ok = np.isfinite(Y) & np.isfinite(X).all(axis=1)
    return X[ok], Y[ok], snapshot


def _fit_run_gp(run_dir: Path, length_scale: float, snapshot: str | None):
    """``(predict, X, Y, snapshot)`` for the GP over a run dir's scored points."""
    X, Y, snapshot = _load_points(run_dir, snapshot)
    return build_gp_landscape(X, Y, float(length_scale)), X, Y, snapshot


def _fit_run_rf(run_dir: Path, n_estimators: int, snapshot: str | None):
    """``(predict, X, Y, snapshot)`` for a random forest over a run dir's scored points.

    Same forest ``visualization/plot_run.py`` draws its RF backgrounds with
    (``RandomForestRegressor(n_estimators, random_state=42)``, sklearn defaults
    otherwise), so the surface is deterministic and refittable from the config.
    """
    from sklearn.ensemble import RandomForestRegressor

    X, Y, snapshot = _load_points(run_dir, snapshot)
    rf = RandomForestRegressor(n_estimators=int(n_estimators), n_jobs=-1, random_state=42)
    rf.fit(X, Y)
    # Single-point queries dominate a simulated run; thread fan-out only costs there.
    rf.set_params(n_jobs=1)

    def predict(Xq: np.ndarray) -> np.ndarray:
        return rf.predict(np.atleast_2d(np.asarray(Xq, dtype=float)))

    predict.rf = rf
    return predict, X, Y, snapshot


def rf_peaks(predict, X: np.ndarray, *, k: int = _RF_PEAK_K,
             min_sep: float = _PEAK_MIN_SEP,
             prominence_frac: float = _PEAK_PROMINENCE_FRAC,
             seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of a random-forest surface, taken among the training points.

    A forest is piecewise constant, so there is no gradient to ascend
    (:func:`ascend_peaks`); its maxima sit where the data does. A training point is a
    candidate when its prediction is at least that of its ``k`` nearest training
    neighbours, and candidates are filtered exactly as :func:`ascend_peaks` filters
    its converged points (prominence over the uniform-draw median, ``min_sep`` apart).

    Returns ``(peaks, values)`` sorted by descending prediction.
    """
    from scipy.spatial import cKDTree

    z = predict(X)
    _, nb = cKDTree(X).query(X, k=k + 1)
    is_max = z >= z[nb[:, 1:]].max(axis=1)

    rng = np.random.default_rng(seed)
    background = float(np.median(predict(rng.dirichlet(np.ones(X.shape[1]),
                                                       _BACKGROUND_SAMPLES))))
    floor = background + prominence_frac * (float(z.max()) - background)
    cand = np.where(is_max & (z >= floor))[0]
    kept: list[int] = []
    for idx in cand[np.argsort(z[cand])[::-1]]:
        if all(np.linalg.norm(X[idx] - X[j]) >= min_sep for j in kept):
            kept.append(idx)
    kept_arr = np.array(kept, dtype=int)
    return X[kept_arr], z[kept_arr]


class RunGPSurface:
    """A rungp / runrf landscape with the ``Ensemble`` interface
    ``visualization/paired_conet`` uses: batched ``predict(X)`` and ``centers`` (the
    reference optima).

    Rebuilt from the ``rungp_config.json`` evaluate.py writes into each rungp / runrf
    run dir (``surrogate`` says which; absent means GP). The surrogate is refit
    (deterministic) but the peaks are taken from the config rather than re-detected,
    so they are exactly the optima the run was scored against.
    """

    def __init__(self, cfg: dict):
        self.dim = int(cfg["dim"])
        self.centers = np.asarray(cfg["centers"], dtype=float)
        if cfg.get("surrogate") == "rf":
            self._predict, _, _, _ = _fit_run_rf(
                Path(cfg["gp_run"]), int(cfg["n_estimators"]), cfg["snapshot"])
        else:
            self._predict, _, _, _ = _fit_run_gp(
                Path(cfg["gp_run"]), float(cfg["length_scale"]), cfg["snapshot"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predict(X)


def rungp_objective(run_dir: str | Path, *, length_scale: float = GP_LENGTH_SCALE,
                    snapshot: str | None = None) -> dict:
    """GP landscape fit to every measured point of a ZoMBI-Hop run directory.

    The run-directory counterpart of :func:`fullgp_objective` for campaigns that
    live in ``runs/run_*`` (e.g. the 6-d hardware run_69af) rather than a DB. The
    surface is the same fixed-length-scale Matern GP; its reference optima come from
    :func:`ascend_peaks`. Composition columns stay in the run's own dimension order.
    The run's hardware bounds are NOT carried over: off the measured region the GP
    relaxes to the mean of Y, which is what a simulated search there will see.
    """
    run_dir = Path(run_dir).resolve()
    predict, X, Y, snapshot = _fit_run_gp(run_dir, length_scale, snapshot)
    peaks, peak_vals = ascend_peaks(predict, X)

    def fn(x: np.ndarray) -> float:
        return float(predict(np.asarray(x, dtype=float).reshape(1, -1))[0])

    return {
        "dim": int(X.shape[1]),
        "fn": fn,
        "predict": predict,
        "peaks": peaks,
        "peak_values": peak_vals,
        "n_points": int(X.shape[0]),
        "run_dir": str(run_dir),
        "snapshot": snapshot,
        "length_scale": float(length_scale),
        "y_max": float(Y.max()),
    }


def runrf_objective(run_dir: str | Path, *, n_estimators: int = RF_N_ESTIMATORS,
                    snapshot: str | None = None) -> dict:
    """:func:`rungp_objective` with a random forest in place of the GP.

    Off the measured region a forest does not relax to the mean the way the GP does:
    it extends the nearest leaves' values, so the unmeasured simplex inherits the
    edge of the data. Peaks come from :func:`rf_peaks`.
    """
    run_dir = Path(run_dir).resolve()
    predict, X, Y, snapshot = _fit_run_rf(run_dir, n_estimators, snapshot)
    peaks, peak_vals = rf_peaks(predict, X)

    def fn(x: np.ndarray) -> float:
        return float(predict(np.asarray(x, dtype=float).reshape(1, -1))[0])

    return {
        "dim": int(X.shape[1]),
        "fn": fn,
        "predict": predict,
        "peaks": peaks,
        "peak_values": peak_vals,
        "n_points": int(X.shape[0]),
        "run_dir": str(run_dir),
        "snapshot": snapshot,
        "n_estimators": int(n_estimators),
        "y_max": float(Y.max()),
    }


# ── figures ──────────────────────────────────────────────────────────────────

_SQRT3_2 = np.sqrt(3) / 2
_LINE_CMAP = "turbo"


def _line_colors(n: int):
    import matplotlib.pyplot as plt

    return plt.get_cmap(_LINE_CMAP)(np.linspace(0.0, 1.0, n))


def _draw_triangle(ax, labels) -> None:
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.12, _SQRT3_2 + 0.16)
    ax.text(-0.03, -0.03, labels[0], ha="right", va="top", fontsize=10)
    ax.text(1.03, -0.03, labels[1], ha="left", va="top", fontsize=10)
    ax.text(0.5, _SQRT3_2 + 0.04, labels[2], ha="center", va="bottom", fontsize=10)


def figure_3d(L: dict, grid, z, peaks, out_png, subtitle, plt) -> None:
    """Ternary: GP contour, the warm-start lines, and one GP peak per line."""
    xy = comp_to_xy(grid)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    tc = ax.tricontourf(xy[:, 0], xy[:, 1], z, levels=60, cmap="viridis")
    _draw_triangle(ax, CAMPAIGNS[3]["columns"])
    fig.colorbar(tc, ax=ax, shrink=0.78, label="GP objective (warm-start fit)")

    # The measured warm-start points, coloured by which line they came from, so
    # the landscape can be read against the data that actually constrains it.
    colors = _line_colors(len(L["picked_lines"]))
    for c, lid in zip(colors, L["picked_lines"]):
        pxy = comp_to_xy(L["X_warm"][L["warm_line_id"] == lid])
        ax.scatter(pxy[:, 0], pxy[:, 1], s=16, color=c, edgecolors="k",
                   linewidths=0.25, zorder=3, label=f"line {lid}")

    # Auto-detected GP peaks, wherever the surface is highest — not tied to lines.
    if len(peaks):
        qxy = comp_to_xy(peaks)
        ax.scatter(qxy[:, 0], qxy[:, 1], s=170, marker="*", c="#e01e37",
                   edgecolors="k", linewidths=0.7, zorder=5,
                   label=f"GP peaks (n={len(peaks)})")

    ax.legend(loc="upper left", fontsize=7, frameon=False,
              bbox_to_anchor=(-0.10, 1.02))
    ax.set_title(subtitle, fontsize=9)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_4d(L: dict, grid, z, peaks, out_png, subtitle, plt) -> None:
    """Tetrahedron: translucent GP volume (top quartile), warm lines, one peak/line."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    fig = plt.figure(figsize=(8.6, 7.8))
    ax = fig.add_subplot(111, projection="3d")

    for i, j in TETRA_EDGES:
        ax.plot(*zip(TETRA_VERTS[i], TETRA_VERTS[j]), color="black", lw=1.1)
    centroid = TETRA_VERTS.mean(axis=0)
    for v, name in zip(TETRA_VERTS, CAMPAIGNS[4]["columns"]):
        p = v + 0.12 * (v - centroid)
        ax.text(p[0], p[1], p[2], name, ha="center", va="center", fontsize=10)

    # Only the top quartile is drawn: a full cloud of low values reads as fog and
    # hides the basins the optima are supposed to sit in.
    gxyz = comp_to_xyz(grid)
    keep = z >= np.percentile(z, 75)
    mappable = ax.scatter(gxyz[keep, 0], gxyz[keep, 1], gxyz[keep, 2], c=z[keep],
                          cmap="viridis", s=8, alpha=0.20, linewidths=0,
                          depthshade=False)
    fig.colorbar(mappable, ax=ax, label="GP objective (warm-start fit)",
                 fraction=0.03, pad=0.02)

    colors = _line_colors(len(L["picked_lines"]))
    for c, lid in zip(colors, L["picked_lines"]):
        pxyz = comp_to_xyz(L["X_warm"][L["warm_line_id"] == lid])
        ax.scatter(pxyz[:, 0], pxyz[:, 1], pxyz[:, 2], s=12, color=[c],
                   edgecolors="k", linewidths=0.25, depthshade=False,
                   label=f"line {lid}")

    if len(peaks):
        qxyz = comp_to_xyz(peaks)
        ax.scatter(qxyz[:, 0], qxyz[:, 1], qxyz[:, 2], s=260, marker="*",
                   c="#e01e37", edgecolors="k", linewidths=0.7,
                   depthshade=False, label=f"GP peaks (n={len(peaks)})")

    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.set_title(subtitle, fontsize=9)
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=18, azim=35)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── driver ───────────────────────────────────────────────────────────────────

def render(dim: int, out_dir: Path, *, seed: int = SEED) -> dict:
    """Build the warm landscape at ``dim``, auto-detect its peaks, draw the figure."""
    import matplotlib.pyplot as plt

    L = build_warm_landscape(dim, seed=seed)
    predict = L["predict"]

    grid = simplex_grid(GRID_3D if dim == 3 else GRID_4D, dim)
    z = predict(grid)
    peaks, peak_values = detect_peaks(grid, z)

    cov = coverage_stats(L["X_warm"])
    spec = CAMPAIGNS[dim]
    best = float(peak_values.max()) if len(peak_values) else float("nan")
    subtitle = (
        f"Warm-start GP landscape, d={dim}   {Path(spec['db']).name}   "
        f"{L['n_warm_lines']} of ~{spec['n_campaign_lines']} campaign lines "
        f"({len(L['X_warm'])} scored pts)\n"
        f"Matern nu=2.5, length_scale={GP_LENGTH_SCALE} (fixed), fit to the warm "
        f"start ONLY   |   warm-start coverage: mean={cov['mean']:.3f} "
        f"max={cov['max']:.3f}\n"
        f"stars = auto-detected GP peaks ({len(peaks)}), wherever the surface is "
        f"highest   |   best GP objective = {best:.3f}"
    )

    out_png = out_dir / f"warm_gp_landscape_{dim}d.png"
    fig = figure_3d if dim == 3 else figure_4d
    fig(L, grid, z, peaks, out_png, subtitle, plt)
    print(f"Saved figure -> {out_png}")

    return {
        "dim": dim,
        "db": spec["db"],
        "columns": list(spec["columns"]),
        "n_campaign_lines": int(np.unique(L["line_id_all"]).size),
        "n_warm_lines": L["n_warm_lines"],
        "picked_lines": [int(i) for i in L["picked_lines"]],
        "n_warm_points": int(len(L["X_warm"])),
        "n_campaign_points": int(len(L["X_all"])),
        "gp_length_scale": GP_LENGTH_SCALE,
        "warm_coverage": cov,
        "n_peaks": int(len(peaks)),
        "peaks": [
            {"composition": [float(v) for v in x], "gp_value": float(v_)}
            for x, v_ in zip(peaks, peak_values)
        ],
        "best_gp_value": best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Warm-start GP landscape and its greedy optima, per dimension.")
    parser.add_argument("--dims", default="3,4",
                        help="Comma-separated dimensions to render (default: 3,4).")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Seed for line selection and find_optima (default: {SEED}).")
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help=f"Output directory (default: {OUT_DIR}).")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    print(f"{'dim':>4} {'warm lines':>11} {'warm pts':>9} "
          f"{'peaks':>7} {'best GP value':>14}")
    print("-" * 52)
    for token in args.dims.split(","):
        dim = int(token.strip())
        if dim not in CAMPAIGNS:
            raise SystemExit(f"No campaign for d={dim} (have {sorted(CAMPAIGNS)})")
        s = render(dim, out_dir, seed=args.seed)
        summaries.append(s)
        print(f"{dim:>4d} {s['n_warm_lines']:>11d} {s['n_warm_points']:>9d} "
              f"{s['n_peaks']:>7d} {s['best_gp_value']:>14.4f}")

    out_json = out_dir / "warm_gp_landscape_summary.json"
    out_json.write_text(json.dumps(summaries, indent=2))
    print(f"\nSaved summary -> {out_json}")


if __name__ == "__main__":
    main()
