"""``oer6d`` -- the Olympus OER plate datasets as a 6-component simplex objective.

Source
------
Four combinatorial electrodeposition plates released with Olympus
(github.com/aspuru-guzik-group/olympus), each a headerless ``data.csv`` of six
composition columns plus ``overpotential``, with the column names carried in a
sibling ``config.json``:

    src/olympus/datasets/dataset_oer_plate_<ID>/{data.csv,config.json}
    ID in {3496, 3851, 3860, 4098}

Olympus itself is never installed -- it pins ``tensorflow==1.15`` and will not
build. :func:`fetch` pulls the eight raw files straight from GitHub into the
gitignored ``data/oer/``. Nothing here is committed; ``--fetch`` regenerates it.

The overpotential is MINIMISED in the original work (``config.json`` says
``"default_goal": "minimize"``). zhbench maximises, so the objective returns
``-overpotential`` in volts. ``meta["sign"]`` records that.

Four plates, four chemistries -- they cannot be pooled
------------------------------------------------------
The first thing recon has to get right, because it is easy to get wrong: all four
plates share a byte-identical composition lattice (same 2121 rows, same order),
which makes them look like replicates. They are not. Their element sets differ::

    3496   Ni Fe Co Mn Ce La
    3851   Ni Fe Co Ta Mn Cu
    3860   Sn Fe Co Ta Mn Cu
    4098   Sn Sb Co Ca Ni Mn

Only Co and Mn are common to all four, and they sit in different columns. Stacking
the plates into one 6-D table would silently claim that column 0 is "Ni or Sn" and
column 4 is "Ce or Mn or Ni". ``oer6d`` is therefore ONE plate (3496 by default);
the other three are separate objectives via ``plate=``. :func:`load_plate` re-reads
the element names from the fetched ``config.json`` on every load and raises if they
have drifted from :data:`EXPECTED_COMPONENTS`, so a silent upstream reshuffle
cannot turn into a silently wrong benchmark.

What the plate actually measures -- and the hole in the middle
--------------------------------------------------------------
Each plate is the COMPLETE set of compositions on the 0.1 lattice with at most
four nonzero components, and nothing else::

    6     unary        (one component at 1.0)
    135   binary       C(6,2) x 9
    720   ternary      C(6,3) x 36
    1260  quaternary   C(6,4) x 84
    ----
    2121  rows, every one summing to exactly 1.0, no duplicates

There are ZERO quinary and ZERO senary rows. That is a stronger statement than the
team's standing objection to this dataset ("quinary/senary interactions are all
theoretical"): those interactions are not theoretical, they are *absent*. The
measured set is the union of the <=3-dimensional faces of the 5-simplex, a
measure-zero subset of the domain ``oer6d`` claims to be an objective over. On a
0.05 lattice 65.7% of grid points have five or six nonzero components, and a
uniform (Dirichlet) draw on the 5-simplex has six nonzero components with
probability 1 -- only 2.6% of such draws land within the benchmark match radius
r = 0.05 of ANY measured row.

So every value ``fn`` returns off the low-order faces is extrapolation, and
:func:`~..metrics.landscape_contrast`, which probes with a Dirichlet, probes
entirely inside the unmeasured region. Read its number with that in mind; the
support-conditioned contrast reported alongside it is the honest one.

The surrogate
-------------
The lattice is discrete, so a continuous surrogate is needed to evaluate anywhere.
:func:`build_surrogate` offers a random forest and a fixed-length-scale
Matern(5/2) GP (the same kernel family ``warm_start.test_greedy_optima_gp`` uses,
so "the GP" means one thing across this repo). The choice is measured, not
asserted -- :func:`cross_validate` runs two protocols and both are reported in
``meta``:

  * ``random_kfold``   5-fold over shuffled rows. Optimistic by construction: the
    design is a full factorial, so a held-out point almost always has lattice
    neighbours in the training set.
  * ``face_extrapolation``  train on the 861 rows with <=3 nonzero components,
    test on the 1260 quaternary rows. This is the question that actually matters,
    because it asks the surrogate to invent an order of mixing it has never seen
    -- the same thing it is asked to do everywhere in the 5-/6-component interior.

Measured on plate 3496 (negated overpotential, y std 0.0396 V):

    model            random_kfold R^2 / RMSE      face_extrap R^2 / RMSE
    rf_400            0.700 / 0.0217              -0.020 / 0.0218
    gp_matern_0.15    0.694 / 0.0219              -0.121 / 0.0229

The random forest wins on both and is ~40x cheaper to evaluate over a 53k-point
lattice, so it is the default. But look at the second column: **on the only split
that tests extrapolation, both surrogates are no better than predicting the
training mean** (R^2 ~ 0). That is the central fact about ``oer6d``. It is not a
defect of the model choice; it is the plate telling us that a quaternary
overpotential is not predictable from unary/binary/ternary data. The interior of
the 5-simplex is worse still, since nothing at all constrains it.

Reference optima
----------------
Same discipline as the rest of the suite (``warm_start.warm_gp_landscape``): scan a
simplex lattice for local maxima of the surrogate, drop anything that fails to rise
``prominence_frac`` of the way from the background (median surrogate value) to the
global max, and thin by ``min_sep`` so one bump yields one reference point. Two
deviations, both forced by d = 6 and both documented at their definitions:
:func:`detect_peaks` uses exact two-step lattice adjacency instead of a KD-tree
ball of radius ``3 * spacing`` (a 0.05 lattice on the 5-simplex has 53130 points
and that ball holds thousands of them), and ``min_sep`` is ``2r = 0.1`` rather than
0.06, because 0.06 is below one lattice step here and because
``metrics.merge_true_optima`` collapses at ``2r`` anyway -- pre-merging makes the
count reported here the count the metrics actually use.

Then the filter this dataset needs and the campaign surrogates do not: a peak is
kept only if some MEASURED plate row lies within ``r = 0.05`` of it AND that row's
value is within ``value_tol`` of the peak's prominence, mirroring
``metrics.reached_flags``. An unsupported peak is a fluctuation of a model that has
no data there; declaring it a "true optimum" would score methods on the surrogate's
imagination. On plate 3496 this cuts the naive peak count from 129 to 20.

Verdict, recorded in ``meta["verdict"]``
----------------------------------------
Run ``python -m benchmarks.zhbench.datasets.oer --report`` to reproduce every
number in this docstring.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from functools import lru_cache
from itertools import combinations

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(_REPO_ROOT, "data", "oer")

_RAW_URL = ("https://raw.githubusercontent.com/aspuru-guzik-group/olympus/main/"
            "src/olympus/datasets/dataset_oer_plate_{plate}/{fname}")
_FILES = ("config.json", "data.csv")

PLATE_IDS = ("3496", "3851", "3860", "4098")
DEFAULT_PLATE = "3496"

#: Element sets as of the fetch on 2026-08-24. Checked against every fetched
#: ``config.json``, because these differing sets are the entire reason the four
#: plates are four objectives instead of one pooled 8484-row table.
EXPECTED_COMPONENTS: dict[str, tuple[str, ...]] = {
    "3496": ("ni", "fe", "co", "mn", "ce", "la"),
    "3851": ("ni", "fe", "co", "ta", "mn", "cu"),
    "3860": ("sn", "fe", "co", "ta", "mn", "cu"),
    "4098": ("sn", "sb", "co", "ca", "ni", "mn"),
}

DIM = 6
LATTICE_STEP = 0.1          # the plate's own composition quantum
GRID_N = 20                 # reference lattice: 1/20 = 0.05 = the match radius
MATCH_RADIUS = 0.05         # optimize.eval_metrics.MATCH_RADIUS; see _match_radius
PEAK_PROMINENCE_FRAC = 0.12  # warm_gp_landscape._PEAK_PROMINENCE_FRAC
PEAK_VALUE_TOL = 0.25       # metrics.reached_flags default value_tol
GP_LENGTH_SCALE = 0.15
RF_TREES = 400
SEED = 0


# --- fetch -------------------------------------------------------------------

def fetch(plates: tuple[str, ...] = PLATE_IDS, *, dest: str = DATA_DIR,
          force: bool = False, timeout: float = 60.0) -> dict[str, dict[str, str]]:
    """Download each plate's ``data.csv`` and ``config.json`` into ``dest``.

    ``config.json`` is fetched too, not just the data: the CSV is headerless, so
    the config is the ONLY record of which column is which element, and a plate
    whose column order changed upstream would otherwise be unfalsifiable. Files
    already on disk are skipped unless ``force``.
    """
    os.makedirs(dest, exist_ok=True)
    out: dict[str, dict[str, str]] = {}
    for plate in plates:
        out[plate] = {}
        for fname in _FILES:
            path = os.path.join(dest, f"plate_{plate}_{fname}")
            if force or not os.path.exists(path):
                url = _RAW_URL.format(plate=plate, fname=fname)
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    blob = resp.read()
                with open(path, "wb") as fh:
                    fh.write(blob)
                out[plate][fname] = f"downloaded {len(blob)} B"
            else:
                out[plate][fname] = f"cached {os.path.getsize(path)} B"
    return out


def _plate_path(plate: str, fname: str) -> str:
    path = os.path.join(DATA_DIR, f"plate_{plate}_{fname}")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing. Run: python -m benchmarks.zhbench.datasets.oer --fetch")
    return path


def _components_from_config(plate: str) -> tuple[tuple[str, ...], str]:
    """Column order and optimisation sense, read from the plate's own config."""
    with open(_plate_path(plate, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    names = tuple(p["name"].removesuffix("_load") for p in cfg["parameters"])
    meas = tuple(m["name"] for m in cfg["measurements"])
    if meas != ("overpotential",):
        raise ValueError(f"plate {plate}: unexpected measurement columns {meas}")
    return names, str(cfg.get("default_goal", "minimize"))


# --- loading and auditing ----------------------------------------------------

@lru_cache(maxsize=8)
def load_plate(plate: str = DEFAULT_PLATE) -> dict:
    """One plate as ``X`` (2121, 6), ``y`` (negated overpotential), plus its audit.

    ``y = -overpotential`` so larger is better, which is what the whole benchmark
    assumes. Rows are renormalised to sum to exactly 1 (they already do to machine
    precision; this only removes float noise so downstream simplex assertions hold).
    """
    plate = str(plate)
    if plate not in EXPECTED_COMPONENTS:
        raise ValueError(f"unknown OER plate {plate!r}; have {PLATE_IDS}")
    components, goal = _components_from_config(plate)
    if components != EXPECTED_COMPONENTS[plate]:
        raise ValueError(
            f"plate {plate} column order changed upstream: config.json says "
            f"{components}, this module was written against "
            f"{EXPECTED_COMPONENTS[plate]}. The four plates are different "
            f"chemistries -- do not proceed until the mapping is re-checked.")
    if goal != "minimize":
        raise ValueError(f"plate {plate}: config goal is {goal!r}, expected 'minimize'")

    A = np.loadtxt(_plate_path(plate, "data.csv"), delimiter=",", dtype=float)
    if A.shape[1] != DIM + 1:
        raise ValueError(f"plate {plate}: expected {DIM + 1} columns, got {A.shape[1]}")
    X, over = A[:, :DIM], A[:, DIM]
    ok = np.isfinite(over) & np.isfinite(X).all(axis=1)
    X, over = X[ok], over[ok]
    X = X / X.sum(axis=1, keepdims=True)
    return {
        "plate": plate,
        "components": components,
        "X": X,
        "overpotential": over,
        "y": -over,                     # maximisation
        "audit": composition_audit(X, over),
    }


def composition_audit(X: np.ndarray, over: np.ndarray) -> dict:
    """What is actually on the plate, in the terms the team argued about.

    ``frac_quinary_senary`` is the direct answer to "the 5- and 6-component
    interactions are all theoretical": it is 0.0 on every plate, i.e. worse than
    theoretical -- unmeasured. ``frac_domain_unmeasured_order`` says how much of a
    0.05 reference lattice on the 5-simplex sits in that unmeasured region.
    """
    nnz = (X > 1e-9).sum(axis=1)
    counts = {int(k): int((nnz == k).sum()) for k in range(1, DIM + 1)}
    G = simplex_lattice(GRID_N, DIM)
    g_nnz = (G > 1e-9).sum(axis=1)
    return {
        "n_rows": int(X.shape[0]),
        "n_unique_compositions": int(np.unique(np.round(X, 9), axis=0).shape[0]),
        "row_sum_min": float(X.sum(1).min()),
        "row_sum_max": float(X.sum(1).max()),
        "levels_per_component": [int(np.unique(np.round(X[:, j], 6)).size)
                                 for j in range(DIM)],
        "n_by_nonzero_components": counts,
        "n_quinary_senary": counts.get(5, 0) + counts.get(6, 0),
        "frac_quinary_senary": float((nnz >= 5).mean()),
        "max_nonzero_components": int(nnz.max()),
        "frac_lattice_with_5plus_components": float((g_nnz >= 5).mean()),
        "overpotential_v": {"min": float(over.min()), "median": float(np.median(over)),
                            "max": float(over.max()), "std": float(over.std())},
    }


# --- simplex lattice ---------------------------------------------------------

@lru_cache(maxsize=4)
def _lattice_int(n: int, d: int) -> np.ndarray:
    """Integer compositions of ``n`` into ``d`` parts -- the lattice, unscaled."""
    rows = []
    for cut in combinations(range(n + d - 1), d - 1):
        prev, parts = -1, []
        for c in cut:
            parts.append(c - prev - 1)
            prev = c
        parts.append(n + d - 2 - prev)
        rows.append(parts)
    return np.asarray(rows, dtype=np.int64)


def simplex_lattice(n: int, d: int = DIM) -> np.ndarray:
    """Uniform ``1/n`` lattice on the (d-1)-simplex, ``C(n+d-1, d-1)`` points.

    ``visualization.plot_run.simplex_grid`` only covers d in {3, 4}; this is the
    same object for arbitrary d. At d = 6, n = 20 gives 53130 points at spacing
    0.05 -- deliberately the match radius, since peaks closer together than that
    are indistinguishable to every metric in the suite anyway.
    """
    return _lattice_int(int(n), int(d)).astype(float) / float(n)


@lru_cache(maxsize=4)
def _lattice_neighbours(n: int, d: int) -> np.ndarray:
    """``(N, d*(d-1))`` index table of one-unit moves on the lattice, -1 if invalid.

    A single move takes one unit from component ``i`` and gives it to ``j``, which
    is the shortest step that stays on the simplex. Lattice points are looked up by
    a mixed-radix code so this is a vectorised ``searchsorted``, not a dict walk.
    """
    K = _lattice_int(n, d)
    base = np.asarray([(n + 1) ** c for c in range(d)], dtype=np.int64)
    code = K @ base
    order = np.argsort(code)
    scode = code[order]

    cols = []
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            moved = code - base[i] + base[j]
            pos = np.searchsorted(scode, moved)
            pos = np.clip(pos, 0, scode.size - 1)
            idx = order[pos]
            bad = (K[:, i] < 1) | (scode[pos] != moved)
            cols.append(np.where(bad, -1, idx))
    return np.stack(cols, axis=1)


def _dilate(z: np.ndarray, nbr: np.ndarray) -> np.ndarray:
    """Max of ``z`` over each point and its one-move lattice neighbourhood."""
    out = z.copy()
    for c in range(nbr.shape[1]):
        idx = nbr[:, c]
        ok = idx >= 0
        out[ok] = np.maximum(out[ok], z[idx[ok]])
    return out


# --- surrogate ---------------------------------------------------------------

def build_surrogate(X: np.ndarray, y: np.ndarray, kind: str = "rf",
                    *, seed: int = SEED, n_jobs: int = 1):
    """Fit a continuous surrogate over the discrete plate and return ``predict``.

    ``rf`` is a random forest; ``gp`` is Matern(nu=2.5) at a fixed length scale,
    matching ``warm_start.test_greedy_optima_gp.build_gp_landscape`` so the two
    families of objective in this suite share a kernel. Hyperparameters are FIXED
    rather than marginal-likelihood-optimised: with 2121 points on a full factorial
    the likelihood happily drives the length scale up until the surface is a plane,
    and an objective whose peak structure depends on an optimiser restart is not a
    stable benchmark. :func:`cross_validate` is what justifies the defaults.

    ``n_jobs=1`` for the forest on purpose -- the benchmark calls ``fn`` one point
    at a time and thread-pool startup dominates a 400-tree single-row predict.
    """
    if kind == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=RF_TREES, random_state=seed,
                                      n_jobs=n_jobs)
        model.fit(X, y)
        return lambda Q: model.predict(np.atleast_2d(np.asarray(Q, dtype=float)))
    if kind == "gp":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import (ConstantKernel, Matern,
                                                      WhiteKernel)
        kernel = (ConstantKernel(1.0, "fixed")
                  * Matern(length_scale=GP_LENGTH_SCALE,
                           length_scale_bounds="fixed", nu=2.5)
                  + WhiteKernel(noise_level=1e-2, noise_level_bounds="fixed"))
        mu, sd = float(y.mean()), float(y.std()) or 1.0
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False,
                                      optimizer=None, random_state=seed)
        gp.fit(X, (y - mu) / sd)

        def predict(Q):
            Q = np.atleast_2d(np.asarray(Q, dtype=float))
            return gp.predict(Q) * sd + mu
        return predict
    raise ValueError(f"unknown surrogate kind {kind!r}")


def _r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    ss = float(((y_true - y_true.mean()) ** 2).sum())
    resid = y_true - y_pred
    r2 = 1.0 - float((resid ** 2).sum()) / ss if ss > 0 else float("nan")
    return r2, float(np.sqrt((resid ** 2).mean()))


def cross_validate(plate: str = DEFAULT_PLATE, kinds: tuple[str, ...] = ("rf", "gp"),
                   *, n_folds: int = 5, seed: int = SEED) -> dict:
    """Held-out accuracy under two protocols; the second is the one that matters.

    ``random_kfold`` shuffles rows. On a full factorial design this leaks: a
    held-out quaternary almost always has a lattice neighbour in training, so it
    measures interpolation between adjacent measured points and nothing else.

    ``face_extrapolation`` trains on every row with <= 3 nonzero components and
    tests on the quaternaries. It is the closest available proxy for what the
    objective actually asks the surrogate to do -- predict a higher-order mixture
    from lower-order ones -- because the plate contains no 5- or 6-component rows
    to test against directly. If a model cannot make this jump, its values in the
    interior of the 5-simplex are decoration.
    """
    D = load_plate(plate)
    X, y = D["X"], D["y"]
    nnz = (X > 1e-9).sum(axis=1)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(X.shape[0]) % n_folds
    tr_lo, te_hi = np.where(nnz <= 3)[0], np.where(nnz == 4)[0]

    out: dict = {"plate": plate, "n_rows": int(X.shape[0]),
                 "y_std": float(y.std()), "y_units": "V (negated overpotential)",
                 "n_train_face_extrap": int(tr_lo.size),
                 "n_test_face_extrap": int(te_hi.size), "models": {}}
    for kind in kinds:
        t0 = time.perf_counter()
        pred = np.empty_like(y)
        for k in range(n_folds):
            te = fold == k
            f = build_surrogate(X[~te], y[~te], kind, seed=seed)
            pred[te] = f(X[te])
        r2_k, rmse_k = _r2_rmse(y, pred)
        f = build_surrogate(X[tr_lo], y[tr_lo], kind, seed=seed)
        r2_e, rmse_e = _r2_rmse(y[te_hi], f(X[te_hi]))
        out["models"][kind] = {
            "random_kfold_r2": r2_k, "random_kfold_rmse": rmse_k,
            "face_extrapolation_r2": r2_e, "face_extrapolation_rmse": rmse_e,
            "fit_seconds": float(time.perf_counter() - t0) / (n_folds + 1),
        }
    return out


# --- reference optima --------------------------------------------------------

def detect_peaks(grid: np.ndarray, z: np.ndarray, nbr: np.ndarray, *,
                 min_sep: float = 2 * MATCH_RADIUS,
                 prominence_frac: float = PEAK_PROMINENCE_FRAC
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of a surface over a simplex lattice -- the naive detector.

    Mirrors ``warm_start.warm_gp_landscape.detect_peaks``: local maxima, a
    prominence floor at ``prominence_frac`` of the way from the median to the
    global max, then descending-value thinning at ``min_sep``. Two changes that d=6
    forces:

      * "local" means no lattice point within TWO one-unit moves is higher,
        evaluated by dilating ``z`` twice over the move table. The KD-tree ball of
        radius ``3 * spacing`` the 3-D/4-D version uses would hold thousands of the
        53130 lattice points here; two moves is the same intent (the immediate
        ring plus one) at a cost that does not blow up with d.
      * ``min_sep`` defaults to ``2r = 0.1``, not 0.06. One lattice step is already
        0.0707, so 0.06 would be inert, and ``metrics.merge_true_optima`` collapses
        anything closer than ``2r`` regardless -- pre-merging makes the count
        reported here the count the metrics will actually score against.
    """
    is_max = z >= _dilate(_dilate(z, nbr), nbr)
    background = float(np.median(z))
    floor = background + prominence_frac * (float(z.max()) - background)
    cand = np.where(is_max & (z >= floor))[0]
    cand = cand[np.argsort(z[cand])[::-1]]

    kept: list[int] = []
    for idx in cand:
        p = grid[idx]
        if all(np.linalg.norm(p - grid[k]) >= min_sep for k in kept):
            kept.append(int(idx))
    sel = np.asarray(kept, dtype=int)
    return grid[sel], z[sel]


def support_mask(peaks: np.ndarray, values: np.ndarray, X: np.ndarray,
                 y: np.ndarray, background: float, *, radius: float = MATCH_RADIUS,
                 value_tol: float = PEAK_VALUE_TOL) -> np.ndarray:
    """Which peaks a real measurement actually vouches for.

    A peak survives when some measured row sits within ``radius`` of it AND that
    row's measured value clears ``v - value_tol * (v - background)`` -- exactly the
    near-AND-high test ``metrics.reached_flags`` applies to samples, turned around
    and applied to the reference set itself.

    This filter exists because of the hole documented at the top of this module.
    Two thirds of the reference lattice has no measurement anywhere near it, so the
    naive detector's peaks out there are shaped by the surrogate's inductive bias
    and nothing else. Scoring a method's recall against those would be scoring it
    on a random forest's extrapolation habits.
    """
    from scipy.spatial import cKDTree

    if peaks.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    tree = cKDTree(X)
    near = tree.query_ball_point(peaks, r=radius)
    keep = np.zeros(peaks.shape[0], dtype=bool)
    for i, idx in enumerate(near):
        if not idx:
            continue
        thresh = values[i] - value_tol * max(values[i] - background, 0.0)
        keep[i] = bool(y[np.asarray(idx, dtype=int)].max() >= thresh)
    return keep


def _contrast(vals_peaks: np.ndarray, vals_probe: np.ndarray) -> dict:
    p99 = float(np.percentile(vals_probe, 99))
    span = float(vals_probe.max() - np.median(vals_probe)) or 1.0
    return {
        "probe_median": float(np.median(vals_probe)),
        "probe_p99": p99,
        "probe_max": float(vals_probe.max()),
        "frac_peaks_above_probe_p99": (float((vals_peaks > p99).mean())
                                       if vals_peaks.size else float("nan")),
        "mean_peak_prominence": (float(((vals_peaks - np.median(vals_probe)) / span).mean())
                                 if vals_peaks.size else float("nan")),
    }


def _reference_path(plate: str, kind: str) -> str:
    return os.path.join(DATA_DIR, f"oer6d_plate{plate}_{kind}_reference.json")


def reference_set(plate: str = DEFAULT_PLATE, kind: str = "rf", *,
                  grid_n: int = GRID_N, seed: int = SEED, rebuild: bool = False,
                  n_probe: int = 4000) -> dict:
    """Peaks, support filter, contrast and CV for one plate -- cached to ``data/``.

    The cache holds only derived numbers (a few hundred floats), so it is cheap to
    delete and regenerate; the surrogate itself is refit on every load because
    pickling a fitted sklearn model across versions is a worse failure mode than a
    two-second refit.
    """
    path = _reference_path(plate, kind)
    if os.path.exists(path) and not rebuild:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    D = load_plate(plate)
    X, y = D["X"], D["y"]
    predict = build_surrogate(X, y, kind, seed=seed)

    grid = simplex_lattice(grid_n, DIM)
    z = np.concatenate([predict(grid[i:i + 8192])
                        for i in range(0, grid.shape[0], 8192)])
    nbr = _lattice_neighbours(grid_n, DIM)
    peaks, vals = detect_peaks(grid, z, nbr)
    background = float(np.median(z))
    keep = support_mask(peaks, vals, X, y, background)

    rng = np.random.default_rng(seed)
    probe_u = rng.dirichlet(np.ones(DIM), size=n_probe)
    v_u = np.concatenate([predict(probe_u[i:i + 8192])
                          for i in range(0, n_probe, 8192)])
    # Second probe distribution: measured rows plus a small jitter, i.e. the part
    # of the domain the surrogate is entitled to speak about. The Dirichlet probe
    # never leaves the unmeasured interior, so on its own it flatters the peaks.
    ridx = rng.integers(0, X.shape[0], size=n_probe)
    probe_s = X[ridx] + rng.normal(0.0, 0.01, size=(n_probe, DIM))
    probe_s = np.clip(probe_s, 0.0, None)
    probe_s /= probe_s.sum(axis=1, keepdims=True)
    v_s = np.concatenate([predict(probe_s[i:i + 8192])
                          for i in range(0, n_probe, 8192)])

    from scipy.spatial import cKDTree
    d_u = cKDTree(X).query(probe_u)[0]

    ref = {
        "plate": plate, "components": list(D["components"]), "surrogate": kind,
        "grid_n": int(grid_n), "n_grid": int(grid.shape[0]), "seed": int(seed),
        "background": background,
        "surrogate_min": float(z.min()), "surrogate_max": float(z.max()),
        "n_peaks_naive": int(peaks.shape[0]),
        "n_peaks_supported": int(keep.sum()),
        "peaks": peaks[keep].tolist(),
        "peak_values": vals[keep].tolist(),
        "peaks_naive": peaks.tolist(),
        "peak_values_naive": vals.tolist(),
        "contrast_uniform_simplex": _contrast(vals[keep], v_u),
        "contrast_uniform_simplex_naive_peaks": _contrast(vals, v_u),
        "contrast_measured_support": _contrast(vals[keep], v_s),
        "frac_uniform_probe_within_r_of_measurement": float((d_u <= MATCH_RADIUS).mean()),
        "median_uniform_probe_dist_to_measurement": float(np.median(d_u)),
        "audit": D["audit"],
        "cv": cross_validate(plate, seed=seed),
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ref, fh, indent=2)
    return ref


# --- the objective -----------------------------------------------------------

def oer6d(plate: str = DEFAULT_PLATE, kind: str = "rf", *, seed: int = SEED,
          grid_n: int = GRID_N, supported_only: bool = True,
          rebuild: bool = False) -> dict:
    """Build the ``oer6d`` objective spec.

    Returns the plain kwargs dict ``objectives.Objective(**spec)`` takes -- ``name,
    dim, fn, true_optima, true_values, maximize, domain, meta`` -- rather than an
    ``Objective``, so this package never imports the registry that imports it.

    ``supported_only=False`` swaps in the naive peak set. It exists to make the
    difference measurable, not because it is ever the right reference.
    """
    ref = reference_set(plate, kind, grid_n=grid_n, seed=seed, rebuild=rebuild)
    D = load_plate(plate)
    predict = build_surrogate(D["X"], D["y"], kind, seed=seed)

    key = "peaks" if supported_only else "peaks_naive"
    vkey = "peak_values" if supported_only else "peak_values_naive"
    P = np.asarray(ref[key], dtype=float).reshape(-1, DIM)
    V = np.asarray(ref[vkey], dtype=float)

    audit, cv = ref["audit"], ref["cv"]
    best = cv["models"][kind]
    meta = {
        "kind": "oer",
        "source": "Olympus dataset_oer_plate_<id> (aspuru-guzik-group/olympus)",
        "url": _RAW_URL.format(plate=plate, fname="data.csv"),
        "plate": plate,
        "components": ref["components"],
        "sign": "y = -overpotential (V); source goal is MINIMISE, benchmark maximises",
        "measured_rows": audit["n_rows"],
        "measured_lattice_step": LATTICE_STEP,

        "emulation": (f"{kind} surrogate fit to all {audit['n_rows']} measured rows; "
                      "fn(x) is a MODEL value, not a measurement, at every x off the "
                      "0.1 lattice"),
        "surrogate_random_kfold_r2": best["random_kfold_r2"],
        "surrogate_random_kfold_rmse": best["random_kfold_rmse"],
        "surrogate_face_extrapolation_r2": best["face_extrapolation_r2"],
        "surrogate_face_extrapolation_rmse": best["face_extrapolation_rmse"],
        "surrogate_cv_all_models": cv["models"],

        "n_peaks_naive": ref["n_peaks_naive"],
        "n_peaks_supported": ref["n_peaks_supported"],
        "peak_support_rule": (f"a measured row within r={MATCH_RADIUS} whose value "
                              f"clears v - {PEAK_VALUE_TOL}*(v - background)"),
        "reference": "supported surrogate peaks; NOT hardware-validated optima",

        "frac_peaks_above_random_p99": ref["contrast_uniform_simplex"]["frac_peaks_above_probe_p99"],
        "contrast_uniform_simplex": ref["contrast_uniform_simplex"],
        "contrast_measured_support": ref["contrast_measured_support"],

        "n_quinary_senary_rows": audit["n_quinary_senary"],
        "frac_quinary_senary_rows": audit["frac_quinary_senary"],
        "max_nonzero_components_measured": audit["max_nonzero_components"],
        "frac_reference_lattice_with_5plus_components": audit["frac_lattice_with_5plus_components"],
        "frac_uniform_probe_within_r_of_measurement":
            ref["frac_uniform_probe_within_r_of_measurement"],

        "caveats": [
            "The four OER plates share an identical composition lattice but NOT "
            "their element sets (3496 Ni/Fe/Co/Mn/Ce/La, 3851 Ni/Fe/Co/Ta/Mn/Cu, "
            "3860 Sn/Fe/Co/Ta/Mn/Cu, 4098 Sn/Sb/Co/Ca/Ni/Mn). They are four "
            "chemistries, not four replicates, and are never pooled.",
            f"The plate contains {audit['n_quinary_senary']} rows with 5 or 6 "
            "nonzero components. Every measurement lies on a face of the 5-simplex "
            "of dimension <= 3 -- a measure-zero subset of the domain. "
            f"{audit['frac_lattice_with_5plus_components']:.1%} of the reference "
            "lattice, and effectively all uniform Dirichlet draws, lie outside it.",
            f"Only {ref['frac_uniform_probe_within_r_of_measurement']:.1%} of "
            f"uniform simplex draws fall within r={MATCH_RADIUS} of any measured "
            "row, so metrics.landscape_contrast probes almost entirely inside the "
            "unmeasured region; contrast_measured_support is the honest counterpart.",
            f"Face-extrapolation R^2 is {best['face_extrapolation_r2']:.3f}: trained "
            "on <=3-component rows the surrogate predicts quaternaries no better "
            "than the training mean. Its values in the 5-/6-component interior are "
            "unvalidated and unvalidatable from this data.",
            "The lattice is discrete at 0.1 while the benchmark match radius is "
            "0.05, so a 'true optimum' can never be more than one measured point.",
        ],
        "verdict": _VERDICT,
    }
    return {
        "name": f"oer6d_plate{plate}" + ("" if supported_only else "_naivepeaks"),
        "dim": DIM,
        "fn": lambda x, _p=predict: float(_p(np.asarray(x, dtype=float).reshape(1, -1))[0]),
        "true_optima": P,
        "true_values": V,
        "maximize": True,
        "domain": "simplex",
        "meta": meta,
    }


_VERDICT = (
    "Include only as a labelled EXTERNAL-DATA sanity check, never as evidence about "
    "multi-optimum search. The plate measures no composition with more than four "
    "nonzero components, so 66% of the 6-D simplex -- including everywhere a "
    "uniform sampler looks -- carries no data, and the surrogate's face-"
    "extrapolation R^2 of ~0 says it cannot fill that in. The supported peak set is "
    "what remains once that is taken seriously, and it lives entirely on the "
    "low-order faces. The 'too sparse/smooth' objection is directionally right but "
    "misdiagnosed: the surface is not smooth, it is unconstrained."
)


def build(spec: dict) -> dict:
    """``objectives.build``-shaped entry point, if the registry ever grows a hook."""
    spec = dict(spec)
    spec.pop("kind", None)
    return oer6d(**spec)


# --- CLI ---------------------------------------------------------------------

def _report(plate: str, kind: str, rebuild: bool) -> None:
    ref = reference_set(plate, kind, rebuild=rebuild)
    a, cv = ref["audit"], ref["cv"]
    print(f"plate {ref['plate']}  components {'/'.join(ref['components'])}  "
          f"{a['n_rows']} rows, {a['n_unique_compositions']} unique, "
          f"sums [{a['row_sum_min']:.6f}, {a['row_sum_max']:.6f}], "
          f"levels/component {a['levels_per_component']}")
    print(f"  rows by nonzero components: {a['n_by_nonzero_components']}")
    print(f"  quinary+senary rows: {a['n_quinary_senary']} "
          f"({a['frac_quinary_senary']:.4%} of the plate)")
    print(f"  0.05 lattice with >=5 components: "
          f"{a['frac_lattice_with_5plus_components']:.2%} of {ref['n_grid']} points")
    print(f"  uniform draws within r=0.05 of a measurement: "
          f"{ref['frac_uniform_probe_within_r_of_measurement']:.2%} "
          f"(median distance {ref['median_uniform_probe_dist_to_measurement']:.4f})")
    print(f"  overpotential V: {a['overpotential_v']}")
    print(f"  CV (y std {cv['y_std']:.4f} V, face split "
          f"{cv['n_train_face_extrap']} -> {cv['n_test_face_extrap']}):")
    for name, m in cv["models"].items():
        print(f"    {name:4s} random_kfold R2 {m['random_kfold_r2']:+.3f} "
              f"RMSE {m['random_kfold_rmse']:.4f}   |   face_extrap R2 "
              f"{m['face_extrapolation_r2']:+.3f} RMSE {m['face_extrapolation_rmse']:.4f}"
              f"   ({m['fit_seconds']:.2f} s/fit)")
    print(f"  peaks: naive {ref['n_peaks_naive']} -> supported "
          f"{ref['n_peaks_supported']}")
    cu, cs = ref["contrast_uniform_simplex"], ref["contrast_measured_support"]
    print(f"  contrast vs uniform simplex   : "
          f"{cu['frac_peaks_above_probe_p99']:.3f} of supported peaks > p99 "
          f"(p99 {cu['probe_p99']:.4f}, median {cu['probe_median']:.4f})")
    print(f"  contrast vs measured support  : "
          f"{cs['frac_peaks_above_probe_p99']:.3f} of supported peaks > p99 "
          f"(p99 {cs['probe_p99']:.4f}, median {cs['probe_median']:.4f})")
    print(f"  contrast vs uniform, naive pks: "
          f"{ref['contrast_uniform_simplex_naive_peaks']['frac_peaks_above_probe_p99']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Olympus OER plates -> oer6d")
    ap.add_argument("--fetch", action="store_true",
                    help="download data.csv + config.json for all four plates")
    ap.add_argument("--force", action="store_true", help="re-download cached files")
    ap.add_argument("--report", action="store_true",
                    help="audit, CV, peaks and contrast for the selected plate(s)")
    ap.add_argument("--rebuild", action="store_true", help="ignore the cached reference")
    ap.add_argument("--plate", default=DEFAULT_PLATE, choices=("all",) + PLATE_IDS)
    ap.add_argument("--surrogate", default="rf", choices=("rf", "gp"))
    args = ap.parse_args()

    if args.fetch:
        for plate, status in fetch(force=args.force).items():
            print(f"plate {plate}: " + ", ".join(f"{k} {v}" for k, v in status.items()))
    if args.report:
        for plate in (PLATE_IDS if args.plate == "all" else (args.plate,)):
            _report(plate, args.surrogate, args.rebuild)
    if not (args.fetch or args.report):
        ap.print_help()
