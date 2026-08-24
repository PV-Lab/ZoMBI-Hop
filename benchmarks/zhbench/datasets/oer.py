"""``oer6d`` -- the Olympus OER plate datasets as a 6-component simplex objective.

Source
------
Four combinatorial electrodeposition plates released with Olympus
(github.com/aspuru-guzik-group/olympus), each a headerless ``data.csv`` of six
composition columns plus ``overpotential``, with the column names carried in a
sibling ``config.json``::

    src/olympus/datasets/dataset_oer_plate_<ID>/{data.csv,config.json}
    ID in {3496, 3851, 3860, 4098}

Olympus itself is never installed -- it pins ``tensorflow==1.15`` and will not
build. :func:`fetch` pulls the eight raw files straight from GitHub into the
gitignored ``data/oer/``. Nothing is committed; ``--fetch`` regenerates it.

The source minimises overpotential (``config.json``: ``"default_goal": "minimize"``)
and this benchmark maximises, so ``fn`` returns ``-overpotential`` in volts.
``meta["sign"]`` says so.

Four plates, four chemistries -- they cannot be pooled
------------------------------------------------------
Easy to get wrong: all four plates share a byte-identical composition lattice --
the same 2121 rows in the same order -- which makes them look like replicates.
They are not. Their element sets differ::

    3496   Ni Fe Co Mn Ce La
    3851   Ni Fe Co Ta Mn Cu
    3860   Sn Fe Co Ta Mn Cu
    4098   Sn Sb Co Ca Ni Mn

Only Co and Mn appear in all four, in different columns. Stacking the plates into
one 8484-row table would silently assert that column 0 means "Ni or Sn" and column
4 means "Ce or Mn or Ni". ``oer6d`` is therefore ONE plate (3496 by default) and
the other three are separate objectives via ``plate=``. :func:`load_plate` re-reads
the element names from the fetched ``config.json`` every time and raises if they
have drifted from :data:`EXPECTED_COMPONENTS`, so an upstream reshuffle cannot
quietly become a wrong benchmark.

What the plate measures -- and the hole in the middle
-----------------------------------------------------
Each plate is the set of 0.1-lattice compositions with at most four nonzero
components -- verified row by row against ``simplex_lattice(10, 6)``, not assumed.
Plates 3496 and 4098 are complete; 3851 and 3860 are missing one and two rows
respectively (2119 / 2120), presumably failed measurements. For 3496::

    6     unary        (one component at 1.0)
    135   binary       C(6,2) x 9
    720   ternary      C(6,3) x 36
    1260  quaternary   C(6,4) x 84
    ----
    2121  rows, every one summing to 1.0 exactly, none duplicated

There are ZERO quinary and ZERO senary rows. That is stronger than the standing
team objection ("quinary/senary interactions are all theoretical"): they are not
theoretical, they are *absent*. Every measurement lies on a face of the 5-simplex
of dimension <= 3, a measure-zero subset of the domain this objective claims to
cover. The remaining 882 of the 3003 points of the plate's own 0.1 lattice (29.4%)
are unmeasured; refine to a 0.05 lattice and 65.7% of it is in that region; draw
uniformly (Dirichlet) on the 5-simplex and *every* sample has six nonzero
components, with only 2.6% of draws landing within the benchmark match radius
r = 0.05 of any measured row.

Consequence for the suite: ``metrics.landscape_contrast`` probes with a Dirichlet,
so on this objective it probes almost entirely inside the unmeasured region.
``meta["contrast_measured_support"]`` is the honest counterpart -- the same
statistic against probes drawn near actual measurements.

The surrogate, and why a random forest
--------------------------------------
The plate is discrete, so a continuous surrogate is needed. :func:`cross_validate`
measures two protocols and reports both:

  * ``random_kfold`` -- 5-fold over shuffled rows. Optimistic by construction: the
    design is a full factorial, so a held-out point nearly always has lattice
    neighbours in training.
  * ``face_extrapolation`` -- train on the 861 rows with <= 3 nonzero components,
    test on the 1260 quaternaries. This is the question that matters, because it
    asks the surrogate to predict an order of mixing it has never seen -- the same
    thing it is asked to do throughout the 5-/6-component interior.

Measured on plate 3496 (``y = -overpotential``, std 0.0539 V, 2121 rows)::

    model                 random_kfold R^2 / RMSE      face_extrap R^2 / RMSE
    rf   (400 trees)       0.800 / 0.0241               0.700 / 0.0248
    gp   (Matern 5/2)      0.761 / 0.0263               0.640 / 0.0272

The forest wins both, so it is the default. The extrapolation column is better
than expected -- overpotential on this plate really is largely additive, so
quaternaries *are* predictable from lower orders. That is a genuine point in the
dataset's favour and is why this module does not simply reject it. What it does
not license is any claim about five- and six-component points, for which no
analogous test exists on this plate at all.

Reference optima
----------------
Same discipline as ``warm_start.warm_gp_landscape.detect_peaks``: local maxima of
the surrogate over a simplex lattice, a prominence floor at ``prominence_frac`` of
the way from the background (median surrogate value) to the global max, then
descending-value thinning at ``min_sep``.

The reference lattice is the plate's OWN 0.1 lattice, and that choice is measured
rather than asserted (``--report`` prints the sensitivity block). Refining to 0.05
does not find physics, it finds model artefacts: the naive detector goes from 10
peaks -- none of them with five or six nonzero components -- to 240, of which 72
sit in the region where the plate has no data. On the plate's own lattice the
support test below is also exact instead of a rounding question, because a lattice
point either IS a measured composition or is one of the 882 unmeasured
five-/six-component points; on a 0.05 lattice every second grid point sits
0.0707 from the nearest measurement, half a printed step, and would be rejected on
geometry rather than on evidence.

Then the filter this dataset needs and the campaign surrogates do not: a peak is
kept only if some MEASURED row lies within ``r = 0.05`` of it AND that row's value
clears ``v - value_tol * (v - background)`` -- the near-AND-high test
``metrics.reached_flags`` applies to samples, turned around onto the reference set.
An unsupported peak is a fluctuation of a model with no data underneath it, and
scoring recall against those would score methods on a forest's extrapolation
habits. On plate 3496 the forest's 10 naive peaks all survive (they land on
measured rows); the GP's 12 become 10. The filter is not vacuous -- on the 0.05
lattice it cuts 240 to 44 -- it is simply well-behaved at the resolution the data
actually has.

Because they survive, every reference optimum of ``oer6d`` is a real measured
composition with a real measured overpotential. That is more than ``real3d`` /
``real4d`` can say, and it is the strongest argument for including this objective.

Two consequences worth stating before anyone reads a score. Every reference
optimum has two or three zero components, since the plate never measured anything
else -- so they all sit on the simplex boundary, and a method that stays in the
interior scores zero by construction (uniform Dirichlet search at a 1000-sample
budget gets ``peak_ratio`` 0.00). And the basins are wide: on plate 3496 the
measured rows within 0.15 of a reference optimum still average 0.73 of its
prominence. :func:`basin_profile` measures that on the raw rows precisely so it
cannot be blamed on the surrogate's smoothing.

How the four plates compare (``--report --plate all``)::

    plate  rf kfold R^2  rf face R^2  peaks  contrast  basin@0.15
    3496       0.800        0.700       10     0.400      0.73
    3851       0.711        0.525       38     0.237      0.24
    3860       0.917        0.885       17     0.529      0.63
    4098       0.706        0.678       14     0.143      0.24

3496 is the default because it is the largest complete plate with a mid-range
contrast; 3851 and 4098 are sharper landscapes but compressed in range, and 3860
is by far the most predictable. ``contrast`` here is the fraction of reference
optima above the 99th percentile of uniform simplex sampling -- the same statistic
``objectives.py`` quotes for the campaign GPs (0.17 at 3-D, 0.46 at 4-D).

Cost
----
``fn`` is one 400-tree forest prediction, ~12 ms, dominated by per-call overhead
rather than by the trees; a 1000-sample budget costs ~12 s and a 4000-point
``landscape_contrast`` sweep ~50 s. Batched calls run at ~10 us/point.

Reproduce every number above with::

    python -m benchmarks.zhbench.datasets.oer --fetch
    python -m benchmarks.zhbench.datasets.oer --report --plate all
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

#: Element sets as fetched on 2026-08-24, checked against every ``config.json`` on
#: load. These differing sets are the whole reason the four plates are four
#: objectives rather than one pooled table, so drift here must be loud.
EXPECTED_COMPONENTS: dict[str, tuple[str, ...]] = {
    "3496": ("ni", "fe", "co", "mn", "ce", "la"),
    "3851": ("ni", "fe", "co", "ta", "mn", "cu"),
    "3860": ("sn", "fe", "co", "ta", "mn", "cu"),
    "4098": ("sn", "sb", "co", "ca", "ni", "mn"),
}

DIM = 6
LATTICE_STEP = 0.1           # the plate's own composition quantum
GRID_N = 10                  # reference lattice == the plate's lattice
FINE_GRID_N = 20             # sensitivity only: 0.05 spacing, 53130 points
PEAK_MOVES = 1               # locality of the local-maximum test, in lattice moves
MATCH_RADIUS = 0.05          # optimize.eval_metrics.MATCH_RADIUS
PEAK_PROMINENCE_FRAC = 0.12  # warm_gp_landscape._PEAK_PROMINENCE_FRAC
PEAK_VALUE_TOL = 0.25        # metrics.reached_flags default value_tol
GP_LENGTH_SCALE = 0.15
RF_TREES = 400
SEED = 0


# --- fetch -------------------------------------------------------------------

def fetch(plates: tuple[str, ...] = PLATE_IDS, *, dest: str = DATA_DIR,
          force: bool = False, timeout: float = 60.0) -> dict[str, dict[str, str]]:
    """Download each plate's ``data.csv`` and ``config.json`` into ``dest``.

    ``config.json`` is fetched too, not just the data: the CSV is headerless, so
    the config is the only record of which column is which element, and a plate
    whose column order changed upstream would otherwise be unfalsifiable. Files
    already present are skipped unless ``force``.
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
    """One plate as ``X`` (2121, 6) and ``y = -overpotential``, plus its audit.

    Rows are renormalised to sum to exactly 1 -- they already do to machine
    precision, so this only removes float noise and lets downstream simplex
    assertions be exact.
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
    if A.ndim != 2 or A.shape[1] != DIM + 1:
        raise ValueError(f"plate {plate}: expected {DIM + 1} columns, got {A.shape}")
    X, over = A[:, :DIM], A[:, DIM]
    ok = np.isfinite(over) & np.isfinite(X).all(axis=1)
    X, over = X[ok], over[ok]
    X = X / X.sum(axis=1, keepdims=True)
    return {
        "plate": plate,
        "components": components,
        "X": X,
        "overpotential": over,
        "y": -over,
        "audit": composition_audit(X, over),
    }


def composition_audit(X: np.ndarray, over: np.ndarray) -> dict:
    """What is actually on the plate, in the terms the team argued about.

    ``frac_quinary_senary`` answers "the 5- and 6-component interactions are all
    theoretical" directly: it is 0.0 on every plate, i.e. worse than theoretical --
    unmeasured. ``frac_plate_lattice_unmeasured`` is the coverage number that
    follows, counted against the plate's OWN lattice rather than assumed from the
    nonzero-count histogram, so the one or two rows plates 3851/3860 are missing
    show up as gaps instead of being rounded away.
    ``covers_all_low_order`` records whether the plate is the complete <= 4-nonzero
    part of that lattice; it is False for exactly those two plates.
    """
    nnz = (X > 1e-9).sum(axis=1)
    counts = [int((nnz == k).sum()) for k in range(1, DIM + 1)]
    coarse = simplex_lattice(GRID_N, DIM)
    c_nnz = (coarse > 1e-9).sum(axis=1)
    fine = simplex_lattice(FINE_GRID_N, DIM)
    f_nnz = (fine > 1e-9).sum(axis=1)
    measured = {tuple(v) for v in np.round(X, 6)}
    on_lattice = np.asarray([tuple(v) in measured for v in np.round(coarse, 6)])
    return {
        "n_rows": int(X.shape[0]),
        "n_unique_compositions": int(np.unique(np.round(X, 9), axis=0).shape[0]),
        "row_sum_min": float(X.sum(1).min()),
        "row_sum_max": float(X.sum(1).max()),
        "levels_per_component": [int(np.unique(np.round(X[:, j], 6)).size)
                                 for j in range(DIM)],
        "n_by_nonzero_components": counts,      # index k-1 == k nonzero components
        "n_quinary_senary": counts[4] + counts[5],
        "frac_quinary_senary": float((nnz >= 5).mean()),
        "max_nonzero_components": int(nnz.max()),
        "all_rows_on_plate_lattice": bool(int(on_lattice.sum()) == X.shape[0]),
        "covers_all_low_order": bool(on_lattice[c_nnz <= 4].all()),
        "n_plate_lattice_points": int(coarse.shape[0]),
        "n_plate_lattice_unmeasured": int((~on_lattice).sum()),
        "frac_plate_lattice_unmeasured": float((~on_lattice).mean()),
        "frac_fine_lattice_with_5plus_components": float((f_nnz >= 5).mean()),
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
    same object for arbitrary d. At d = 6, ``n = 10`` reproduces the plate's own
    lattice (3003 points) and ``n = 20`` halves the spacing (53130 points).
    """
    return _lattice_int(int(n), int(d)).astype(float) / float(n)


@lru_cache(maxsize=4)
def _lattice_neighbours(n: int, d: int) -> np.ndarray:
    """``(N, d*(d-1))`` index table of one-unit moves on the lattice, -1 if invalid.

    A move takes one unit from component ``i`` and gives it to ``j`` -- the
    shortest step that stays on the simplex, and exactly the set of lattice points
    at one spacing (checked against a KD-tree ball in the tests). Points are looked
    up through a mixed-radix code so this is a vectorised ``searchsorted`` rather
    than a per-point dict walk; at ``n = 20`` that is the difference between two
    seconds and two minutes.
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
            pos = np.clip(np.searchsorted(scode, moved), 0, scode.size - 1)
            bad = (K[:, i] < 1) | (scode[pos] != moved)
            cols.append(np.where(bad, -1, order[pos]))
    return np.stack(cols, axis=1)


def _dilate(z: np.ndarray, nbr: np.ndarray) -> np.ndarray:
    """Max of ``z`` over each lattice point and its one-move neighbourhood."""
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
    matching ``warm_start.test_greedy_optima_gp.build_gp_landscape`` so "the GP"
    means one thing across this repo. Hyperparameters are FIXED rather than
    marginal-likelihood-optimised: on a full factorial the likelihood happily
    pushes the length scale up until the surface is nearly a plane, and an
    objective whose peak structure depends on which optimiser restart won is not a
    stable benchmark. :func:`cross_validate` is what justifies the defaults.

    ``n_jobs=1`` on purpose: the benchmark calls ``fn`` one point at a time, and
    joblib's dispatch dominates a single-row 400-tree predict.
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

    ``random_kfold`` shuffles rows. On a full factorial this leaks: a held-out
    quaternary nearly always has a lattice neighbour in training, so it measures
    interpolation between adjacent measured points and little else.

    ``face_extrapolation`` trains on every row with <= 3 nonzero components and
    tests on the 1260 quaternaries. It is the closest available proxy for what the
    objective actually demands -- predict a higher-order mixture from lower-order
    ones -- because the plate has no 5- or 6-component rows to test against
    directly. A model that fails this has nothing to say about the interior.
    """
    D = load_plate(plate)
    X, y = D["X"], D["y"]
    nnz = (X > 1e-9).sum(axis=1)
    rng = np.random.default_rng(seed)
    fold = rng.permutation(X.shape[0]) % n_folds
    tr_lo, te_hi = np.where(nnz <= 3)[0], np.where(nnz == 4)[0]

    out: dict = {"plate": plate, "n_rows": int(X.shape[0]),
                 "y_std": float(y.std()), "y_units": "V (negated overpotential)",
                 "n_folds": int(n_folds),
                 "n_train_face_extrap": int(tr_lo.size),
                 "n_test_face_extrap": int(te_hi.size), "models": {}}
    for kind in kinds:
        t0 = time.perf_counter()
        pred = np.empty_like(y)
        for k in range(n_folds):
            te = fold == k
            f = build_surrogate(X[~te], y[~te], kind, seed=seed, n_jobs=-1)
            pred[te] = f(X[te])
        r2_k, rmse_k = _r2_rmse(y, pred)
        f = build_surrogate(X[tr_lo], y[tr_lo], kind, seed=seed, n_jobs=-1)
        r2_e, rmse_e = _r2_rmse(y[te_hi], f(X[te_hi]))
        out["models"][kind] = {
            "random_kfold_r2": r2_k, "random_kfold_rmse": rmse_k,
            "face_extrapolation_r2": r2_e, "face_extrapolation_rmse": rmse_e,
            "seconds_per_fit": float(time.perf_counter() - t0) / (n_folds + 1),
        }
    return out


# --- reference optima --------------------------------------------------------

def detect_peaks(grid: np.ndarray, z: np.ndarray, nbr: np.ndarray, *,
                 n_moves: int = PEAK_MOVES, min_sep: float = 2 * MATCH_RADIUS,
                 prominence_frac: float = PEAK_PROMINENCE_FRAC
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of a surface over a simplex lattice -- the naive detector.

    Mirrors ``warm_start.warm_gp_landscape.detect_peaks`` -- local maxima, a
    prominence floor ``prominence_frac`` of the way from the median to the global
    max, then descending-value thinning at ``min_sep`` -- with two changes d = 6
    forces:

      * "local" means no lattice point within ``n_moves`` one-unit moves is higher,
        evaluated by dilating ``z`` that many times over the move table. The 3-D /
        4-D version uses a KD-tree ball of radius ``3 * spacing``; on a 0.05 lattice
        over the 5-simplex that ball holds thousands of the 53130 points, and even
        one move here is 0.14 in composition L2, already far more global relative to
        the domain than three steps of a 220-point ternary grid.
      * ``min_sep`` defaults to ``2r = 0.1`` rather than 0.06. One lattice move is
        0.1414, so 0.06 would be inert, and ``metrics.merge_true_optima`` collapses
        anything closer than ``2r`` anyway -- pre-merging makes the count reported
        here the count the metrics will actually score against.
    """
    dil = z
    for _ in range(int(n_moves)):
        dil = _dilate(dil, nbr)
    is_max = z >= dil
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

    A peak survives when some measured row lies within ``radius`` of it AND that
    row's measured value clears ``v - value_tol * (v - background)`` -- exactly the
    near-AND-high test ``metrics.reached_flags`` applies to samples, turned around
    onto the reference set itself.

    This filter exists because of the hole documented at the top of this module.
    Nearly a third of the plate's own lattice, and two thirds of a 0.05 refinement,
    has no measurement anywhere near it, so a naive peak out there is shaped by the
    surrogate's inductive bias and nothing else. Declaring one a "true optimum"
    would score methods on a random forest's extrapolation habits.
    """
    from scipy.spatial import cKDTree

    if peaks.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    near = cKDTree(X).query_ball_point(peaks, r=radius)
    keep = np.zeros(peaks.shape[0], dtype=bool)
    for i, idx in enumerate(near):
        if not idx:
            continue
        thresh = values[i] - value_tol * max(values[i] - background, 0.0)
        keep[i] = bool(y[np.asarray(idx, dtype=int)].max() >= thresh)
    return keep


def basin_profile(peaks: np.ndarray, X: np.ndarray, y: np.ndarray,
                  radii: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
                  ) -> dict:
    """How fast the MEASURED value falls away from each reference optimum.

    For each radius, the mean measured ``y`` of the plate rows within that distance
    of a reference optimum, expressed as a fraction of the optimum's own prominence
    above the plate median. 1.0 means the neighbourhood is as good as the peak;
    0.0 means it is ordinary.

    Deliberately computed on raw rows rather than on the surrogate, because the
    question it answers -- "are these needles or hills?" -- is exactly the standing
    objection to this dataset ("too sparse/smooth compared to our data"), and a
    number computed from a model could be dismissed as the model's own smoothing.
    A random forest is piecewise constant and a fixed-length-scale GP is smooth by
    construction; the plate is neither.

    Radii below 0.15 are uninformative here and report 1.00 by definition: adjacent
    rows of a 0.1 lattice are 0.1414 apart, so the only row within 0.10 of a
    reference optimum is the optimum itself.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(X)
    bg = float(np.median(y))
    out: dict[str, dict] = {}
    for r in radii:
        fracs, counts = [], []
        for p in peaks:
            idx = tree.query_ball_point(p, r=r)
            v_pk = float(y[tree.query(p)[1]])
            if not idx or v_pk <= bg:
                continue
            fracs.append(float((y[np.asarray(idx, dtype=int)].mean() - bg) / (v_pk - bg)))
            counts.append(len(idx))
        out[f"{r:.2f}"] = {"mean_prominence_fraction": float(np.mean(fracs)) if fracs else float("nan"),
                           "mean_n_rows": float(np.mean(counts)) if counts else 0.0}
    return out


def _contrast(vals_peaks: np.ndarray, vals_probe: np.ndarray) -> dict:
    """``metrics.landscape_contrast``'s statistic against an arbitrary probe set."""
    p99 = float(np.percentile(vals_probe, 99))
    span = float(vals_probe.max() - np.median(vals_probe)) or 1.0
    return {
        "n_peaks": int(vals_peaks.size),
        "probe_median": float(np.median(vals_probe)),
        "probe_p99": p99,
        "probe_max": float(vals_probe.max()),
        "frac_peaks_above_probe_p99": (float((vals_peaks > p99).mean())
                                       if vals_peaks.size else float("nan")),
        "mean_peak_prominence": (float(((vals_peaks - np.median(vals_probe)) / span).mean())
                                 if vals_peaks.size else float("nan")),
    }


def _predict_chunked(predict, Q: np.ndarray, chunk: int = 8192) -> np.ndarray:
    return np.concatenate([predict(Q[i:i + chunk]) for i in range(0, Q.shape[0], chunk)])


def _peak_pass(predict, X, y, grid_n: int, n_moves: int) -> dict:
    """One (lattice, locality) peak-detection pass, naive and supported."""
    grid = simplex_lattice(grid_n, DIM)
    z = _predict_chunked(predict, grid)
    peaks, vals = detect_peaks(grid, z, _lattice_neighbours(grid_n, DIM),
                               n_moves=n_moves)
    background = float(np.median(z))
    keep = support_mask(peaks, vals, X, y, background)
    nnz = (peaks > 1e-9).sum(axis=1)
    return {"grid_n": int(grid_n), "n_moves": int(n_moves),
            "n_grid": int(grid.shape[0]), "background": background,
            "surrogate_min": float(z.min()), "surrogate_max": float(z.max()),
            "n_peaks_naive": int(peaks.shape[0]),
            "n_peaks_naive_unmeasured_order": int((nnz >= 5).sum()),
            "n_peaks_supported": int(keep.sum()),
            "peaks": peaks[keep], "values": vals[keep],
            "peaks_naive": peaks, "values_naive": vals}


def _reference_path(plate: str, kind: str) -> str:
    return os.path.join(DATA_DIR, f"oer6d_plate{plate}_{kind}_reference.json")


def reference_set(plate: str = DEFAULT_PLATE, kind: str = "rf", *,
                  grid_n: int = GRID_N, n_moves: int = PEAK_MOVES, seed: int = SEED,
                  rebuild: bool = False, n_probe: int = 4000) -> dict:
    """Peaks, support filter, contrast, CV and the grid sensitivity -- cached JSON.

    The cache holds only derived numbers (a few hundred floats), so it is cheap to
    delete and regenerate. The surrogate itself is refit on every load rather than
    pickled: an sklearn pickle that silently fails to load across a version bump is
    a far worse failure mode than a two-second refit.
    """
    path = _reference_path(plate, kind)
    if os.path.exists(path) and not rebuild:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    D = load_plate(plate)
    X, y = D["X"], D["y"]
    predict = build_surrogate(X, y, kind, seed=seed)
    main = _peak_pass(predict, X, y, grid_n, n_moves)

    rng = np.random.default_rng(seed)
    probe_u = rng.dirichlet(np.ones(DIM), size=n_probe)
    v_u = _predict_chunked(predict, probe_u)
    # Second probe distribution: measured rows plus a small jitter -- the part of
    # the domain the surrogate is entitled to speak about. The Dirichlet probe
    # never leaves the unmeasured interior, so on its own it flatters the peaks.
    probe_s = X[rng.integers(0, X.shape[0], size=n_probe)] \
        + rng.normal(0.0, 0.01, size=(n_probe, DIM))
    probe_s = np.clip(probe_s, 0.0, None)
    probe_s /= probe_s.sum(axis=1, keepdims=True)
    v_s = _predict_chunked(predict, probe_s)

    from scipy.spatial import cKDTree
    d_u = cKDTree(X).query(probe_u)[0]

    fine = _peak_pass(predict, X, y, FINE_GRID_N, n_moves)
    ref = {
        "plate": plate, "components": list(D["components"]), "surrogate": kind,
        "seed": int(seed), "n_probe": int(n_probe),
        **{k: main[k] for k in ("grid_n", "n_moves", "n_grid", "background",
                                "surrogate_min", "surrogate_max", "n_peaks_naive",
                                "n_peaks_naive_unmeasured_order", "n_peaks_supported")},
        "peaks": main["peaks"].tolist(),
        "peak_values": main["values"].tolist(),
        "peaks_naive": main["peaks_naive"].tolist(),
        "peak_values_naive": main["values_naive"].tolist(),
        "frac_peaks_on_measured_rows": float(
            (cKDTree(X).query(main["peaks"])[0] <= 1e-9).mean())
        if main["peaks"].shape[0] else float("nan"),
        "contrast_uniform_simplex": _contrast(main["values"], v_u),
        "contrast_uniform_simplex_naive_peaks": _contrast(main["values_naive"], v_u),
        "contrast_measured_support": _contrast(main["values"], v_s),
        "frac_uniform_probe_within_r_of_measurement": float((d_u <= MATCH_RADIUS).mean()),
        "median_uniform_probe_dist_to_measurement": float(np.median(d_u)),
        "grid_sensitivity_fine": {
            k: fine[k] for k in ("grid_n", "n_moves", "n_grid", "n_peaks_naive",
                                 "n_peaks_naive_unmeasured_order", "n_peaks_supported")},
        "basin_profile_measured": basin_profile(main["peaks"], X, y),
        "peak_nonzero_components": [int(v) for v in (main["peaks"] > 1e-9).sum(axis=1)],
        "audit": D["audit"],
        "cv": cross_validate(plate, seed=seed),
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ref, fh, indent=2)
    return ref


# --- the objective -----------------------------------------------------------

_VERDICT = (
    "Include as a secondary, clearly labelled external-data objective; do not make "
    "it headline evidence. FOR: all 10 reference optima are real measured "
    "compositions with real measured overpotentials -- neither real3d nor real4d "
    "can say that, both being peaks of a surrogate -- the plate is public and "
    "regenerable in seconds, and the contrast (0.40 of peaks above the uniform "
    "p99) sits between real3d (0.17) and real4d (0.46), so it discriminates about "
    "as well as the campaign objectives already in the suite. AGAINST, and both "
    "objections are confirmed by measurement, not by argument: (1) coverage -- the "
    "plate contains NO composition with more than four nonzero components, so 29% "
    "of its own lattice and 66% of a 0.05 refinement is model output rather than "
    "data, every reference optimum sits on the simplex boundary, and a uniform "
    "sampler scores exactly 0; (2) smoothness -- measured rows 0.15 away from an "
    "optimum still average 0.73 of its prominence, so these are hills, not "
    "needles. Brianna's 'too sparse/smooth compared to our data' is right on both "
    "counts. What it misses is that the sparsity is structured rather than random: "
    "the plate is dense on the low-order faces and empty everywhere else, which is "
    "a different failure from thin coverage and makes uniform-probe statistics "
    "such as landscape_contrast read optimistically. Use oer6d to show the method "
    "transfers to somebody else's hardware data; keep the sharp multi-optimum "
    "claim on the ensemble suite. If a sharper OER landscape is wanted, plates "
    "3851 and 4098 fall to 0.24 of prominence at 0.15 (versus 0.73 on 3496) and "
    "carry 38 and 14 reference optima, but they pay for it with lower contrast "
    "(0.24 and 0.14) because their overpotential range is compressed."
)


def oer6d(plate: str = DEFAULT_PLATE, kind: str = "rf", *, seed: int = SEED,
          grid_n: int = GRID_N, n_moves: int = PEAK_MOVES,
          supported_only: bool = True, rebuild: bool = False) -> dict:
    """Build the ``oer6d`` objective spec.

    Returns the plain kwargs dict ``objectives.Objective(**spec)`` takes -- ``name,
    dim, fn, true_optima, true_values, maximize, domain, meta`` -- rather than an
    ``Objective``, so this package never imports the registry that would import it
    back.

    ``supported_only=False`` swaps in the unfiltered peak set. It exists so the
    difference is measurable, not because it is ever the right reference.
    """
    ref = reference_set(plate, kind, grid_n=grid_n, n_moves=n_moves, seed=seed,
                        rebuild=rebuild)
    D = load_plate(plate)
    predict = build_surrogate(D["X"], D["y"], kind, seed=seed)

    key = "peaks" if supported_only else "peaks_naive"
    vkey = "peak_values" if supported_only else "peak_values_naive"
    P = np.asarray(ref[key], dtype=float).reshape(-1, DIM)
    V = np.asarray(ref[vkey], dtype=float)

    audit, cv = ref["audit"], ref["cv"]
    model = cv["models"][kind]
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
                      f"fn(x) is a MODEL value, not a measurement, at every x off "
                      f"the {LATTICE_STEP} lattice"),
        "surrogate_random_kfold_r2": model["random_kfold_r2"],
        "surrogate_random_kfold_rmse": model["random_kfold_rmse"],
        "surrogate_face_extrapolation_r2": model["face_extrapolation_r2"],
        "surrogate_face_extrapolation_rmse": model["face_extrapolation_rmse"],
        "surrogate_cv_all_models": cv["models"],
        "surrogate_cv_note": ("face_extrapolation trains on <=3-component rows and "
                              "tests on quaternaries; no test of 5-/6-component "
                              "prediction is possible on this plate at all"),

        "n_peaks_naive": ref["n_peaks_naive"],
        "n_peaks_supported": ref["n_peaks_supported"],
        "n_peaks_naive_unmeasured_order": ref["n_peaks_naive_unmeasured_order"],
        "frac_peaks_on_measured_rows": ref["frac_peaks_on_measured_rows"],
        "peak_support_rule": (f"a measured row within r={MATCH_RADIUS} whose value "
                              f"clears v - {PEAK_VALUE_TOL}*(v - background)"),
        "peak_grid": {"grid_n": ref["grid_n"], "n_points": ref["n_grid"],
                      "n_moves": ref["n_moves"]},
        "peak_grid_sensitivity": ref["grid_sensitivity_fine"],
        "reference": ("supported surrogate peaks; each one lies ON a measured "
                      "composition, unlike real3d/real4d"),

        "frac_peaks_above_random_p99": ref["contrast_uniform_simplex"]["frac_peaks_above_probe_p99"],
        "contrast_uniform_simplex": ref["contrast_uniform_simplex"],
        "contrast_uniform_simplex_naive_peaks": ref["contrast_uniform_simplex_naive_peaks"],
        "contrast_measured_support": ref["contrast_measured_support"],
        "basin_profile_measured": ref["basin_profile_measured"],
        "peak_nonzero_components": ref["peak_nonzero_components"],

        "n_quinary_senary_rows": audit["n_quinary_senary"],
        "frac_quinary_senary_rows": audit["frac_quinary_senary"],
        "max_nonzero_components_measured": audit["max_nonzero_components"],
        "frac_plate_lattice_unmeasured": audit["frac_plate_lattice_unmeasured"],
        "frac_fine_lattice_with_5plus_components": audit["frac_fine_lattice_with_5plus_components"],
        "frac_uniform_probe_within_r_of_measurement":
            ref["frac_uniform_probe_within_r_of_measurement"],

        "caveats": [
            "The four OER plates share an identical composition lattice but NOT "
            "their element sets (3496 Ni/Fe/Co/Mn/Ce/La, 3851 Ni/Fe/Co/Ta/Mn/Cu, "
            "3860 Sn/Fe/Co/Ta/Mn/Cu, 4098 Sn/Sb/Co/Ca/Ni/Mn). Four chemistries, "
            "not four replicates; never pooled.",
            f"The plate contains {audit['n_quinary_senary']} rows with 5 or 6 "
            f"nonzero components ({audit['frac_quinary_senary']:.1%}). Every "
            "measurement lies on a face of the 5-simplex of dimension <= 3, a "
            "measure-zero subset of the domain. "
            f"{audit['frac_plate_lattice_unmeasured']:.1%} of the plate's own 0.1 "
            f"lattice and {audit['frac_fine_lattice_with_5plus_components']:.1%} of "
            "a 0.05 refinement lie outside it.",
            f"Only {ref['frac_uniform_probe_within_r_of_measurement']:.1%} of "
            f"uniform simplex draws fall within r={MATCH_RADIUS} of any measured "
            "row (median distance "
            f"{ref['median_uniform_probe_dist_to_measurement']:.3f}), so "
            "metrics.landscape_contrast probes almost entirely inside the "
            "unmeasured region; contrast_measured_support is the honest counterpart.",
            f"Face-extrapolation R^2 is {model['face_extrapolation_r2']:.3f}: the "
            "surrogate does generalise from <=3-component rows to quaternaries, so "
            "this plate's chemistry is largely additive. Nothing on this plate "
            "tests the next step, to 5 and 6 components.",
            f"The measured lattice is quantised at {LATTICE_STEP} while the match "
            f"radius is {MATCH_RADIUS}, so no more than one measured composition "
            "can ever sit inside a match ball; peak_ratio here is a coarse "
            "statistic on a small reference set, not a fine one.",
            f"Reference set is small (n={ref['n_peaks_supported']}). A single "
            "matched or missed optimum moves peak_ratio by "
            f"{1.0 / max(ref['n_peaks_supported'], 1):.2f}; report a CI.",
            "Every reference optimum lies on the BOUNDARY -- each has 2 or 3 zero "
            f"components (nonzero counts {ref['peak_nonzero_components']}), because "
            "the plate never measured anything else. A method that stays in the "
            "interior scores 0 by construction; uniform Dirichlet search on this "
            "objective gets peak_ratio 0.00 at a 1000-sample budget.",
            "Basins are wide, and that is in the DATA, not the surrogate: measured "
            "rows within 0.15 of a reference optimum still average "
            f"{ref['basin_profile_measured']['0.15']['mean_prominence_fraction']:.2f} "
            "of its prominence, and within 0.30 still "
            f"{ref['basin_profile_measured']['0.30']['mean_prominence_fraction']:.2f}. "
            "These are hills, not needles. The 'too smooth' objection is correct at "
            "the scale of the match radius.",
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


def build(spec: dict) -> dict:
    """``objectives.build``-shaped entry point, if the registry grows a hook."""
    spec = dict(spec)
    spec.pop("kind", None)
    return oer6d(**spec)


# --- CLI ---------------------------------------------------------------------

def _report(plate: str, kind: str, rebuild: bool) -> None:
    ref = reference_set(plate, kind, rebuild=rebuild)
    a, cv, s = ref["audit"], ref["cv"], ref["grid_sensitivity_fine"]
    print(f"plate {ref['plate']}  components {'/'.join(ref['components'])}  "
          f"surrogate {ref['surrogate']}")
    print(f"  {a['n_rows']} rows, {a['n_unique_compositions']} unique, sums "
          f"[{a['row_sum_min']:.6f}, {a['row_sum_max']:.6f}], levels/component "
          f"{a['levels_per_component']}")
    print("  rows by nonzero components: "
          + " ".join(f"{k + 1}:{n}" for k, n in enumerate(a["n_by_nonzero_components"])))
    print(f"  quinary+senary rows: {a['n_quinary_senary']} "
          f"({a['frac_quinary_senary']:.4%} of the plate); every row on the 0.1 "
          f"lattice: {a['all_rows_on_plate_lattice']}; complete over <=4 nonzero: "
          f"{a['covers_all_low_order']}")
    print(f"  unmeasured: {a['n_plate_lattice_unmeasured']}/"
          f"{a['n_plate_lattice_points']} of the plate's own lattice "
          f"({a['frac_plate_lattice_unmeasured']:.1%}); "
          f"{a['frac_fine_lattice_with_5plus_components']:.1%} of a 0.05 lattice")
    print(f"  uniform simplex draws within r={MATCH_RADIUS} of a measurement: "
          f"{ref['frac_uniform_probe_within_r_of_measurement']:.2%} "
          f"(median distance {ref['median_uniform_probe_dist_to_measurement']:.4f})")
    print(f"  overpotential V: min {a['overpotential_v']['min']:.4f} median "
          f"{a['overpotential_v']['median']:.4f} max {a['overpotential_v']['max']:.4f}")
    print(f"  CV (y std {cv['y_std']:.4f} V; face split "
          f"{cv['n_train_face_extrap']} -> {cv['n_test_face_extrap']}):")
    for name, m in cv["models"].items():
        print(f"    {name:3s}  random_kfold R2 {m['random_kfold_r2']:+.3f} RMSE "
              f"{m['random_kfold_rmse']:.4f}  |  face_extrap R2 "
              f"{m['face_extrapolation_r2']:+.3f} RMSE "
              f"{m['face_extrapolation_rmse']:.4f}   ({m['seconds_per_fit']:.2f} s/fit)")
    print(f"  peaks on the plate lattice (n={ref['grid_n']}, {ref['n_grid']} pts, "
          f"{ref['n_moves']}-move): naive {ref['n_peaks_naive']} "
          f"({ref['n_peaks_naive_unmeasured_order']} in the unmeasured region) "
          f"-> supported {ref['n_peaks_supported']}  "
          f"[{ref['frac_peaks_on_measured_rows']:.0%} sit on a measured row]")
    print(f"  sensitivity, 0.05 lattice (n={s['grid_n']}, {s['n_grid']} pts): naive "
          f"{s['n_peaks_naive']} ({s['n_peaks_naive_unmeasured_order']} unmeasured) "
          f"-> supported {s['n_peaks_supported']}")
    for label, c in (("uniform simplex ", ref["contrast_uniform_simplex"]),
                     ("measured support", ref["contrast_measured_support"])):
        print(f"  contrast vs {label}: "
              f"{c['frac_peaks_above_probe_p99']:.3f} of {c['n_peaks']} peaks > p99 "
              f"(p99 {c['probe_p99']:.4f}, median {c['probe_median']:.4f}, "
              f"mean prominence {c['mean_peak_prominence']:.2f})")
    print(f"  contrast vs uniform, NAIVE peaks: "
          f"{ref['contrast_uniform_simplex_naive_peaks']['frac_peaks_above_probe_p99']:.3f}")
    print(f"  reference optima nonzero components: {ref['peak_nonzero_components']} "
          "(all on the simplex boundary)")
    print("  basin profile, MEASURED rows near an optimum (fraction of its "
          "prominence):")
    print("    " + "  ".join(
        f"r={r}: {b['mean_prominence_fraction']:.2f} (n={b['mean_n_rows']:.0f})"
        for r, b in ref["basin_profile_measured"].items()))


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
