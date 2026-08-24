"""``poisson5d`` -- the ZoMBI negative-Poisson's-ratio task, on a unit cube.

Source
------
Two artefacts ship with the ZoMBI-Hop repo and both were audited before anything
here was written:

    data/Material_Properties_Negative_Poisson   pandas DataFrame, 6628 x 8
    data/poisson_RF_trained.pkl                 zlib'd joblib, sklearn 1.1.1

The first is what it is claimed to be: a *pickled pandas DataFrame that needs
pymatgen*, not a stale sklearn pickle. Its ``composition`` column holds live
``pymatgen.core.composition.Composition`` objects, so ``find_class`` imports
pymatgen during unpickling and ``pd.read_pickle`` dies with ``ModuleNotFoundError:
No module named 'pymatgen'`` until it is installed. (It also carries a
``pandas.core.indexes.numeric.Int64Index``, removed in pandas 2.0, which survives
only because ``pandas.compat.pickle_compat`` still re-routes that name.)

The second will NOT load under modern sklearn -- ``ValueError: node array from the
pickle has an incompatible dtype``, because ``Tree`` gained a
``missing_go_to_left`` field. It is recoverable read-only by stubbing
``sklearn.tree._tree.Tree``, and doing so is how the provenance below was
established rather than assumed:

    upstream n_features_in_ = 5, n_estimators = 500, root n_node_samples = 4203
    (~0.634 x 6628, the expected bootstrap in-bag count)
    leaf values span exactly [-130.93308747429336, 77.04691460866245]

Those two leaf extremes are bit-identical to ``min``/``max`` of
``homogeneous_poisson`` in the DataFrame. The frame IS the training set of the
shipped model, and the shipped model was trained on the target unfiltered. Nothing
here needs the stale pickle; it is retrained from the frame.

The recipe, reproduced exactly
------------------------------
``github.com/PV-Lab/ZoMBI``, ``data/poisson/train_RF.py``: pull the Materials
Project summary for ``[nelements, composition, density, energy_per_atom, efermi,
energy_above_hull, band_gap, homogeneous_poisson]``, drop rows with NaN
``homogeneous_poisson`` or ``efermi``, take ``X = df.iloc[:, 2:-1]``, normalise
each column by ``|x| / max|x|``, and fit ``RandomForestRegressor(n_estimators=500)``.

``iloc[:, 2:-1]`` is columns 2..6, so the task is **5-dimensional**, not 6:

    density, energy_per_atom, efermi, energy_above_hull, band_gap

confirmed three ways -- the slice, the shipped model's ``n_features_in_ = 5``, and
the upstream notebook cell (``dimensions = 5 # 5D X-dataset used to train the RF
model``). ``dim=6`` is available and prepends ``nelements``, but that column is
what ``iloc[:, 2:-1]`` deliberately skips, so it is NOT the published task and
says so in ``meta``. If the registry wants a name, this is ``poisson5d``.

Because every column is divided by its own ``max|x|``, the search space is the
**unit cube** ``[0, 1]^5`` -- hence ``domain="cube"``, and the upstream notebook
samples ``lower = zeros(5), upper = ones(5)``. The ``abs()`` is lossy and upstream:
``efermi`` is negative for 491 of 6628 rows and those fold onto their positive
mirror, so two materials at -3 eV and +3 eV become the same point.

Direction. ZoMBI maximises whatever it is handed (``fX_best = max(fX_new)``) and
reports ``minimum.accumulate(-1 * fX)``; the notebook hands it
``-1 * poisson_model(X)``. So the paper MINIMISES Poisson's ratio, and the thing
its optimiser maximises is ``-nu``. zhbench maximises, so ``fn(x) = -RF(x)`` --
identical to what ZoMBI actually optimises. ``meta["sign"]`` records it.

What the data actually is, and three reasons to distrust it
-----------------------------------------------------------
6628 rows survive the NaN filter out of the ~146k-material Materials Project pull
of 30 June 2022 -- i.e. 95.5% of MP has no elasticity data at all. 54 of the 6628
(0.81%) have ``nu < 0``, matching the upstream README's needle-in-a-haystack claim.

1. **The target contains physically impossible values.** An isotropic solid has
   ``nu`` in ``[-1, 0.5]``. 181 rows violate that, 38 by more than 1.0, and the
   extremes are elemental Dy at ``nu = -130.93`` and Ti3Ga at ``nu = +77.05`` --
   Materials Project elastic-tensor fitting artefacts, not auxetics. Training on
   them (which upstream does) is why the faithful recipe scores **5-fold CV
   R^2 = -22.4** and OOB R^2 = -0.65: worse than predicting the mean. It also puts
   a single 79-high spike on a landscape whose median is -0.5, which is not a
   multi-optimum benchmark, it is one artefact. ``target_range`` defaults to
   ``(-1.0, 0.5)`` here for that reason; ``target_range=None`` reproduces upstream
   byte-for-byte and both are measured in ``meta`` and ``--report``.

2. **Five crude scalars do not determine a Poisson ratio.** The median material has
   16 other materials within the benchmark match radius ``r = 0.05``, and a pair
   inside ``r`` differs in ``nu`` by a median of 0.046 -- 58% of the whole standard
   deviation of the physical target (0.0796). Averaging that within-radius variance
   gives an **R^2 ceiling of 0.363** for *any* model on these features at this
   resolution. The retrained forest reaches 0.153, so it captures ~40% of the only
   signal that is there. Grouping folds by chemical system (4070 systems, so
   polymorphs cannot leak) barely moves it: 0.143. This is not a tuning problem.

3. **The cube is almost entirely empty.** Only **0.08%** of ``[0, 1]^5`` lies within
   ``r = 0.05`` of any measured material; the median uniform draw is 0.58 away.
   ``energy_above_hull`` and ``band_gap`` normalise to means of 0.008 and 0.048, so
   real materials occupy a thin slab and the other 99.92% of the domain is forest
   extrapolation. ``metrics.landscape_contrast``, which probes uniformly, probes
   almost entirely outside the data -- the support-conditioned contrast reported
   beside it is the honest one.

Reference optima
----------------
Same discipline as ``oer.py`` and ``warm_start.warm_gp_landscape``: local maxima of
the surrogate on a ``1/20`` grid (spacing = the match radius; 21^5 = 4084101
points), where "local" means no grid point within two axis-steps is higher; a
prominence floor at ``prominence_frac`` of the way from the background median to
the global max; thinning at ``min_sep = 2r = 0.1`` so ``metrics.merge_true_optima``
has nothing left to collapse.

Then the filter this dataset cannot do without, given (3): a peak survives only if
some MEASURED material lies within ``r`` of it AND that material's own ``-nu``
clears ``v - value_tol * (v - background)`` -- ``metrics.reached_flags``' near-AND-
high test, turned around onto the reference set. A peak with no measurement near it
is a fact about a random forest, not about materials.

Reproduce every number here with::

    python -m benchmarks.zhbench.datasets.poisson --fetch     # data, from scratch
    python -m benchmarks.zhbench.datasets.poisson --retrain   # cache the forest
    python -m benchmarks.zhbench.datasets.poisson --report    # both variants
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from functools import lru_cache

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(_REPO_ROOT, "data", "poisson")

#: The frame ships in the ZoMBI-Hop repo, not in PV-Lab/ZoMBI (which ships only the
#: model and ``train_RF.py``). Verified 200 / 827475 B, byte-identical to the copy
#: already sitting in ``data/``.
_FRAME_URL = ("https://raw.githubusercontent.com/PV-Lab/ZoMBI-Hop/main/"
              "data/Material_Properties_Negative_Poisson")
_RECIPE_URL = ("https://raw.githubusercontent.com/PV-Lab/ZoMBI/main/"
               "data/poisson/train_RF.py")

_PICKLE_NAME = "Material_Properties_Negative_Poisson"
_FRAME_CSV = os.path.join(DATA_DIR, "material_properties_negative_poisson.csv")

#: Every column of the pull, in the order ``train_RF.py`` builds them. The feature
#: slice is ``iloc[:, 2:-1]``, i.e. index 2 through 6.
ALL_COLUMNS = ("nelements", "composition", "density", "energy_per_atom", "efermi",
               "energy_above_hull", "band_gap", "homogeneous_poisson")
FEATURES = ("density", "energy_per_atom", "efermi", "energy_above_hull", "band_gap")
TARGET = "homogeneous_poisson"

#: ``dim=6`` prepends ``nelements`` -- the column the published slice skips. Kept
#: only so a 6-D variant is measurable rather than imagined; it is not the paper's.
FEATURE_SETS: dict[int, tuple[str, ...]] = {5: FEATURES, 6: ("nelements",) + FEATURES}

DIM = 5
RF_TREES = 500              # train_RF.py: RandomForestRegressor(n_estimators = 500)
GRID_N = 20                 # 1/20 = 0.05 = the match radius, as in oer.py
MATCH_RADIUS = 0.05         # optimize.eval_metrics.MATCH_RADIUS
PEAK_PROMINENCE_FRAC = 0.12  # warm_gp_landscape._PEAK_PROMINENCE_FRAC
PEAK_VALUE_TOL = 0.25       # metrics.reached_flags default value_tol
MAX_PEAK_CANDIDATES = 20000
SEED = 0

#: An isotropic solid cannot have ``nu`` outside this interval; rows that do are
#: failed elastic-tensor fits. Default because a benchmark landscape built on
#: ``nu = -130`` measures Materials Project's error bars. ``None`` = upstream.
PHYSICAL_RANGE = (-1.0, 0.5)


# --- fetch -------------------------------------------------------------------

def fetch(*, dest: str = DATA_DIR, force: bool = False, timeout: float = 120.0,
          from_mp: bool = False, api_key: str | None = None) -> dict:
    """Regenerate the dataset from scratch into ``dest``; nothing is committed.

    Default path downloads the 827 kB frame from the ZoMBI-Hop repo and transcodes
    it to CSV. The transcode is the point: the raw pickle can only be opened with
    pymatgen installed, and a benchmark that needs a 60 MB crystallography stack to
    read five floats per row is a benchmark nobody will rerun. The CSV keeps
    ``composition`` as a reduced formula string, which is all it was ever used for
    (it is not a feature).

    ``from_mp=True`` re-runs ``train_RF.py``'s original Materials Project query
    instead. It needs ``mp_api`` and an API key, and it will NOT reproduce this
    file: the pull is dated 30 June 2022 and MP has been recomputed many times
    since. Offered so the provenance is executable, not because it is equivalent.
    """
    os.makedirs(dest, exist_ok=True)
    if from_mp:
        return _fetch_from_materials_project(dest=dest, api_key=api_key)

    raw = os.path.join(dest, _PICKLE_NAME)
    status = {}
    if force or not os.path.exists(raw):
        with urllib.request.urlopen(_FRAME_URL, timeout=timeout) as resp:
            blob = resp.read()
        with open(raw, "wb") as fh:
            fh.write(blob)
        status["frame_pickle"] = f"downloaded {len(blob)} B"
    else:
        status["frame_pickle"] = f"cached {os.path.getsize(raw)} B"

    n = _transcode(raw, _FRAME_CSV, force=force)
    status["frame_csv"] = f"{n} rows -> {_FRAME_CSV} ({os.path.getsize(_FRAME_CSV)} B)"
    return status


def _transcode(pickle_path: str, csv_path: str, *, force: bool = False) -> int:
    """Pickle -> pymatgen-free CSV. The one place pymatgen is ever needed."""
    import pandas as pd

    if os.path.exists(csv_path) and not force:
        return int(sum(1 for _ in open(csv_path, encoding="utf-8")) - 1)
    try:
        df = pd.read_pickle(pickle_path)
    except ModuleNotFoundError as exc:      # pragma: no cover - environment issue
        raise ModuleNotFoundError(
            f"{pickle_path} stores live pymatgen Composition/Element objects, so "
            f"unpickling it imports pymatgen ({exc}). Install it once "
            "(`pip install pymatgen`) to produce the CSV; nothing downstream of "
            "the CSV needs it.") from exc
    if tuple(df.columns) != ALL_COLUMNS:
        raise ValueError(f"frame columns changed upstream: {tuple(df.columns)} != "
                         f"{ALL_COLUMNS}; re-check the train_RF.py slice before use")
    out = df.copy()
    out["composition"] = out["composition"].map(
        lambda c: getattr(c, "reduced_formula", None) or str(c))
    out["elements"] = df["composition"].map(
        lambda c: "-".join(sorted(e.symbol for e in c.elements))
        if hasattr(c, "elements") else "")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    out.to_csv(csv_path, index=True, index_label="mp_row")
    return int(out.shape[0])


def _fetch_from_materials_project(*, dest: str, api_key: str | None) -> dict:
    """``train_RF.py``'s query, verbatim in structure. Will not reproduce 2022 data."""
    key = api_key or os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError("set MP_API_KEY (or pass --api-key) to pull from "
                           "Materials Project; note the result will differ from the "
                           "30 June 2022 snapshot the published model was fit to")
    import pandas as pd
    from mp_api.client import MPRester

    props = list(ALL_COLUMNS)
    with MPRester(key) as mpr:
        docs = mpr.summary.search(fields=props)
    frame = pd.DataFrame({p: [getattr(d, p) for d in docs] for p in props})
    frame = frame[~np.isnan(frame[TARGET])]
    frame = frame[~np.isnan(frame["efermi"])]
    path = os.path.join(dest, "material_properties_negative_poisson_repull.csv")
    frame = frame.copy()
    frame["composition"] = frame["composition"].map(
        lambda c: getattr(c, "reduced_formula", None) or str(c))
    frame.to_csv(path, index=False)
    return {"repull_rows": int(frame.shape[0]), "path": path,
            "warning": "fresh MP pull; NOT the 30 June 2022 snapshot"}


def _frame_path() -> str:
    """The CSV, transcoding the pickle from wherever it already lives if needed."""
    if os.path.exists(_FRAME_CSV):
        return _FRAME_CSV
    for cand in (os.path.join(DATA_DIR, _PICKLE_NAME),
                 os.path.join(_REPO_ROOT, "data", _PICKLE_NAME)):
        if os.path.exists(cand):
            _transcode(cand, _FRAME_CSV)
            return _FRAME_CSV
    raise FileNotFoundError(
        "no Poisson frame on disk. Run: "
        "python -m benchmarks.zhbench.datasets.poisson --fetch")


# --- loading -----------------------------------------------------------------

@lru_cache(maxsize=4)
def load_frame(dim: int = DIM, target_range: tuple | None = PHYSICAL_RANGE) -> dict:
    """Features, target and the normalisation, exactly as ``train_RF.py`` builds them.

    ``scale`` is ``max|x|`` per column over the WHOLE frame, so ``Xn = |X| / scale``
    lands in ``[0, 1]`` -- the cube the benchmark searches. It is computed before
    ``target_range`` drops any row, so the two variants share one coordinate system
    and their peaks are directly comparable.
    """
    import pandas as pd

    df = pd.read_csv(_frame_path())
    cols = FEATURE_SETS[int(dim)]
    X_raw = df[list(cols)].to_numpy(dtype=float)
    y_nu = df[TARGET].to_numpy(dtype=float)
    scale = np.abs(X_raw).max(axis=0)
    Xn = np.abs(X_raw) / scale

    keep = np.ones(y_nu.size, dtype=bool)
    rng_used = None
    if target_range is not None:
        rng_used = (float(target_range[0]), float(target_range[1]))
        keep = (y_nu >= rng_used[0]) & (y_nu <= rng_used[1])
    return {
        "columns": cols,
        "X": Xn[keep],
        "y": -y_nu[keep],                   # maximisation: ZoMBI optimises -nu
        "nu": y_nu[keep],
        "nu_all": y_nu,
        "scale": scale,
        "formula": df["composition"].to_numpy(dtype=object)[keep],
        "elements": (df["elements"].to_numpy(dtype=object)[keep]
                     if "elements" in df.columns else None),
        "n_dropped": int((~keep).sum()),
        "target_range": rng_used,
        "audit": _audit(Xn, y_nu, keep),
    }


def _audit(Xn: np.ndarray, nu: np.ndarray, keep: np.ndarray) -> dict:
    """The numbers that decide whether this objective means anything.

    ``within_r_*`` is the load-bearing one: it measures how much the target varies
    among materials the benchmark cannot tell apart, which caps the R^2 any model
    can reach on these features regardless of how it is fit.
    """
    from scipy.spatial import cKDTree

    Xk, nk = Xn[keep], nu[keep]
    tree = cKDTree(Xk)
    pairs = tree.query_pairs(MATCH_RADIUS, output_type="ndarray")
    dnu = (np.abs(nk[pairs[:, 0]] - nk[pairs[:, 1]]) if pairs.size
           else np.zeros(0))
    nbr = tree.query_ball_point(Xk, r=MATCH_RADIUS)
    within_var = np.asarray([float(np.var(nk[np.asarray(i, dtype=int)]))
                             for i in nbr if len(i) >= 3])
    total_var = float(np.var(nk)) or 1.0

    rng = np.random.default_rng(SEED)
    probe = rng.random((20000, Xk.shape[1]))
    dist = tree.query(probe)[0]
    return {
        "n_rows_total": int(nu.size),
        "n_rows_used": int(keep.sum()),
        "n_rows_dropped": int((~keep).sum()),
        # Independent of `keep`, so both variants report the same audit of how much
        # of the published target is physically impossible.
        "n_rows_nonphysical": int(((nu < PHYSICAL_RANGE[0]) | (nu > PHYSICAL_RANGE[1])).sum()),
        "n_rows_nu_gt_1": int((np.abs(nu) > 1.0).sum()),
        "nu_min_all": float(nu.min()), "nu_max_all": float(nu.max()),
        "nu_min_used": float(nk.min()), "nu_max_used": float(nk.max()),
        "nu_std_used": float(nk.std()),
        "frac_negative_nu_used": float((nk < 0).mean()),
        "n_negative_nu_used": int((nk < 0).sum()),
        "median_neighbours_within_r": float(np.median([len(i) for i in nbr])),
        "within_r_median_abs_dnu": float(np.median(dnu)) if dnu.size else float("nan"),
        "within_r_mean_variance": float(within_var.mean()) if within_var.size else float("nan"),
        "implied_r2_ceiling": (1.0 - float(within_var.mean()) / total_var
                               if within_var.size else float("nan")),
        "frac_cube_within_r_of_measurement": float((dist <= MATCH_RADIUS).mean()),
        "median_cube_dist_to_measurement": float(np.median(dist)),
        "normalised_column_means": [float(v) for v in Xk.mean(axis=0)],
    }


# --- surrogate ---------------------------------------------------------------

def _model_path(dim: int, tag: str) -> str:
    return os.path.join(DATA_DIR, f"poisson{dim}d_{tag}_rf{RF_TREES}.joblib")


def _tag(target_range: tuple | None) -> str:
    return "asPublished" if target_range is None else "physical"


def train_rf(dim: int = DIM, target_range: tuple | None = PHYSICAL_RANGE, *,
             seed: int = SEED, cache: bool = True, n_jobs: int = -1):
    """``train_RF.py``'s forest: 500 trees, defaults everywhere else, on ``|x|/max|x|``.

    Fits in ~1 s, so the on-disk cache is a convenience, not a dependency, and it is
    only reused when the recording sklearn version matches. That check is the whole
    lesson of ``poisson_RF_trained.pkl``: a forest pickled under 1.1.1 is dead
    weight the moment ``Tree`` grows a field, and silently refitting beats raising
    ``incompatible dtype`` at benchmark time.
    """
    import joblib
    import sklearn
    from sklearn.ensemble import RandomForestRegressor

    path = _model_path(dim, _tag(target_range))
    meta_path = path + ".json"
    if cache and os.path.exists(path) and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            info = json.load(fh)
        if info.get("sklearn") == sklearn.__version__ and info.get("seed") == seed:
            return joblib.load(path), info
    D = load_frame(dim, target_range)
    t0 = time.perf_counter()
    rf = RandomForestRegressor(n_estimators=RF_TREES, random_state=seed,
                               n_jobs=n_jobs, oob_score=True)
    rf.fit(D["X"], D["nu"])                 # fit on nu; fn negates. Upstream order.
    info = {"sklearn": sklearn.__version__, "seed": int(seed), "dim": int(dim),
            "features": list(D["columns"]), "target": TARGET,
            "target_range": D["target_range"], "n_train": int(D["X"].shape[0]),
            "n_estimators": RF_TREES, "oob_r2": float(rf.oob_score_),
            "fit_seconds": float(time.perf_counter() - t0),
            "scale": [float(v) for v in D["scale"]]}
    if cache:
        os.makedirs(DATA_DIR, exist_ok=True)
        joblib.dump(rf, path, compress=3)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)
        info["cached_at"] = path
    return rf, info


def _predictors(rf) -> tuple:
    """Deterministic single-point and batched ``predict``, bypassing sklearn's.

    Two problems with ``RandomForestRegressor.predict``, both load-bearing here.

    Speed: it costs 15.6 ms on one row -- 70 ms with ``n_jobs=-1``, which is worse,
    because thread-pool setup dominates a 500-tree single-row descent. The benchmark
    calls ``fn`` one point at a time, so a 1000-sample run would spend 16 s inside
    input validation and ``Parallel``. Walking ``estimator.tree_`` in a fixed order
    costs 1.3 ms.

    Reproducibility, which matters more: with ``n_jobs=-1`` sklearn accumulates tree
    outputs into a shared buffer from several threads, so repeated calls differ by
    ~3e-16 and are not bitwise reproducible. A forest is piecewise constant -- 387
    of 20000 uniform probes hit an exact tie -- so ``z >= dilate(z)`` flips on
    plateaus and the SAME cached model yielded 8 supported peaks on one build and 9
    on the next. Summing in list order fixes the reference set. Both closures use
    the identical accumulate-then-scale sequence, so ``fn(peak)`` equals the cached
    ``true_values`` exactly rather than approximately.
    """
    trees = [e.tree_ for e in rf.estimators_]
    inv = 1.0 / len(trees)
    buf = np.empty((1, rf.n_features_in_), dtype=np.float32)

    def predict_one(x) -> float:
        buf[0, :] = np.asarray(x, dtype=float).ravel()
        total = 0.0
        for t in trees:
            total += t.predict(buf)[0, 0]
        return float(total * inv)

    def predict_many(X) -> np.ndarray:
        Q = np.ascontiguousarray(X, dtype=np.float32)
        out = np.zeros(Q.shape[0], dtype=float)
        for t in trees:
            out += t.predict(Q)[:, 0]
        out *= inv
        return out
    return predict_one, predict_many


def _point_predictor(rf):
    return _predictors(rf)[0]


def _batch_predict(rf, X: np.ndarray, chunk: int = 131072) -> np.ndarray:
    many = _predictors(rf)[1]
    out = np.empty(X.shape[0], dtype=float)
    for i in range(0, X.shape[0], chunk):
        out[i:i + chunk] = many(X[i:i + chunk])
    return out


# --- held-out accuracy -------------------------------------------------------

def cross_validate(dim: int = DIM, target_range: tuple | None = PHYSICAL_RANGE, *,
                   n_folds: int = 5, seed: int = SEED) -> dict:
    """Two protocols, because a random split can flatter this data and does not.

    ``random_kfold`` shuffles rows. ``grouped_by_chemistry`` splits on the element
    system (4070 of them), so the 21 Al2O3 polymorphs and the 20 carbon allotropes
    -- which share nearly identical features and can share a Poisson ratio -- cannot
    straddle the split. The two agreeing (0.153 vs 0.143 on the physical variant) is
    the useful result: the weakness is the feature set, not leakage.

    Spearman is reported alongside R^2 because it is the only statistic that stays
    meaningful on the as-published variant, where a handful of |nu| > 1 rows make
    the squared error unbounded.
    """
    from scipy.stats import spearmanr
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold, KFold

    D = load_frame(dim, target_range)
    X, nu = D["X"], D["nu"]
    groups = D["elements"] if D["elements"] is not None else np.arange(nu.size)

    def run(splits) -> dict:
        r2, rmse, mae, rho = [], [], [], []
        for tr, te in splits:
            m = RandomForestRegressor(n_estimators=RF_TREES, random_state=seed,
                                      n_jobs=-1).fit(X[tr], nu[tr])
            p = m.predict(X[te])
            ss = float(((nu[te] - nu[te].mean()) ** 2).sum())
            r2.append(1.0 - float(((nu[te] - p) ** 2).sum()) / ss if ss > 0 else np.nan)
            rmse.append(float(np.sqrt(((nu[te] - p) ** 2).mean())))
            mae.append(float(np.abs(nu[te] - p).mean()))
            rho.append(float(spearmanr(nu[te], p).statistic))
        return {"r2": float(np.mean(r2)), "rmse": float(np.mean(rmse)),
                "mae": float(np.mean(mae)), "spearman": float(np.mean(rho)),
                "r2_folds": [float(v) for v in r2]}

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    gk = GroupKFold(n_splits=n_folds)
    _, info = train_rf(dim, target_range, seed=seed, cache=False)
    return {
        "n_rows": int(nu.size), "nu_std": float(nu.std()),
        "n_groups": int(np.unique(groups).size),
        "random_kfold": run(kf.split(X)),
        "grouped_by_chemistry": run(gk.split(X, nu, groups)),
        "oob_r2_full_fit": info["oob_r2"],
        "r2_ceiling_from_within_radius_spread": D["audit"]["implied_r2_ceiling"],
    }


# --- reference optima --------------------------------------------------------

def _grid_coords(flat_idx: np.ndarray, n: int, d: int) -> np.ndarray:
    return (np.stack(np.unravel_index(flat_idx, (n + 1,) * d), axis=1).astype(float)
            / float(n))


def _dilate(Z: np.ndarray) -> np.ndarray:
    """Max over each cell and its 2d axis-aligned neighbours on a regular grid."""
    out = Z.copy()
    for ax in range(Z.ndim):
        lo = [slice(None)] * Z.ndim
        hi = [slice(None)] * Z.ndim
        lo[ax], hi[ax] = slice(0, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        np.maximum(out[lo], Z[hi], out=out[lo])
        np.maximum(out[hi], Z[lo], out=out[hi])
    return out


def detect_peaks(rf, dim: int, *, grid_n: int = GRID_N,
                 min_sep: float = 2 * MATCH_RADIUS,
                 prominence_frac: float = PEAK_PROMINENCE_FRAC,
                 max_candidates: int = MAX_PEAK_CANDIDATES
                 ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Local maxima of ``-RF`` on a ``1/grid_n`` cube lattice.

    A forest is piecewise constant, so gradient ascent is meaningless and an
    exhaustive lattice is both simpler and exact at the resolution that matters:
    21^5 = 4084101 points at spacing 0.05, which is the match radius, so two peaks
    the grid cannot separate are two peaks no metric in this suite can separate
    either. "Local" is two axis-steps, matching ``oer.detect_peaks``' two lattice
    moves -- a full Chebyshev ball would be 3^5 shifts for the same intent.

    Ties matter here and do not on a smooth GP: plateaus of exactly equal forest
    output are common, and ``z >= dilate(z)`` marks every cell of a plateau. The
    ``min_sep`` thinning that follows collapses each plateau to its first member,
    which is the desired behaviour but means the raw local-max count is not a peak
    count and is reported separately as ``n_local_max_cells``.
    """
    n_total = (grid_n + 1) ** dim
    many = _predictors(rf)[1]
    z = np.empty(n_total, dtype=float)
    for start in range(0, n_total, 131072):
        idx = np.arange(start, min(start + 131072, n_total))
        z[idx] = -many(_grid_coords(idx, grid_n, dim))

    Z = z.reshape((grid_n + 1,) * dim)
    is_max = (Z >= _dilate(_dilate(Z))).ravel()
    background = float(np.median(z))
    floor = background + prominence_frac * (float(z.max()) - background)
    cand = np.where(is_max & (z >= floor))[0]
    order = cand[np.argsort(z[cand])[::-1]][:max_candidates]

    kept: list[int] = []
    pts: list[np.ndarray] = []
    for idx in order:
        p = _grid_coords(np.asarray([idx]), grid_n, dim)[0]
        if all(np.linalg.norm(p - q) >= min_sep for q in pts):
            kept.append(int(idx))
            pts.append(p)
    sel = np.asarray(kept, dtype=int)
    stats = {"grid_n": int(grid_n), "n_grid": int(n_total),
             "n_local_max_cells": int(is_max.sum()),
             "n_above_prominence_floor": int(cand.size),
             "candidates_truncated": bool(cand.size > max_candidates),
             "background": background, "prominence_floor": float(floor),
             "surface_min": float(z.min()), "surface_max": float(z.max())}
    P = (np.stack(pts) if pts else np.empty((0, dim)))
    return P, z[sel], stats


def support_mask(peaks: np.ndarray, values: np.ndarray, X: np.ndarray, y: np.ndarray,
                 background: float, *, radius: float = MATCH_RADIUS,
                 value_tol: float = PEAK_VALUE_TOL) -> np.ndarray:
    """Which peaks a real measurement vouches for -- near AND high, per peak.

    Identical in form to ``oer.support_mask`` and to ``metrics.reached_flags``' test
    on samples. It carries far more weight here: only 0.08% of the cube is within
    ``radius`` of any material, so without it the reference set would be a catalogue
    of the forest's behaviour in the 99.92% of the domain where no material exists.
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


def _peak_support(peaks: np.ndarray, values: np.ndarray, D: dict,
                  *, radius: float = MATCH_RADIUS) -> list[dict]:
    """Name the material vouching for each surviving peak, and its actual nu.

    Recorded because the support test alone can flatter a peak: the threshold
    ``v - value_tol * (v - background)`` is easy to clear when ``v`` barely exceeds
    the background, so a low peak sitting in dense data passes on the strength of an
    ordinary material. Writing the supporting formula and its nu into the reference
    makes that visible instead of leaving a bare count of 'supported' peaks.
    """
    from scipy.spatial import cKDTree

    if peaks.shape[0] == 0:
        return []
    tree = cKDTree(D["X"])
    out = []
    for p, v in zip(peaks, values):
        idx = np.asarray(tree.query_ball_point(p, r=radius), dtype=int)
        best = int(idx[np.argmin(D["nu"][idx])]) if idx.size else -1
        out.append({
            "peak_value": float(v),
            "n_materials_within_r": int(idx.size),
            "best_formula": str(D["formula"][best]) if best >= 0 else None,
            "best_nu": float(D["nu"][best]) if best >= 0 else None,
            "nearest_dist": float(tree.query(p)[0]),
        })
    return out


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


def _reference_path(dim: int, tag: str) -> str:
    return os.path.join(DATA_DIR, f"poisson{dim}d_{tag}_reference.json")


def reference_set(dim: int = DIM, target_range: tuple | None = PHYSICAL_RANGE, *,
                  grid_n: int = GRID_N, seed: int = SEED, rebuild: bool = False,
                  n_probe: int = 4000) -> dict:
    """Peaks, support filter, contrast and CV -- cached as a few hundred floats.

    The lattice sweep is ~4.1M forest evaluations (about a minute), so it is cached;
    the forest itself is not stored in here, only rebuilt or reloaded, for the
    version reason in :func:`train_rf`.
    """
    tag = _tag(target_range)
    path = _reference_path(dim, tag)
    if os.path.exists(path) and not rebuild:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    D = load_frame(dim, target_range)
    rf, info = train_rf(dim, target_range, seed=seed)
    t0 = time.perf_counter()
    peaks, vals, stats = detect_peaks(rf, dim, grid_n=grid_n)
    keep = support_mask(peaks, vals, D["X"], D["y"], stats["background"])

    rng = np.random.default_rng(seed)
    probe_u = rng.random((n_probe, dim))
    v_u = -_batch_predict(rf, probe_u)
    # Second probe: measured materials plus a jitter smaller than the match radius,
    # i.e. the sliver of the cube the forest is entitled to speak about. The uniform
    # probe never visits it, so on its own it makes every peak look decisive.
    ridx = rng.integers(0, D["X"].shape[0], size=n_probe)
    probe_s = np.clip(D["X"][ridx] + rng.normal(0.0, 0.02, size=(n_probe, dim)), 0.0, 1.0)
    v_s = -_batch_predict(rf, probe_s)

    ref = {
        "dim": int(dim), "tag": tag, "features": list(D["columns"]),
        "target": TARGET, "target_range": D["target_range"], "seed": int(seed),
        "scale": [float(v) for v in D["scale"]],
        "n_peaks_naive": int(peaks.shape[0]),
        "n_peaks_supported": int(keep.sum()),
        "peaks": peaks[keep].tolist(), "peak_values": vals[keep].tolist(),
        "peaks_naive": peaks.tolist(), "peak_values_naive": vals.tolist(),
        "peak_support": _peak_support(peaks[keep], vals[keep], D),
        "n_peaks_auxetic_support": int(sum(
            1 for s in _peak_support(peaks[keep], vals[keep], D)
            if s["best_nu"] is not None and s["best_nu"] < 0.0)),
        "grid": stats,
        "contrast_uniform_cube": _contrast(vals[keep], v_u),
        "contrast_uniform_cube_naive_peaks": _contrast(vals, v_u),
        "contrast_measured_support": _contrast(vals[keep], v_s),
        "audit": D["audit"],
        "cv": cross_validate(dim, target_range, seed=seed),
        "model": info,
        "build_seconds": float(time.perf_counter() - t0),
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(ref, fh, indent=2)
    return ref


# --- the objective -----------------------------------------------------------

def poisson5d(dim: int = DIM, target_range: tuple | None = PHYSICAL_RANGE, *,
              seed: int = SEED, grid_n: int = GRID_N, supported_only: bool = True,
              rebuild: bool = False) -> dict:
    """Build the objective spec: the kwargs ``objectives.Objective(**spec)`` takes.

    Returns a plain dict (``name, dim, fn, true_optima, true_values, maximize,
    domain, meta``) rather than an ``Objective``, so this package never imports the
    registry that would import it back.

    ``target_range=None`` reproduces the published model exactly, artefacts and all;
    ``supported_only=False`` swaps in the unfiltered peak set. Both exist so the
    difference is measurable, not because either is the right default.
    """
    ref = reference_set(dim, target_range, grid_n=grid_n, seed=seed, rebuild=rebuild)
    rf, _ = train_rf(dim, target_range, seed=seed)
    predict_one = _point_predictor(rf)

    key, vkey = (("peaks", "peak_values") if supported_only
                 else ("peaks_naive", "peak_values_naive"))
    P = np.asarray(ref[key], dtype=float).reshape(-1, dim)
    V = np.asarray(ref[vkey], dtype=float)

    a, cv, tag = ref["audit"], ref["cv"], ref["tag"]
    name = f"poisson{dim}d"
    if tag != "physical":
        name += "_asPublished"
    if not supported_only:
        name += "_naivepeaks"
    if dim != DIM:
        name += "_nelements"

    meta = {
        "kind": "poisson",
        "source": "PV-Lab/ZoMBI data/poisson/train_RF.py; frame from PV-Lab/ZoMBI-Hop",
        "url": _FRAME_URL, "recipe_url": _RECIPE_URL,
        "provenance_pull": "Materials Project summary, 30 June 2022 (train_RF.py header)",
        "features": ref["features"],
        "normalisation": "x -> |x| / max|x| per column, over the full frame "
                         "(train_RF.py); domain is therefore the unit cube",
        "abs_normalisation_is_lossy": "efermi is negative for 491/6628 rows and "
                                      "folds onto its positive mirror",
        "feature_scale_max_abs": ref["scale"],
        "target": TARGET,
        "sign": "fn = -nu. ZoMBI MINIMISES Poisson's ratio and feeds its maximiser "
                "-1 * RF(x) (examples.ipynb); zhbench maximises, so fn is that same "
                "quantity. Larger fn = more auxetic.",
        "measured_rows": a["n_rows_used"],
        "n_negative_nu": a["n_negative_nu_used"],
        "frac_negative_nu": a["frac_negative_nu_used"],

        "emulation": (f"RandomForestRegressor({RF_TREES}) refit here from the frame; "
                      "fn(x) is a MODEL value everywhere. The shipped "
                      "poisson_RF_trained.pkl is sklearn 1.1.1 and cannot be loaded "
                      "(node dtype gained missing_go_to_left), so it is never used."),
        "surrogate_random_kfold_r2": cv["random_kfold"]["r2"],
        "surrogate_random_kfold_rmse": cv["random_kfold"]["rmse"],
        "surrogate_grouped_chemistry_r2": cv["grouped_by_chemistry"]["r2"],
        "surrogate_spearman": cv["random_kfold"]["spearman"],
        "surrogate_oob_r2": cv["oob_r2_full_fit"],
        "surrogate_r2_ceiling": cv["r2_ceiling_from_within_radius_spread"],
        "surrogate_cv_full": cv,

        "n_peaks_naive": ref["n_peaks_naive"],
        "n_peaks_supported": ref["n_peaks_supported"],
        "n_peaks_auxetic_support": ref["n_peaks_auxetic_support"],
        "peak_support": ref["peak_support"],
        "peak_support_rule": (f"a measured material within r={MATCH_RADIUS} whose "
                              f"-nu clears v - {PEAK_VALUE_TOL}*(v - background)"),
        "reference": "supported surrogate peaks; NOT experimentally validated optima",
        "grid": ref["grid"],

        "frac_peaks_above_random_p99": ref["contrast_uniform_cube"]["frac_peaks_above_probe_p99"],
        "contrast_uniform_cube": ref["contrast_uniform_cube"],
        "contrast_measured_support": ref["contrast_measured_support"],

        "frac_cube_within_r_of_measurement": a["frac_cube_within_r_of_measurement"],
        "median_cube_dist_to_measurement": a["median_cube_dist_to_measurement"],
        "within_r_median_abs_dnu": a["within_r_median_abs_dnu"],
        "nu_std": a["nu_std_used"],

        "caveats": [
            f"The published task is {DIM}-dimensional, not 6. train_RF.py takes "
            f"iloc[:, 2:-1] = {list(FEATURES)}; the shipped forest reports "
            f"n_features_in_ = 5; examples.ipynb sets dimensions = 5. dim=6 here "
            f"prepends nelements and is NOT the paper's task."
            + ("" if dim == DIM else "  THIS OBJECTIVE IS THE dim=6 VARIANT."),
            (f"{a['n_rows_nonphysical']} of {a['n_rows_total']} rows have a "
             f"physically impossible Poisson ratio (outside [-1, 0.5]); "
             f"{a['n_rows_nu_gt_1']} "
             f"exceed |nu| = 1. The extremes are elemental Dy at nu = {a['nu_min_all']:.2f} "
             f"and Ti3Ga at nu = {a['nu_max_all']:.2f}, which are Materials Project "
             "elastic-fit artefacts. Upstream trains on all of them; this build "
             + ("does too (target_range=None) and its landscape is one artefact spike."
                if ref["target_range"] is None else
                "drops them, which is a deliberate departure from the recipe.")),
            (f"Five scalars do not determine nu at this resolution: the median "
             f"material has {a['median_neighbours_within_r']:.0f} others within "
             f"r={MATCH_RADIUS}, differing in nu by a median of "
             f"{a['within_r_median_abs_dnu']:.4f} against a target std of "
             f"{a['nu_std_used']:.4f}. That within-radius spread caps ANY model on "
             f"these features at R^2 ~ {cv['r2_ceiling_from_within_radius_spread']:.3f}; "
             f"this forest reaches {cv['random_kfold']['r2']:.3f}."),
            (f"Of the {ref['n_peaks_supported']} supported peaks, "
             f"{ref['n_peaks_auxetic_support']} is/are vouched for by a material that "
             "is actually auxetic; the rest pass only because the support threshold "
             "v - value_tol*(v - background) is easy to clear for a peak that barely "
             "exceeds the background. meta['peak_support'] names the material behind "
             "each one so this is checkable rather than implied."),
            (f"Only {a['frac_cube_within_r_of_measurement']:.2%} of the unit cube is "
             f"within r={MATCH_RADIUS} of any measured material (median distance "
             f"{a['median_cube_dist_to_measurement']:.3f}), so metrics."
             "landscape_contrast probes almost entirely outside the data; "
             "contrast_measured_support is the honest counterpart."),
            "The forest is refit here, so fn is NOT numerically identical to the "
            "surface the ZoMBI paper optimised -- that surface is only recoverable "
            "by pinning sklearn 1.1.1. Feature set, normalisation, tree count and "
            "training rows are identical.",
        ],
        "verdict": _VERDICT,
    }
    return {
        "name": name,
        "dim": int(dim),
        "fn": lambda x, _p=predict_one: -_p(x),
        "true_optima": P,
        "true_values": V,
        "maximize": True,
        "domain": "cube",
        "meta": meta,
    }


_VERDICT = (
    "Marginal. Include only as a labelled EXTERNAL-SURROGATE objective, only in the "
    "physical variant, and never as the evidence for a multi-optimum claim. Four "
    "facts have to travel with any number it produces. (1) It is 5-D, not 6-D -- a "
    "'poisson6d' registry name is simply wrong. (2) The surrogate is weak for a "
    "reason no model can fix: materials the benchmark cannot tell apart (within "
    "r = 0.05) differ in Poisson ratio by 58% of the target's own standard "
    "deviation, capping R^2 near 0.36; the forest gets 0.15. (3) 99.92% of the cube "
    "has no material within r, so the reference optima are credible only after the "
    "support filter, and a uniform-probe contrast describes the forest's "
    "extrapolation, not the landscape -- which is why only 1 of the 8 supported "
    "peaks clears the uniform p99 while 3 clear the measured-slab p99. (4) The "
    "support filter leaves 8 peaks but only ONE with a genuinely auxetic material "
    "behind it (LiLa3, nu = -0.97, itself sitting against the edge of the physical "
    "window). This is closer to a single-needle problem with 7 marginal bumps than "
    "to the 20-optimum landscapes the ensemble suite provides. As published -- "
    "trained on the unfiltered target -- it is not a benchmark at all: CV R^2 is "
    "-23.6, and the reference set collapses to 1 peak."
)


def build(spec: dict) -> dict:
    """``objectives.build``-shaped entry point, if the registry grows a hook."""
    spec = dict(spec)
    spec.pop("kind", None)
    return poisson5d(**spec)


# --- CLI ---------------------------------------------------------------------

def _report(dim: int, target_range: tuple | None, rebuild: bool) -> None:
    ref = reference_set(dim, target_range, rebuild=rebuild)
    a, cv, g = ref["audit"], ref["cv"], ref["grid"]
    print(f"--- poisson{dim}d [{ref['tag']}]  features {'/'.join(ref['features'])}"
          f"  target {ref['target']}  domain cube")
    print(f"  rows {a['n_rows_used']}/{a['n_rows_total']} "
          f"(dropped {a['n_rows_dropped']}; {a['n_rows_nu_gt_1']} had |nu|>1); "
          f"nu in [{a['nu_min_used']:.3f}, {a['nu_max_used']:.3f}] std {a['nu_std_used']:.4f}; "
          f"{a['n_negative_nu_used']} negative ({a['frac_negative_nu_used']:.2%})")
    print(f"  feature max|x| scale: "
          + ", ".join(f"{n}={v:.4g}" for n, v in zip(ref['features'], ref['scale'])))
    print(f"  degeneracy: median {a['median_neighbours_within_r']:.0f} materials within "
          f"r={MATCH_RADIUS}; median |dnu| there {a['within_r_median_abs_dnu']:.4f}; "
          f"R^2 ceiling {a['implied_r2_ceiling']:.3f}")
    print(f"  support: {a['frac_cube_within_r_of_measurement']:.2%} of the cube is "
          f"within r of a material (median dist {a['median_cube_dist_to_measurement']:.3f})")
    for proto in ("random_kfold", "grouped_by_chemistry"):
        m = cv[proto]
        print(f"  CV {proto:22s} R2 {m['r2']:+.4f}  RMSE {m['rmse']:.4f}  "
              f"MAE {m['mae']:.4f}  Spearman {m['spearman']:+.4f}")
    print(f"  CV oob_r2 {cv['oob_r2_full_fit']:+.4f}  ({cv['n_groups']} chemical systems)")
    print(f"  grid {g['n_grid']} pts @ 1/{g['grid_n']}: surface [{g['surface_min']:.4f}, "
          f"{g['surface_max']:.4f}], background {g['background']:.4f}, "
          f"floor {g['prominence_floor']:.4f}, {g['n_local_max_cells']} local-max cells, "
          f"{g['n_above_prominence_floor']} above floor")
    print(f"  peaks: naive {ref['n_peaks_naive']} -> supported {ref['n_peaks_supported']}"
          f", of which {ref['n_peaks_auxetic_support']} are vouched for by a material "
          f"that is actually auxetic (nu < 0)")
    for i, s in enumerate(ref["peak_support"][:10]):
        print(f"    peak {i}  v={s['peak_value']:+.4f}  {s['n_materials_within_r']:3d} "
              f"materials within r  best nu={s['best_nu']:+.4f} ({s['best_formula']})")
    cu, cn, cs = (ref["contrast_uniform_cube"], ref["contrast_uniform_cube_naive_peaks"],
                  ref["contrast_measured_support"])
    print(f"  contrast vs uniform cube  : {cu['frac_peaks_above_probe_p99']:.3f} of "
          f"supported peaks > p99 (median {cu['probe_median']:.4f}, p99 {cu['probe_p99']:.4f}, "
          f"max {cu['probe_max']:.4f}); naive peaks {cn['frac_peaks_above_probe_p99']:.3f}")
    print(f"  contrast vs measured slab : {cs['frac_peaks_above_probe_p99']:.3f} "
          f"(median {cs['probe_median']:.4f}, p99 {cs['probe_p99']:.4f})")
    print(f"  built in {ref['build_seconds']:.1f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ZoMBI negative-Poisson task -> poisson5d")
    ap.add_argument("--fetch", action="store_true",
                    help="download the frame and transcode it to a pymatgen-free CSV")
    ap.add_argument("--from-mp", action="store_true",
                    help="re-pull from Materials Project (needs MP_API_KEY; will NOT "
                         "reproduce the 30 June 2022 snapshot)")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--retrain", action="store_true",
                    help="fit the 500-tree forest and cache it under data/poisson/")
    ap.add_argument("--report", action="store_true",
                    help="audit, CV, peaks and contrast")
    ap.add_argument("--rebuild", action="store_true", help="ignore the cached reference")
    ap.add_argument("--force", action="store_true", help="re-download / re-transcode")
    ap.add_argument("--dim", type=int, default=DIM, choices=(5, 6))
    ap.add_argument("--variant", default="physical",
                    choices=("physical", "asPublished", "both"),
                    help="'asPublished' keeps the 181 impossible-nu rows train_RF.py "
                         "keeps; 'physical' drops them (default)")
    args = ap.parse_args()

    ranges = ({"physical": [PHYSICAL_RANGE], "asPublished": [None],
               "both": [PHYSICAL_RANGE, None]})[args.variant]

    if args.fetch:
        for k, v in fetch(force=args.force, from_mp=args.from_mp,
                          api_key=args.api_key).items():
            print(f"{k}: {v}")
    if args.retrain:
        for tr in ranges:
            _, info = train_rf(args.dim, tr, cache=True)
            print(f"retrained [{_tag(tr)}] n_train={info['n_train']} "
                  f"trees={info['n_estimators']} oob_r2={info['oob_r2']:+.4f} "
                  f"fit={info['fit_seconds']:.1f}s -> {info.get('cached_at', 'cache hit')}")
    if args.report:
        for tr in ranges:
            _report(args.dim, tr, args.rebuild)
    if not (args.fetch or args.retrain or args.report):
        ap.print_help()
