"""
ELA feature vector for simplex landscapes (Muñoz 8 + ZoMBI 2).

Protocol
--------
* Characterize the **RF surrogate** on a fixed dense simplex sample (N = dim × 1000).
* Meta-model / classification features use **Helmert ILR** coordinates.
* Distribution features (H, gamma, PKS) use objective values Y only.
* OOB R² uses **measured campaign rows** (sparse).
* Median Lipschitz uses the dense sample at composition L₂ radius 0.064.
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PolynomialFeatures

ROOT = Path(__file__).resolve().parent.parent

DB_COMP_COLS = ["FAPbI3", "MAPbI3", "MAPbBr3"]
DEFAULT_OBJECTIVE = "Objective"
DEFAULT_INPUT_NOISE = 0.064
RF_N_ESTIMATORS = 500
DIST_BAND = (0.5, 1.5)

FEATURE_NAMES: tuple[str, ...] = (
    "R2_Q",
    "CN",
    "H_Y",
    "xi_1",
    "gamma_Y",
    "EL25",
    "LQ25",
    "PKS",
    "oob_r2",
    "median_lipschitz",
)

# Spatial roughness extras (also registered for fitness via tier1.ALLOWED_FITNESS_NAMES).
SPATIAL_LIP_NAMES: tuple[str, ...] = (
    "interior_median_lipschitz",
    "lipschitz_p20",
    "lipschitz_p80",
    "edge_interior_lip_ratio",
    "ma_region_median_lipschitz",
    "fa_region_median_lipschitz",
    "br_region_median_lipschitz",
    "local_lipschitz_cv",
    "tile_min_median_lipschitz",
)

# Composition column indices in DB_COMP_COLS / dense X: FAPbI3, MAPbI3, MAPbBr3.
_COMP_FA = 0
_COMP_MA = 1
_COMP_BR = 2
# Region mask: fraction of that component ≥ this → "near that vertex lobe".
REGION_COMP_MIN = 0.40
# Barycentric tile count per edge for tile_min_median_lipschitz.
TILE_N_BINS = 4
TILE_MIN_POINTS = 12


def composition_to_ilr(x: np.ndarray) -> np.ndarray:
    """Helmert ILR; matches ``synthetic_data.oracles.composition_to_ilr_np``."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    eps = 1e-10
    log_x = np.log(x + eps)
    d = x.shape[1]
    out = np.empty((x.shape[0], d - 1), dtype=float)
    for i in range(d - 1):
        coef = math.sqrt((i + 1) / (i + 2))
        term1 = log_x[:, : i + 1].sum(axis=1) / (i + 1)
        term2 = log_x[:, i + 1]
        out[:, i] = coef * (term1 - term2)
    return out


def ilr_to_composition(ilr: np.ndarray, d: int) -> np.ndarray:
    """Inverse Helmert ILR."""
    ilr = np.asarray(ilr, dtype=float)
    if ilr.ndim == 1:
        ilr = ilr.reshape(1, -1)
    n = ilr.shape[0]
    log_x = np.zeros((n, d), dtype=float)
    for i in range(d - 1):
        coef = math.sqrt((i + 1) / (i + 2))
        contribution = ilr[:, i] * coef
        log_x[:, : i + 1] += contribution[:, None] / (i + 1)
        log_x[:, i + 1] -= contribution
    x = np.exp(log_x)
    return x / x.sum(axis=1, keepdims=True)


def _resolve_db_path(db_arg: str | Path) -> Path:
    p = Path(db_arg)
    if p.is_file():
        return p
    candidate = ROOT / "data" / db_arg
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Database not found: {db_arg}")


def _infer_comp_cols_from_names(names: list[str], explicit: list[str] | None) -> list[str]:
    """Resolve composition columns: explicit, FA/MA/Br campaign, or Comp1..CompN."""
    if explicit:
        missing = [c for c in explicit if c not in names]
        if missing:
            raise ValueError(f"composition columns missing: {missing}")
        return list(explicit)
    if all(c in names for c in DB_COMP_COLS):
        return list(DB_COMP_COLS)
    comps = [c for c in names if c.startswith("Comp")]
    comps_sorted: list[str] = []
    i = 1
    while f"Comp{i}" in names:
        comps_sorted.append(f"Comp{i}")
        i += 1
    if len(comps_sorted) >= 2:
        return comps_sorted
    if len(comps) >= 2:
        return sorted(comps)
    raise ValueError(
        f"Could not infer composition columns from {names}; "
        f"expected {DB_COMP_COLS} or Comp1..CompN."
    )


def load_campaign_rows(
    db_path: str | Path,
    *,
    comp_cols: list[str] | None = None,
    objective_column: str = DEFAULT_OBJECTIVE,
) -> tuple[np.ndarray, np.ndarray]:
    """Load measured compositions and objective from a results DB or campaign CSV."""
    db_path = Path(db_path)
    if db_path.suffix.lower() == ".csv":
        import csv

        with open(db_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise RuntimeError(f"Empty CSV: {db_path}")
            names = list(reader.fieldnames)
            resolved = _infer_comp_cols_from_names(names, comp_cols)
            if objective_column not in names:
                raise ValueError(f"{objective_column!r} missing from {db_path}")
            rows = []
            for row in reader:
                try:
                    xvals = [float(row[c]) for c in resolved]
                    yval = float(row[objective_column])
                except (TypeError, ValueError):
                    continue
                if any(v != v for v in xvals + [yval]):  # NaN check
                    continue
                rows.append(xvals + [yval])
        if not rows:
            raise RuntimeError(f"No complete rows in {db_path}")
        arr = np.asarray(rows, dtype=float)
        x = arr[:, :-1]
        y = arr[:, -1]
        s = x.sum(axis=1, keepdims=True)
        x = x / np.where(s == 0, 1.0, s)
        return x, y

    resolved = comp_cols or list(DB_COMP_COLS)
    cols = resolved + [objective_column]
    con = sqlite3.connect(str(db_path))
    try:
        # Prefer explicit columns; if FA/MA/Br missing, try Comp* from schema.
        table_info = con.execute("PRAGMA table_info(results)").fetchall()
        schema_names = [r[1] for r in table_info]
        if schema_names and not all(c in schema_names for c in resolved):
            resolved = _infer_comp_cols_from_names(schema_names, comp_cols)
            cols = resolved + [objective_column]
        sel = ", ".join(f'"{c}"' for c in cols)
        where = " AND ".join(f'"{c}" IS NOT NULL' for c in cols)
        try:
            rows = con.execute(f"SELECT {sel} FROM results WHERE {where}").fetchall()
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc).lower():
                raise
            tables = [
                r[0]
                for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"
                ).fetchall()
            ]
            raise RuntimeError(
                f"{db_path.resolve()} has no 'results' table (found: {tables or 'none'}; "
                f"size={db_path.stat().st_size} bytes). "
                "Allowlisted campaign DBs live under data/ (see .gitignore). Sync with:\n"
                "  rsync -av --progress data/2nd_real_run.db "
                "eve_lal@login007:~/orcd/scratch/ZoMBI-Hop/data/"
            ) from exc
    finally:
        con.close()
    arr = np.asarray(rows, dtype=float)
    if arr.shape[0] == 0:
        raise RuntimeError(f"No complete rows in {db_path}")
    x = arr[:, : len(resolved)]
    y = arr[:, len(resolved)]
    s = x.sum(axis=1, keepdims=True)
    x = x / np.where(s == 0, 1.0, s)
    return x, y


def train_rf_surrogate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = RF_N_ESTIMATORS,
    random_state: int = 42,
) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=1,
        random_state=random_state,
    )
    rf.fit(x, y)
    return rf


def rf_transform_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    *,
    n_estimators: int = RF_N_ESTIMATORS,
    random_state: int = 42,
    return_oob: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Fit RF on ``(x_train, y_train)`` and predict on ``x_query``.

    Used for ELA(RF_g): sample the evolved expression, fit an RF, then
    evaluate ELA features on the RF surface (same family as λ_T).

    When ``return_oob`` is True, also return the RF out-of-bag R².
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=1,
        random_state=random_state,
        oob_score=bool(return_oob),
        bootstrap=True,
    )
    rf.fit(x_train, y_train)
    pred = np.asarray(rf.predict(x_query), dtype=float).ravel()
    if return_oob:
        return pred, float(rf.oob_score_)
    return pred


def load_campaign_rf(
    db_path: str | Path,
    *,
    objective_column: str = DEFAULT_OBJECTIVE,
) -> tuple[RandomForestRegressor, np.ndarray, np.ndarray]:
    """Measured rows + fitted RF (in-sample; use oob_r2 separately)."""
    x, y = load_campaign_rows(db_path, objective_column=objective_column)
    return train_rf_surrogate(x, y), x, y


def sample_simplex_sobol(dim: int, n: int, *, seed: int = 42) -> np.ndarray:
    """Low-discrepancy uniform sample on ``dim``-simplex via ILR box mapping."""
    from scipy.stats import qmc

    ilr_dim = dim - 1
    # Map Sobol [0,1]^ilr_dim through a bounded ILR box learned from Dirichlet spread.
    rng = np.random.default_rng(seed)
    probe = composition_to_ilr(rng.dirichlet(np.ones(dim), size=5000))
    lo = probe.min(axis=0) - 0.25
    hi = probe.max(axis=0) + 0.25
    engine = qmc.Sobol(d=ilr_dim, scramble=True, seed=seed)
    u = engine.random(n)
    z = lo + u * (hi - lo)
    return ilr_to_composition(z, dim)


def _adjusted_r2(y: np.ndarray, y_hat: np.ndarray, n_params: int) -> float:
    n = len(y)
    if n <= n_params + 1:
        return float("nan")
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-15:
        return 0.0
    r2 = 1.0 - ss_res / ss_tot
    return 1.0 - (1.0 - r2) * (n - 1) / max(n - n_params - 1, 1)


def feature_r2_q(z: np.ndarray, y: np.ndarray) -> float:
    """Adjusted R² of a purely quadratic model (ILR coords, no linear terms)."""
    d = z.shape[1]
    cols = [z[:, i] ** 2 for i in range(d)]
    # pure quadratic main effects only (Muñoz ``R2_Q``)
    xq = np.column_stack(cols)
    reg = LinearRegression(fit_intercept=True)
    reg.fit(xq, y)
    y_hat = reg.predict(xq)
    return _adjusted_r2(y, y_hat, n_params=d + 1)


def feature_cn(z: np.ndarray, y: np.ndarray) -> float:
    """Condition number proxy: min/max |quadratic coefficients|."""
    d = z.shape[1]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    xp = poly.fit_transform(z)
    names = poly.get_feature_names_out([f"z{i}" for i in range(d)])
    quad_idx = [i for i, nm in enumerate(names) if "^2" in nm or " " in nm]
    if not quad_idx:
        return 1.0
    reg = LinearRegression(fit_intercept=True)
    reg.fit(xp[:, quad_idx], y)
    coefs = np.abs(reg.coef_)
    cmax = float(np.max(coefs))
    cmin = float(np.min(coefs))
    if cmax < 1e-15:
        return 1.0
    return cmin / cmax


def feature_h_y(y: np.ndarray, *, n_bins: int = 32) -> float:
    """Entropy of binned objective distribution."""
    y = np.asarray(y, dtype=float).ravel()
    y = y[np.isfinite(y)]
    if y.size < 2:
        return 0.0
    y_min = float(y.min())
    y_max = float(y.max())
    if not np.isfinite(y_min) or not np.isfinite(y_max) or (y_max - y_min) < 1e-15:
        return 0.0
    counts, _ = np.histogram(y, bins=n_bins, range=(y_min, y_max))
    counts = counts.astype(float)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def feature_gamma_y(y: np.ndarray) -> float:
    return float(stats.skew(y))


def feature_xi_1(z: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> float:
    """
    First-order entropic significance (Seo & Moon style proxy).

    Average normalized mutual information between each ILR coordinate and
    binned objective levels.
    """
    y_disc = np.digitize(y, np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1]))
    h_y = feature_h_y(y_disc.astype(float), n_bins=n_bins)
    if h_y < 1e-12:
        return 0.0
    scores: list[float] = []
    for j in range(z.shape[1]):
        z_disc = np.digitize(z[:, j], np.quantile(z[:, j], np.linspace(0, 1, n_bins + 1)[1:-1]))
        # MI(Y; Z_j) via contingency table
        contingency = np.histogram2d(y_disc, z_disc, bins=[n_bins, n_bins])[0]
        pxy = contingency / max(contingency.sum(), 1)
        px = pxy.sum(axis=1)
        py = pxy.sum(axis=0)
        mi = 0.0
        for i in range(px.shape[0]):
            for k in range(py.shape[0]):
                if pxy[i, k] <= 0:
                    continue
                mi += pxy[i, k] * math.log(pxy[i, k] / (px[i] * py[k] + 1e-15) + 1e-15)
        scores.append(max(mi / h_y, 0.0))
    return float(np.mean(scores))


def _class_labels_top_fraction(y: np.ndarray, frac: float, *, maximize: bool) -> np.ndarray:
    """Binary labels: top ``frac`` objective values (maximize) or bottom (minimize)."""
    n_good = max(1, int(round(len(y) * frac)))
    order = np.argsort(y)
    labels = np.zeros(len(y), dtype=int)
    if maximize:
        labels[order[-n_good:]] = 1
    else:
        labels[order[:n_good]] = 1
    return labels


def feature_el25(z: np.ndarray, y: np.ndarray, *, maximize: bool = True) -> float:
    """LDA cross-validated accuracy at top-25% threshold."""
    labels = _class_labels_top_fraction(y, 0.25, maximize=maximize)
    if labels.sum() in (0, len(labels)):
        return 0.5
    lda = LinearDiscriminantAnalysis()
    try:
        scores = cross_val_score(lda, z, labels, cv=5, scoring="accuracy")
    except Exception:
        return 0.5
    return float(np.mean(scores))


def feature_eq25(z: np.ndarray, y: np.ndarray, *, maximize: bool = True) -> float:
    labels = _class_labels_top_fraction(y, 0.25, maximize=maximize)
    if labels.sum() in (0, len(labels)):
        return 0.5
    qda = QuadraticDiscriminantAnalysis()
    try:
        scores = cross_val_score(qda, z, labels, cv=5, scoring="accuracy")
    except Exception:
        return 0.5
    return float(np.mean(scores))


def feature_lq25(el25: float, eq25: float) -> float:
    if eq25 < 1e-12:
        return float("nan")
    return el25 / eq25


def feature_pks(y: np.ndarray, *, n_bins: int = 64) -> float:
    """Number of peaks in the binned objective histogram."""
    y = np.asarray(y, dtype=float).ravel()
    y = y[np.isfinite(y)]
    if y.size < 2:
        return 0.0
    y_min = float(y.min())
    y_max = float(y.max())
    if not np.isfinite(y_min) or not np.isfinite(y_max) or (y_max - y_min) < 1e-15:
        return 0.0
    counts, _ = np.histogram(y, bins=n_bins, range=(y_min, y_max))
    peaks, _ = find_peaks(counts)
    return float(len(peaks))


def feature_oob_r2(x: np.ndarray, y: np.ndarray, *, n_estimators: int = RF_N_ESTIMATORS) -> float:
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=1,
        random_state=42,
        oob_score=True,
        bootstrap=True,
    )
    rf.fit(x, y)
    return float(rf.oob_score_)


# Spatial roughness diagnostics (fitness extras). Interior = min(composition) ≥ this.
INTERIOR_MIN_COMP = 0.10
EDGE_MAX_COMP = 0.05


def pairwise_lipschitz_slopes(
    x: np.ndarray,
    y: np.ndarray,
    *,
    input_noise: float = DEFAULT_INPUT_NOISE,
    knn_fallback: int = 5,
) -> np.ndarray:
    """|Δy|/Δx over unique composition-L₂ neighbor pairs within ``input_noise``.

    If the radius graph is empty (sparse samples), fall back to ``knn_fallback``
    nearest neighbors so interior-only subsets still yield a defined median.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = x.shape[0]
    if n < 2:
        return np.asarray([], dtype=float)
    nn = NearestNeighbors(radius=input_noise, metric="euclidean", n_jobs=1)
    nn.fit(x)
    neighbors = nn.radius_neighbors(x, return_distance=True)
    slopes: list[float] = []
    for i in range(n):
        for j, dist in zip(neighbors[1][i], neighbors[0][i]):
            if j <= i or dist <= 1e-12:
                continue
            slopes.append(abs(float(y[i] - y[j])) / float(dist))
    if slopes:
        return np.asarray(slopes, dtype=float)
    k = int(min(max(knn_fallback, 1) + 1, n))
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=1)
    knn.fit(x)
    dists, idxs = knn.kneighbors(x, return_distance=True)
    for i in range(n):
        for j, dist in zip(idxs[i], dists[i]):
            if int(j) <= i or dist <= 1e-12:
                continue
            slopes.append(abs(float(y[i] - y[j])) / float(dist))
    return np.asarray(slopes, dtype=float)


def per_point_local_lipschitz(
    x: np.ndarray,
    y: np.ndarray,
    *,
    input_noise: float = DEFAULT_INPUT_NOISE,
) -> np.ndarray:
    """Per-point median |Δy|/Δx to neighbors within ``input_noise`` (NaN if none)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if n < 2:
        return out
    nn = NearestNeighbors(radius=input_noise, metric="euclidean", n_jobs=1)
    nn.fit(x)
    dists, idxs = nn.radius_neighbors(x, return_distance=True)
    for i in range(n):
        slopes: list[float] = []
        for j, dist in zip(idxs[i], dists[i]):
            if int(j) == i or dist <= 1e-12:
                continue
            slopes.append(abs(float(y[i] - y[j])) / float(dist))
        if slopes:
            out[i] = float(np.median(slopes))
    return out


def feature_median_lipschitz(
    x: np.ndarray,
    y: np.ndarray,
    *,
    input_noise: float = DEFAULT_INPUT_NOISE,
) -> float:
    """Median |Δy|/Δx over composition-L₂ neighbor pairs within ``input_noise``."""
    slopes = pairwise_lipschitz_slopes(x, y, input_noise=input_noise)
    if slopes.size == 0:
        return float("nan")
    return float(np.median(slopes))


def feature_lipschitz_percentile(
    x: np.ndarray,
    y: np.ndarray,
    q: float,
    *,
    input_noise: float = DEFAULT_INPUT_NOISE,
) -> float:
    """Percentile of pairwise Lipschitz slopes (q in 0..100)."""
    slopes = pairwise_lipschitz_slopes(x, y, input_noise=input_noise)
    if slopes.size == 0:
        return float("nan")
    return float(np.percentile(slopes, q))


def feature_interior_median_lipschitz(
    x: np.ndarray,
    y: np.ndarray,
    *,
    interior_min: float = INTERIOR_MIN_COMP,
    input_noise: float = DEFAULT_INPUT_NOISE,
) -> float:
    """Median Lipschitz using only interior points (``min(x_i) ≥ interior_min``)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    mask = x.min(axis=1) >= float(interior_min)
    if int(mask.sum()) < 8:
        return float("nan")
    return feature_median_lipschitz(x[mask], y[mask], input_noise=input_noise)


def feature_edge_interior_lip_ratio(
    x: np.ndarray,
    y: np.ndarray,
    *,
    interior_min: float = INTERIOR_MIN_COMP,
    edge_max: float = EDGE_MAX_COMP,
    input_noise: float = DEFAULT_INPUT_NOISE,
) -> float:
    """
    Ratio of near-edge to interior per-point median Lipschitz.

    Values ≫ 1 indicate edge-localized roughness (GP cheat mode vs RF ~1–1.5).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    local = per_point_local_lipschitz(x, y, input_noise=input_noise)
    interior = x.min(axis=1) >= float(interior_min)
    edge = x.min(axis=1) < float(edge_max)
    mi = float(np.nanmedian(local[interior])) if np.any(interior) else float("nan")
    me = float(np.nanmedian(local[edge])) if np.any(edge) else float("nan")
    if not np.isfinite(mi) or not np.isfinite(me) or mi < 1e-12:
        return float("nan")
    return float(me / mi)


def feature_region_median_lipschitz(
    x: np.ndarray,
    y: np.ndarray,
    *,
    comp_index: int,
    comp_min: float = REGION_COMP_MIN,
    input_noise: float = DEFAULT_INPUT_NOISE,
) -> float:
    """Median Lipschitz on the lobe where composition[:, comp_index] ≥ comp_min."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if x.ndim != 2 or x.shape[1] <= int(comp_index):
        return float("nan")
    mask = x[:, int(comp_index)] >= float(comp_min)
    if int(mask.sum()) < 8:
        return float("nan")
    return feature_median_lipschitz(x[mask], y[mask], input_noise=input_noise)


def feature_local_lipschitz_cv(
    x: np.ndarray,
    y: np.ndarray,
    *,
    interior_min: float = INTERIOR_MIN_COMP,
    input_noise: float = DEFAULT_INPUT_NOISE,
) -> float:
    """
    Coefficient of variation (std/mean) of per-point local Lipschitz in the interior.

    Low values ≈ spatially uniform grain; high values ≈ roughness concentrated in bands.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    local = per_point_local_lipschitz(x, y, input_noise=input_noise)
    interior = x.min(axis=1) >= float(interior_min)
    vals = local[interior]
    vals = vals[np.isfinite(vals)]
    if vals.size < 8:
        return float("nan")
    mean = float(np.mean(vals))
    if mean < 1e-12:
        return float("nan")
    return float(np.std(vals) / mean)


def feature_tile_min_median_lipschitz(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = TILE_N_BINS,
    min_points: int = TILE_MIN_POINTS,
    input_noise: float = DEFAULT_INPUT_NOISE,
) -> float:
    """
    Minimum median Lipschitz across barycentric composition tiles.

    Quiet tiles pull this down, so midline/edge stripes that leave lobes smooth fail
    even when the global median Lip looks fine.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if x.ndim != 2 or x.shape[0] < min_points or x.shape[1] < 3:
        return float("nan")
    n_bins = max(int(n_bins), 2)
    qx = np.floor(np.clip(x[:, :3], 0.0, 1.0 - 1e-12) * n_bins).astype(int)
    # Encode (fa_bin, ma_bin, br_bin) as a single tile id.
    stride = n_bins + 1
    tile_ids = qx[:, 0] * stride * stride + qx[:, 1] * stride + qx[:, 2]
    medians: list[float] = []
    for tid in np.unique(tile_ids):
        mask = tile_ids == tid
        if int(mask.sum()) < int(min_points):
            continue
        med = feature_median_lipschitz(x[mask], y[mask], input_noise=input_noise)
        if np.isfinite(med):
            medians.append(float(med))
    if not medians:
        return float("nan")
    return float(min(medians))


def compute_spatial_lipschitz_features(
    x: np.ndarray,
    y: np.ndarray,
    *,
    input_noise: float = DEFAULT_INPUT_NOISE,
) -> dict[str, float]:
    """Bundle of spatial-roughness diagnostics derived from one neighbor pass."""
    slopes = pairwise_lipschitz_slopes(x, y, input_noise=input_noise)
    local = per_point_local_lipschitz(x, y, input_noise=input_noise)
    x = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    interior = x.min(axis=1) >= INTERIOR_MIN_COMP
    edge = x.min(axis=1) < EDGE_MAX_COMP

    if slopes.size == 0:
        med = p20 = p80 = float("nan")
    else:
        med = float(np.median(slopes))
        p20 = float(np.percentile(slopes, 20))
        p80 = float(np.percentile(slopes, 80))

    if int(interior.sum()) >= 8:
        interior_med = feature_median_lipschitz(
            x[interior], y_arr[interior],
            input_noise=input_noise,
        )
    else:
        interior_med = float("nan")

    mi = float(np.nanmedian(local[interior])) if np.any(interior) else float("nan")
    me = float(np.nanmedian(local[edge])) if np.any(edge) else float("nan")
    if np.isfinite(mi) and np.isfinite(me) and mi >= 1e-12:
        ratio = float(me / mi)
    else:
        ratio = float("nan")

    interior_local = local[interior]
    interior_local = interior_local[np.isfinite(interior_local)]
    if interior_local.size >= 8:
        mean_loc = float(np.mean(interior_local))
        local_cv = (
            float(np.std(interior_local) / mean_loc)
            if mean_loc >= 1e-12
            else float("nan")
        )
    else:
        local_cv = float("nan")

    return {
        "median_lipschitz": med,
        "interior_median_lipschitz": float(interior_med),
        "lipschitz_p20": p20,
        "lipschitz_p80": p80,
        "edge_interior_lip_ratio": ratio,
        "ma_region_median_lipschitz": feature_region_median_lipschitz(
            x, y_arr, comp_index=_COMP_MA, input_noise=input_noise,
        ),
        "fa_region_median_lipschitz": feature_region_median_lipschitz(
            x, y_arr, comp_index=_COMP_FA, input_noise=input_noise,
        ),
        "br_region_median_lipschitz": feature_region_median_lipschitz(
            x, y_arr, comp_index=_COMP_BR, input_noise=input_noise,
        ),
        "local_lipschitz_cv": local_cv,
        "tile_min_median_lipschitz": feature_tile_min_median_lipschitz(
            x, y_arr, input_noise=input_noise,
        ),
    }


def compute_feature_vector(
    z: np.ndarray,
    y: np.ndarray,
    x_comp: np.ndarray,
    *,
    x_campaign: np.ndarray,
    y_campaign: np.ndarray,
    maximize: bool = True,
    input_noise: float = DEFAULT_INPUT_NOISE,
) -> dict[str, float]:
    """
    Compute all 10 features.

    ``z``, ``y``, ``x_comp`` — dense surrogate sample (ILR, objective, compositions).
    ``x_campaign``, ``y_campaign`` — measured campaign rows for OOB R².
    """
    el25 = feature_el25(z, y, maximize=maximize)
    eq25 = feature_eq25(z, y, maximize=maximize)
    lip = compute_spatial_lipschitz_features(x_comp, y, input_noise=input_noise)
    return {
        "R2_Q": feature_r2_q(z, y),
        "CN": feature_cn(z, y),
        "H_Y": feature_h_y(y),
        "xi_1": feature_xi_1(z, y),
        "gamma_Y": feature_gamma_y(y),
        "EL25": el25,
        "LQ25": feature_lq25(el25, eq25),
        "PKS": feature_pks(y),
        "oob_r2": feature_oob_r2(x_campaign, y_campaign),
        **lip,
    }


def characterize_campaign_surrogate(
    db_path: str | Path,
    *,
    objective_column: str = DEFAULT_OBJECTIVE,
    comp_cols: list[str] | None = None,
    maximize: bool = True,
    sample_seed: int = 42,
    n_samples: int | None = None,
    full: bool = False,
    return_model: bool = False,
) -> dict[str, Any]:
    """Full λ_T pipeline for a campaign DB/CSV (any simplex dim).

    Trains the deterministic RF surrogate on measured rows, evaluates it on a
    dense Sobol simplex sample, and returns ELA features plus the RF y-range
    used to affine-scale evolved GP trees (same recipe as ``2nd_real_run``).
    """
    db_path = _resolve_db_path(db_path)
    x_campaign, y_campaign = load_campaign_rows(
        db_path, objective_column=objective_column, comp_cols=comp_cols,
    )
    dim = x_campaign.shape[1]
    target_n = dim * 1000
    n_dense = n_samples if n_samples is not None else (1 << (target_n - 1).bit_length())
    rf = train_rf_surrogate(x_campaign, y_campaign)
    x_dense = sample_simplex_sobol(dim, n_dense, seed=sample_seed)
    y_dense = rf.predict(x_dense)
    z_dense = composition_to_ilr(x_dense)

    if full:
        from ela.pflacco_port import compute_all_ela_feature_groups

        feature_groups = compute_all_ela_feature_groups(
            z_dense, y_dense, x_dense,
            x_campaign=x_campaign, y_campaign=y_campaign,
            maximize=maximize, dim=dim, seed=sample_seed,
        )
        features = feature_groups["munoz_33"]
    else:
        feature_groups = None
        features = compute_feature_vector(
            z_dense, y_dense, x_dense,
            x_campaign=x_campaign, y_campaign=y_campaign,
            maximize=maximize,
        )

    result: dict[str, Any] = {
        "db_path": str(db_path),
        "objective_column": objective_column,
        "maximize": maximize,
        "dim": dim,
        "n_campaign": int(x_campaign.shape[0]),
        "n_dense_sample": int(n_dense),
        "sample_seed": sample_seed,
        "input_noise": DEFAULT_INPUT_NOISE,
        "y_campaign_range": [float(y_campaign.min()), float(y_campaign.max())],
        "y_dense_range": [float(y_dense.min()), float(y_dense.max())],
        "features": features,
    }
    if feature_groups is not None:
        result["feature_groups"] = feature_groups
        result["n_features_total"] = sum(
            len(v) for v in feature_groups.values() if isinstance(v, dict)
        )
    if return_model:
        result["_rf"] = rf
        result["_x_campaign"] = x_campaign
        result["_y_campaign"] = y_campaign
        result["_x_dense"] = x_dense
        result["_y_dense"] = y_dense
    return result


def save_lambda_target(result: dict[str, Any], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return _sanitize(obj.tolist())
        return obj

    with open(out, "w", encoding="utf-8") as f:
        json.dump(_sanitize(result), f, indent=2, allow_nan=False)
        f.write("\n")
    return out
