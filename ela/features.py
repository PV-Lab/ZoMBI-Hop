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


def load_campaign_rows(
    db_path: str | Path,
    *,
    comp_cols: list[str] | None = None,
    objective_column: str = DEFAULT_OBJECTIVE,
) -> tuple[np.ndarray, np.ndarray]:
    """Load measured compositions and objective from a results DB."""
    comp_cols = comp_cols or list(DB_COMP_COLS)
    cols = comp_cols + [objective_column]
    con = sqlite3.connect(str(db_path))
    try:
        sel = ", ".join(f'"{c}"' for c in cols)
        where = " AND ".join(f'"{c}" IS NOT NULL' for c in cols)
        rows = con.execute(f"SELECT {sel} FROM results WHERE {where}").fetchall()
    finally:
        con.close()
    arr = np.asarray(rows, dtype=float)
    if arr.shape[0] == 0:
        raise RuntimeError(f"No complete rows in {db_path}")
    x = arr[:, : len(comp_cols)]
    y = arr[:, len(comp_cols)]
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
    counts, _ = np.histogram(y, bins=n_bins)
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
    counts, _ = np.histogram(y, bins=n_bins)
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


def feature_median_lipschitz(
    x: np.ndarray,
    y: np.ndarray,
    *,
    input_noise: float = DEFAULT_INPUT_NOISE,
) -> float:
    """Median |Δy|/Δx over composition-L₂ neighbor pairs within ``input_noise``."""
    n = x.shape[0]
    if n < 2:
        return float("nan")
    nn = NearestNeighbors(radius=input_noise, metric="euclidean", n_jobs=1)
    nn.fit(x)
    neighbors = nn.radius_neighbors(x, return_distance=True)
    slopes: list[float] = []
    for i in range(n):
        for j, dist in zip(neighbors[1][i], neighbors[0][i]):
            if j <= i or dist <= 1e-12:
                continue
            slopes.append(abs(float(y[i] - y[j])) / float(dist))
    if not slopes:
        return float("nan")
    return float(np.median(slopes))


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
        "median_lipschitz": feature_median_lipschitz(
            x_comp, y, input_noise=input_noise,
        ),
    }


def characterize_campaign_surrogate(
    db_path: str | Path,
    *,
    objective_column: str = DEFAULT_OBJECTIVE,
    maximize: bool = True,
    sample_seed: int = 42,
    n_samples: int | None = None,
    full: bool = False,
) -> dict[str, Any]:
    """Full λ_T pipeline for a 3D (or general d) campaign DB."""
    db_path = _resolve_db_path(db_path)
    x_campaign, y_campaign = load_campaign_rows(
        db_path, objective_column=objective_column,
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
    return result


def save_lambda_target(result: dict[str, Any], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return out
