"""Port of core pflacco classical ELA calculators (numpy-only, no pflacco dependency)."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde, pearsonr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier


def _as_df(X: np.ndarray) -> pd.DataFrame:
    X = np.asarray(X, dtype=float)
    cols = [f"x{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols)


def calculate_ela_meta(X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Match ``pflacco.classical_ela_features.calculate_ela_meta``."""
    t0 = time.monotonic()
    X = _as_df(X)
    y = pd.Series(np.asarray(y, dtype=float).ravel())

    model = LinearRegression().fit(X, y)
    lin_simple_adj_r2 = 1 - (1 - model.score(X, y)) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
    lin_simple_intercept = float(model.intercept_)
    lin_simple_coef_min = float(np.abs(model.coef_).min())
    lin_simple_coef_max = float(np.abs(model.coef_).max())
    lin_simple_coef_max_by_min = lin_simple_coef_max / max(lin_simple_coef_min, 1e-15)

    poly = PolynomialFeatures(interaction_only=True, include_bias=False)
    x_int = poly.fit_transform(X)
    m_int = LinearRegression().fit(x_int, y)
    lin_w_interact_adj_r2 = 1 - (1 - m_int.score(x_int, y)) * (len(y) - 1) / (len(y) - x_int.shape[1] - 1)

    x_sq = pd.concat([X, X.pow(2).add_suffix("^2")], axis=1)
    m_quad = LinearRegression().fit(x_sq, y)
    quad_simple_adj_r2 = 1 - (1 - m_quad.score(x_sq, y)) * (len(y) - 1) / (len(y) - x_sq.shape[1] - 1)
    qcoef = np.abs(m_quad.coef_[int(x_sq.shape[1] / 2) :])
    quad_simple_cond = float(qcoef.max() / max(qcoef.min(), 1e-15))

    x_arr = x_sq.to_numpy()
    for idx in range(len(x_sq.columns)):
        tmp = idx + 1
        while tmp < len(x_sq.columns):
            x_arr = np.hstack((x_arr, (x_arr[:, idx] * x_arr[:, tmp]).reshape(-1, 1)))
            tmp += 1
    m_qi = LinearRegression().fit(x_arr, y)
    quad_w_interact_adj_r2 = 1 - (1 - m_qi.score(x_arr, y)) * (len(y) - 1) / (len(y) - x_arr.shape[1] - 1)

    return {
        "ela_meta.lin_simple.adj_r2": float(lin_simple_adj_r2),
        "ela_meta.lin_simple.intercept": lin_simple_intercept,
        "ela_meta.lin_simple.coef.min": lin_simple_coef_min,
        "ela_meta.lin_simple.coef.max": lin_simple_coef_max,
        "ela_meta.lin_simple.coef.max_by_min": float(lin_simple_coef_max_by_min),
        "ela_meta.lin_w_interact.adj_r2": float(lin_w_interact_adj_r2),
        "ela_meta.quad_simple.adj_r2": float(quad_simple_adj_r2),
        "ela_meta.quad_simple.cond": quad_simple_cond,
        "ela_meta.quad_w_interact.adj_r2": float(quad_w_interact_adj_r2),
        "ela_meta.costs_runtime": time.monotonic() - t0,
    }


def calculate_ela_distribution(y: np.ndarray) -> dict[str, float]:
    t0 = time.monotonic()
    y = np.asarray(y, dtype=float).ravel()
    y = y[~np.isnan(y)]
    n = len(y)
    if n < 4:
        raise ValueError("At least 4 observations required for ela_distr")

    y_skew = y - y.mean()
    skewness = float(np.sqrt(n) * (y_skew**3).sum() / ((y_skew**2).sum() ** 1.5) * (1 - 1 / n) ** 1.5)

    y_kurt = y - y.mean()
    r = n * (y_kurt**4).sum() / (y_kurt**2).sum() ** 2
    kurtosis = float(r * (1 - 1 / n) ** 2 - 3)

    kernel = gaussian_kde(y)
    low_ = y.min() - 3 * kernel.covariance_factor() * y.std()
    upp_ = y.max() + 3 * kernel.covariance_factor() * y.std()
    positions = np.mgrid[low_:upp_:512j]
    d = kernel(positions)
    idx = np.arange(1, len(d) - 2)
    min_index = [x for x in idx if d[x] < d[x - 1] and d[x] < d[x + 1]]
    min_index = [0, *min_index, len(d)]
    modemass = []
    for i in range(len(min_index) - 1):
        a, b = int(min_index[i]), int(min_index[i + 1] - 1)
        modemass.append(d[a:b].mean() + abs(positions[a] - positions[b]))
    n_peaks = int((np.array(modemass) > 0.1).sum())

    return {
        "ela_distr.skewness": skewness,
        "ela_distr.kurtosis": kurtosis,
        "ela_distr.number_of_peaks": n_peaks,
        "ela_distr.costs_runtime": time.monotonic() - t0,
    }


def calculate_ela_level(
    X: np.ndarray,
    y: np.ndarray,
    *,
    quantiles: list[float] | None = None,
    n_splits: int = 10,
    maximize: bool = False,
) -> dict[str, float]:
    """Level-set classifiers; returns MCVA (= 1 - MMCE) for LDA/QDA/CART."""
    t0 = time.monotonic()
    quantiles = quantiles or [0.10, 0.25, 0.50]
    X = _as_df(X)
    y = np.asarray(y, dtype=float).ravel()

    el: dict[str, list[float]] = {"lda": [], "qda": [], "cart": []}
    for prob in quantiles:
        if maximize:
            y_quant = np.quantile(y, 1.0 - prob)
            y_class = (y >= y_quant).astype(int)
        else:
            y_quant = np.quantile(y, prob)
            y_class = (y < y_quant).astype(int)
        if y_class.sum() in (0, len(y_class)) or y_class.sum() < n_splits:
            for k in el:
                el[k].append(float("nan"))
            continue
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        acc = {"lda": [], "qda": [], "cart": []}
        for train, test in kf.split(X, y_class):
            for name, clf in (
                ("lda", LinearDiscriminantAnalysis()),
                ("qda", QuadraticDiscriminantAnalysis()),
                ("cart", DecisionTreeClassifier(random_state=42)),
            ):
                clf.fit(X.iloc[train], y_class[train])
                acc[name].append(float((y_class[test] == clf.predict(X.iloc[test])).mean()))
        for name in el:
            el[name].append(float(np.mean(acc[name])))

    out: dict[str, float] = {}
    for i, prob in enumerate(quantiles):
        pct = int(round(prob * 100))
        ela = el["lda"][i]
        eqa = el["qda"][i]
        eta = el["cart"][i]
        out[f"ela_level.mcva_lda_{pct:02d}"] = ela
        out[f"ela_level.mcva_qda_{pct:02d}"] = eqa
        out[f"ela_level.mcva_cart_{pct:02d}"] = eta
        out[f"ela_level.lq_lda_qda_{pct:02d}"] = ela / eqa if eqa > 1e-12 else float("nan")
        out[f"ela_level.le_lda_cart_{pct:02d}"] = ela / eta if eta > 1e-12 else float("nan")
    out["ela_level.costs_runtime"] = time.monotonic() - t0
    return out


def calculate_dispersion(
    X: np.ndarray,
    y: np.ndarray,
    *,
    disp_quantiles: list[float] | None = None,
    maximize: bool = False,
) -> dict[str, float]:
    t0 = time.monotonic()
    disp_quantiles = disp_quantiles or [0.02, 0.05, 0.10, 0.25]
    X = _as_df(X)
    y = np.asarray(y, dtype=float).ravel()
    if not maximize:
        y_work = y
    else:
        y_work = -y

    quantile_vals = np.quantile(y_work, disp_quantiles)
    dist_full = squareform(pdist(X, metric="euclidean"))
    full_nonzero = dist_full[dist_full != 0]
    means, medians = [], []
    for qv in quantile_vals:
        idx = np.where(y_work <= qv)[0]
        if len(idx) < 2:
            means.append(float("nan"))
            medians.append(float("nan"))
            continue
        sub = squareform(pdist(X.iloc[idx], metric="euclidean"))
        nz = sub[sub != 0]
        means.append(float(np.mean(nz)))
        medians.append(float(np.median(nz)))

    out: dict[str, float] = {}
    for i, q in enumerate(disp_quantiles):
        tag = f"{int(round(q * 100)):02d}"
        out[f"disp.ratio_mean_{tag}"] = means[i] / np.mean(full_nonzero)
        out[f"disp.ratio_median_{tag}"] = medians[i] / np.median(full_nonzero)
        out[f"disp.diff_mean_{tag}"] = means[i] - np.mean(full_nonzero)
        out[f"disp.diff_median_{tag}"] = medians[i] - np.median(full_nonzero)
    out["disp.costs_runtime"] = time.monotonic() - t0
    return out


def calculate_fdc(X: np.ndarray, y: np.ndarray, *, maximize: bool = False) -> float:
    """Fitness-distance correlation (Jones & Forrest 1995)."""
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    best_i = int(np.argmax(y) if maximize else np.argmin(y))
    dists = np.linalg.norm(X - X[best_i], axis=1)
    if np.std(dists) < 1e-15 or np.std(y) < 1e-15:
        return 0.0
    return float(pearsonr(y, dists)[0])


def calculate_disp1pct(X: np.ndarray, y: np.ndarray, *, maximize: bool = False) -> float:
    """DISP1% — mean pairwise distance among points within 1% of best objective."""
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    y_rng = float(y.max() - y.min())
    if y_rng < 1e-15:
        return 0.0
    if maximize:
        mask = y >= y.max() - 0.01 * y_rng
    else:
        mask = y <= y.min() + 0.01 * y_rng
    pts = X[mask]
    if len(pts) < 2:
        return 0.0
    d = pdist(pts, metric="euclidean")
    return float(np.mean(d[d > 0])) if np.any(d > 0) else 0.0


def calculate_entropy_y(y: np.ndarray, *, n_bins: int = 32) -> float:
    counts, _ = np.histogram(y, bins=n_bins)
    p = counts.astype(float) / max(counts.sum(), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def calculate_information_content(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int = 42,
    ic_nn_neighborhood: int = 20,
) -> dict[str, float]:
    """Port of pflacco ``calculate_information_content`` (NN tour)."""
    t0 = time.monotonic()
    from sklearn.neighbors import NearestNeighbors

    X = _as_df(X).reset_index(drop=True)
    y = pd.Series(np.asarray(y, dtype=float).ravel())
    rng = np.random.default_rng(seed)
    ic_epsilon = np.insert(10.0 ** np.linspace(-5, 15, 1000), 0, 0.0)
    ic_settling_sensitivity = 0.05
    ic_info_sensitivity = 0.5

    n_pts = X.shape[0]
    ic_nn_start = int(rng.integers(0, n_pts))
    nbrs = NearestNeighbors(n_neighbors=min(ic_nn_neighborhood, n_pts), algorithm="kd_tree").fit(X)
    distances, indices = nbrs.kneighbors(X)

    current = ic_nn_start
    candidates = [i for i in range(n_pts) if i != current]
    permutation = [current]
    dists: list[float | None] = [None]
    for _ in range(1, n_pts):
        currents = indices[permutation[-1]]
        nxt = [x for x in currents if x in candidates]
        if nxt:
            current = int(nxt[0])
        else:
            nbrs2 = NearestNeighbors(n_neighbors=1).fit(X.iloc[candidates].to_numpy())
            d2, i2 = nbrs2.kneighbors(X.iloc[permutation[-1]].to_numpy().reshape(1, -1))
            current = int(candidates[int(i2.ravel()[0])])
            dists.append(float(d2.ravel()[0]))
            permutation.append(current)
            candidates = [c for c in candidates if c != current]
            continue
        permutation.append(current)
        candidates = [c for c in candidates if c != current]
        dists.append(float(distances[permutation[-2], currents == current][0]))

    d = np.array([v for v in dists[1:] if v is not None], dtype=float)
    y_perm = y.iloc[permutation].to_numpy()
    diff_y = np.diff(y_perm)
    ratio = diff_y / d
    epsilon = np.unique(ic_epsilon)
    psi_eps = np.array([[0 if abs(x) < eps else np.sign(x) for x in ratio] for eps in epsilon])
    h_vals, m_vals = [], []
    for row in psi_eps:
        a, b = row[:-1], row[1:]
        probs = [
            np.mean((a == -1) & (b == 0)),
            np.mean((a == -1) & (b == 1)),
            np.mean((a == 0) & (b == -1)),
            np.mean((a == 0) & (b == 1)),
            np.mean((a == 1) & (b == -1)),
            np.mean((a == 1) & (b == 0)),
        ]
        h_vals.append(-sum(0 if p == 0 else p * np.log(p) / np.log(6) for p in probs))
        nz = row[row != 0]
        len_row = len(nz[np.insert(np.diff(nz) != 0, 0, False)]) if len(nz) > 0 else 0
        m_vals.append(len_row / (len(row) - 1))
    h_arr = np.array(h_vals)
    m_arr = np.array(m_vals)
    eps_s_mask = epsilon[h_arr < ic_settling_sensitivity]
    eps_s = float(np.log10(eps_s_mask.min())) if len(eps_s_mask) > 0 else float("nan")
    m0 = float(m_arr[epsilon == 0][0])
    eps05_idx = np.where(m_arr > ic_info_sensitivity * m0)[0]
    eps_ratio = float(np.log10(epsilon[eps05_idx].max())) if len(eps05_idx) > 0 else float("nan")

    return {
        "ic.h_max": float(h_arr.max()),
        "ic.eps_s": eps_s,
        "ic.eps_max": float(np.median(epsilon[h_arr == h_arr.max()])),
        "ic.eps_ratio": eps_ratio,
        "ic.m0": m0,
        "ic.costs_runtime": time.monotonic() - t0,
    }


def entropic_significance(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_bins: int = 10,
    n_bootstrap: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """
    Seo & Moon entropic significance (orders 1 and 2) with bootstrap ``sigma``.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    d = X.shape[1]

    def _mi_pair(a: np.ndarray, b: np.ndarray) -> float:
        aq = np.quantile(a, np.linspace(0, 1, n_bins + 1)[1:-1])
        bq = np.quantile(b, np.linspace(0, 1, n_bins + 1)[1:-1])
        ad = np.digitize(a, aq)
        bd = np.digitize(b, bq)
        cont = np.histogram2d(ad, bd, bins=[n_bins, n_bins])[0].astype(float)
        pxy = cont / max(cont.sum(), 1)
        px, py = pxy.sum(1), pxy.sum(0)
        mi = 0.0
        for i in range(px.shape[0]):
            for j in range(py.shape[0]):
                if pxy[i, j] > 0:
                    mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j] + 1e-15) + 1e-15)
        return float(mi)

    hy = calculate_entropy_y(y, n_bins=n_bins)

    def order1_scores() -> np.ndarray:
        return np.array([_mi_pair(X[:, j], y) / max(hy, 1e-15) for j in range(d)])

    def order2_score() -> float:
        if d < 2:
            return 0.0
        vals = [_mi_pair(X[:, i] + X[:, j], y) / max(hy, 1e-15) for i in range(d) for j in range(i + 1, d)]
        return float(np.mean(vals)) if vals else 0.0

    xi1 = float(np.mean(order1_scores()))
    xi2 = order2_score()
    xi_d = xi2 if d > 1 else xi1

    boot1, boot2 = [], []
    n = len(y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        Xb, yb = X[idx], y[idx]
        hy_b = calculate_entropy_y(yb, n_bins=n_bins)
        b1 = np.array([
            _mi_pair(Xb[:, j], yb) / max(hy_b, 1e-15) for j in range(d)
        ])
        boot1.append(float(np.mean(b1)))
        if d >= 2:
            b2 = float(np.mean([
                _mi_pair(Xb[:, i] + Xb[:, j], yb) / max(hy_b, 1e-15)
                for i in range(d) for j in range(i + 1, d)
            ]))
            boot2.append(b2)

    return {
        "entropic.xi_1": xi1,
        "entropic.xi_2": xi2,
        "entropic.xi_D": xi_d,
        "entropic.sigma_1": float(np.std(boot1)) if boot1 else 0.0,
        "entropic.sigma_2": float(np.std(boot2)) if boot2 else 0.0,
    }


def munoz_table1_features(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    maximize: bool = True,
    dim: int,
) -> dict[str, float]:
    """Muñoz et al. (2019) Table 1 — all 33 named features."""
    meta = calculate_ela_meta(Z, y)
    level = calculate_ela_level(Z, y, quantiles=[0.10, 0.25, 0.50], maximize=maximize)
    distr = calculate_ela_distribution(y)
    ic = calculate_information_content(Z, y)

    out: dict[str, float] = {
        "FDC": calculate_fdc(Z, y, maximize=maximize),
        "DISP1pct": calculate_disp1pct(Z, y, maximize=maximize),
        "R2_L": meta["ela_meta.lin_simple.adj_r2"],
        "R2_LI": meta["ela_meta.lin_w_interact.adj_r2"],
        "R2_Q": meta["ela_meta.quad_simple.adj_r2"],
        "R2_QI": meta["ela_meta.quad_w_interact.adj_r2"],
        "beta_min": meta["ela_meta.lin_simple.coef.min"],
        "beta_max": meta["ela_meta.lin_simple.coef.max"],
        "CN": meta["ela_meta.quad_simple.cond"],
        "EL10": level["ela_level.mcva_lda_10"],
        "EQ10": level["ela_level.mcva_qda_10"],
        "LQ10": level["ela_level.lq_lda_qda_10"],
        "ET10": level["ela_level.mcva_cart_10"],
        "EL25": level["ela_level.mcva_lda_25"],
        "EQ25": level["ela_level.mcva_qda_25"],
        "LQ25": level["ela_level.lq_lda_qda_25"],
        "ET25": level["ela_level.mcva_cart_25"],
        "EL50": level["ela_level.mcva_lda_50"],
        "EQ50": level["ela_level.mcva_qda_50"],
        "LQ50": level["ela_level.lq_lda_qda_50"],
        "ET50": level["ela_level.mcva_cart_50"],
        "gamma_Y": distr["ela_distr.skewness"],
        "H_Y": calculate_entropy_y(y),
        "kappa_Y": distr["ela_distr.kurtosis"],
        "PKS": distr["ela_distr.number_of_peaks"],
        "Hmax": ic["ic.h_max"],
        "eps_S": ic["ic.eps_s"],
        "M0": ic["ic.m0"],
    }
    ent = entropic_significance(Z, y)
    out["xi_D"] = ent["entropic.xi_D"]
    out["xi_1"] = ent["entropic.xi_1"]
    out["xi_2"] = ent["entropic.xi_2"]
    out["sigma_1"] = ent["entropic.sigma_1"]
    out["sigma_2"] = ent["entropic.sigma_2"]
    return out


def compute_all_ela_feature_groups(
    Z: np.ndarray,
    y: np.ndarray,
    x_comp: np.ndarray,
    *,
    x_campaign: np.ndarray,
    y_campaign: np.ndarray,
    maximize: bool = True,
    dim: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Full ELA dump: Muñoz-33, flacco classical blocks, ZoMBI extras."""
    from ela.features import feature_median_lipschitz, feature_oob_r2

    groups: dict[str, Any] = {}
    groups["munoz_33"] = munoz_table1_features(Z, y, maximize=maximize, dim=dim)
    groups["flacco_meta"] = calculate_ela_meta(Z, y)
    groups["flacco_distr"] = calculate_ela_distribution(y)
    groups["flacco_level"] = calculate_ela_level(Z, y, maximize=maximize)
    groups["flacco_dispersion"] = calculate_dispersion(Z, y, maximize=maximize)
    groups["flacco_ic"] = calculate_information_content(Z, y, seed=seed)
    groups["flacco_entropic"] = entropic_significance(Z, y, seed=seed)
    groups["zombi"] = {
        "oob_r2": feature_oob_r2(x_campaign, y_campaign),
        "median_lipschitz": feature_median_lipschitz(x_comp, y),
    }
    return groups
