"""
llm/surrogate.py
================
A **generative surrogate** of the ``campaign2_all.db`` experiment.

Where ``visualization/plot_run.py`` fits a single Random-Forest that maps
*composition → Objective* (a deterministic interpolation), this module builds a
much richer object: a model you can *sample* from, that emits a **whole
synthetic measurement** — the Objective **and** every other measured feature —
that is statistically consistent with the real campaign, conditioned on the
three things you always know before you run a spot:

    the composition (FAPbI3, MAPbI3, MAPbBr3),  the wall-clock time,  the BO iteration.

Design (chosen with data_exploration.ipynb)
-------------------------------------------
The campaign has ~640 raw columns, most of them highly redundant *curves*
(absorption spectrum, stability voltage sweep, initial/final PL spectra). We
first collapse each curve group to a handful of **functional-PCA** scores
(exactly as the notebook does), leaving ~33 informative features. Then:

1.  **Conditional mean (what composition/time/iteration explain).**
    For every feature we fit a Random-Forest regressor on
    ``Z = (FAPbI3, MAPbBr3, time, iteration)`` and take its **out-of-bag**
    prediction as the conditional mean.  OOB (rather than in-sample) prediction
    is essential here: an RF fits its training rows almost perfectly, so
    in-sample residuals would badly *under*-estimate the noise we need to sample.

2.  **Residuals.**  ``resid = feature − OOB_mean``.  Everything the conditioning
    variables do *not* explain lives in these residuals — and, crucially, the
    residuals of different features **co-vary** (a spot that absorbs unusually
    strongly for its composition also tends to be off-trend in PL, objective, …).

3.  **Factor model of the residual covariance.**  We standardise the residuals
    and estimate their joint covariance Σ.  Because the interesting curve groups
    (PL, kinetics variances) are only measured on ~247 / 953 rows, Σ is estimated
    by **EM for a multivariate Gaussian with missing data** — this uses *every*
    row for the blocks it observed, instead of throwing away 2/3 of the data to
    complete-case.  Σ is then regularised with **diagonal (Ledoit-Wolf-style)
    shrinkage** so it is well-conditioned and positive-definite, and its leading
    eigen-structure is reported as the shared **factors** that drive the
    co-variation.

4.  **Sampling.**  Given any ``(composition, time, iteration)`` we predict each
    feature's conditional mean with its RF, draw a joint residual vector from
    ``N(0, Σ)``, de-standardise, and add — yielding a full synthetic row whose
    feature↔objective and feature↔feature relationships match the real data.

Why a factor / shrunk-covariance model (not plain sample covariance)?
    With ~33 features and as few as ~230 jointly-observed rows for the sparse
    blocks, the raw empirical covariance is noisy and can be near-singular —
    Cholesky (needed to sample) would fail and the sampled correlations would be
    over-fit to noise.  Shrinking toward the diagonal guarantees a valid,
    stable Σ and is the standard robust estimator in the p-not-≪-n regime; the
    eigen-factors give the same "shared latent drivers" interpretation you asked
    for from a factor model, with none of the fragility of fitting a fixed
    factor count to incomplete data.

Usage
-----
    conda activate zombi-hop
    python llm/surrogate.py                 # fit, print diagnostics
    python llm/surrogate.py --plot          # + ternary demo PNG
    python llm/surrogate.py --save model.pkl

    # programmatic
    from surrogate import Surrogate
    surr = Surrogate.fit()
    row  = surr.sample(comp=(0.7, 0.2, 0.1), time_sec=None, iteration=None)
    df   = surr.sample_grid(grid_n=60)      # a synthetic ternary sweep
"""
from __future__ import annotations

import argparse
import pickle
import re
import sqlite3
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, eigh
from sklearn.ensemble import RandomForestRegressor

# skfda is only needed at *fit* time (to compress the curve groups); a pickled
# Surrogate carries the fitted reconstructors and does not import it to sample.
from skfda import FDataGrid
from skfda.representation.basis import BSplineBasis
from skfda.preprocessing.dim_reduction import FPCA

warnings.filterwarnings("ignore")

# Windows consoles default to cp1252 and choke on the δ/…/→ used in diagnostics.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# ── paths & column groups (mirror data_exploration.ipynb) ─────────────────────
_HERE = Path(__file__).resolve().parent
DEFAULT_DB = _HERE / "data" / "campaign2_all.db"

COMP = ["FAPbI3", "MAPbI3", "MAPbBr3"]              # ternary simplex (sums to 1)
COND_COLS = ["FAPbI3", "MAPbBr3", "_time_sec", "Iteration"]  # conditioning Z (MAPbI3 redundant)

# scalar features kept in the joint model.  k2 is dropped: it is numerically
# identical to Stability (Spearman rho = 1.00) and would make Σ singular.
SUBTARGETS = ["Bandgap", "Photoconductance", "Stability"]
ENV = ["Temperature_in", "Temperature_out", "Humidity_in", "Humidity_out",
       "Pressure_in", "Pressure_out", "DMF_ppm_in", "DMF_ppm_out"]
KIN = ["k1", "k3", "k1_var", "k2_var", "k3_var"]   # k2 excluded (== Stability)

# functional (curve) groups → compressed to a few fPCA scores each.
FPCA_CONFIG = {
    #  group            n_basis  n_comp
    "absorption":       (20, 4),
    "stability_dark":   (8, 3),
    "stability_light":  (8, 3),
    "PL_initial":       (10, 3),
    "PL_final":         (10, 3),
}


# ── data loading ──────────────────────────────────────────────────────────────

def load_frame(db_path: Path = DEFAULT_DB) -> tuple[pd.DataFrame, dict]:
    """Load the ``results`` table, add ``_time_sec``, and classify columns."""
    df = pd.read_sql("select * from results", sqlite3.connect(str(db_path)))

    t = pd.to_datetime(df["Timestamp"])
    df["_time_sec"] = (t - t.min()).dt.total_seconds()

    def _isnum(s):
        try:
            float(s); return True
        except ValueError:
            return False

    absorp = [c for c in df.columns if _isnum(c)]                       # absorption wavelengths
    stab = [c for c in df.columns if re.match(r"^-?\d+_(dark|light)_\d+$", c)]
    ipl = [c for c in df.columns if c.endswith("_i_pl")]
    fpl = [c for c in df.columns if c.endswith("_f_pl")]
    modules = [c for c in df.columns if re.match(r"^Module\d$", c)]     # identically zero
    df = df.drop(columns=modules)

    cols = {"absorption": absorp, "stability_sweep": stab, "PL_initial": ipl, "PL_final": fpl}
    return df, cols


# ── functional-PCA compression (curve groups → scores) ────────────────────────

@dataclass
class FunctionalReconstructor:
    """Everything needed to turn fPCA scores back into a curve on the grid."""
    grid: np.ndarray              # functional index (wavelength / voltage)
    mean_grid: np.ndarray         # (G,) mean curve evaluated on grid
    comps_grid: np.ndarray        # (n_comp, G) eigen-functions on grid
    score_names: list[str]

    def reconstruct(self, scores: np.ndarray) -> np.ndarray:
        """scores (..., n_comp) → curve (..., G) = mean + Σ score_j · comp_j."""
        scores = np.asarray(scores, float)
        return self.mean_grid + scores @ self.comps_grid


def _fpca_group(values: np.ndarray, grid: np.ndarray, n_basis: int, n_comp: int,
                short: str) -> tuple[np.ndarray, list[str], FunctionalReconstructor, float]:
    """Smooth curves onto a B-spline basis, FPCA → (scores[n,n_comp], names, recon, cum_evr)."""
    V = np.asarray(values, float)
    valid = np.isfinite(V).mean(1) > 0.5
    Vv = V[valid]
    col_mean = np.nanmean(Vv, 0)
    Vv = np.where(np.isfinite(Vv), Vv, col_mean)      # impute stray NaNs

    fd = FDataGrid(Vv, grid_points=grid)
    basis = BSplineBasis(n_basis=n_basis, domain_range=(float(grid.min()), float(grid.max())))
    fp = FPCA(n_components=n_comp)
    scores_valid = fp.fit_transform(fd.to_basis(basis))

    scores = np.full((len(V), n_comp), np.nan)
    scores[valid] = scores_valid

    mean_grid = fd.mean().to_grid(grid).data_matrix.ravel()
    comps_grid = fp.components_.to_grid(grid).data_matrix.reshape(n_comp, -1)
    names = [f"{short}_fPC{j + 1}" for j in range(n_comp)]
    recon = FunctionalReconstructor(grid, mean_grid, comps_grid, names)
    return scores, names, recon, float(fp.explained_variance_ratio_.sum())


def build_features(df: pd.DataFrame, cols: dict, verbose: bool = True):
    """Assemble the joint feature matrix F (n×p) and conditioning Z (n×4).

    Returns ``(Z, F, feat_names, groups, reconstructors)`` where F holds NaN for
    unmeasured features and ``groups[name]`` labels each feature's origin.
    """
    short = {"absorption": "absorp", "stability_dark": "stab_dark",
             "stability_light": "stab_light", "PL_initial": "PLi", "PL_final": "PLf"}

    feat_cols, feat_names, groups, reconstructors = [], [], {}, {}

    def _add(name, values, group):
        feat_cols.append(np.asarray(values, float))
        feat_names.append(name)
        groups[name] = group

    # 1) scalars ---------------------------------------------------------------
    _add("Objective", df["Objective"].values, "objective")
    for c in SUBTARGETS:
        _add(c, df[c].values, "subtarget")
    for c in ENV:
        _add(c, df[c].values, "environment")
    for c in KIN:
        _add(c, df[c].values, "kinetics")

    # 2) absorption curve ------------------------------------------------------
    wl_abs = np.array([float(c) for c in cols["absorption"]])
    nb, nc = FPCA_CONFIG["absorption"]
    sc, names, recon, evr = _fpca_group(df[cols["absorption"]].values, wl_abs, nb, nc,
                                        short["absorption"])
    for j, nm in enumerate(names):
        _add(nm, sc[:, j], "absorption")
    reconstructors["absorption"] = recon
    if verbose:
        print(f"  absorption      -> {nc} scores (cum EVR {evr:.3f})")

    # 3) stability sweep: average 3 repeats → dark(V) & light(V) curves --------
    volts = sorted({int(c.split("_")[0]) for c in cols["stability_sweep"]})
    vgrid = np.array(volts, float)
    for cond, gname in [("dark", "stability_dark"), ("light", "stability_light")]:
        curve = np.stack(
            [df[[f"{v}_{cond}_{r}" for r in (1, 2, 3)]].mean(axis=1).values for v in volts],
            axis=1)
        nb, nc = FPCA_CONFIG[gname]
        sc, names, recon, evr = _fpca_group(curve, vgrid, nb, nc, short[gname])
        for j, nm in enumerate(names):
            _add(nm, sc[:, j], gname)
        reconstructors[gname] = recon
        if verbose:
            print(f"  {gname:15s} -> {nc} scores (cum EVR {evr:.3f})")

    # 4) PL spectra ------------------------------------------------------------
    wl_pl = np.array([int(c.split("_")[0]) for c in cols["PL_initial"]], float)
    for group, raw in [("PL_initial", cols["PL_initial"]), ("PL_final", cols["PL_final"])]:
        nb, nc = FPCA_CONFIG[group]
        sc, names, recon, evr = _fpca_group(df[raw].values, wl_pl, nb, nc, short[group])
        for j, nm in enumerate(names):
            _add(nm, sc[:, j], group)
        reconstructors[group] = recon
        if verbose:
            print(f"  {group:15s} -> {nc} scores (cum EVR {evr:.3f})")

    F = np.column_stack(feat_cols)
    Z = df[COND_COLS].values.astype(float)
    return Z, F, feat_names, groups, reconstructors


# ── step 1-2: conditional means (RF-OOB) and residuals ────────────────────────

def fit_conditional_means(Z: np.ndarray, F: np.ndarray, feat_names: list[str],
                          n_estimators: int = 400, seed: int = 42, verbose: bool = True):
    """Fit an RF per feature; return (models, residuals, oob_r2).

    ``residuals[i, j] = F[i, j] − OOB_prediction`` (NaN where F is NaN). The
    out-of-bag prediction is an honest held-out mean, so the residual scale is
    not deflated by the RF over-fitting its own training rows.
    """
    n, p = F.shape
    models, resid, oob_r2 = [], np.full((n, p), np.nan), np.full(p, np.nan)
    zfinite = np.isfinite(Z).all(1)                     # Z is complete in practice
    for j in range(p):
        m = zfinite & np.isfinite(F[:, j])
        rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=True,
                                   n_jobs=-1, random_state=seed)
        rf.fit(Z[m], F[m, j])
        models.append(rf)
        resid[m, j] = F[m, j] - rf.oob_prediction_
        oob_r2[j] = rf.oob_score_
        if verbose:
            print(f"    {feat_names[j]:18s} n={m.sum():4d}  OOB R2={rf.oob_score_:+.3f}")
    return models, resid, oob_r2


# ── step 3: EM covariance under missingness + shrinkage + factors ─────────────

def em_gaussian_cov(X: np.ndarray, max_iter: int = 500, tol: float = 1e-4,
                    ridge: float = 1e-6):
    """MLE mean & covariance of a multivariate Gaussian with missing (NaN) entries.

    Standard EM (Little & Rubin): rows sharing a missingness pattern are handled
    together; the M-step covariance gets the usual conditional-variance
    correction for the imputed cells.  Uses every observed block of every row.
    """
    X = np.asarray(X, float)
    n, p = X.shape
    obs = np.isfinite(X)

    mu = np.where(np.isfinite(np.nanmean(X, 0)), np.nanmean(X, 0), 0.0)
    imp = np.where(obs, X, mu)
    Sigma = np.cov(imp, rowvar=False) + ridge * np.eye(p)

    # group row indices by identical observed-pattern
    patterns: dict[bytes, list[int]] = {}
    for i in range(n):
        patterns.setdefault(obs[i].tobytes(), []).append(i)

    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        T1 = np.zeros(p)
        T2 = np.zeros((p, p))
        for key, rows in patterns.items():
            rows = np.asarray(rows)
            o = np.frombuffer(key, dtype=bool)
            oi, mi = np.where(o)[0], np.where(~o)[0]
            Ec = np.zeros((len(rows), p))
            Ec[:, oi] = X[np.ix_(rows, oi)]
            corr = np.zeros((p, p))
            if mi.size:
                Soo = Sigma[np.ix_(oi, oi)] + ridge * np.eye(oi.size)
                Smo = Sigma[np.ix_(mi, oi)]
                A = Smo @ np.linalg.inv(Soo)                       # (nm, no)
                Ec[:, mi] = mu[mi] + (X[np.ix_(rows, oi)] - mu[oi]) @ A.T
                Cmm = Sigma[np.ix_(mi, mi)] - A @ Smo.T            # conditional cov (shared)
                corr[np.ix_(mi, mi)] = Cmm * len(rows)
            T1 += Ec.sum(0)
            T2 += Ec.T @ Ec + corr
        mu_new = T1 / n
        Sigma_new = T2 / n - np.outer(mu_new, mu_new)
        Sigma_new = 0.5 * (Sigma_new + Sigma_new.T) + ridge * np.eye(p)
        delta = np.abs(Sigma_new - Sigma).max()
        mu, Sigma = mu_new, Sigma_new
        if delta < tol:
            break
    return mu, Sigma, n_iter


def shrink_to_psd(Sigma: np.ndarray, max_cond: float = 1e4, floor: float = 1e-3):
    """Shrink toward the diagonal until well-conditioned & positive-definite.

    ``Σ_δ = (1−δ)Σ + δ·diag(Σ)`` — the Ledoit-Wolf diagonal target.  Returns the
    smallest δ on a grid that gives all eigenvalues > ``floor·λ_max`` and
    condition number < ``max_cond`` (so Cholesky sampling is stable).
    """
    D = np.diag(np.diag(Sigma))
    for delta in [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]:
        S = (1 - delta) * Sigma + delta * D
        w = np.linalg.eigvalsh(S)
        if w.min() > floor * w.max() and (w.max() / w.min()) < max_cond:
            return S, delta
    return 0.5 * Sigma + 0.5 * D, 0.5


def cov_to_corr(S: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(S))
    d[d == 0] = 1.0
    return S / np.outer(d, d)


def factor_decompose(Sigma: np.ndarray, feat_names: list[str]):
    """Eigen-factor summary of the residual correlation matrix.

    Returns ``(n_factors, loadings, evr)`` where ``n_factors`` is the Kaiser
    count (eigenvalues > 1) and ``loadings[:, :n_factors]`` are the shared
    factors Λ with Σ_corr ≈ ΛΛᵀ + Ψ.
    """
    corr = cov_to_corr(Sigma)
    w, V = eigh(corr)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    evr = w / w.sum()
    k = int(max(1, np.sum(w > 1.0)))
    loadings = V * np.sqrt(np.clip(w, 0, None))         # (p, p); take [:, :k]
    return k, loadings, evr


# ── the surrogate ─────────────────────────────────────────────────────────────

@dataclass
class Surrogate:
    feat_names: list[str]
    groups: dict
    models: list                       # RandomForestRegressor per feature
    resid_mean: np.ndarray             # (p,) OOB residual bias
    resid_std: np.ndarray              # (p,) residual scale
    Sigma: np.ndarray                  # (p,p) shrunk covariance of standardised residuals
    reconstructors: dict               # group -> FunctionalReconstructor
    oob_r2: np.ndarray
    shrink_delta: float
    n_factors: int
    loadings: np.ndarray
    evr: np.ndarray
    time_ref: dict                     # {'min','max','median'} of _time_sec
    iter_ref: dict
    _chol: np.ndarray = field(default=None, repr=False)

    # -- construction ----------------------------------------------------------
    @classmethod
    def fit(cls, db_path: Path = DEFAULT_DB, n_estimators: int = 400,
            seed: int = 42, verbose: bool = True) -> "Surrogate":
        if verbose:
            print(f"Loading {db_path} …")
        df, cols = load_frame(db_path)
        if verbose:
            print(f"  {len(df)} rows; compressing curve groups with fPCA:")
        Z, F, feat_names, groups, recon = build_features(df, cols, verbose)
        if verbose:
            print(f"  joint feature vector: p = {len(feat_names)} features\n"
                  f"Fitting {len(feat_names)} RF conditional means (OOB residuals):")
        models, resid, oob_r2 = fit_conditional_means(Z, F, feat_names, n_estimators,
                                                       seed, verbose)

        resid_mean = np.nanmean(resid, 0)
        resid_std = np.nanstd(resid, 0)
        resid_std[resid_std == 0] = 1.0
        Rz = (resid - resid_mean) / resid_std             # standardised, NaN-preserving

        if verbose:
            print("Estimating residual covariance by EM (missing-data) …")
        _, Sigma0, n_it = em_gaussian_cov(Rz)
        Sigma, delta = shrink_to_psd(Sigma0)
        k, loadings, evr = factor_decompose(Sigma, feat_names)
        if verbose:
            print(f"  EM converged in {n_it} iters; shrinkage δ = {delta:.3f}; "
                  f"{k} factors (eig>1) explain {evr[:k].sum():.2f} of residual variance")

        surr = cls(
            feat_names=feat_names, groups=groups, models=models,
            resid_mean=resid_mean, resid_std=resid_std, Sigma=Sigma,
            reconstructors=recon, oob_r2=oob_r2, shrink_delta=delta,
            n_factors=k, loadings=loadings, evr=evr,
            time_ref={"min": float(df["_time_sec"].min()),
                      "max": float(df["_time_sec"].max()),
                      "median": float(df["_time_sec"].median())},
            iter_ref={"min": int(df["Iteration"].min()),
                      "max": int(df["Iteration"].max()),
                      "median": float(df["Iteration"].median())},
        )
        surr._chol = cholesky(Sigma, lower=True)
        return surr

    # -- sampling --------------------------------------------------------------
    def _resolve_cond(self, time_sec, iteration):
        if time_sec is None:
            time_sec = self.time_ref["max"]               # default: end of campaign
        if iteration is None:
            iteration = self.iter_ref["max"]
        return float(time_sec), float(iteration)

    def _cond_means(self, Z: np.ndarray) -> np.ndarray:
        """Conditional mean of every feature at rows Z (m×4) → (m, p)."""
        return np.column_stack([rf.predict(Z) for rf in self.models])

    def _draw_resid(self, m: int, rng) -> np.ndarray:
        """m joint residual draws (m×p) in ORIGINAL feature units."""
        z = rng.standard_normal((m, len(self.feat_names)))
        std_resid = z @ self._chol.T                      # ~ N(0, Σ)
        return self.resid_mean + std_resid * self.resid_std

    def sample_at(self, Z: np.ndarray, seed: int | None = None) -> pd.DataFrame:
        """Sample one synthetic full row per conditioning row ``Z`` (m×4)."""
        Z = np.atleast_2d(np.asarray(Z, float))
        rng = np.random.default_rng(seed)
        samples = self._cond_means(Z) + self._draw_resid(len(Z), rng)
        out = pd.DataFrame(samples, columns=self.feat_names)
        out.insert(0, "Iteration", Z[:, 3])
        out.insert(0, "_time_sec", Z[:, 2])
        out.insert(0, "MAPbBr3", Z[:, 1])
        out.insert(0, "MAPbI3", 1.0 - Z[:, 0] - Z[:, 1])
        out.insert(0, "FAPbI3", Z[:, 0])
        return out

    def sample(self, comp, time_sec=None, iteration=None, n: int = 1, seed=None):
        """Sample ``n`` synthetic rows at a single composition + condition.

        ``comp`` is (FAPbI3, MAPbI3, MAPbBr3) (renormalised to sum 1).
        ``time_sec`` / ``iteration`` default to the end of the campaign.
        """
        comp = np.asarray(comp, float)
        comp = comp / comp.sum()
        t, it = self._resolve_cond(time_sec, iteration)
        Z = np.tile([comp[0], comp[2], t, it], (n, 1))
        return self.sample_at(Z, seed)

    def sample_grid(self, grid_n: int = 60, time_sec=None, iteration=None,
                    stochastic: bool = True, seed=0) -> pd.DataFrame:
        """Sample the full ternary simplex at a fixed time & iteration.

        With ``stochastic=False`` returns the deterministic RF conditional means
        (the analogue of plot_run.py's surrogate); otherwise one joint draw per
        grid point.
        """
        pts = []
        for i in range(grid_n + 1):
            for j in range(grid_n + 1 - i):
                pts.append([i / grid_n, j / grid_n, (grid_n - i - j) / grid_n])
        comp = np.asarray(pts, float)                     # (M,3) FAPbI3,MAPbI3,MAPbBr3
        t, it = self._resolve_cond(time_sec, iteration)
        Z = np.column_stack([comp[:, 0], comp[:, 2],
                             np.full(len(comp), t), np.full(len(comp), it)])
        if stochastic:
            return self.sample_at(Z, seed)
        means = self._cond_means(Z)
        out = pd.DataFrame(means, columns=self.feat_names)
        out.insert(0, "FAPbI3", comp[:, 0])
        out.insert(1, "MAPbI3", comp[:, 1])
        out.insert(2, "MAPbBr3", comp[:, 2])
        return out

    def reconstruct_curve(self, group: str, sample_row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Turn a sampled row's fPCA scores back into a curve (grid, values)."""
        recon = self.reconstructors[group]
        scores = np.array([sample_row[nm] for nm in recon.score_names])
        return recon.grid, recon.reconstruct(scores)

    # -- persistence -----------------------------------------------------------
    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path) -> "Surrogate":
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ── validation / diagnostics ──────────────────────────────────────────────────

def validate(surr: Surrogate, db_path: Path = DEFAULT_DB, seed: int = 0):
    """Check the surrogate reproduces (a) marginals and (b) feature↔feature
    co-variation of the real data.  Prints a compact report and returns a dict.
    """
    from scipy.stats import ks_2samp

    df, cols = load_frame(db_path)
    Z, F, feat_names, _, _ = build_features(df, cols, verbose=False)

    # one synthetic row at each real spot's conditioning
    synth = surr.sample_at(Z, seed=seed)[feat_names].values

    # (a) marginal fidelity: KS between real & synthetic per feature
    ks = []
    for j, nm in enumerate(feat_names):
        real = F[np.isfinite(F[:, j]), j]
        if real.size >= 30:
            ks.append((nm, ks_2samp(real, synth[:, j]).statistic))
    ks_stat = np.array([s for _, s in ks])

    # (b) covariance fidelity: pairwise-complete corr(real) vs corr(synthetic)
    real_corr = pd.DataFrame(F, columns=feat_names).corr().values
    synth_corr = pd.DataFrame(synth, columns=feat_names).corr().values
    mask = ~np.isnan(real_corr)
    iu = np.triu_indices_from(real_corr, k=1)
    keep = mask[iu]
    corr_mae = np.mean(np.abs(real_corr[iu][keep] - synth_corr[iu][keep]))

    # spotlight: does the surrogate keep the key discovered relationships?
    obj = feat_names.index("Objective")
    spotlight = {}
    for nm in ["Photoconductance", "absorp_fPC1", "Humidity_in", "DMF_ppm_out"]:
        if nm in feat_names:
            j = feat_names.index(nm)
            r = pd.DataFrame(F[:, [obj, j]]).corr().iloc[0, 1]
            s = np.corrcoef(synth[:, obj], synth[:, j])[0, 1]
            spotlight[nm] = (r, s)

    print("\n── validation ─────────────────────────────────────────────")
    print(f"marginal KS statistic   : median {np.median(ks_stat):.3f}  "
          f"90th pct {np.quantile(ks_stat, 0.9):.3f}   (0 = identical dist.)")
    print(f"corr-matrix MAE          : {corr_mae:.3f}   "
          f"(mean |Δ corr| over feature pairs; lower = better co-variation match)")
    print("key Objective correlations (real → synthetic):")
    for nm, (r, s) in spotlight.items():
        print(f"    Objective ~ {nm:16s} {r:+.3f} → {s:+.3f}")

    # factor loadings: which features drive the top shared factors
    print(f"\ntop factor loadings ({surr.n_factors} factors, eig>1):")
    for f in range(min(3, surr.n_factors)):
        load = surr.loadings[:, f]
        top = np.argsort(np.abs(load))[::-1][:5]
        items = ", ".join(f"{surr.feat_names[i]}({load[i]:+.2f})" for i in top)
        print(f"    factor {f + 1} (EVR {surr.evr[f]:.2f}): {items}")

    return {"ks": ks, "corr_mae": corr_mae, "spotlight": spotlight}


# ── optional ternary demo plot ────────────────────────────────────────────────

def demo_plot(surr: Surrogate, out: Path, grid_n: int = 80):
    """RF-mean Objective over the simplex + one stochastic sampled sweep."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _s32 = np.sqrt(3) / 2

    def to_xy(c):
        c = np.asarray(c, float)
        return np.column_stack([c[:, 1] + 0.5 * c[:, 2], _s32 * c[:, 2]])

    mean_df = surr.sample_grid(grid_n, stochastic=False)
    samp_df = surr.sample_grid(grid_n // 2, stochastic=True, seed=1)
    gxy = to_xy(mean_df[COMP].values)
    sxy = to_xy(samp_df[COMP].values)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (xy, val, title) in zip(
            axes,
            [(gxy, mean_df["Objective"].values, "RF conditional mean  E[Objective | comp]"),
             (sxy, samp_df["Objective"].values, "one stochastic joint sample")]):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=val, cmap="viridis", s=14)
        ax.plot([0, 1, 0.5, 0], [0, 0, _s32, 0], "k-", lw=1)
        for (px, py, lbl) in [(0.5, _s32, "MAPbBr3"), (0, 0, "FAPbI3"), (1, 0, "MAPbI3")]:
            ax.annotate(lbl, (px, py), ha="center",
                        va="bottom" if py > 0 else "top", fontsize=9, fontweight="bold")
        ax.set_title(title); ax.axis("off"); ax.set_aspect("equal")
        fig.colorbar(sc, ax=ax, label="Objective", shrink=0.75)
    fig.suptitle(f"Surrogate at t={surr.time_ref['max']/3600:.0f}h, "
                 f"iter={surr.iter_ref['max']}", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved demo plot → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Fit a generative surrogate of campaign2_all.db")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="path to campaign2_all.db")
    ap.add_argument("--n-estimators", type=int, default=400, help="RF trees per feature")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default=None, help="pickle the fitted surrogate to this path")
    ap.add_argument("--plot", action="store_true", help="write a ternary demo PNG")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args()

    surr = Surrogate.fit(Path(args.db), args.n_estimators, args.seed)

    if not args.no_validate:
        validate(surr, Path(args.db))

    # demonstrate a draw
    print("\n── example draw at composition (0.7, 0.2, 0.1), end of campaign ──")
    row = surr.sample((0.7, 0.2, 0.1), n=1, seed=0).iloc[0]
    print(row[["FAPbI3", "MAPbI3", "MAPbBr3", "Objective", "Bandgap",
               "Photoconductance", "Stability"]].round(4).to_string())

    if args.plot:
        demo_plot(surr, _HERE / "surrogate_demo.png")
    if args.save:
        surr.save(args.save)
        print(f"saved surrogate → {args.save}")


if __name__ == "__main__":
    main()
