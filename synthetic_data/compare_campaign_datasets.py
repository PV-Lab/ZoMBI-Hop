"""
compare_campaign_datasets.py
============================
Compare campaign1a (experimental 3D RF campaign) with synthetic 3D datasets on
objective distributions, sampling geometry, RF surrogate quality (comparison metric
only — MOBO on synthetics uses direct oracles), and landscape
roughness.  Writes a multi-panel dashboard PNG + metrics CSV.

Usage
-----
  python synthetic_data/compare_campaign_datasets.py
  MPLBACKEND=Agg python synthetic_data/compare_campaign_datasets.py --no-show
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

if not sys.stdin.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from synthetic_data.campaign_datasets import (
    RF_N_ESTIMATORS,
    composition_column_names,
    load_metadata,
    normalize_rows,
    train_rf,
)
from synthetic_data.oracles import ORACLE_CHOICES

DATA_DIR = Path(__file__).resolve().parent / "data"
PLOTS_DIR = DATA_DIR / "plots"
CAMPAIGN1A_COLS = ["FAPbI3", "MAPbI3", "MAPbBr3"]
OBJECTIVE_COL = "Objective"
TERNARY_GRID_N = 80
_SQRT3_2 = math.sqrt(3) / 2

# campaign1a measured properties (when present).
CAMPAIGN_PROPERTIES = ("Objective", "Bandgap", "Photoconductance", "Stability")


@dataclass
class DatasetPanel:
    key: str
    label: str
    X: np.ndarray
    y: np.ndarray
    comp_cols: list[str]
    source: str
    maximize: bool | None = None
    oracle: str | None = None


def comp_to_xy(comp: np.ndarray) -> np.ndarray:
    p = np.asarray(comp, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    s = p.sum(axis=-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return np.column_stack([p[:, 1] + 0.5 * p[:, 2], _SQRT3_2 * p[:, 2]])


def ternary_grid(n: int = TERNARY_GRID_N) -> np.ndarray:
    return np.array([
        [i / n, j / n, (n - i - j) / n]
        for i in range(n + 1)
        for j in range(n + 1 - i)
    ], dtype=float)


def resolve_campaign1a_path(explicit: str | None = None) -> Path:
    candidates = [
        Path(explicit) if explicit else None,
        DATA_DIR / "campaign1a.csv",
        _REPO / "data" / "campaign1a.csv",
        _REPO / "interactive_testing" / "data" / "campaign1a.csv",
        _REPO / "interactive_testing" / "campaign1a.csv",
    ]
    for path in candidates:
        if path and path.is_file():
            return path
    tried = "\n".join(f"  {p}" for p in candidates if p)
    raise FileNotFoundError(f"campaign1a.csv not found. Tried:\n{tried}")


def discover_synthetic_csvs() -> list[Path]:
    paths = sorted(DATA_DIR.glob("campaign3d_synthetic_*.csv"))
    return [p for p in paths if "_meta" not in p.name]


def load_campaign1a(path: Path) -> DatasetPanel:
    df = pd.read_csv(path).dropna(subset=CAMPAIGN1A_COLS + [OBJECTIVE_COL])
    X = normalize_rows(df[CAMPAIGN1A_COLS].values.astype(float))
    y = df[OBJECTIVE_COL].values.astype(float)
    return DatasetPanel(
        key="campaign1a",
        label=f"campaign1a (n={len(df)})",
        X=X,
        y=y,
        comp_cols=CAMPAIGN1A_COLS,
        source=str(path),
        maximize=False,
    )


def load_synthetic_csv(path: Path) -> DatasetPanel:
    meta_path = path.with_name(path.stem + "_meta.json")
    meta = load_metadata(meta_path) if meta_path.is_file() else {}
    comp_cols = meta.get("composition_columns") or composition_column_names(3)
    df = pd.read_csv(path).dropna(subset=comp_cols + [OBJECTIVE_COL])
    X = normalize_rows(df[comp_cols].values.astype(float))
    y = df[OBJECTIVE_COL].values.astype(float)
    oracle = meta.get("oracle") or path.stem.replace("campaign3d_synthetic_", "")
    return DatasetPanel(
        key=oracle,
        label=f"{oracle} (n={len(df)})",
        X=X,
        y=y,
        comp_cols=list(comp_cols),
        source=str(path),
        maximize=bool(meta.get("maximize", True)),
        oracle=oracle,
    )


def normalized_objective(y: np.ndarray) -> np.ndarray:
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo if hi > lo else 1.0
    return (y - lo) / span


def distribution_shape_metrics(y: np.ndarray) -> dict:
    """Spread / asymmetry on per-dataset normalized objectives."""
    yn = normalized_objective(y)
    q25, q75 = np.quantile(yn, [0.25, 0.75])
    std = float(yn.std()) or 1.0
    centred = (yn - yn.mean()) / std
    return {
        "y_norm_iqr": float(q75 - q25),
        "y_norm_skew": float(np.mean(centred ** 3)),
        "y_norm_kurtosis": float(np.mean(centred ** 4) - 3.0),
    }


def oracle_grid_std(panel: DatasetPanel, grid_pts: np.ndarray) -> float | None:
    """True analytic-oracle roughness on the shared ternary grid (synthetics only)."""
    if panel.oracle is None:
        return None
    from synthetic_data.oracles import build_oracle

    meta_path = Path(panel.source).with_name(Path(panel.source).stem + "_meta.json")
    meta = load_metadata(meta_path) if meta_path.is_file() else {}
    layout = str(meta.get("layout", "2"))
    seed = int(meta.get("seed", 42))
    fn, _, _ = build_oracle(panel.oracle, 3, layout, seed=seed)
    vals = np.array([float(fn(x)) for x in grid_pts], dtype=float)
    return float(np.std(vals))


def rf_metrics(X: np.ndarray, y: np.ndarray, grid_pts: np.ndarray) -> dict:
    rf = train_rf(X, y)
    train_r2 = float(rf.score(X, y))
    grid_vals = rf.predict(grid_pts)
    resid = y - rf.predict(X)
    # Basin volume: fraction of uniform simplex grid in top quartile of RF predictions.
    top_q = float(np.quantile(grid_vals, 0.75))
    basin_frac = float((grid_vals >= top_q).mean())
    return {
        "rf_train_r2": train_r2,
        "rf_grid_std": float(np.std(grid_vals)),
        "rf_grid_range": float(grid_vals.max() - grid_vals.min()),
        "rf_resid_std": float(np.std(resid)),
        "rf_basin_topq_frac": basin_frac,
        "rf_grid_vals": grid_vals,
        "rf": rf,
    }


def sampling_metrics(X: np.ndarray, y: np.ndarray) -> dict:
    centroid = np.full(X.shape[1], 1.0 / X.shape[1])
    dist_centre = np.linalg.norm(X - centroid, axis=1)
    # Chord-length proxy: consecutive rows (campaign line order; synthetic line blocks).
    step_dists = np.linalg.norm(np.diff(X, axis=0), axis=1) if len(X) > 1 else np.array([])
    return {
        "dist_centre_mean": float(dist_centre.mean()),
        "dist_centre_std": float(dist_centre.std()),
        "step_dist_median": float(np.median(step_dists)) if len(step_dists) else 0.0,
        "step_dist_p90": float(np.quantile(step_dists, 0.9)) if len(step_dists) else 0.0,
        "comp_std_mean": float(X.std(axis=0).mean()),
    }


def objective_summary(y: np.ndarray) -> dict:
    return {
        "n": len(y),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
        "y_p25": float(np.quantile(y, 0.25)),
        "y_p50": float(np.quantile(y, 0.50)),
        "y_p75": float(np.quantile(y, 0.75)),
    }


def build_metrics_table(panels: list[DatasetPanel], grid_pts: np.ndarray) -> pd.DataFrame:
    rows = []
    for panel in panels:
        obj = objective_summary(panel.y)
        samp = sampling_metrics(panel.X, panel.y)
        shape = distribution_shape_metrics(panel.y)
        rf = rf_metrics(panel.X, panel.y, grid_pts)
        o_std = oracle_grid_std(panel, grid_pts)
        rf_std = rf["rf_grid_std"]
        rows.append({
            "dataset": panel.key,
            "label": panel.label,
            "maximize": panel.maximize,
            "source": panel.source,
            **obj,
            **samp,
            **shape,
            "rf_train_r2": rf["rf_train_r2"],
            "rf_grid_std": rf_std,
            "rf_grid_range": rf["rf_grid_range"],
            "rf_resid_std": rf["rf_resid_std"],
            "rf_basin_topq_frac": rf["rf_basin_topq_frac"],
            "oracle_grid_std": o_std,
            "rf_over_oracle_std": (rf_std / o_std) if o_std else None,
        })
    return pd.DataFrame(rows)


def _dataset_colors(keys: list[str]) -> list[str]:
    return ["#2166ac" if k == "campaign1a" else "#ef8a62" for k in keys]


def plot_rf_r2_bars(ax, metrics_df: pd.DataFrame) -> None:
    order = metrics_df.sort_values("rf_train_r2")
    colors = _dataset_colors(order["dataset"].tolist())
    bars = ax.barh(order["dataset"], order["rf_train_r2"], color=colors, edgecolor="white", height=0.7)
    ax.axvline(1.0, color="#999", ls="--", lw=0.8)
    for bar, val in zip(bars, order["rf_train_r2"]):
        ax.text(
            min(val + 0.008, 1.01), bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=7,
        )
    ax.set_xlim(0.82, 1.03)
    ax.set_xlabel("RF train R²")
    ax.set_title("RF surrogate fit (lower = harder to interpolate)", fontsize=10)
    ax.grid(axis="x", alpha=0.25)


def plot_roughness_comparison(ax, metrics_df: pd.DataFrame) -> None:
    """True oracle grid σ vs RF grid σ — shows smoothing / miss of fine structure."""
    names = metrics_df["dataset"].tolist()
    x = np.arange(len(names))
    w = 0.36
    rf = metrics_df["rf_grid_std"].to_numpy(dtype=float)
    oracle = metrics_df["oracle_grid_std"].to_numpy(dtype=float)
    ax.bar(x - w / 2, rf, w, label="RF grid σ", color="#67a9cf", edgecolor="white")
    has_oracle = ~np.isnan(oracle)
    if has_oracle.any():
        ax.bar(
            x[has_oracle] + w / 2, oracle[has_oracle], w,
            label="true oracle grid σ", color="#b2182b", edgecolor="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("grid prediction σ (log)")
    ax.set_title("Landscape roughness: RF vs true oracle", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.25)


def plot_norm_iqr_bars(ax, metrics_df: pd.DataFrame) -> None:
    """IQR of min–max normalized objectives — how spread out sample values are."""
    order = metrics_df.sort_values("y_norm_iqr")
    colors = _dataset_colors(order["dataset"].tolist())
    ax.barh(order["dataset"], order["y_norm_iqr"], color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("normalized objective IQR")
    ax.set_title("Sample spread (0 = collapsed, 1 = full range used)", fontsize=10)
    ax.set_xlim(0, 1.0)
    ax.grid(axis="x", alpha=0.25)


def plot_campaign_property_cv(ax, campaign_df: pd.DataFrame) -> None:
    """Coefficient of variation per measured campaign property."""
    props, cvs = [], []
    for prop in CAMPAIGN_PROPERTIES:
        if prop not in campaign_df.columns:
            continue
        v = campaign_df[prop].dropna().values.astype(float)
        denom = abs(float(v.mean())) or 1.0
        props.append(prop)
        cvs.append(float(v.std() / denom))
    if not props:
        ax.text(0.5, 0.5, "no properties", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.bar(props, cvs, color="#2166ac", edgecolor="white")
    ax.set_ylabel("coefficient of variation (σ/|μ|)")
    ax.set_title("campaign1a — property variability", fontsize=10)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)


def plot_objective_histograms(ax, panels: list[DatasetPanel]) -> None:
    all_y = np.concatenate([p.y for p in panels])
    bins = np.linspace(all_y.min(), all_y.max(), 40)
    for panel in panels:
        ax.hist(
            panel.y, bins=bins, alpha=0.45, density=True,
            label=panel.key, edgecolor="white", linewidth=0.4,
        )
    ax.set_xlabel("Objective (raw)")
    ax.set_ylabel("density")
    ax.set_title("Objective distributions (shared bins)", fontsize=10)
    ax.legend(fontsize=7, ncol=2)


def plot_normalized_cdfs(ax, panels: list[DatasetPanel]) -> None:
    t = np.linspace(0, 1, 200)
    for panel in panels:
        lo, hi = panel.y.min(), panel.y.max()
        span = hi - lo if hi > lo else 1.0
        y_norm = np.sort((panel.y - lo) / span)
        cdf = np.searchsorted(y_norm, t, side="right") / len(y_norm)
        ax.plot(t, cdf, lw=1.8, label=panel.key)
    ax.set_xlabel("normalized objective (0=worst, 1=best in dataset)")
    ax.set_ylabel("CDF")
    ax.set_title("Per-dataset normalized objective CDFs", fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)


def plot_rf_quality_scatter(ax, metrics_df: pd.DataFrame) -> None:
    ax.scatter(
        metrics_df["rf_train_r2"], metrics_df["rf_grid_std"],
        s=70, c=range(len(metrics_df)), cmap="tab10", edgecolors="black", linewidths=0.5,
    )
    for _, row in metrics_df.iterrows():
        ax.annotate(row["dataset"], (row["rf_train_r2"], row["rf_grid_std"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("RF train R²")
    ax.set_ylabel("RF grid prediction std (roughness)")
    ax.set_title("Surrogate fit vs landscape roughness", fontsize=10)
    ax.grid(alpha=0.25)


def plot_rf_residual_bars(ax, metrics_df: pd.DataFrame) -> None:
    order = metrics_df.sort_values("rf_resid_std")
    colors = _dataset_colors(order["dataset"].tolist())
    ax.barh(order["dataset"], order["rf_resid_std"], color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("RF residual σ on training samples")
    ax.set_title("RF interpolation error (higher = noisier / less smooth fit)", fontsize=10)
    ax.grid(axis="x", alpha=0.25)


def plot_step_length_cdfs(ax, panels: list[DatasetPanel]) -> None:
    """Consecutive-sample chord lengths — reveals line-structured campaign sampling."""
    for panel in panels:
        if len(panel.X) < 2:
            continue
        steps = np.linalg.norm(np.diff(panel.X, axis=0), axis=1)
        t = np.sort(steps)
        cdf = np.arange(1, len(t) + 1) / len(t)
        ax.plot(t, cdf, lw=1.6, label=panel.key)
    ax.set_xlabel("consecutive sample step length ‖Δx‖₂")
    ax.set_ylabel("CDF")
    ax.set_title("Sampling geometry — line continuity", fontsize=10)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.25)


def plot_ternary_samples(ax, panel: DatasetPanel, *, max_pts: int = 700) -> None:
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    idx = np.arange(len(panel.X))
    if len(idx) > max_pts:
        idx = np.random.default_rng(0).choice(idx, size=max_pts, replace=False)
    Xs, ys = panel.X[idx], panel.y[idx]
    xy = comp_to_xy(Xs)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=ys, cmap="viridis", s=6, alpha=0.75, linewidths=0)
    ax.set_title(panel.key, fontsize=8)
    return sc


def plot_dist_from_centre(ax, panels: list[DatasetPanel]) -> None:
    for panel in panels:
        c = np.full(panel.X.shape[1], 1.0 / panel.X.shape[1])
        d = np.linalg.norm(panel.X - c, axis=1)
        ax.scatter(d, panel.y, s=8, alpha=0.35, label=panel.key)
    ax.set_xlabel("‖x − centroid‖₂")
    ax.set_ylabel("Objective")
    ax.set_title("Sampling bias vs objective", fontsize=10)
    ax.legend(fontsize=6, ncol=2)


def make_dashboard(
    panels: list[DatasetPanel],
    campaign_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    *,
    output_png: Path,
    show: bool,
) -> None:
    grid_pts = ternary_grid(TERNARY_GRID_N)
    grid_xy = comp_to_xy(grid_pts)

    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(6, 3, figure=fig, hspace=0.38, wspace=0.28)

    ax_a = fig.add_subplot(gs[0, 0])
    plot_rf_r2_bars(ax_a, metrics_df)
    ax_b = fig.add_subplot(gs[0, 1])
    plot_roughness_comparison(ax_b, metrics_df)
    ax_c = fig.add_subplot(gs[0, 2])
    plot_norm_iqr_bars(ax_c, metrics_df)

    ax_d = fig.add_subplot(gs[1, 0])
    plot_campaign_property_cv(ax_d, campaign_df)
    ax_e = fig.add_subplot(gs[1, 1:])
    plot_objective_histograms(ax_e, panels)

    ax_f = fig.add_subplot(gs[2, 0])
    plot_normalized_cdfs(ax_f, panels)
    ax_g = fig.add_subplot(gs[2, 1])
    plot_rf_quality_scatter(ax_g, metrics_df)
    ax_h = fig.add_subplot(gs[2, 2])
    plot_rf_residual_bars(ax_h, metrics_df)

    ax_i = fig.add_subplot(gs[3, :2])
    plot_step_length_cdfs(ax_i, panels)
    ax_j = fig.add_subplot(gs[3, 2])
    plot_dist_from_centre(ax_j, panels)

    # Ternary sample coverage (first 6 panels).
    sample_panels = panels[:6]
    inner_gs = gs[4, :].subgridspec(1, len(sample_panels), wspace=0.05)
    for j, panel in enumerate(sample_panels):
        ax = fig.add_subplot(inner_gs[j])
        plot_ternary_samples(ax, panel)

    # RF landscapes on shared grid.
    landscape_panels = panels[:6]
    inner_gs2 = gs[5, :].subgridspec(1, len(landscape_panels), wspace=0.05)
    for j, panel in enumerate(landscape_panels):
        ax = fig.add_subplot(inner_gs2[j])
        rf = train_rf(panel.X, panel.y)
        vals = rf.predict(grid_pts)
        ax.scatter(grid_xy[:, 0], grid_xy[:, 1], c=vals, cmap="viridis", s=2, alpha=0.9, rasterized=True)
        ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"RF: {panel.key}", fontsize=7)

    row_titles = (
        "Row 1 — RF R², landscape roughness, sample spread",
        "Row 2 — campaign1a property variability + raw objective scale",
        "Row 3 — CDFs, RF fit vs roughness, RF residual error",
        "Row 4 — Sampling line continuity + bias vs centroid",
        "Row 5 — Ternary sample maps (color = objective)",
        "Row 6 — RF surrogate landscapes (comparison only)",
    )
    for row_idx, title in enumerate(row_titles):
        fig.text(
            0.01, 0.92 - row_idx * 0.155, title,
            fontsize=9, fontweight="bold", color="#333", va="top", rotation=90,
        )

    fig.suptitle(
        "campaign1a vs synthetic 3D datasets\n"
        "(MOBO uses direct oracles on synthetics; RF panels = surrogate comparison vs campaign1a)",
        fontsize=13, y=0.998,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"Saved dashboard → {output_png}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def make_regime_comparison_figure(
    panels: list[DatasetPanel],
    metrics_df: pd.DataFrame,
    *,
    output_png: Path,
    show: bool,
) -> None:
    """Focused figure: RF R², CDFs, roughness."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    plot_rf_r2_bars(axes[0], metrics_df)
    plot_normalized_cdfs(axes[1], panels)
    plot_roughness_comparison(axes[2], metrics_df)

    fig.suptitle("Key metrics — campaign1a vs synthetic", fontsize=12)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    print(f"Saved regime figure → {output_png}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare campaign1a with synthetic 3D datasets.")
    parser.add_argument("--campaign1a", default=None, help="Path to campaign1a.csv")
    parser.add_argument(
        "--synthetic-dir",
        default=str(DATA_DIR),
        help="Directory with campaign3d_synthetic_*.csv files",
    )
    parser.add_argument(
        "--output",
        default=str(PLOTS_DIR / "campaign_comparison_dashboard.png"),
        help="Output dashboard PNG",
    )
    parser.add_argument(
        "--regime-output",
        default=str(PLOTS_DIR / "campaign_regime_comparison.png"),
        help="Output focused regime PNG",
    )
    parser.add_argument(
        "--metrics-output",
        default=str(PLOTS_DIR / "campaign_comparison_metrics.csv"),
        help="Output metrics CSV",
    )
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    campaign_path = resolve_campaign1a_path(args.campaign1a)
    syn_dir = Path(args.synthetic_dir)

    panels: list[DatasetPanel] = [load_campaign1a(campaign_path)]
    for csv_path in sorted(syn_dir.glob("campaign3d_synthetic_*.csv")):
        if "_meta" in csv_path.name:
            continue
        try:
            panels.append(load_synthetic_csv(csv_path))
        except Exception as exc:
            print(f"  [skip] {csv_path.name}: {exc}")

    # Stable order: campaign first, then oracle name order.
    synth_order = {name: i for i, name in enumerate(ORACLE_CHOICES)}
    panels = [panels[0]] + sorted(
        panels[1:],
        key=lambda p: synth_order.get(p.key, 99),
    )

    print("=" * 70)
    print("Campaign vs synthetic comparison")
    for p in panels:
        print(f"  {p.label}  |  maximize={p.maximize}  |  {p.source}")
    print("=" * 70)

    grid_pts = ternary_grid(TERNARY_GRID_N)
    metrics_df = build_metrics_table(panels, grid_pts)
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics → {metrics_path}")

    campaign_df = pd.read_csv(campaign_path)
    make_dashboard(
        panels, campaign_df, metrics_df,
        output_png=Path(args.output),
        show=not args.no_show,
    )
    make_regime_comparison_figure(
        panels, metrics_df,
        output_png=Path(args.regime_output),
        show=not args.no_show,
    )
    print("Done.")


if __name__ == "__main__":
    main()
