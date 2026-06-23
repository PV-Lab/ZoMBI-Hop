"""
generate_synthetic_campaign.py
================================
Build synthetic composition "campaign" CSVs from planted oracles (3D–10D+),
train an RF surrogate, and optionally plot oracle vs RF on a ternary grid (d=3).

Usage
-----
  conda activate zombi-hop-linebo
  python synthetic_data/generate_synthetic_campaign.py

  # All standard 3D presets for MOBO benchmarking:
  python synthetic_data/generate_synthetic_campaign.py --all-3d

  # 10D messy campaign (for high-D RF MOBO once run_mobo supports d>3 RF):
  python synthetic_data/generate_synthetic_campaign.py --dim 10 --oracle messy

  MPLBACKEND=Agg python synthetic_data/generate_synthetic_campaign.py --no-show
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib

if not sys.stdin.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from synthetic_data.campaign_datasets import (
    DEFAULT_LAYOUT,
    OBJECTIVE_COL,
    RF_N_ESTIMATORS,
    SYNTHETIC_3D_PRESETS,
    TARGET_DATASET_SIZE,
    composition_column_names,
    generate_campaign_files,
    metadata_path_for_csv,
    normalize_rows,
    train_rf,
)
from synthetic_data.oracles import ORACLE_CHOICES

TERNARY_GRID_N = 120
_SQRT3_2 = math.sqrt(3) / 2
CAMPAIGN1A_COLS = ["FAPbI3", "MAPbI3", "MAPbBr3"]
CAMPAIGN1A_CORNER_LABELS = tuple(CAMPAIGN1A_COLS)


def comp_to_xy(comp: np.ndarray) -> np.ndarray:
    p = np.asarray(comp, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    s = p.sum(axis=-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return np.column_stack([p[:, 1] + 0.5 * p[:, 2], _SQRT3_2 * p[:, 2]])


def draw_ternary_frame(
    ax,
    pad: float = 0.04,
    corner_labels: tuple[str, str, str] | None = None,
) -> None:
    labels = corner_labels or tuple(composition_column_names(3))
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, _SQRT3_2 + 0.16)
    ax.axis("off")
    ax.text(-pad, -pad, labels[0], ha="right", va="top", fontsize=9)
    ax.text(1 + pad, -pad, labels[1], ha="left", va="top", fontsize=9)
    ax.text(0.5, _SQRT3_2 + pad, labels[2], ha="center", va="bottom", fontsize=9)


def ternary_grid(n: int = TERNARY_GRID_N) -> np.ndarray:
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append([i / n, j / n, (n - i - j) / n])
    return np.array(pts, dtype=float)


def resolve_campaign1a_path(explicit: str | None = None) -> str:
    if explicit and os.path.isfile(explicit):
        return os.path.normpath(explicit)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        explicit,
        os.path.join(script_dir, "data", "campaign1a.csv"),
        os.path.join(script_dir, "campaign1a.csv"),
        os.path.join(script_dir, "..", "data", "campaign1a.csv"),
        os.path.join(script_dir, "..", "interactive_testing", "data", "campaign1a.csv"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.normpath(path)
    tried = "\n".join(f"  {os.path.normpath(p)}" for p in candidates if p)
    raise FileNotFoundError(f"campaign1a.csv not found. Tried:\n{tried}")


def load_campaign1a(csv_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=CAMPAIGN1A_COLS + [OBJECTIVE_COL])
    X = normalize_rows(df[CAMPAIGN1A_COLS].values.astype(float))
    y = df[OBJECTIVE_COL].values.astype(float)
    return X, y, len(df)


def train_hgb(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
    hgb = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_depth=8, random_state=42
    )
    hgb.fit(X, y)
    return hgb


def plot_ternary_landscapes(
    *,
    grid_pts: np.ndarray,
    panels: list[dict],
    save_path: str | None,
    show: bool,
) -> None:
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(7.5 * n_panels, 6.8))
    if n_panels == 1:
        axes = [axes]
    gxy = comp_to_xy(grid_pts)

    for ax, panel in zip(axes, panels):
        corner_labels = panel.get("corner_labels", tuple(composition_column_names(3)))
        draw_ternary_frame(ax, corner_labels=corner_labels)
        ax.set_title(panel["title"], fontsize=10)
        sc = ax.scatter(
            gxy[:, 0], gxy[:, 1], c=panel["vals"],
            cmap="viridis", s=8, alpha=0.80, zorder=2, rasterized=True,
        )
        if panel.get("reference_optima"):
            ref_xy = comp_to_xy(np.array(panel["reference_optima"]))
            ax.scatter(
                ref_xy[:, 0], ref_xy[:, 1], marker="*", s=340, c="blue",
                zorder=12, edgecolors="navy", linewidths=1.3,
            )
        fig.colorbar(sc, ax=ax, label="Objective", fraction=0.046, pad=0.04)

    fig.suptitle("Δ³ surrogate comparison — synthetic vs campaign1a RF", fontsize=11, y=1.02)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot → {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def default_output_paths(oracle: str, dim: int) -> tuple[str, str]:
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    stem = f"campaign{dim}d_synthetic_{oracle}"
    return (
        os.path.join(data_dir, f"{stem}.csv"),
        os.path.join(data_dir, "plots", f"{stem}.png"),
    )


def run_one(
    *,
    oracle: str,
    dim: int,
    layout: str,
    seed: int,
    n_samples: int,
    sampling: str,
    noise_std: float,
    outlier_frac: float,
    output: str,
    plot_path: str | None,
    with_hgb: bool,
    skip_campaign1a: bool,
    campaign1a: str | None,
    show: bool,
) -> dict:
    from synthetic_data.oracles import build_oracle

    print("=" * 70)
    print(f"Synthetic {dim}D campaign — oracle={oracle}")
    print("=" * 70)

    df, meta = generate_campaign_files(
        output_csv=output,
        oracle=oracle,
        dim=dim,
        layout=layout,
        seed=seed,
        n_samples=n_samples,
        sampling=sampling,
        noise_std=noise_std,
        outlier_frac=outlier_frac,
    )
    print(f"  Oracle: {meta['oracle_label']}")
    print(f"  Sampling: {meta['sampling_desc']}")
    print(f"  {len(df)} rows → {output}")
    meta_out = metadata_path_for_csv(output)
    print(f"  metadata → {meta_out}")

    comp_cols = meta["composition_columns"]
    X = normalize_rows(df[comp_cols].values.astype(float))
    y = df[OBJECTIVE_COL].values.astype(float)
    rf = train_rf(X, y)
    print(f"  RF train R² = {rf.score(X, y):.4f}")

    obj, centers, _ = build_oracle(oracle, dim, layout, seed=seed)
    for i, c in enumerate(centers):
        print(f"    peak {i + 1}: {np.round(c, 4)}  y={obj(c):.4f}")

    if dim == 3 and plot_path:
        grid_pts = ternary_grid(TERNARY_GRID_N)
        oracle_vals = np.array([obj(x) for x in grid_pts])
        panels = [
            {"title": f"Oracle: {meta['oracle_label']}", "vals": oracle_vals, "reference_optima": centers},
            {"title": f"Synthetic RF (R²={rf.score(X, y):.4f})", "vals": rf.predict(grid_pts), "reference_optima": centers},
        ]
        if with_hgb:
            hgb = train_hgb(X, y)
            panels.append({
                "title": f"Synthetic HGB (R²={hgb.score(X, y):.4f})",
                "vals": hgb.predict(grid_pts),
                "reference_optima": centers,
            })
        if not skip_campaign1a:
            c1a_path = resolve_campaign1a_path(campaign1a)
            X_c1a, y_c1a, n_c1a = load_campaign1a(c1a_path)
            rf_c1a = train_rf(X_c1a, y_c1a)
            panels.append({
                "title": f"campaign1a RF ({n_c1a} rows, R²={rf_c1a.score(X_c1a, y_c1a):.4f})",
                "vals": rf_c1a.predict(grid_pts),
                "corner_labels": CAMPAIGN1A_CORNER_LABELS,
            })
        plot_ternary_landscapes(grid_pts=grid_pts, panels=panels, save_path=plot_path, show=show)

    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic campaign CSV + metadata.")
    parser.add_argument("--dim", type=int, default=3, help="Simplex dimension (default 3).")
    parser.add_argument("--layout", default=DEFAULT_LAYOUT, choices=["1", "2", "3"])
    parser.add_argument("--oracle", choices=ORACLE_CHOICES, default="messy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=TARGET_DATASET_SIZE)
    parser.add_argument("--sampling", choices=("campaign", "uniform"), default="campaign")
    parser.add_argument("--noise-std", type=float, default=0.07)
    parser.add_argument("--outlier-frac", type=float, default=0.06)
    parser.add_argument("--output", default=None)
    parser.add_argument("--plot", default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--with-hgb", action="store_true")
    parser.add_argument("--campaign1a", default=None)
    parser.add_argument("--skip-campaign1a", action="store_true")
    parser.add_argument(
        "--all-3d",
        action="store_true",
        help="Generate all standard 3D synthetic presets (11 expression families).",
    )
    args = parser.parse_args()

    if args.all_3d:
        for name in SYNTHETIC_3D_PRESETS:
            out, plot = default_output_paths(name, 3)
            run_one(
                oracle=name,
                dim=3,
                layout=args.layout,
                seed=args.seed,
                n_samples=args.n_samples,
                sampling=args.sampling,
                noise_std=args.noise_std,
                outlier_frac=args.outlier_frac,
                output=out,
                plot_path=plot,
                with_hgb=args.with_hgb,
                skip_campaign1a=args.skip_campaign1a,
                campaign1a=args.campaign1a,
                show=not args.no_show,
            )
        print("Done (all 3D presets).")
        return

    output = args.output or default_output_paths(args.oracle, args.dim)[0]
    plot_path = args.plot
    if plot_path is None and args.dim == 3:
        plot_path = default_output_paths(args.oracle, args.dim)[1]

    run_one(
        oracle=args.oracle,
        dim=args.dim,
        layout=args.layout,
        seed=args.seed,
        n_samples=args.n_samples,
        sampling=args.sampling,
        noise_std=args.noise_std,
        outlier_frac=args.outlier_frac,
        output=output,
        plot_path=plot_path,
        with_hgb=args.with_hgb,
        skip_campaign1a=args.skip_campaign1a,
        campaign1a=args.campaign1a,
        show=not args.no_show,
    )
    print("Done.")


if __name__ == "__main__":
    main()
