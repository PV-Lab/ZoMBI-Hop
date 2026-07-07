"""
visualization/geometry_validation.py
====================================
Geometry validation for the ILR → compositional-scale change.

On real simplex data (default: ``data/2nd_real_run.db``), compare Euclidean
composition distance vs Aitchison distance where the algorithm actually operates
(at the measured input-noise scale, default ℓ = 0.064 composition L₂).

Reports
-------
  * Correlation and ratio d_A / d_E for neighbor pairs (d_E ≤ ℓ)
  * How the ratio varies with corner proximity and distance from simplex centre
  * |ΔY| vs d_E and vs d_A for each measured property (neighbor pairs)
  * Ternary overlays: sample locations, local metric distortion, radial maps

Usage
-----
  python visualization/geometry_validation.py
  python visualization/geometry_validation.py --db data/2nd_real_run.db \\
      --out-dir data/2nd_real_run_geometry
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
sys.path.insert(0, str(ROOT))

from synthetic_data.plot_aitchison_vs_euclidean import (  # noqa: E402
    CENTER_3,
    aitchison_distance,
    comp_to_xy,
    draw_ternary_frame,
    euclidean_distance,
    simplex_rgb,
    ternary_grid,
)

DB_COMP_COLS = ["FAPbI3", "MAPbI3", "MAPbBr3"]
DB_PROPERTIES = ["Bandgap", "Photoconductance", "Stability", "Objective"]
DEFAULT_DB = "2nd_real_run.db"
DEFAULT_INPUT_NOISE = 0.064
_SQRT3_2 = np.sqrt(3) / 2
CORNER_LABELS = ("FAPbI3", "MAPbI3", "MAPbBr3")


def _resolve_db_path(db_arg: str) -> Path:
    p = Path(db_arg)
    if p.is_file():
        return p
    candidate = ROOT / "data" / db_arg
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Data file not found: {db_arg}")


def _load_db_rows(db_path: Path, cols: list[str]) -> np.ndarray:
    con = sqlite3.connect(str(db_path))
    try:
        sel = ", ".join(f'"{c}"' for c in cols)
        where = " AND ".join(f'"{c}" IS NOT NULL' for c in cols)
        rows = con.execute(f"SELECT {sel} FROM results WHERE {where}").fetchall()
    finally:
        con.close()
    return np.asarray(rows, dtype=float)


def load_db_compositions(db_path: Path) -> np.ndarray:
    arr = _load_db_rows(db_path, DB_COMP_COLS)
    X = arr[:, :3]
    s = X.sum(axis=1, keepdims=True)
    return X / np.where(s == 0, 1.0, s)


def load_db_compositions_and_properties(
    db_path: Path, properties: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    cols = DB_COMP_COLS + properties
    arr = _load_db_rows(db_path, cols)
    if arr.shape[0] == 0:
        raise RuntimeError(f"No complete rows in {db_path}")
    X = arr[:, :3]
    s = X.sum(axis=1, keepdims=True)
    X = X / np.where(s == 0, 1.0, s)
    ys = {prop: arr[:, 3 + i] for i, prop in enumerate(properties)}
    return X, ys


def corner_proximity(X: np.ndarray) -> np.ndarray:
    """Distance to nearest simplex boundary: min component (0 at corners)."""
    return X.min(axis=1)


def dist_from_center(X: np.ndarray) -> np.ndarray:
    c = np.full(X.shape[1], 1.0 / X.shape[1])
    return np.linalg.norm(X - c, axis=1)


def collect_neighbor_pairs(
    X: np.ndarray, radius: float,
) -> dict[str, np.ndarray]:
    """All unordered pairs with 0 < d_E ≤ radius."""
    n = X.shape[0]
    nn = NearestNeighbors(radius=radius, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    neighbors = nn.radius_neighbors(X, return_distance=True)

    i_idx: list[int] = []
    j_idx: list[int] = []
    d_e_list: list[float] = []
    d_a_list: list[float] = []

    for i in range(n):
        for j, d_e in zip(neighbors[1][i], neighbors[0][i]):
            j = int(j)
            if j <= i or d_e <= 1e-12:
                continue
            d_a = float(aitchison_distance(X[i : i + 1], X[j : j + 1])[0])
            i_idx.append(i)
            j_idx.append(j)
            d_e_list.append(float(d_e))
            d_a_list.append(d_a)

    i_arr = np.asarray(i_idx, dtype=int)
    j_arr = np.asarray(j_idx, dtype=int)
    d_e = np.asarray(d_e_list, dtype=float)
    d_a = np.asarray(d_a_list, dtype=float)
    ratio = d_a / np.maximum(d_e, 1e-12)

    mid = 0.5 * (X[i_arr] + X[j_arr])
    return {
        "i": i_arr,
        "j": j_arr,
        "d_euclidean": d_e,
        "d_aitchison": d_a,
        "ratio": ratio,
        "corner_prox_mid": corner_proximity(mid),
        "dist_center_mid": dist_from_center(mid),
        "corner_prox_min": np.minimum(corner_proximity(X[i_arr]), corner_proximity(X[j_arr])),
    }


def per_point_local_ratio(X: np.ndarray, radius: float) -> np.ndarray:
    """Median d_A / d_E over Euclidean neighbors within radius (per point)."""
    pairs = collect_neighbor_pairs(X, radius)
    n = X.shape[0]
    buckets: list[list[float]] = [[] for _ in range(n)]
    for i, j, r in zip(pairs["i"], pairs["j"], pairs["ratio"]):
        buckets[i].append(r)
        buckets[j].append(r)
    out = np.full(n, np.nan)
    for i, vals in enumerate(buckets):
        if vals:
            out[i] = float(np.median(vals))
    return out


def summarize_pairs(pairs: dict[str, np.ndarray]) -> dict:
    d_e = pairs["d_euclidean"]
    d_a = pairs["d_aitchison"]
    ratio = pairs["ratio"]
    if len(d_e) == 0:
        return {"n_pairs": 0}

    pearson = float(stats.pearsonr(d_e, d_a)[0]) if len(d_e) > 2 else float("nan")
    spearman = float(stats.spearmanr(d_e, d_a)[0]) if len(d_e) > 2 else float("nan")
    log_pearson = (
        float(stats.pearsonr(np.log(d_e + 1e-12), np.log(d_a + 1e-12))[0])
        if len(d_e) > 2 else float("nan")
    )

    corner = pairs["corner_prox_min"]
    center = pairs["dist_center_mid"]
    corner_corr = (
        float(stats.spearmanr(corner, ratio)[0]) if len(ratio) > 2 else float("nan")
    )
    center_corr = (
        float(stats.spearmanr(center, ratio)[0]) if len(ratio) > 2 else float("nan")
    )

    return {
        "n_pairs": int(len(d_e)),
        "d_euclidean": {
            "median": float(np.median(d_e)),
            "p90": float(np.percentile(d_e, 90)),
            "max": float(np.max(d_e)),
        },
        "d_aitchison": {
            "median": float(np.median(d_a)),
            "p90": float(np.percentile(d_a, 90)),
            "max": float(np.max(d_a)),
        },
        "ratio_dA_over_dE": {
            "median": float(np.median(ratio)),
            "p10": float(np.percentile(ratio, 10)),
            "p90": float(np.percentile(ratio, 90)),
            "max": float(np.max(ratio)),
        },
        "correlation_dE_dA": {
            "pearson": pearson,
            "spearman": spearman,
            "log_log_pearson": log_pearson,
        },
        "ratio_vs_corner_proximity_spearman": corner_corr,
        "ratio_vs_dist_from_center_spearman": center_corr,
        "fraction_ratio_above_1.5": float(np.mean(ratio > 1.5)),
        "fraction_ratio_above_2.0": float(np.mean(ratio > 2.0)),
    }


def property_dy_correlations(
    pairs: dict[str, np.ndarray], ys: dict[str, np.ndarray],
) -> dict[str, dict]:
    i, j = pairs["i"], pairs["j"]
    d_e, d_a = pairs["d_euclidean"], pairs["d_aitchison"]
    out: dict[str, dict] = {}
    for prop, y in ys.items():
        dy = np.abs(y[i] - y[j])
        out[prop] = {
            "median_abs_dy": float(np.median(dy)),
            "spearman_dy_vs_dE": float(stats.spearmanr(dy, d_e)[0]) if len(dy) > 2 else float("nan"),
            "spearman_dy_vs_dA": float(stats.spearmanr(dy, d_a)[0]) if len(dy) > 2 else float("nan"),
        }
    return out


def plot_metric_scatter_real(
    pairs: dict[str, np.ndarray],
    X: np.ndarray,
    out_path: Path,
    input_noise: float,
) -> None:
    d_e = pairs["d_euclidean"]
    d_a = pairs["d_aitchison"]
    mid = 0.5 * (X[pairs["i"]] + X[pairs["j"]])
    rgb = simplex_rgb(mid)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(d_e, d_a, c=rgb, s=14, alpha=0.55, linewidths=0)
    lim = max(d_e.max(), input_noise) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5, label="d_A = d_E")
    ax.set_xlim(0, lim)
    ax.set_ylim(max(d_a.min() * 0.8, 1e-4), d_a.max() * 1.08)
    ax.set_xlabel("Euclidean distance (composition L₂)")
    ax.set_ylabel("Aitchison distance")
    ax.set_title(
        f"Neighbor pairs with d_E ≤ {input_noise:.3f}\n"
        f"(n={len(d_e):,}; color = midpoint composition)",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ratio_on_ternary(
    X: np.ndarray,
    local_ratio: np.ndarray,
    out_path: Path,
    input_noise: float,
) -> None:
    valid = np.isfinite(local_ratio)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    draw_ternary_frame(ax)
    ax.text(-0.04, -0.04, CORNER_LABELS[0], ha="right", va="top", fontsize=9)
    ax.text(1.04, -0.04, CORNER_LABELS[1], ha="left", va="top", fontsize=9)
    ax.text(0.5, _SQRT3_2 + 0.04, CORNER_LABELS[2], ha="center", va="bottom", fontsize=9)

    xy = comp_to_xy(X[valid])
    vals = local_ratio[valid]
    vmin, vmax = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
    sc = ax.scatter(
        xy[:, 0], xy[:, 1], c=vals, cmap="coolwarm",
        vmin=vmin, vmax=vmax, s=28, edgecolors="black", linewidths=0.4, zorder=5,
    )
    fig.colorbar(sc, ax=ax, label="median d_A / d_E (local)", fraction=0.046, pad=0.02)
    ax.set_title(
        f"Local metric distortion on real samples\n"
        f"(median d_A/d_E over Euclidean neighbors ≤ {input_noise:.3f})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_corner_vs_ratio(
    pairs: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.scatter(
        pairs["corner_prox_min"], pairs["ratio"],
        s=10, alpha=0.35, c="#4477aa", linewidths=0,
    )
    ax.set_xlabel("min composition component (pair)")
    ax.set_ylabel("d_A / d_E")
    ax.set_title("Metric distortion vs corner proximity\n(smaller min → nearer a corner)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_radial_with_real_overlay(
    X: np.ndarray,
    ref: np.ndarray,
    out_path: Path,
    grid_n: int = 80,
) -> None:
    grid = ternary_grid(grid_n)
    d_e_grid = euclidean_distance(grid, ref)
    d_a_grid = aitchison_distance(grid, ref)
    ratio_grid = d_a_grid / np.maximum(d_e_grid, 1e-12)

    d_e_pts = euclidean_distance(X, ref)
    d_a_pts = aitchison_distance(X, ref)
    ratio_pts = d_a_pts / np.maximum(d_e_pts, 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    xy_g = comp_to_xy(grid)
    ref_xy = comp_to_xy(ref.reshape(1, -1))[0]
    xy_pts = comp_to_xy(X)

    for ax, vals, title, cmap in zip(
        axes,
        [d_e_grid, d_a_grid, ratio_grid],
        ["Euclidean from ref", "Aitchison from ref", "d_A / d_E from ref"],
        ["Blues", "Oranges", "coolwarm"],
    ):
        tri = mtri.Triangulation(xy_g[:, 0], xy_g[:, 1])
        if "ratio" in title.lower():
            vmin, vmax = 0.8, 3.5
        else:
            vmin, vmax = 0.0, float(np.percentile(vals, 99))
        ax.tripcolor(tri, vals, cmap=cmap, vmin=vmin, vmax=vmax, shading="gouraud", rasterized=True)
        draw_ternary_frame(ax)
        ax.text(-0.04, -0.04, CORNER_LABELS[0], ha="right", va="top", fontsize=8)
        ax.text(1.04, -0.04, CORNER_LABELS[1], ha="left", va="top", fontsize=8)
        ax.text(0.5, _SQRT3_2 + 0.04, CORNER_LABELS[2], ha="center", va="bottom", fontsize=8)
        ax.plot(ref_xy[0], ref_xy[1], "k*", ms=14, zorder=11, markeredgecolor="white", markeredgewidth=0.5)
        ax.scatter(xy_pts[:, 0], xy_pts[:, 1], s=12, c="white", edgecolors="black", linewidths=0.35, zorder=12, alpha=0.9)
        ax.set_title(title, fontsize=10)

    fig.suptitle(
        f"Radial geometry + real sample overlay (ref = data centroid)\n"
        f"white dots: n={X.shape[0]} measured compositions",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dy_vs_metrics(
    pairs: dict[str, np.ndarray],
    ys: dict[str, np.ndarray],
    out_path: Path,
    properties: list[str],
) -> None:
    i, j = pairs["i"], pairs["j"]
    d_e, d_a = pairs["d_euclidean"], pairs["d_aitchison"]
    n = len(properties)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.8 * n), squeeze=False)

    for row, prop in enumerate(properties):
        dy = np.abs(ys[prop][i] - ys[prop][j])
        for col, dist, label in [(0, d_e, "Euclidean d"), (1, d_a, "Aitchison d")]:
            ax = axes[row, col]
            ax.scatter(dist, dy, s=8, alpha=0.3, c="#334488", linewidths=0)
            rho = stats.spearmanr(dy, dist)[0] if len(dy) > 2 else float("nan")
            ax.set_xlabel(label)
            if col == 0:
                ax.set_ylabel(f"|Δ{prop}|")
            ax.set_title(f"{prop}: |ΔY| vs {label}  (ρ={rho:.3f})", fontsize=9)
            ax.grid(True, alpha=0.25)

    fig.suptitle("Property change vs geometry (neighbor pairs)", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_summary(
    summary: dict, input_noise: float, X: np.ndarray,
) -> None:
    ps = summary["pair_summary"]
    print(
        f"Geometry points: {summary['n_points_geometry']}  "
        f"| property-complete: {summary['n_points_properties']}  "
        f"| neighbor pairs (d_E ≤ {input_noise:.4f}): {ps['n_pairs']}"
    )
    print()
    print("Distance stats (neighbor pairs):")
    print(f"  d_E  median={ps['d_euclidean']['median']:.4f}  p90={ps['d_euclidean']['p90']:.4f}")
    print(f"  d_A  median={ps['d_aitchison']['median']:.4f}  p90={ps['d_aitchison']['p90']:.4f}")
    print(f"  d_A/d_E  median={ps['ratio_dA_over_dE']['median']:.3f}  "
          f"p10–p90=[{ps['ratio_dA_over_dE']['p10']:.3f}, {ps['ratio_dA_over_dE']['p90']:.3f}]")
    print(f"  pairs with ratio > 1.5: {ps['fraction_ratio_above_1.5']*100:.1f}%  "
          f"> 2.0: {ps['fraction_ratio_above_2.0']*100:.1f}%")
    print()
    c = summary["correlation_dE_dA"]
    print("Correlation (neighbor pairs):")
    print(f"  Pearson(d_E, d_A)={c['pearson']:.4f}  Spearman={c['spearman']:.4f}  "
          f"log-log Pearson={c['log_log_pearson']:.4f}")
    print(f"  Spearman(ratio, corner proximity)={ps['ratio_vs_corner_proximity_spearman']:.4f}  "
          f"(negative → higher ratio near corners)")
    print(f"  Spearman(ratio, dist from centre)={ps['ratio_vs_dist_from_center_spearman']:.4f}")
    print()
    print("Sample region:")
    sr = summary["sample_region"]
    print(f"  min component: median={sr['corner_prox_median']:.3f}  "
          f"p10={sr['corner_prox_p10']:.3f}  (0 = on corner)")
    print(f"  dist from centre: median={sr['dist_center_median']:.3f}")
    print()
    print("|ΔY| correlation with distance (neighbor pairs):")
    for prop, row in summary["property_dy"].items():
        print(f"  {prop:18s}  Spearman(|ΔY|, d_E)={row['spearman_dy_vs_dE']:.3f}  "
              f"Spearman(|ΔY|, d_A)={row['spearman_dy_vs_dA']:.3f}")
    print()
    print("Interpretation:")
    interp = summary["interpretation"]
    for line in interp:
        print(f"  • {line}")


def _interpret(summary: dict, input_noise: float) -> list[str]:
    ps = summary["pair_summary"]
    sr = summary["sample_region"]
    lines: list[str] = []

    med_ratio = ps["ratio_dA_over_dE"]["median"]
    if med_ratio < 1.15:
        lines.append(
            f"At d_E ≤ {input_noise:.3f}, Aitchison ≈ Euclidean (median ratio {med_ratio:.2f}); "
            "compositional-scale GP geometry is close to ILR/Aitchison in the sampled region."
        )
    elif med_ratio < 1.5:
        lines.append(
            f"Moderate distortion: median d_A/d_E = {med_ratio:.2f} on neighbor pairs — "
            "compositional and Aitchison distances differ but are correlated."
        )
    else:
        lines.append(
            f"Strong distortion: median d_A/d_E = {med_ratio:.2f} — "
            "Euclidean composition distance understates compositional separation."
        )

    if ps["correlation_dE_dA"]["spearman"] > 0.9:
        lines.append("Rank ordering of neighbors is nearly identical under both metrics (Spearman > 0.9).")
    elif ps["correlation_dE_dA"]["spearman"] > 0.75:
        lines.append("Neighbor rankings are similar but not identical across metrics.")
    else:
        lines.append("Neighbor rankings diverge — geometry choice could change which points are 'close'.")

    if sr["corner_prox_p10"] < 0.08:
        lines.append(
            f"Some samples lie near simplex corners (10th pct min component = {sr['corner_prox_p10']:.3f}); "
            "metric distortion may be worse there."
        )
    else:
        lines.append(
            f"Samples avoid extreme corners (10th pct min component = {sr['corner_prox_p10']:.3f}); "
            "corner compression is less of a concern for this dataset."
        )

    rc = ps.get("ratio_vs_corner_proximity_spearman", float("nan"))
    if np.isfinite(rc) and rc < -0.2:
        lines.append("d_A/d_E rises near corners — ILR/Aitchison would weight those regions differently.")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Aitchison vs Euclidean validation on real data.")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--properties", default=",".join(DB_PROPERTIES))
    parser.add_argument("--input-noise", type=float, default=DEFAULT_INPUT_NOISE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--grid-n", type=int, default=80)
    args = parser.parse_args()

    db_path = _resolve_db_path(args.db)
    properties = [p.strip() for p in args.properties.split(",") if p.strip()]
    out_dir = args.out_dir or (db_path.parent / f"{db_path.stem}_geometry")

    X, ys = load_db_compositions_and_properties(db_path, properties)
    X_geom = load_db_compositions(db_path)
    pairs = collect_neighbor_pairs(X_geom, args.input_noise)
    pair_summary = summarize_pairs(pairs)
    local_ratio = per_point_local_ratio(X_geom, args.input_noise)
    ref = X_geom.mean(axis=0)
    ref = ref / ref.sum()

    # Property |ΔY| uses rows where all requested properties are present.
    pairs_y = collect_neighbor_pairs(X, args.input_noise)

    summary = {
        "database": str(db_path),
        "n_points_geometry": int(X_geom.shape[0]),
        "n_points_properties": int(X.shape[0]),
        "input_noise": args.input_noise,
        "reference_centroid": ref.tolist(),
        "pair_summary": pair_summary,
        "correlation_dE_dA": pair_summary.get("correlation_dE_dA", {}),
        "sample_region": {
            "corner_prox_median": float(np.median(corner_proximity(X_geom))),
            "corner_prox_p10": float(np.percentile(corner_proximity(X_geom), 10)),
            "corner_prox_p90": float(np.percentile(corner_proximity(X_geom), 90)),
            "dist_center_median": float(np.median(dist_from_center(X_geom))),
        },
        "property_dy": property_dy_correlations(pairs_y, ys),
        "local_ratio_median": float(np.nanmedian(local_ratio)),
    }
    summary["interpretation"] = _interpret(summary, args.input_noise)

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "geometry_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"Database: {db_path}")
    print(f"Output:   {out_dir}/")
    print()
    _print_summary(summary, args.input_noise, X_geom)

    plot_metric_scatter_real(pairs, X_geom, out_dir / "euclidean_vs_aitchison_pairs.png", args.input_noise)
    plot_ratio_on_ternary(X_geom, local_ratio, out_dir / "local_ratio_ternary.png", args.input_noise)
    plot_corner_vs_ratio(pairs, out_dir / "ratio_vs_corner.png")
    plot_radial_with_real_overlay(X_geom, ref, out_dir / "radial_maps_real_overlay.png", args.grid_n)
    plot_dy_vs_metrics(pairs_y, ys, out_dir / "dy_vs_metrics.png", properties)

    print(f"Wrote plots and {json_path.name} -> {out_dir}/")


if __name__ == "__main__":
    main()
