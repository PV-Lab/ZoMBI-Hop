"""Aitchison vs Euclidean distance on the 3-simplex (ternary + scatter)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

EPS = 1e-10
_SQRT3_2 = math.sqrt(3) / 2
CENTER_3 = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
REF_POINT_3 = np.array([0.10, 0.10, 0.80])
CORNER_LABELS = ("x₁", "x₂", "x₃")


def comp_to_xy(comp: np.ndarray) -> np.ndarray:
    p = np.asarray(comp, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    s = p.sum(axis=-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return np.column_stack([p[:, 1] + 0.5 * p[:, 2], _SQRT3_2 * p[:, 2]])


def simplex_rgb(comp: np.ndarray) -> np.ndarray:
    """x₁→R, x₂→G, x₃→B for ternary coloring."""
    p = np.asarray(comp, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    p = p / p.sum(axis=-1, keepdims=True)
    return np.clip(p, 0.0, 1.0)


def draw_ternary_frame(ax, pad: float = 0.04) -> None:
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.2, zorder=10)
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, _SQRT3_2 + 0.16)
    ax.axis("off")
    ax.text(-pad, -pad, CORNER_LABELS[0], ha="right", va="top", fontsize=9)
    ax.text(1 + pad, -pad, CORNER_LABELS[1], ha="left", va="top", fontsize=9)
    ax.text(0.5, _SQRT3_2 + pad, CORNER_LABELS[2], ha="center", va="bottom", fontsize=9)


def ternary_grid(n: int) -> np.ndarray:
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append([i / n, j / n, (n - i - j) / n])
    return np.asarray(pts, dtype=float)


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x - y, axis=-1)


def aitchison_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    log_x = np.log(x + EPS)
    log_y = np.log(y + EPS)
    clr_x = log_x - log_x.mean(axis=-1, keepdims=True)
    clr_y = log_y - log_y.mean(axis=-1, keepdims=True)
    return np.linalg.norm(clr_x - clr_y, axis=-1)


def _aitchison_lognorm(vals: np.ndarray) -> tuple[mcolors.LogNorm, float]:
    positive = vals[vals > 0]
    vmin = max(positive.min(), 1e-3) if len(positive) > 0 else 1e-3
    return mcolors.LogNorm(vmin=vmin, vmax=vals.max()), vmin


def _norm_for(vals: np.ndarray, metric: str) -> mcolors.Normalize:
    if metric == "aitchison":
        norm, _ = _aitchison_lognorm(vals)
        return norm
    return mcolors.Normalize(vmin=0.0, vmax=vals.max())


def plot_radial_ternary(
    ax,
    grid: np.ndarray,
    ref: np.ndarray,
    vals: np.ndarray,
    metric: str,
) -> plt.cm.ScalarMappable:
    xy = comp_to_xy(grid)
    cmap = "Blues" if metric == "euclidean" else "Oranges"
    label = "Euclidean distance" if metric == "euclidean" else "Aitchison distance (log scale)"

    norm = _norm_for(vals, metric)
    plot_vals = vals.astype(float)
    if metric == "aitchison":
        _, vmin = _aitchison_lognorm(vals)
        plot_vals = np.where(plot_vals > 0, plot_vals, vmin * 0.5)

    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    mappable = ax.tripcolor(
        tri, plot_vals, cmap=cmap, norm=norm, shading="gouraud", rasterized=True,
    )
    draw_ternary_frame(ax)

    ref_xy = comp_to_xy(ref.reshape(1, -1))[0]
    ax.plot(
        ref_xy[0], ref_xy[1], "k*", ms=16, zorder=11,
        markeredgecolor="white", markeredgewidth=0.6,
    )
    ax.set_title(label, fontsize=10)
    return mappable


def plot_metric_scatter(ax, d_e: np.ndarray, d_a: np.ndarray, grid: np.ndarray) -> None:
    rgb = simplex_rgb(grid)
    lim_e = d_e.max() * 1.05
    _, vmin_a = _aitchison_lognorm(d_a)
    plot_a = np.where(d_a > 0, d_a, vmin_a * 0.5)
    ax.scatter(d_e, plot_a, c=rgb, s=12, alpha=0.7, linewidths=0)
    ax.set_xlim(0, lim_e)
    ax.set_yscale("log")
    ax.set_ylim(vmin_a * 0.8, d_a.max() * 1.05)
    ax.set_box_aspect(1)
    ax.set_xlabel("Euclidean distance")
    ax.set_ylabel("Aitchison distance (log scale)")
    ax.set_title("Aitchison vs Euclidean\n(color = composition)", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aitchison vs Euclidean distance on the 3-simplex.",
    )
    parser.add_argument("--grid-n", type=int, default=80)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "data"
        / "plots"
        / "aitchison_vs_euclidean_3simplex.png",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    grid = ternary_grid(args.grid_n)
    refs = [
        ("from center (⅓, ⅓, ⅓)", CENTER_3),
        ("from (0.1, 0.1, 0.8)", REF_POINT_3),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for row, (row_title, ref) in enumerate(refs):
        d_e = euclidean_distance(grid, ref)
        d_a = aitchison_distance(grid, ref)

        for col, (metric, vals) in enumerate(
            [("euclidean", d_e), ("aitchison", d_a)],
        ):
            mappable = plot_radial_ternary(
                axes[row, col], grid, ref, vals, metric,
            )
            fig.colorbar(mappable, ax=axes[row, col], fraction=0.046, pad=0.02)

        plot_metric_scatter(axes[row, 2], d_e, d_a, grid)

        axes[row, 0].text(
            -0.20, 0.5, row_title, rotation=90, va="center", ha="center",
            fontsize=10, transform=axes[row, 0].transAxes,
        )

    fig.suptitle("Aitchison vs Euclidean distance on the 3-simplex", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0.05, 0, 1, 0.97))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
