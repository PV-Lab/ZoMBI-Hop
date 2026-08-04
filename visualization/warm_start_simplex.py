"""
visualization/warm_start_simplex.py
===================================
Visualise the 15%-budget **line-constrained** warm-start design at each
benchmark dimension, one static PNG per dimension:

    warm_start_simplex_3d.png    ternary triangle          4 lines (96 pts)
    warm_start_simplex_4d.png    3D tetrahedron            8 lines (192 pts)
    warm_start_simplex_10d.png   2D embedding of the 10-simplex  125 lines (3000 pts)

The hardware measures a *line* at a time: 24 evenly spaced compositions between
a free start and a free end point.  The warm start is therefore a union of
segments, chosen greedily to cover the simplex — see ``warm_start/warm_start.py``
(:func:`greedy_lines`).  Each figure draws the segments as translucent
polylines with their 24 measured points on top, coloured by selection order, so
you can read both *where* the lines were laid and *in what order* the greedy
coverage rule got there.

The diagrams follow the project's existing conventions: the ternary matches
plot_run.py's ``comp_to_xy`` (unit triangle, same corner order), the tetrahedron
matches its ``TETRA_VERTS`` corner mapping, and the 10d panel is the same kind
of 2D embedding plot_10d.py's CoNet uses (UMAP when umap-learn is installed,
otherwise a PCA projection, which needs no extra dependency).

Titles report the coverage radius — the distance from a uniform simplex probe to
its nearest measured composition — which is the metric that matters for a line
design (nearest-neighbour spacing is fixed *within* a line by construction).

Run:
  uv run python visualization/warm_start_simplex.py
  # one dimension only:
  uv run python visualization/warm_start_simplex.py --dims 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # repo root, so warm_start is importable

# Single source of truth for the sampler + metrics.
from warm_start.warm_start import (  # noqa: E402
    INPUT_NOISE,
    POINTS_PER_LINE,
    coverage_stats,
    greedy_lines,
    n_lines,
)

OUT_TEMPLATE = "warm_start_simplex_{dim}d.png"
CMAP = "turbo"          # line colour = selection order
LINE_ALPHA = 0.55

_SQRT3_2 = np.sqrt(3) / 2

# Component names at the corners of the 3d / 4d diagrams (plot_run.py's labels).
LABELS_3D = ("FAPbI3", "MAPbI3", "MAPbBr3")
LABELS_4D = ("FAPbI3", "MAPbI3", "MAPbBr3", "CsPbI3")

# Vertices of a regular unit-edge tetrahedron; composition column i -> corner i
# (mirrors plot_run.TETRA_VERTS / comp_to_xyz).
TETRA_VERTS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, _SQRT3_2, 0.0],
    [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3],
], dtype=float)
TETRA_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


# ── simplex mappings ─────────────────────────────────────────────────────────

def comp_to_xy(comp: np.ndarray) -> np.ndarray:
    """(N,3) simplex compositions → (N,2) ternary Cartesian (matches plot_run.py).

    col0 → (0,0) bottom-left, col1 → (1,0) bottom-right, col2 → (0.5,√3/2) top.
    """
    p = np.asarray(comp, float)
    s = p.sum(-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return np.column_stack([p[:, 1] + 0.5 * p[:, 2], _SQRT3_2 * p[:, 2]])


def comp_to_xyz(comp: np.ndarray) -> np.ndarray:
    """(N,4) simplex compositions → (N,3) tetrahedron Cartesian (matches plot_run.py)."""
    p = np.asarray(comp, float)
    s = p.sum(-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return p @ TETRA_VERTS


def embed_highd(X: np.ndarray, seed: int = 0) -> tuple[np.ndarray, str]:
    """(N,d) high-dimensional compositions → (N,2), the way plot_10d's CoNet does.

    Uses UMAP when ``umap-learn`` is installed (plot_10d's own embedding); falls
    back to a PCA projection otherwise, so this script runs in a plain
    numpy/scipy/sklearn environment.  Returns ``(xy, method_name)``.
    """
    try:
        import umap  # noqa: F401

        reducer = umap.UMAP(n_components=2, min_dist=0.35, n_neighbors=15,
                            random_state=seed)
        return np.asarray(reducer.fit_transform(X), float), "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA

        xy = PCA(n_components=2, random_state=seed).fit_transform(X)
        return np.asarray(xy, float), "PCA"


# ── shared drawing pieces ────────────────────────────────────────────────────

def _line_colors(n: int):
    import matplotlib.pyplot as plt

    return plt.get_cmap(CMAP)(np.linspace(0.0, 1.0, n))


def _title(dim: int, L: int, X: np.ndarray, extra: str = "") -> str:
    c = coverage_stats(X)
    return (f"Line-constrained warm start, d={dim}  —  {L} lines x "
            f"{POINTS_PER_LINE} pts = {len(X)}{extra}\n"
            f"coverage: mean={c['mean']:.3f}  p95={c['p95']:.3f}  "
            f"max={c['max']:.3f}   (noise r={INPUT_NOISE})")


def _draw_triangle(ax, labels) -> None:
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.12, _SQRT3_2 + 0.16)
    ax.text(-0.03, -0.03, labels[0], ha="right", va="top", fontsize=10)
    ax.text(1.03, -0.03, labels[1], ha="left", va="top", fontsize=10)
    ax.text(0.5, _SQRT3_2 + 0.04, labels[2], ha="center", va="bottom", fontsize=10)


def _order_colorbar(fig, ax, n_lines_: int) -> None:
    """Colourbar mapping the line colours back to greedy selection order."""
    import matplotlib as mpl

    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=mpl.colors.Normalize(1, n_lines_))
    fig.colorbar(sm, ax=ax, shrink=0.72, pad=0.02, label="line (selection order)")


# ── per-dimension figures ────────────────────────────────────────────────────

def figure_3d(out_png: Path, seed: int = 0) -> None:
    """Ternary triangle: every line drawn as a chord with its 24 points."""
    import matplotlib.pyplot as plt

    dim, L = 3, n_lines(3)
    lines, X = greedy_lines(L, dim, seed=seed)
    colors = _line_colors(L)

    fig, ax = plt.subplots(figsize=(6.6, 5.9))
    _draw_triangle(ax, LABELS_3D)
    for i, (p, q) in enumerate(lines):
        e = comp_to_xy(np.stack([p, q]))
        ax.plot(e[:, 0], e[:, 1], "-", color=colors[i], lw=1.6, alpha=LINE_ALPHA,
                zorder=2)
    xy = comp_to_xy(X)
    ax.scatter(xy[:, 0], xy[:, 1], s=20,
               c=np.repeat(colors, POINTS_PER_LINE, axis=0),
               edgecolors="k", linewidths=0.3, zorder=3)
    _order_colorbar(fig, ax, L)
    ax.set_title(_title(dim, L, X), fontsize=9)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out_png}")


def figure_4d(out_png: Path, seed: int = 0) -> None:
    """3D tetrahedron (mplot3d): the four components are the four corners."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    dim, L = 4, n_lines(4)
    lines, X = greedy_lines(L, dim, seed=seed)
    colors = _line_colors(L)

    fig = plt.figure(figsize=(7.4, 6.8))
    ax = fig.add_subplot(111, projection="3d")

    for i, j in TETRA_EDGES:
        ax.plot(*zip(TETRA_VERTS[i], TETRA_VERTS[j]), color="black", lw=1.2)
    centroid = TETRA_VERTS.mean(axis=0)
    for v, name in zip(TETRA_VERTS, LABELS_4D):
        p = v + 0.12 * (v - centroid)
        ax.text(p[0], p[1], p[2], name, ha="center", va="center", fontsize=9)

    for i, (p, q) in enumerate(lines):
        e = comp_to_xyz(np.stack([p, q]))
        ax.plot(e[:, 0], e[:, 1], e[:, 2], "-", color=colors[i], lw=1.5,
                alpha=LINE_ALPHA)
    pxyz = comp_to_xyz(X)
    ax.scatter(pxyz[:, 0], pxyz[:, 1], pxyz[:, 2], s=14,
               c=np.repeat(colors, POINTS_PER_LINE, axis=0),
               edgecolors="black", linewidths=0.3, depthshade=False)

    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    _order_colorbar(fig, ax, L)
    ax.set_title(_title(dim, L, X), fontsize=9)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out_png}")


def figure_10d(out_png: Path, seed: int = 0) -> None:
    """2D embedding of the 10-simplex, in the spirit of plot_10d.py's CoNet.

    The 10-simplex has no faithful 2D picture, so the design is embedded the way
    plot_10d embeds a run's compositions.  A line is a straight segment in 10d
    but a curve in the embedding, so its 24 points are joined in order rather
    than drawn as a chord.
    """
    import matplotlib.pyplot as plt

    dim, L = 10, n_lines(10)
    lines, X = greedy_lines(L, dim, seed=seed)
    colors = _line_colors(L)

    xy, method = embed_highd(X, seed=seed)
    seg = xy.reshape(L, POINTS_PER_LINE, 2)

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    for i in range(L):
        ax.plot(seg[i, :, 0], seg[i, :, 1], "-", color=colors[i], lw=0.9,
                alpha=0.45, zorder=2)
    ax.scatter(xy[:, 0], xy[:, 1], s=7,
               c=np.repeat(colors, POINTS_PER_LINE, axis=0),
               linewidths=0, zorder=3)

    ax.set_aspect("equal")
    ax.set_xlabel(f"{method} 1", fontsize=9)
    ax.set_ylabel(f"{method} 2", fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _order_colorbar(fig, ax, L)
    ax.set_title(_title(dim, L, X, extra=f"   ({method} embedding)"), fontsize=9)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure -> {out_png}")


FIGURES = {3: figure_3d, 4: figure_4d, 10: figure_10d}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static PNGs of the line-constrained warm-start design.")
    parser.add_argument("--dims", default="3,4,10",
                        help="Comma-separated dimensions to render (default: 3,4,10).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for the greedy line design (default: 0).")
    parser.add_argument("--out-dir", default=str(_HERE),
                        help="Directory for the PNGs (default: alongside this script).")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for token in args.dims.split(","):
        dim = int(token.strip())
        if dim not in FIGURES:
            raise SystemExit(f"No figure defined for d={dim} (have {sorted(FIGURES)})")
        FIGURES[dim](out_dir / OUT_TEMPLATE.format(dim=dim), seed=args.seed)


if __name__ == "__main__":
    main()
