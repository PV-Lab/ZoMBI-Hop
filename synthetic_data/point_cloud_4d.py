"""Interactive 3D point-cloud view of a 4D negated-Ackley objective.

The companion to ``synthetic_data/plot_4d.py``. Where that script slices the
objective by x4 and shows one ternary cross-section at a time behind a slider,
this one shows the *whole* 4D function at once as a single semi-transparent 3D
point cloud you can orbit, pan, and zoom.

The trick is that the 4-element probability simplex (x1 + x2 + x3 + x4 = 1) is a
regular tetrahedron: every composition is a convex combination of the four
vertices, so we map each lattice point (a, b, c, d) -> a*V1 + b*V2 + c*V3 + d*V4
into 3D and colour it by the objective. No dimension is hidden behind a slider --
all four coordinates are encoded in the 3D position, and the objective is the
colour. Output is a self-contained interactive HTML file that opens in your
default browser (rotatable WebGL scene; no notebook, kernel, or GUI backend).

Usage:
    python point_cloud_4d.py --ackley {centroid|edge|vertex|multimodal|realistic}

The variant catalogue is defined once in ``synthetic_data/ackley.py``; adding a
new mode there makes it available here (and in plot_4d.py) automatically.
"""

import argparse
import sys
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
# Make the repo root importable so this runs from any working directory.
sys.path.insert(0, str(HERE.parent))
from synthetic_data.ackley import Ackley  # noqa: E402

DIM = 4           # the 4-element simplex (x1 + x2 + x3 + x4 = 1) is a tetrahedron
GRID_N = 44       # lattice resolution: points where the four integer counts sum to GRID_N
MARKER_SIZE = 3.5
MARKER_OPACITY = 0.18   # low so the dense interior reads as a translucent volume
FIG_W, FIG_H = 950, 850

# Vertices of a regular tetrahedron, one per simplex coordinate (x1..x4). Any
# affine-independent placement works; these are the canonical "alternating cube
# corners" set, recentred on the origin for nice default framing.
TETRA_VERTICES = np.array([
    [1.0,  1.0,  1.0],
    [1.0, -1.0, -1.0],
    [-1.0,  1.0, -1.0],
    [-1.0, -1.0,  1.0],
])
TETRA_VERTICES = TETRA_VERTICES - TETRA_VERTICES.mean(axis=0)
VERTEX_LABELS = ["x1", "x2", "x3", "x4"]


def build_simplex_lattice(grid_n):
    """All (x1, x2, x3, x4) on the 4-simplex with denominator ``grid_n``.

    Returns an (N, 4) array of compositions summing to 1, one per integer
    4-tuple (i, j, k, l) with i + j + k + l = grid_n.
    """
    pts = [
        (i, j, k, grid_n - i - j - k)
        for i in range(grid_n + 1)
        for j in range(grid_n + 1 - i)
        for k in range(grid_n + 1 - i - j)
    ]
    return np.array(pts, dtype=float) / grid_n


def to_3d(comp):
    """Map simplex compositions (N, 4) into 3D tetrahedron coordinates (N, 3)."""
    return comp @ TETRA_VERTICES


def tetra_edges_trace():
    """Wireframe of the tetrahedron's six edges, for spatial orientation."""
    xs, ys, zs = [], [], []
    for i in range(4):
        for j in range(i + 1, 4):
            xs += [TETRA_VERTICES[i, 0], TETRA_VERTICES[j, 0], None]
            ys += [TETRA_VERTICES[i, 1], TETRA_VERTICES[j, 1], None]
            zs += [TETRA_VERTICES[i, 2], TETRA_VERTICES[j, 2], None]
    return go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines", name="simplex edges",
        line=dict(color="rgba(60,60,60,0.6)", width=3), hoverinfo="skip",
    )


def vertex_labels_trace():
    """Corner labels x1..x4, nudged outward so they clear the cloud."""
    pos = TETRA_VERTICES * 1.12
    return go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="text", text=VERTEX_LABELS,
        textfont=dict(size=18, color="black"), name="vertices", hoverinfo="skip",
        showlegend=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot a 4D negated-Ackley objective as a rotatable 3D point cloud."
    )
    parser.add_argument(
        "--ackley", required=True, choices=sorted(Ackley.VARIANTS),
        help="Which analytic Ackley variant to plot (lifted to the 4-element simplex).",
    )
    args = parser.parse_args()

    variant = args.ackley
    fn = Ackley(variant, dim=DIM)

    comp = build_simplex_lattice(GRID_N)
    obj = fn.predict(comp)
    xyz = to_3d(comp)

    obj_min, obj_max = float(obj.min()), float(obj.max())

    hover = [
        f"x=[{a:.2f}, {b:.2f}, {c:.2f}, {d:.2f}]<br>obj={v:.2f}"
        for (a, b, c, d), v in zip(comp, obj)
    ]

    cloud = go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers", name="objective",
        text=hover, hoverinfo="text",
        marker=dict(
            color=obj, colorscale="Viridis", cmin=obj_min, cmax=obj_max,
            size=MARKER_SIZE, opacity=MARKER_OPACITY,
            showscale=True, colorbar=dict(title="Objective"),
        ),
    )

    # Known analytic peaks, mapped into the same 3D frame, as opaque red stars.
    peaks = np.array(fn.centers)
    peaks_xyz = to_3d(peaks)
    peaks_trace = go.Scatter3d(
        x=peaks_xyz[:, 0], y=peaks_xyz[:, 1], z=peaks_xyz[:, 2], mode="markers",
        name="known peak",
        marker=dict(symbol="diamond", color="red", size=6,
                    line=dict(color="white", width=1)),
        hoverinfo="name",
    )

    fig = go.Figure(data=[cloud, tetra_edges_trace(), vertex_labels_trace(), peaks_trace])
    fig.update_layout(
        title=f"Negated Ackley ('{variant}') on the 4-simplex as a 3D point cloud",
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data",
        ),
        legend=dict(x=0.0, y=1.0),
        width=FIG_W, height=FIG_H,
    )

    out_html = HERE / "point_cloud_plot.html"
    fig.write_html(out_html, include_plotlyjs="cdn", auto_open=False)
    print(f"Wrote {out_html}")
    webbrowser.open(out_html.as_uri())


if __name__ == "__main__":
    main()
