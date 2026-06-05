"""Interactive ternary plot of a 3D negated-Ackley objective.

Evaluates one of the analytic negated-Ackley variants from
``synthetic_data/ackley.py`` on the 3-element probability simplex
(x1 + x2 + x3 = 1) and renders it as a single ternary heatmap. Because the
3-simplex *is* the ternary, the whole objective fits in one plot -- there is no
hidden dimension to slice, so (unlike plot_4d.py) there is no slider. The
objective is analytic, so the grid is evaluated directly (no scattered-data
interpolation). Output is a self-contained interactive HTML file that opens in
your default browser -- no notebook, no kernel, no ipywidgets, no blocking GUI
backend.

Usage:
    python plot_3d.py --ackley {centroid|edge|vertex|multimodal|realistic}

The variant catalogue is defined once in ``synthetic_data/ackley.py``; adding a
new mode there makes it available here automatically.
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

DIM = 3          # plot the 3-element simplex (x1 + x2 + x3 = 1)
GRID_N = 200     # resolution of the x1/x2/x3 ternary grid
MARKER_SIZE = 4.0  # heatmap cell size in px; tune alongside GRID_N + FIG_W so cells just touch
FIG_W, FIG_H = 900, 820


def build_grid(grid_n):
    """Barycentric (a, b, c) nodes on the unit simplex, each summing to 1."""
    pts = [
        (i / grid_n, j / grid_n, (grid_n - i - j) / grid_n)
        for i in range(grid_n + 1)
        for j in range(grid_n + 1 - i)
    ]
    bary = np.array(pts)
    return bary[:, 0], bary[:, 1], bary[:, 2]


def main():
    parser = argparse.ArgumentParser(
        description="Plot a 3D negated-Ackley objective as a ternary heatmap."
    )
    parser.add_argument(
        "--ackley", required=True, choices=sorted(Ackley.VARIANTS),
        help="Which analytic Ackley variant to plot (on the 3-element simplex).",
    )
    args = parser.parse_args()

    variant = args.ackley
    fn = Ackley(variant, dim=DIM)

    ga, gb, gc = build_grid(GRID_N)
    X = np.column_stack([ga, gb, gc])
    obj = fn.predict(X)
    obj_min = float(np.nanmin(obj))
    obj_max = float(np.nanmax(obj))

    heat = go.Scatterternary(
        a=ga, b=gb, c=gc, mode="markers", name="objective", hoverinfo="skip",
        marker=dict(color=obj, colorscale="Viridis",
                    cmin=obj_min, cmax=obj_max, size=MARKER_SIZE,
                    showscale=True, colorbar=dict(title="Objective", x=1.02)),
    )
    traces = [heat]

    # Overlay the known analytic peak(s) as red stars.
    peaks = np.array(fn.centers)
    if len(peaks):
        traces.append(go.Scatterternary(
            a=peaks[:, 0], b=peaks[:, 1], c=peaks[:, 2], mode="markers",
            name="known peak",
            marker=dict(symbol="star", color="red", size=14,
                        line=dict(color="white", width=1)),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Negated Ackley ('{variant}') on the 3-simplex",
        ternary=dict(sum=1, aaxis=dict(title="x1"), baxis=dict(title="x2"),
                     caxis=dict(title="x3")),
        # push the legend to the right of the colorbar (x=1.02) so they don't overlap
        legend=dict(x=1.18, y=1.0),
        width=FIG_W, height=FIG_H,
    )

    out_html = HERE / "simplex_plot_3d.html"
    fig.write_html(out_html, include_plotlyjs="cdn", auto_open=False)
    print(f"Wrote {out_html}")
    webbrowser.open(out_html.as_uri())


if __name__ == "__main__":
    main()
