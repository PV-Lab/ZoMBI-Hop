"""Interactive ternary cross-section plot of a 4D negated-Ackley objective.

Evaluates one of the analytic negated-Ackley variants from
``synthetic_data/ackley.py`` -- but lifted onto the 4-element probability simplex
(x1 + x2 + x3 + x4 = 1) -- and renders it over the x1/x2/x3 ternary with a slider
that steps through exact x4 cross-sections. Because the objective is analytic, each
slice is evaluated directly on the grid (no scattered-data interpolation). Output is
a self-contained interactive HTML file that opens in your default browser -- no
notebook, no kernel, no ipywidgets, no blocking GUI backend.

For a fixed slice x4 = s, a ternary node (a, b, c) with a + b + c = 1 corresponds to
the 4D composition ((1-s)*a, (1-s)*b, (1-s)*c, s), which sums to 1.

Usage:
    python plot_simplex.py --ackley {centroid|edge|vertex|multimodal}
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
from synthetic_data.ackley import _negated_ackley, ACKLEY_B, ACKLEY_B_SKINNY  # noqa: E402

N_SLICES = 40    # number of exact x4 cross-sections the slider steps through
GRID_N = 200     # resolution of the x1/x2/x3 ternary grid per slice
MARKER_SIZE = 4.0  # heatmap cell size in px; tune alongside GRID_N + FIG_W so cells just touch
FIG_W, FIG_H = 900, 820

# Canonical peak locations of each variant, lifted to the 4-element simplex.
# These mirror the 3-simplex centres in synthetic_data/ackley.py with a 4th
# coordinate appended (centroid spread across all four).
CENTERS_4D = {
    "centroid":   [np.array([0.25, 0.25, 0.25, 0.25])],
    "edge":       [np.array([0.5, 0.5, 0.0, 0.0])],
    "vertex":     [np.array([1.0, 0.0, 0.0, 0.0])],
    "multimodal": [np.array([0.25, 0.25, 0.25, 0.25]),
                   np.array([0.5, 0.5, 0.0, 0.0]),
                   np.array([1.0, 0.0, 0.0, 0.0])],
}


def make_predict(variant):
    """Return a ``predict((N, 4)) -> (N,)`` closure for the chosen variant.

    Matches ``synthetic_data.ackley.Ackley.predict`` semantics: a single negated
    Ackley for the unimodal variants, or the sum of three skinnier-peaked Ackleys
    for ``"multimodal"``.
    """
    centers = CENTERS_4D[variant]
    b = ACKLEY_B_SKINNY if variant == "multimodal" else ACKLEY_B

    def predict(X):
        total = np.zeros(X.shape[0], dtype=float)
        for center in centers:
            total = total + _negated_ackley(X, center, b=b)
        return total

    return predict


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
        description="Plot a 4D negated-Ackley objective as x4-sliced ternary cross-sections."
    )
    parser.add_argument(
        "--ackley", required=True, choices=sorted(CENTERS_4D),
        help="Which analytic Ackley variant to plot (lifted to the 4-element simplex).",
    )
    args = parser.parse_args()

    variant = args.ackley
    predict = make_predict(variant)

    ga, gb, gc = build_grid(GRID_N)
    n_nodes = len(ga)

    # Exact x4 slice values in [0, 1); x4=1 is the degenerate vertex (skipped).
    slice_values = np.linspace(0.0, 1.0, N_SLICES + 1)[:-1]

    # Evaluate every slice up front so the colour scale is shared across frames.
    grid_objs = []
    for x4 in slice_values:
        scale = 1.0 - x4
        if scale <= 1e-9:
            grid_objs.append(np.full(n_nodes, np.nan))
            continue
        X = np.column_stack([ga * scale, gb * scale, gc * scale,
                             np.full(n_nodes, x4)])
        grid_objs.append(predict(X))

    stacked = np.concatenate(grid_objs)
    obj_min = float(np.nanmin(stacked))
    obj_max = float(np.nanmax(stacked))

    # Assign each known peak to the slice whose x4 is closest, for an overlay star.
    peaks_by_slice = {}
    for center in CENTERS_4D[variant]:
        s_idx = int(np.argmin(np.abs(slice_values - center[3])))
        scale = 1.0 - center[3]
        tern = center[:3] / scale if scale > 1e-9 else center[:3]
        peaks_by_slice.setdefault(s_idx, []).append(tern)

    def slice_traces(s):
        heat = go.Scatterternary(
            a=ga, b=gb, c=gc, mode="markers", name="objective", hoverinfo="skip",
            marker=dict(color=grid_objs[s], colorscale="Viridis",
                        cmin=obj_min, cmax=obj_max, size=MARKER_SIZE,
                        showscale=True, colorbar=dict(title="Objective", x=1.02)),
        )
        traces = [heat]
        if s in peaks_by_slice:
            peaks = np.array(peaks_by_slice[s])
            traces.append(go.Scatterternary(
                a=peaks[:, 0], b=peaks[:, 1], c=peaks[:, 2], mode="markers",
                name="known peak",
                marker=dict(symbol="star", color="red", size=14,
                            line=dict(color="white", width=1)),
            ))
        return traces

    frames = [go.Frame(data=slice_traces(s), name=str(s)) for s in range(N_SLICES)]
    steps = [
        dict(method="animate", label=f"{slice_values[s]:.3f}",
             args=[[str(s)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                  transition=dict(duration=0))])
        for s in range(N_SLICES)
    ]

    fig = go.Figure(data=slice_traces(0), frames=frames)
    fig.update_layout(
        title=f"Negated Ackley ('{variant}') on the 4-simplex, sliced by x4",
        ternary=dict(sum=1, aaxis=dict(title="x1"), baxis=dict(title="x2"),
                     caxis=dict(title="x3")),
        sliders=[dict(active=0, currentvalue=dict(prefix="x4 = "),
                      pad=dict(t=50), steps=steps)],
        # push the legend to the right of the colorbar (x=1.02) so they don't overlap
        legend=dict(x=1.18, y=1.0),
        width=FIG_W, height=FIG_H,
    )

    out_html = HERE / "simplex_plot.html"
    fig.write_html(out_html, include_plotlyjs="cdn", auto_open=False)
    print(f"Wrote {out_html}")
    webbrowser.open(out_html.as_uri())


if __name__ == "__main__":
    main()
