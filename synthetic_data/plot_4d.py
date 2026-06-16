"""Interactive 3D point-cloud view of a tunable 4D Ackley + noise objective.

The 4-element probability simplex is a tetrahedron: each composition
(x1, x2, x3, x4) maps to a 3D point via the tetrahedron vertices. The
objective value is encoded as colour.

Runs a Dash app with sliders for noise frequency, noise amplitude, number of
Ackley optima, and grid resolution.  Basin width (b) is not tunable here: it is
the hardcoded ``BASIN_WIDTH_BY_DIM[4]`` in ``ackley.py``.  Click "Save as Default"
to persist slider values to ``synthetic_data/ackley/defaults.json``.

Usage:
    python point_cloud_4d.py

Overlay API
-----------
``add_simplex_overlays`` lets a caller draw ZoMBI-Hop-style annotations on top
of the point cloud, all specified as plain simplex compositions (no ZoMBI data
structures required). It supports:

  * **pared points**  -- discrete sampled compositions, coloured by objective;
  * **main / cache lines** -- LineBO's suggested lines as 3D segments;
  * **needles** -- discovered-needle markers plus penalization ellipsoids.
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from synthetic_data.ackley import Ackley, load_config, save_config, BASIN_WIDTH_BY_DIM  # noqa: E402

DIM = 4
DEFAULT_GRID_N = 44
GRID_N = DEFAULT_GRID_N
MARKER_SIZE = 3.5
MARKER_OPACITY = 0.18
FIG_W, FIG_H = 950, 850

# ── Overlay styling ──────────────────────────────────────────────────────────
PARED_SIZE = 8.0
MAIN_LINE_COLOR = "orange"
CACHE_LINE_COLOR = "deepskyblue"
NEEDLE_MARKER_COLOR = "red"
NEEDLE_ELL_COLOR = "purple"
NEEDLE_ELL_OPACITY = 0.16

TETRA_VERTICES = np.array([
    [1.0,  1.0,  1.0],
    [1.0, -1.0, -1.0],
    [-1.0,  1.0, -1.0],
    [-1.0, -1.0,  1.0],
])
TETRA_VERTICES = TETRA_VERTICES - TETRA_VERTICES.mean(axis=0)
VERTEX_LABELS = ["x1", "x2", "x3", "x4"]


def build_simplex_lattice(grid_n: int) -> np.ndarray:
    pts = [
        (i, j, k, grid_n - i - j - k)
        for i in range(grid_n + 1)
        for j in range(grid_n + 1 - i)
        for k in range(grid_n + 1 - i - j)
    ]
    return np.array(pts, dtype=float) / grid_n


def to_3d(comp: np.ndarray) -> np.ndarray:
    return comp @ TETRA_VERTICES


def tetra_edges_trace():
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
    pos = TETRA_VERTICES * 1.12
    return go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="text", text=VERTEX_LABELS,
        textfont=dict(size=18, color="black"), name="vertices", hoverinfo="skip",
        showlegend=False,
    )


# ── Overlay API (unchanged from original) ────────────────────────────────────

def composition_to_tetra(comp: np.ndarray) -> np.ndarray:
    comp = np.atleast_2d(np.asarray(comp, dtype=float))
    if comp.shape[1] != TETRA_VERTICES.shape[0]:
        raise ValueError(
            f"expected compositions with {TETRA_VERTICES.shape[0]} components, "
            f"got shape {comp.shape}"
        )
    return comp @ TETRA_VERTICES


def pared_points_trace(
    comp, values=None, *, recency=None, cmin=None, cmax=None,
    size=PARED_SIZE, name="pared points",
):
    xyz = composition_to_tetra(comp)
    n = xyz.shape[0]
    if recency is not None:
        r = np.asarray(recency, dtype=float).ravel()
        rng = r.max() - r.min()
        norm = (r - r.min()) / rng if rng > 0 else np.ones(n)
        sizes = size * (0.6 + 0.9 * norm)
    else:
        sizes = size
    marker = dict(size=sizes, opacity=0.95, line=dict(color="white", width=0.5))
    if values is not None:
        values = np.asarray(values, dtype=float).ravel()
        marker.update(
            color=values, colorscale="Viridis",
            cmin=cmin if cmin is not None else float(values.min()),
            cmax=cmax if cmax is not None else float(values.max()),
            showscale=False,
        )
        text = [f"obj={v:.3f}" for v in values]
        hoverinfo = "text"
    else:
        marker.update(color=MAIN_LINE_COLOR)
        text = None
        hoverinfo = "name"
    return go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
        name=name, text=text, hoverinfo=hoverinfo, marker=marker,
    )


def line_trace(endpoints, *, name, color, dash=None, width=6.0):
    pts = composition_to_tetra(endpoints)
    return go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
        name=name, hoverinfo="name",
        line=dict(color=color, width=width, dash=dash),
    )


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])


def needle_marker_trace(centers, *, name="needle", color=NEEDLE_MARKER_COLOR):
    xyz = composition_to_tetra(centers)
    return go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
        name=name, hoverinfo="name",
        marker=dict(symbol="x", color=color, size=7,
                    line=dict(color="darkred", width=1)),
    )


def needle_ellipsoid_mesh(
    center, M, *, name, color=NEEDLE_ELL_COLOR, opacity=NEEDLE_ELL_OPACITY,
    show_legend=False, n=600,
):
    import torch
    from src.utils.simplex import composition_to_ilr, ilr_to_composition

    center = np.asarray(center, dtype=float).ravel()
    d = center.shape[0]
    M = np.asarray(M, dtype=float)
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, 1e-12)
    sphere = _fibonacci_sphere(n)
    u = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ sphere.T).T
    c_ilr = composition_to_ilr(
        torch.as_tensor(center.reshape(1, -1), dtype=torch.float64)
    ).squeeze(0).cpu().numpy()
    z = c_ilr + u
    comp = ilr_to_composition(torch.as_tensor(z, dtype=torch.float64), d).cpu().numpy()
    xyz = composition_to_tetra(comp)
    return go.Mesh3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        alphahull=0, color=color, opacity=opacity, flatshading=True,
        name=name, showlegend=show_legend, hoverinfo="skip",
    )


def add_simplex_overlays(
    fig, *, pared_points=None, pared_values=None, recency=None,
    main_line=None, cache_line=None, needles=None, needle_ell_M=None,
    obj_cmin=None, obj_cmax=None,
):
    traces = []
    if needles is not None:
        needles = np.atleast_2d(np.asarray(needles, dtype=float))
        M_list = needle_ell_M or []
        first_ell = True
        for i, c in enumerate(needles):
            Mi = M_list[i] if i < len(M_list) else None
            if Mi is None:
                continue
            mesh = needle_ellipsoid_mesh(c, Mi, name="needle region", show_legend=first_ell)
            if mesh is not None:
                traces.append(mesh)
                first_ell = False
        traces.append(needle_marker_trace(needles))
    if pared_points is not None:
        traces.append(pared_points_trace(
            pared_points, pared_values, recency=recency,
            cmin=obj_cmin, cmax=obj_cmax,
        ))
    if main_line is not None:
        traces.append(line_trace(main_line, name="LineBO (main)", color=MAIN_LINE_COLOR, width=7))
    if cache_line is not None:
        traces.append(line_trace(cache_line, name="LineBO (cache)", color=CACHE_LINE_COLOR, dash="dot", width=9))
    fig.add_traces(traces)
    return fig


# ── Static per-iteration export ───────────────────────────────────────────────

COORD_COLS_4D = ["x1", "x2", "x3", "x4"]


def _read_needles_at_iter(needles_csv: Path, n: int) -> np.ndarray:
    """Load the (M, 4) needle compositions discovered up to and including iteration ``n``.

    The needles.csv (written by ``optimize/evaluate.py``) stamps each needle with
    the ``iteration`` it was found, so the run's state *during* iteration ``n`` is
    every needle with ``iteration <= n``.  Returns an empty (0, 4) array if none
    qualify.
    """
    import pandas as pd

    df = pd.read_csv(needles_csv)
    missing = [c for c in COORD_COLS_4D + ["iteration"] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{needles_csv} is missing expected column(s) {missing}; "
            f"this exporter only handles 4D (x1..x4) needles.csv files."
        )
    df = df[df["iteration"] <= n]
    if df.empty:
        return np.empty((0, 4), dtype=float)
    return df[COORD_COLS_4D].to_numpy(dtype=float)


def render_iteration_point_cloud(run_dir, n: int, *, grid_n: int = GRID_N):
    """Write ``point_cloud_<n>.html`` for the run in ``run_dir`` at iteration ``n``.

    Shows only the background objective point cloud (the current default 4D
    "realistic" Ackley), the known peaks, and the needles discovered up to
    iteration ``n``.  Returns the path to the written HTML file.
    """
    run_dir = Path(run_dir)
    needles_csv = run_dir / "needles.csv"
    if not needles_csv.exists():
        raise FileNotFoundError(f"No needles.csv in {run_dir}")

    needles = _read_needles_at_iter(needles_csv, n)
    fn = Ackley("realistic", dim=DIM)

    comp = build_simplex_lattice(int(grid_n))
    obj = fn.predict(comp)
    xyz = to_3d(comp)
    obj_min, obj_max = float(obj.min()), float(obj.max())

    hover = [
        f"x=[{a:.2f}, {b:.2f}, {c:.2f}, {d:.2f}]<br>obj={v:.2f}"
        for (a, b, c, d), v in zip(comp, obj)
    ]
    cloud = go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
        name="objective", text=hover, hoverinfo="text",
        marker=dict(
            color=obj, colorscale="Viridis", cmin=obj_min, cmax=obj_max,
            size=MARKER_SIZE, opacity=MARKER_OPACITY,
            showscale=True, colorbar=dict(title="Objective"),
        ),
    )

    peaks_xyz = to_3d(np.array(fn.centers))
    peaks_trace = go.Scatter3d(
        x=peaks_xyz[:, 0], y=peaks_xyz[:, 1], z=peaks_xyz[:, 2], mode="markers",
        name="known peak",
        marker=dict(symbol="diamond", color="red", size=6,
                    line=dict(color="white", width=1)),
        hoverinfo="name",
    )

    data = [cloud, tetra_edges_trace(), vertex_labels_trace(), peaks_trace]
    if needles.shape[0] > 0:
        data.append(needle_marker_trace(needles))

    fig = go.Figure(data=data)
    fig.update_layout(
        title=f"ZoMBI-Hop state at iteration {n} on the 4-simplex "
              f"(negated Ackley) — {needles.shape[0]} needle(s)",
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False), aspectmode="data",
        ),
        legend=dict(x=0.0, y=1.0),
        width=FIG_W, height=FIG_H,
    )

    out_path = run_dir / f"point_cloud_{n}.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn", auto_open=False)
    return out_path


# ── Dash app ─────────────────────────────────────────────────────────────────

def build_app():
    """Construct the interactive Dash app (imported lazily so the static
    exporter and overlay API don't require Dash)."""
    from dash import Dash, Input, Output, State, callback, dcc, html

    cfg = load_config()

    app = Dash(__name__)

    app.layout = html.Div([
        html.H2("Tunable Realistic Ackley on the 4-Simplex (Point Cloud)",
                style={"textAlign": "center"}),
        html.Div([
            html.Div([
                html.Label("Number of Optima"),
                dcc.Slider(id="n-optima", min=1, max=30, step=1,
                           value=cfg["n_optima"],
                           marks={i: str(i) for i in range(1, 31, 5)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"padding": "10px"}),
            html.Div([
                html.Label("Noise Frequency"),
                dcc.Slider(id="noise-freq", min=0, max=40, step=0.5,
                           value=cfg["noise_freq"],
                           marks={i: str(i) for i in range(0, 41, 5)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"padding": "10px"}),
            html.Div([
                html.Label("Noise Amplitude (relative to peaks)"),
                dcc.Slider(id="noise-amp", min=0, max=2000, step=20,
                           value=cfg["noise_amp"],
                           marks={i: str(i) for i in range(0, 2000, 200)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"padding": "10px"}),
            html.Div([
                html.Label("Intensity Offset Mean"),
                dcc.Slider(id="intensity-mean", min=0, max=100, step=1,
                           value=cfg.get("intensity_mean", 0),
                           marks={i: str(i) for i in range(0, 101, 20)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"padding": "10px"}),
            html.Div([
                html.Label("Intensity Offset Variance"),
                dcc.Slider(id="intensity-var", min=0, max=2000, step=10,
                           value=cfg.get("intensity_var", 0),
                           marks={i: str(i) for i in range(0, 2001, 400)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"padding": "10px"}),
            html.Div([
                html.Label("Grid Resolution"),
                dcc.Slider(id="grid-res", min=15, max=50, step=5, value=DEFAULT_GRID_N,
                           marks={i: str(i) for i in range(15, 55, 5)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], style={"padding": "10px"}),
            html.Div([
                html.Button("Save as Default", id="save-btn", n_clicks=0,
                            style={"marginTop": "10px", "padding": "8px 24px"}),
                html.Span(id="save-status", style={"marginLeft": "12px"}),
            ], style={"padding": "10px", "textAlign": "center"}),
        ], style={"width": "60%", "margin": "0 auto"}),
        dcc.Graph(id="cloud-plot", style={"height": f"{FIG_H}px"}),
    ])

    @callback(
        Output("save-status", "children"),
        Input("save-btn", "n_clicks"),
        State("n-optima", "value"),
        State("intensity-mean", "value"),
        State("intensity-var", "value"),
        State("noise-freq", "value"),
        State("noise-amp", "value"),
        prevent_initial_call=True,
    )
    def save_defaults(n_clicks, n_optima, intensity_mean, intensity_var, noise_freq, noise_amp):
        # basin_width is no longer tunable here; preserve whatever is already in the
        # config (the 3d default) so saving the other params doesn't drop it.
        save_config({
            "n_optima": int(n_optima),
            "basin_width": float(load_config().get("basin_width", 50.0)),
            "intensity_mean": float(intensity_mean),
            "intensity_var": float(intensity_var),
            "noise_freq": float(noise_freq),
            "noise_amp": float(noise_amp),
        })
        return "Saved!"

    @callback(
        Output("cloud-plot", "figure"),
        Input("n-optima", "value"),
        Input("noise-freq", "value"),
        Input("noise-amp", "value"),
        Input("intensity-mean", "value"),
        Input("intensity-var", "value"),
        Input("grid-res", "value"),
    )
    def update_plot(n_optima, noise_freq, noise_amp, intensity_mean, intensity_var, grid_res):
        # basin_width (b) is not passed: Ackley uses the hardcoded BASIN_WIDTH_BY_DIM
        # value for this dim.
        fn = Ackley(
            "realistic", dim=DIM,
            n_optima=int(n_optima),
            intensity_mean=float(intensity_mean),
            intensity_var=float(intensity_var),
            noise_freq=float(noise_freq),
            noise_amp=float(noise_amp),
        )

        comp = build_simplex_lattice(int(grid_res))
        obj = fn.predict(comp)
        xyz = to_3d(comp)
        obj_min, obj_max = float(obj.min()), float(obj.max())

        hover = [
            f"x=[{a:.2f}, {b:.2f}, {c:.2f}, {d:.2f}]<br>obj={v:.2f}"
            for (a, b, c, d), v in zip(comp, obj)
        ]

        cloud = go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
            name="objective", text=hover, hoverinfo="text",
            marker=dict(
                color=obj, colorscale="Viridis",
                cmin=obj_min, cmax=obj_max,
                size=MARKER_SIZE, opacity=MARKER_OPACITY,
                showscale=True, colorbar=dict(title="Objective"),
            ),
        )

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
            title=f"Tunable Realistic Ackley ({n_optima} peaks, "
                  f"b={BASIN_WIDTH_BY_DIM[DIM]} at dim {DIM})",
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode="data",
            ),
            legend=dict(x=0.0, y=1.0),
            width=FIG_W, height=FIG_H,
        )
        return fig

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Render a 4-simplex Ackley point cloud. With RUN_DIR and "
                    "ITERATION, export a static point_cloud_<n>.html showing the "
                    "needles discovered up to that iteration; with no positional "
                    "args, launch the interactive Dash app.")
    parser.add_argument("run_dir", nargs="?", default=None,
                        help="Run directory containing a needles.csv.")
    parser.add_argument("iteration", nargs="?", type=int, default=None,
                        help="Iteration n to render the state of.")
    parser.add_argument("--grid-n", type=int, default=GRID_N,
                        help=f"Simplex lattice resolution (default: {GRID_N}).")
    args = parser.parse_args()

    if args.run_dir is not None and args.iteration is not None:
        out = render_iteration_point_cloud(args.run_dir, args.iteration, grid_n=args.grid_n)
        print(f"Wrote {out}")
    elif args.run_dir is not None or args.iteration is not None:
        parser.error("provide BOTH run_dir and iteration to export, or neither "
                     "to launch the interactive app.")
    else:
        print("Starting Dash app at http://127.0.0.1:8051")
        build_app().run(debug=True, port=8051)


if __name__ == "__main__":
    main()
