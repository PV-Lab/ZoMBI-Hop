"""Interactive viewer for tunable synthetic objectives on the simplex.

Runs a single Dash app with two dropdowns:

  * **dimensionality** — "3D" draws the 3-simplex as a ternary heatmap; "4D"
    draws the 4-simplex as a tetrahedron point cloud (each composition mapped to
    a 3D point, objective value encoded as colour).
  * **objective** — "ackley" (realistic negated-Ackley + noise) or "bumps"
    (planted bump field).  The sliders swap to match the selected objective.

The plot updates in real time as you drag any slider.  For ackley, basin width
(b) is not tunable here: it is the hardcoded ``BASIN_WIDTH_BY_DIM[dim]`` in
``ackley.py``.  Click "Save as Default" to persist slider values for the
selected objective to ``synthetic_data/defaults/ackley.json`` or
``synthetic_data/defaults/bumps.json``.

The ackley panel adds a **Random Seed** slider that drives *both* optima
placement and the background noise.

Usage:
    python plot.py

Overlay API
-----------
``add_simplex_overlays`` lets a caller draw ZoMBI-Hop-style annotations on top
of the 4D point cloud, all specified as plain simplex compositions (no ZoMBI
data structures required). It supports:

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
from synthetic_data.ackley import (  # noqa: E402
    Ackley,
    BASIN_WIDTH_BY_DIM,
    load_config as load_ackley_config,
    save_config as save_ackley_config,
)
from synthetic_data.bumps import (  # noqa: E402
    Bumps,
    load_config as load_bumps_config,
    save_config as save_bumps_config,
)

DIM = 4
DEFAULT_GRID_N = 44
GRID_N = DEFAULT_GRID_N
MARKER_SIZE = 3.5
MARKER_OPACITY = 0.18
FIG_W, FIG_H = 950, 850

# Objective threshold separating near-optimal "basins" from the background.
# Compositions whose objective falls below it are de-emphasised: blacked out in
# the 3D ternary heatmap, made transparent (hidden) in the 4D point cloud. The
# same value is shared by both views and imported by analyze_basin_vol.py, which
# draws it as the "basin threshold" line. Objectives are normalised to [0.5, 1].
BASIN_THRESHOLD = 0.9

# 3D ternary view (3-simplex heatmap).
DEFAULT_GRID_N_3D = 120
TERNARY_MARKER_SIZE = 4.0

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


def build_ternary_grid(grid_n: int) -> np.ndarray:
    """Regular 3-simplex lattice as (N, 3) compositions for the ternary view."""
    pts = [
        (i / grid_n, j / grid_n, (grid_n - i - j) / grid_n)
        for i in range(grid_n + 1)
        for j in range(grid_n + 1 - i)
    ]
    return np.array(pts)


def basin_mask(obj: np.ndarray, threshold: float = BASIN_THRESHOLD) -> np.ndarray:
    """Boolean mask of compositions at or above ``threshold`` (the basins)."""
    return np.asarray(obj) >= threshold


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


# ── Realistic Ackley builder ─────────────────────────────────────────────────

def build_ackley(dim, n_optima, intensity_mean, intensity_var, noise_freq, seed):
    """Build a "realistic" Ackley for the live viewer.

    The ``seed`` drives *both* the optima placement (``peak_seed``) and the
    background noise (``noise_seed``).  The background noise amplitude and basin
    width come from the per-dim tables (``NOISE_AMP_BY_DIM[dim]`` /
    ``BASIN_WIDTH_BY_DIM[dim]``); there is no noise-amplitude slider.
    """
    return Ackley(
        "realistic", dim=int(dim), n_optima=int(n_optima),
        intensity_mean=float(intensity_mean), intensity_var=float(intensity_var),
        noise_freq=float(noise_freq), noise_amp=None,  # -> NOISE_AMP_BY_DIM[dim]
        peak_seed=int(seed), noise_seed=int(seed),
    )


# ── Dash app ─────────────────────────────────────────────────────────────────

def build_app():
    """Construct the interactive Dash app (imported lazily so the static
    exporter and overlay API don't require Dash)."""
    from dash import Dash, Input, Output, State, callback, dcc, html

    ack_cfg = load_ackley_config()
    bmp_cfg = load_bumps_config()

    app = Dash(__name__)

    ackley_panel = html.Div([
        html.Div([
            html.Label("Number of Optima"),
            dcc.Slider(id="n-optima", min=1, max=30, step=1,
                       value=ack_cfg["n_optima"],
                       marks={i: str(i) for i in range(1, 31, 5)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"padding": "10px"}),
        html.Div([
            html.Label("Noise Frequency"),
            dcc.Slider(id="noise-freq", min=0, max=40, step=0.5,
                       value=ack_cfg["noise_freq"],
                       marks={i: str(i) for i in range(0, 41, 5)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"padding": "10px"}),
        html.Div([
            html.Label("Intensity Offset Mean"),
            dcc.Slider(id="intensity-mean", min=0, max=100, step=1,
                       value=ack_cfg.get("intensity_mean", 0),
                       marks={i: str(i) for i in range(0, 101, 20)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"padding": "10px"}),
        html.Div([
            html.Label("Intensity Offset Variance"),
            dcc.Slider(id="intensity-var", min=0, max=2000, step=10,
                       value=ack_cfg.get("intensity_var", 0),
                       marks={i: str(i) for i in range(0, 2001, 400)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"padding": "10px"}),
        html.Div([
            html.Label("Random Seed (optima placement + noise)"),
            dcc.Slider(id="ack-seed", min=0, max=100, step=1,
                       value=ack_cfg.get("seed", 0),
                       marks={i: str(i) for i in range(0, 101, 20)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"padding": "10px"}),
    ])

    bumps_panel = html.Div([
        html.Div([
            html.Label("Number of Major Bumps (true optima)"),
            dcc.Slider(id="n-major", min=1, max=30, step=1,
                       value=bmp_cfg["n_major"],
                       marks={i: str(i) for i in range(1, 31, 2)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"padding": "10px"}),
        html.Div([
            html.Label("Number of Micro Bumps"),
            dcc.Slider(id="n-micro", min=0, max=150, step=5,
                       value=bmp_cfg["n_micro"],
                       marks={i: str(i) for i in range(0, 151, 25)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"padding": "10px"}),
        html.Div([
            html.Label("Major Bump Width (σ)"),
            dcc.Slider(id="major-sigma", min=0.01, max=0.30, step=0.005,
                       value=bmp_cfg["major_sigma"],
                       marks={round(i, 2): f"{i:.2f}" for i in np.arange(0.05, 0.31, 0.05)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"padding": "10px"}),
        html.Div([
            html.Label("Micro Bump Width Range (σ)"),
            dcc.RangeSlider(id="micro-sigma", min=0.005, max=0.20, step=0.005,
                            value=[bmp_cfg["micro_sigma_min"], bmp_cfg["micro_sigma_max"]],
                            marks={round(i, 2): f"{i:.2f}" for i in np.arange(0.05, 0.21, 0.05)},
                            tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"padding": "10px"}),
        html.Div([
            html.Label("Micro Bump Weight Range (negative = dents)"),
            dcc.RangeSlider(id="micro-weight", min=-1.0, max=1.0, step=0.05,
                            value=[bmp_cfg["micro_weight_min"], bmp_cfg["micro_weight_max"]],
                            marks={round(i, 1): f"{i:.1f}" for i in np.arange(-1.0, 1.01, 0.5)},
                            tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"padding": "10px"}),
        html.Div([
            html.Label("Placement Seed"),
            dcc.Slider(id="seed", min=0, max=100, step=1,
                       value=bmp_cfg["seed"],
                       marks={i: str(i) for i in range(0, 101, 20)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ], style={"padding": "10px"}),
    ])

    app.layout = html.Div([
        html.H2("Tunable Synthetic Objective on the Simplex",
                style={"textAlign": "center"}),
        html.Div([
            html.Div([
                html.Label("Dimensionality"),
                dcc.Dropdown(id="dim-select", clearable=False,
                             options=[{"label": "3D (ternary heatmap)", "value": "3d"},
                                      {"label": "4D (tetrahedron point cloud)", "value": "4d"}],
                             value="3d"),
            ], style={"padding": "10px"}),
            html.Div([
                html.Label("Objective"),
                dcc.Dropdown(id="objective", clearable=False,
                             options=[{"label": "Ackley (realistic + noise)", "value": "ackley"},
                                      {"label": "Planted bumps", "value": "bumps"}],
                             value="ackley"),
            ], style={"padding": "10px"}),
            html.Div(ackley_panel, id="ackley-wrap"),
            html.Div(bumps_panel, id="bumps-wrap", style={"display": "none"}),
            html.Div([
                html.Label("Grid Resolution"),
                dcc.Slider(id="grid-res-3d", min=30, max=180, step=10, value=DEFAULT_GRID_N_3D,
                           marks={i: str(i) for i in range(30, 181, 30)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], id="grid-3d-wrap", style={"padding": "10px"}),
            html.Div([
                html.Label("Grid Resolution"),
                dcc.Slider(id="grid-res-4d", min=15, max=50, step=5, value=DEFAULT_GRID_N,
                           marks={i: str(i) for i in range(15, 55, 5)},
                           tooltip={"placement": "bottom", "always_visible": True}),
            ], id="grid-4d-wrap", style={"padding": "10px", "display": "none"}),
            html.Div([
                html.Label("Basin Threshold (below: blacked out in 3D, "
                           "transparent in 4D)"),
                dcc.Slider(id="basin-threshold", min=0.5, max=1.0, step=0.01,
                           value=BASIN_THRESHOLD,
                           marks={round(i, 2): f"{i:.2f}" for i in np.arange(0.5, 1.01, 0.1)},
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
        Output("ackley-wrap", "style"),
        Output("bumps-wrap", "style"),
        Input("objective", "value"),
    )
    def toggle_panels(objective):
        show, hide = {}, {"display": "none"}
        if objective == "bumps":
            return hide, show
        return show, hide

    @callback(
        Output("grid-3d-wrap", "style"),
        Output("grid-4d-wrap", "style"),
        Input("dim-select", "value"),
    )
    def toggle_grid_controls(dim_sel):
        show = {"padding": "10px"}
        hide = {"padding": "10px", "display": "none"}
        if dim_sel == "4d":
            return hide, show
        return show, hide

    @callback(
        Output("save-status", "children"),
        Input("save-btn", "n_clicks"),
        State("objective", "value"),
        State("n-optima", "value"),
        State("intensity-mean", "value"),
        State("intensity-var", "value"),
        State("noise-freq", "value"),
        State("n-major", "value"),
        State("n-micro", "value"),
        State("major-sigma", "value"),
        State("micro-sigma", "value"),
        State("micro-weight", "value"),
        State("seed", "value"),
        State("ack-seed", "value"),
        prevent_initial_call=True,
    )
    def save_defaults(n_clicks, objective, n_optima, intensity_mean, intensity_var,
                      noise_freq, n_major, n_micro, major_sigma,
                      micro_sigma, micro_weight, seed, ack_seed):
        if objective == "bumps":
            save_bumps_config({
                "n_major": int(n_major),
                "n_micro": int(n_micro),
                "major_sigma": float(major_sigma),
                "micro_sigma_min": float(micro_sigma[0]),
                "micro_sigma_max": float(micro_sigma[1]),
                "micro_weight_min": float(micro_weight[0]),
                "micro_weight_max": float(micro_weight[1]),
                "seed": int(seed),
            })
        else:
            # basin_width is not tunable here; preserve whatever is already in the
            # config so saving the other params doesn't drop it.
            # basin_width and noise_amp aren't tunable here; preserve whatever is
            # already in the config (noise_amp stays the fallback for dims absent
            # from NOISE_AMP_BY_DIM) so saving the other params doesn't drop them.
            existing = load_ackley_config()
            save_ackley_config({
                "n_optima": int(n_optima),
                "basin_width": float(existing.get("basin_width", 50.0)),
                "intensity_mean": float(intensity_mean),
                "intensity_var": float(intensity_var),
                "noise_freq": float(noise_freq),
                "noise_amp": float(existing.get("noise_amp", 5.0)),
                "seed": int(ack_seed),
            })
        return "Saved!"

    @callback(
        Output("cloud-plot", "figure"),
        Input("dim-select", "value"),
        Input("objective", "value"),
        Input("n-optima", "value"),
        Input("noise-freq", "value"),
        Input("intensity-mean", "value"),
        Input("intensity-var", "value"),
        Input("n-major", "value"),
        Input("n-micro", "value"),
        Input("major-sigma", "value"),
        Input("micro-sigma", "value"),
        Input("micro-weight", "value"),
        Input("seed", "value"),
        Input("grid-res-3d", "value"),
        Input("grid-res-4d", "value"),
        Input("ack-seed", "value"),
        Input("basin-threshold", "value"),
    )
    def update_plot(dim_sel, objective, n_optima, noise_freq,
                    intensity_mean, intensity_var, n_major, n_micro, major_sigma,
                    micro_sigma, micro_weight, seed, grid_res_3d, grid_res_4d,
                    ack_seed, basin_threshold):
        dim = 3 if dim_sel == "3d" else 4

        if objective == "bumps":
            fn = Bumps(
                dim=dim,
                n_major=int(n_major),
                n_micro=int(n_micro),
                major_sigma=float(major_sigma),
                micro_sigma_range=(float(micro_sigma[0]), float(micro_sigma[1])),
                micro_weight_range=(float(micro_weight[0]), float(micro_weight[1])),
                seed=int(seed),
            )
            peaks = np.array(fn.major_centers)
            title = f"Planted bumps ({n_major} major + {n_micro} micro, σ_major={major_sigma})"
        else:
            # basin_width (b) is not passed: Ackley uses the hardcoded
            # BASIN_WIDTH_BY_DIM value for this dim.
            fn = build_ackley(
                dim, n_optima, intensity_mean, intensity_var, noise_freq, ack_seed,
            )
            peaks = np.array(fn.centers)
            title = (f"Realistic Ackley ({n_optima} peaks, "
                     f"b={BASIN_WIDTH_BY_DIM[dim]} at dim {dim})")

        if dim == 3:
            comp = build_ternary_grid(int(grid_res_3d))
            obj = fn.predict(comp)
            obj_min, obj_max = float(np.nanmin(obj)), float(np.nanmax(obj))
            above = basin_mask(obj, basin_threshold)
            traces = []
            if (~above).any():
                traces.append(go.Scatterternary(
                    a=comp[~above, 2], b=comp[~above, 0], c=comp[~above, 1],
                    mode="markers", name="below basin threshold",
                    hoverinfo="skip", showlegend=False,
                    marker=dict(color="black", size=TERNARY_MARKER_SIZE),
                ))
            heat = go.Scatterternary(
                a=comp[above, 2], b=comp[above, 0], c=comp[above, 1], mode="markers",
                name="objective", hoverinfo="skip",
                marker=dict(
                    color=obj[above], colorscale="Viridis",
                    cmin=obj_min, cmax=obj_max, size=TERNARY_MARKER_SIZE,
                    showscale=True, colorbar=dict(title="Objective", x=1.02),
                ),
            )
            traces.append(heat)
            if len(peaks):
                traces.append(go.Scatterternary(
                    a=peaks[:, 2], b=peaks[:, 0], c=peaks[:, 1], mode="markers",
                    name="known peak",
                    marker=dict(symbol="star", color="red", size=14,
                                line=dict(color="white", width=1)),
                ))
            fig = go.Figure(data=traces)
            fig.update_layout(
                title=title,
                ternary=dict(
                    sum=1,
                    aaxis=dict(title="x3"),
                    baxis=dict(title="x1"),
                    caxis=dict(title="x2"),
                ),
                legend=dict(x=1.18, y=1.0),
                width=FIG_W, height=FIG_H,
                margin=dict(t=60),
            )
            return fig

        # dim == 4: tetrahedron point cloud
        comp = build_simplex_lattice(int(grid_res_4d))
        obj = fn.predict(comp)
        xyz = to_3d(comp)
        obj_min, obj_max = float(obj.min()), float(obj.max())

        # Below-threshold points become transparent (dropped) so the basins show
        # through; the colourbar stays pinned to the full objective range.
        above = basin_mask(obj, basin_threshold)
        comp_v, xyz_v, obj_v = comp[above], xyz[above], obj[above]

        hover = [
            f"x=[{a:.2f}, {b:.2f}, {c:.2f}, {d:.2f}]<br>obj={v:.2f}"
            for (a, b, c, d), v in zip(comp_v, obj_v)
        ]

        cloud = go.Scatter3d(
            x=xyz_v[:, 0], y=xyz_v[:, 1], z=xyz_v[:, 2], mode="markers",
            name="objective", text=hover, hoverinfo="text",
            marker=dict(
                color=obj_v, colorscale="Viridis",
                cmin=obj_min, cmax=obj_max,
                size=MARKER_SIZE, opacity=MARKER_OPACITY,
                showscale=True, colorbar=dict(title="Objective"),
            ),
        )

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
            title=title,
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
        description="Interactive simplex objective viewer (3D ternary / 4D point "
                    "cloud). With RUN_DIR and ITERATION, instead export a static "
                    "4-simplex point_cloud_<n>.html showing the needles discovered "
                    "up to that iteration; with no positional args, launch the "
                    "interactive Dash app.")
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
        print("Starting Dash app at http://127.0.0.1:8050")
        build_app().run(debug=True, port=8050)


if __name__ == "__main__":
    main()
