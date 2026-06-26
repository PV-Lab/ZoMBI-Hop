"""Interactive viewer for the layered :class:`~synthetic_data.ensemble.Ensemble`
objective on the simplex.

A single Dash app shows **one plot at a time**:

  * **dimensionality** dropdown — "3D" draws the 3-simplex as a ternary heatmap;
    "4D" draws the 4-simplex as a tetrahedron point cloud (objective -> colour).

Every parameter of every feature (true optima + their placement/clustering, weak
optima, ridges incl. length, roughness, anisotropy, plateaus, and the edge /
structural bias) has its own control, plus a **negative-fraction** slider (share
of features that subtract mass), a master **random seed** slider, **grid
resolution** slider, and a **basin threshold**.  Each optional feature has an
on/off checkbox; unchecking it passes the disabling value (count/amplitude 0)
without losing the slider position.  There is intentionally no save/load of
defaults.

Geometry helpers (simplex lattices, tetra rendering) are reused from
``plot_ackley.py`` so the two viewers stay visually consistent.

Usage:
    python plot_ensemble.py
"""

import random
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from synthetic_data.ensemble import (  # noqa: E402
    Ensemble,
    random_ensemble_config,
)
from synthetic_data.plot_ackley import (  # noqa: E402
    BASIN_THRESHOLD,
    DEFAULT_GRID_N,
    DEFAULT_GRID_N_3D,
    FIG_H,
    FIG_W,
    MARKER_OPACITY,
    MARKER_SIZE,
    TERNARY_MARKER_SIZE,
    basin_mask,
    build_simplex_lattice,
    build_ternary_grid,
    tetra_edges_trace,
    to_3d,
    vertex_labels_trace,
)


# ── Slider helper ────────────────────────────────────────────────────────────

def _slider(label, sid, lo, hi, step, value, marks_every, fmt=str):
    n_marks = int(round((hi - lo) / marks_every)) + 1
    marks = {}
    for i in range(n_marks):
        v = lo + i * marks_every
        v = round(v, 6)
        marks[v] = fmt(v)
    return html.Div([
        html.Label(label),
        dcc.Slider(id=sid, min=lo, max=hi, step=step, value=value, marks=marks,
                   tooltip={"placement": "bottom", "always_visible": True}),
    ], style={"padding": "8px"})


def _feature_block(title, toggle_id, children, enabled=True):
    return html.Div([
        dcc.Checklist(id=toggle_id, options=[{"label": f"  {title}", "value": "on"}],
                      value=["on"] if enabled else [],
                      style={"fontWeight": "bold", "marginTop": "6px"}),
        html.Div(children, style={"paddingLeft": "14px"}),
    ], style={"borderTop": "1px solid #ddd", "padding": "4px 0"})


# ── Dash app ─────────────────────────────────────────────────────────────────

def build_app():
    global html, dcc  # used by the helpers above after Dash import
    from dash import Dash, Input, Output, State, callback, dcc, html, no_update  # noqa: F811

    app = Dash(__name__)

    controls = html.Div([
        html.Div([
            html.Label("Dimensionality"),
            dcc.Dropdown(id="dim-select", clearable=False,
                         options=[{"label": "3D (ternary heatmap)", "value": "3d"},
                                  {"label": "4D (tetrahedron point cloud)", "value": "4d"}],
                         value="3d"),
        ], style={"padding": "8px"}),

        # True optima (always on).
        html.Div([
            html.Label("True Optima", style={"fontWeight": "bold"}),
            _slider("Number of Optima", "n-optima", 1, 150, 1, 4, 25, str),
            _slider("Basin Width (b)", "basin-width", 5, 200, 1, 65, 35, str),
            _slider("Optima Margin (normalized gap above background)",
                    "optima-margin", 0.0, 0.5, 0.01, 0.1, 0.1,
                    lambda v: f"{v:.2f}"),
            html.Div([
                html.Label("Placement / Clustering"),
                dcc.Dropdown(id="optima-layout", clearable=False,
                             options=[{"label": "Scatter (uniform)", "value": "scatter"},
                                      {"label": "Cluster @ corners", "value": "corners"},
                                      {"label": "Cluster @ edges", "value": "edges"},
                                      {"label": "Cluster @ faces", "value": "faces"},
                                      {"label": "Cluster @ middle", "value": "middle"}],
                             value="scatter"),
            ], style={"padding": "8px"}),
            _slider("Cluster Count", "n-clusters", 1, 6, 1, 3, 1, str),
            _slider("Cluster Concentration (higher = tighter)",
                    "cluster-conc", 20, 250, 5, 80, 50, str),
            _slider("Cluster Spread (higher = looser hug of region)",
                    "cluster-spread", 0.0, 1.0, 0.05, 0.0, 0.25, str),
        ], style={"borderTop": "2px solid #999", "padding": "4px 0"}),

        _feature_block("Weak Optima (distractors)", "tog-weak", [
            _slider("Count", "n-weak", 0, 30, 1, 6, 5, str),
            _slider("Basin Width (b)", "weak-width", 5, 300, 1, 120, 50, str),
            _slider("Prominence", "weak-amp", 0.0, 1.0, 0.02, 0.6, 0.25,
                    lambda v: f"{v:.2f}"),
        ]),

        _feature_block("Ridges", "tog-ridges", [
            _slider("Count", "n-ridges", 0, 8, 1, 2, 2, str),
            _slider("Tube Width", "ridge-width", 0.01, 0.25, 0.005, 0.06, 0.06,
                    lambda v: f"{v:.2f}"),
            _slider("Prominence", "ridge-amp", 0.0, 1.0, 0.02, 0.6, 0.25,
                    lambda v: f"{v:.2f}"),
            _slider("Length", "ridge-length", 0.1, 1.0, 0.05, 1.0, 0.3,
                    lambda v: f"{v:.2f}"),
        ]),

        _feature_block("Roughness (Perlin noise)", "tog-rough", [
            _slider("Frequency", "noise-freq", 0, 40, 0.5, 8, 10, str),
            _slider("Amplitude (raw)", "noise-amp", 0, 2000, 100, 120, 200, str),
            _slider("Octaves", "noise-oct", 1, 6, 1, 4, 1, str),
        ]),

        _feature_block("Anisotropy (stretch axes)", "tog-aniso", [
            _slider("Strength", "aniso-strength", 0.0, 50, 1, 0.0, 10,
                    lambda v: f"{v:.1f}"),
        ]),

        _feature_block("Plateaus", "tog-plateaus", [
            _slider("Count", "n-plateaus", 0, 8, 1, 2, 2, str),
            _slider("Radius", "plateau-radius", 0.02, 0.40, 0.01, 0.12, 0.1,
                    lambda v: f"{v:.2f}"),
            _slider("Mesa Height", "plateau-amp", 0.0, 1.0, 0.02, 0.7, 0.25,
                    lambda v: f"{v:.2f}"),
        ]),

        _feature_block("Edge / Structural Bias", "tog-edge", [
            html.Div([
                html.Label("Target Region"),
                dcc.Dropdown(id="edge-region", clearable=False,
                             options=[{"label": "Corners", "value": "corners"},
                                      {"label": "Edges", "value": "edges"},
                                      {"label": "Faces", "value": "faces"},
                                      {"label": "Middle", "value": "middle"}],
                             value="corners"),
            ], style={"padding": "8px"}),
            _slider("Prominence", "edge-amp", 0.0, 1.0, 0.02, 0.4, 0.25,
                    lambda v: f"{v:.2f}"),
            _slider("Reach", "edge-reach", 0.05, 0.80, 0.05, 0.3, 0.25,
                    lambda v: f"{v:.2f}"),
        ], enabled=False),

        # Global controls.
        html.Div([
            html.Label("Global", style={"fontWeight": "bold"}),
            _slider("Random Seed", "seed", 0, 100, 1, 0, 20, str),
            _slider("Negative Fraction (share of features that subtract mass)",
                    "neg-frac", 0.0, 1.0, 0.02, 0.5, 0.25, lambda v: f"{v:.2f}"),
            html.Div(_slider("Grid Resolution", "grid-res-3d", 30, 180, 10,
                             160, 30, str), id="grid-3d-wrap"),
            html.Div(_slider("Grid Resolution", "grid-res-4d", 15, 50, 5,
                             DEFAULT_GRID_N, 5, str), id="grid-4d-wrap",
                     style={"display": "none"}),
            _slider("Basin Threshold (below: blacked out 3D / transparent 4D)",
                    "basin-threshold", 0.0, 1.0, 0.01, 0.0, 0.1,
                    lambda v: f"{v:.2f}"),
        ], style={"borderTop": "2px solid #999", "padding": "4px 0"}),
        html.Div([
            html.Button("Randomize", id="randomize-btn", n_clicks=0,
                         style={"fontSize": "16px", "padding": "8px 24px",
                                "margin": "12px auto", "display": "block",
                                "cursor": "pointer"}),
        ]),
    ], style={"width": "60%", "margin": "0 auto"})

    app.layout = html.Div([
        html.H2("Layered Synthetic Objective (Ensemble) on the Simplex",
                style={"textAlign": "center"}),
        controls,
        dcc.Graph(id="cloud-plot", style={"height": f"{FIG_H}px"}),
    ])

    @callback(
        Output("grid-3d-wrap", "style"),
        Output("grid-4d-wrap", "style"),
        Input("dim-select", "value"),
    )
    def toggle_grid_controls(dim_sel):
        show, hide = {}, {"display": "none"}
        return (hide, show) if dim_sel == "4d" else (show, hide)

    @callback(
        Output("n-optima", "value"),
        Output("basin-width", "value"),
        Output("optima-layout", "value"),
        Output("n-clusters", "value"),
        Output("cluster-conc", "value"),
        Output("cluster-spread", "value"),
        Output("n-weak", "value"),
        Output("weak-width", "value"),
        Output("weak-amp", "value"),
        Output("n-ridges", "value"),
        Output("ridge-width", "value"),
        Output("ridge-amp", "value"),
        Output("ridge-length", "value"),
        Output("noise-freq", "value"),
        Output("noise-amp", "value"),
        Output("noise-oct", "value"),
        Output("aniso-strength", "value"),
        Output("n-plateaus", "value"),
        Output("plateau-radius", "value"),
        Output("plateau-amp", "value"),
        Output("edge-region", "value"),
        Output("edge-amp", "value"),
        Output("edge-reach", "value"),
        Output("neg-frac", "value"),
        Output("tog-weak", "value"),
        Output("tog-ridges", "value"),
        Output("tog-rough", "value"),
        Output("tog-aniso", "value"),
        Output("tog-plateaus", "value"),
        Output("tog-edge", "value"),
        Input("randomize-btn", "n_clicks"),
        State("dim-select", "value"),
        prevent_initial_call=True,
    )
    def randomize(_n_clicks, dim_sel):
        # Draw exactly what optimize/run_mobo.py and optimize/evaluate.py generate
        # per run, so the viewer's "Randomize" matches the benchmark landscapes.
        # A random Sobol' index + scramble seed gives a fresh landscape per click.
        dim = 3 if dim_sel == "3d" else 4
        cfg = random_ensemble_config(dim, index=random.randrange(1 << 20),
                                     seed=random.randrange(1 << 16))
        on = lambda v: ["on"] if v else []  # noqa: E731
        return (
            cfg["n_optima"],                             # n-optima
            cfg["basin_width"],                          # basin-width
            cfg["optima_layout"],                        # optima-layout
            cfg["n_optima_clusters"],                    # n-clusters
            cfg["optima_cluster_conc"],                  # cluster-conc
            cfg["optima_cluster_spread"],                # cluster-spread
            cfg["n_weak"],                               # n-weak
            cfg["weak_width"],                           # weak-width
            cfg["weak_amp"],                             # weak-amp
            cfg["n_ridges"],                             # n-ridges
            cfg["ridge_width"],                          # ridge-width
            cfg["ridge_amp"],                            # ridge-amp
            cfg["ridge_length"],                         # ridge-length
            cfg["noise_freq"],                           # noise-freq
            cfg["noise_amp"],                            # noise-amp
            cfg["noise_octaves"],                        # noise-oct
            cfg["aniso_strength"],                       # aniso-strength
            cfg["n_plateaus"],                           # n-plateaus
            cfg["plateau_radius"],                       # plateau-radius
            cfg["plateau_amp"],                          # plateau-amp
            cfg["edge_region"] or "corners",             # edge-region
            cfg["edge_amp"],                             # edge-amp
            cfg["edge_reach"],                           # edge-reach
            cfg["neg_frac"],                             # neg-frac
            on(cfg["n_weak"] > 0),                       # tog-weak
            on(cfg["n_ridges"] > 0),                     # tog-ridges
            on(cfg["noise_amp"] > 0),                    # tog-rough
            on(cfg["aniso_strength"] > 0),               # tog-aniso
            on(cfg["n_plateaus"] > 0),                   # tog-plateaus
            on(cfg["edge_region"] is not None),          # tog-edge
        )

    @callback(
        Output("cloud-plot", "figure"),
        Input("dim-select", "value"),
        Input("n-optima", "value"),
        Input("basin-width", "value"),
        Input("optima-margin", "value"),
        Input("optima-layout", "value"),
        Input("n-clusters", "value"),
        Input("cluster-conc", "value"),
        Input("cluster-spread", "value"),
        Input("tog-weak", "value"),
        Input("n-weak", "value"),
        Input("weak-width", "value"),
        Input("weak-amp", "value"),
        Input("tog-ridges", "value"),
        Input("n-ridges", "value"),
        Input("ridge-width", "value"),
        Input("ridge-amp", "value"),
        Input("ridge-length", "value"),
        Input("tog-rough", "value"),
        Input("noise-freq", "value"),
        Input("noise-amp", "value"),
        Input("noise-oct", "value"),
        Input("tog-aniso", "value"),
        Input("aniso-strength", "value"),
        Input("tog-plateaus", "value"),
        Input("n-plateaus", "value"),
        Input("plateau-radius", "value"),
        Input("plateau-amp", "value"),
        Input("tog-edge", "value"),
        Input("edge-region", "value"),
        Input("edge-amp", "value"),
        Input("edge-reach", "value"),
        Input("neg-frac", "value"),
        Input("seed", "value"),
        Input("grid-res-3d", "value"),
        Input("grid-res-4d", "value"),
        Input("basin-threshold", "value"),
    )
    def update_plot(dim_sel, n_optima, basin_width, optima_margin,
                    optima_layout, n_clusters, cluster_conc, cluster_spread,
                    tog_weak, n_weak, weak_width, weak_amp,
                    tog_ridges, n_ridges, ridge_width, ridge_amp, ridge_length,
                    tog_rough, noise_freq, noise_amp, noise_oct,
                    tog_aniso, aniso_strength,
                    tog_plateaus, n_plateaus, plateau_radius, plateau_amp,
                    tog_edge, edge_region, edge_amp, edge_reach, neg_frac,
                    seed, grid_res_3d, grid_res_4d, basin_threshold):
        dim = 3 if dim_sel == "3d" else 4
        on = lambda t: bool(t)  # noqa: E731

        fn = Ensemble(
            dim=dim,
            n_optima=int(n_optima),
            basin_width=float(basin_width),
            optima_margin=float(optima_margin),
            optima_layout=str(optima_layout),
            n_optima_clusters=int(n_clusters),
            optima_cluster_conc=float(cluster_conc),
            optima_cluster_spread=float(cluster_spread),
            n_weak=int(n_weak) if on(tog_weak) else 0,
            weak_width=float(weak_width),
            weak_amp=float(weak_amp),
            n_ridges=int(n_ridges) if on(tog_ridges) else 0,
            ridge_width=float(ridge_width),
            ridge_amp=float(ridge_amp),
            ridge_length=float(ridge_length),
            noise_freq=float(noise_freq),
            noise_amp=float(noise_amp) if on(tog_rough) else 0.0,
            noise_octaves=int(noise_oct),
            aniso_strength=float(aniso_strength) if on(tog_aniso) else 0.0,
            n_plateaus=int(n_plateaus) if on(tog_plateaus) else 0,
            plateau_radius=float(plateau_radius),
            plateau_amp=float(plateau_amp),
            edge_region=str(edge_region) if on(tog_edge) else None,
            edge_amp=float(edge_amp),
            edge_reach=float(edge_reach),
            neg_frac=float(neg_frac),
            seed=int(seed),
        )
        peaks = np.asarray(fn.centers)
        title = (f"Ensemble — {len(peaks)} optima, margin={optima_margin:g} "
                 f"(dim {dim}, seed {seed})")

        if dim == 3:
            return _ternary_figure(fn, peaks, int(grid_res_3d), basin_threshold, title)
        return _point_cloud_figure(fn, peaks, int(grid_res_4d), basin_threshold, title)

    return app


# ── Figure builders ──────────────────────────────────────────────────────────

def _ternary_figure(fn, peaks, grid_n, basin_threshold, title):
    comp = build_ternary_grid(grid_n)
    obj = fn.predict(comp)
    obj_min, obj_max = float(np.nanmin(obj)), float(np.nanmax(obj))
    above = basin_mask(obj, basin_threshold)

    traces = []
    if (~above).any():
        traces.append(go.Scatterternary(
            a=comp[~above, 2], b=comp[~above, 0], c=comp[~above, 1],
            mode="markers", name="below basin threshold", hoverinfo="skip",
            showlegend=False, marker=dict(color="black", size=TERNARY_MARKER_SIZE),
        ))
    traces.append(go.Scatterternary(
        a=comp[above, 2], b=comp[above, 0], c=comp[above, 1], mode="markers",
        name="objective", hoverinfo="skip",
        marker=dict(color=obj[above], colorscale="Viridis", cmin=obj_min, cmax=obj_max,
                    size=TERNARY_MARKER_SIZE, showscale=True,
                    colorbar=dict(title="Objective", x=1.02)),
    ))
    if len(peaks):
        traces.append(go.Scatterternary(
            a=peaks[:, 2], b=peaks[:, 0], c=peaks[:, 1], mode="markers",
            name="known peak", visible="legendonly",
            marker=dict(symbol="star", color="red", size=14,
                        line=dict(color="white", width=1)),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        ternary=dict(sum=1, aaxis=dict(title="x3"), baxis=dict(title="x1"),
                     caxis=dict(title="x2")),
        legend=dict(x=1.18, y=1.0), width=FIG_W, height=FIG_H, margin=dict(t=60),
    )
    return fig


def _point_cloud_figure(fn, peaks, grid_n, basin_threshold, title):
    comp = build_simplex_lattice(grid_n)
    obj = fn.predict(comp)
    xyz = to_3d(comp)
    obj_min, obj_max = float(obj.min()), float(obj.max())

    above = basin_mask(obj, basin_threshold)
    xyz_v, obj_v, comp_v = xyz[above], obj[above], comp[above]
    hover = [f"x=[{a:.2f}, {b:.2f}, {c:.2f}, {d:.2f}]<br>obj={v:.2f}"
             for (a, b, c, d), v in zip(comp_v, obj_v)]

    cloud = go.Scatter3d(
        x=xyz_v[:, 0], y=xyz_v[:, 1], z=xyz_v[:, 2], mode="markers",
        name="objective", text=hover, hoverinfo="text",
        marker=dict(color=obj_v, colorscale="Viridis", cmin=obj_min, cmax=obj_max,
                    size=MARKER_SIZE, opacity=MARKER_OPACITY, showscale=True,
                    colorbar=dict(title="Objective")),
    )
    data = [cloud, tetra_edges_trace(), vertex_labels_trace()]
    if len(peaks):
        peaks_xyz = to_3d(peaks)
        data.append(go.Scatter3d(
            x=peaks_xyz[:, 0], y=peaks_xyz[:, 1], z=peaks_xyz[:, 2], mode="markers",
            name="known peak", visible="legendonly",
            marker=dict(symbol="diamond", color="red", size=6,
                        line=dict(color="white", width=1)),
            hoverinfo="name",
        ))

    fig = go.Figure(data=data)
    fig.update_layout(
        title=title,
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                   zaxis=dict(visible=False), aspectmode="data"),
        legend=dict(x=0.0, y=1.0), width=FIG_W, height=FIG_H,
    )
    return fig


def main() -> None:
    print("Starting Dash app at http://127.0.0.1:8051")
    build_app().run(debug=True, port=8051)


if __name__ == "__main__":
    main()
