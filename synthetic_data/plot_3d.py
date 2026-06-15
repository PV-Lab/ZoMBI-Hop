"""Interactive ternary plot of a tunable realistic Ackley + noise objective.

Runs a Dash app with sliders for noise frequency, noise amplitude, number of
Ackley optima, and intensity offsets.  The ternary heatmap updates in real time
as you drag any slider.  Basin width (b) is not tunable here: it is the hardcoded
``BASIN_WIDTH_BY_DIM[3]`` in ``ackley.py``.  Click "Save as Default" to persist
the current slider values to ``synthetic_data/ackley/defaults.json``.

Usage:
    python plot_3d.py
"""

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, ctx, dcc, html

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from synthetic_data.ackley import Ackley, load_config, save_config, BASIN_WIDTH_BY_DIM  # noqa: E402

DIM = 3
GRID_N = 150
MARKER_SIZE = 4.0
FIG_W, FIG_H = 920, 860


def build_grid(grid_n: int) -> np.ndarray:
    pts = [
        (i / grid_n, j / grid_n, (grid_n - i - j) / grid_n)
        for i in range(grid_n + 1)
        for j in range(grid_n + 1 - i)
    ]
    return np.array(pts)


GRID = build_grid(GRID_N)
GA, GB, GC = GRID[:, 2], GRID[:, 0], GRID[:, 1]

cfg = load_config()

app = Dash(__name__)

app.layout = html.Div([
    html.H2("Tunable Realistic Ackley on the 3-Simplex",
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
            html.Button("Save as Default", id="save-btn", n_clicks=0,
                        style={"marginTop": "10px", "padding": "8px 24px"}),
            html.Span(id="save-status", style={"marginLeft": "12px"}),
        ], style={"padding": "10px", "textAlign": "center"}),
    ], style={"width": "60%", "margin": "0 auto"}),
    dcc.Graph(id="ternary-plot", style={"height": f"{FIG_H}px"}),
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
    # basin_width is no longer tunable here (it's hardcoded per-dim in ackley.py);
    # preserve whatever is already in the config so saving doesn't drop it.
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
    Output("ternary-plot", "figure"),
    Input("n-optima", "value"),
    Input("noise-freq", "value"),
    Input("noise-amp", "value"),
    Input("intensity-mean", "value"),
    Input("intensity-var", "value"),
)
def update_plot(n_optima, noise_freq, noise_amp, intensity_mean, intensity_var):
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
    obj = fn.predict(GRID)
    obj_min, obj_max = float(np.nanmin(obj)), float(np.nanmax(obj))

    heat = go.Scatterternary(
        a=GA, b=GB, c=GC, mode="markers", name="objective", hoverinfo="skip",
        marker=dict(
            color=obj, colorscale="Viridis",
            cmin=obj_min, cmax=obj_max, size=MARKER_SIZE,
            showscale=True, colorbar=dict(title="Objective", x=1.02),
        ),
    )
    traces = [heat]

    peaks = np.array(fn.centers)
    if len(peaks):
        traces.append(go.Scatterternary(
            a=peaks[:, 2], b=peaks[:, 0], c=peaks[:, 1], mode="markers",
            name="known peak",
            marker=dict(symbol="star", color="red", size=14,
                        line=dict(color="white", width=1)),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Tunable Realistic Ackley ({n_optima} peaks, b={BASIN_WIDTH_BY_DIM[DIM]} at dim {DIM})",
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


if __name__ == "__main__":
    print("Starting Dash app at http://127.0.0.1:8050")
    app.run(debug=True)
