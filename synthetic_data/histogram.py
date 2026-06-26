"""Interactive Dash app to histogram the objective-value distribution of a
synthetic benchmark over the probability simplex, alongside the campaign1a RF
surrogate.

A dropdown selects the mode (what the old ``--ackley`` / ``--ensemble`` command
line flags used to control), a checkbox toggles the log y-axis (the old
``--log`` flag), and a slider controls the basin width for the ``one-peak`` mode.

Modes
-----
``ackley``
    For each dimensionality in ``DIMENSIONS`` draw a large uniform sample from the
    probability simplex, evaluate the analytic Ackley objective
    (synthetic_data/ackley.py) at every point, and histogram the resulting
    objective values -- one panel per dimensionality.  A final panel does the same
    for the Random-Forest surrogate that
    interactive_testing/interactive_test_zombi.py trains on campaign1a.csv
    (3-component composition -> Objective), so its landscape can be compared on
    equal footing.

``ensemble``
    Two panels.  On the **left**, a *combined* histogram of ``ENSEMBLE_N_RUNS``
    random :class:`~synthetic_data.ensemble.Ensemble` landscapes (all 3-D): every
    run is sampled uniformly on the simplex and all the objective values across the
    runs are pooled into one shared set of bins, so the panel shows what the
    distribution looks like aggregated over many random benchmark instances.  On
    the **right**, the same campaign1a RF surrogate panel as the Ackley mode.

``one-peak``
    Three panels: an Ackley objective with a single peak at the simplex centroid,
    evaluated at dim = 3, 4 and 10.  The basin width ``b`` (slider, 2.2 .. 50) is
    held *identical* across all three dimensionalities -- the Ackley envelope
    normalizes the squared distance by the dimension internally, so the same ``b``
    means the same basin shape regardless of ``d`` (it does not scale with ``d``).

``gaussian-bump``
    Like ``one-peak``, but the envelope is a plain Gaussian bump
    ``exp(-r^2 / (2 w^2))`` centered at the simplex centroid, evaluated at dim =
    3, 4 and 10.  The squared distance ``r^2`` is normalized by the dimension, so
    the bump width ``w`` (slider, 0.02 .. 1.0) is held *identical* across all
    three dimensionalities and does not scale with ``d``.

Reading the plot as a basin-volume diagnostic: the objective is highest at the
optima and falls off away from them, so the histogram is the distribution of "how
good" a uniformly random point is.  Mass piled up at the low end with only a thin
tail reaching the top means the near-optimal basins occupy a tiny fraction of the
simplex volume.

Usage:
    python histogram.py            # launches the Dash app at http://127.0.0.1:8050
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
# Make the repo root importable so this runs from any working directory.
sys.path.insert(0, str(HERE.parent))
from synthetic_data.ackley import (  # noqa: E402
    Ackley, _negated_ackley, ACKLEY_A, ACKLEY_SCALE,
)
from synthetic_data.ensemble import Ensemble, random_ensemble_config  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────────
DIMENSIONS = [3, 4, 10]   # simplex dimensionalities to examine (ackley/one-peak)
N_SAMPLES = 50_000        # uniform simplex samples used to estimate the distribution
N_BINS = 80               # histogram bins per panel
SEED = 0                  # RNG seed for reproducible sampling

# Ensemble mode: pool this many random 3-D landscapes into one combined histogram.
ENSEMBLE_DIM = 3
ENSEMBLE_N_RUNS = 10
ENSEMBLE_SEED = 0         # seeds the per-run random configs (reproducible)

# One-peak mode: basin-width slider bounds.
ONE_PEAK_B_MIN = 2.2
ONE_PEAK_B_MAX = 50.0
ONE_PEAK_B_DEFAULT = 20.0

# Gaussian-bump mode: bump-width (sigma) slider bounds.
GAUSS_W_MIN = 0.02
GAUSS_W_MAX = 1.0
GAUSS_W_DEFAULT = 0.2

# RF surrogate settings -- mirror interactive_testing/interactive_test_zombi.py.
RF_CSV = HERE.parent / "interactive_testing" / "campaign1a.csv"
RF_COMPOSITION_COLS = ["FAPbI3", "MAPbI3", "MAPbBr3"]
RF_OBJECTIVE_COL = "Objective"
RF_N_ESTIMATORS = 500
RF_RANDOM_STATE = 42

# Shared histogram range (objective values are mapped onto [0.5, 1]).
HIST_RANGE = (0.5, 1.0)


def sample_simplex(dim: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` points uniformly from the ``dim``-element probability simplex."""
    return rng.dirichlet(np.ones(dim), size=n)


@lru_cache(maxsize=1)
def load_rf_surrogate() -> RandomForestRegressor:
    """Train the campaign1a Random-Forest surrogate (same recipe as the
    interactive ZoMBI-Hop test: 3 composition columns row-normalized to the
    simplex -> Objective, 500 trees, fixed seed).  Cached so the app trains once."""
    df = pd.read_csv(RF_CSV).dropna(subset=RF_COMPOSITION_COLS + [RF_OBJECTIVE_COL])
    X = df[RF_COMPOSITION_COLS].to_numpy(dtype=float)
    X = X / np.where(X.sum(axis=1, keepdims=True) == 0, 1.0, X.sum(axis=1, keepdims=True))
    y = df[RF_OBJECTIVE_COL].to_numpy(dtype=float)
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, n_jobs=-1, random_state=RF_RANDOM_STATE
    )
    rf.fit(X, y)
    return rf


def rf_panel(rng: np.random.Generator) -> tuple:
    """The campaign1a RF-surrogate histogram panel (shared by both modes)."""
    rf = load_rf_surrogate()
    y_rf = rf.predict(sample_simplex(len(RF_COMPOSITION_COLS), N_SAMPLES, rng))
    return (f"RF surrogate<br>(campaign1a, dim = {len(RF_COMPOSITION_COLS)})",
            y_rf, "indianred", 1.0)


def ackley_panels(rng: np.random.Generator) -> list:
    """One Ackley panel per dimensionality in ``DIMENSIONS``."""
    panels = []
    for dim in DIMENSIONS:
        fn = Ackley("realistic", dim=dim)
        y = fn.predict(sample_simplex(dim, N_SAMPLES, rng))
        panels.append((f"Ackley 'realistic'<br>dim = {dim}", y, "steelblue", 1.0))
    return panels


def one_peak_panels(rng: np.random.Generator, basin_width: float) -> list:
    """One Ackley panel per dim with a *single* peak at the simplex centroid.

    The same ``basin_width`` (``b``) is used for every dimensionality.  Inside
    :func:`_negated_ackley` the squared distance to the center is divided by the
    dimension, so a fixed ``b`` produces the same basin shape at d = 3, 4 and 10
    -- the basin width does not scale with dimension.

    Each panel maps the negated-Ackley envelope onto [0, 1] with a *fixed*
    analytic transform (peak -> 1, far-field -> 0): no per-panel min-max rescale,
    so the panels read on a true common scale across dims.
    """
    panels = []
    for dim in DIMENSIONS:
        center = np.full(dim, 1.0 / dim)
        X = sample_simplex(dim, N_SAMPLES, rng)
        raw = _negated_ackley(X, center, b=basin_width)
        # ``raw`` lives in [-scale*a, 0] (0 at the peak).  Divide by the constant
        # envelope depth ``scale * a`` and shift to land on [0, 1] with the peak
        # at 1.  The constant is identical across dims -- unlike a per-panel
        # min-max it keeps the panels directly comparable.
        y = raw / (ACKLEY_SCALE * ACKLEY_A) + 1.0
        panels.append((f"Ackley one-peak<br>dim = {dim}, b = {basin_width:g}",
                       y, "darkorange", 1.0))
    return panels


def gaussian_bump_panels(rng: np.random.Generator, bump_width: float) -> list:
    """One Gaussian-bump panel per dim with a *single* peak at the simplex centroid.

    Mirrors :func:`one_peak_panels`, but the envelope is a plain Gaussian bump
    ``exp(-r^2 / (2 * w^2))`` where ``r^2`` is the squared distance to the center
    divided by the dimension (the same normalization :func:`_negated_ackley` uses).
    Because of that per-dimension normalization the same ``bump_width`` (``w``)
    produces the same bump shape at d = 3, 4 and 10 -- the width does not scale
    with dimension.

    The Gaussian bump is already in [0, 1] (peak -> 1, far-field -> 0), so it is
    plotted directly with no per-panel rescale -- the values sit on a true common
    scale across dims.
    """
    panels = []
    for dim in DIMENSIONS:
        center = np.full(dim, 1.0 / dim)
        X = sample_simplex(dim, N_SAMPLES, rng)
        r2 = np.sum((X - center) ** 2, axis=1) / dim
        y = np.exp(-r2 / (2.0 * bump_width ** 2))
        panels.append((f"Gaussian bump<br>dim = {dim}, w = {bump_width:g}",
                       y, "mediumpurple", 1.0))
    return panels


def build_figure(mode: str, log: bool, basin_width: float,
                 bump_width: float) -> go.Figure:
    """Build the multi-panel histogram figure for the selected ``mode``."""
    rng = np.random.default_rng(SEED)

    if mode == "ensemble":
        panels = [ensemble_combined_panel(rng), rf_panel(rng)]
        suptitle = "Objective distribution: Ensemble vs. campaign1a RF"
    elif mode == "one-peak":
        panels = one_peak_panels(rng, basin_width)
        suptitle = "Objective distribution: single centered Ackley peak across dims"
    elif mode == "gaussian-bump":
        panels = gaussian_bump_panels(rng, bump_width)
        suptitle = "Objective distribution: single centered Gaussian bump across dims"
    else:  # "ackley"
        panels = ackley_panels(rng) + [rf_panel(rng)]
        suptitle = "Objective distribution: Ackley vs. campaign1a RF"

    # one-peak / gaussian-bump are no longer min-max rescaled, so their values
    # span the full [0, 1] envelope rather than the [0.5, 1] predict() range.
    hist_range = (0.0, 1.0) if mode in ("one-peak", "gaussian-bump") else HIST_RANGE
    bins = np.linspace(hist_range[0], hist_range[1], N_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    titles = [title for title, *_ in panels]
    fig = make_subplots(
        rows=1, cols=len(panels), shared_yaxes=True,
        subplot_titles=titles, horizontal_spacing=0.04,
    )

    for col, (title, y, color, weight) in enumerate(panels, start=1):
        counts, _ = np.histogram(y, bins=bins, weights=np.full(y.shape, weight))
        fig.add_trace(
            go.Bar(x=centers, y=counts, width=(bins[1] - bins[0]),
                   marker_color=color, marker_line_color="white",
                   marker_line_width=0.3, showlegend=False),
            row=1, col=col,
        )
        fig.update_xaxes(range=list(hist_range), title_text="objective value",
                         row=1, col=col)

    fig.update_yaxes(title_text="count", type="log" if log else "linear",
                     row=1, col=1)
    if log:
        for col in range(2, len(panels) + 1):
            fig.update_yaxes(type="log", row=1, col=col)

    fig.update_layout(
        title=suptitle, bargap=0, template="plotly_white",
        height=520, margin=dict(t=90, b=60),
    )
    return fig


def ensemble_combined_panel(rng: np.random.Generator) -> tuple:
    """Pool ``ENSEMBLE_N_RUNS`` random 3-D Ensemble landscapes into one panel.

    Each run draws a fresh random Ensemble config (same recipe the benchmark and
    the plot_ensemble.py "Randomize" button use): walking the Sobol' sweep
    ``index = 0, 1, ..., ENSEMBLE_N_RUNS - 1`` at ``seed=ENSEMBLE_SEED`` gives a
    low-discrepancy spread of landscapes.  Each is sampled uniformly on the
    simplex, and the objective values from every run are concatenated so they all
    fall into one shared set of bins.

    Each sample is weighted by ``1 / ENSEMBLE_N_RUNS`` so the bin heights are the
    *average* per-run count -- the same magnitude as the single-run RF panel, so
    both panels read on the same scale.
    """
    ys = []
    for i in range(ENSEMBLE_N_RUNS):
        cfg = random_ensemble_config(ENSEMBLE_DIM, index=i, seed=ENSEMBLE_SEED)
        fn = Ensemble(**cfg)
        ys.append(fn.predict(sample_simplex(ENSEMBLE_DIM, N_SAMPLES, rng)))
    y = np.concatenate(ys)
    return (f"Ensemble (combined)<br>{ENSEMBLE_N_RUNS} random runs, dim = {ENSEMBLE_DIM}",
            y, "seagreen", 1.0 / ENSEMBLE_N_RUNS)


# ── Dash app ──────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)

app.layout = html.Div(
    style={"maxWidth": "1400px", "margin": "0 auto", "fontFamily": "sans-serif"},
    children=[
        html.H2("Objective-distribution histograms"),
        html.Div(
            style={"display": "flex", "gap": "32px", "alignItems": "flex-end",
                   "flexWrap": "wrap", "marginBottom": "12px"},
            children=[
                html.Div([
                    html.Label("Mode"),
                    dcc.Dropdown(
                        id="mode",
                        options=[
                            {"label": "Ackley (per-dim) + RF", "value": "ackley"},
                            {"label": "Ensemble (combined) + RF", "value": "ensemble"},
                            {"label": "One-peak (centered, per-dim)", "value": "one-peak"},
                            {"label": "Gaussian bump (centered, per-dim)", "value": "gaussian-bump"},
                        ],
                        value="ackley", clearable=False,
                        style={"width": "320px"},
                    ),
                ]),
                html.Div([
                    dcc.Checklist(
                        id="log",
                        options=[{"label": " log y-axis", "value": "log"}],
                        value=[],
                    ),
                ]),
            ],
        ),
        html.Div(
            id="basin-controls",
            children=[
                html.Label(id="basin-label"),
                dcc.Slider(
                    id="basin-width",
                    min=ONE_PEAK_B_MIN, max=ONE_PEAK_B_MAX, step=0.1,
                    value=ONE_PEAK_B_DEFAULT,
                    marks={int(v): str(int(v))
                           for v in np.linspace(ONE_PEAK_B_MIN, ONE_PEAK_B_MAX, 6)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ],
        ),
        html.Div(
            id="bump-controls",
            children=[
                html.Label(id="bump-label"),
                dcc.Slider(
                    id="bump-width",
                    min=GAUSS_W_MIN, max=GAUSS_W_MAX, step=0.01,
                    value=GAUSS_W_DEFAULT,
                    marks={round(v, 2): f"{v:g}"
                           for v in np.linspace(GAUSS_W_MIN, GAUSS_W_MAX, 6)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ],
        ),
        dcc.Graph(id="histogram", style={"height": "560px"}),
    ],
)


@app.callback(
    Output("basin-controls", "style"),
    Input("mode", "value"),
)
def toggle_basin_controls(mode):
    """Only show the basin-width slider in one-peak mode."""
    if mode == "one-peak":
        return {"maxWidth": "700px", "marginBottom": "16px"}
    return {"display": "none"}


@app.callback(
    Output("basin-label", "children"),
    Input("basin-width", "value"),
)
def basin_label(value):
    return f"Basin width b = {value:g}  (held constant across dim = 3, 4, 10)"


@app.callback(
    Output("bump-controls", "style"),
    Input("mode", "value"),
)
def toggle_bump_controls(mode):
    """Only show the bump-width slider in gaussian-bump mode."""
    if mode == "gaussian-bump":
        return {"maxWidth": "700px", "marginBottom": "16px"}
    return {"display": "none"}


@app.callback(
    Output("bump-label", "children"),
    Input("bump-width", "value"),
)
def bump_label(value):
    return f"Bump width w = {value:g}  (held constant across dim = 3, 4, 10)"


@app.callback(
    Output("histogram", "figure"),
    Input("mode", "value"),
    Input("log", "value"),
    Input("basin-width", "value"),
    Input("bump-width", "value"),
)
def update_histogram(mode, log, basin_width, bump_width):
    return build_figure(mode, "log" in (log or []), float(basin_width),
                        float(bump_width))


if __name__ == "__main__":
    app.run(debug=True)
