"""Analyze the objective-value distribution of the Ackley objective across
different simplex dimensionalities, alongside the campaign1a RF surrogate.

For each dimensionality in ``DIMENSIONS`` we draw a large uniform sample from the
probability simplex, evaluate the analytic Ackley objective (synthetic_data/ackley.py)
at every point, and histogram the resulting objective values. One extra panel does
the same for the Random-Forest surrogate that interactive_testing/interactive_test_zombi.py
trains on campaign1a.csv (3-component composition -> Objective), so its landscape can
be compared on equal footing.

Reading the plot as a basin-volume diagnostic: the objective is highest at the
optima and falls off away from them, so the histogram is the distribution of "how
good" a uniformly random point is. Mass piled up at the low end with only a thin
tail reaching the top means the near-optimal basins occupy a tiny fraction of the
simplex volume -- and for Ackley that fraction shrinks as the dimensionality grows
(the curse of dimensionality, made visible).

This is an interactive Dash app:
  * a "noise" switch toggles the background simplex noise for every Ackley panel;
  * a "log y-axis" switch toggles the histogram count scale;
  * per-dimension sliders set the basin width and noise amplitude (the values that
    otherwise come from ``BASIN_WIDTH_BY_DIM`` / ``NOISE_AMP_BY_DIM`` in ackley.py);
  * "Reset" restores the ackley.py defaults.
The RF panel does not depend on these controls and is computed once.

Nothing is written to disk.

Usage:
    python analyze_basin_vol.py            # serve at http://127.0.0.1:8050
    python analyze_basin_vol.py --port 8060
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from dash import Dash, dcc, html, Input, Output, State, ctx, ALL

HERE = Path(__file__).resolve().parent
# Make the repo root importable so this runs from any working directory.
sys.path.insert(0, str(HERE.parent))
from synthetic_data.ackley import (  # noqa: E402
    Ackley,
    BASIN_WIDTH_BY_DIM,
    NOISE_AMP_BY_DIM,
    load_config,
    scaled_n_optima,
)

# ── Configuration ─────────────────────────────────────────────────────────────
DIMENSIONS = [3, 4, 10]   # simplex dimensionalities to examine (one panel each)
N_SAMPLES = 500_000       # uniform simplex samples used to estimate the distribution
N_BINS = 80               # histogram bins per panel
SEED = 0                  # RNG seed for reproducible sampling

INCLUDE_RF = True         # add a panel for the campaign1a RF surrogate (dim = 3)
# RF surrogate settings -- mirror interactive_testing/interactive_test_zombi.py.
RF_CSV = HERE.parent / "interactive_testing" / "campaign1a.csv"
RF_COMPOSITION_COLS = ["FAPbI3", "MAPbI3", "MAPbBr3"]
RF_OBJECTIVE_COL = "Objective"
RF_N_ESTIMATORS = 500
RF_RANDOM_STATE = 42

# Slider ranges for the interactive controls.
BASIN_WIDTH_RANGE = (1.0, 150.0)
NOISE_AMP_RANGE = (0.0, 600.0)
N_OPTIMA_RANGE = (1, 50)
N_SAMPLES_RANGE = (10_000, 1_000_000)
N_SAMPLES_STEP = 10_000
NOISE_FREQ_RANGE = (0.5, 30.0)
NOISE_FREQ_STEP = 0.5

ACKLEY_COLOR = "steelblue"
RF_COLOR = "indianred"


def sample_simplex(dim: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` points uniformly from the ``dim``-element probability simplex."""
    return rng.dirichlet(np.ones(dim), size=n)


def default_basin_width(dim: int) -> float:
    """The ackley.py default basin width for ``dim`` (config fallback included)."""
    return float(BASIN_WIDTH_BY_DIM.get(dim, load_config()["basin_width"]))


def default_noise_amp(dim: int) -> float:
    """The ackley.py default noise amplitude for ``dim`` (config fallback included)."""
    return float(NOISE_AMP_BY_DIM.get(dim, load_config()["noise_amp"]))


def default_n_optima(dim: int) -> int:
    """The ackley.py default peak count for ``dim`` (scaled from the d=3 baseline)."""
    return scaled_n_optima(int(load_config()["n_optima"]), dim)


def default_noise_freq() -> float:
    return float(load_config()["noise_freq"])


def load_rf_surrogate() -> RandomForestRegressor:
    """Train the campaign1a Random-Forest surrogate (same recipe as the
    interactive ZoMBI-Hop test: 3 composition columns row-normalized to the
    simplex -> Objective, 500 trees, fixed seed)."""
    df = pd.read_csv(RF_CSV).dropna(subset=RF_COMPOSITION_COLS + [RF_OBJECTIVE_COL])
    X = df[RF_COMPOSITION_COLS].to_numpy(dtype=float)
    X = X / np.where(X.sum(axis=1, keepdims=True) == 0, 1.0, X.sum(axis=1, keepdims=True))
    y = df[RF_OBJECTIVE_COL].to_numpy(dtype=float)
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, n_jobs=-1, random_state=RF_RANDOM_STATE
    )
    rf.fit(X, y)
    return rf


def ackley_objective(dim: int, samples: np.ndarray, *,
                     basin_width: float, noise_amp: float, noise_freq: float,
                     n_optima: int, noise_on: bool,
                     ackley_on: bool = True) -> np.ndarray:
    """Evaluate the "realistic" Ackley objective at ``samples`` for one panel.

    ``noise_amp`` is forced to 0 when ``noise_on`` is False; the Ackley peaks
    are dropped entirely when ``ackley_on`` is False (noise-only mode).
    """
    fn = Ackley(
        "realistic",
        dim=dim,
        basin_width=basin_width,
        noise_amp=(noise_amp if noise_on else 0.0),
        noise_freq=noise_freq,
        n_optima=n_optima,
    )
    if ackley_on:
        return fn.predict(samples)
    raw = fn._noise_raw(samples)
    span = raw.max() - raw.min()
    if span < 1e-12:
        return np.full(raw.shape, 0.75)
    return 0.5 + 0.5 * (raw - raw.min()) / span


# ── Pre-computed, control-independent data (built once at import) ─────────────
_MAX_SAMPLES = N_SAMPLES_RANGE[1]
_rng = np.random.default_rng(SEED)
SAMPLES_BY_DIM = {dim: sample_simplex(dim, _MAX_SAMPLES, _rng) for dim in DIMENSIONS}
RF_Y = None
if INCLUDE_RF:
    _rf = load_rf_surrogate()
    _rf_samples = sample_simplex(len(RF_COMPOSITION_COLS), _MAX_SAMPLES, _rng)
    RF_Y = _rf.predict(_rf_samples)


def _add_histogram(fig, col, y, title, color):
    """Add one histogram panel (column ``col``, 1-indexed) to the subplot grid."""
    bin_size = 0.5 / N_BINS  # x range is pinned to [0.5, 1.0]
    fig.add_trace(
        go.Histogram(
            x=y,
            xbins=dict(start=0.5, end=1.0, size=bin_size),
            marker=dict(color=color, line=dict(color="white", width=0.3)),
            showlegend=False,
        ),
        row=1, col=col,
    )
    best = float(np.max(y))
    fig.add_vline(
        x=best, line=dict(color="crimson", dash="dash", width=1.2),
        row=1, col=col,
    )
    suffix = "" if col == 1 else str(col)  # plotly axis ids: x, x2, x3, ...
    fig.add_annotation(
        xref=f"x{suffix} domain", yref=f"y{suffix} domain",
        x=0.97, y=0.97, xanchor="right", yanchor="top",
        text=f"best sampled<br>{best:.2f}",
        showarrow=False, font=dict(color="crimson", size=11),
    )
    fig.update_xaxes(title_text="objective value", range=[0.5, 1.0], row=1, col=col)
    # Subplot titles are set via make_subplots; annotate per-panel here instead.
    fig.layout.annotations[col - 1].text = title


def build_figure(basin_widths, noise_amps, n_optima_map, noise_on, log_y, n_samples,
                 ackley_on=True, noise_freqs=None):
    """Build the full multi-panel histogram figure for the current controls."""
    n_panels = len(DIMENSIONS) + (1 if INCLUDE_RF else 0)
    tags = []
    if ackley_on:
        tags.append("peaks on")
    else:
        tags.append("peaks off")
    tags.append("noise on" if noise_on else "noise off")
    tag_str = ", ".join(tags)
    titles = [f"Ackley 'realistic'<br>dim = {d} ({tag_str})" for d in DIMENSIONS]
    if INCLUDE_RF:
        titles.append(f"RF surrogate<br>(campaign1a, dim = {len(RF_COMPOSITION_COLS)})")

    fig = make_subplots(
        rows=1, cols=n_panels, shared_yaxes=True,
        subplot_titles=titles, horizontal_spacing=0.04,
    )

    for i, dim in enumerate(DIMENSIONS):
        _nf = noise_freqs[dim] if noise_freqs and dim in noise_freqs else default_noise_freq()
        y = ackley_objective(
            dim, SAMPLES_BY_DIM[dim][:n_samples],
            basin_width=basin_widths[dim], noise_amp=noise_amps[dim],
            noise_freq=_nf,
            n_optima=n_optima_map[dim], noise_on=noise_on,
            ackley_on=ackley_on,
        )
        _add_histogram(fig, i + 1, y, titles[i], ACKLEY_COLOR)

    if INCLUDE_RF:
        _add_histogram(fig, n_panels, RF_Y[:n_samples], titles[-1], RF_COLOR)

    fig.update_yaxes(type=("log" if log_y else "linear"))
    fig.update_yaxes(title_text="count", row=1, col=1)
    fig.update_layout(
        title_text=(
            f"Objective distribution over the simplex "
            f"(Ackley 'realistic' vs. campaign1a RF, {n_samples:,} uniform samples each)"
        ),
        bargap=0.02,
        height=560,
        margin=dict(t=110, b=60),
        template="plotly_white",
    )
    return fig


# ── Dash app ──────────────────────────────────────────────────────────────────
app = Dash(__name__)
app.title = "Ackley basin-volume explorer"

_switch_style = {"display": "inline-block", "marginRight": "32px"}


def _dim_column(dim):
    col_style = {"padding": "0 1%"}
    return html.Div(
        [
            html.H4(f"d = {dim}", style={"marginBottom": "4px", "fontSize": "14px"}),
            html.Label("basin_width", style={"fontSize": "13px"}),
            dcc.Slider(
                id={"type": "basin", "dim": dim},
                min=BASIN_WIDTH_RANGE[0], max=BASIN_WIDTH_RANGE[1],
                value=default_basin_width(dim),
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="mouseup",
            ),
            html.Label("noise_amp", style={"fontSize": "13px"}),
            dcc.Slider(
                id={"type": "noise", "dim": dim},
                min=NOISE_AMP_RANGE[0], max=NOISE_AMP_RANGE[1],
                value=default_noise_amp(dim),
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="mouseup",
            ),
            html.Label("n_optima", style={"fontSize": "13px"}),
            dcc.Slider(
                id={"type": "optima", "dim": dim},
                min=N_OPTIMA_RANGE[0], max=N_OPTIMA_RANGE[1], step=1,
                value=default_n_optima(dim),
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="mouseup",
            ),
            html.Label("noise_freq", style={"fontSize": "13px"}),
            dcc.Slider(
                id={"type": "freq", "dim": dim},
                min=NOISE_FREQ_RANGE[0], max=NOISE_FREQ_RANGE[1],
                step=NOISE_FREQ_STEP,
                value=default_noise_freq(),
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="mouseup",
            ),
        ],
        style=col_style,
    )


_n_dim_cols = len(DIMENSIONS)
_col_pct = f"{100 // _n_dim_cols}%"

app.layout = html.Div(
    [
        html.H2("Ackley basin-volume explorer", style={"marginBottom": "4px"}),
        html.Div(
            [
                dcc.Checklist(
                    id="ackley-toggle",
                    options=[{"label": " Ackley peaks", "value": "on"}],
                    value=["on"], style=_switch_style,
                ),
                dcc.Checklist(
                    id="noise-toggle",
                    options=[{"label": " noise", "value": "on"}],
                    value=["on"], style=_switch_style,
                ),
                dcc.Checklist(
                    id="logy-toggle",
                    options=[{"label": " log y-axis", "value": "on"}],
                    value=[], style=_switch_style,
                ),
                html.Button("Reset to defaults", id="reset-btn", n_clicks=0),
            ],
            style={"margin": "8px 0 12px 0"},
        ),
        html.Div(
            [
                html.Label("n_samples", style={"fontSize": "13px"}),
                dcc.Slider(
                    id="n-samples-slider",
                    min=N_SAMPLES_RANGE[0], max=N_SAMPLES_RANGE[1],
                    step=N_SAMPLES_STEP,
                    value=N_SAMPLES,
                    marks={v: f"{v // 1000}k" for v in range(
                        N_SAMPLES_RANGE[0], N_SAMPLES_RANGE[1] + 1, 100_000)},
                    tooltip={"placement": "bottom", "always_visible": False},
                    updatemode="mouseup",
                ),
            ],
            style={"marginBottom": "12px"},
        ),
        html.Div(
            [html.Div(_dim_column(dim),
                       style={"width": _col_pct, "display": "inline-block",
                              "verticalAlign": "top"})
             for dim in DIMENSIONS],
            style={"marginBottom": "12px"},
        ),
        dcc.Loading(dcc.Graph(id="hist-graph"), type="default"),
    ],
    style={"maxWidth": "1500px", "margin": "0 auto", "fontFamily": "sans-serif",
           "padding": "16px"},
)


@app.callback(
    Output("hist-graph", "figure"),
    Input("ackley-toggle", "value"),
    Input("noise-toggle", "value"),
    Input("logy-toggle", "value"),
    Input("n-samples-slider", "value"),
    Input({"type": "basin", "dim": ALL}, "value"),
    Input({"type": "noise", "dim": ALL}, "value"),
    Input({"type": "optima", "dim": ALL}, "value"),
    Input({"type": "freq", "dim": ALL}, "value"),
    State({"type": "basin", "dim": ALL}, "id"),
    State({"type": "noise", "dim": ALL}, "id"),
    State({"type": "optima", "dim": ALL}, "id"),
    State({"type": "freq", "dim": ALL}, "id"),
)
def _update_figure(ackley_value, noise_value, logy_value, n_samples,
                   basin_vals, noise_vals, optima_vals, freq_vals,
                   basin_ids, noise_ids, optima_ids, freq_ids):
    basin_widths = {bid["dim"]: v for bid, v in zip(basin_ids, basin_vals)}
    noise_amps = {nid["dim"]: v for nid, v in zip(noise_ids, noise_vals)}
    n_optima = {oid["dim"]: int(v) for oid, v in zip(optima_ids, optima_vals)}
    noise_freqs = {fid["dim"]: float(v) for fid, v in zip(freq_ids, freq_vals)}
    ackley_on = "on" in (ackley_value or [])
    noise_on = "on" in (noise_value or [])
    log_y = "on" in (logy_value or [])
    return build_figure(basin_widths, noise_amps, n_optima, noise_on, log_y,
                        int(n_samples), ackley_on, noise_freqs)


@app.callback(
    Output({"type": "basin", "dim": ALL}, "value"),
    Output({"type": "noise", "dim": ALL}, "value"),
    Output({"type": "optima", "dim": ALL}, "value"),
    Output({"type": "freq", "dim": ALL}, "value"),
    Output("n-samples-slider", "value"),
    Input("reset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _reset_sliders(_n_clicks):
    return (
        [default_basin_width(dim) for dim in DIMENSIONS],
        [default_noise_amp(dim) for dim in DIMENSIONS],
        [default_n_optima(dim) for dim in DIMENSIONS],
        [default_noise_freq() for _ in DIMENSIONS],
        N_SAMPLES,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8050, help="port to serve on")
    parser.add_argument("--host", default="127.0.0.1", help="host/interface to bind")
    parser.add_argument("--debug", action="store_true", help="run Dash in debug mode")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
