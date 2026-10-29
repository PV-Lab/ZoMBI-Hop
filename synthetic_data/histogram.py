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

``showdowns``
    One panel per optimize/showdown.py campaign in ``SHOWDOWN_DIRS`` (or
    ``--showdowns``), with the landscape rebuilt from its showdown_manifest.json
    exactly as evaluate.py ran it: the RF / GP fit to a run, or the needles /
    ensemble landscapes at the campaign's indices, seed and overrides (several
    landscapes are pooled per panel). The landscapes sit on different scales, so the
    "normalize" checkbox maps each panel to 0 = sample min, 1 = best true optimum.

Reading the plot as a basin-volume diagnostic: the objective is highest at the
optima and falls off away from them, so the histogram is the distribution of "how
good" a uniformly random point is.  Mass piled up at the low end with only a thin
tail reaching the top means the near-optimal basins occupy a tiny fraction of the
simplex volume.

Usage:
    python histogram.py            # launches the Dash app at http://127.0.0.1:8050

    # Static figure, no Dash server (fits an RF, so run it in an sbatch):
    python synthetic_data/histogram.py --mode showdowns --log --normalize \\
        --n-samples 200000 --save landscape_hist.png
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

# Showdowns mode: the landscapes optimize/showdown.py campaigns were benchmarked on,
# rebuilt from each showdown_manifest.json. Override with --showdowns on the CLI.
SHOWDOWN_DIRS = [
    "optimize/runs/showdown_6d_dup1c_lin_axial_run69af_rf500_reps3",
    "optimize/runs/showdown_6d_dup1c_needles_bw22_opt10",
    "optimize/runs/showdown_6d_dup1c_needles_bw6_opt10",
    "optimize/runs/showdown_6d_dup1c_needles_bw10_opt10",
    "optimize/runs/showdown_6d_dup1c_mix_rf69af_needles_rf_unit_v2",
]
SHOWDOWN_COLORS = ["indianred", "steelblue", "seagreen", "darkorange",
                   "mediumpurple", "goldenrod", "teal", "deeppink", "slategray"]


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


def _surrogate_landscape(dataset: str, gp_run, *, rf_n_estimators=None,
                         gp_length_scale=None) -> tuple[str, int, list, float]:
    """``(label, dim, [predict], peak)`` of a runrf / rungp landscape.

    ``gp_run`` is a run dir or a points ``.npz`` (optimize/scripts/needles_rf_points.py),
    which the label then describes by the landscape its points were drawn from.
    """
    from warm_start.warm_gp_landscape import runrf_objective, rungp_objective
    gp_run = Path(gp_run)
    if gp_run.suffix == ".npz":
        z = np.load(gp_run)
        src = (f"{len(z['Y'])} uniform pts of<br>needles "
               f"(n_optima = {int(z['n_optima'])}, basin_width = {float(z['basin_width']):g}"
               + (", unit_range" if "unit_range" in z and bool(z["unit_range"]) else "") + ")")
    else:
        src = gp_run.name
    if dataset == "runrf":
        o = runrf_objective(gp_run, n_estimators=int(rf_n_estimators))
        label = f"RF of {src}<br>({o['n_estimators']} trees, dim = {o['dim']})"
    else:
        o = rungp_objective(gp_run, length_scale=float(gp_length_scale))
        label = f"GP of {src}<br>(length scale {o['length_scale']:g}, dim = {o['dim']})"
    return label, int(o["dim"]), [o["predict"]], float(max(o["peak_values"]))


def _synthetic_landscape(dataset: str, dim: int, indices, seed: int,
                         overrides: dict) -> tuple[str, int, list, float]:
    """``(label, dim, [predict per landscape], peak)`` of needles / ensemble draws."""
    import evaluate
    fns = []
    for idx in indices:
        if dataset == "needles":
            cfg = evaluate.needles_config(dim, idx, seed, overrides)
        else:
            cfg = random_ensemble_config(dim, idx, seed=seed)
            if overrides:
                cfg = evaluate.apply_ensemble_overrides(cfg, overrides)
        fns.append(Ensemble(**cfg))
    peak = max(v for fn in fns for _, v in fn.known_maxima)
    # Two knobs per title line, so a long override list does not run into the
    # neighbouring panel's title.
    items = [f"{k} = {v}" for k, v in overrides.items()]
    knobs = ",<br>".join(", ".join(items[i:i + 2]) for i in range(0, len(items), 2))
    label = (f"{dataset} ({knobs})<br>{len(fns)} landscape(s), dim = {dim}" if knobs
             else f"{dataset}<br>{len(fns)} landscape(s), dim = {dim}")
    return label, dim, [fn.predict for fn in fns], peak


def showdown_landscapes(showdown_dir: str) -> list[tuple[str, int, list, float]]:
    """``[(label, dim, [batch predict per landscape], peak value)]`` for one showdown.

    Built exactly as optimize/evaluate.py builds the campaign's runs — the same
    needles_config / random_ensemble_config indices, seed and overrides, or the same
    run-fitted RF / GP — so the histogram is of the objectives that were benchmarked.
    The peak is the landscape's best true optimum, used by the normalized view. A
    showdown has one entry (its landscapes pooled), except a mixed one
    (showdown.py --mix), which has one per landscape.
    """
    import json
    sys.path.insert(0, str(HERE.parent / "optimize"))
    path = Path(showdown_dir)
    if not path.is_absolute():
        path = HERE.parent / path
    m = json.loads((path / "showdown_manifest.json").read_text())
    dataset, dim, seed = m.get("dataset") or "ensemble", int(m["dim"]), int(m["landscape_seed"])
    if dataset == "mixed":
        return [_surrogate_landscape(sp["dataset"], sp["gp_run"],
                                     rf_n_estimators=sp.get("rf_n_estimators"),
                                     gp_length_scale=sp.get("gp_length_scale"))
                if sp["dataset"] in ("runrf", "rungp")
                else _synthetic_landscape(sp["dataset"], dim, [sp["index"]], seed,
                                          sp.get("overrides") or {})
                for sp in m["landscape_specs"]]
    if dataset in ("runrf", "rungp"):
        return [_surrogate_landscape(dataset, m["gp_run"],
                                     rf_n_estimators=m.get("rf_n_estimators"),
                                     gp_length_scale=m.get("gp_length_scale"))]
    return [_synthetic_landscape(dataset, dim, m["landscapes"], seed,
                                 m.get("ensemble_overrides") or {})]


def showdown_rows(showdown_dirs) -> list[int]:
    """Figure row of each showdown panel: plain showdowns share a row, and every mixed
    showdown starts its own, so its landscapes read side by side."""
    import json
    rows, row, prev_mixed = [], -1, True
    for d in showdown_dirs:
        path = Path(d) if Path(d).is_absolute() else HERE.parent / d
        m = json.loads((path / "showdown_manifest.json").read_text())
        mixed = m.get("dataset") == "mixed"
        if mixed or prev_mixed:
            row += 1
        rows += [row] * (len(m["landscape_specs"]) if mixed else 1)
        prev_mixed = mixed
    return rows


@lru_cache(maxsize=4)
def showdown_samples(showdown_dirs: tuple, n_samples: int) -> tuple:
    """``(label, y, weight, peak)`` per showdown; cached because the RF fit is slow.

    A showdown with several landscapes pools a uniform sample of each into one panel,
    weighted ``1 / n_landscapes`` as in :func:`ensemble_combined_panel`, so every panel
    reads as the count for ONE landscape.
    """
    rng = np.random.default_rng(SEED)
    out = []
    for d in showdown_dirs:
        for label, dim, predicts, peak in showdown_landscapes(d):
            y = np.concatenate([np.asarray(p(sample_simplex(dim, n_samples, rng)),
                                           dtype=float) for p in predicts])
            out.append((label, y, 1.0 / len(predicts), peak))
            print(f"  [hist] {label.replace('<br>', ' ')}: min {y.min():.3f}  "
                  f"median {np.median(y):.3f}  max {y.max():.3f}  peak {peak:.3f}")
    return tuple(out)


def showdown_panels(normalize: bool, showdown_dirs=None,
                    n_samples: int = N_SAMPLES) -> list:
    """One panel per showdown landscape.

    The landscapes live on different scales (needles: plain 0.75, peaks 1.0; the RF
    inherits run_69af's measured values), so raw panels each get their own x-range.
    ``normalize`` maps every panel to ``(y - min) / (peak - min)`` — 0 at the sample
    minimum, 1 at the best true optimum — so the panels share one [0, 1] axis and read
    directly as "how close to the peak is a random point".
    """
    panels = []
    for k, (label, y, weight, peak) in enumerate(
            showdown_samples(tuple(showdown_dirs or SHOWDOWN_DIRS), int(n_samples))):
        if normalize:
            y = (y - y.min()) / max(peak - y.min(), 1e-12)
        panels.append((label, y, SHOWDOWN_COLORS[k % len(SHOWDOWN_COLORS)], weight))
    return panels


def panel_range(mode: str, normalize: bool, y: np.ndarray) -> tuple:
    """The x-range a panel is histogrammed over."""
    if mode == "showdowns":
        return (0.0, 1.0) if normalize else (float(y.min()), float(y.max()))
    # one-peak / gaussian-bump are no longer min-max rescaled, so their values
    # span the full [0, 1] envelope rather than the [0.5, 1] predict() range.
    return (0.0, 1.0) if mode in ("one-peak", "gaussian-bump") else HIST_RANGE


def build_figure(mode: str, log: bool, basin_width: float,
                 bump_width: float, normalize: bool = False,
                 showdown_dirs=None, n_samples: int = N_SAMPLES) -> go.Figure:
    """Build the multi-panel histogram figure for the selected ``mode``."""
    rng = np.random.default_rng(SEED)

    if mode == "showdowns":
        panels = showdown_panels(normalize, showdown_dirs, n_samples)
        suptitle = ("Objective distribution of the benchmarked showdown landscapes"
                    + (" (normalized: 0 = sample min, 1 = best optimum)" if normalize
                       else ""))
    elif mode == "ensemble":
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

    # Grid position of each panel: one row, except showdowns with a mixed showdown.
    rows = (showdown_rows(showdown_dirs or SHOWDOWN_DIRS) if mode == "showdowns"
            else [0] * len(panels))
    cols = [sum(1 for r in rows[:i] if r == rows[i]) for i in range(len(rows))]
    n_rows, n_cols = max(rows) + 1, max(cols) + 1
    titles = [""] * (n_rows * n_cols)
    for (title, *_), r, c in zip(panels, rows, cols):
        titles[r * n_cols + c] = title
    fig = make_subplots(
        rows=n_rows, cols=n_cols, shared_yaxes="all",
        subplot_titles=titles, horizontal_spacing=0.04,
        vertical_spacing=0.18 if n_rows > 1 else 0.0,
    )

    for (title, y, color, weight), r, c in zip(panels, rows, cols):
        row, col = r + 1, c + 1
        hist_range = panel_range(mode, normalize, y)
        bins = np.linspace(hist_range[0], hist_range[1], N_BINS + 1)
        centers = 0.5 * (bins[:-1] + bins[1:])
        counts, _ = np.histogram(y, bins=bins, weights=np.full(y.shape, weight))
        fig.add_trace(
            go.Bar(x=centers, y=counts, width=(bins[1] - bins[0]),
                   marker_color=color, marker_line_color="white",
                   marker_line_width=0.3, showlegend=False),
            row=row, col=col,
        )
        fig.update_xaxes(range=list(hist_range), title_text="objective value",
                         row=row, col=col)

    fig.update_yaxes(type="log" if log else "linear")
    for r in range(1, n_rows + 1):
        fig.update_yaxes(title_text="count", row=r, col=1)

    fig.update_layout(
        title=suptitle, bargap=0, template="plotly_white",
        height=520 * n_rows, margin=dict(t=90, b=60),
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
                            {"label": "Showdown landscapes (RF run_69af + needles)", "value": "showdowns"},
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
                html.Div([
                    dcc.Checklist(
                        id="normalize",
                        options=[{"label": " normalize to peak (showdowns)",
                                  "value": "normalize"}],
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
    Input("normalize", "value"),
)
def update_histogram(mode, log, basin_width, bump_width, normalize):
    return build_figure(mode, "log" in (log or []), float(basin_width),
                        float(bump_width), normalize="normalize" in (normalize or []))


def save_png(path: str, mode: str, log: bool, normalize: bool,
             showdown_dirs=None, n_samples: int = N_SAMPLES) -> None:
    """Static matplotlib twin of :func:`build_figure` (plotly has no PNG export here)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if mode != "showdowns":
        raise SystemExit("--save *.png supports --mode showdowns only; use .html")
    panels = showdown_panels(normalize, showdown_dirs, n_samples)
    rows = showdown_rows(showdown_dirs or SHOWDOWN_DIRS)
    cols = [sum(1 for r in rows[:i] if r == rows[i]) for i in range(len(rows))]
    n_rows, n_cols = max(rows) + 1, max(cols) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.8 * n_rows),
                             sharey=True, squeeze=False)
    used = set(zip(rows, cols))
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) not in used:
                axes[r][c].set_visible(False)
    for (title, y, color, weight), r, c in zip(panels, rows, cols):
        ax = axes[r][c]
        lo, hi = panel_range(mode, normalize, y)
        ax.hist(y, bins=np.linspace(lo, hi, N_BINS + 1), weights=np.full(y.shape, weight),
                color=color, edgecolor="white", linewidth=0.3)
        ax.set_title(title.replace("<br>", "\n"), fontsize=10)
        ax.set_xlabel("objective value" + (" (normalized)" if normalize else ""))
        if log:
            ax.set_yscale("log")
    for r in range(n_rows):
        axes[r][0].set_ylabel("count per landscape")
    fig.suptitle("Objective distribution of the benchmarked showdown landscapes"
                 + (" — 0 = sample min, 1 = best optimum" if normalize else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save", default=None, metavar="PATH",
                    help="write the figure to PATH (.html, or .png for --mode showdowns) "
                         "instead of launching the Dash app — for running in an sbatch")
    ap.add_argument("--mode", default="showdowns",
                    choices=("ackley", "ensemble", "one-peak", "gaussian-bump", "showdowns"))
    ap.add_argument("--showdowns", default=None, metavar="DIR[,DIR...]",
                    help="showdown dirs for --mode showdowns (default: SHOWDOWN_DIRS)")
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES,
                    help="uniform simplex samples per landscape (default: %(default)s)")
    ap.add_argument("--log", action="store_true", help="log y-axis")
    ap.add_argument("--normalize", action="store_true",
                    help="showdowns: map each panel to [0, 1] (sample min -> best optimum)")
    a = ap.parse_args()
    if a.save is None:
        app.run(debug=True)
    else:
        dirs = [d.strip() for d in a.showdowns.split(",") if d.strip()] if a.showdowns else None
        if a.save.endswith(".png"):
            save_png(a.save, a.mode, a.log, a.normalize, dirs, a.n_samples)
        else:
            build_figure(a.mode, a.log, ONE_PEAK_B_DEFAULT, GAUSS_W_DEFAULT,
                         normalize=a.normalize, showdown_dirs=dirs,
                         n_samples=a.n_samples).write_html(a.save)
        print(f"  wrote {a.save}")
