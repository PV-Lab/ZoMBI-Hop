"""
visualization/plot_warm_start.py
================================
Interactive Dash app that shows *how informative a warm start is* by putting two
Gaussian-Process reconstructions of the same simplex side by side.

  * **Left — GP reconstruction of the full run.** A GP is fit on every measured
    ``(X, Y)`` point of a real ZoMBI-Hop run (``runs/run_7eb9`` for d=3,
    ``runs/run_9dfe`` for d=4) and evaluated over the simplex. This is treated as
    the ground-truth surface.
  * **Right — GP over only the warm-start points.** The line-constrained warm
    start (``warm_start.greedy_lines``) lays a handful of 24-point lines on the
    same simplex. Each warm-start composition is *sampled from the left GP* (its
    posterior mean), and a second GP — same length scale — is fit on those warm
    points alone. Comparing the two surfaces shows how much of the full surface a
    warm start recovers before any adaptive search begins.

Both GPs use the same fixed Matern length scale (chosen with the slider, default
0.05 — the smallest in ``plot_run.py``'s range), and both panels share one
viridis colour scale so the surfaces are directly comparable.

A dropdown switches between the d=3 run (ternary triangle) and the d=4 run
(3D tetrahedron); the diagram type follows the dimension automatically, reusing
``plot_run.build_figure``.

Usage
-----
  python visualization/plot_warm_start.py                 # launch the app
  python visualization/plot_warm_start.py --port 8052     # custom port
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

# ── project root on sys.path so `visualization` / `warm_start` imports resolve ──
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from visualization.plot_run import (  # noqa: E402
    TETRA_GRID_MAX_N, _color_limits, _resolve_run_dir, build_figure,
    load_run_source, simplex_grid,
)
from warm_start.warm_start import greedy_lines, n_lines  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────────
# Which real run stands in for each dimension.
RUN_FOR_DIM: dict[int, str] = {3: "run_7eb9", 4: "run_9dfe"}
# Default length scale = smallest in plot_run.py's GP length-scale slider (0.05).
DEFAULT_LENGTH_SCALE = 0.05
DEFAULT_SEED = 0
# Grid resolution for the background surface (d=4 is clamped internally by
# build_figure to keep the O(n^3) tetrahedron grid responsive).
DEFAULT_GRID_N = 120


# ── Gaussian-Process reconstruction we can sample at arbitrary points ──────────

def fit_gp(X: np.ndarray, Y: np.ndarray, length_scale: float):
    """Fit a fixed-length-scale Matern GP on (X, Y); return a mean-predictor.

    Mirrors ``plot_run.fit_gp_background``'s kernel exactly (ConstantKernel *
    Matern(nu=2.5, fixed length scale) + WhiteKernel), so the surface this
    predicts is the same one that appears as the left panel's background. Y is
    standardised for numerical stability and the predictor maps back to the
    original scale.

    Returns ``predict(P) -> (M,)`` giving the GP posterior *mean* at points ``P``
    — this is what "sample the reconstruction at the warm-start compositions"
    means here: read the reconstructed surface at each warm-start point.
    """
    y_mean = float(Y.mean())
    y_std = float(Y.std()) or 1.0
    y = (Y - y_mean) / y_std

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=length_scale, length_scale_bounds="fixed", nu=2.5)
        + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, normalize_y=False, n_restarts_optimizer=2, random_state=42
    )
    gp.fit(X, y)

    def predict(P: np.ndarray) -> np.ndarray:
        return gp.predict(P) * y_std + y_mean

    return predict


# ── caches: the run datasets and warm-start designs are dimension-only ─────────

_RUN_CACHE: dict[int, tuple] = {}
_WARM_CACHE: dict[tuple[int, int], np.ndarray] = {}


def load_run(dim: int) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], str]:
    """``(X, Y, labels, title)`` for the run that stands in for ``dim``."""
    if dim not in _RUN_CACHE:
        run = RUN_FOR_DIM[dim]
        _RUN_CACHE[dim] = load_run_source(_resolve_run_dir(run), None)
    return _RUN_CACHE[dim]


def warm_start_points(dim: int, seed: int) -> np.ndarray:
    """The (n, dim) warm-start compositions for ``dim`` (line-constrained design)."""
    key = (dim, seed)
    if key not in _WARM_CACHE:
        _, X = greedy_lines(n_lines(dim), dim, seed=seed)
        _WARM_CACHE[key] = X
    return _WARM_CACHE[key]


# ── the two figures ────────────────────────────────────────────────────────────

def build_comparison(dim: int, length_scale: float, seed: int, grid_n: int,
                     show_points: bool):
    """Build ``(left_fig, right_fig, info)`` for the current controls.

    Left: GP reconstruction of the full run. Right: GP over warm-start points
    whose values are sampled from the left GP. Both share one colour scale.
    """
    X, Y, labels, run_title = load_run(dim)

    # Warm-start compositions and their values, sampled from the left GP surface.
    X_warm = warm_start_points(dim, seed)
    predict = fit_gp(X, Y, length_scale)
    Y_warm = predict(X_warm)

    # One colour scale for both panels, derived exactly the way plot_run does:
    # the 10th/90th percentiles over the (left) GP reconstruction *surface* plus
    # its measured points. Evaluating the predictor we already fit over the same
    # simplex grid build_figure uses means no extra GP fit. d=4 uses the clamped
    # grid resolution that build_quaternary_figure renders at.
    eff_grid_n = min(grid_n, TETRA_GRID_MAX_N) if dim == 4 else grid_n
    left_grid_vals = predict(simplex_grid(eff_grid_n, dim))
    color_limits = _color_limits(left_grid_vals, Y)

    common = dict(
        grid_n=grid_n, n_estimators=1, background="gp",
        show_points=show_points, gp_length_scale=length_scale,
        value_name="Objective", color_limits=color_limits,
    )

    left = build_figure(
        X, Y, labels,
        title=f"Full run — GP reconstruction  (ℓ={length_scale:g})",
        **common,
    )
    right = build_figure(
        X_warm, Y_warm, labels,
        title=f"Warm start only — GP over {X_warm.shape[0]} pts  (ℓ={length_scale:g})",
        **common,
    )

    info = (
        f"run: {RUN_FOR_DIM[dim]}  ({run_title})\n"
        f"full run: {X.shape[0]} points\n"
        f"warm start: {n_lines(dim)} lines × 24 = {X_warm.shape[0]} points\n"
        f"length scale ℓ = {length_scale:g}   seed = {seed}\n"
        f"colour range: [{color_limits[0]:.3f}, {color_limits[1]:.3f}]"
    )
    return left, right, info


# ── Dash app ──────────────────────────────────────────────────────────────────

def build_app(grid_n: int = DEFAULT_GRID_N):
    from dash import Dash, Input, Output, dcc, html, no_update

    label_style = {"fontWeight": "600", "marginTop": "12px", "display": "block"}
    panel_style = {
        "width": "300px", "padding": "18px", "boxSizing": "border-box",
        "borderRight": "1px solid #e3e3e3", "background": "#fafafa",
        "height": "100vh", "overflowY": "auto",
    }

    app = Dash(__name__, update_title=None)
    app.title = "ZoMBI-Hop Warm-Start Comparison"

    app.layout = html.Div(
        style={"display": "flex", "fontFamily": "system-ui, sans-serif"},
        children=[
            html.Div(style=panel_style, children=[
                html.H3("Warm-start informativeness", style={"marginTop": 0}),
                html.Div(
                    "Left: GP on the full run. Right: GP on only the warm-start "
                    "points, sampled from the left surface.",
                    style={"fontSize": "13px", "color": "#666"},
                ),

                html.Label("Dimension", style=label_style),
                dcc.Dropdown(
                    id="dim",
                    options=[
                        {"label": "3d  (run_7eb9, ternary)", "value": 3},
                        {"label": "4d  (run_9dfe, tetrahedron)", "value": 4},
                    ],
                    value=3, clearable=False,
                ),

                html.Label("GP length scale ℓ", style=label_style),
                dcc.Slider(
                    id="length-scale", min=0.05, max=1.0, step=0.05,
                    value=DEFAULT_LENGTH_SCALE,
                    marks={0.05: "0.05", 0.3: "0.3", 0.6: "0.6", 1.0: "1.0"},
                ),

                html.Label("Warm-start seed", style=label_style),
                dcc.Slider(
                    id="seed", min=0, max=9, step=1, value=DEFAULT_SEED,
                    marks={i: str(i) for i in range(0, 10, 3)},
                ),

                html.Label("Grid resolution", style=label_style),
                dcc.Slider(
                    id="grid-n", min=40, max=240, step=20, value=grid_n,
                    marks={40: "40", 120: "120", 240: "240"},
                ),

                dcc.Checklist(
                    id="show-points",
                    options=[{"label": " Show points", "value": "show"}],
                    value=["show"],
                    style={"marginTop": "12px"},
                ),

                html.Div(id="status", style={
                    "marginTop": "18px", "fontSize": "13px", "color": "#666",
                    "whiteSpace": "pre-wrap",
                }),
            ]),

            html.Div(style={"flex": 1, "display": "flex"}, children=[
                dcc.Loading(html.Div(
                    dcc.Graph(id="left-graph", style={"height": "92vh"}),
                    style={"flex": 1},
                ), style={"flex": 1}),
                dcc.Loading(html.Div(
                    dcc.Graph(id="right-graph", style={"height": "92vh"}),
                    style={"flex": 1},
                ), style={"flex": 1}),
            ]),
        ],
    )

    @app.callback(
        Output("left-graph", "figure"),
        Output("right-graph", "figure"),
        Output("status", "children"),
        Input("dim", "value"),
        Input("length-scale", "value"),
        Input("seed", "value"),
        Input("grid-n", "value"),
        Input("show-points", "value"),
    )
    def _render(dim, length_scale, seed, gn, show_points):
        try:
            left, right, info = build_comparison(
                int(dim), float(length_scale), int(seed), int(gn),
                bool(show_points),
            )
            return left, right, info
        except Exception as e:  # surface load/fit errors in the UI
            return no_update, no_update, f"Error: {e}"

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side GP comparison of a full ZoMBI-Hop run vs. its "
                    "warm start, for d=3 and d=4."
    )
    parser.add_argument("--port", type=int, default=8052,
                        help="Port for the Dash app (default: 8052).")
    parser.add_argument("--grid-n", type=int, default=DEFAULT_GRID_N,
                        help="Background grid resolution (default: 120).")
    args = parser.parse_args()

    app = build_app(grid_n=args.grid_n)
    print(f"Dash app running at http://127.0.0.1:{args.port}")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
