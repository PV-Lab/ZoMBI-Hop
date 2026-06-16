"""
plot_oracles_interactive.py
===========================
Interactive Dash explorer for all 3D synthetic oracles (and the tunable
realistic Ackley from plot_3d.py).  Sliders update the ternary landscape and
an optional campaign-style sample preview in real time.

Usage
-----
  python synthetic_data/plot_oracles_interactive.py
  python synthetic_data/plot_3d.py          # shim → realistic Ackley preset

Open http://127.0.0.1:8050
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    from dash import Dash, Input, Output, State, ctx, dcc, html
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: dash (and plotly).\n"
        "  pip install dash plotly\n"
        "Then rerun: python synthetic_data/plot_oracles_interactive.py"
    ) from exc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from synthetic_data.interactive_oracle_app import (  # noqa: E402
    DISPLAY_LABELS,
    ORACLE_CHOICES,
    SLIDER_GROUPS,
    build_grid,
    build_tuned_oracle,
    evaluate_oracle,
    load_oracle_params,
    preview_campaign_samples,
    samples_figure,
    save_oracle_params,
    ternary_figure,
)

GRID_N = 120
FIG_W, FIG_H = 880, 780

HIDDEN = {"display": "none"}
VISIBLE = {"display": "block", "padding": "8px 10px"}

GROUP_STYLE = {g: HIDDEN for g in ("common", "realistic", "messy", "ackley", "gaussian", "planted", "rastrigin")}


def _slider_block(group_id: str, label: str, slider: dcc.Slider) -> html.Div:
    return html.Div(
        [html.Label(label, style={"fontWeight": 600}), slider],
        id=f"group-{group_id}",
        style=GROUP_STYLE[group_id],
    )


def _make_layout(initial_oracle: str) -> html.Div:
    params = load_oracle_params(initial_oracle)
    return html.Div([
        html.H2("Interactive Synthetic Oracle Explorer (Δ³)",
                style={"textAlign": "center", "marginBottom": "8px"}),
        html.P(
            "Choose an oracle, tune sliders, and watch the landscape + sample "
            "preview update. Save persists defaults (realistic Ackley → "
            "ackley/defaults.json; others → oracle_configs/).",
            style={"textAlign": "center", "color": "#444", "fontSize": "14px"},
        ),
        html.Div([
            html.Div([
                html.Label("Oracle"),
                dcc.Dropdown(
                    id="oracle-type",
                    options=[{"label": DISPLAY_LABELS[k], "value": k} for k in ORACLE_CHOICES],
                    value=initial_oracle,
                    clearable=False,
                ),
            ], style={"width": "42%", "display": "inline-block", "padding": "8px"}),
            html.Div([
                html.Label("Peak layout"),
                dcc.Dropdown(
                    id="layout",
                    options=[
                        {"label": "1 — 3 peaks (centroid + v0 + edge)", "value": "1"},
                        {"label": "2 — 5 peaks (+ v1, v2)", "value": "2"},
                        {"label": "3 — 7 peaks (needs d≥5 for full set)", "value": "3"},
                    ],
                    value=str(params.get("layout", "2")),
                    clearable=False,
                ),
            ], style={"width": "28%", "display": "inline-block", "padding": "8px"}),
            html.Div([
                dcc.Checklist(
                    id="show-samples",
                    options=[{"label": " Show campaign-style sample preview", "value": "on"}],
                    value=["on"],
                    style={"marginTop": "28px"},
                ),
            ], style={"width": "28%", "display": "inline-block", "padding": "8px"}),
        ], style={"width": "92%", "margin": "0 auto"}),

        html.Div([
            _slider_block("common", "RNG seed", dcc.Slider(
                id="seed", min=0, max=99, step=1, value=int(params.get("seed", 42)),
                tooltip={"placement": "bottom", "always_visible": True},
            )),
            _slider_block("realistic", "Number of optima", dcc.Slider(
                id="n-optima", min=1, max=30, step=1,
                value=int(params.get("n_optima", 10)),
                marks={i: str(i) for i in range(1, 31, 5)},
                tooltip={"placement": "bottom", "always_visible": True},
            )),
            _slider_block("realistic", "Basin width (b)", dcc.Slider(
                id="basin-width", min=1, max=200, step=1,
                value=float(params.get("basin_width", 50.0)),
                marks={i: str(i) for i in range(0, 201, 25)},
                tooltip={"placement": "bottom", "always_visible": True},
            )),
            _slider_block("realistic", "Noise frequency", dcc.Slider(
                id="noise-freq", min=0, max=40, step=0.5,
                value=float(params.get("noise_freq", 8.0)),
                marks={i: str(i) for i in range(0, 41, 5)},
                tooltip={"placement": "bottom", "always_visible": True},
            )),
            _slider_block("realistic", "Noise amplitude", dcc.Slider(
                id="noise-amp", min=0, max=2000, step=20,
                value=float(params.get("noise_amp", 5.0)),
                marks={i: str(i) for i in range(0, 2001, 200)},
                tooltip={"placement": "bottom", "always_visible": True},
            )),
            _slider_block("messy", "Signed micro-bumps", dcc.Slider(
                id="n-micro", min=0, max=300, step=5,
                value=int(params.get("n_micro", 150)),
                marks={i: str(i) for i in range(0, 301, 50)},
                tooltip={"placement": "bottom", "always_visible": True},
            )),
            _slider_block("messy", "ILR ripples", dcc.Slider(
                id="n-ripples", min=0, max=80, step=2,
                value=int(params.get("n_ripples", 30)),
                marks={i: str(i) for i in range(0, 81, 20)},
                tooltip={"placement": "bottom", "always_visible": True},
            )),
            _slider_block("messy", "Major bump σ", dcc.Slider(
                id="major-sigma-messy", min=0.02, max=0.15, step=0.005,
                value=float(params.get("major_sigma", 0.055)),
                tooltip={"placement": "bottom", "always_visible": True},
            )),
            _slider_block("ackley", "Ackley peak width b", dcc.Slider(
                id="ackley-b", min=0.2, max=3.0, step=0.1,
                value=float(params.get("ackley_b", 1.2)),
                marks={0.2: "0.2", 1.2: "1.2", 3.0: "3"},
                tooltip={"placement": "bottom", "always_visible": True},
            )),
            _slider_block("gaussian", "Gaussian σ", dcc.Slider(
                id="gaussian-sigma", min=0.03, max=0.15, step=0.005,
                value=float(params.get("sigma", 0.07)),
                tooltip={"placement": "bottom", "always_visible": True},
            )),
            _slider_block("planted", "Micro-bumps", dcc.Slider(
                id="n-micro-planted", min=0, max=150, step=5,
                value=int(params.get("n_micro", 40)),
                marks={i: str(i) for i in range(0, 151, 25)},
                tooltip={"placement": "bottom", "always_visible": True},
            )),
            _slider_block("planted", "Major bump σ", dcc.Slider(
                id="major-sigma-planted", min=0.03, max=0.15, step=0.005,
                value=float(params.get("major_sigma", 0.09)),
                tooltip={"placement": "bottom", "always_visible": True},
            )),
            _slider_block("planted", "Signed micro weights", dcc.Dropdown(
                id="signed-micro",
                options=[
                    {"label": "Unsigned (positive micro only)", "value": "0"},
                    {"label": "Signed (± micro, messier)", "value": "1"},
                ],
                value="1" if params.get("signed_micro") else "0",
                clearable=False,
            )),
            _slider_block("rastrigin", "Rastrigin amplitude", dcc.Slider(
                id="rastrigin-amp", min=1.0, max=30.0, step=0.5,
                value=float(params.get("amplitude", 10.0)),
                marks={1: "1", 10: "10", 30: "30"},
                tooltip={"placement": "bottom", "always_visible": True},
            )),
            _slider_block("common", "Preview samples", dcc.Slider(
                id="n-preview", min=100, max=700, step=50,
                value=300,
                marks={100: "100", 300: "300", 700: "700"},
                tooltip={"placement": "bottom", "always_visible": True},
            )),
            html.Div([
                html.Button("Save as Default", id="save-btn", n_clicks=0,
                            style={"padding": "8px 24px", "marginRight": "12px"}),
                html.Span(id="save-status"),
            ], style={"textAlign": "center", "padding": "12px"}),
        ], style={"width": "70%", "margin": "0 auto"}),

        html.Div([
            dcc.Graph(id="oracle-plot", style={"width": "49%", "display": "inline-block"}),
            dcc.Graph(id="samples-plot", style={"width": "49%", "display": "inline-block"}),
        ], style={"width": "98%", "margin": "0 auto"}),
    ])


def create_app(initial_oracle: str = "messy") -> Dash:
    app = Dash(__name__)
    app.layout = _make_layout(initial_oracle)
    _register_callbacks(app)
    return app


def _params_from_inputs(
    oracle: str,
    layout, seed,
    n_optima, basin_width, noise_freq, noise_amp,
    n_micro, n_ripples, major_sigma_messy,
    ackley_b, gaussian_sigma,
    n_micro_planted, major_sigma_planted, signed_micro,
    rastrigin_amp,
) -> dict:
    p: dict = {"layout": str(layout), "seed": int(seed)}
    if oracle == "realistic_ackley":
        p.update({
            "n_optima": int(n_optima),
            "basin_width": float(basin_width),
            "noise_freq": float(noise_freq),
            "noise_amp": float(noise_amp),
        })
    elif oracle == "messy":
        p.update({
            "n_micro": int(n_micro),
            "n_ripples": int(n_ripples),
            "major_sigma": float(major_sigma_messy),
        })
    elif oracle == "ackley":
        p["ackley_b"] = float(ackley_b)
    elif oracle == "gaussian":
        p["sigma"] = float(gaussian_sigma)
    elif oracle == "planted_bumps":
        p.update({
            "n_micro": int(n_micro_planted),
            "major_sigma": float(major_sigma_planted),
            "signed_micro": signed_micro == "1",
        })
    elif oracle == "rastrigin_ilr":
        p["amplitude"] = float(rastrigin_amp)
    return p


def _register_callbacks(app: Dash) -> None:
    group_ids = [f"group-{g}" for g in GROUP_STYLE]

    @app.callback(
        Output("layout", "value"),
        Output("seed", "value"),
        Output("n-optima", "value"),
        Output("basin-width", "value"),
        Output("noise-freq", "value"),
        Output("noise-amp", "value"),
        Output("n-micro", "value"),
        Output("n-ripples", "value"),
        Output("major-sigma-messy", "value"),
        Output("ackley-b", "value"),
        Output("gaussian-sigma", "value"),
        Output("n-micro-planted", "value"),
        Output("major-sigma-planted", "value"),
        Output("signed-micro", "value"),
        Output("rastrigin-amp", "value"),
        Input("oracle-type", "value"),
    )
    def load_sliders_for_oracle(oracle):
        p = load_oracle_params(oracle)
        return (
            str(p.get("layout", "2")),
            int(p.get("seed", 42)),
            int(p.get("n_optima", 10)),
            float(p.get("basin_width", 50.0)),
            float(p.get("noise_freq", 8.0)),
            float(p.get("noise_amp", 5.0)),
            int(p.get("n_micro", 150)),
            int(p.get("n_ripples", 30)),
            float(p.get("major_sigma", 0.055)),
            float(p.get("ackley_b", 1.2)),
            float(p.get("sigma", 0.07)),
            int(p.get("n_micro", 40)),
            float(p.get("major_sigma", 0.09)),
            "1" if p.get("signed_micro") else "0",
            float(p.get("amplitude", 10.0)),
        )

    @app.callback(
        [Output(gid, "style") for gid in group_ids],
        Input("oracle-type", "value"),
    )
    def toggle_slider_groups(oracle):
        active = set(SLIDER_GROUPS.get(oracle, ("common",)))
        return [
            VISIBLE if g.replace("group-", "") in active else HIDDEN
            for g in group_ids
        ]

    @app.callback(
        Output("save-status", "children"),
        Input("save-btn", "n_clicks"),
        State("oracle-type", "value"),
        State("layout", "value"),
        State("seed", "value"),
        State("n-optima", "value"),
        State("basin-width", "value"),
        State("noise-freq", "value"),
        State("noise-amp", "value"),
        State("n-micro", "value"),
        State("n-ripples", "value"),
        State("major-sigma-messy", "value"),
        State("ackley-b", "value"),
        State("gaussian-sigma", "value"),
        State("n-micro-planted", "value"),
        State("major-sigma-planted", "value"),
        State("signed-micro", "value"),
        State("rastrigin-amp", "value"),
        prevent_initial_call=True,
    )
    def save_defaults(n_clicks, oracle, *args):
        params = _params_from_inputs(oracle, *args)
        save_oracle_params(oracle, params)
        return "Saved!"

    @app.callback(
        Output("oracle-plot", "figure"),
        Output("samples-plot", "figure"),
        Input("oracle-type", "value"),
        Input("layout", "value"),
        Input("seed", "value"),
        Input("n-optima", "value"),
        Input("basin-width", "value"),
        Input("noise-freq", "value"),
        Input("noise-amp", "value"),
        Input("n-micro", "value"),
        Input("n-ripples", "value"),
        Input("major-sigma-messy", "value"),
        Input("ackley-b", "value"),
        Input("gaussian-sigma", "value"),
        Input("n-micro-planted", "value"),
        Input("major-sigma-planted", "value"),
        Input("signed-micro", "value"),
        Input("rastrigin-amp", "value"),
        Input("show-samples", "value"),
        Input("n-preview", "value"),
    )
    def update_plots(oracle, *args):
        show_samples, n_preview = args[-2:]
        slider_args = args[:-2]
        params = _params_from_inputs(oracle, *slider_args)
        realistic = oracle == "realistic_ackley"

        fn, optima, suffix = build_tuned_oracle(oracle, params)
        grid = build_grid(GRID_N)
        vals = evaluate_oracle(fn, grid, realistic_ackley=realistic)
        title = f"{DISPLAY_LABELS[oracle]} — {suffix}"
        fig_oracle = ternary_figure(
            grid, vals, optima, title=title, width=FIG_W, height=FIG_H,
        )

        if show_samples and "on" in (show_samples or []):
            X, y = preview_campaign_samples(
                fn, optima,
                n_samples=int(n_preview),
                seed=int(params.get("seed", 42)),
                realistic_ackley=realistic,
            )
            fig_samples = samples_figure(
                X, y,
                title=f"Campaign-style preview (n={len(X)}, noise σ=0.07)",
                width=FIG_W, height=FIG_H,
            )
        else:
            fig_samples = samples_figure(
                np.zeros((0, 3)), np.zeros(0),
                title="Sample preview disabled",
                width=FIG_W, height=FIG_H,
            )

        return fig_oracle, fig_samples


def main(initial_oracle: str = "messy") -> None:
    app = create_app(initial_oracle=initial_oracle)
    print("Starting Dash app at http://127.0.0.1:8050")
    print(f"  Oracle preset: {initial_oracle}")
    app.run(debug=True)


if __name__ == "__main__":
    main()
