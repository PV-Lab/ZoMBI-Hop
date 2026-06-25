"""
visualization/discrepancy.py
=============================
Interactive Dash web app for inspecting the *discrepancy* between the line
ZoMBI-Hop **asked** the hardware to sample (the "expected" line) and the line
the hardware actually **returned** (the "actual" line).

Each LineBO line is a sequence of ``NUM_EXPERIMENTS`` (24) composition points.
For a chosen line the ternary shows:

  * the expected line  — 24 points drawn in **grey**;
  * the actual line     — 24 points coloured by their **objective value**
                          (viridis), mirroring how points are rendered in
                          ``interface/app.py``;
  * thin grey segments connecting each actual point to its expected twin.

A "Random line" button picks a random line.  The scrollable list on the right
shows every line, labelled by the order in which it was placed and ranked
*worst-first* by closeness (mean per-point Euclidean distance between the
actual and expected points).  That same metric is shown beneath the ternary.

Only d=3 (true ternary) runs are supported; loading a higher-d run raises a
clear error.

Usage
-----
  conda activate zombi-hop
  python visualization/discrepancy.py
  # then open the printed http://127.0.0.1:8050 in a browser
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

# ── project root on sys.path so `src` imports resolve ──────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.utils.datahandler import reconstruct_snapshot_tensors  # noqa: E402

import dash  # noqa: E402
from dash import Input, Output, State, ctx, dcc, html  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────────
NUM_EXPERIMENTS = 24                       # points per LineBO line
RUNS_DIR        = _HERE.parent / "runs"
DEFAULT_RUN     = "run_63b5"               # preferred default if present

# Cache of loaded line data, keyed by (run, snapshot), to avoid re-reading
# snapshots on every callback. The dev server is single-process so this is safe.
_CACHE: dict[tuple[str, str], list[dict]] = {}


# ── run / snapshot discovery ────────────────────────────────────────────────────

def list_runs() -> list[str]:
    """Run directories under runs/ that have a config and at least one snapshot."""
    if not RUNS_DIR.exists():
        return []
    return [
        d.name
        for d in sorted(RUNS_DIR.iterdir())
        if d.is_dir() and (d / "config.json").exists() and (d / "snapshots").exists()
    ]


def list_snapshots(run: str) -> list[str]:
    snap_dir = RUNS_DIR / run / "snapshots"
    if not snap_dir.exists():
        return []
    return sorted(s.name for s in snap_dir.iterdir() if s.is_dir())


def latest_snapshot(run: str) -> str | None:
    snaps = list_snapshots(run)
    if not snaps:
        return None
    lt = RUNS_DIR / run / "latest.txt"
    if lt.exists():
        name = lt.read_text().strip()
        if name in snaps:
            return name
    return snaps[-1]


# ── line loading ─────────────────────────────────────────────────────────────────

def load_lines(run: str, snapshot: str) -> list[dict]:
    """
    Reconstruct a snapshot and split its paired expected/actual point clouds into
    LineBO lines of ``NUM_EXPERIMENTS`` points each.

    Returns a list (in placement order) of dicts with keys:
      placed   – int placement index (0 = first line sampled)
      expected – (24, 3) expected composition points
      actual   – (24, 3) actual composition points returned by hardware
      y        – (24,)   objective value per actual point
      avg_dist – mean per-point ‖actual − expected‖₂ over the 24 points
    """
    s  = reconstruct_snapshot_tensors(RUNS_DIR / run, snapshot, device="cpu")
    xa = s["X_all_actual"].float().numpy()
    xe = s["X_all_expected"].float().numpy()
    y  = s["Y_all"].float().numpy().ravel()

    if xa.ndim != 2 or xa.shape[1] != 3:
        d = xa.shape[1] if xa.ndim == 2 else "?"
        raise ValueError(
            f"discrepancy.py supports only d=3 (ternary) runs; "
            f"'{run}/{snapshot}' has d={d}."
        )

    n_lines = xa.shape[0] // NUM_EXPERIMENTS
    lines: list[dict] = []
    for i in range(n_lines):
        sl   = slice(i * NUM_EXPERIMENTS, (i + 1) * NUM_EXPERIMENTS)
        a, e = xa[sl], xe[sl]
        dist = np.linalg.norm(a - e, axis=1)
        lines.append({
            "placed":   i,
            "expected": e,
            "actual":   a,
            "y":        y[sl],
            "avg_dist": float(dist.mean()),
        })
    return lines


def get_lines(run: str, snapshot: str) -> list[dict]:
    key = (run, snapshot)
    if key not in _CACHE:
        _CACHE[key] = load_lines(run, snapshot)
    return _CACHE[key]


def ranked(lines: list[dict]) -> list[dict]:
    """Lines sorted worst-first (largest mean discrepancy first)."""
    return sorted(lines, key=lambda ln: ln["avg_dist"], reverse=True)


# ── figure ───────────────────────────────────────────────────────────────────────

def make_figure(line: dict) -> go.Figure:
    e, a, y = line["expected"], line["actual"], line["y"]

    # Connector segments (expected → actual) for each of the 24 points.
    ca, cb, cc = [], [], []
    for i in range(len(a)):
        ca += [e[i, 0], a[i, 0], None]
        cb += [e[i, 1], a[i, 1], None]
        cc += [e[i, 2], a[i, 2], None]

    fig = go.Figure()
    fig.add_trace(go.Scatterternary(
        a=ca, b=cb, c=cc, mode="lines",
        line=dict(color="rgba(150,150,150,0.4)", width=1),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatterternary(
        a=e[:, 0], b=e[:, 1], c=e[:, 2], mode="lines+markers",
        line=dict(color="lightgrey", width=1),
        marker=dict(size=8, color="lightgrey",
                    line=dict(color="grey", width=1)),
        name="Expected", hovertemplate="expected<extra></extra>",
    ))
    fig.add_trace(go.Scatterternary(
        a=a[:, 0], b=a[:, 1], c=a[:, 2], mode="lines+markers",
        line=dict(color="rgba(0,0,0,0.2)", width=1),
        marker=dict(
            size=11, color=y, colorscale="Viridis",
            colorbar=dict(title="Objective Y", len=0.6),
            line=dict(color="black", width=1),
        ),
        name="Actual",
        customdata=y,
        hovertemplate="actual<br>Y = %{customdata:.5f}<extra></extra>",
    ))
    fig.update_layout(
        ternary=dict(
            sum=1,
            aaxis=dict(title="x0", min=0),
            baxis=dict(title="x1", min=0),
            caxis=dict(title="x2", min=0),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=40, r=40, t=60, b=40),
        height=620,
    )
    return fig


def line_list_options(lines: list[dict]) -> list[dict]:
    """Radio options for the right-hand list: worst-first, labelled by placement."""
    opts = []
    for rank, ln in enumerate(ranked(lines), start=1):
        opts.append({
            "label": html.Span(
                f"#{rank}  ·  placed {ln['placed']}  ·  avg dist {ln['avg_dist']:.4f}",
                style={"fontFamily": "monospace", "fontSize": "13px"},
            ),
            "value": ln["placed"],
        })
    return opts


def metric_text(lines: list[dict], placed: int) -> str:
    by_placed = {ln["placed"]: ln for ln in lines}
    ln = by_placed.get(placed)
    if ln is None:
        return "No line selected."
    order = ranked(lines)
    rank  = next(i for i, x in enumerate(order, start=1) if x["placed"] == placed)
    return (f"Line placed #{placed}  —  mean expected↔actual distance "
            f"= {ln['avg_dist']:.5f}   (closeness rank {rank} of {len(lines)}, "
            f"worst = rank 1)")


# ── app ────────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__)
app.title = "ZoMBI-Hop · Line Discrepancy"

_runs        = list_runs()
_default_run = DEFAULT_RUN if DEFAULT_RUN in _runs else (_runs[0] if _runs else None)

app.layout = html.Div(
    style={"fontFamily": "system-ui, sans-serif", "padding": "12px"},
    children=[
        html.H2("Expected vs. actual LineBO lines"),
        html.Div(
            style={"display": "flex", "gap": "16px", "alignItems": "flex-end",
                   "flexWrap": "wrap", "marginBottom": "8px"},
            children=[
                html.Div([
                    html.Label("Run"),
                    dcc.Dropdown(
                        id="run-dd",
                        options=[{"label": r, "value": r} for r in _runs],
                        value=_default_run, clearable=False,
                        style={"width": "240px"},
                    ),
                ]),
                html.Div([
                    html.Label("Snapshot"),
                    dcc.Dropdown(id="snap-dd", clearable=False,
                                 style={"width": "320px"}),
                ]),
                html.Button("🎲 Random line", id="random-btn", n_clicks=0,
                            style={"height": "38px", "cursor": "pointer"}),
            ],
        ),
        html.Div(
            style={"display": "flex", "gap": "16px", "alignItems": "stretch"},
            children=[
                html.Div(
                    style={"flex": "1 1 640px", "minWidth": "480px"},
                    children=[
                        dcc.Graph(id="ternary"),
                        html.Div(id="metric", style={
                            "fontFamily": "monospace", "fontSize": "14px",
                            "padding": "8px 10px", "background": "#f3f3f3",
                            "borderRadius": "6px", "marginTop": "4px"}),
                    ],
                ),
                html.Div(
                    style={"flex": "0 0 320px"},
                    children=[
                        html.Label("Lines — ranked worst → best",
                                   style={"fontWeight": "bold"}),
                        html.Div(
                            dcc.RadioItems(
                                id="line-list", options=[], value=None,
                                labelStyle={"display": "block",
                                            "padding": "3px 4px", "cursor": "pointer"},
                            ),
                            style={"height": "560px", "overflowY": "auto",
                                   "border": "1px solid #ccc", "borderRadius": "6px",
                                   "padding": "6px", "marginTop": "4px"},
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ── callbacks ────────────────────────────────────────────────────────────────────

@app.callback(
    Output("snap-dd", "options"),
    Output("snap-dd", "value"),
    Input("run-dd", "value"),
)
def _update_snapshots(run):
    if not run:
        return [], None
    snaps = list_snapshots(run)
    return ([{"label": s, "value": s} for s in snaps], latest_snapshot(run))


@app.callback(
    Output("line-list", "options"),
    Output("line-list", "value"),
    Input("snap-dd", "value"),
    State("run-dd", "value"),
)
def _populate_lines(snapshot, run):
    if not run or not snapshot:
        return [], None
    lines = get_lines(run, snapshot)
    if not lines:
        return [], None
    options = line_list_options(lines)
    chosen  = random.choice(lines)["placed"]
    return options, chosen


@app.callback(
    Output("line-list", "value", allow_duplicate=True),
    Input("random-btn", "n_clicks"),
    State("run-dd", "value"),
    State("snap-dd", "value"),
    prevent_initial_call=True,
)
def _random_line(_n, run, snapshot):
    if not run or not snapshot:
        return dash.no_update
    lines = get_lines(run, snapshot)
    if not lines:
        return dash.no_update
    return random.choice(lines)["placed"]


@app.callback(
    Output("ternary", "figure"),
    Output("metric", "children"),
    Input("line-list", "value"),
    State("run-dd", "value"),
    State("snap-dd", "value"),
)
def _render(placed, run, snapshot):
    if placed is None or not run or not snapshot:
        return go.Figure(), "Select a run and line."
    lines    = get_lines(run, snapshot)
    by_placed = {ln["placed"]: ln for ln in lines}
    line = by_placed.get(placed)
    if line is None:
        return go.Figure(), "Line not found."
    return make_figure(line), metric_text(lines, placed)


if __name__ == "__main__":
    if not _runs:
        print(f"No runs found under {RUNS_DIR}", file=sys.stderr)
        sys.exit(1)
    app.run(debug=True)
