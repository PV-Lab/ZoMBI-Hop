"""
visualization/discrepancy.py
=============================
Interactive Dash web app for inspecting the *discrepancy* between the line
ZoMBI-Hop **asked** the hardware to print (the "expected"/sent line) and the
line the hardware actually **returned** (the "actual"/measured line), on a
ternary diagram.

Two data sources, picked automatically per run
----------------------------------------------
1. ``composition_log.jsonl``  (preferred — written by scripts/run_zombi_main.py)
   Records, per objective call, the sent and measured compositions for **both
   hardware rails** (the hardware prints two lines at a time: a *main* rail and
   a *cache* rail). When present, each call is shown as two ternaries labelled
   e.g. ``5a`` (main) and ``5b`` (cache):
     * expected/sent line  — grey points + line;
     * actual/measured line — points coloured by objective value (viridis,
                               mirroring interface/app.py).
   The closeness metric is the mean per-point ‖measured − sent‖₂.

2. Legacy fallback (older runs with no composition log)
   Older runs only stored the *main* rail's measured points, and the saved
   "expected" array is a degenerate single PCA fit through the batch — the true
   requested line was never logged. So for these runs we segment the saved
   points into one line per optimizer iteration (using the per-snapshot point
   counts, NOT a fixed chunk of 24 — dedup makes lines variable-length) and draw
   each iteration's measured points against a **best-fit reference line** (grey).
   The metric is then the RMS perpendicular distance to that fit (how straight /
   noisy the rail was), and a banner makes clear the true requested line and the
   second rail are unavailable for these runs.

3. Standalone pickle pair (``data/sent_compositions.pkl`` + ``actual_compositions.pkl``)
   Two ``(24, 3, n)`` NumPy arrays — the sent (requested) and actual (measured)
   lines, aligned 1:1 (line ``i`` is ``arr[:, :, i]``). Each line is shown as one
   rail with the mean per-point ‖measured − sent‖₂. The actual array is supplied
   externally in the identical format the recreate script writes for `sent`.

A "Random line" button jumps to a random iteration. The scrollable list on the
right shows every rail, labelled by placement order and **ranked worst-first**
by the closeness metric; the same metric is shown beneath the ternaries.

Only d=3 (true ternary) runs are supported.

Usage
-----
  conda activate zombi-hop
  python visualization/discrepancy.py
  # open the printed http://127.0.0.1:8050 in a browser
"""
from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

# ── project root on sys.path so `src` imports resolve ──────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.utils.datahandler import reconstruct_snapshot_tensors  # noqa: E402

import dash  # noqa: E402
from dash import Input, Output, State, dcc, html  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────────
RUNS_DIR    = _HERE.parent / "runs"
DEFAULT_RUN = "run_7eb9"
COMP_LOG    = "composition_log.jsonl"

# Standalone pickled array pair (see module docstring, source 3). Both are numpy
# arrays of shape (24, 3, n) — the 24 points per line, 3 composition ratios, and n
# lines — as written by visualization/recreate_composition_csvs.py (sent) and
# produced externally in the identical format (actual). Line i is arr[:, :, i]; the
# two arrays are aligned 1:1.
SENT_PKL    = _HERE.parent / "data" / "sent_compositions.pkl"
ACTUAL_PKL  = _HERE.parent / "data" / "actual_compositions.pkl"
PKL_RUN     = "📦 PKL: sent vs actual"

# Cache of loaded groups, keyed by (run, snapshot). Single-process dev server.
_CACHE: dict[tuple[str, str], tuple] = {}


# ── run / snapshot discovery ────────────────────────────────────────────────────

def list_runs() -> list[str]:
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


def has_comp_log(run: str) -> bool:
    return (RUNS_DIR / run / COMP_LOG).exists()


# ── geometry helpers ────────────────────────────────────────────────────────────

def _avg_dist(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return 0.0
    return float(np.linalg.norm(a - b, axis=1).mean())


def fit_line_feet(P: np.ndarray) -> np.ndarray:
    """Return, for each point, the foot of its perpendicular on the best-fit line."""
    if len(P) < 2:
        return P.copy()
    mean = P.mean(axis=0)
    c = P - mean
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    direction = Vt[0]
    t = c @ direction
    return mean + np.outer(t, direction)


# ── group / rail construction ────────────────────────────────────────────────────
# A "group" is one optimizer iteration (one objective call). It holds 1–2 rails.
# A "rail" dict: {label, expected (N,3), actual (N,3), y (N,), avg_dist, kind}.

def _rail(label, expected, actual, y, kind):
    return {
        "label":    label,
        "expected": np.asarray(expected, float),
        "actual":   np.asarray(actual, float),
        "y":        np.asarray(y, float).ravel(),
        "avg_dist": _avg_dist(np.asarray(actual, float), np.asarray(expected, float)),
        "kind":     kind,
    }


def load_from_comp_log(run: str) -> list[dict]:
    """Build groups (with both rails) from composition_log.jsonl."""
    groups: list[dict] = []
    path = RUNS_DIR / run / COMP_LOG
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        idx = int(rec.get("call", len(groups)))
        rails = []
        for ri, rail in enumerate(rec.get("rails", [])):
            sent = np.asarray(rail.get("sent", []), float)
            meas = np.asarray(rail.get("measured", []), float)
            yv   = np.asarray(rail.get("y", []), float).ravel()
            if sent.ndim != 2 or sent.shape[0] == 0 or sent.shape != meas.shape:
                continue
            if sent.shape[1] != 3:
                raise ValueError(
                    f"discrepancy.py supports only d=3; run '{run}' has d={sent.shape[1]}.")
            suffix = "ab"[ri] if ri < 2 else str(ri)
            rails.append(_rail(f"{idx}{suffix}", sent, meas, yv, kind="sent"))
        if rails:
            groups.append({"index": idx, "label": f"call {idx}", "rails": rails})
    return groups


def _snapshot_boundaries(run: str, snapshot: str) -> list[int]:
    """Cumulative point counts at each snapshot up to `snapshot`, from summary.json."""
    counts: list[int] = []
    for sn in list_snapshots(run):
        summ = RUNS_DIR / run / "snapshots" / sn / "summary.json"
        try:
            n = int(json.loads(summ.read_text()).get("n_points", 0))
        except Exception:
            n = 0
        counts.append(n)
        if sn == snapshot:
            break
    return counts


def load_legacy(run: str, snapshot: str) -> list[dict]:
    """
    Fallback for runs without a composition log: one rail per optimizer iteration,
    segmented by per-snapshot point counts, compared against a best-fit reference.
    """
    s  = reconstruct_snapshot_tensors(RUNS_DIR / run, snapshot, device="cpu")
    xa = s["X_all_actual"].float().numpy()
    y  = s["Y_all"].float().numpy().ravel()
    if xa.ndim != 2 or xa.shape[1] != 3:
        d = xa.shape[1] if xa.ndim == 2 else "?"
        raise ValueError(f"discrepancy.py supports only d=3; run '{run}' has d={d}.")

    # Iteration boundaries = the distinct increasing cumulative counts.
    bounds = sorted({c for c in _snapshot_boundaries(run, snapshot) if c > 0})
    groups: list[dict] = []
    prev, gi = 0, 0
    for b in bounds:
        b = min(b, len(xa))
        if b <= prev:
            continue
        actual = xa[prev:b]
        feet   = fit_line_feet(actual)            # grey best-fit reference points
        groups.append({
            "index": gi, "label": f"iter {gi}",
            "rails": [_rail(str(gi), feet, actual, y[prev:b], kind="fit")],
        })
        prev, gi = b, gi + 1
    return groups


# ── standalone pickle pair (source 3) ─────────────────────────────────────────────

def _load_pkl_array(path: Path, width: int | None = None) -> np.ndarray:
    """Load a pickled (points, C, n) composition array; validate its shape.

    `width`, when given, is the exact component count C the array must have.
    """
    with open(path, "rb") as f:
        arr = np.asarray(pickle.load(f), dtype=float)
    if arr.ndim != 3 or (width is not None and arr.shape[1] != width):
        want = f"(points, {width}, n_lines)" if width else "(points, C, n_lines)"
        raise ValueError(f"{path.name} must have shape {want}; got {arr.shape}.")
    return arr


def _actual_wide_flat(actual: np.ndarray) -> np.ndarray:
    """(points, C, n) → (n*points, C), line-major (line 0 point 0 first)."""
    if actual.size == 0:
        return np.empty((0, actual.shape[1] if actual.ndim == 3 else 0))
    return actual.transpose(2, 0, 1).reshape(-1, actual.shape[1])


def _load_pkl_pair() -> tuple[np.ndarray, np.ndarray, bool]:
    """Load the sent/actual pickle pair. Returns (sent (24,3,n), actual (24,C,n), ok).

    `ok` is True only when both files are present, valid and mutually consistent
    (same point count and line count). `sent` is the requested lines (3 ratios);
    `actual` is the measured lines embedded in the wider hardware channel space
    (C channels, of which only 3 are active). If the sent pickle is present but the
    actual one (obtained externally, same format) is missing or inconsistent, an
    error is printed and `ok` is False.
    """
    empty3, emptyC = np.empty((0, 3, 0)), np.empty((0, 0, 0))
    if not SENT_PKL.exists():
        return empty3, emptyC, False
    try:
        sent = _load_pkl_array(SENT_PKL, width=3)
    except (OSError, ValueError, pickle.UnpicklingError) as exc:
        print(f"Error reading {SENT_PKL}: {exc}", file=sys.stderr)
        return empty3, emptyC, False
    if not ACTUAL_PKL.exists():
        print(f"Error: found {SENT_PKL.name} but no actual pickle at {ACTUAL_PKL}. "
              f"Provide a measured array of shape (points, C, n) in the same format "
              f"to enable the sent-vs-actual view.", file=sys.stderr)
        return sent, emptyC, False
    try:
        actual = _load_pkl_array(ACTUAL_PKL)
    except (OSError, ValueError, pickle.UnpicklingError) as exc:
        print(f"Error reading {ACTUAL_PKL}: {exc}", file=sys.stderr)
        return sent, emptyC, False
    if (sent.shape[0], sent.shape[2]) != (actual.shape[0], actual.shape[2]):
        print(f"Error: {SENT_PKL.name} shape {sent.shape} and {ACTUAL_PKL.name} shape "
              f"{actual.shape} disagree on points/lines; they must match 1:1.",
              file=sys.stderr)
        return sent, actual, False
    if actual.shape[1] < 3:
        print(f"Error: {ACTUAL_PKL.name} has only {actual.shape[1]} channels; "
              f"need at least 3.", file=sys.stderr)
        return sent, actual, False
    return sent, actual, True


def _auto_col_map(actual_raw: np.ndarray, anchor: np.ndarray) -> tuple[int, int, int]:
    """Infer which 3 wide-channel columns correspond to sent components x0,x1,x2.

    The measured array stores every hardware channel but only three are ever
    non-zero. The very first measured point echoes the first sent point, so we
    match each sent component to the active column nearest to it at that anchor
    row — an unambiguous, order-preserving assignment.
    """
    active = [j for j in range(actual_raw.shape[1]) if np.abs(actual_raw[:, j]).max() > 1e-9]
    mapping, used = [], set()
    for k in range(3):
        target = anchor[k]
        cand = [j for j in active if j not in used] or [j for j in range(actual_raw.shape[1]) if j not in used]
        best = min(cand, key=lambda j: abs(actual_raw[0, j] - target))
        mapping.append(best)
        used.add(best)
    return tuple(mapping)  # type: ignore[return-value]


# Parsed once at import; the arrays are tiny and never change during a session.
_SENT_ARR, _ACTUAL_ARR, _HAS_PKL = _load_pkl_pair()
_ACTUAL_COLS = [str(j) for j in range(_ACTUAL_ARR.shape[1])] if _ACTUAL_ARR.ndim == 3 else []
if _HAS_PKL:
    _AUTO_COL_MAP = _auto_col_map(_actual_wide_flat(_ACTUAL_ARR), _SENT_ARR[0, :, 0])
else:
    _AUTO_COL_MAP = (0, 1, 2)


def load_pkl_pair(col_map: tuple[int, int, int] | None) -> list[dict]:
    """Build groups from the sent/actual pickle pair — one rail per line.

    Each line's 24 points are aligned 1:1 between sent and actual (line i =
    arr[:, :, i]). `col_map` selects which 3 wide actual-channels feed x0/x1/x2;
    when None the auto-inferred mapping is used.
    """
    cmap = list(col_map or _AUTO_COL_MAP)
    groups: list[dict] = []
    n = _SENT_ARR.shape[2]
    for gi in range(n):
        sent = _SENT_ARR[:, :, gi]              # (24, 3)
        act  = _ACTUAL_ARR[:, cmap, gi]         # (24, 3)
        label = f"line{gi + 1}"
        rail = _rail(label, sent, act, [], kind="sent")
        rail["connect"] = True                  # every line is a swept 24-point gradient
        groups.append({"index": gi, "label": label, "rails": [rail]})
    return groups


def get_groups(run: str, snapshot: str,
               col_map: tuple[int, int, int] | None = None) -> tuple[list[dict], str]:
    """Return (groups, mode) where mode is 'pkl', 'log' or 'legacy'."""
    if run == PKL_RUN:
        key = ("__pkl__", col_map)
        if key not in _CACHE:
            _CACHE[key] = (load_pkl_pair(col_map), "pkl")
        return _CACHE[key]
    key = (run, snapshot if not has_comp_log(run) else "__log__")
    if key not in _CACHE:
        if has_comp_log(run):
            _CACHE[key] = (load_from_comp_log(run), "log")
        else:
            _CACHE[key] = (load_legacy(run, snapshot), "legacy")
    return _CACHE[key]


def all_rails(groups: list[dict]) -> list[tuple[dict, dict]]:
    """Flat list of (group, rail) pairs, worst-first by avg_dist."""
    pairs = [(g, r) for g in groups for r in g["rails"]]
    return sorted(pairs, key=lambda gr: gr[1]["avg_dist"], reverse=True)


# ── figures ──────────────────────────────────────────────────────────────────────

def make_figure(rail: dict) -> go.Figure:
    e, a, y = rail["expected"], rail["actual"], rail["y"]
    # `init`-style rails are scatters of unordered seed points, not a swept line, so
    # drawing a polyline through them is misleading — show markers only for those.
    connect = rail.get("connect", True)
    pt_mode = "lines+markers" if connect else "markers"

    fig = go.Figure()
    # The grey "expected/sent" line + connectors are only meaningful when we have a
    # real sent line to compare against (log/csv mode). For legacy runs (kind="fit")
    # there is no true expected line, so just show the measured points.
    if rail["kind"] != "fit":
        ca, cb, cc = [], [], []
        for i in range(len(a)):
            ca += [e[i, 0], a[i, 0], None]
            cb += [e[i, 1], a[i, 1], None]
            cc += [e[i, 2], a[i, 2], None]
        fig.add_trace(go.Scatterternary(
            a=ca, b=cb, c=cc, mode="lines",
            line=dict(color="rgba(150,150,150,0.45)", width=1),
            hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatterternary(
            a=e[:, 0], b=e[:, 1], c=e[:, 2], mode=pt_mode,
            line=dict(color="lightgrey", width=1),
            marker=dict(size=7, color="lightgrey", line=dict(color="grey", width=1)),
            name="Expected (sent)", hovertemplate="Expected (sent)<extra></extra>"))
    has_y = y.size == len(a) and np.isfinite(y).any()
    fig.add_trace(go.Scatterternary(
        a=a[:, 0], b=a[:, 1], c=a[:, 2], mode=pt_mode,
        line=dict(color="rgba(0,0,0,0.2)", width=1),
        marker=dict(
            size=10,
            color=(y if has_y else "steelblue"),
            colorscale="Viridis" if has_y else None,
            colorbar=(dict(title="Objective Y", len=0.7) if has_y else None),
            line=dict(color="black", width=1)),
        name="Actual (measured)",
        customdata=(y if has_y else None),
        hovertemplate=("actual<br>Y = %{customdata:.5f}<extra></extra>"
                       if has_y else "actual<extra></extra>")))
    fig.update_layout(
        title=dict(text=f"Line {rail['label']}   ·   avg dist {rail['avg_dist']:.4f}",
                   x=0.5, font=dict(size=14)),
        ternary=dict(sum=1, aaxis=dict(title="x0", min=0),
                     baxis=dict(title="x1", min=0), caxis=dict(title="x2", min=0)),
        showlegend=False, margin=dict(l=30, r=30, t=50, b=30), height=520)
    return fig


def list_options(groups: list[dict]) -> list[dict]:
    opts = []
    for rank, (g, r) in enumerate(all_rails(groups), start=1):
        opts.append({
            "label": html.Span(
                f"#{rank}  ·  {r['label']}  ·  avg dist {r['avg_dist']:.4f}",
                style={"fontFamily": "monospace", "fontSize": "13px"}),
            "value": g["index"],
        })
    return opts


def metric_children(groups: list[dict], group_index: int):
    order = all_rails(groups)
    rank_of = {id(r): i for i, (_, r) in enumerate(order, start=1)}
    grp = next((g for g in groups if g["index"] == group_index), None)
    if grp is None:
        return "No line selected."
    parts = []
    for r in grp["rails"]:
        label = ("mean perpendicular distance to best-fit"
                 if r["kind"] == "fit" else "mean sent↔measured distance")
        parts.append(html.Div(
            f"Line {r['label']}  —  {label} = {r['avg_dist']:.5f}   "
            f"(closeness rank {rank_of[id(r)]} of {len(order)}, worst = rank 1)"))
    return parts


# ── app ────────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__)
app.title = "ZoMBI-Hop · Line Discrepancy"

_runs         = list_runs()
_run_options  = ([{"label": PKL_RUN, "value": PKL_RUN}] if _HAS_PKL else []) \
                + [{"label": r, "value": r} for r in _runs]
# Default to the pickle pair when present (it's the current focus); else fall back.
_default_run  = PKL_RUN if _HAS_PKL else \
                (DEFAULT_RUN if DEFAULT_RUN in _runs else (_runs[0] if _runs else None))

# Options for the "which wide channel feeds x0/x1/x2" override dropdowns.
_col_options  = [{"label": f"col {name}", "value": j}
                 for j, name in enumerate(_ACTUAL_COLS)]

app.layout = html.Div(
    style={"fontFamily": "system-ui, sans-serif", "padding": "12px"},
    children=[
        html.H2("Expected (sent) vs. actual (measured) hardware lines"),
        html.Div(
            style={"display": "flex", "gap": "16px", "alignItems": "flex-end",
                   "flexWrap": "wrap", "marginBottom": "6px"},
            children=[
                html.Div([
                    html.Label("Run"),
                    dcc.Dropdown(
                        id="run-dd",
                        options=_run_options,
                        value=_default_run, clearable=False,
                        style={"width": "240px"}),
                ]),
                html.Div([
                    html.Label("Snapshot (legacy runs only)"),
                    dcc.Dropdown(id="snap-dd", clearable=False, style={"width": "320px"}),
                ]),
                html.Button("🎲 Random line", id="random-btn", n_clicks=0,
                            style={"height": "38px", "cursor": "pointer"}),
            ],
        ),
        # Column-mapping override — pickle mode only. The mapping is auto-inferred
        # from the anchor point; these let you correct it if the guess is ever wrong.
        html.Div(
            id="csv-controls",
            style={"display": "flex", "gap": "10px", "alignItems": "flex-end",
                   "flexWrap": "wrap", "marginBottom": "6px"},
            children=[
                html.Div("Actual-column → component mapping:",
                         style={"fontSize": "13px", "alignSelf": "center"}),
                *[html.Div([
                    html.Label(f"x{k}", style={"fontSize": "12px"}),
                    dcc.Dropdown(id=f"col-x{k}", options=_col_options,
                                 value=_AUTO_COL_MAP[k], clearable=False,
                                 style={"width": "110px"}),
                  ]) for k in range(3)],
            ],
        ),
        html.Div(id="mode-banner", style={
            "fontSize": "13px", "padding": "6px 10px", "borderRadius": "6px",
            "marginBottom": "8px"}),
        html.Div(
            style={"display": "flex", "gap": "16px", "alignItems": "stretch"},
            children=[
                html.Div(
                    style={"flex": "1 1 700px", "minWidth": "480px"},
                    children=[
                        html.Div(id="ternaries",
                                 style={"display": "flex", "gap": "12px",
                                        "flexWrap": "wrap"}),
                        html.Div(id="metric", style={
                            "fontFamily": "monospace", "fontSize": "14px",
                            "padding": "8px 10px", "background": "#f3f3f3",
                            "borderRadius": "6px", "marginTop": "4px"}),
                    ],
                ),
                html.Div(
                    style={"flex": "0 0 320px"},
                    children=[
                        html.Label("Rails — ranked worst → best",
                                   style={"fontWeight": "bold"}),
                        html.Div(
                            dcc.RadioItems(
                                id="line-list", options=[], value=None,
                                labelStyle={"display": "block", "padding": "3px 4px",
                                            "cursor": "pointer"}),
                            style={"height": "560px", "overflowY": "auto",
                                   "border": "1px solid #ccc", "borderRadius": "6px",
                                   "padding": "6px", "marginTop": "4px"}),
                    ],
                ),
            ],
        ),
    ],
)


# ── callbacks ────────────────────────────────────────────────────────────────────

def _col_map(x0, x1, x2):
    """Assemble the three override dropdown values into a col_map tuple (or None)."""
    if None in (x0, x1, x2):
        return None
    return (int(x0), int(x1), int(x2))


@app.callback(
    Output("csv-controls", "style"),
    Input("run-dd", "value"),
)
def _toggle_csv_controls(run):
    base = {"display": "flex", "gap": "10px", "alignItems": "flex-end",
            "flexWrap": "wrap", "marginBottom": "6px"}
    return base if run == PKL_RUN else {"display": "none"}


@app.callback(
    Output("snap-dd", "options"),
    Output("snap-dd", "value"),
    Output("snap-dd", "disabled"),
    Input("run-dd", "value"),
)
def _update_snapshots(run):
    if not run:
        return [], None, True
    if run == PKL_RUN:
        return [], None, True  # the pickle pair spans a whole fixed dataset; no snapshot axis.
    snaps = list_snapshots(run)
    # In log mode the snapshot axis is irrelevant (the log spans the whole run).
    return ([{"label": s, "value": s} for s in snaps],
            latest_snapshot(run), has_comp_log(run))


@app.callback(
    Output("line-list", "options"),
    Output("line-list", "value"),
    Output("mode-banner", "children"),
    Output("mode-banner", "style"),
    Input("snap-dd", "value"),
    Input("random-btn", "n_clicks"),
    Input("col-x0", "value"),
    Input("col-x1", "value"),
    Input("col-x2", "value"),
    State("run-dd", "value"),
    State("line-list", "value"),
)
def _populate(snapshot, _n_clicks, cx0, cx1, cx2, run, current_value):
    base_style = {"fontSize": "13px", "padding": "6px 10px", "borderRadius": "6px",
                  "marginBottom": "8px"}
    if not run or (run != PKL_RUN and not snapshot):
        return [], None, "", base_style
    groups, mode = get_groups(run, snapshot, _col_map(cx0, cx1, cx2))
    if not groups:
        return [], None, "No lines found for this run.", base_style

    trigger = dash.callback_context.triggered_id
    if trigger == "random-btn":
        value = random.choice(groups)["index"]
    elif current_value is not None and any(g["index"] == current_value for g in groups):
        value = current_value
    else:
        value = random.choice(groups)["index"]

    if mode == "pkl":
        banner = ("✓ Using data/sent_compositions.pkl vs. data/actual_compositions.pkl "
                  "— (24, 3, n) sent and (24, C, n) actual arrays aligned 1:1, one rail "
                  "per line. Adjust the channel→component mapping above if the "
                  "auto-guess is wrong.")
        style = {**base_style, "background": "#e6f0fb", "border": "1px solid #9bf"}
    elif mode == "log":
        banner = ("✓ Using composition_log.jsonl — true sent vs. measured "
                  "compositions, both rails (a = main, b = cache). "
                  "Snapshot selector is ignored in this mode.")
        style = {**base_style, "background": "#e6f5e6", "border": "1px solid #9c9"}
    else:
        banner = ""
        style = {**base_style, "display": "none"}
    return list_options(groups), value, banner, style


@app.callback(
    Output("ternaries", "children"),
    Output("metric", "children"),
    Input("line-list", "value"),
    State("run-dd", "value"),
    State("snap-dd", "value"),
    State("col-x0", "value"),
    State("col-x1", "value"),
    State("col-x2", "value"),
)
def _render(group_index, run, snapshot, cx0, cx1, cx2):
    if group_index is None or not run or (run != PKL_RUN and not snapshot):
        return [], "Select a run and line."
    groups, _ = get_groups(run, snapshot, _col_map(cx0, cx1, cx2))
    grp = next((g for g in groups if g["index"] == group_index), None)
    if grp is None:
        return [], "Line not found."
    graphs = [
        dcc.Graph(figure=make_figure(r),
                  style={"flex": "1 1 460px", "minWidth": "360px"})
        for r in grp["rails"]
    ]
    return graphs, metric_children(groups, group_index)


if __name__ == "__main__":
    if not _runs and not _HAS_PKL:
        print(f"No runs found under {RUNS_DIR} and no pickle pair at "
              f"{SENT_PKL} / {ACTUAL_PKL}", file=sys.stderr)
        sys.exit(1)
    app.run(debug=True)
