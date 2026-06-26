"""
visualization/plot_run.py
=========================
Interactive Dash app (with an optional static-PNG export mode) that plots every
datapoint collected during a *real* ZoMBI-Hop run on a ternary diagram, over a
Random-Forest-interpolated background of those same points.

Two data sources are supported, selectable in the app:

  * **Run directory** (e.g. ``runs/run_7eb9``) — the full measured dataset
    ``(X, Y)`` is reconstructed from the run's delta snapshots
    (``reconstruct_snapshot_tensors``).
  * **Database** (e.g. ``data/2nd_real_run.db``) — the ``results`` table is read
    directly; the three composition columns (FAPbI3, MAPbI3, MAPbBr3) form ``X``
    and a chosen value column (default ``Objective``) forms ``Y``.

For whichever source, the app:
  1. Trains a Random-Forest surrogate on ``(X, Y)`` and evaluates it over a
     dense ternary grid — the smooth background heatmap (viridis).
  2. Overlays every collected datapoint on top, each coloured by its measured
     value (same viridis scale as the background) with a black outline so
     individual measurements stay legible.

Only d=3 (true ternary) data is supported.

Usage
-----
  conda activate zombi-hop
  python visualization/plot_run.py                      # launch the Dash app
  python visualization/plot_run.py --port 8051          # app on a custom port

  # Static PNG export (legacy behaviour), no server:
  python visualization/plot_run.py --export --run runs/run_7eb9
  python visualization/plot_run.py --export --db data/2nd_real_run.db --out db.png

Flags
-----
  --port N          Port for the Dash app (default: 8050).
  --export          Render a static PNG instead of launching the app.
  --run PATH        Run directory for --export (default: runs/run_7eb9).
  --db PATH         Database file for --export (overrides --run if given).
  --value COL       DB value column to plot as the objective (default: Objective).
  --snapshot NAME   Snapshot to reconstruct up to (default: latest.txt).
  --out PATH        Output PNG path for --export.
  --grid-n N        Ternary grid resolution for the RF background (default: 120).
  --n-estimators N  Number of trees in the RF surrogate (default: 500).
  --corner-dims D   Run dims placed at [bottom-left,bottom-right,top]
                    (default: 9,8,0 — dim 9 bottom-left, dim 0 at the top).
  --labels A,B,C    Corner labels for [bottom-left,bottom-right,top]
                    (default: FAPbI3,MAPbI3,MAPbBr3).
  --show            (export only) Display the matplotlib figure as well as saving.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

# ── project root on sys.path so `src` imports resolve ──────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.utils.datahandler import reconstruct_snapshot_tensors  # noqa: E402

# ── constants ──────────────────────────────────────────────────────────────────
ROOT = _HERE.parent
RUNS_DIR = ROOT / "runs"
DATA_DIR = ROOT / "data"
DEFAULT_RUN = "run_7eb9"
RF_N_ESTIMATORS = 500
TERNARY_GRID_N = 120
_SQRT3_2 = np.sqrt(3) / 2

# Physical identity of each hardware dim, and which ternary corner it occupies.
# CORNER_DIM_ORDER lists dims for [bottom-left, bottom-right, top]; the measured
# composition columns (ordered by the run's optimizing dims) are reindexed to
# match so each labelled component lands in the requested corner.
DIM_LABELS: dict[int, str] = {9: "FAPbI3", 0: "MAPbBr3", 8: "MAPbI3"}
CORNER_DIM_ORDER: list[int] = [9, 8, 0]   # bottom-left, bottom-right, top

# Composition columns in a ``results`` DB table, ordered [bottom-left,
# bottom-right, top] to match comp_to_xy / the run default corner layout.
DB_COMP_COLS: list[str] = ["FAPbI3", "MAPbI3", "MAPbBr3"]
DB_VALUE_COLS: list[str] = ["Objective", "Bandgap", "Photoconductance", "Stability"]
DEFAULT_DB_VALUE = "Objective"


# ── ternary utilities (mirror interactive_test_zombi.py) ───────────────────────

def comp_to_xy(comp: np.ndarray) -> np.ndarray:
    """(N, 3) simplex compositions → (N, 2) Cartesian ternary coordinates.

    Corner mapping:  comp[:,0] → (0,0),  comp[:,1] → (1,0),  comp[:,2] → (0.5, √3/2).
    """
    p = np.asarray(comp, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    s = p.sum(axis=-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return np.column_stack([p[:, 1] + 0.5 * p[:, 2], _SQRT3_2 * p[:, 2]])


def ternary_grid(n: int = 120) -> np.ndarray:
    """Return (N, 3) uniform grid on the probability simplex."""
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append([i / n, j / n, (n - i - j) / n])
    return np.array(pts, dtype=float)


# ── data loading: run directories ────────────────────────────────────────────

def _resolve_run_dir(run_arg: str) -> Path:
    """Accept a full path, or a bare run name resolved against runs/."""
    p = Path(run_arg)
    if p.is_dir():
        return p.resolve()
    candidate = RUNS_DIR / run_arg
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"Run directory not found: {run_arg}")


def _default_snapshot(run_dir: Path) -> str:
    """The snapshot named in latest.txt, or the last snapshot directory."""
    latest = run_dir / "latest.txt"
    if latest.exists():
        name = latest.read_text().strip()
        if name:
            return name
    snaps = sorted(s.name for s in (run_dir / "snapshots").iterdir() if s.is_dir())
    if not snaps:
        raise FileNotFoundError(f"No snapshots found under {run_dir}")
    return snaps[-1]


def _run_dims(run_dir: Path) -> list[int] | None:
    """The run's optimizing dims, in measured-column order, or None."""
    lp = run_dir / "live_plot_state.json"
    if lp.exists():
        try:
            dims = json.loads(lp.read_text()).get("optimizing_dims")
            if dims:
                return [int(d) for d in dims]
        except Exception:
            pass
    hw = run_dir / "hw_config.json"
    if hw.exists():
        try:
            raw = json.loads(hw.read_text()).get("dims", "")
            dims = [int(x) for x in str(raw).split(",") if x.strip() != ""]
            if dims:
                return dims
        except Exception:
            pass
    return None


def _corner_config(
    run_dir: Path,
    corner_dims_override: str | None,
    labels_override: str | None,
) -> tuple[list[int] | None, tuple[str, str, str]]:
    """Resolve the corner dim ordering and corner labels.

    Returns ``(corner_dims, labels)`` where ``corner_dims`` lists the run dims
    placed at [bottom-left, bottom-right, top] (or None if the run dims are
    unknown / don't match), and ``labels`` are the matching corner labels.
    """
    run_dims = _run_dims(run_dir)

    # Corner dim ordering: CLI override, else the default physical layout when
    # all its dims are present in this run, else the run's own column order.
    if corner_dims_override:
        corner_dims = [int(x) for x in corner_dims_override.split(",") if x.strip() != ""]
        if len(corner_dims) != 3:
            raise ValueError("--corner-dims must be three comma-separated dim indices")
    elif run_dims is not None and all(d in run_dims for d in CORNER_DIM_ORDER):
        corner_dims = list(CORNER_DIM_ORDER)
    else:
        corner_dims = list(run_dims) if run_dims else None

    # Labels: CLI override, else the physical DIM_LABELS map, else "dim N" / A,B,C.
    if labels_override:
        parts = [s.strip() for s in labels_override.split(",")]
        if len(parts) != 3:
            raise ValueError("--labels must be three comma-separated names")
        labels = (parts[0], parts[1], parts[2])
    elif corner_dims is not None:
        labels = tuple(DIM_LABELS.get(d, f"dim {d}") for d in corner_dims)  # type: ignore[assignment]
    else:
        labels = ("A", "B", "C")

    return corner_dims, labels  # type: ignore[return-value]


def _reorder_columns(
    X: np.ndarray, run_dims: list[int] | None, corner_dims: list[int] | None
) -> np.ndarray:
    """Reindex composition columns so corner_dims map to [col0, col1, col2]."""
    if not run_dims or not corner_dims:
        return X
    try:
        idx = [run_dims.index(d) for d in corner_dims]
    except ValueError:
        return X
    return X[:, idx]


def load_run_dataset(run_dir: Path, snapshot: str) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the full measured dataset ``(X (N,3), Y (N,))`` for a run."""
    tensors = reconstruct_snapshot_tensors(run_dir, snapshot, device="cpu")
    X = tensors.get("X_all_actual")
    Y = tensors.get("Y_all")
    if X is None or Y is None or X.shape[0] == 0:
        raise RuntimeError(f"No datapoints reconstructed from {run_dir}/{snapshot}")
    X = X.detach().cpu().numpy().astype(float)
    Y = Y.detach().cpu().numpy().astype(float).ravel()
    if X.shape[1] != 3:
        raise ValueError(f"Only d=3 runs are supported (got d={X.shape[1]}).")
    # Normalise rows to sum 1 (guards against tiny numerical drift).
    s = X.sum(axis=1, keepdims=True)
    X = X / np.where(s == 0, 1.0, s)
    return X, Y


def load_run_source(
    run_dir: Path,
    snapshot: str | None,
    corner_dims_override: str | None = None,
    labels_override: str | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, str, str], str]:
    """Load a run directory → ``(X, Y, labels, title)`` ready for plotting."""
    snapshot = snapshot or _default_snapshot(run_dir)
    run_dims = _run_dims(run_dir)
    corner_dims, labels = _corner_config(run_dir, corner_dims_override, labels_override)
    X, Y = load_run_dataset(run_dir, snapshot)
    X = _reorder_columns(X, run_dims, corner_dims)
    title = f"{run_dir.name} — collected datapoints  ({snapshot})"
    return X, Y, labels, title


# ── data loading: database ────────────────────────────────────────────────────

def _resolve_db_path(db_arg: str) -> Path:
    """Accept a full path, or a bare db name resolved against data/."""
    p = Path(db_arg)
    if p.is_file():
        return p.resolve()
    candidate = DATA_DIR / db_arg
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Database file not found: {db_arg}")


def db_value_columns(db_path: Path) -> list[str]:
    """Candidate value columns present (with data) in the db's results table."""
    con = sqlite3.connect(str(db_path))
    try:
        have = {r[1] for r in con.execute("PRAGMA table_info(results)")}
        cols = []
        for c in DB_VALUE_COLS:
            if c in have:
                n = con.execute(
                    f'SELECT COUNT(*) FROM results WHERE "{c}" IS NOT NULL'
                ).fetchone()[0]
                if n > 0:
                    cols.append(c)
        return cols or [c for c in DB_VALUE_COLS if c in have]
    finally:
        con.close()


def load_db_dataset(
    db_path: Path, value_col: str = DEFAULT_DB_VALUE
) -> tuple[np.ndarray, np.ndarray, tuple[str, str, str], str]:
    """Read the ``results`` table → ``(X (N,3), Y (N,), labels, title)``.

    Rows missing the value column or any composition column are dropped, and the
    composition rows are renormalised to sum 1.
    """
    cols = DB_COMP_COLS + [value_col]
    con = sqlite3.connect(str(db_path))
    try:
        sel = ", ".join(f'"{c}"' for c in cols)
        where = " AND ".join(f'"{c}" IS NOT NULL' for c in cols)
        rows = con.execute(f"SELECT {sel} FROM results WHERE {where}").fetchall()
    finally:
        con.close()
    if not rows:
        raise RuntimeError(f"No rows with non-null {DB_COMP_COLS} + {value_col} in {db_path}")
    arr = np.asarray(rows, dtype=float)
    X = arr[:, :3]
    Y = arr[:, 3]
    s = X.sum(axis=1, keepdims=True)
    X = X / np.where(s == 0, 1.0, s)
    labels = (DB_COMP_COLS[0], DB_COMP_COLS[1], DB_COMP_COLS[2])
    title = f"{db_path.name} — {X.shape[0]} measured points  ({value_col})"
    return X, Y, labels, title


# ── RF surrogate ─────────────────────────────────────────────────────────────

def fit_rf_background(
    X: np.ndarray, Y: np.ndarray, grid_n: int, n_estimators: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an RF surrogate on (X, Y) and evaluate it over a ternary grid.

    Returns ``(grid_pts (M,3), grid_vals (M,))``.
    """
    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    rf.fit(X, Y)
    grid_pts = ternary_grid(grid_n)
    grid_vals = rf.predict(grid_pts)
    return grid_pts, grid_vals


# ── plotly figure ─────────────────────────────────────────────────────────────

def build_ternary_figure(
    X: np.ndarray,
    Y: np.ndarray,
    labels: tuple[str, str, str],
    *,
    grid_n: int,
    n_estimators: int,
    title: str,
    value_name: str = "Objective",
):
    """Interactive Plotly ternary: RF background grid + measured points overlay.

    Corner mapping mirrors comp_to_xy / matplotlib:
      col0 → bottom-left (b),  col1 → bottom-right (c),  col2 → top (a).
    """
    import plotly.graph_objects as go

    grid_pts, grid_vals = fit_rf_background(X, Y, grid_n, n_estimators)

    vmin = float(min(grid_vals.min(), Y.min()))
    vmax = float(max(grid_vals.max(), Y.max()))
    if vmax <= vmin:
        vmax = vmin + 1e-9

    fig = go.Figure()

    # Background: RF-interpolated heatmap rendered as a dense marker grid.
    fig.add_trace(go.Scatterternary(
        a=grid_pts[:, 2], b=grid_pts[:, 0], c=grid_pts[:, 1],
        mode="markers",
        marker=dict(
            symbol="circle", size=5, color=grid_vals,
            colorscale="Viridis", cmin=vmin, cmax=vmax,
            opacity=0.80, line=dict(width=0),
        ),
        hoverinfo="skip",
        name="RF background",
        showlegend=False,
    ))

    # Foreground: every measured datapoint, coloured by value with a black edge.
    customdata = np.column_stack([X[:, 0], X[:, 1], X[:, 2], Y])
    fig.add_trace(go.Scatterternary(
        a=X[:, 2], b=X[:, 0], c=X[:, 1],
        mode="markers",
        marker=dict(
            symbol="circle", size=9, color=Y,
            colorscale="Viridis", cmin=vmin, cmax=vmax,
            line=dict(width=1.0, color="black"),
            colorbar=dict(title=value_name, thickness=16, len=0.75),
        ),
        customdata=customdata,
        hovertemplate=(
            f"{labels[0]}=%{{customdata[0]:.3f}}<br>"
            f"{labels[1]}=%{{customdata[1]:.3f}}<br>"
            f"{labels[2]}=%{{customdata[2]:.3f}}<br>"
            f"{value_name}=%{{customdata[3]:.4f}}<extra></extra>"
        ),
        name="measured",
        showlegend=False,
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        ternary=dict(
            sum=1,
            aaxis=dict(title=labels[2], min=0, ticks="outside"),   # top
            baxis=dict(title=labels[0], min=0, ticks="outside"),   # bottom-left
            caxis=dict(title=labels[1], min=0, ticks="outside"),   # bottom-right
            bgcolor="white",
        ),
        margin=dict(l=60, r=60, t=60, b=40),
        height=720,
        annotations=[dict(
            text=f"{X.shape[0]} measured points · RF-interpolated background",
            showarrow=False, x=0.5, y=-0.06, xref="paper", yref="paper",
            font=dict(size=11, color="#555"),
        )],
    )
    return fig


# ── Dash app ──────────────────────────────────────────────────────────────────

def _list_run_dirs() -> list[str]:
    """Names of run directories that contain snapshots, newest first."""
    if not RUNS_DIR.is_dir():
        return []
    runs = [p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "snapshots").is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in runs]


def _list_db_files() -> list[str]:
    """Names of .db files under data/."""
    if not DATA_DIR.is_dir():
        return []
    return sorted(p.name for p in DATA_DIR.glob("*.db"))


def build_app(grid_n: int = TERNARY_GRID_N, n_estimators: int = RF_N_ESTIMATORS):
    """Construct the Dash application."""
    from dash import Dash, dcc, html, Input, Output, no_update

    run_names = _list_run_dirs()
    db_names = _list_db_files()

    default_db = "2nd_real_run.db" if "2nd_real_run.db" in db_names else (
        db_names[0] if db_names else None)
    default_run = DEFAULT_RUN if DEFAULT_RUN in run_names else (
        run_names[0] if run_names else None)

    label_style = {"fontWeight": "600", "marginTop": "12px", "display": "block"}
    panel_style = {
        "width": "300px", "padding": "18px", "boxSizing": "border-box",
        "borderRight": "1px solid #e3e3e3", "background": "#fafafa",
        "height": "100vh", "overflowY": "auto",
    }

    app = Dash(__name__)
    app.title = "ZoMBI-Hop Run Plot"

    app.layout = html.Div(style={"display": "flex", "fontFamily": "system-ui, sans-serif"}, children=[
        html.Div(style=panel_style, children=[
            html.H3("ZoMBI-Hop ternary", style={"marginTop": 0}),

            html.Label("Source", style=label_style),
            dcc.RadioItems(
                id="source-type",
                options=[
                    {"label": " Run directory", "value": "run"},
                    {"label": " Database (.db)", "value": "db"},
                ],
                value="db" if default_db else "run",
                labelStyle={"display": "block"},
            ),

            html.Div(id="run-controls", children=[
                html.Label("Run", style=label_style),
                dcc.Dropdown(
                    id="run-dropdown",
                    options=[{"label": n, "value": n} for n in run_names],
                    value=default_run, clearable=False,
                ),
            ]),

            html.Div(id="db-controls", children=[
                html.Label("Database", style=label_style),
                dcc.Dropdown(
                    id="db-dropdown",
                    options=[{"label": n, "value": n} for n in db_names],
                    value=default_db, clearable=False,
                ),
                html.Label("Value column", style=label_style),
                dcc.Dropdown(id="db-value-dropdown", clearable=False),
            ]),

            html.Label("RF grid resolution", style=label_style),
            dcc.Slider(
                id="grid-n", min=40, max=200, step=20, value=grid_n,
                marks={40: "40", 120: "120", 200: "200"},
            ),

            html.Label("RF trees", style=label_style),
            dcc.Slider(
                id="n-estimators", min=100, max=800, step=100, value=n_estimators,
                marks={100: "100", 500: "500", 800: "800"},
            ),

            html.Div(id="status", style={
                "marginTop": "18px", "fontSize": "13px", "color": "#666",
                "whiteSpace": "pre-wrap",
            }),
        ]),

        html.Div(style={"flex": 1, "padding": "10px"}, children=[
            dcc.Loading(dcc.Graph(id="ternary-graph", style={"height": "90vh"})),
        ]),
    ])

    # Show/hide the run vs db control groups.
    @app.callback(
        Output("run-controls", "style"),
        Output("db-controls", "style"),
        Input("source-type", "value"),
    )
    def _toggle(source_type):
        show, hide = {}, {"display": "none"}
        return (show, hide) if source_type == "run" else (hide, show)

    # Populate the db value-column dropdown when the db selection changes.
    @app.callback(
        Output("db-value-dropdown", "options"),
        Output("db-value-dropdown", "value"),
        Input("db-dropdown", "value"),
    )
    def _db_values(db_name):
        if not db_name:
            return [], None
        try:
            cols = db_value_columns(_resolve_db_path(db_name))
        except Exception:
            return [], None
        value = DEFAULT_DB_VALUE if DEFAULT_DB_VALUE in cols else (cols[0] if cols else None)
        return [{"label": c, "value": c} for c in cols], value

    # Main render callback.
    @app.callback(
        Output("ternary-graph", "figure"),
        Output("status", "children"),
        Input("source-type", "value"),
        Input("run-dropdown", "value"),
        Input("db-dropdown", "value"),
        Input("db-value-dropdown", "value"),
        Input("grid-n", "value"),
        Input("n-estimators", "value"),
    )
    def _render(source_type, run_name, db_name, db_value, gn, ntrees):
        try:
            if source_type == "run":
                if not run_name:
                    return no_update, "No run selected."
                X, Y, labels, title = load_run_source(_resolve_run_dir(run_name), None)
                value_name = "Objective"
            else:
                if not db_name or not db_value:
                    return no_update, "No database / value column selected."
                X, Y, labels, title = load_db_dataset(_resolve_db_path(db_name), db_value)
                value_name = db_value
            fig = build_ternary_figure(
                X, Y, labels,
                grid_n=int(gn), n_estimators=int(ntrees),
                title=title, value_name=value_name,
            )
            status = (
                f"{X.shape[0]} points\n"
                f"Y range: [{Y.min():.4f}, {Y.max():.4f}]\n"
                f"corners (BL, BR, top): {labels}"
            )
            return fig, status
        except Exception as e:  # surface load/plot errors in the UI
            return no_update, f"Error: {e}"

    return app


# ── static export (legacy matplotlib path) ────────────────────────────────────

def export_png(args: argparse.Namespace) -> None:
    """Render a static PNG using matplotlib (the original behaviour)."""
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.db:
        db_path = _resolve_db_path(args.db)
        X, Y, labels, title = load_db_dataset(db_path, args.value)
        out_default = db_path.with_suffix(".png")
        value_name = args.value
    else:
        run_dir = _resolve_run_dir(args.run)
        X, Y, labels, title = load_run_source(
            run_dir, args.snapshot, args.corner_dims, args.labels)
        out_default = run_dir / "run_ternary.png"
        value_name = "Objective"

    print(f"Title  : {title}")
    print(f"Labels : {labels}")
    print(f"Points : {X.shape[0]}   Y range: [{Y.min():.4f}, {Y.max():.4f}]")

    grid_pts, grid_vals = fit_rf_background(X, Y, args.grid_n, args.n_estimators)
    vmin = float(min(grid_vals.min(), Y.min()))
    vmax = float(max(grid_vals.max(), Y.max()))
    if vmax <= vmin:
        vmax = vmin + 1e-9

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, _SQRT3_2 + 0.16)
    ax.axis("off")
    ax.text(-0.04, -0.04, labels[0], ha="right", va="top", fontsize=9)
    ax.text(1.04, -0.04, labels[1], ha="left", va="top", fontsize=9)
    ax.text(0.5, _SQRT3_2 + 0.04, labels[2], ha="center", va="bottom", fontsize=9)
    ax.set_title(title, fontsize=11)

    gxy = comp_to_xy(grid_pts)
    ax.scatter(gxy[:, 0], gxy[:, 1], c=grid_vals, cmap="viridis",
               vmin=vmin, vmax=vmax, s=8, alpha=0.80, zorder=2, rasterized=True)
    pxy = comp_to_xy(X)
    sc = ax.scatter(pxy[:, 0], pxy[:, 1], c=Y, cmap="viridis", vmin=vmin, vmax=vmax,
                    s=30, alpha=0.95, zorder=4, edgecolors="black", linewidths=0.9)
    fig.colorbar(sc, ax=ax, label=value_name, fraction=0.046, pad=0.04)
    fig.tight_layout()

    out = Path(args.out) if args.out else out_default
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {out}")
    if args.show:
        plt.show()


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive Dash app (or static PNG) for a ZoMBI-Hop run's "
                    "datapoints on an RF-interpolated ternary."
    )
    parser.add_argument("--port", type=int, default=8050,
                        help="Port for the Dash app (default: 8050).")
    parser.add_argument("--export", action="store_true",
                        help="Render a static PNG instead of launching the app.")
    parser.add_argument("--run", default=DEFAULT_RUN,
                        help="Run directory or bare run name (export; default: run_7eb9).")
    parser.add_argument("--db", default=None,
                        help="Database file or bare name (export; overrides --run).")
    parser.add_argument("--value", default=DEFAULT_DB_VALUE,
                        help="DB value column to plot (default: Objective).")
    parser.add_argument("--snapshot", default=None,
                        help="Snapshot to reconstruct up to (default: latest.txt).")
    parser.add_argument("--out", default=None, help="Output PNG path (export).")
    parser.add_argument("--grid-n", type=int, default=TERNARY_GRID_N,
                        help="Ternary grid resolution for the RF background.")
    parser.add_argument("--n-estimators", type=int, default=RF_N_ESTIMATORS,
                        help="Number of trees in the RF surrogate.")
    parser.add_argument("--corner-dims", default=None,
                        help="Comma-separated run dims placed at "
                             "[bottom-left,bottom-right,top] (default: 9,8,0).")
    parser.add_argument("--labels", default=None,
                        help="Comma-separated corner labels for "
                             "[bottom-left,bottom-right,top].")
    parser.add_argument("--show", action="store_true",
                        help="(export only) Display the matplotlib figure as well.")
    args = parser.parse_args()

    if args.export:
        export_png(args)
        return

    app = build_app(grid_n=args.grid_n, n_estimators=args.n_estimators)
    print(f"Dash app running at http://127.0.0.1:{args.port}")
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
