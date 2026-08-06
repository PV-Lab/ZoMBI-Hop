"""
visualization/plot_run.py
=========================
Interactive Dash app (with an optional static-PNG export mode) that plots every
datapoint collected during a *real* ZoMBI-Hop run on a simplex diagram, over a
Random-Forest-interpolated background of those same points.

Both d=3 and d=4 runs are supported. A d=3 (3-component) composition lives on a
triangle, so it is drawn as the usual **ternary** diagram. A d=4 (4-component)
composition lives on a 3-simplex, so it is drawn as a 3D **quaternary**
tetrahedron: the four components are the four corners, every measured point sits
inside the solid, and the optional interpolated background fills the volume.

Two data sources are supported, selectable in the app:

  * **Run directory** (e.g. ``runs/run_7eb9``) — the full measured dataset
    ``(X, Y)`` is reconstructed from the run's delta snapshots
    (``reconstruct_snapshot_tensors``).
  * **Data file** (e.g. ``data/2nd_real_run.db`` or ``data/campaign1a.csv``) —
    the ``results`` table (``.db``) or the campaign table (``.csv``) is read
    directly; the three composition columns (FAPbI3, MAPbI3, MAPbBr3) form ``X``
    and a chosen value column (default ``Objective``) forms ``Y``. The format is
    detected automatically from the file extension.

For whichever source, the app:
  1. Trains a Random-Forest surrogate on ``(X, Y)`` and evaluates it over a
     dense ternary grid — the smooth background heatmap (viridis).
  2. Overlays every collected datapoint on top, each coloured by its measured
     value (same viridis scale as the background) with a black outline so
     individual measurements stay legible.

The diagram type (ternary vs. tetrahedron) is chosen automatically from the
number of composition columns; d=3 and d=4 are supported.

Usage
-----
  conda activate zombi-hop
  python visualization/plot_run.py                      # launch the Dash app
  python visualization/plot_run.py --port 8051          # app on a custom port

  # Static PNG export (legacy behaviour), no server:
  python visualization/plot_run.py --export --run runs/run_7eb9
  python visualization/plot_run.py --export --db data/2nd_real_run.db --out db.png
  python visualization/plot_run.py --export --db data/campaign1a.csv --out csv.png

Flags
-----
  --port N          Port for the Dash app (default: 8050).
  --export          Render a static PNG instead of launching the app.
  --run PATH        Run directory for --export (default: runs/run_7eb9).
  --db PATH         Data file (.db or .csv) for --export (overrides --run).
  --value COL       DB value column to plot as the objective (default: Objective).
  --snapshot NAME   Snapshot to reconstruct up to (default: latest.txt).
  --out PATH        Output PNG path for --export.
  --grid-n N        Ternary grid resolution for the background (default: 200).
  --n-estimators N  Number of trees in the RF surrogate (default: 500).
  --background M    Background surrogate: rf, gp, or none (default: rf).
  --no-points       (export only) Hide the measured sampled points.
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

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
TERNARY_GRID_N = 200
# A d=4 grid has O(n^3) points (vs O(n^2) for a triangle), and every point is a
# 3D marker that must be depth-composited, so the tetrahedron uses a much coarser
# default grid to stay responsive.
TETRA_GRID_N = 28
# Hard cap on the d=4 grid resolution (the Dash slider is shared with the 2D
# ternary, which allows up to 400; a tetrahedron at that resolution is millions
# of points). simplex_grid_4d(44) ≈ 16k interior points.
TETRA_GRID_MAX_N = 44
_SQRT3_2 = np.sqrt(3) / 2

# Vertices of a regular (unit-edge) tetrahedron. Composition column i is mapped
# to corner TETRA_VERTS[i]; a (N,4) simplex point becomes ``comp @ TETRA_VERTS``.
TETRA_VERTS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, _SQRT3_2, 0.0],
    [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3],
], dtype=float)
# The six edges of the tetrahedron, as index pairs into TETRA_VERTS.
TETRA_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# Physical identity of each hardware dim, and which ternary corner it occupies.
# CORNER_DIM_ORDER lists dims for [bottom-left, bottom-right, top]; the measured
# composition columns (ordered by the run's optimizing dims) are reindexed to
# match so each labelled component lands in the requested corner.
DIM_LABELS: dict[int, str] = {0: "FAPbI3", 9: "MAPbBr3", 8: "MAPbI3", 2: "CsPbI3"}
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


def comp_to_xyz(comp: np.ndarray) -> np.ndarray:
    """(N, 4) simplex compositions → (N, 3) Cartesian tetrahedron coordinates.

    Corner mapping:  comp[:, i] → TETRA_VERTS[i].  Rows are renormalised to sum 1.
    """
    p = np.asarray(comp, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    s = p.sum(axis=-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return p @ TETRA_VERTS


def simplex_grid_4d(n: int) -> np.ndarray:
    """Return (N, 4) uniform grid on the 3-simplex (4-component probabilities)."""
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            for k in range(n + 1 - i - j):
                pts.append([i / n, j / n, k / n, (n - i - j - k) / n])
    return np.array(pts, dtype=float)


def simplex_grid(n: int, d: int) -> np.ndarray:
    """Uniform grid on the (d-1)-simplex for d in {3, 4}."""
    if d == 3:
        return ternary_grid(n)
    if d == 4:
        return simplex_grid_4d(n)
    raise ValueError(f"Only d=3 or d=4 simplex grids are supported (got d={d}).")


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
    placed at each simplex corner (or None if the run dims are unknown / don't
    match), and ``labels`` are the matching corner labels. Works for both d=3
    (triangle: [bottom-left, bottom-right, top]) and d=4 (tetrahedron corners).
    """
    run_dims = _run_dims(run_dir)
    n_dims = len(run_dims) if run_dims else None

    # Corner dim ordering: CLI override, else — only for the d=3 physical layout —
    # the default corner order when all its dims are present, else the run's own
    # column order.
    if corner_dims_override:
        corner_dims = [int(x) for x in corner_dims_override.split(",") if x.strip() != ""]
        if n_dims is not None and len(corner_dims) != n_dims:
            raise ValueError(
                f"--corner-dims must list {n_dims} comma-separated dim indices "
                f"for this d={n_dims} run"
            )
    elif run_dims is not None and n_dims == 3 and all(d in run_dims for d in CORNER_DIM_ORDER):
        corner_dims = list(CORNER_DIM_ORDER)
    else:
        corner_dims = list(run_dims) if run_dims else None

    # Labels: CLI override, else the physical DIM_LABELS map, else "dim N".
    if labels_override:
        parts = [s.strip() for s in labels_override.split(",")]
        if corner_dims is not None and len(parts) != len(corner_dims):
            raise ValueError(f"--labels must list {len(corner_dims)} comma-separated names")
        labels = tuple(parts)
    elif corner_dims is not None:
        labels = tuple(DIM_LABELS.get(d, f"dim {d}") for d in corner_dims)
    else:
        labels = tuple("ABCD"[: n_dims or 3])

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
    if X.shape[1] not in (3, 4):
        raise ValueError(f"Only d=3 or d=4 runs are supported (got d={X.shape[1]}).")
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


# ── data loading: database / csv ──────────────────────────────────────────────

def _is_csv(path: Path) -> bool:
    """Whether ``path`` should be read as a CSV (vs a SQLite .db)."""
    return path.suffix.lower() == ".csv"


def _resolve_db_path(db_arg: str) -> Path:
    """Accept a full path, or a bare .db/.csv name resolved against data/."""
    p = Path(db_arg)
    if p.is_file():
        return p.resolve()
    candidate = DATA_DIR / db_arg
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Data file not found: {db_arg}")


def db_value_columns(db_path: Path) -> list[str]:
    """Candidate value columns present (with data) in a .db or .csv data file."""
    if _is_csv(db_path):
        import pandas as pd

        df = pd.read_csv(db_path, usecols=lambda c: c in set(DB_VALUE_COLS))
        cols = [c for c in DB_VALUE_COLS if c in df.columns and df[c].notna().any()]
        return cols or [c for c in DB_VALUE_COLS if c in df.columns]

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


def _load_csv_rows(csv_path: Path, cols: list[str]) -> np.ndarray:
    """Read ``cols`` from a campaign CSV, dropping rows with any null in them."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Columns {missing} not found in {csv_path}")
    df = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    return df.to_numpy(dtype=float)


def _load_db_rows(db_path: Path, cols: list[str]) -> np.ndarray:
    """Read ``cols`` from a .db results table, dropping rows with any null."""
    con = sqlite3.connect(str(db_path))
    try:
        sel = ", ".join(f'"{c}"' for c in cols)
        where = " AND ".join(f'"{c}" IS NOT NULL' for c in cols)
        rows = con.execute(f"SELECT {sel} FROM results WHERE {where}").fetchall()
    finally:
        con.close()
    return np.asarray(rows, dtype=float)


def load_db_dataset(
    db_path: Path, value_col: str = DEFAULT_DB_VALUE
) -> tuple[np.ndarray, np.ndarray, tuple[str, str, str], str]:
    """Read a .db ``results`` table or campaign .csv → ``(X (N,3), Y (N,), labels, title)``.

    The format is chosen from the file extension. Rows missing the value column
    or any composition column are dropped, and composition rows are renormalised
    to sum 1.
    """
    cols = DB_COMP_COLS + [value_col]
    arr = _load_csv_rows(db_path, cols) if _is_csv(db_path) else _load_db_rows(db_path, cols)
    if arr.shape[0] == 0:
        raise RuntimeError(f"No rows with non-null {DB_COMP_COLS} + {value_col} in {db_path}")
    X = arr[:, :3]
    Y = arr[:, 3]
    s = X.sum(axis=1, keepdims=True)
    X = X / np.where(s == 0, 1.0, s)
    labels = (DB_COMP_COLS[0], DB_COMP_COLS[1], DB_COMP_COLS[2])
    title = f"{db_path.name} — {X.shape[0]} measured points  ({value_col})"
    return X, Y, labels, title


# ── background surrogates ────────────────────────────────────────────────────

BACKGROUND_MODES: tuple[str, ...] = ("rf", "gp", "none")


def fit_rf_background(
    X: np.ndarray, Y: np.ndarray, grid_n: int, n_estimators: int
) -> tuple[np.ndarray, np.ndarray]:
    """Fit an RF surrogate on (X, Y) and evaluate it over a ternary grid.

    Returns ``(grid_pts (M,3), grid_vals (M,))``.
    """
    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42)
    rf.fit(X, Y)
    grid_pts = simplex_grid(grid_n, X.shape[1])
    grid_vals = rf.predict(grid_pts)
    return grid_pts, grid_vals


def fit_gp_background(
    X: np.ndarray, Y: np.ndarray, grid_n: int, length_scale: float = 0.3
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Gaussian-Process surrogate on (X, Y) and evaluate over the grid.

    A Matern(nu=2.5) kernel plus a white-noise term gives a smooth interpolation
    across the simplex. ``length_scale`` is held fixed so it directly controls
    smoothness (smaller = more local/wiggly, larger = smoother). Y is
    standardised for numerical stability and mapped back to the original scale.

    Returns ``(grid_pts (M,3), grid_vals (M,))``.
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
    grid_pts = simplex_grid(grid_n, X.shape[1])
    grid_vals = gp.predict(grid_pts) * y_std + y_mean
    return grid_pts, grid_vals


def fit_background(
    X: np.ndarray, Y: np.ndarray, grid_n: int, n_estimators: int, mode: str,
    *, length_scale: float = 0.3,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Dispatch to the requested background surrogate.

    ``mode`` is one of ``"rf"``, ``"gp"``, ``"none"``. Returns ``None`` when no
    background is requested, else ``(grid_pts, grid_vals)``.
    """
    if mode == "none":
        return None
    if mode == "gp":
        return fit_gp_background(X, Y, grid_n, length_scale=length_scale)
    return fit_rf_background(X, Y, grid_n, n_estimators)


# ── plotly figure ─────────────────────────────────────────────────────────────

_BG_LABELS: dict[str, str] = {
    "rf": "RF-interpolated background",
    "gp": "GP-interpolated background",
    "none": "no background",
}


def _color_limits(
    grid_vals: np.ndarray | None, Y: np.ndarray, lo_pct: float = 10.0, hi_pct: float = 90.0
) -> tuple[float, float]:
    """Colorbar (vmin, vmax) from the ``lo_pct``/``hi_pct`` percentiles of the data.

    Using the 10th/90th percentiles instead of the raw min/max clips the extreme
    tails, so the viridis range is spent on the bulk of the values and smaller
    changes are easier to see. Percentiles are taken over the background grid and
    the measured points together (or just the points when there is no background).
    """
    vals = Y if grid_vals is None else np.concatenate([np.ravel(grid_vals), np.ravel(Y)])
    vmin = float(np.percentile(vals, lo_pct))
    vmax = float(np.percentile(vals, hi_pct))
    if vmax <= vmin:
        vmax = vmin + 1e-9
    return vmin, vmax


def _dropped_title(label: str, scale: float) -> str:
    """Axis title text preceded by a blank spacer line, to push it further down.

    Plotly fixes the ternary axis-title offset at ``tickfont size + ticklen + 3``
    below the tick numbers (see ``drawAxes`` in plotly.js) and exposes no
    standoff, so the two bottom corner titles collide with the corner tick
    labels. A leading blank line whose font size we control adds exactly the
    extra gap we want, and it scales with the rest of the fonts.
    """
    gap = max(int(round(12 * scale)), 1)
    return f"<span style='font-size:{gap}px'> </span><br>{label}"


def _bg_marker_size(grid_n: int) -> float:
    """Marker size that keeps the background grid visually gap-free as grid_n grows.

    The grid spacing on the plotted ternary shrinks like 1/grid_n, so the marker
    size is scaled the same way (clamped). Sized to slightly overlap so there is
    no white space between neighbouring background points.
    """
    return float(np.clip(1100.0 / max(grid_n, 1), 3.0, 16.0))


def build_ternary_figure(
    X: np.ndarray,
    Y: np.ndarray,
    labels: tuple[str, str, str],
    *,
    grid_n: int,
    n_estimators: int,
    title: str,
    value_name: str = "Objective",
    background: str = "rf",
    show_points: bool = True,
    gp_length_scale: float = 0.3,
    scale: float = 1.0,
    plot_size: float = 0.80,
    color_limits: tuple[float, float] | None = None,
):
    """Interactive Plotly ternary: optional interpolated background + points overlay.

    ``background`` selects the surrogate heatmap: ``"rf"``, ``"gp"``, or ``"none"``.
    ``show_points`` toggles the measured-datapoint overlay.

    ``scale`` multiplies the measured-point marker size, corner-label fonts and
    the objective colorbar fonts (the triangle geometry itself is unchanged).

    ``plot_size`` is the fraction of the figure width given to the triangle; the
    colorbar is parked just past it. Lowering it shrinks the triangle relative
    to the colorbar (the d=4 tetrahedron gets the same effect from zooming).

    ``color_limits`` forces the viridis (vmin, vmax); pass it when two figures
    must share one colour scale, else it is derived from this figure's data.

    Corner mapping mirrors comp_to_xy / matplotlib:
      col0 → bottom-left (b),  col1 → bottom-right (c),  col2 → top (a).
    """
    import plotly.graph_objects as go

    bg = fit_background(X, Y, grid_n, n_estimators, background, length_scale=gp_length_scale)

    if bg is not None:
        grid_pts, grid_vals = bg
    else:
        grid_pts = grid_vals = None
    vmin, vmax = color_limits if color_limits is not None else _color_limits(grid_vals, Y)

    fig = go.Figure()

    # Colorbar rides on whichever trace is present (background if points hidden).
    # Pulled in toward the plot (x) and enlarged (thicker + taller, bigger fonts).
    colorbar = dict(
        title=dict(text=value_name, font=dict(size=14 * scale)),
        thickness=28, len=0.9, x=min(plot_size + 0.06, 0.98), xpad=0,
        tickfont=dict(size=14 * scale),
    )

    # Background: interpolated heatmap rendered as a dense marker grid.
    if bg is not None:
        fig.add_trace(go.Scatterternary(
            a=grid_pts[:, 2], b=grid_pts[:, 0], c=grid_pts[:, 1],
            mode="markers",
            marker=dict(
                symbol="circle", size=_bg_marker_size(grid_n), color=grid_vals,
                colorscale="Viridis", cmin=vmin, cmax=vmax,
                opacity=0.80, line=dict(width=0),
                colorbar=None if show_points else colorbar,
            ),
            hoverinfo="skip",
            name=f"{background.upper()} background",
            showlegend=False,
        ))

    # Foreground: every measured datapoint, coloured by value with a black edge.
    if show_points:
        customdata = np.column_stack([X[:, 0], X[:, 1], X[:, 2], Y])
        fig.add_trace(go.Scatterternary(
            a=X[:, 2], b=X[:, 0], c=X[:, 1],
            mode="markers",
            marker=dict(
                symbol="circle", size=9 * scale, color=Y,
                colorscale="Viridis", cmin=vmin, cmax=vmax,
                line=dict(width=1.0, color="black"),
                colorbar=colorbar,
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

    # Ternary axes: bigger corner-label + tick fonts, and a visible triangle
    # outline (linecolor/linewidth) that shows even when there is no background.
    axis_common = dict(
        min=0, ticks="outside",
        title_font=dict(size=16 * scale),
        tickfont=dict(size=14 * scale),
        linecolor="black", linewidth=2, showline=True,
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, y=0.98, yanchor="top", font=dict(size=17)),
        font=dict(size=15),
        ternary=dict(
            sum=1,
            aaxis=dict(title=labels[2], **axis_common),                    # top
            baxis=dict(title=_dropped_title(labels[0], scale), **axis_common),  # bottom-left
            caxis=dict(title=_dropped_title(labels[1], scale), **axis_common),  # bottom-right
            bgcolor="white",
            # Triangle occupies this slice of the figure; the colorbar sits just
            # past its right edge, so shrinking the domain shrinks the plot
            # relative to the colorbar.
            domain=dict(x=[0.0, plot_size], y=[0.0, 1.0]),
        ),
        # Bottom margin has to hold the tick numbers, the spacer line and the
        # corner titles, all of which scale with `scale`.
        margin=dict(l=int(40 + 40 * scale), r=40, t=90, b=int(60 + 45 * scale)),
        height=720,
    )
    return fig


def _tetra_bg_marker_size(grid_n: int) -> float:
    """Marker size for the 3D background grid; smaller than 2D so points show through."""
    return float(np.clip(260.0 / max(grid_n, 1), 2.0, 8.0))


def build_quaternary_figure(
    X: np.ndarray,
    Y: np.ndarray,
    labels: tuple[str, ...],
    *,
    grid_n: int,
    n_estimators: int,
    title: str,
    value_name: str = "Objective",
    background: str = "rf",
    show_points: bool = True,
    gp_length_scale: float = 0.3,
    scale: float = 1.0,
    color_limits: tuple[float, float] | None = None,
):
    """Interactive Plotly 3D tetrahedron for d=4 compositions.

    The four composition columns are the four tetrahedron corners (column i →
    ``TETRA_VERTS[i]``). Every measured point sits inside the solid, and the
    optional interpolated background fills the interior as a translucent 3D grid
    (its markers are kept small and semi-transparent so the opaque, black-edged
    measured points stay legible through the volume).
    """
    import plotly.graph_objects as go

    # A d=4 grid grows as O(grid_n^3), so clamp it well below the 2D ternary range
    # (the UI slider is shared) to keep the fit + 3D render responsive.
    grid_n = int(min(grid_n, TETRA_GRID_MAX_N))
    bg = fit_background(X, Y, grid_n, n_estimators, background, length_scale=gp_length_scale)

    if bg is not None:
        grid_pts, grid_vals = bg
    else:
        grid_pts = grid_vals = None
    vmin, vmax = color_limits if color_limits is not None else _color_limits(grid_vals, Y)

    colorbar = dict(
        title=dict(text=value_name, font=dict(size=14 * scale)),
        thickness=28, len=0.9, x=0.86, xpad=0,
        tickfont=dict(size=14 * scale),
    )

    fig = go.Figure()

    # Tetrahedron wireframe: all six edges as a single line trace with None gaps.
    ex, ey, ez = [], [], []
    for i, j in TETRA_EDGES:
        ex += [TETRA_VERTS[i, 0], TETRA_VERTS[j, 0], None]
        ey += [TETRA_VERTS[i, 1], TETRA_VERTS[j, 1], None]
        ez += [TETRA_VERTS[i, 2], TETRA_VERTS[j, 2], None]
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode="lines",
        line=dict(color="black", width=3),
        hoverinfo="skip", showlegend=False,
    ))

    # Corner labels at each vertex, nudged outward from the centroid.
    centroid = TETRA_VERTS.mean(axis=0)
    lbl = TETRA_VERTS + 0.12 * (TETRA_VERTS - centroid)
    fig.add_trace(go.Scatter3d(
        x=lbl[:, 0], y=lbl[:, 1], z=lbl[:, 2], mode="text",
        text=list(labels[:4]),
        textfont=dict(size=14 * scale, color="black"),
        hoverinfo="skip", showlegend=False,
    ))

    # Background: translucent interior grid coloured by the surrogate.
    if bg is not None:
        gxyz = comp_to_xyz(grid_pts)
        fig.add_trace(go.Scatter3d(
            x=gxyz[:, 0], y=gxyz[:, 1], z=gxyz[:, 2], mode="markers",
            marker=dict(
                size=_tetra_bg_marker_size(grid_n), color=grid_vals,
                colorscale="Viridis", cmin=vmin, cmax=vmax,
                opacity=0.28, line=dict(width=0),
                colorbar=None if show_points else colorbar,
            ),
            hoverinfo="skip", name=f"{background.upper()} background", showlegend=False,
        ))

    # Foreground: every measured datapoint, coloured by value with a black edge.
    if show_points:
        pxyz = comp_to_xyz(X)
        customdata = np.column_stack([X, Y])
        hover = "<br>".join(
            f"{labels[k]}=%{{customdata[{k}]:.3f}}" for k in range(X.shape[1])
        ) + f"<br>{value_name}=%{{customdata[{X.shape[1]}]:.4f}}<extra></extra>"
        fig.add_trace(go.Scatter3d(
            x=pxyz[:, 0], y=pxyz[:, 1], z=pxyz[:, 2], mode="markers",
            marker=dict(
                size=5 * scale, color=Y,
                colorscale="Viridis", cmin=vmin, cmax=vmax,
                line=dict(width=1.0, color="black"),
                colorbar=colorbar,
            ),
            customdata=customdata, hovertemplate=hover,
            name="measured", showlegend=False,
        ))

    # Explicit *cube* scene box centered on the top vertex's vertical axis.
    #
    # The apex (TETRA_VERTS[3], e.g. MAPbBr3) sits directly above the base; we
    # want the auto-rotation to spin about the vertical line through it. Plotly's
    # camera orbits about scene-normalized (0,0,0), which maps to the centre of
    # the axis ranges — so we build equal-length ranges centred on
    # (apex_x, apex_y, mid_z). That puts the orbit axis on the apex, keeps all
    # three axes at one scale (so the shape is undistorted and the spin stays
    # circular, not wobbly/elliptical), and pads the box so nothing clips on zoom.
    apex = TETRA_VERTS[3]
    box_pts = np.vstack([TETRA_VERTS, lbl])          # verts + outward corner labels
    half_xy = float(np.abs(box_pts[:, :2] - apex[:2]).max())
    z_lo, z_hi = float(box_pts[:, 2].min()), float(box_pts[:, 2].max())
    half = max(half_xy, (z_hi - z_lo) / 2.0) * 1.08  # 8% padding to avoid clipping
    cx, cy, cz = apex[0], apex[1], (z_lo + z_hi) / 2.0

    def _axis_range(center: float) -> list[float]:
        return [center - half, center + half]

    hidden_axis = dict(
        showbackground=False, showgrid=False, zeroline=False,
        showticklabels=False, title="", visible=False,
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, y=0.98, yanchor="top", font=dict(size=17)),
        font=dict(size=15),
        scene=dict(
            xaxis=dict(range=_axis_range(cx), **hidden_axis),
            yaxis=dict(range=_axis_range(cy), **hidden_axis),
            zaxis=dict(range=_axis_range(cz), **hidden_axis),
            # Equal ranges + cube => uniform scale on every axis, so the tetra is
            # undistorted and the vertical-axis spin is a true circle.
            aspectmode="cube",
            camera=dict(eye=dict(x=1.35, y=1.35, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=90, b=10),
        autosize=True,   # fill the 90vh graph container rather than a fixed height
    )
    return fig


def build_figure(X: np.ndarray, Y: np.ndarray, labels: tuple[str, ...], **kwargs):
    """Dispatch to the ternary (d=3) or quaternary tetrahedron (d=4) builder."""
    d = X.shape[1]
    # plot_size only applies to the flat ternary; the d=4 scene is sized by the
    # camera, so zooming already covers it.
    plot_size = kwargs.pop("plot_size", None)
    if d == 3:
        if plot_size is not None:
            kwargs["plot_size"] = plot_size
        return build_ternary_figure(X, Y, labels, **kwargs)
    if d == 4:
        return build_quaternary_figure(X, Y, labels, **kwargs)
    raise ValueError(f"Only d=3 or d=4 is supported for plotting (got d={d}).")


# ── Dash app ──────────────────────────────────────────────────────────────────

def _list_run_dirs() -> list[str]:
    """Names of run directories that contain snapshots, newest first."""
    if not RUNS_DIR.is_dir():
        return []
    runs = [p for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "snapshots").is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in runs]


def _list_db_files() -> list[str]:
    """Names of .db and .csv data files under data/."""
    if not DATA_DIR.is_dir():
        return []
    files = list(DATA_DIR.glob("*.db")) + list(DATA_DIR.glob("*.csv"))
    return sorted(p.name for p in files)


def build_app(grid_n: int = TERNARY_GRID_N, n_estimators: int = RF_N_ESTIMATORS):
    """Construct the Dash application."""
    from dash import Dash, dcc, html, Input, Output, State, no_update

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

    # update_title=None stops Dash flashing "Updating..." in the tab on every
    # rotation tick (the interval fires ~20x/s while auto-rotating).
    app = Dash(__name__, update_title=None)
    app.title = "ZoMBI-Hop Run Plot"

    app.layout = html.Div(style={"display": "flex", "fontFamily": "system-ui, sans-serif"}, children=[
        html.Div(style=panel_style, children=[
            html.H3("ZoMBI-Hop ternary", style={"marginTop": 0}),

            html.Label("Source", style=label_style),
            dcc.RadioItems(
                id="source-type",
                options=[
                    {"label": " Run directory", "value": "run"},
                    {"label": " Data file (.db/.csv)", "value": "db"},
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
                html.Label("Data file", style=label_style),
                dcc.Dropdown(
                    id="db-dropdown",
                    options=[{"label": n, "value": n} for n in db_names],
                    value=default_db, clearable=False,
                ),
                html.Label("Value column", style=label_style),
                dcc.Dropdown(id="db-value-dropdown", clearable=False),
            ]),

            html.Label("Background", style=label_style),
            dcc.RadioItems(
                id="background-mode",
                options=[
                    {"label": " Random Forest", "value": "rf"},
                    {"label": " Gaussian Process", "value": "gp"},
                    {"label": " None", "value": "none"},
                ],
                value="rf",
                labelStyle={"display": "block"},
            ),

            dcc.Checklist(
                id="show-points",
                options=[{"label": " Show sampled points", "value": "show"}],
                value=["show"],
                style={"marginTop": "12px"},
            ),

            html.Label("Grid resolution", style=label_style),
            dcc.Slider(
                id="grid-n", min=40, max=400, step=20, value=grid_n,
                marks={40: "40", 120: "120", 240: "240", 400: "400"},
            ),

            html.Label("RF trees", style=label_style),
            dcc.Slider(
                id="n-estimators", min=100, max=800, step=100, value=n_estimators,
                marks={100: "100", 500: "500", 800: "800"},
            ),

            html.Label("GP length scale", style=label_style),
            dcc.Slider(
                id="gp-length-scale", min=0.05, max=1.0, step=0.05, value=0.3,
                marks={0.05: "0.05", 0.3: "0.3", 0.6: "0.6", 1.0: "1.0"},
            ),

            html.Label("Scale (dots & labels)", style=label_style),
            dcc.Slider(
                id="scale", min=0.5, max=3.0, step=0.1, value=1.0,
                marks={0.5: "0.5", 1.0: "1", 2.0: "2", 3.0: "3"},
            ),

            # d=3 only: triangle width vs the colorbar. The d=4 tetrahedron is
            # sized by the scene camera, so scroll-zoom already does this.
            html.Label("Plot size vs colorbar (d=3)", style=label_style),
            dcc.Slider(
                id="plot-size", min=0.4, max=0.95, step=0.05, value=0.80,
                marks={0.4: "small", 0.7: "0.7", 0.95: "large"},
            ),

            # Color-scale override: when checked, the viridis (vmin, vmax) is
            # taken from the two inputs below instead of the data percentiles.
            dcc.Checklist(
                id="color-override",
                options=[{"label": " Override color scale", "value": "on"}],
                value=[],
                style={"marginTop": "12px"},
            ),
            html.Div(
                style={"display": "flex", "gap": "8px", "marginTop": "6px"},
                children=[
                    html.Div(children=[
                        html.Label("Min", style={"fontSize": "12px", "color": "#666"}),
                        dcc.Input(
                            id="color-min", type="number", debounce=True,
                            style={"width": "100%", "boxSizing": "border-box"},
                        ),
                    ], style={"flex": 1}),
                    html.Div(children=[
                        html.Label("Max", style={"fontSize": "12px", "color": "#666"}),
                        dcc.Input(
                            id="color-max", type="number", debounce=True,
                            style={"width": "100%", "boxSizing": "border-box"},
                        ),
                    ], style={"flex": 1}),
                ],
            ),

            # Auto-rotate controls (only affect the 3D d=4 tetrahedron; a no-op
            # for the flat d=3 ternary, which has no scene camera).
            html.Label("3D auto-rotate", style=label_style),
            html.Button(
                "Start rotation", id="rotate-toggle", n_clicks=0,
                style={"width": "100%", "padding": "8px", "cursor": "pointer"},
            ),
            html.Label("Rotation speed", style=label_style),
            dcc.Slider(
                id="rotate-speed", min=0.2, max=5.0, step=0.2, value=1.0,
                marks={0.2: "slow", 1.0: "1", 3.0: "3", 5.0: "fast"},
            ),
            dcc.Interval(id="rotate-interval", interval=50, disabled=True),
            dcc.Store(id="rotate-on", data=False),

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

    # Start/stop auto-rotation: flip the stored on/off state, relabel the button,
    # and enable/disable the interval that drives the camera spin.
    @app.callback(
        Output("rotate-on", "data"),
        Output("rotate-toggle", "children"),
        Output("rotate-interval", "disabled"),
        Input("rotate-toggle", "n_clicks"),
    )
    def _toggle_rotate(n_clicks):
        rotating = bool(n_clicks) and n_clicks % 2 == 1
        return rotating, ("Stop rotation" if rotating else "Start rotation"), not rotating

    # Client-side camera spin: each interval tick rotates the 3D scene camera's
    # eye about the vertical axis by an angle set by the speed slider. Runs only
    # when a scene camera exists (i.e. the d=4 tetrahedron), a no-op otherwise.
    app.clientside_callback(
        """
        function(n_intervals, rotating, speed) {
            const no = window.dash_clientside.no_update;
            if (!rotating) { return no; }
            // The element carrying the Dash id is a wrapper; the real Plotly
            // graph div (with _fullLayout / relayout) is the .js-plotly-plot child.
            const outer = document.getElementById('ternary-graph');
            if (!outer) { return no; }
            const gd = outer.querySelector('.js-plotly-plot') || outer;
            const fl = gd._fullLayout;
            // No 3D scene => flat d=3 ternary, nothing to rotate.
            if (!fl || !fl.scene) { return no; }
            // Prefer the *live* gl3d camera (getCamera) so an in-progress
            // scroll-zoom is reflected immediately; fall back to the layout copy,
            // which only updates when an interaction ends.
            const sc = fl.scene._scene;
            let eye;
            if (sc && typeof sc.getCamera === 'function') {
                eye = sc.getCamera().eye;
            } else if (fl.scene.camera) {
                eye = fl.scene.camera.eye;
            }
            if (!eye) { return no; }
            // eye may be {x,y,z} or an [x,y,z] array depending on Plotly version.
            const ex = Array.isArray(eye) ? eye[0] : eye.x;
            const ey = Array.isArray(eye) ? eye[1] : eye.y;
            const ez = Array.isArray(eye) ? eye[2] : eye.z;
            const ang = (speed || 1.0) * 0.01;
            const cos = Math.cos(ang), sin = Math.sin(ang);
            const x = ex * cos - ey * sin;
            const y = ex * sin + ey * cos;
            window.Plotly.relayout(gd, {'scene.camera.eye': {x: x, y: y, z: ez}});
            return no;
        }
        """,
        Output("rotate-on", "data", allow_duplicate=True),
        Input("rotate-interval", "n_intervals"),
        State("rotate-on", "data"),
        State("rotate-speed", "value"),
        prevent_initial_call=True,
    )

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
        Input("background-mode", "value"),
        Input("show-points", "value"),
        Input("grid-n", "value"),
        Input("n-estimators", "value"),
        Input("gp-length-scale", "value"),
        Input("scale", "value"),
        Input("plot-size", "value"),
        Input("color-override", "value"),
        Input("color-min", "value"),
        Input("color-max", "value"),
    )
    def _render(source_type, run_name, db_name, db_value, background, show_points,
                gn, ntrees, gp_ls, scale, plot_size, color_override, color_min,
                color_max):
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

            # Manual color-scale override: only applied when the box is checked
            # and both bounds are valid (min < max); otherwise fall back to the
            # data-derived percentile limits inside build_figure.
            color_limits = None
            if color_override and color_min is not None and color_max is not None:
                lo, hi = float(color_min), float(color_max)
                if hi > lo:
                    color_limits = (lo, hi)

            fig = build_figure(
                X, Y, labels,
                grid_n=int(gn), n_estimators=int(ntrees),
                title=title, value_name=value_name,
                background=background, show_points=bool(show_points),
                gp_length_scale=float(gp_ls), scale=float(scale),
                plot_size=float(plot_size), color_limits=color_limits,
            )
            diagram = "tetrahedron (d=4)" if X.shape[1] == 4 else "ternary (d=3)"
            status = (
                f"{X.shape[0]} points\n"
                f"Y range: [{Y.min():.4f}, {Y.max():.4f}]\n"
                f"diagram: {diagram}\n"
                f"background: {background}\n"
                f"corners: {labels}"
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

    d = X.shape[1]
    print(f"Title  : {title}")
    print(f"Labels : {labels}")
    print(f"Diagram: {'tetrahedron (d=4)' if d == 4 else 'ternary (d=3)'}")
    print(f"Points : {X.shape[0]}   Y range: [{Y.min():.4f}, {Y.max():.4f}]")

    # d=4 grids grow as O(n^3); clamp as the interactive builder does.
    grid_n = min(args.grid_n, TETRA_GRID_MAX_N) if d == 4 else args.grid_n
    bg = fit_background(X, Y, grid_n, args.n_estimators, args.background,
                        length_scale=args.gp_length_scale)
    if bg is not None:
        grid_pts, grid_vals = bg
    else:
        grid_pts = grid_vals = None
    vmin, vmax = _color_limits(grid_vals, Y)

    if d == 4:
        fig = _export_tetra_png(
            X, Y, labels, title, value_name, grid_pts, grid_vals,
            vmin, vmax, grid_n, args.no_points, plt)
        out = Path(args.out) if args.out else (out_default.with_name("run_tetra.png")
                                               if not args.db else out_default)
    else:
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

        # Marker size scales inversely with grid resolution so denser grids stay
        # gap-free without blotting; mirrors the Plotly background sizing.
        mappable = None
        if bg is not None:
            gxy = comp_to_xy(grid_pts)
            bg_s = float(np.clip(6000.0 / max(grid_n, 1), 2.0, 40.0))
            mappable = ax.scatter(gxy[:, 0], gxy[:, 1], c=grid_vals, cmap="viridis",
                                  vmin=vmin, vmax=vmax, s=bg_s, alpha=0.80, zorder=2,
                                  rasterized=True)
        if not args.no_points:
            pxy = comp_to_xy(X)
            mappable = ax.scatter(pxy[:, 0], pxy[:, 1], c=Y, cmap="viridis",
                                  vmin=vmin, vmax=vmax, s=30, alpha=0.95, zorder=4,
                                  edgecolors="black", linewidths=0.9)
        if mappable is not None:
            fig.colorbar(mappable, ax=ax, label=value_name, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = Path(args.out) if args.out else out_default

    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved figure -> {out}")
    if args.show:
        plt.show()


def _export_tetra_png(X, Y, labels, title, value_name, grid_pts, grid_vals,
                      vmin, vmax, grid_n, no_points, plt):
    """Render a static 3D tetrahedron PNG for a d=4 run (matplotlib mplot3d)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    fig = plt.figure(figsize=(8.6, 7.6))
    ax = fig.add_subplot(111, projection="3d")

    # Tetrahedron wireframe.
    for i, j in TETRA_EDGES:
        ax.plot(*zip(TETRA_VERTS[i], TETRA_VERTS[j]), color="black", lw=1.2)

    # Corner labels, nudged outward from the centroid.
    centroid = TETRA_VERTS.mean(axis=0)
    for v, name in zip(TETRA_VERTS, labels[:4]):
        p = v + 0.12 * (v - centroid)
        ax.text(p[0], p[1], p[2], name, ha="center", va="center", fontsize=10)

    mappable = None
    if grid_pts is not None:
        gxyz = comp_to_xyz(grid_pts)
        bg_s = float(np.clip(1200.0 / max(grid_n, 1), 2.0, 20.0))
        mappable = ax.scatter(gxyz[:, 0], gxyz[:, 1], gxyz[:, 2], c=grid_vals,
                              cmap="viridis", vmin=vmin, vmax=vmax, s=bg_s,
                              alpha=0.12, linewidths=0)
    if not no_points:
        pxyz = comp_to_xyz(X)
        mappable = ax.scatter(pxyz[:, 0], pxyz[:, 1], pxyz[:, 2], c=Y, cmap="viridis",
                              vmin=vmin, vmax=vmax, s=26, alpha=0.98,
                              edgecolors="black", linewidths=0.6, depthshade=False)
    if mappable is not None:
        fig.colorbar(mappable, ax=ax, label=value_name, fraction=0.03, pad=0.02)

    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    fig.tight_layout()
    return fig


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
                        help="Data file (.db or .csv) or bare name (export; overrides --run).")
    parser.add_argument("--value", default=DEFAULT_DB_VALUE,
                        help="DB value column to plot (default: Objective).")
    parser.add_argument("--snapshot", default=None,
                        help="Snapshot to reconstruct up to (default: latest.txt).")
    parser.add_argument("--out", default=None, help="Output PNG path (export).")
    parser.add_argument("--grid-n", type=int, default=TERNARY_GRID_N,
                        help="Ternary grid resolution for the interpolated background.")
    parser.add_argument("--n-estimators", type=int, default=RF_N_ESTIMATORS,
                        help="Number of trees in the RF surrogate.")
    parser.add_argument("--background", choices=BACKGROUND_MODES, default="rf",
                        help="Background surrogate: rf, gp, or none (default: rf).")
    parser.add_argument("--gp-length-scale", type=float, default=0.3,
                        help="Fixed Matern length scale for the GP background (default: 0.3).")
    parser.add_argument("--no-points", action="store_true",
                        help="(export only) Hide the measured sampled points.")
    parser.add_argument("--corner-dims", default=None,
                        help="Comma-separated run dims placed at the simplex "
                             "corners (d=3: [bottom-left,bottom-right,top], "
                             "default 9,8,0; d=4: the four tetrahedron corners).")
    parser.add_argument("--labels", default=None,
                        help="Comma-separated corner labels, one per composition "
                             "column (3 for a ternary run, 4 for a tetrahedron run).")
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
