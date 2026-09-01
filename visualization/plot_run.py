"""
visualization/plot_run.py
=========================
Interactive Dash app (with an optional static-PNG export mode) that plots every
datapoint collected during a *real* ZoMBI-Hop run on a simplex diagram, over a
Random-Forest-interpolated background of those same points.

Both the **constraint** and the **dimensionality** are detected from the data, and
the diagram follows from the pair — the data is never projected down, or onto a
constraint it does not have, to fit a fixed diagram.

*Simplex* sources (compositions: rows sum to 1) get the composition diagrams:

  * **d=3** — a 3-component composition lives on a triangle, drawn as the usual
    **ternary** diagram over an interpolated background.
  * **d=4** — a 4-component composition lives on a 3-simplex, drawn as a 3D
    **quaternary** tetrahedron: the four components are the four corners, every
    measured point sits inside the solid, and the background fills the volume.

*Non-simplex* sources (a box of independent physical parameters, each with its
own unit and range) get plain axes in those units instead — putting them on a
triangle would assert a sum-to-one relationship they do not have:

  * **d=2** — a **heatmap** of the surrogate with the measured points over it.
  * **d=3** — a **3D scatter** in the parameter box, ticked and labelled, with a
    translucent voxel cloud of the surrogate filling the volume.
  * **d=4** — a **scatter-plot matrix**, every axis pair coloured by the
    objective. A 4D box, unlike a 4-component simplex, has no sum-to-one
    constraint to collapse it into a solid, so there is no faithful 3D embedding
    to draw. Deliberately no surrogate background: see ``build_splom_figure``.

  * **d>=5** — in *either* family, 5+ dimensions have no faithful 2D/3D
    embedding, so the run is drawn as a **CoNet** (``plot_10d``'s
    co-occurrence-network UMAP map, the same view ``conet.png`` shows). This is
    the plain single-dataset CoNet; the uniform baseline of ``paired_conet.py``
    is not involved. Non-simplex inputs are **unit-scaled before embedding**
    (``Dataset.embed_X``) — on raw ``hplc`` units one column would otherwise be
    94% of every row and the map would describe that column alone.

Three data sources are supported, selectable in the app:

  * **Run directory** (e.g. ``runs/run_7eb9``) — the full measured dataset
    ``(X, Y)`` is reconstructed from the run's delta snapshots
    (``reconstruct_snapshot_tensors``).
  * **Data file** (e.g. ``data/6d.db`` or ``data/campaign1a.csv``) — the
    ``results`` table (``.db``) or the campaign table (``.csv``) is read
    directly. The composition columns are **discovered** rather than hard-coded
    (see ``detect_comp_columns``), so ``3d.db``/``4d.db``/``6d.db`` each plot at
    their own dimensionality, and a chosen value column (default ``Objective``)
    forms ``Y``. The format is detected from the file extension.
  * **Public dataset** (e.g. ``hplc``) — an Olympus dataset cached under
    ``benchmarks/public_db/data`` by ``benchmarks/public_db/olympus.py``. These
    are the only sources that are not necessarily compositions and not
    necessarily maximised, so they are what carry ``simplex``/``bounds``/``goal``
    onto ``Dataset``. Of the four curated ones, ``photo_pce10`` and
    ``photo_wf3`` are 4-component simplices (tetrahedron), ``crossed_barrel`` is
    a 4-parameter box (scatter-plot matrix) and ``hplc`` is a 6-parameter box
    (CoNet). ``crossed_barrel`` against the ``photo_*`` pair is the clearest
    case for detecting the constraint rather than assuming it: same ``d=4``,
    different diagram entirely.

Where a background applies (every diagram but the d=4 scatter-plot matrix and the
CoNet) the app trains a Random-Forest (or GP) surrogate on ``(X, Y)``, evaluates
it over a dense grid — a simplex grid for a composition, a box grid in the
columns' own units otherwise — and overlays every measured datapoint coloured by
its value on the same viridis scale, with a black outline so individual
measurements stay legible.

Two optional panels, each behind a checkbox in the left panel:

  * **Time slider** — replays the run in *line* increments. A line is one
    deposition line of (nominally) 24 points; because some points are culled, the
    lines are not all 24 long, so the boundaries are taken from the data's own
    ``Iteration`` column rather than assumed (see ``line_index``). The slider runs
    from 0 lines (an empty plot) on the left to every line on the right. On the
    simplex diagrams each step rings the points its newest line added in red, so
    what that step contributed is visible at a glance.
  * **Convergence plot** — a panel below the main diagram in the style of
    ``optimize/run_mobo.py``'s ``plot_convergence``: every observed Y against
    sample index, the running-best step envelope (reset at each activation), and
    a crimson dashed rule at each declared needle. With the time slider on, it
    fills in alongside the main plot and keeps fixed axes, so points appear
    rather than the whole plot rescaling.

For the CoNet + time slider, the UMAP map is fitted **once on the full dataset**
and every intermediate step replays that fitted frame (``build_conet(frame=...)``)
on its prefix of the points, so a sample sits at exactly the same coordinates at
every step instead of the map re-fitting and shifting under the animation.

Slider steps are **precomputed**, not rendered on demand: selecting a source with
the slider on starts a background pass that renders every step of the range,
starting from the step on screen, and what it stores is the finished artefact —
the PNG the browser displays, or the serialised figure Dash sends it. It all
lives on disk, so a range survives closing the app, and CoNet steps are served
as plain image URLs rather than inlined into the callback. Scrubbing then costs
a file read. See "precomputed time-slider steps" below.

Usage
-----
  .venv/Scripts/python.exe visualization/plot_run.py    # launch the Dash app
  python visualization/plot_run.py --port 8051          # app on a custom port

  # Static PNG export (legacy behaviour), no server:
  python visualization/plot_run.py --export --run runs/run_7eb9
  python visualization/plot_run.py --export --db data/3d.db --out db.png
  python visualization/plot_run.py --export --db data/6d.db --out conet.png
  python visualization/plot_run.py --export --db data/campaign1a.csv --out csv.png

  # Public Olympus datasets (fetch them first):
  python benchmarks/public_db/olympus.py --fetch all
  python visualization/plot_run.py --export --public photo_pce10     # tetrahedron
  python visualization/plot_run.py --export --public crossed_barrel  # SPLOM
  python visualization/plot_run.py --export --public hplc            # CoNet

Flags
-----
  --port N          Port for the Dash app (default: 8050).
  --export          Render a static PNG instead of launching the app.
  --run PATH        Run directory for --export (default: runs/run_7eb9).
  --db PATH         Data file (.db or .csv) for --export (overrides --run).
  --public NAME     Public Olympus dataset for --export (overrides --db/--run):
                    photo_pce10, photo_wf3, hplc, crossed_barrel, or any other
                    name cached under benchmarks/public_db/data.
  --value COL       DB value column to plot as the objective (default: Objective).
  --snapshot NAME   Snapshot to reconstruct up to (default: latest.txt).
  --out PATH        Output PNG path for --export.
  --grid-n N        Grid resolution for the background (default: 200). Capped
                    per diagram: a simplex/box grid grows as O(n^(d-1))/O(n^d),
                    so d>=3 clamps far below the 2D range.
  --n-estimators N  Number of trees in the RF surrogate (default: 500).
  --background M    Background surrogate: rf, gp, or none (default: rf).
  --no-points       (export only) Hide the measured sampled points.
  --corner-dims D   Run dims placed at [bottom-left,bottom-right,top]
                    (default: 9,8,0 — dim 9 bottom-left, dim 0 at the top).
  --labels A,B,C    Corner labels for [bottom-left,bottom-right,top]
                    (default: FAPbI3,MAPbI3,MAPbBr3).
  --show            (export only) Display the matplotlib figure as well as saving.
  --cache-dir PATH  Where rendered slider steps are cached (kept between runs).
  --no-precompute   Render time-slider steps on demand instead of ahead of time.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace as _dc_replace
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

# ── project root (and this dir, for `plot_10d`) on sys.path ────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

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

DB_VALUE_COLS: list[str] = ["Objective", "Bandgap", "Photoconductance", "Stability"]
DEFAULT_DB_VALUE = "Objective"

# The 3-component layout of the original real-run campaign, ordered [bottom-left,
# bottom-right, top] to match comp_to_xy. Plotting no longer assumes it — the
# composition columns are detected per file (``detect_comp_columns``) so a d=4 or
# d=6 file is not squashed into three — but ``input_noise.py`` analyses that
# specific campaign and imports this triple, so it stays as the d=3 default.
DB_COMP_COLS: list[str] = ["FAPbI3", "MAPbI3", "MAPbBr3"]

# A ``results`` table lays its ten hardware module slots out contiguously between
# the ``Iteration`` and ``X`` columns. A slot loaded with a real precursor carries
# that precursor's name (FAPbI3, CsPbI3, MACl, …); an empty slot keeps its
# placeholder name ``ModuleN`` and is all zeros. Detecting the composition columns
# this way — rather than hard-coding one fixed triple — is what lets 3d.db, 4d.db
# and 6d.db each plot at their own true dimensionality.
_COMP_SPAN = ("Iteration", "X")
_MODULE_RE = re.compile(r"module\d+$", re.IGNORECASE)

# d>=5 has no faithful simplex diagram, so it is drawn as a CoNet instead. The
# same threshold applies to non-simplex sources: d=2 gets real axes, d=3 a 3-D
# box, d=4 a scatter-plot matrix, and d>=5 the CoNet.
CONET_MIN_D = 5

# Total points allowed in a non-simplex box background grid. A box grid is n**d,
# so this caps the product rather than the per-axis resolution (see _box_grid_n):
# ~360x360 at d=2, ~50^3 at d=3.
BOX_GRID_MAX_PTS = 130_000

# Per-axis cap for the d=3 box background specifically. Every grid point is a
# translucent 3-D marker the browser must depth-composite, which bites well before
# the fit does — the same reason TETRA_GRID_N is far below the ternary's.
BOX3D_GRID_MAX_N = 30

# Above this many samples the d=4 scatter-plot matrix draws smaller, more
# transparent markers — a SPLOM puts every point in every one of its d*(d-1)
# panels, so a few thousand samples becomes a solid block without it.
SPLOM_DENSE_N = 400

# Nominal points per deposition line. Only a fallback: run directories carry no
# ``Iteration`` column, so their lines are assumed to be exactly this long. Data
# files use their real ``Iteration`` values, which is what handles culled
# (shorter-than-nominal) lines correctly — see ``line_index``.
NOMINAL_LINE_POINTS = 24


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


# ── the loaded dataset ────────────────────────────────────────────────────────

@dataclass
class Dataset:
    """One loaded source, ready to plot, whatever its dimensionality.

    ``X`` is (N, d) row-normalised composition, ``Y`` is (N,) the measured value,
    and ``labels`` names the d components. ``lines`` is the per-point deposition
    line id (0-based, ascending, not necessarily contiguous in length — see
    ``line_index``); the time slider steps through it one line at a time.

    ``activations`` is the per-point ZoMBI activation id, or None when the source
    does not record one. It is optimizer state, so only run directories have it
    (recovered from their snapshots); a ``.db``/``.csv`` is a hardware result log
    with no such column. The convergence plot resets its running-best envelope at
    each activation boundary when it is present — see ``run_activations``.

    ``needles`` is an ``(n, 2)`` array of ``[sample index, points collected when
    declared]`` for every needle the run declared, or None when the source keeps no
    needle record (again, ``.db``/``.csv``). The convergence plot rules a line at
    each. The two columns differ because a needle is declared *at the best point so
    far*, which is usually an earlier sample than the moment of declaration — see
    ``run_needles``.

    ``simplex`` says whether ``X`` lives on a simplex (rows summing to 1) or in a
    plain box of independent parameters. It is what selects the *family* of
    diagram, before dimensionality selects the member: a composition goes on a
    ternary/tetrahedron, a box of unrelated physical parameters (ml, ml/min, Hz,
    s) goes on ordinary axes in its own units. Run directories and the
    ``.db``/``.csv`` result logs are all compositions, so it defaults True; the
    public Olympus datasets set it from their upstream ``constraints`` field.

    ``bounds`` is the (d, 2) ``[low, high]`` per column for non-simplex sources —
    the declared design-space extent, which fixes the axis ranges and the
    background grid so those do not float with whatever subset is on screen. It
    is None for simplex sources, which get their extent from the simplex itself.

    ``goal`` is ``"minimize"`` or ``"maximize"``. The convergence panel's
    best-so-far envelope follows it; a hard-coded ``max`` would draw a rising
    curve for a dataset whose objective is a degradation to be minimised.
    """

    X: np.ndarray
    Y: np.ndarray
    labels: tuple[str, ...]
    title: str
    lines: np.ndarray
    value_name: str = DEFAULT_DB_VALUE
    activations: np.ndarray | None = None
    needles: np.ndarray | None = None
    simplex: bool = True
    bounds: np.ndarray | None = None
    goal: str = "maximize"

    def __iter__(self):
        """Unpack as the legacy ``(X, Y, labels, title)`` tuple.

        ``load_db_dataset`` / ``load_run_source`` used to return that 4-tuple, and
        several sibling modules unpack it that way (``needle_overlay``,
        ``random_baseline``, ``plot_warm_start``, ``optimize/evaluate``,
        ``warm_start/test_greedy_optima_gp``). Keeping the unpacking contract means
        the richer Dataset can carry ``lines``/``value_name`` for the time slider
        without any of them having to change.
        """
        return iter((self.X, self.Y, self.labels, self.title))

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    @property
    def axis_bounds(self) -> np.ndarray:
        """(d, 2) plotting extent per column, always defined.

        The declared ``bounds`` when the source has them, else the observed
        column range padded by 5% so points do not sit on the frame. A degenerate
        column (every row identical) is widened to a unit interval, since a
        zero-width axis range is not renderable.
        """
        if self.bounds is not None and len(self.bounds) == self.d:
            return np.asarray(self.bounds, dtype=float)
        lo = self.X.min(axis=0) if len(self.X) else np.zeros(self.d)
        hi = self.X.max(axis=0) if len(self.X) else np.ones(self.d)
        pad = np.where(hi > lo, (hi - lo) * 0.05, 0.5)
        return np.column_stack([lo - pad, hi + pad])

    @property
    def unit_X(self) -> np.ndarray:
        """``X`` rescaled into [0, 1] per column against ``axis_bounds``.

        Anything that measures a *distance* between samples must use this rather
        than ``X`` when the source is not a simplex. A simplex is already
        commensurate — every column is a fraction of the same whole — but a box
        of physical parameters is not: ``hplc`` mixes millilitres (0-0.08) with
        hertz (80-150), so on raw units the Euclidean distance between two
        samples is, to three decimal places, the difference in ``push_speed``
        alone, and any embedding built on it describes that one column.
        """
        b = self.axis_bounds
        lo, hi = b[:, 0], b[:, 1]
        span = np.where(hi > lo, hi - lo, 1.0)
        return np.clip((self.X - lo) / span, 0.0, 1.0)

    @property
    def embed_X(self) -> np.ndarray:
        """The coordinates a distance-based view (the CoNet) should embed.

        A simplex passes its compositions through untouched; a box is unit-scaled
        first (see ``unit_X``). ``plot_10d._reduce_comp`` row-normalises whatever
        it is handed, which is meaningful for a composition and meaningless for a
        box of mixed units — unit-scaling first at least makes every column
        contribute on comparable terms before that happens.
        """
        return self.X if self.simplex else self.unit_X

    @property
    def n_lines(self) -> int:
        return int(self.lines.max()) + 1 if len(self.lines) else 0

    def prefix_rows(self, n_lines: int) -> np.ndarray:
        """Row indices belonging to the first ``n_lines`` deposition lines.

        Returned as explicit indices rather than a count: the rows of a line are
        contiguous in every file seen so far, but nothing in the format guarantees
        it, and a CoNet step selects its embedding rows by exactly this index array
        (``E_raw[rows]``). Indices keep that correct either way, where a bare count
        would silently mis-pair points with coordinates.
        """
        return np.flatnonzero(self.lines < int(n_lines))

    def prefix(self, n_lines: int) -> "Dataset":
        """This dataset restricted to its first ``n_lines`` deposition lines."""
        rows = self.prefix_rows(n_lines)
        return _dc_replace(
            self, X=self.X[rows], Y=self.Y[rows], lines=self.lines[rows],
            activations=None if self.activations is None else self.activations[rows],
            needles=self.needles_within(rows),
        )

    def needles_within(self, rows: np.ndarray) -> np.ndarray | None:
        """Needles already declared by ``rows``, re-indexed into that selection.

        Both conditions matter: the needle's own sample must be in the selection,
        *and* it must have been declared by then. A needle sits at the best point
        found so far, so its sample can be drawn long before the optimizer decided
        it was a needle; showing it early would put a needle on the plot at a step
        where the run had not found one.
        """
        if self.needles is None:
            return None
        pos = np.full(len(self.Y), -1, dtype=int)
        pos[rows] = np.arange(len(rows))
        keep = [(pos[i], when) for i, when in self.needles
                if 0 <= i < len(pos) and pos[i] >= 0 and when <= len(rows)]
        return np.asarray(keep, dtype=int).reshape(-1, 2)


def line_index(iteration: np.ndarray | None, n: int) -> np.ndarray:
    """Per-point deposition-line id, 0-based and ascending.

    A line is nominally ``NOMINAL_LINE_POINTS`` points, but the pipeline culls
    points, so real lines run shorter (in ``data/6d.db`` they range from 14 to 24).
    Slicing at a fixed stride would therefore drift out of phase with the actual
    lines within a few steps. When the source carries an ``Iteration`` column that
    column *is* the line marker, so it is used directly (densified to 0..L-1 so the
    slider maps onto contiguous steps). Only when there is no such column — a run
    directory reconstructed from snapshots — do we fall back to fixed-size blocks.
    """
    if iteration is None:
        return np.arange(n, dtype=int) // NOMINAL_LINE_POINTS
    _, dense = np.unique(np.asarray(iteration, dtype=float), return_inverse=True)
    return dense.astype(int)


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


def run_activations(run_dir: Path, snapshot: str, n_points: int) -> np.ndarray | None:
    """Per-point ZoMBI activation id for a run, or None if it cannot be recovered.

    Each snapshot's ``summary.json`` records the activation that was running and the
    cumulative ``n_points`` at that moment, which is exactly the ``(n_points,
    activation)`` record stream ``run_mobo._activation_zoom_per_point`` buckets: the
    points in ``[prev_n, n)`` were measured under that snapshot's activation. Walking
    the snapshots up to and including ``snapshot`` therefore reconstructs the
    activation of every stored point, which is what lets the convergence plot restart
    its running-best envelope at each activation boundary.

    Returns None when there are no readable snapshots, so callers fall back to a
    single global running best rather than inventing boundaries.
    """
    snaps_dir = run_dir / "snapshots"
    if not snaps_dir.is_dir():
        return None
    records: list[tuple[int, int]] = []
    for snap in sorted(p for p in snaps_dir.iterdir() if p.is_dir()):
        try:
            s = json.loads((snap / "summary.json").read_text())
            records.append((int(s["n_points"]), int(s["activation"])))
        except Exception:
            continue                      # a partial/unreadable snapshot is skipped
        if snap.name == snapshot:
            break                         # the view is reconstructed only this far
    if not records:
        return None

    act = np.zeros(int(n_points), dtype=int)
    prev = 0
    for n, a in records:
        n = min(n, n_points)
        if n > prev:
            act[prev:n] = a
            prev = n
    if prev < n_points:                   # tail: points past the last snapshot
        act[prev:] = records[-1][1]
    return act


def _needle_sample(rec: dict, X: np.ndarray, Y: np.ndarray) -> int | None:
    """Index of the sample a needle record names, or None if it cannot be pinned.

    A needle is declared *at* one of the collected points, so its recorded
    composition should appear verbatim in ``X`` — matching on it recovers the
    sample index ``run_mobo.plot_convergence`` rules its line at. The recorded
    objective value is the fallback for a run whose needle was stored at a
    different dimensionality than the reconstructed X.
    """
    point = np.asarray(rec.get("point") or [], dtype=float)
    if point.size == X.shape[1] and point.sum() > 0:
        dist = np.abs(X - point / point.sum()).max(axis=1)
    else:
        value = rec.get("value")
        if value is None:
            return None
        dist = np.abs(Y - float(value))
    i = int(np.argmin(dist))
    # Only an essentially exact hit counts: a near miss means the record does not
    # correspond to a stored point, and a line drawn on the wrong sample would be
    # worse than no line at all.
    return i if dist[i] <= 1e-6 else None


def run_needles(run_dir: Path, snapshot: str, X: np.ndarray,
                Y: np.ndarray) -> np.ndarray | None:
    """``(n, 2)`` array of ``[sample index, points collected when declared]``.

    Each snapshot carries the cumulative ``needles.json`` list, so a needle is
    "declared" at the first snapshot whose list grew to include it — that
    snapshot's ``n_points`` is when the run knew about it. The needle's own sample
    index is recovered separately (``_needle_sample``), because the two are not the
    same: a needle is declared at the best point found so far, which is typically
    an earlier sample.

    Returns an empty ``(0, 2)`` array for a run that declared no needles, and None
    when there is no snapshot record at all, so callers can tell "none found" apart
    from "not recorded" (a ``.db``/``.csv`` result log has no needle record).
    """
    snaps_dir = run_dir / "snapshots"
    if not snaps_dir.is_dir():
        return None
    records: list[dict] = []
    declared: list[int] = []
    for snap in sorted(p for p in snaps_dir.iterdir() if p.is_dir()):
        try:
            n_points = int(json.loads((snap / "summary.json").read_text())["n_points"])
            found = json.loads((snap / "needles.json").read_text())
        except Exception:
            continue                      # a partial/unreadable snapshot is skipped
        while len(records) < len(found):  # everything this snapshot newly declared
            records.append(found[len(records)])
            declared.append(n_points)
        if snap.name == snapshot:
            break                         # the view is reconstructed only this far

    out = []
    for rec, when in zip(records, declared):
        i = _needle_sample(rec, X, Y)
        if i is not None:
            out.append((i, min(when, len(Y))))
    return np.asarray(out, dtype=int).reshape(-1, 2)


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
    """Reconstruct the full measured dataset ``(X (N,d), Y (N,))`` for a run.

    Any d is returned; the caller picks the diagram from it (d>=5 becomes a CoNet).
    """
    tensors = reconstruct_snapshot_tensors(run_dir, snapshot, device="cpu")
    X = tensors.get("X_all_actual")
    Y = tensors.get("Y_all")
    if X is None or Y is None or X.shape[0] == 0:
        raise RuntimeError(f"No datapoints reconstructed from {run_dir}/{snapshot}")
    X = X.detach().cpu().numpy().astype(float)
    Y = Y.detach().cpu().numpy().astype(float).ravel()
    # Normalise rows to sum 1 (guards against tiny numerical drift).
    s = X.sum(axis=1, keepdims=True)
    X = X / np.where(s == 0, 1.0, s)
    return X, Y


def load_run_source(
    run_dir: Path,
    snapshot: str | None,
    corner_dims_override: str | None = None,
    labels_override: str | None = None,
) -> Dataset:
    """Load a run directory into a ``Dataset`` ready for plotting."""
    snapshot = snapshot or _default_snapshot(run_dir)
    run_dims = _run_dims(run_dir)
    corner_dims, labels = _corner_config(run_dir, corner_dims_override, labels_override)
    X, Y = load_run_dataset(run_dir, snapshot)
    # Matched before the columns are reordered for the diagram: needles.json
    # records a composition in the run's own dimension order.
    needles = run_needles(run_dir, snapshot, X, Y)
    X = _reorder_columns(X, run_dims, corner_dims)
    if len(labels) != X.shape[1]:      # d>=5: _corner_config's triple-oriented default
        dims = corner_dims or run_dims or list(range(X.shape[1]))
        labels = tuple(DIM_LABELS.get(d, f"dim {d}") for d in dims[: X.shape[1]])
    return Dataset(
        X=X, Y=Y, labels=tuple(labels),
        title=f"{run_dir.name} — collected datapoints  ({snapshot})",
        # Snapshots carry no Iteration column, so lines fall back to fixed blocks.
        lines=line_index(None, len(X)),
        activations=run_activations(run_dir, snapshot, len(X)),
        needles=needles,
    )


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


def _table_columns(path: Path) -> list[str]:
    """Column names of a data file's table, in declaration order."""
    if _is_csv(path):
        import pandas as pd

        return list(pd.read_csv(path, nrows=0).columns)
    con = sqlite3.connect(str(path))
    try:
        return [r[1] for r in con.execute("PRAGMA table_info(results)")]
    finally:
        con.close()


def _nonzero_columns(path: Path, cols: list[str]) -> list[str]:
    """Those of ``cols`` that hold at least one non-zero, non-null value."""
    if _is_csv(path):
        import pandas as pd

        df = pd.read_csv(path, usecols=lambda c: c in set(cols))
        return [c for c in cols
                if c in df.columns
                and pd.to_numeric(df[c], errors="coerce").fillna(0.0).abs().max() > 0]
    con = sqlite3.connect(str(path))
    try:
        keep = []
        for c in cols:
            hi = con.execute(
                f'SELECT MAX(ABS("{c}")) FROM results WHERE "{c}" IS NOT NULL'
            ).fetchone()[0]
            if hi is not None and float(hi) > 0:
                keep.append(c)
        return keep
    finally:
        con.close()


def detect_comp_columns(path: Path) -> list[str]:
    """The composition columns actually used by a data file, in table order.

    This is what gives the app its dimensionality: the length of the returned list
    IS ``d``, so a 6-component file is plotted as 6 components rather than being
    squeezed into a fixed three.

    The candidates are the hardware module slots, which sit contiguously between
    the ``Iteration`` and ``X`` columns (see ``_COMP_SPAN``). Two filters narrow
    them to the components a campaign really varied:

      * slots still carrying their ``ModuleN`` placeholder name were never loaded
        with a precursor, so they are not components at all;
      * slots that are all-zero across every row contribute nothing to any
        composition — keeping them would inflate d with a constant column and, on
        the simplex, add a corner no sample ever approaches.

    A campaign CSV has no module slots; its span between ``Iteration`` and ``X``
    is already exactly the composition columns, and the same filters are no-ops.
    """
    cols = _table_columns(path)
    lo, hi = _COMP_SPAN
    if lo not in cols or hi not in cols:
        raise RuntimeError(
            f"{path.name}: expected the composition columns between "
            f"'{lo}' and '{hi}', but the table has no such span."
        )
    span = cols[cols.index(lo) + 1: cols.index(hi)]
    named = [c for c in span if not _MODULE_RE.match(c)]
    active = _nonzero_columns(path, named)
    if len(active) < 2:
        raise RuntimeError(
            f"{path.name}: found {len(active)} varying composition column(s) "
            f"{active} among {named}; need at least 2 to plot."
        )
    return active


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


def load_db_dataset(db_path: Path, value_col: str = DEFAULT_DB_VALUE) -> Dataset:
    """Read a .db ``results`` table or campaign .csv into a ``Dataset``.

    The format is chosen from the file extension and the width from
    ``detect_comp_columns``, so d is whatever the file actually contains. Rows
    missing the value column or any composition column are dropped, and
    composition rows are renormalised to sum 1. The ``Iteration`` column is read
    alongside so the time slider can step by real deposition lines.
    """
    comp_cols = detect_comp_columns(db_path)
    d = len(comp_cols)
    cols = comp_cols + [value_col, _COMP_SPAN[0]]
    arr = _load_csv_rows(db_path, cols) if _is_csv(db_path) else _load_db_rows(db_path, cols)
    if arr.shape[0] == 0:
        raise RuntimeError(f"No rows with non-null {comp_cols} + {value_col} in {db_path}")
    X, Y, iteration = arr[:, :d], arr[:, d], arr[:, d + 1]
    s = X.sum(axis=1, keepdims=True)
    X = X / np.where(s == 0, 1.0, s)
    lines = line_index(iteration, len(X))
    return Dataset(
        X=X, Y=Y, labels=tuple(comp_cols),
        title=(f"{db_path.name} — {X.shape[0]} measured points, d={d}, "
               f"{int(lines.max()) + 1 if len(lines) else 0} lines  ({value_col})"),
        lines=lines, value_name=value_col,
    )


# ── data loading: public datasets (Olympus) ───────────────────────────────────

#: Preference order for the app's default public dataset. Falls back to whatever
#: is cached; kept here so plot_run does not import the benchmarks package just to
#: pick a default when the package is absent.
CURATED_PUBLIC: tuple[str, ...] = ("photo_pce10", "photo_wf3", "hplc",
                                  "crossed_barrel")


def _list_public_datasets() -> list[str]:
    """Names of the public datasets cached under ``benchmarks/public_db/data``.

    Returns [] if the package is missing rather than raising, so the app still
    starts (with the public source simply offering nothing) on a checkout where
    the datasets were never fetched.
    """
    try:
        from benchmarks.public_db import available
    except Exception:
        return []
    try:
        return available()
    except Exception:
        return []


def load_public_dataset(name: str) -> Dataset:
    """Load an Olympus dataset from ``benchmarks/public_db`` into a ``Dataset``.

    Carries three things across that the run/``.db`` sources never have to think
    about, because for them they are constant:

      * ``simplex`` — whether the parameters are a composition. The ``photo_*``
        sets are (4-component, so they draw as a tetrahedron); ``hplc`` is not
        (6 independent process parameters, so it draws as a CoNet), and neither
        is ``crossed_barrel`` (4 independent geometry parameters, so it draws as
        a scatter-plot matrix rather than as the tetrahedron its ``d`` alone
        would suggest).
      * ``bounds`` — the declared design-space extent per column, which fixes the
        non-simplex axis ranges to the space that was searched rather than to the
        part of it that happened to be sampled.
      * ``goal`` — ``photo_*`` minimise degradation while ``hplc`` maximises peak
        area and ``crossed_barrel`` maximises toughness, and the convergence
        envelope follows it.

    ``lines`` is a **pseudo**-deposition-line index: these are published datasets,
    not ZoMBI-Hop campaigns, so there is no deposition-line column and no
    guarantee that file order is acquisition order (for the ``photo_*`` grids it
    is certainly a design enumeration instead). Rows are chunked into blocks of
    ``NOMINAL_LINE_POINTS`` in file order purely so the time slider has something
    to step through; it replays the file, and should not be read as a campaign
    unfolding.
    """
    from benchmarks.public_db import load as _load_public

    src = _load_public(name)
    X = np.asarray(src.X, dtype=float)
    Y = np.asarray(src.Y, dtype=float)
    lines = np.arange(len(X)) // NOMINAL_LINE_POINTS
    kind = "simplex" if src.simplex else "box"
    return Dataset(
        X=X, Y=Y, labels=tuple(src.labels),
        title=(f"{src.name} — {len(X)} points, d={src.d} ({kind}), "
               f"{src.goal} {src.target_name}"),
        lines=lines, value_name=src.target_name,
        simplex=src.simplex, bounds=np.asarray(src.bounds, dtype=float),
        goal=src.goal,
    )


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


def box_grid(n: int, bounds: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Regular ``n**d`` lattice filling the box ``bounds`` ((d, 2) low/high).

    Returns ``(pts (n**d, d), axes)`` where ``axes[i]`` is the 1-D tick vector of
    column i. The axes come back alongside the points because the 2-D background
    is drawn as a ``Heatmap``, which wants the two vectors and a rectangular
    ``z``, not a flat list of coordinates.

    Built with ``indexing="ij"``, so reshaping values to ``(n,) * d`` puts column
    0 on the first axis.
    """
    axes = [np.linspace(lo, hi, n) for lo, hi in np.asarray(bounds, dtype=float)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([m.ravel() for m in mesh]), axes


def _box_grid_n(grid_n: int, d: int) -> int:
    """Per-axis resolution for a d-dimensional box grid, capped by total points.

    A box grid is ``n**d`` points, so the UI's single resolution slider (which
    goes to 400, sensible for a 2-D heatmap) has to mean something much smaller
    at d=3, where 400 would be 64 million points to fit a surrogate over and
    depth-composite. The cap is on the product, mirroring how ``TETRA_GRID_MAX_N``
    keeps the d=4 simplex grid tractable.
    """
    n = max(int(grid_n), 2)
    if d <= 1:
        return min(n, BOX_GRID_MAX_PTS)
    return max(2, min(n, int(BOX_GRID_MAX_PTS ** (1.0 / d))))


def fit_box_background(
    X: np.ndarray, Y: np.ndarray, grid_n: int, n_estimators: int, mode: str,
    bounds: np.ndarray, *, length_scale: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]] | None:
    """``fit_background``'s non-simplex counterpart: a surrogate over a box grid.

    The surrogate is fitted on **unit-scaled** inputs and asked for predictions at
    unit-scaled grid points, while the grid points are *returned* in the columns'
    own physical units. That split matters for the GP: a single isotropic Matern
    length scale is only meaningful when every axis is commensurate, and on raw
    ``hplc`` units (ml against Hz) one fixed length scale is simultaneously far
    too wide for one column and far too narrow for another. The RF is invariant
    to the rescaling, so it is unaffected either way.

    Returns ``(grid_pts (M, d), grid_vals (M,), axes)``, or None for ``"none"``.
    """
    if mode == "none":
        return None
    b = np.asarray(bounds, dtype=float)
    n = _box_grid_n(grid_n, X.shape[1])
    grid_pts, axes = box_grid(n, b)

    lo, hi = b[:, 0], b[:, 1]
    span = np.where(hi > lo, hi - lo, 1.0)
    Xu = (X - lo) / span
    Gu = (grid_pts - lo) / span

    if mode == "gp":
        y_mean, y_std = float(Y.mean()), float(Y.std()) or 1.0
        kernel = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=length_scale, length_scale_bounds="fixed", nu=2.5)
            + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-6, 1e1))
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=False, n_restarts_optimizer=2, random_state=42
        )
        gp.fit(Xu, (Y - y_mean) / y_std)
        vals = gp.predict(Gu) * y_std + y_mean
    else:
        rf = RandomForestRegressor(
            n_estimators=n_estimators, n_jobs=-1, random_state=42)
        rf.fit(Xu, Y)
        vals = rf.predict(Gu)
    return grid_pts, vals, axes


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


# The time slider's newest deposition line is ringed in this colour so the points
# that step added stand out against the black-edged points already on the plot.
NEW_POINT_COLOR = "red"
NEW_POINT_EDGE_WIDTH = 3.0


def _point_edges(n: int, highlight: np.ndarray | None) -> tuple:
    """Per-point (edge colour, edge width) arrays for the measured-point overlay.

    ``highlight`` is a boolean mask over the plotted points (the rows added by the
    time slider's newest line). Unhighlighted points keep the usual thin black
    edge; highlighted ones get a thick red ring. Returns plain scalars when
    nothing is highlighted so the common case sends no per-point arrays.
    """
    if highlight is None or not np.any(highlight):
        return "black", 1.0
    mask = np.asarray(highlight, dtype=bool)
    return (np.where(mask, NEW_POINT_COLOR, "black"),
            np.where(mask, NEW_POINT_EDGE_WIDTH, 1.0))


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
    highlight: np.ndarray | None = None,
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

    ``highlight`` is an optional boolean mask over the rows of ``X``; those points
    are ringed in red instead of black (the time slider uses it to show which
    points the current deposition line added).

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

    # Foreground: every measured datapoint, coloured by value with a black edge
    # (red for the points the time slider's newest line added).
    if show_points:
        edge_color, edge_width = _point_edges(len(X), highlight)
        customdata = np.column_stack([X[:, 0], X[:, 1], X[:, 2], Y])
        fig.add_trace(go.Scatterternary(
            a=X[:, 2], b=X[:, 0], c=X[:, 1],
            mode="markers",
            marker=dict(
                symbol="circle", size=9 * scale, color=Y,
                colorscale="Viridis", cmin=vmin, cmax=vmax,
                line=dict(width=edge_width, color=edge_color),
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
    highlight: np.ndarray | None = None,
):
    """Interactive Plotly 3D tetrahedron for d=4 compositions.

    The four composition columns are the four tetrahedron corners (column i →
    ``TETRA_VERTS[i]``). Every measured point sits inside the solid, and the
    optional interpolated background fills the interior as a translucent 3D grid
    (its markers are kept small and semi-transparent so the opaque, black-edged
    measured points stay legible through the volume).

    ``highlight`` is an optional boolean mask over the rows of ``X``; those points
    are ringed in red and drawn slightly larger, so the points the time slider's
    newest line added stand out inside the volume.
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
    #
    # scatter3d's marker.line.width is a single number (unlike the 2D ternary,
    # which takes a per-point array), so the points the time slider's newest line
    # added go in their own trace: red-ringed and enlarged, since a thin ring
    # alone is easy to lose inside the volume. The two traces are disjoint, so
    # nothing is drawn twice at the same depth.
    if show_points:
        pxyz = comp_to_xyz(X)
        customdata = np.column_stack([X, Y])
        hover = "<br>".join(
            f"{labels[k]}=%{{customdata[{k}]:.3f}}" for k in range(X.shape[1])
        ) + f"<br>{value_name}=%{{customdata[{X.shape[1]}]:.4f}}<extra></extra>"

        new_mask = (np.zeros(len(X), dtype=bool) if highlight is None
                    else np.asarray(highlight, dtype=bool))
        layers = [(~new_mask, "black", 1.0, 5 * scale, "measured")]
        if new_mask.any():
            layers.append((new_mask, NEW_POINT_COLOR, NEW_POINT_EDGE_WIDTH,
                           8 * scale, "newest line"))
        # The colorbar rides on the first non-empty layer.
        bar_taken = False
        for mask, edge_color, edge_width, msize, name in layers:
            if not mask.any():
                continue
            fig.add_trace(go.Scatter3d(
                x=pxyz[mask, 0], y=pxyz[mask, 1], z=pxyz[mask, 2], mode="markers",
                marker=dict(
                    size=msize, color=Y[mask],
                    colorscale="Viridis", cmin=vmin, cmax=vmax,
                    line=dict(width=edge_width, color=edge_color),
                    colorbar=None if bar_taken else colorbar,
                ),
                customdata=customdata[mask], hovertemplate=hover,
                name=name, showlegend=False,
            ))
            bar_taken = True

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


# ── non-simplex diagrams ──────────────────────────────────────────────────────
#
# A source whose parameters are *not* a composition has no simplex to sit on: its
# columns are independent physical quantities with their own units and ranges, and
# projecting them onto a triangle or a tetrahedron would assert a
# sum-to-one relationship that does not exist. They are drawn on ordinary axes in
# their own units instead, with dimensionality choosing the diagram exactly as it
# does on the simplex side:
#
#   d=2  → a flat heatmap of the surrogate with the measured points over it: the
#          direct analogue of the ternary, and the only case where the background
#          is a genuine dense picture of the landscape rather than a slice of one.
#   d=3  → a 3-D scatter in the parameter box (the analogue of the tetrahedron),
#          with a translucent voxel cloud of the surrogate filling the volume.
#   d=4  → a scatter-plot matrix. A 4-D box has no faithful 2-D or 3-D embedding,
#          and unlike a 4-component simplex it cannot borrow one from a
#          sum-to-one constraint (which is what collapses that case to a solid
#          tetrahedron). Every axis pair is shown instead, coloured by the
#          objective, with no surrogate background: a 2-D projection of a 4-D
#          surrogate would have to marginalise over the two hidden axes, and the
#          resulting smooth field would look far more informative than it is.
#   d>=5 → the CoNet, the same as the simplex side.

def _box_axis(label: str, rng: tuple[float, float], scale: float) -> dict:
    """Shared linear-axis styling for the non-simplex diagrams."""
    return dict(
        title=dict(text=label, font=dict(size=15 * scale)),
        range=list(rng), tickfont=dict(size=12 * scale),
        showline=True, linecolor="black", linewidth=1,
        gridcolor="rgba(0,0,0,0.10)", zeroline=False,
    )


def build_box2d_figure(
    X: np.ndarray,
    Y: np.ndarray,
    labels: tuple[str, ...],
    *,
    bounds: np.ndarray,
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
    highlight: np.ndarray | None = None,
):
    """d=2 non-simplex: surrogate heatmap in real units + measured points.

    The counterpart of ``build_ternary_figure`` for two independent parameters.
    Axis ranges come from ``bounds`` (the declared design space) rather than from
    the plotted subset, so the frame holds still while the time slider fills it.
    """
    import plotly.graph_objects as go

    bg = fit_box_background(X, Y, grid_n, n_estimators, background, bounds,
                            length_scale=gp_length_scale)
    grid_vals = None if bg is None else bg[1]
    vmin, vmax = (color_limits if color_limits is not None
                  else _color_limits(grid_vals, Y))

    fig = go.Figure()
    colorbar = dict(
        title=dict(text=value_name, font=dict(size=14 * scale)),
        thickness=28, len=0.9, x=min(plot_size + 0.06, 0.98), xpad=0,
        tickfont=dict(size=14 * scale),
    )

    if bg is not None:
        _, vals, axes = bg
        n = len(axes[0])
        # box_grid builds with indexing="ij", so vals reshapes to [x, y]; Heatmap
        # indexes z as [y][x], hence the transpose.
        fig.add_trace(go.Heatmap(
            x=axes[0], y=axes[1], z=vals.reshape(n, n).T,
            colorscale="Viridis", zmin=vmin, zmax=vmax, zsmooth="best",
            hoverinfo="skip", showscale=not show_points,
            colorbar=colorbar if not show_points else None,
        ))

    if show_points:
        edge_color, edge_width = _point_edges(len(X), highlight)
        fig.add_trace(go.Scatter(
            x=X[:, 0], y=X[:, 1], mode="markers",
            marker=dict(
                size=9 * scale, color=Y, colorscale="Viridis",
                cmin=vmin, cmax=vmax, showscale=True, colorbar=colorbar,
                line=dict(width=edge_width, color=edge_color),
            ),
            customdata=np.column_stack([Y]),
            hovertemplate=(
                f"{labels[0]}=%{{x:.4g}}<br>{labels[1]}=%{{y:.4g}}<br>"
                f"{value_name}=%{{customdata[0]:.4g}}<extra></extra>"),
            name="measured", showlegend=False,
        ))

    b = np.asarray(bounds, dtype=float)
    fig.update_layout(
        title=dict(text=title, x=0.5, y=0.98, yanchor="top", font=dict(size=17)),
        font=dict(size=15),
        xaxis=_box_axis(labels[0], (b[0, 0], b[0, 1]), scale),
        yaxis=_box_axis(labels[1], (b[1, 0], b[1, 1]), scale),
        plot_bgcolor="white",
        margin=dict(l=int(40 + 40 * scale), r=40, t=90, b=int(50 + 30 * scale)),
        height=720,
    )
    fig.update_xaxes(domain=[0.0, plot_size])
    return fig


def build_box3d_figure(
    X: np.ndarray,
    Y: np.ndarray,
    labels: tuple[str, ...],
    *,
    bounds: np.ndarray,
    grid_n: int,
    n_estimators: int,
    title: str,
    value_name: str = "Objective",
    background: str = "rf",
    show_points: bool = True,
    gp_length_scale: float = 0.3,
    scale: float = 1.0,
    color_limits: tuple[float, float] | None = None,
    highlight: np.ndarray | None = None,
):
    """d=3 non-simplex: 3-D scatter in the parameter box, real units on all axes.

    The counterpart of ``build_quaternary_figure``. Unlike the tetrahedron there
    is no enclosing solid to draw — the domain *is* the axis box — so the frame is
    the axes themselves, kept at ``bounds`` and shown with ticks, because here
    the numbers carry units worth reading.

    ``aspectmode="cube"`` is deliberate: the three axes are different physical
    quantities, so there is no meaningful common scale between them and a cube is
    the honest neutral choice.
    """
    import plotly.graph_objects as go

    grid_n = int(min(grid_n, BOX3D_GRID_MAX_N))
    bg = fit_box_background(X, Y, grid_n, n_estimators, background, bounds,
                            length_scale=gp_length_scale)
    grid_vals = None if bg is None else bg[1]
    vmin, vmax = (color_limits if color_limits is not None
                  else _color_limits(grid_vals, Y))

    fig = go.Figure()
    colorbar = dict(
        title=dict(text=value_name, font=dict(size=14 * scale)),
        thickness=28, len=0.9, tickfont=dict(size=14 * scale),
    )

    if bg is not None:
        pts, vals, _ = bg
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
            marker=dict(size=_tetra_bg_marker_size(grid_n), color=vals,
                        colorscale="Viridis", cmin=vmin, cmax=vmax,
                        opacity=0.10, line=dict(width=0),
                        colorbar=None if show_points else colorbar,
                        showscale=not show_points),
            hoverinfo="skip", name=f"{background.upper()} background",
            showlegend=False,
        ))

    if show_points:
        edge_color, edge_width = _point_edges(len(X), highlight)
        fig.add_trace(go.Scatter3d(
            x=X[:, 0], y=X[:, 1], z=X[:, 2], mode="markers",
            marker=dict(size=5 * scale, color=Y, colorscale="Viridis",
                        cmin=vmin, cmax=vmax, showscale=True, colorbar=colorbar,
                        line=dict(width=edge_width, color=edge_color)),
            customdata=np.column_stack([Y]),
            hovertemplate=(
                f"{labels[0]}=%{{x:.4g}}<br>{labels[1]}=%{{y:.4g}}<br>"
                f"{labels[2]}=%{{z:.4g}}<br>"
                f"{value_name}=%{{customdata[0]:.4g}}<extra></extra>"),
            name="measured", showlegend=False,
        ))

    b = np.asarray(bounds, dtype=float)
    fig.update_layout(
        title=dict(text=title, x=0.5, y=0.98, yanchor="top", font=dict(size=17)),
        font=dict(size=15),
        scene=dict(
            xaxis=_box_axis(labels[0], (b[0, 0], b[0, 1]), scale),
            yaxis=_box_axis(labels[1], (b[1, 0], b[1, 1]), scale),
            zaxis=_box_axis(labels[2], (b[2, 0], b[2, 1]), scale),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=90, b=10),
        autosize=True,
    )
    return fig


def build_splom_figure(
    X: np.ndarray,
    Y: np.ndarray,
    labels: tuple[str, ...],
    *,
    bounds: np.ndarray,
    title: str,
    value_name: str = "Objective",
    scale: float = 1.0,
    color_limits: tuple[float, float] | None = None,
    highlight: np.ndarray | None = None,
    **_ignored,
):
    """d=4 non-simplex: scatter-plot matrix of every axis pair, coloured by value.

    Only the lower triangle is drawn — the upper half is the same six panels
    transposed — and the diagonal is dropped, so the grid shows each of the six
    distinct pairings once.

    No surrogate background, deliberately: see the note above ``_box_axis``. Each
    panel here is an honest *projection* of the samples (every point appears in
    every panel, at its true coordinates), and adding a fitted field behind it
    would imply a 2-D landscape that the other two axes are silently averaged out
    of.

    ``**_ignored`` swallows the background/grid knobs the shared renderer passes
    to every builder; none of them apply to a SPLOM.
    """
    import plotly.graph_objects as go

    vmin, vmax = color_limits if color_limits is not None else _color_limits(None, Y)
    dense = len(X) > SPLOM_DENSE_N
    edge_color, edge_width = _point_edges(len(X), highlight)
    if highlight is None or not np.any(highlight):
        # A SPLOM repeats every sample across six panels, so at these counts the
        # per-point outline is most of the ink. Drop it unless it is carrying the
        # time slider's red ring.
        edge_color, edge_width = ("rgba(0,0,0,0.35)", 0.5) if dense else ("black", 1.0)

    b = np.asarray(bounds, dtype=float)
    fig = go.Figure(go.Splom(
        dimensions=[dict(label=labels[i], values=X[:, i],
                         axis=dict(matches=False))
                    for i in range(X.shape[1])],
        showupperhalf=False, diagonal=dict(visible=False),
        marker=dict(
            size=(4 if dense else 7) * scale, color=Y, colorscale="Viridis",
            cmin=vmin, cmax=vmax, showscale=True,
            opacity=0.65 if dense else 0.9,
            line=dict(width=edge_width, color=edge_color),
            colorbar=dict(title=dict(text=value_name, font=dict(size=14 * scale)),
                          thickness=24, len=0.85, tickfont=dict(size=13 * scale)),
        ),
        text=[f"{value_name}={v:.4g}" for v in Y],
        hovertemplate="%{text}<extra></extra>",
    ))

    axis_style = dict(showline=True, linecolor="black", linewidth=1,
                      gridcolor="rgba(0,0,0,0.10)", zeroline=False,
                      tickfont=dict(size=11 * scale))
    layout = {"xaxis" if i == 0 else f"xaxis{i + 1}":
              dict(range=[b[i, 0], b[i, 1]], **axis_style)
              for i in range(X.shape[1])}
    layout |= {"yaxis" if i == 0 else f"yaxis{i + 1}":
               dict(range=[b[i, 0], b[i, 1]], **axis_style)
               for i in range(X.shape[1])}
    fig.update_layout(
        title=dict(text=title, x=0.5, y=0.99, yanchor="top", font=dict(size=17)),
        font=dict(size=13), plot_bgcolor="white", dragmode="select",
        margin=dict(l=70, r=40, t=95, b=60), height=760, **layout,
    )
    return fig


def build_figure(X: np.ndarray, Y: np.ndarray, labels: tuple[str, ...], **kwargs):
    """Dispatch to the diagram for this source's constraint *and* dimensionality.

    ``simplex=True`` (the default, and every run directory / result log) picks the
    composition diagrams: ternary at d=3, tetrahedron at d=4. ``simplex=False``
    picks the plain-axes diagrams: heatmap at d=2, 3-D box at d=3, scatter-plot
    matrix at d=4.

    d>=5 has no diagram in either family and is handled by ``conet_png_src``
    instead; ``render_state`` routes it there before ever calling this.
    """
    d = X.shape[1]
    simplex = kwargs.pop("simplex", True)
    bounds = kwargs.pop("bounds", None)
    # plot_size only applies to the flat 2-D diagrams; the 3-D scenes are sized by
    # the camera, so zooming already covers them, and a SPLOM sizes its own grid.
    plot_size = kwargs.pop("plot_size", None)

    if simplex:
        if d == 3:
            if plot_size is not None:
                kwargs["plot_size"] = plot_size
            return build_ternary_figure(X, Y, labels, **kwargs)
        if d == 4:
            return build_quaternary_figure(X, Y, labels, **kwargs)
        raise ValueError(
            f"d={d} has no simplex diagram; use conet_png_src for d>={CONET_MIN_D}.")

    if bounds is None:
        raise ValueError("a non-simplex source needs bounds to fix its axes")
    if d == 2:
        if plot_size is not None:
            kwargs["plot_size"] = plot_size
        return build_box2d_figure(X, Y, labels, bounds=bounds, **kwargs)
    if d == 3:
        return build_box3d_figure(X, Y, labels, bounds=bounds, **kwargs)
    if d == 4:
        return build_splom_figure(X, Y, labels, bounds=bounds, **kwargs)
    raise ValueError(
        f"d={d} has no box diagram; use conet_png_src for d>={CONET_MIN_D}.")


# ── CoNet (d >= 5) ────────────────────────────────────────────────────────────
#
# plot_10d's CoNet is a matplotlib render, so it is delivered to the browser as a
# PNG data URI rather than as a Plotly figure. This is the plain single-dataset
# CoNet (what `conet.png` shows) — paired_conet.py's uniform-baseline variant is
# deliberately not used here: there is no synthetic landscape to draw a baseline
# from, and the second panel would say nothing about a real campaign.

class CoNetTooFewPoints(RuntimeError):
    """Raised when a time-slider step holds too few samples to embed."""


# plot_10d.build_conet_structure refuses to embed fewer than 5 points.
CONET_MIN_POINTS = 5

# A 1x1 transparent PNG. Used for the CoNet's first few time-slider steps, which
# hold too few samples for UMAP to embed: the slider's left end is meant to show an
# empty plot, so the panel is blanked rather than left displaying a stale render.
BLANK_PNG_SRC = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)
BLANK_PNG_BYTES = base64.b64decode(BLANK_PNG_SRC.split(",", 1)[1])


def _data_uri(png: bytes) -> str:
    """Wrap raw PNG bytes as the ``src`` of an ``html.Img``."""
    return "data:image/png;base64," + base64.b64encode(png).decode()

# The fitted map, keyed by dataset identity. The UMAP fit dominates the cost (~30s
# for 2k points vs <1s to replay it), and the time slider re-renders on every drag,
# so caching the fit is what makes scrubbing usable at all.
_CONET_CACHE: dict[tuple, dict] = {}
_CONET_CACHE_MAX = 4


def fit_conet_frame(ds: Dataset, *, key: tuple) -> dict:
    """Fit the CoNet map ONCE on a dataset's full point set, and cache it.

    Everything the time slider needs to draw an intermediate step is fixed here,
    on all the data:

      * ``frame`` + ``E_raw`` — the fitted UMAP embedding and its gap/purity warp
        parameters. Replaying these on a prefix of the points (rather than
        refitting UMAP on that prefix) is what holds each sample at the same
        coordinates through the whole animation; refitting would rebuild the map
        from scratch at every step and the points would jump around.
      * ``iters`` — the per-point deposition line, which is what plot_10d rims in
        red as the "current iteration" (see below).
      * ``limits`` — the axis extent, so the frame never rescales mid-animation.
      * ``bounds`` — the 10th-90th-percentile colour bounds, so a value keeps the
        same colour at every step.
    """
    import plot_10d as p10

    # embed_X, not X: _reduce_comp row-normalises whatever it is given, which is
    # the identity on a composition but arbitrary on a box of mixed physical
    # units. Unit-scaling a non-simplex source first is what stops the single
    # widest-ranged column (hplc's push_speed, 80-150 Hz against sample_loop's
    # 0-0.08 ml) from being the only thing the embedding can see.
    comp, active = p10._reduce_comp(ds.embed_X)
    names = [n for n, a in zip(ds.labels, active) if a]
    resp = ds.Y.reshape(-1, 1)
    # plot_10d rims the points of the LATEST iteration in red. Its standalone
    # entry points have no iteration column and pass acquisition order (one point
    # per "iteration"), which would rim a single sample; here the deposition line
    # is the real unit, so the rim marks every point the newest line added — the
    # same thing the simplex diagrams highlight.
    iters = np.asarray(ds.lines, dtype=float)
    M, _ = p10.build_conet(comp, names, resp, iters)
    fit = {
        "comp": comp, "names": names, "resp": resp, "iters": iters,
        "frame": M["frame"], "E_raw": M["E_raw"],
        "limits": p10._cn_view_limits(M["E"]),
        "bounds": p10.conet_bounds(resp),
    }
    if len(_CONET_CACHE) >= _CONET_CACHE_MAX:
        _CONET_CACHE.pop(next(iter(_CONET_CACHE)))
    _CONET_CACHE[key] = fit
    return fit


def conet_png_bytes(ds: Dataset, *, key: tuple, rows: np.ndarray | None = None) -> bytes:
    """Render the CoNet to PNG bytes, optionally restricted to ``rows``.

    ``rows`` (the time slider's selection) keeps only those samples. They are drawn
    in the map fitted on the FULL dataset — the fitted frame is replayed on those
    same rows of ``E_raw``, which reproduces the full fit's coordinates exactly, so
    points appear in place instead of the map shifting under them.

    The dominance fields (the coloured composition regions) *are* recomputed per
    step: they describe where the currently-drawn samples sit, so they should grow
    with the data. The axis limits and colour bounds they are computed against come
    from the full fit, so the frame itself stays put.

    The points of the step's newest deposition line come out red-rimmed: the fit
    carries the line index as plot_10d's iteration, and plot_10d rims the highest
    iteration present — which, on a prefix, is the line that step added.
    """
    import plot_10d as p10

    fit = _CONET_CACHE.get(key) or fit_conet_frame(ds, key=key)
    total = len(fit["comp"])
    sel = np.arange(total) if rows is None else np.asarray(rows, dtype=int)
    if len(sel) < CONET_MIN_POINTS:
        raise CoNetTooFewPoints(
            f"the CoNet needs at least {CONET_MIN_POINTS} samples to embed; "
            f"the time slider is showing {len(sel)}."
        )

    frame = dict(fit["frame"])
    frame["E_raw"] = fit["E_raw"][sel]
    M, F = p10.build_conet(
        fit["comp"][sel], fit["names"], fit["resp"][sel], fit["iters"][sel],
        frame=frame, limits=fit["limits"],
    )
    title = f"{ds.title} · CoNet (d={ds.d}"
    title += ")" if rows is None else f", {len(sel)}/{total} samples)"

    buf = io.BytesIO()
    p10.save_png(M, F, fit["bounds"], title, buf)
    return buf.getvalue()


def conet_png_src(ds: Dataset, *, key: tuple, rows: np.ndarray | None = None) -> str:
    """``conet_png_bytes`` as a data URI, for direct use as an image ``src``."""
    return _data_uri(conet_png_bytes(ds, key=key, rows=rows))


# ── precomputed time-slider steps ─────────────────────────────────────────────
#
# A slider step is not cheap to draw. A CoNet step replays the fitted map but
# still rebuilds the co-occurrence graph and the dominance fields over the
# visible points (seconds, growing with the prefix), and a simplex step refits
# the background surrogate over the whole grid. Rendering on demand means every
# drag pays that again, which is what made scrubbing unusable.
#
# So a step is rendered at most ONCE, and what is kept is the *finished* artefact
# — the PNG the browser displays, or the serialised figure Dash sends it — not
# the inputs it was made from. Redrawing a step costs a file read, not a render.
#
# Everything lands on disk, under a directory named for a *signature* covering
# whatever decides what a step looks like: the source (with its mtime, so edited
# data misses), and the render knobs. The step index is deliberately not part of
# it, so one signature owns a whole slider range. Because it is all on disk, a
# range survives closing the app: reopening a dataset you have already scrubbed
# costs nothing.
#
# Turning the slider on (or switching source, or moving a knob) starts a
# background pass that walks the range and fills whatever is missing, beginning
# at the step on screen. A new signature cancels the pass in flight, so switching
# datasets does not leave stale work competing for the CPU.
#
# CoNet PNGs are never inlined into the callback response. They are served as
# ordinary image URLs off a Flask route (``/plot-run-step/...``), so the browser
# streams the file straight from disk and — since the URL is content-addressed —
# keeps it in its own cache afterwards. Inlining them as base64 data URIs meant
# pushing ~2MB of JSON through the callback on every single step.

CACHE_DIR = Path(os.environ.get("PLOT_RUN_CACHE_DIR")
                 or Path(tempfile.gettempdir()) / "zombi_plot_run_cache")
CACHE_MAX_BYTES = 4 * 1024 ** 3   # least-recently-used signatures pruned past this
# Part of every on-disk name. A signature covers the data and the render knobs,
# but not the renderers themselves, so a change to what a panel *draws* would
# otherwise be masked by steps cached before it. Bump this whenever that happens
# (adding the needle rules did): old steps are then simply never looked up, and
# age out with the prune, instead of being served stale.
STEP_FORMAT = 3
PRECOMPUTE = True                 # --no-precompute renders steps on demand instead
STEP_URL_PREFIX = "/plot-run-step"

# Decoded steps are also kept in memory, so a step already on screen costs
# nothing at all to redraw. Bounded, since a figure at a high grid resolution is
# not small; the disk copy is what makes a miss cheap.
_STEPS: "OrderedDict[tuple, OrderedDict[int, object]]" = OrderedDict()
_STEPS_MAX_SIGS = 4

# Renders are serialised: they are CPU-bound, and the CoNet path touches
# matplotlib's global rcParams. `foreground()` lets a step the user is actually
# waiting on jump the queue instead of sitting behind a speculative one.
_RENDER_LOCK = threading.Lock()
_STATE_LOCK = threading.Lock()
_FOREGROUND = threading.Event()
_FG_LOCK = threading.Lock()
_FG_COUNT = 0

FULL_STEP = -1     # the "timeline off" render (all data), cached alongside the steps
_CACHED = object()  # sentinel: the step is cached, but the caller asked for no value


@contextlib.contextmanager
def foreground():
    """Mark the enclosed render as one a user is waiting on.

    The background pass yields the render lock while this is held. Counted, not a
    bare flag: Dash serves callbacks on several threads, so two foreground
    renders can overlap and the first to finish must not un-flag the other.
    """
    global _FG_COUNT
    with _FG_LOCK:
        _FG_COUNT += 1
        _FOREGROUND.set()
    try:
        yield
    finally:
        with _FG_LOCK:
            _FG_COUNT -= 1
            if _FG_COUNT == 0:
                _FOREGROUND.clear()


def _figure_to_bytes(fig) -> bytes:
    """Serialise a plotly figure exactly as Dash would send it."""
    import plotly.io as pio
    return pio.to_json(fig).encode("utf-8")


def _figure_from_bytes(raw: bytes) -> dict:
    """The stored figure, as the plain dict a callback can return.

    Deliberately *not* rebuilt into a ``go.Figure``: Dash serialises a dict
    straight through, so this skips both figure construction and the numpy
    encoding that dominates the cost of handing a big figure back.
    """
    return json.loads(raw)


@dataclass(frozen=True)
class Codec:
    """How one kind of step artefact is stored and handed back.

    ``ram`` is how many decoded steps to hold in memory for this kind; 0 means
    none, which is right for the CoNet PNGs — nothing ever reads their bytes in
    Python, the browser fetches them from disk by URL.
    """
    ext: str
    encode: object
    decode: object
    ram: int


PNG_STEP = Codec("png", lambda b: b, lambda b: b, ram=0)
FIGURE_STEP = Codec("json", _figure_to_bytes, _figure_from_bytes, ram=200)


def _sig_hash(sig: tuple) -> str:
    """Short stable name for a signature; also the URL path of its PNG steps."""
    return hashlib.sha1(repr((STEP_FORMAT, sig)).encode()).hexdigest()[:16]


# Signature -> its directory, for the signatures this session has touched. Also
# the set the prune must not evict: their steps are being served by URL right now.
_SIG_DIRS: dict[tuple, Path] = {}


def _prune_cache(keep: Path) -> None:
    """Drop least-recently-used signature directories past ``CACHE_MAX_BYTES``."""
    live = {keep, *_SIG_DIRS.values()}
    try:
        dirs = []
        for c in CACHE_DIR.iterdir():
            if c.is_dir() and c not in live:
                size = sum(f.stat().st_size for f in c.glob("*") if f.is_file())
                dirs.append((c.stat().st_mtime, size, c))
    except OSError:
        return
    total = sum(size for _, size, _ in dirs)
    for _, size, c in sorted(dirs):          # oldest first
        if total <= CACHE_MAX_BYTES:
            break
        shutil.rmtree(c, ignore_errors=True)
        total -= size


def _sig_dir(sig: tuple) -> Path:
    """Directory holding one signature's steps, created and LRU-touched."""
    ck = (str(CACHE_DIR), sig)
    d = _SIG_DIRS.get(ck)
    if d is not None:
        return d
    d = CACHE_DIR / _sig_hash(sig)
    fresh = not d.is_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        if fresh:
            _prune_cache(d)
        else:
            os.utime(d, None)   # touch, so the prune order is by last use
    except OSError:
        pass
    _SIG_DIRS[ck] = d
    return d


def _step_path(sig: tuple, n: int, codec: Codec) -> Path:
    return _sig_dir(sig) / f"{n:+06d}.{codec.ext}"


def _steps_for(sig: tuple, limit: int) -> "OrderedDict[int, object]":
    """The in-memory step store for ``sig``, evicting the least recent signature."""
    with _STATE_LOCK:
        store = _STEPS.get(sig)
        if store is None:
            while len(_STEPS) >= _STEPS_MAX_SIGS:
                _STEPS.popitem(last=False)
            store = _STEPS[sig] = OrderedDict()
        _STEPS.move_to_end(sig)
        while len(store) > limit:
            store.popitem(last=False)
        return store


def cached_step(sig: tuple, n: int, render, *, codec: Codec, want_value: bool = True):
    """Return the artefact for one slider step, rendering it at most once.

    ``render(n)`` builds the step; ``n == FULL_STEP`` means the whole dataset.
    The encoded form is written under ``CACHE_DIR``, so a step outlives both
    eviction from memory and the app itself.

    ``want_value=False`` says the caller only needs the step to *exist* (the
    CoNet path, which hands the browser a URL) — an already-cached step then
    costs a single ``stat`` and returns ``None``, rather than reading a megabyte
    of PNG into Python for nothing.
    """
    store = _steps_for(sig, codec.ram) if codec.ram else None
    path = _step_path(sig, n, codec)

    def hit():
        if store is not None:
            v = store.get(n)
            if v is not None:
                store.move_to_end(n)
                return v
        if path.is_file():
            if not want_value:
                return _CACHED
            try:
                v = codec.decode(path.read_bytes())
            except OSError:
                return None
            if store is not None:
                store[n] = v
                _steps_for(sig, codec.ram)
            return v
        return None

    v = hit()
    if v is None:
        with _RENDER_LOCK:
            v = hit()               # another thread may have rendered it meanwhile
            if v is None:
                raw = codec.encode(render(n))
                _write_step(path, raw)
                v = _CACHED if not want_value else codec.decode(raw)
                if store is not None and v is not _CACHED:
                    store[n] = v
                    _steps_for(sig, codec.ram)
    return None if v is _CACHED else v


def _write_step(path: Path, raw: bytes) -> None:
    """Write a step atomically — a torn file must never be read back or served."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(raw)
        tmp.replace(path)
    except OSError:
        pass


def step_image_url(sig: tuple, n: int) -> str:
    """URL serving a PNG step, for the ``src`` of an ``html.Img``.

    Content-addressed (the signature covers the source and its mtime), so the
    route can tell the browser to keep it indefinitely.
    """
    return f"{STEP_URL_PREFIX}/{_sig_hash(sig)}/{n:+06d}.png"


def prefetch_urls(sig: tuple, n: int, n_max: int, *, span: int = 5) -> list[str]:
    """URLs of the already-rendered steps either side of ``n``.

    Handed to the browser to pull into its image cache, so stepping to a
    neighbour paints from memory with no request at all.
    """
    out = []
    for d in range(1, span + 1):
        for m in (n + d, n - d):
            if 0 <= m <= n_max and _step_path(sig, m, PNG_STEP).is_file():
                out.append(step_image_url(sig, m))
    return out


@dataclass
class _Warmer:
    """One background pass filling every step of a set of panels."""
    key: tuple
    order: tuple[int, ...]
    cancel: threading.Event = field(default_factory=threading.Event)
    done: int = 0
    error: str | None = None

    @property
    def total(self) -> int:
        return len(self.order)


_WARMER: _Warmer | None = None


def start_warm(order, tracks) -> None:
    """Fill every step of ``tracks`` in the background, cancelling any earlier pass.

    ``tracks`` is a list of ``(signature, render, codec)`` — one per visible panel,
    filled in lockstep so the main diagram and its convergence plot are ready at
    the same step rather than one lagging the other.

    Re-called on every render; a pass already running (or finished) for the same
    signatures is left alone, so this is cheap to call unconditionally.
    """
    global _WARMER
    key = tuple(sig for sig, _, _ in tracks)
    with _STATE_LOCK:
        if _WARMER is not None:
            if _WARMER.key == key:
                return
            _WARMER.cancel.set()
        w = _WARMER = _Warmer(key=key, order=tuple(order))

    def _run() -> None:
        for n in w.order:
            if w.cancel.is_set():
                return
            # Never make a step the user is waiting on queue behind a speculative
            # one: hold off while a foreground render is pending.
            while _FOREGROUND.is_set() and not w.cancel.is_set():
                time.sleep(0.05)
            for sig, render, codec in tracks:
                if w.cancel.is_set():
                    return
                try:
                    cached_step(sig, n, render, codec=codec, want_value=False)
                except Exception as e:  # a broken step must not kill the pass
                    w.error = f"{type(e).__name__}: {e}"
                    return
            w.done += 1

    threading.Thread(target=_run, name="plot_run-warm", daemon=True).start()


def warm_order(current: int, n_max: int) -> list[int]:
    """Step order for the background pass: outward from the step being viewed.

    Forward first (scrubbing is normally left-to-right from where you are), then
    back to the start, then the all-data render that unchecking the slider shows.
    """
    current = max(0, min(int(current), int(n_max)))
    return (list(range(current, n_max + 1)) + list(range(current - 1, -1, -1))
            + [FULL_STEP])


def warm_status_text() -> str:
    """One line on the background pass, for the panel under the slider."""
    w = _WARMER
    if w is None:
        return ""
    if w.error:
        return f"precompute stopped: {w.error}"
    if w.done >= w.total:
        return f"all {w.total} steps precomputed - scrubbing is instant"
    return f"precomputing steps... {w.done}/{w.total}"


def conet_step_renderer(ds: Dataset, key: tuple):
    """``render(n)`` -> the CoNet PNG for the first ``n`` lines of ``ds``.

    The first few steps hold too few samples for UMAP to embed; they render as
    the blank image (which is what that end of the slider means) so the pass can
    fill the whole range without special cases.
    """
    def render(n: int) -> bytes:
        rows = None if n == FULL_STEP else ds.prefix_rows(n)
        if rows is not None and len(rows) < CONET_MIN_POINTS:
            return BLANK_PNG_BYTES
        return conet_png_bytes(ds, key=key, rows=rows)
    return render


def simplex_step_renderer(ds: Dataset, **kw):
    """``render(n)`` -> the simplex figure for the first ``n`` lines of ``ds``.

    Each step also rings the points its newest line (line ``n-1``) added in red,
    so scrubbing the slider shows at a glance what that step contributed rather
    than only that the cloud grew. The full-dataset step has no "newest" line and
    is drawn plain.
    """
    background = kw.pop("background")

    def render(n: int):
        sub = ds if n == FULL_STEP else ds.prefix(n)
        highlight = None if n == FULL_STEP else sub.lines == int(n) - 1
        # A background surrogate needs something to fit; the early steps can be
        # empty or near-empty, so drop to no background there.
        bg = background if len(sub.X) >= 2 else "none"
        # simplex/bounds come from the FULL dataset, not the prefix: they say what
        # kind of space this is and how far it extends, which a slider step must
        # not be able to change under the animation.
        return build_figure(sub.X, sub.Y, sub.labels, background=bg,
                            highlight=highlight, simplex=ds.simplex,
                            bounds=ds.axis_bounds, **kw)
    return render


def convergence_step_renderer(ds: Dataset, *, show_line_marks: bool):
    """``render(n)`` -> the convergence figure for the first ``n`` lines of ``ds``."""
    def render(n: int):
        rows = None if n == FULL_STEP else ds.prefix_rows(n)
        return build_convergence_figure(ds, rows=rows,
                                        show_line_marks=show_line_marks)
    return render


# ── convergence panel ─────────────────────────────────────────────────────────

def build_convergence_figure(
    ds: Dataset,
    *,
    rows: np.ndarray | None = None,
    show_line_marks: bool = False,
    height: int = 300,
):
    """Plotly convergence panel in the style of ``run_mobo.plot_convergence``.

    Every observed Y against sample index as steel-blue dots, with the
    running-best envelope over them as a dark-orange ``steps-post`` line — the same
    two marks, colours and axis labels the run's ``convergence.png`` uses.

    When the source records activations (``ds.activations``), the envelope **resets
    at every activation boundary**, matching ``plot_convergence(activations=...)``:
    each activation gets its own best-so-far sawtooth, drawn as its own disconnected
    segment with a dotted rule at the boundary. Each activation is a fresh ZoMBI
    search phase, so letting one curve coast on an earlier phase's peak would hide
    how much the new phase re-explores. Sources with no activation record (a
    ``.db``/``.csv`` result log) fall back to a single global running best.

    ``rows`` (the time slider's selection) restricts the drawn data, while the axis
    ranges stay pinned to the FULL dataset. That is what makes the panel fill in
    under the slider: points appear inside a fixed frame instead of the whole plot
    rescaling at every step. Each point keeps its own sample index on the x axis, so
    a point never moves once drawn.

    When the source records needles (``ds.needles``), a crimson dashed rule marks
    each one, the way ``plot_convergence`` does — at the *sample that was declared
    the needle*, which is where its ``convergence.png`` draws it. Because that is
    usually an earlier point than the moment of declaration, a needle only appears
    once the run had actually declared it, so the time slider never shows a needle
    before the optimizer found one.

    ``show_line_marks`` adds a very faint rule at each deposition-line boundary —
    it is what ties a slider position (measured in lines) to a place on this axis.
    """
    import plotly.graph_objects as go

    Y_all = np.asarray(ds.Y, dtype=float).ravel()
    n_total = len(Y_all)
    sel = np.arange(n_total) if rows is None else np.asarray(rows, dtype=int)
    acts = None if ds.activations is None else np.asarray(ds.activations).ravel()
    if acts is not None and len(acts) != n_total:
        acts = None

    # Ranges from the FULL data, padded, so the frame is identical at every step.
    if n_total:
        lo, hi = float(Y_all.min()), float(Y_all.max())
        pad = (hi - lo) * 0.06 or 1e-6
        y_range = [lo - pad, hi + pad]
        x_range = [-0.02 * n_total, n_total * 1.02]
    else:
        y_range, x_range = None, None

    fig = go.Figure()

    if show_line_marks and n_total:
        # Boundaries of the real (variably-sized) lines. Kept fainter than the
        # activation rules below: there are far more of them (109 lines vs ~14
        # activations in these runs), so they must read as a background scale.
        for b in np.flatnonzero(np.diff(ds.lines)) + 1:
            fig.add_vline(x=float(b) - 0.5, line=dict(color="#cccccc", width=0.6, dash="dot"),
                          opacity=0.25, layer="below")

    # Activation boundaries, drawn over the FULL dataset so the frame is fixed.
    if acts is not None:
        for b in np.flatnonzero(np.diff(acts)) + 1:
            fig.add_vline(x=float(b) - 0.5, line=dict(color="#888888", width=0.7, dash="dot"),
                          opacity=0.35, layer="below")

    # Needle rules, under the data. Drawn as traces rather than layout shapes so
    # they carry one shared legend entry and say on hover which needle they are.
    if ds.needles is not None and y_range is not None:
        shown = [(i, when) for i, when in ds.needles if when <= len(sel)]
        for k, (i, when) in enumerate(shown):
            fig.add_trace(go.Scatter(
                x=[float(i), float(i)], y=y_range, mode="lines",
                name="needle found", legendgroup="needle", showlegend=(k == 0),
                line=dict(color="crimson", width=1.2, dash="dash"), opacity=0.55,
                hovertemplate=(f"needle {k + 1} at sample {int(i)}<br>"
                               f"declared after {int(when)} points<extra></extra>"),
            ))

    fig.add_trace(go.Scatter(
        x=sel, y=Y_all[sel], mode="markers", name="obs",
        marker=dict(size=5, color="steelblue", opacity=0.65),
        hovertemplate=f"sample %{{x}}<br>{ds.value_name}=%{{y:.4f}}<extra></extra>",
    ))

    # One envelope per activation (or a single global one when there is no
    # activation record). Each segment is its own trace, so nothing bridges a
    # boundary — the reset is a real break in the line, not a steep climb.
    # "Best" follows the source's own goal. Every simplex source here maximises,
    # but the public photodegradation sets minimise, and running np.maximum over
    # those would draw a confidently rising curve for a campaign whose whole
    # object is to drive the value down.
    minimizing = str(ds.goal).lower().startswith("min")
    accumulate = np.minimum.accumulate if minimizing else np.maximum.accumulate
    if len(sel):
        best_word = "running best (min)" if minimizing else "running best"
        if acts is None:
            groups = [sel]
            best_label = best_word
        else:
            sel_acts = acts[sel]
            cuts = np.flatnonzero(np.diff(sel_acts)) + 1
            groups = np.split(sel, cuts)
            best_label = f"{best_word} (reset/activation)"
        for gi, g in enumerate(groups):
            if not len(g):
                continue
            fig.add_trace(go.Scatter(
                x=g, y=accumulate(Y_all[g]), mode="lines",
                name=best_label, legendgroup="best", showlegend=(gi == 0),
                line=dict(color="darkorange", width=2, shape="hv"),
                hovertemplate="best through %{x}: %{y:.4f}<extra></extra>",
            ))

    n_needles = 0 if ds.needles is None else sum(1 for _, w in ds.needles
                                                 if w <= len(sel))
    needle_note = "" if ds.needles is None else f", {n_needles} needles"

    fig.update_layout(
        xaxis=dict(title="Sample index", range=x_range, showgrid=True, gridcolor="#eee",
                   zeroline=False, linecolor="black", showline=True, mirror=False),
        yaxis=dict(title=f"{ds.value_name} Y", range=y_range, showgrid=True,
                   gridcolor="#eee", zeroline=False, linecolor="black", showline=True),
        title=dict(text=f"Convergence  ({len(sel)}/{n_total} pts{needle_note})",
                   x=0.5, y=0.97,
                   yanchor="top", font=dict(size=13)),
        showlegend=True,
        legend=dict(x=0.99, y=0.02, xanchor="right", yanchor="bottom",
                    font=dict(size=10), bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=60, r=40, t=40, b=45),
        height=height,
        plot_bgcolor="white", paper_bgcolor="white",
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
    """Names of .db and .csv data files under data/."""
    if not DATA_DIR.is_dir():
        return []
    files = list(DATA_DIR.glob("*.db")) + list(DATA_DIR.glob("*.csv"))
    return sorted(p.name for p in files)


# Loaded datasets, keyed by source signature. The time slider re-renders on every
# drag and each render needs the full dataset (to slice a prefix from), so without
# this every step would re-read the .db from disk.
_DATA_CACHE: dict[tuple, Dataset] = {}
_DATA_CACHE_MAX = 8


def _source_key(source_type: str, run_name: str | None, db_name: str | None,
                db_value: str | None, public_name: str | None = None) -> tuple:
    """Cache key identifying a source, including its mtime so edits are picked up."""
    if source_type == "run":
        p = _resolve_run_dir(run_name or "")
        return ("run", str(p), p.stat().st_mtime)
    if source_type == "public":
        # Keyed on the cached data.csv's mtime, the same way the others are keyed
        # on their file, so a re-fetch of the upstream dataset invalidates it.
        from benchmarks.public_db.olympus import DATA_DIR as _PUB_DIR
        p = _PUB_DIR / (public_name or "") / "data.csv"
        return ("public", public_name or "", p.stat().st_mtime)
    p = _resolve_db_path(db_name or "")
    return ("db", str(p), p.stat().st_mtime, db_value)


def load_source(key: tuple) -> Dataset:
    """Load (or reuse) the ``Dataset`` for a cache key produced by ``_source_key``."""
    hit = _DATA_CACHE.get(key)
    if hit is not None:
        return hit
    if key[0] == "run":
        ds = load_run_source(Path(key[1]), None)
    elif key[0] == "public":
        ds = load_public_dataset(key[1])
    else:
        ds = load_db_dataset(Path(key[1]), key[3])
    if len(_DATA_CACHE) >= _DATA_CACHE_MAX:
        _DATA_CACHE.pop(next(iter(_DATA_CACHE)))
    _DATA_CACHE[key] = ds
    return ds


def build_app(grid_n: int = TERNARY_GRID_N, n_estimators: int = RF_N_ESTIMATORS):
    """Construct the Dash application."""
    from dash import Dash, ctx, dcc, html, Input, Output, State, no_update

    run_names = _list_run_dirs()
    db_names = _list_db_files()
    public_names = _list_public_datasets()

    step_btn_style = {
        "flex": "0 0 auto", "width": "28px", "height": "28px", "padding": "0",
        "lineHeight": "1", "fontSize": "11px", "color": "#555",
        "background": "#fff", "border": "1px solid #d5d5d5",
        "borderRadius": "4px", "cursor": "pointer",
    }

    default_db = next((n for n in ("3d.db", "4d.db", "6d.db") if n in db_names),
                      db_names[0] if db_names else None)
    default_run = DEFAULT_RUN if DEFAULT_RUN in run_names else (
        run_names[0] if run_names else None)
    default_public = next((n for n in CURATED_PUBLIC if n in public_names),
                          public_names[0] if public_names else None)

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
                    {"label": " Public dataset (Olympus)", "value": "public"},
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

            html.Div(id="public-controls", children=[
                html.Label("Public dataset", style=label_style),
                dcc.Dropdown(
                    id="public-dropdown",
                    options=[{"label": n, "value": n} for n in public_names],
                    value=default_public, clearable=False,
                ),
                html.Div(
                    "Fetched from the-matter-lab/olympus into "
                    "benchmarks/public_db/data. Run "
                    "`python benchmarks/public_db/olympus.py --fetch all` "
                    "to populate.",
                    style={"fontSize": "11px", "color": "#777", "marginTop": "6px"},
                ),
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

            # ── Time slider ───────────────────────────────────────────────────
            # Replays the run one deposition line at a time. Fully left = 0 lines
            # (an empty plot), fully right = every line. The steps are real lines
            # read from the data's Iteration column, not fixed 24-point blocks, so
            # culled (short) lines stay in phase — see line_index().
            html.Hr(style={"marginTop": "18px", "border": "none",
                           "borderTop": "1px solid #e3e3e3"}),
            dcc.Checklist(
                id="timeline-on",
                options=[{"label": " Enable time slider", "value": "on"}],
                value=[],
            ),
            html.Div(id="timeline-controls", style={"display": "none"}, children=[
                # A step button either side of the slider: one line back / one
                # line on, without having to land the handle on an exact tick.
                html.Div(style={"display": "flex", "alignItems": "center",
                                "gap": "6px"}, children=[
                    html.Button("\u25c0", id="timeline-prev", n_clicks=0,
                                title="one line back", style=step_btn_style),
                    html.Div(style={"flex": 1, "minWidth": 0}, children=[
                        dcc.Slider(id="timeline", min=0, max=1, step=1, value=1,
                                   marks=None,
                                   tooltip={"placement": "bottom",
                                            "always_visible": True}),
                    ]),
                    html.Button("\u25b6", id="timeline-next", n_clicks=0,
                                title="one line on", style=step_btn_style),
                ]),
                html.Div(id="timeline-status",
                         style={"fontSize": "12px", "color": "#666", "marginTop": "4px"}),
                # Every step of the range is rendered in the background as soon as
                # the slider is switched on; this says how far that has got, so a
                # step that is briefly still slow is explained rather than puzzling.
                html.Div(id="warm-status",
                         style={"fontSize": "11px", "color": "#999", "marginTop": "2px"}),
                dcc.Interval(id="warm-poll", interval=800, disabled=True),
                dcc.Store(id="conet-prefetch"),
                dcc.Store(id="conet-prefetch-sink"),
            ]),

            # ── Convergence panel ─────────────────────────────────────────────
            dcc.Checklist(
                id="convergence-on",
                options=[{"label": " Show convergence plot", "value": "on"}],
                value=[],
                style={"marginTop": "12px"},
            ),
            html.Hr(style={"marginTop": "12px", "border": "none",
                           "borderTop": "1px solid #e3e3e3"}),

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

        # Main plot area. d<=4 renders into the Plotly graph and d>=5 into the
        # CoNet <img> (plot_10d draws with matplotlib, so it arrives as a PNG,
        # streamed from the step cache by URL); exactly one of the two is ever
        # visible. The convergence panel sits below whichever is showing.
        html.Div(style={"flex": 1, "padding": "10px", "minWidth": 0}, children=[
            dcc.Loading(children=[
                html.Div(id="simplex-container", children=[
                    dcc.Graph(id="ternary-graph", style={"height": "90vh"}),
                ]),
                html.Div(id="conet-container", style={"display": "none"}, children=[
                    html.Img(id="conet-image",
                             style={"width": "100%", "height": "auto",
                                    "display": "block"}),
                ]),
                html.Div(id="convergence-container", style={"display": "none"}, children=[
                    dcc.Graph(id="convergence-graph", style={"height": "300px"}),
                ]),
            ]),
        ]),
    ])

    # Show/hide the run vs db vs public control groups.
    @app.callback(
        Output("run-controls", "style"),
        Output("db-controls", "style"),
        Output("public-controls", "style"),
        Input("source-type", "value"),
    )
    def _toggle(source_type):
        show, hide = {}, {"display": "none"}
        return tuple(show if source_type == t else hide
                     for t in ("run", "db", "public"))

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

    # Serve rendered CoNet steps as ordinary images, straight off the disk cache.
    # The URL is content-addressed (its signature covers the source file and its
    # mtime), so the browser is told it never expires: coming back to a step it
    # has already drawn costs no request at all, let alone a re-render.
    _URL_DIR_RE = re.compile(r"^[0-9a-f]{16}$")
    _URL_STEP_RE = re.compile(r"^[-+]\d{5}\.png$")

    @app.server.route(f"{STEP_URL_PREFIX}/<sig>/<step>")
    def _serve_step(sig, step):
        from flask import abort, send_file
        if not (_URL_DIR_RE.match(sig) and _URL_STEP_RE.match(step)):
            abort(404)
        path = CACHE_DIR / sig / step
        if not path.is_file():
            abort(404)
        return send_file(path, mimetype="image/png", max_age=31536000)

    # Pull the neighbouring steps into the browser's image cache, so stepping to
    # one paints from memory instead of waiting on a request.
    app.clientside_callback(
        """
        function (urls) {
            (urls || []).forEach(function (u) { var img = new Image(); img.src = u; });
            return window.dash_clientside.no_update;
        }
        """,
        Output("conet-prefetch-sink", "data"),
        Input("conet-prefetch", "data"),
    )

    # Show/hide the time-slider controls with its checkbox.
    @app.callback(
        Output("timeline-controls", "style"),
        Input("timeline-on", "value"),
    )
    def _toggle_timeline(timeline_on):
        return {"display": "block", "marginTop": "8px"} if timeline_on else {"display": "none"}

    # Re-range the time slider whenever the source changes: one step per deposition
    # line, 0 (empty) on the left and every line on the right. Snapped to the new
    # maximum so switching sources never leaves the slider past the end of the data.
    @app.callback(
        Output("timeline", "max"),
        Output("timeline", "value"),
        Output("timeline", "marks"),
        Input("source-type", "value"),
        Input("run-dropdown", "value"),
        Input("db-dropdown", "value"),
        Input("db-value-dropdown", "value"),
        Input("public-dropdown", "value"),
    )
    def _timeline_range(source_type, run_name, db_name, db_value, public_name):
        try:
            ds = load_source(_source_key(source_type, run_name, db_name,
                                         db_value, public_name))
        except Exception:
            return 1, 1, None
        n = max(ds.n_lines, 1)
        return n, n, {0: "0", n: str(n)}

    # Step the slider one line at a time. `allow_duplicate` because the slider's
    # value is also set by _timeline_range when the source changes.
    @app.callback(
        Output("timeline", "value", allow_duplicate=True),
        Input("timeline-prev", "n_clicks"),
        Input("timeline-next", "n_clicks"),
        State("timeline", "value"),
        State("timeline", "max"),
        prevent_initial_call=True,
    )
    def _step_timeline(_prev, _next, value, n_max):
        delta = -1 if ctx.triggered_id == "timeline-prev" else 1
        stepped = int(value or 0) + delta
        stepped = max(0, min(stepped, int(n_max or 0)))
        return no_update if stepped == value else stepped

    @app.callback(
        Output("timeline-prev", "disabled"),
        Output("timeline-next", "disabled"),
        Input("timeline", "value"),
        Input("timeline", "max"),
    )
    def _step_buttons(value, n_max):
        value, n_max = int(value or 0), int(n_max or 0)
        return value <= 0, value >= n_max

    # Poll the precompute pass only while the time slider is in use.
    @app.callback(
        Output("warm-poll", "disabled"),
        Input("timeline-on", "value"),
    )
    def _poll_enabled(timeline_on):
        return not bool(timeline_on)

    @app.callback(
        Output("warm-status", "children"),
        Input("warm-poll", "n_intervals"),
    )
    def _warm_status(_n):
        return warm_status_text()

    # Main render callback.
    @app.callback(
        Output("ternary-graph", "figure"),
        Output("simplex-container", "style"),
        Output("conet-image", "src"),
        Output("conet-container", "style"),
        Output("convergence-graph", "figure"),
        Output("convergence-container", "style"),
        Output("timeline-status", "children"),
        Output("status", "children"),
        Output("conet-prefetch", "data"),
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
        Input("timeline-on", "value"),
        Input("timeline", "value"),
        Input("convergence-on", "value"),
        # Appended last so render_state's existing positional signature — which
        # several tests and sibling scripts call directly — keeps its meaning.
        Input("public-dropdown", "value"),
    )
    def _render(*args):
        return render_state(*args, no_update=no_update)

    return app


_SIMPLEX_DIAGRAMS = {3: "ternary triangle (d=3, simplex)",
                     4: "tetrahedron (d=4, simplex)"}
_BOX_DIAGRAMS = {2: "axes heatmap (d=2, non-simplex)",
                 3: "3-D box scatter (d=3, non-simplex)",
                 4: "scatter-plot matrix (d=4, non-simplex)"}


def _diagram_name(ds: Dataset) -> str:
    """Human-readable name of the diagram ``ds`` dispatches to, for the status panel.

    Says which *family* was chosen as well as which member, since that is the
    part a reader cannot infer from d alone — a d=4 source is a tetrahedron or a
    scatter-plot matrix depending entirely on whether its columns are a
    composition.
    """
    if ds.d >= CONET_MIN_D:
        return f"CoNet (d={ds.d}, {'simplex' if ds.simplex else 'non-simplex'})"
    table = _SIMPLEX_DIAGRAMS if ds.simplex else _BOX_DIAGRAMS
    return table.get(ds.d, f"unsupported (d={ds.d})")


def render_state(source_type, run_name, db_name, db_value, background, show_points,
                 gn, ntrees, gp_ls, scale, plot_size, color_override, color_min,
                 color_max, timeline_on, timeline, convergence_on, public_name=None,
                 *, no_update=None, precompute=None):
    """Everything the main Dash callback renders, as a plain function.

    Split out of the callback body so it can be exercised (and its eight outputs
    checked) without standing up a server. Returns the callback's output tuple:
    ``(simplex figure, simplex style, conet src, conet style, convergence figure,
    convergence style, timeline status, status, conet prefetch urls)``.

    ``precompute`` (default: the module-level ``PRECOMPUTE``) controls whether the
    background pass that fills the rest of the slider range is started.
    """
    precompute = PRECOMPUTE if precompute is None else bool(precompute)
    hide = {"display": "none"}
    show_block = {"display": "block"}
    try:
        if source_type == "run" and not run_name:
            return (no_update,) * 7 + ("No run selected.", no_update)
        if source_type == "db" and not (db_name and db_value):
            return ((no_update,) * 7
                    + ("No database / value column selected.", no_update))
        if source_type == "public" and not public_name:
            return ((no_update,) * 7
                    + ("No public dataset selected — fetch one with "
                       "`python benchmarks/public_db/olympus.py --fetch all`.",
                       no_update))

        key = _source_key(source_type, run_name, db_name, db_value, public_name)
        full = load_source(key)

        # The time slider cuts the dataset to its first N deposition lines.
        # Everything downstream draws `ds` but keeps its frame (axis limits, colour
        # bounds, CoNet map) pinned to `full`, so the view fills in rather than
        # rescaling at each step.
        timeline_on = bool(timeline_on)
        n_lines = int(timeline) if timeline is not None else full.n_lines
        rows = full.prefix_rows(n_lines) if timeline_on else None
        ds = full.prefix(n_lines) if timeline_on else full
        tl_status = (f"{n_lines}/{full.n_lines} lines · {len(ds.X)}/{len(full.X)} points"
                     if timeline_on else "")

        # Manual color-scale override: only applied when the box is checked and both
        # bounds are valid (min < max). Otherwise the limits come from the FULL
        # data's percentiles — not the visible prefix's — so a point's colour does
        # not change as the time slider advances.
        color_limits = None
        if color_override and color_min is not None and color_max is not None:
            lo, hi = float(color_min), float(color_max)
            if hi > lo:
                color_limits = (lo, hi)
        if color_limits is None and timeline_on:
            color_limits = _color_limits(None, full.Y)

        # Every step goes through the cache, under a signature holding everything
        # that changes what a step looks like (the step index deliberately not
        # among them, so one signature covers the whole slider range). The step
        # on screen is rendered here if it is not cached yet; `start_warm` then
        # fills the rest of the range in the background, so the next drag lands
        # on an already-rendered step instead of paying for it.
        step = n_lines if timeline_on else FULL_STEP
        tracks = []          # (signature, render, codec) per visible panel

        fig = no_update
        conet_src = no_update
        prefetch = no_update
        conet = full.d >= CONET_MIN_D
        if conet:
            simplex_style, conet_style = hide, show_block
            diagram = _diagram_name(full)
            # None of the surrogate knobs feed the CoNet, so its signature is the
            # source alone — twiddling them never invalidates a rendered step.
            sig = ("conet", key)
            render = conet_step_renderer(full, key)
            tracks.append((sig, render, PNG_STEP))
            # The slider's leftmost steps hold too few samples to embed; they
            # render blank (an empty plot is what that end means) and say so
            # under the slider rather than erroring out.
            if rows is not None and len(rows) < CONET_MIN_POINTS:
                tl_status = (
                    f"{tl_status} — the CoNet needs at least {CONET_MIN_POINTS} "
                    f"samples to embed; the time slider is showing {len(rows)}.")
        else:
            simplex_style, conet_style = show_block, hide
            diagram = _diagram_name(full)
            sig = ("simplex", key, int(gn), int(ntrees), background,
                   bool(show_points), float(gp_ls), float(scale),
                   float(plot_size), color_limits)
            render = simplex_step_renderer(
                full,
                grid_n=int(gn), n_estimators=int(ntrees),
                title=full.title, value_name=full.value_name,
                background=background, show_points=bool(show_points),
                gp_length_scale=float(gp_ls), scale=float(scale),
                plot_size=float(plot_size), color_limits=color_limits,
            )
            tracks.append((sig, render, FIGURE_STEP))

        conv_fig = no_update
        conv_style = hide
        if convergence_on:
            conv_style = show_block
            # Its own signature: the convergence panel depends on the source and
            # on whether the line marks are drawn, not on any surrogate knob.
            conv_sig = ("convergence", key, bool(timeline_on))
            conv_render = convergence_step_renderer(
                full, show_line_marks=bool(timeline_on))
            tracks.append((conv_sig, conv_render, FIGURE_STEP))

        # Flagged as foreground so the background pass yields the render lock
        # rather than making these wait behind a step nobody asked for.
        with foreground():
            if conet:
                # The bytes are never read into Python: the step is written to
                # disk and the browser is handed a URL to stream it from.
                cached_step(sig, step, render, codec=PNG_STEP, want_value=False)
                conet_src = step_image_url(sig, step)
                prefetch = prefetch_urls(sig, step, full.n_lines)
            else:
                fig = cached_step(sig, step, render, codec=FIGURE_STEP)
            if convergence_on:
                conv_fig = cached_step(conv_sig, step, conv_render,
                                       codec=FIGURE_STEP)

        if timeline_on and precompute:
            start_warm(warm_order(step, full.n_lines), tracks)

        yr = (f"[{ds.Y.min():.4f}, {ds.Y.max():.4f}]" if len(ds.Y) else "(no points)")
        # Say plainly whether the running best resets: a .db/.csv result log has no
        # activation column, so there the envelope is necessarily one global curve.
        if full.activations is None:
            act_note = "activations: none recorded (single global running best)"
        else:
            act_note = f"activations: {int(full.activations.max()) + 1} (running best resets)"
        # Same story for needles: they are optimizer state, recorded in a run's
        # snapshots and nowhere in a result log.
        if full.needles is None:
            act_note += "\nneedles: none recorded"
        else:
            act_note += f"\nneedles: {len(full.needles)} (marked on the convergence plot)"
        # The background knobs do nothing on a CoNet, and nothing on the d=4
        # scatter-plot matrix either (which deliberately fits no surrogate), so
        # say 'n/a' rather than naming a mode that is not being applied.
        has_bg = full.d < CONET_MIN_D and not (full.d == 4 and not full.simplex)
        status = (
            f"{len(ds.X)}/{len(full.X)} points · {full.n_lines} lines\n"
            f"Y range: {yr}\n"
            f"space: {'simplex' if full.simplex else 'box (non-simplex)'}"
            f" · d={full.d} · goal={full.goal}\n"
            f"diagram: {diagram}\n"
            f"background: {background if has_bg else 'n/a'}\n"
            f"components: {full.labels}\n"
            f"{act_note}"
        )
        return (fig, simplex_style, conet_src, conet_style,
                conv_fig, conv_style, tl_status, status, prefetch)
    except Exception as e:  # surface load/plot errors in the UI
        return (no_update,) * 7 + (f"Error: {e}", no_update)


# ── static export (legacy matplotlib path) ────────────────────────────────────

def export_png(args: argparse.Namespace) -> None:
    """Render a static PNG using matplotlib (the original behaviour)."""
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.public:
        ds = load_public_dataset(args.public)
        out_default = DATA_DIR / f"{args.public}.png"
    elif args.db:
        db_path = _resolve_db_path(args.db)
        ds = load_db_dataset(db_path, args.value)
        out_default = db_path.with_suffix(".png")
    else:
        run_dir = _resolve_run_dir(args.run)
        ds = load_run_source(run_dir, args.snapshot, args.corner_dims, args.labels)
        out_default = run_dir / "run_ternary.png"

    X, Y, labels, title, value_name, d = (
        ds.X, ds.Y, ds.labels, ds.title, ds.value_name, ds.d)
    print(f"Title  : {title}")
    print(f"Labels : {labels}")
    print(f"Space  : {'simplex' if ds.simplex else 'box (non-simplex)'}"
          f"   goal: {ds.goal}")
    print(f"Diagram: {_diagram_name(ds)}")
    print(f"Points : {X.shape[0]}   Y range: [{Y.min():.4f}, {Y.max():.4f}]")

    # A non-simplex source below the CoNet threshold gets the plain-axes diagrams
    # rather than any of the composition ones.
    if not ds.simplex and d < CONET_MIN_D:
        out = Path(args.out) if args.out else out_default.with_name(
            f"{out_default.stem}_box.png")
        fig = _export_box_png(ds, args, plt)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved figure -> {out}")
        if args.show:
            plt.show()
        return

    # d>=5 has no diagram in either family; export the CoNet PNG the app shows.
    if d >= CONET_MIN_D:
        out = Path(args.out) if args.out else out_default.with_name(
            f"{out_default.stem}_conet.png")
        src = conet_png_src(ds, key=("export", str(out_default)))
        out.write_bytes(base64.b64decode(src.split(",", 1)[1]))
        print(f"Saved figure -> {out}")
        return

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


def _export_box_png(ds: Dataset, args: argparse.Namespace, plt):
    """Static PNG for a **non-simplex** source below the CoNet threshold.

    The matplotlib counterpart of ``build_box2d_figure`` /
    ``build_box3d_figure`` / ``build_splom_figure``, and it makes the same three
    choices for the same reasons: d=2 gets a filled surrogate field with the
    points over it, d=3 a 3-D scatter in the parameter box, and d=4 a lower-
    triangle scatter-plot matrix with no surrogate behind it.

    Returns the figure; the caller saves it.
    """
    b = ds.axis_bounds
    X, Y, labels, d = ds.X, ds.Y, ds.labels, ds.d

    if d == 2:
        bg = fit_box_background(X, Y, args.grid_n, args.n_estimators,
                                args.background, b,
                                length_scale=args.gp_length_scale)
        vmin, vmax = _color_limits(None if bg is None else bg[1], Y)
        fig, ax = plt.subplots(figsize=(8.4, 7.0))
        mappable = None
        if bg is not None:
            _, vals, axes = bg
            n = len(axes[0])
            # box_grid is indexing="ij" (column 0 first), imshow wants row=y.
            mappable = ax.imshow(
                vals.reshape(n, n).T, origin="lower", aspect="auto",
                cmap="viridis", vmin=vmin, vmax=vmax,
                extent=(b[0, 0], b[0, 1], b[1, 0], b[1, 1]), zorder=1)
        if not args.no_points:
            mappable = ax.scatter(X[:, 0], X[:, 1], c=Y, cmap="viridis",
                                  vmin=vmin, vmax=vmax, s=28, zorder=3,
                                  edgecolors="black", linewidths=0.7)
        ax.set_xlim(b[0, 0], b[0, 1])
        ax.set_ylim(b[1, 0], b[1, 1])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_title(ds.title, fontsize=11)
        if mappable is not None:
            fig.colorbar(mappable, ax=ax, label=ds.value_name,
                         fraction=0.046, pad=0.04)
        fig.tight_layout()
        return fig

    if d == 3:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d)

        vmin, vmax = _color_limits(None, Y)
        fig = plt.figure(figsize=(8.6, 7.6))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap="viridis",
                        vmin=vmin, vmax=vmax, s=22, depthshade=False,
                        edgecolors="black", linewidths=0.4)
        ax.set_xlim(*b[0])
        ax.set_ylim(*b[1])
        ax.set_zlim(*b[2])
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        ax.set_title(ds.title, fontsize=11)
        fig.colorbar(sc, ax=ax, label=ds.value_name, fraction=0.03, pad=0.10)
        return fig

    # d == 4: lower-triangle scatter-plot matrix, no surrogate behind it.
    vmin, vmax = _color_limits(None, Y)
    fig, axes = plt.subplots(d - 1, d - 1, figsize=(10.0, 9.2),
                             squeeze=False)
    sc = None
    for r in range(d - 1):          # y is column r+1
        for c in range(d - 1):      # x is column c
            ax = axes[r][c]
            if c > r:
                ax.axis("off")
                continue
            sc = ax.scatter(X[:, c], X[:, r + 1], c=Y, cmap="viridis",
                            vmin=vmin, vmax=vmax, s=8, alpha=0.75,
                            linewidths=0.0)
            ax.set_xlim(*b[c])
            ax.set_ylim(*b[r + 1])
            ax.tick_params(labelsize=7)
            if r == d - 2:
                ax.set_xlabel(labels[c], fontsize=9)
            else:
                ax.set_xticklabels([])
            if c == 0:
                ax.set_ylabel(labels[r + 1], fontsize=9)
            else:
                ax.set_yticklabels([])
    fig.suptitle(ds.title, fontsize=11)
    if sc is not None:
        fig.colorbar(sc, ax=axes, label=ds.value_name, fraction=0.030, pad=0.02)
    return fig


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
    global CACHE_DIR, PRECOMPUTE

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
    parser.add_argument("--public", default=None,
                        help="Public Olympus dataset name (export; overrides "
                             "--db/--run). One of "
                             f"{', '.join(CURATED_PUBLIC)}, or any other name "
                             "cached under benchmarks/public_db/data.")
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
    parser.add_argument("--cache-dir", default=None,
                        help="Directory holding precomputed slider steps, kept "
                             f"between sessions (default: {CACHE_DIR}).")
    parser.add_argument("--no-precompute", action="store_true",
                        help="Render time-slider steps on demand instead of "
                             "precomputing the whole range in the background.")
    parser.add_argument("--show", action="store_true",
                        help="(export only) Display the matplotlib figure as well.")
    args = parser.parse_args()

    if args.export:
        export_png(args)
        return

    if args.cache_dir:
        CACHE_DIR = Path(args.cache_dir)
    PRECOMPUTE = not args.no_precompute

    app = build_app(grid_n=args.grid_n, n_estimators=args.n_estimators)
    print(f"Dash app running at http://127.0.0.1:{args.port}")
    print(f"Rendered steps are cached under {CACHE_DIR}"
          + (" and precomputed in the background" if PRECOMPUTE else ""))
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
