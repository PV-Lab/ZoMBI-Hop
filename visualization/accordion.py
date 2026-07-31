"""
visualization/accordion.py
==========================
The paper's summary figure.  Everything is drawn on **one** hand-picked
3-simplex landscape (the "accordion landscape") so each panel of the figure
tells a different part of the story about the *same* surface.

The landscape is an :class:`~synthetic_data.ensemble.Ensemble` — the same
layered objective the benchmarks use — with its true optima placed by hand
instead of at random:

    * exactly **three** optima, one per ternary corner;
    * each pulled in from its corner by :data:`CORNER_INSET` (toward the simplex
      centroid) so the basin's slope is visible rising on *every* side rather
      than being clipped by the triangle boundary;
    * each jittered by a small random tangential offset
      (:data:`CORNER_JITTER`, driven by :data:`OPTIMA_SEED`) so the three peaks
      do not sit in perfectly symmetric, obviously-synthetic positions.

Everything else (roughness, weak optima/distractors, ridges) is stock
``Ensemble`` background, so the surface still looks like a plausible materials
response rather than three clean Gaussians.

Panels
------
``--base``    the landscape on its own, as a filled ternary heat map.
``--slices``  the 3D panel: one ZoMBI-Hop run's per-iteration lines stacked along
              a time axis running front-right.  Each slice is a bare ternary
              triangle carrying **only** the 24-point line measured at that
              iteration — no landscape behind it, so the eye follows where the
              optimizer went.  Candidate runs come from
              ``optimize/scripts/accordion_runs.sbatch`` (5 parallel runs on this
              exact landscape via ``evaluate.py --dataset accordion``).

Run:
  uv run python visualization/accordion.py --base
  uv run python visualization/accordion.py --slices RUN_DIR --stride 15 --max-slices 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # repo root, so synthetic_data is importable

from synthetic_data.ensemble import Ensemble  # noqa: E402

_SQRT3_2 = np.sqrt(3) / 2

# Corner labels, in comp_to_xy column order (bottom-left, bottom-right, top).
LABELS = ("FAPbI3", "MAPbI3", "MAPbBr3")

# ── the accordion landscape ─────────────────────────────────────────────────
# How far each optimum is pulled from its corner toward the centroid, as a
# fraction of the corner->centroid distance.  Large enough that the basin's
# uphill slope is fully inside the triangle on all sides.
CORNER_INSET = 0.5
# Extra random offset (in composition units, tangent to the simplex) so the
# three peaks are not exactly symmetric.
CORNER_JITTER = 0.1
OPTIMA_SEED = 7        # drives the jitter only
# Master seed for every background feature.  ``Ensemble`` derives one stream per
# feature from this single seed (noise, weak optima, ridges, plateaus, signs, …),
# so there is no separate "noise seed" to set.  8 is chosen because its signed-
# feature draw comes out mixed — weak optima, ridges and plateaus each get both a
# raised and a sunk instance, instead of e.g. five pits and no bumps.
LANDSCAPE_SEED = 3

# ── the three main optima ────────────────────────────────────────────────────
# WIDTH: the negated-Ackley sharpness ``b`` in ``exp(-b * rms_delta)``.  SMALLER is
# WIDER (a broad hill), larger is a narrow spike.  Ensemble's own random sweep
# draws this from (2.2, 15.0); 6.5 sits in the middle.
BASIN_WIDTH = 20
# STRENGTH: how far the optima stand above the background, NOT their height — the
# true optima are pinned to the top of the raw field by construction (``_PEAK``)
# and cannot be raised.  What this does is cap every upward background excursion at
# ``_PEAK * (1 - 2*margin)``, so 0 lets distractors reach the peak and 0.5 flattens
# the background to the neutral level.  0.2 caps the background at 60% of peak.
OPTIMA_MARGIN = 0.2

GRID_N = 300           # ternary lattice resolution for the rendered surface
CMAP = "viridis"
OUT_DEFAULT = _HERE / "accordion_base.png"


def comp_to_xy(comp: np.ndarray) -> np.ndarray:
    """(N,3) simplex compositions → (N,2) ternary Cartesian (matches plot_run.py).

    col0 → (0,0) bottom-left, col1 → (1,0) bottom-right, col2 → (0.5,√3/2) top.
    """
    p = np.asarray(comp, float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    s = p.sum(axis=-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return np.column_stack([p[:, 1] + 0.5 * p[:, 2], _SQRT3_2 * p[:, 2]])


def ternary_grid(n: int = GRID_N) -> np.ndarray:
    """(N,3) uniform lattice on the probability simplex (mirrors plot_run.py)."""
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    keep = (i + j) <= n
    i, j = i[keep], j[keep]
    return np.column_stack([i, j, n - i - j]).astype(float) / n


def corner_optima(inset: float = CORNER_INSET, jitter: float = CORNER_JITTER,
                  seed: int = OPTIMA_SEED) -> np.ndarray:
    """The three hand-placed optima: one per corner, inset and jittered.

    Each centre starts at a vertex, is moved ``inset`` of the way to the
    centroid (so the whole basin — including its uphill slope — sits inside the
    triangle), then nudged by a random sum-zero offset of magnitude up to
    ``jitter`` so the placement is not perfectly symmetric.  The result is
    clipped back onto the simplex.
    """
    rng = np.random.default_rng(seed)
    centroid = np.full(3, 1.0 / 3.0)
    out = np.empty((3, 3), dtype=float)
    for k in range(3):
        v = np.zeros(3)
        v[k] = 1.0
        c = (1.0 - inset) * v + inset * centroid
        d = rng.standard_normal(3)
        d -= d.mean()                      # tangent to the simplex (sum-zero)
        d /= max(np.linalg.norm(d), 1e-12)
        c = c + d * jitter * rng.uniform(0.5, 1.0)
        c = np.clip(c, 1e-3, None)
        out[k] = c / c.sum()
    return out


class AccordionLandscape(Ensemble):
    """``Ensemble`` whose true optima are :func:`corner_optima` instead of random."""

    def __init__(self, *, inset: float = CORNER_INSET, jitter: float = CORNER_JITTER,
                 optima_seed: int = OPTIMA_SEED, **kwargs):
        self._corner_kw = (inset, jitter, optima_seed)
        kwargs.setdefault("n_optima", 3)
        super().__init__(dim=3, **kwargs)

    def _sample_optima(self, n: int) -> np.ndarray:  # noqa: D102 (see base class)
        inset, jitter, seed = self._corner_kw
        return corner_optima(inset, jitter, seed)


def build_landscape(*, inset: float = CORNER_INSET, jitter: float = CORNER_JITTER,
                    optima_seed: int = OPTIMA_SEED,
                    basin_width: float = BASIN_WIDTH,
                    optima_margin: float = OPTIMA_MARGIN,
                    seed: int = LANDSCAPE_SEED) -> AccordionLandscape:
    """The single landscape every panel of the accordion figure is drawn on.

    **Every** ``Ensemble`` background feature is switched on — weak optima,
    ridges, Perlin roughness, plateaus, the structural edge bias and anisotropy
    — so the figure advertises the full generator rather than a subset.  The
    amplitudes are tuned for legibility, not difficulty: each feature is large
    enough to be visible at figure size, but the true optima still dominate
    (``optima_margin`` caps every upward background excursion below the peak).

    Two choices are deliberate rather than incidental:

    * ``edge_region="middle"`` with the seed's negative edge sign digs the
      simplex interior *down*, which throws the three corner basins into relief.
    * ``noise_amp`` is set well above the ``Ensemble`` default because the output
      map squishes the raw field twice (once by ``scale``, once into
      ``[0.5, 1]``), so raw roughness reaches the colour scale at ~1/4 strength.
    """
    return AccordionLandscape(
        inset=inset, jitter=jitter, optima_seed=optima_seed,
        # true optima — broad enough that the slope reads clearly at figure size
        basin_width=basin_width,
        optima_margin=optima_margin,
        input_noise=0.0,          # keep all three peaks tagged as true optima
        # distractors / texture
        n_weak=5, weak_width=100.0, weak_amp=0.045,
        n_ridges=2, ridge_width=0.1, ridge_amp=0.35, ridge_length=0.6,
        noise_freq=12.0, noise_amp=200.0, noise_octaves=4,
        n_plateaus=0, plateau_radius=0.10, plateau_amp=0.40,
        edge_region="middle", edge_amp=0.30, edge_reach=0.45,
        aniso_strength=0,
        neg_frac=0.5,
        seed=seed,
    )


# ── panels ───────────────────────────────────────────────────────────────────

def draw_triangle(ax, labels=LABELS) -> None:
    """Triangle outline, corner labels and axis housekeeping (warm_start style)."""
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.3, zorder=4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, _SQRT3_2 + 0.16)
    ax.text(-0.03, -0.03, labels[0], ha="right", va="top", fontsize=10)
    ax.text(1.03, -0.03, labels[1], ha="left", va="top", fontsize=10)
    ax.text(0.5, _SQRT3_2 + 0.04, labels[2], ha="center", va="bottom", fontsize=10)


def draw_base(ax, fn, *, grid_n: int = GRID_N, show_optima: bool = True,
              colorbar: bool = True):
    """Render the landscape itself: filled ternary heat map + optima markers."""
    grid = ternary_grid(grid_n)
    vals = fn.predict(grid)
    xy = comp_to_xy(grid)

    tpc = ax.tricontourf(xy[:, 0], xy[:, 1], vals, levels=60, cmap=CMAP, zorder=1)
    # Kill the hairline seams between filled levels (ContourSet is itself an
    # artist on matplotlib >= 3.8; older versions expose .collections).
    if hasattr(tpc, "set_edgecolor"):
        tpc.set_edgecolor("face")
    else:  # pragma: no cover - matplotlib < 3.8
        for coll in tpc.collections:
            coll.set_edgecolor("face")

    if show_optima:
        pk = comp_to_xy(fn.centers)
        ax.scatter(pk[:, 0], pk[:, 1], s=70, marker="*", c="white",
                   edgecolors="k", linewidths=0.8, zorder=5)

    draw_triangle(ax)
    if colorbar:
        cb = ax.figure.colorbar(tpc, ax=ax, shrink=0.72, pad=0.02)
        cb.set_label("objective", fontsize=9)
    return tpc


def figure_base(out_png: Path, *, grid_n: int = GRID_N, seed: int = LANDSCAPE_SEED,
                show_optima: bool = True, basin_width: float = BASIN_WIDTH,
                optima_margin: float = OPTIMA_MARGIN) -> None:
    """``--base``: just the landscape, on its own axes."""
    import matplotlib.pyplot as plt

    fn = build_landscape(seed=seed, basin_width=basin_width,
                         optima_margin=optima_margin)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    draw_base(ax, fn, grid_n=grid_n, show_optima=show_optima)
    ax.set_title("Accordion landscape — three corner optima", fontsize=10)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure -> {out_png}")
    for c, v in fn.known_maxima:
        print(f"  optimum at [{c[0]:.3f} {c[1]:.3f} {c[2]:.3f}]  y={v:.4f}")


# ── time-slice panel (3D) ────────────────────────────────────────────────────
# Points measured per objective call — one LineBO *line*, so one iteration.  The
# hardware (and ``run_mobo.physics_simulate_line``) measures a fixed 24-point line
# each call, so ``points.csv`` row order alone recovers the per-iteration grouping:
# rows [24*i, 24*i+24) are the line printed at iteration i.  Kept in sync with
# ``optimize/run_mobo.NUM_EXPERIMENTS``.
POINTS_PER_ITER = 24

# 3D layout.  The time axis runs along mpl's +x and the ternary slice lives in the
# (y, z) plane, so the view angle below sends time toward the front-right.
SLICE_SPACING = 0.90      # gap between consecutive slices, in triangle-edge units
SLICE_ELEV = 10.0
SLICE_AZIM = -55.0
SLICE_POINT_ALPHA = 0.55  # semi-transparent, like plot_run.py's point cloud
# Landscape slice opacity, matching the sampled points by default.  Independent of
# draw order: the surface is forced behind everything regardless of this value.
LANDSCAPE_ALPHA = SLICE_POINT_ALPHA
# Canvas caps (inches).  The slice panel sizes itself from the slice count; these
# stop a --stride 1 render of a ~80-iteration run from asking for a canvas so large
# that the PNG becomes hundreds of megapixels.
MAX_SLICE_FIG_W = 20.0
MAX_SLICE_FIG_H = 8.0
SLICE_DPI = 300
MAX_SLICE_PIXELS = 120e6
SLICE_OUTLINE_ALPHA = 0.30
OUT_SLICES_DEFAULT = _HERE / "accordion_slices.png"


def load_run_points(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read an evaluate.py run's ``points.csv`` → ``(X (N,3), Y (N,), iters (N,))``.

    ``iters`` is the iteration each point was measured at, derived from row order
    (see :data:`POINTS_PER_ITER`).  A trailing partial line — the run's time limit
    landing mid-call — is kept and labelled as its own iteration.
    """
    import pandas as pd

    csv = Path(run_dir) / "points.csv"
    if not csv.is_file():
        raise FileNotFoundError(f"No points.csv under {run_dir}")
    df = pd.read_csv(csv)
    coord_cols = [c for c in ("x1", "x2", "x3") if c in df.columns]
    if len(coord_cols) != 3:
        coord_cols = [c for c in ("FA", "MA", "Br") if c in df.columns]
    if len(coord_cols) != 3:
        raise ValueError(f"{csv}: expected 3 composition columns, found {list(df.columns)}")

    X = df[coord_cols].to_numpy(dtype=float)
    s = X.sum(axis=1, keepdims=True)
    X = X / np.where(s == 0, 1.0, s)
    Y = df["Y"].to_numpy(dtype=float)
    iters = np.arange(len(df)) // POINTS_PER_ITER
    return X, Y, iters


def _trim_png(path: Path, pad: int = 12) -> None:
    """Crop uniform white margins off a saved PNG.

    A 3D axes always reports its full rectangle as its tight bbox — the drawing
    inside it is inscribed with a wide, uncroppable margin — so
    ``bbox_inches="tight"`` leaves a short, wide slice stack floating in a sea of
    white.  Trimming the rendered pixels is the reliable way to get a compact
    figure.  Silently does nothing when Pillow is unavailable.
    """
    try:
        from PIL import Image, ImageChops
    except ImportError:  # pragma: no cover - Pillow is optional
        return
    # Pillow refuses to open images past ~179 Mpx as a decompression-bomb guard.
    # That guard is about untrusted input; this file is the one matplotlib just
    # wrote, so lift the limit rather than fail on a wide many-slice figure.
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(path).convert("RGB")
    bg = Image.new("RGB", img.size, (255, 255, 255))
    bbox = ImageChops.difference(img, bg).getbbox()
    if bbox is None:
        return
    l, t, r, b = bbox
    box = (max(l - pad, 0), max(t - pad, 0),
           min(r + pad, img.width), min(b + pad, img.height))
    img.crop(box).save(path)


def load_run_needles(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an evaluate.py run's ``needles.csv`` → ``(X (M,3), iters (M,))``.

    ``needles.csv`` carries its own ``iteration`` column on the same global
    objective-call basis as :func:`load_run_points`' row-order index (each needle
    is recorded one line before its activation's last), so the two line up
    directly and a needle can be drawn on exactly the slice it was placed at.
    Returns empty arrays when the run recorded no needles.
    """
    import pandas as pd

    csv = Path(run_dir) / "needles.csv"
    if not csv.is_file():
        return np.empty((0, 3)), np.empty(0, dtype=int)
    df = pd.read_csv(csv)
    if df.empty:
        return np.empty((0, 3)), np.empty(0, dtype=int)
    coord_cols = [c for c in ("x1", "x2", "x3") if c in df.columns]
    if len(coord_cols) != 3:
        coord_cols = [c for c in ("FA", "MA", "Br") if c in df.columns]
    X = df[coord_cols].to_numpy(dtype=float)
    s = X.sum(axis=1, keepdims=True)
    X = X / np.where(s == 0, 1.0, s)
    return X, df["iteration"].to_numpy(dtype=int)


def _rerun_config(run_dir: Path) -> dict:
    """The nearest ``rerun_config.json`` at or above ``run_dir`` (``{}`` if none).

    evaluate.py writes it at the top of the output directory, so it sits two or
    three levels above an individual ``run_<k>``; it names the dataset the run was
    scored on.
    """
    p = Path(run_dir).resolve()
    for cand in (p, *p.parents):
        cfg = cand / "rerun_config.json"
        if cfg.is_file():
            try:
                import json

                return json.loads(cfg.read_text())
            except Exception:
                return {}
        if cand.name == "runs":  # don't climb out of optimize/runs
            break
    return {}


def load_run_landscape(run_dir: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """The landscape **this run was evaluated on**, as ``(grid_pts, grid_vals)``.

    Deliberately read from the run directory rather than rebuilt from
    :func:`build_landscape`, so the surface shown is the one the optimizer actually
    walked — the slice panel is routinely pointed at runs on other objectives
    (fullgp, ensemble, …), where accordion's own landscape would be a lie.

    Two sources, in order:

    1. ``coverage_ground_truth.npz`` in the run dir — the exact grid and values
       evaluate.py scored the run against.  Written only when the run was made with
       ``--coverage``.
    2. Failing that, a ``rerun_config.json`` naming ``dataset: "accordion"`` lets
       the deterministic :func:`build_landscape` reproduce it exactly.  Runs made
       before the rename still say ``"figure_one"``; the landscape itself is
       unchanged, so both names are accepted.

    Returns ``None`` when neither applies (an unknown objective with no saved
    ground truth), so the caller can carry on without the landscape slice.
    """
    npz = Path(run_dir) / "coverage_ground_truth.npz"
    if npz.is_file():
        z = np.load(npz)
        if "grid_pts" in z and "grid_vals" in z:
            return np.asarray(z["grid_pts"], float), np.asarray(z["grid_vals"], float)

    if _rerun_config(run_dir).get("dataset") in ("accordion", "figure_one"):
        fn = build_landscape()
        grid = ternary_grid(200)
        return grid, fn.predict(grid)

    return None


def _draw_landscape_slice(ax, grid_pts: np.ndarray, grid_vals: np.ndarray,
                          offset: float, cmap, norm,
                          alpha: float = LANDSCAPE_ALPHA) -> None:
    """Draw the landscape as one filled triangle at ``x = offset``.

    mplot3d has no ``tricontourf``, so the ternary grid is Delaunay-triangulated in
    the slice plane and emitted as a ``Poly3DCollection`` of flat-shaded faces (one
    colour per triangle, from its mean value).  Edges are painted the same colour
    as their face to kill the hairline seams between neighbouring triangles.
    """
    from matplotlib.tri import Triangulation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    xy = comp_to_xy(grid_pts)
    tri = Triangulation(xy[:, 0], xy[:, 1])
    faces = tri.triangles
    verts = [[(offset, xy[i, 0], xy[i, 1]) for i in f] for f in faces]
    colors = cmap(norm(grid_vals[faces].mean(axis=1)))

    # Opacity is baked into the per-face RGBA rather than set via ``set_alpha``:
    # mplot3d's own alpha handling re-derives face colours and would drop the
    # per-triangle colouring.  ``set_alpha(None)`` then tells it to leave them be.
    colors[:, 3] = float(np.clip(alpha, 0.0, 1.0))
    poly = Poly3DCollection(verts, facecolors=colors, edgecolors=colors,
                            linewidths=0.0, shade=False)
    poly.set_alpha(None)

    # Force it to the back.  mplot3d ignores a plain ``zorder`` on 3D artists: each
    # draw it sorts them by ``do_3d_projection()`` *descending* and hands out
    # increasing zorder, so the largest returned depth is drawn first (farthest).
    # The landscape sits at the near end of the time axis, so by true depth it
    # legitimately occludes the sampled points.  Wrapping the projection call —
    # rather than skipping it — keeps the geometry correct while reporting +inf
    # depth, i.e. infinitely far away, so it is always drawn first.
    _orig_proj = poly.do_3d_projection

    def _behind_everything(*args, **kwargs):
        _orig_proj(*args, **kwargs)
        return np.inf

    poly.do_3d_projection = _behind_everything
    ax.add_collection3d(poly)


def _colorbar_rect(fig, ax, span: float, y_hi: float, z_hi: float,
                   *, pad: float = 0.015, width: float = 0.012,
                   height_scale: float = 0.72) -> list[float]:
    """Colorbar rectangle (figure fraction) hugging the projected slice stack.

    A fixed rectangle like ``[0.875, …]`` strands the colorbar far to the right
    whenever the stack does not fill the axes — e.g. ``--stride 1 --spacing 0.05``
    packs 78 slices into the left ~44% of a capped-width canvas, leaving a gulf of
    interior white that ``_trim_png`` cannot touch (it only crops outer margins).
    Projecting the data box through the current 3D view gives where the drawing
    actually ends, so the bar can sit just past it at any slice count or spacing.
    """
    from mpl_toolkits.mplot3d import proj3d

    fig.canvas.draw()  # ensure the projection and transforms are current
    proj = ax.get_proj()
    corners = [(x, y, z) for x in (0.0, span) for y in (0.0, y_hi) for z in (0.0, z_hi)]
    xs, ys = [], []
    for cx, cy, cz in corners:
        px, py, _ = proj3d.proj_transform(cx, cy, cz, proj)
        dx, dy = ax.transData.transform((px, py))
        fx, fy = fig.transFigure.inverted().transform((dx, dy))
        xs.append(fx)
        ys.append(fy)

    x1 = min(max(xs) + pad, 1.0 - width)
    # Height tracks the stack's own vertical extent, not a fixed figure fraction:
    # a bar taller than the drawing re-introduces the same dead space vertically
    # that the x-anchoring just removed horizontally.
    y_mid = 0.5 * (min(ys) + max(ys))
    height = max((max(ys) - min(ys)) * height_scale, 0.10)
    return [x1, max(y_mid - height / 2, 0.01), width, min(height, 0.98)]


def _slice_frame(offset: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed triangle outline for the slice at time ``offset``, as ``(x, y, z)``."""
    tx = np.array([0.0, 1.0, 0.5, 0.0])
    ty = np.array([0.0, 0.0, _SQRT3_2, 0.0])
    return np.full_like(tx, offset), tx, ty


def figure_slices(run_dir: Path, out_png: Path, *, stride: int = 1,
                  max_slices: int | None = None, spacing: float = SLICE_SPACING,
                  elev: float = SLICE_ELEV, azim: float = SLICE_AZIM,
                  point_size: float = 30.0, no_landscape: bool = False,
                  landscape_gap: float | None = None,
                  landscape_alpha: float = LANDSCAPE_ALPHA,
                  at_iters: list[int] | None = None) -> None:
    """``--slices``: the run's per-iteration lines stacked along a 3D time axis.

    Each slice is one ternary triangle carrying **only** the 24-point line measured
    at that iteration — no landscape background, so the eye follows where the
    optimizer went rather than what it was walking over.  Points are coloured by
    measured objective on a single scale shared by every slice (so a brightening
    trend down the time axis *is* the optimizer improving) and drawn
    semi-transparent, matching ``plot_run.py``'s point-cloud styling.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

    X, Y, iters = load_run_points(run_dir)
    nx, n_iters = load_run_needles(run_dir)
    land = None if no_landscape else load_run_landscape(run_dir)
    avail = np.unique(iters)
    if at_iters is not None:
        # Explicit iterations win over stride/max_slices.  Needles land on specific
        # iterations, and an evenly-strided selection usually misses every one of
        # them (stride 16 on a 95-iteration run hits none of 6, 13, 21, 28, …), so
        # picking the slices by hand is the only way to get a panel that shows both
        # the sampling and the needles it produced.
        keep = np.array([i for i in at_iters if i in set(avail.tolist())], dtype=int)
        missing = [i for i in at_iters if i not in set(avail.tolist())]
        if missing:
            print(f"  [warn] --at-iters: no such iteration(s) {missing}; "
                  f"run has 0..{int(avail.max())}")
    else:
        keep = avail[::stride]
        if max_slices is not None:
            keep = keep[:max_slices]

    xy = comp_to_xy(X)
    nxy = comp_to_xy(nx) if len(nx) else np.empty((0, 2))
    # Colour limits from the 10th/90th percentiles rather than the raw min/max
    # (mirrors plot_run._color_limits): the extremes are a handful of outlier
    # points, and spending the whole viridis range on them flattens the bulk of the
    # data into a narrow band of near-identical colour.  Clipping the tails makes
    # the differences that matter visible; values outside the range simply saturate.
    vmin, vmax = float(np.percentile(Y, 10.0)), float(np.percentile(Y, 90.0))
    if vmax <= vmin:
        vmax = vmin + 1e-9

    # Size the canvas from the slice count so the triangles stay legible: a fixed
    # figure width would shrink every slice as more are added.
    # Figure proportions track the 3D box: the slice stack is wide and short, and
    # a taller canvas just adds white margin mplot3d refuses to give back.
    # Width grows per slice so the triangles stay legible, but is capped: at
    # --stride 1 a long run asks for ~80 slices, and an uncapped canvas would be
    # ~90 inches wide (hundreds of megapixels, and unopenable downstream). Past the
    # cap the slices simply get thinner, which is the honest rendering of "you asked
    # for more slices than fit".
    # The landscape occupies a slot of its own at the head of the stack.
    n_slots = len(keep) + (1 if land is not None else 0)
    fig_w = min(2.0 + 1.15 * n_slots, MAX_SLICE_FIG_W)
    fig_h = min(1.4 + 0.30 * n_slots, MAX_SLICE_FIG_H)
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0.0, 0.0, 0.88, 1.0])

    # Iteration ticks collide once the slices get thin (e.g. --stride 1 on a long
    # run), so past ~12 slices only every Nth is labelled.
    label_every = max(1, int(np.ceil(len(keep) / 12)))

    # The landscape sits at x=0 and the iteration slices start after it, one normal
    # slice-width along — it is a slot in the same sequence, not a separated legend.
    if land is not None:
        gap = landscape_gap if landscape_gap is not None else spacing
        norm = mpl.colors.Normalize(vmin, vmax)
        _draw_landscape_slice(ax, land[0], land[1], 0.0, plt.get_cmap(CMAP), norm,
                              alpha=landscape_alpha)
        ax.plot(*_slice_frame(0.0), color="0.35", lw=0.9,
                alpha=SLICE_OUTLINE_ALPHA, zorder=1)
    else:
        gap = 0.0

    for k, it in enumerate(keep):
        off = gap + k * spacing
        fx, fy, fz = _slice_frame(off)
        ax.plot(fx, fy, fz, color="0.35", lw=0.9, alpha=SLICE_OUTLINE_ALPHA, zorder=1)

        m = iters == it
        ax.scatter(np.full(int(m.sum()), off), xy[m, 0], xy[m, 1],
                   c=Y[m], cmap=CMAP, vmin=vmin, vmax=vmax,
                   s=point_size, alpha=SLICE_POINT_ALPHA,
                   edgecolors="black", linewidths=0.35, depthshade=False, zorder=3)
        # Needles placed at this exact iteration, and nowhere else: a needle marks
        # the moment ZoMBI-Hop committed to a location, so carrying it forward onto
        # later slices would misread as "still being sampled there".
        if len(nxy):
            nm = n_iters == it
            if nm.any():
                ax.scatter(np.full(int(nm.sum()), off), nxy[nm, 0], nxy[nm, 1],
                           marker="*", s=point_size * 6.0, c="red",
                           edgecolors="black", linewidths=0.5, depthshade=False,
                           zorder=6)

        if k % label_every == 0:
            ax.text(off, 0.5, -0.16, f"{int(it)}", ha="center", va="top", fontsize=8,
                    color="0.35")

    # Corner labels once, on the first slice, so the stack stays readable.  The
    # bottom-right label is pushed *down* rather than out: the +y direction points
    # straight at the next slice in this projection, so an outward offset would
    # land the text on top of it.
    for (ly, lz), name, ha, va in (
            ((-0.06, -0.06), LABELS[0], "right", "top"),
            ((1.06, -0.13), LABELS[1], "center", "top"),
            ((0.5, _SQRT3_2 + 0.06), LABELS[2], "center", "bottom")):
        ax.text(0.0, ly, lz, name, ha=ha, va=va, fontsize=8)

    span = gap + (len(keep) - 1) * spacing
    x_lo, x_hi = -0.1, span + 0.1
    y_lo, y_hi = -0.14, 1.16
    z_lo, z_hi = -0.22, _SQRT3_2 + 0.12
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_zlim(z_lo, z_hi)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)
    # Orthographic, not perspective: every slice must be the same size and shape
    # so the reader compares them directly instead of reading the far ones as
    # smaller triangles.
    ax.set_proj_type("ortho")
    # Box aspect proportional to the *data* ranges, which is what keeps each
    # triangle undistorted — anything else (e.g. a fixed (n, 1, 1)) rescales the
    # y/z plane the triangles live in and shears them as the slice count changes.
    ax.set_box_aspect((max(x_hi - x_lo, 1e-6), y_hi - y_lo, z_hi - z_lo))

    sm = mpl.cm.ScalarMappable(cmap=CMAP, norm=mpl.colors.Normalize(vmin, vmax))
    cax = fig.add_axes(_colorbar_rect(fig, ax, span, y_hi, z_hi))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("measured objective", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    # No title: a paper panel gets a caption, and a figure-level title would sit
    # far above the short slice stack and defeat the whitespace trim below.

    # Cap total pixels as well as inches: a wide canvas at 300 dpi still lands in
    # the hundreds of megapixels, which is slow to write and awkward to open.
    dpi = min(SLICE_DPI, (MAX_SLICE_PIXELS / max(fig_w * fig_h, 1e-9)) ** 0.5)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    _trim_png(out_png)
    print(f"Saved figure -> {out_png}")
    if land is not None:
        print(f"  landscape slice: {len(land[1])} grid points from "
              f"{'coverage_ground_truth.npz' if (Path(run_dir) / 'coverage_ground_truth.npz').is_file() else 'rebuilt accordion landscape'}")
    elif not no_landscape:
        print("  landscape slice: SKIPPED — no coverage_ground_truth.npz in the run "
              "dir and the run is not an accordion one; re-run evaluate.py with "
              "--coverage to record the objective it was scored against")
    shown = int(np.isin(n_iters, keep).sum()) if len(n_iters) else 0
    print(f"  {len(np.unique(iters))} iterations, {len(Y)} points, "
          f"colour range (p10-p90) [{vmin:.4f}, {vmax:.4f}]; drew {len(keep)} slice(s)")
    if len(n_iters):
        print(f"  needles: {shown}/{len(n_iters)} drawn "
              f"(a needle only appears if its iteration is among the drawn slices; "
              f"iterations {sorted(int(i) for i in n_iters)})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", action="store_true",
                   help="render only the base landscape panel")
    p.add_argument("--slices", metavar="RUN_DIR", type=Path, default=None,
                   help="render the 3D per-iteration time-slice panel for a run "
                        "directory (the one holding points.csv)")
    p.add_argument("--stride", type=int, default=1,
                   help="draw every Nth iteration (default 1 = all)")
    p.add_argument("--at-iters", default=None, metavar="I,I,...",
                   help="draw exactly these iterations as the slices, overriding "
                        "--stride/--max-slices. Use it to land slices on the "
                        "needle iterations (printed in this script's output), "
                        "which an even stride usually misses entirely.")
    p.add_argument("--max-slices", type=int, default=None,
                   help="cap the number of slices drawn")
    p.add_argument("--no-landscape", action="store_true",
                   help="omit the opaque landscape slice at the head of the stack")
    p.add_argument("--landscape-alpha", type=float, default=LANDSCAPE_ALPHA,
                   help=f"landscape slice opacity, 0-1 (default {LANDSCAPE_ALPHA}, "
                        "matching the sampled points); it stays behind everything "
                        "at any value")
    p.add_argument("--landscape-gap", type=float, default=None,
                   help="extra separation between the landscape slice and the "
                        "first iteration slice (default: --spacing, i.e. the same "
                        "distance as between any two slices)")
    p.add_argument("--spacing", type=float, default=SLICE_SPACING,
                   help=f"gap between slices (default {SLICE_SPACING})")
    p.add_argument("--elev", type=float, default=SLICE_ELEV, help="3D elevation angle")
    p.add_argument("--azim", type=float, default=SLICE_AZIM, help="3D azimuth angle")
    p.add_argument("--out", type=Path, default=None, help="output PNG path")
    p.add_argument("--grid-n", type=int, default=GRID_N,
                   help=f"ternary lattice resolution (default {GRID_N})")
    p.add_argument("--seed", type=int, default=LANDSCAPE_SEED,
                   help="background-feature seed for the landscape")
    p.add_argument("--basin-width", type=float, default=BASIN_WIDTH,
                   help=f"optima sharpness b (default {BASIN_WIDTH}); SMALLER is "
                        "WIDER, Ensemble's own sweep uses 2.2-15")
    p.add_argument("--optima-margin", type=float, default=OPTIMA_MARGIN,
                   help=f"gap between the optima and the tallest background feature, "
                        f"0-0.5 (default {OPTIMA_MARGIN}); higher = the three optima "
                        "dominate more")
    p.add_argument("--no-optima", action="store_true",
                   help="hide the true-optima markers")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    if args.base:
        out = args.out or OUT_DEFAULT
        figure_base(out, grid_n=args.grid_n, seed=args.seed,
                    show_optima=not args.no_optima,
                    basin_width=args.basin_width,
                    optima_margin=args.optima_margin)
        return

    if args.slices is not None:
        out = args.out or OUT_SLICES_DEFAULT
        figure_slices(args.slices, out, stride=args.stride,
                      max_slices=args.max_slices, spacing=args.spacing,
                      elev=args.elev, azim=args.azim,
                      no_landscape=args.no_landscape,
                      landscape_gap=args.landscape_gap,
                      landscape_alpha=args.landscape_alpha,
                      at_iters=([int(v) for v in args.at_iters.split(",") if v.strip()]
                                if args.at_iters else None))
        return

    p.error("nothing to render yet — pass --base or --slices RUN_DIR "
            "(the combined figure is WIP)")


if __name__ == "__main__":
    main()
