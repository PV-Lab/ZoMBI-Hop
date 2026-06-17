"""Show a coverage plot: ground-truth landscape + all sampled points, and
produce a coverage.mp4 video of points appearing over time.

3D (ternary) datasets render as a triangle; 4D datasets render as a tetrahedron
(the 4-simplex), reusing the geometry from ``synthetic_data/plot.py``. The 4D
video slowly rotates so the structure reads in 3D.

Usage
-----
    python -m optimize.coverage_plot <trial_dir>

``trial_dir`` is any folder that contains ``points.csv`` and whose parent (or
itself) contains ``run_config.json`` or ``rerun_config.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the "3d" projection)
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestRegressor

VIDEO_TARGET_DURATION_S = 60.0
VIDEO_MIN_FPS = 1.0
VIDEO_MAX_FPS = 60.0

# ─── Ternary geometry (mirrors run_mobo.py) ──────────────────────────────────

_SQRT3_2 = math.sqrt(3) / 2
CORNER_LABELS = ("FAPbI₃", "MAPbI₃", "MAPbBr₃")
GRID_N = 80
RF_N_ESTIMATORS = 500


def comp_to_xy(comp: np.ndarray) -> np.ndarray:
    p = np.asarray(comp, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    s = p.sum(axis=-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return np.column_stack([p[:, 1] + 0.5 * p[:, 2], _SQRT3_2 * p[:, 2]])


def ternary_grid(n: int = GRID_N) -> np.ndarray:
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append([i / n, j / n, (n - i - j) / n])
    return np.array(pts, dtype=float)


def draw_ternary_frame(ax, pad: float = 0.04) -> None:
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, _SQRT3_2 + 0.16)
    ax.axis("off")
    ax.text(-pad, -pad, CORNER_LABELS[0], ha="right", va="top", fontsize=10)
    ax.text(1 + pad, -pad, CORNER_LABELS[1], ha="left", va="top", fontsize=10)
    ax.text(0.5, _SQRT3_2 + pad, CORNER_LABELS[2], ha="center", va="bottom", fontsize=10)


# ─── Quaternary geometry (4-simplex → tetrahedron, mirrors plot.py) ────────

GRID_N_4D = 24
VERTEX_LABELS_4D = ("x1", "x2", "x3", "x4")

TETRA_VERTICES = np.array([
    [1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
], dtype=float)
TETRA_VERTICES = TETRA_VERTICES - TETRA_VERTICES.mean(axis=0)


def comp_to_xyz(comp: np.ndarray) -> np.ndarray:
    p = np.asarray(comp, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    s = p.sum(axis=-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return p @ TETRA_VERTICES


def tetra_grid(n: int = GRID_N_4D) -> np.ndarray:
    pts = [
        (i, j, k, n - i - j - k)
        for i in range(n + 1)
        for j in range(n + 1 - i)
        for k in range(n + 1 - i - j)
    ]
    return np.array(pts, dtype=float) / n


def draw_tetra_frame(ax) -> None:
    for i in range(4):
        for j in range(i + 1, 4):
            seg = TETRA_VERTICES[[i, j]]
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], "k-", lw=1.0, alpha=0.4, zorder=1)
    for p, lab in zip(TETRA_VERTICES * 1.15, VERTEX_LABELS_4D):
        ax.text(p[0], p[1], p[2], lab, fontsize=11, ha="center", va="center")
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


# ─── Ground-truth builders ───────────────────────────────────────────────────

def _build_rf_ground_truth(csv_path: str):
    df = pd.read_csv(csv_path).dropna(subset=["FAPbI3", "MAPbI3", "MAPbBr3", "Objective"])
    X = df[["FAPbI3", "MAPbI3", "MAPbBr3"]].values.astype(float)
    X /= X.sum(axis=1, keepdims=True)
    y = df["Objective"].values.astype(float)
    rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    grid_pts = ternary_grid()
    grid_vals = rf.predict(grid_pts)
    return grid_pts, grid_vals


def _build_ackley_ground_truth(variant: str, dim: int):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from synthetic_data.ackley import Ackley
    fn = Ackley(variant, dim=dim)
    grid_pts = ternary_grid() if dim == 3 else tetra_grid()
    grid_vals = fn.predict(grid_pts)
    return grid_pts, grid_vals


# ─── Config discovery ────────────────────────────────────────────────────────

def _find_config(trial_dir: str) -> dict:
    """Search trial_dir and ancestors for run_config.json or rerun_config.json."""
    for name in ("run_config.json", "rerun_config.json"):
        path = os.path.join(trial_dir, name)
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)

    parent = os.path.dirname(os.path.normpath(trial_dir))
    for name in ("run_config.json", "rerun_config.json"):
        path = os.path.join(parent, name)
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)

    grandparent = os.path.dirname(parent)
    for name in ("run_config.json", "rerun_config.json"):
        path = os.path.join(grandparent, name)
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)

    sys.exit(f"Could not find run_config.json or rerun_config.json in or above {trial_dir}")


def _detect_comp_cols(df: pd.DataFrame, dim: int = 3) -> list[str]:
    if dim == 3 and {"FA", "MA", "Br"}.issubset(df.columns):
        return ["FA", "MA", "Br"]
    x_cols = sorted([c for c in df.columns if c.startswith("x") and c[1:].isdigit()],
                    key=lambda c: int(c[1:]))
    if len(x_cols) >= dim:
        return x_cols[:dim]
    sys.exit(f"Cannot detect {dim} composition columns in points.csv")


# ─── Video rendering ─────────────────────────────────────────────────────────

def _stamp_counter(img: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image as PILImage, ImageDraw, ImageFont  # lazy (video path only)
    pil = PILImage.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", size=28)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    margin, pad = 12, 6
    x = pil.width - tw - margin
    y = pil.height - th - margin
    draw.rounded_rectangle(
        [x - pad, y - pad, x + tw + pad, y + th + pad],
        radius=6, fill=(0, 0, 0, 180),
    )
    draw.text((x, y), text, fill="white", font=font)
    return np.array(pil)


def _render_coverage_frame(
    n_visible: int,
    pxy: np.ndarray,
    y_vals: np.ndarray,
    gxy: np.ndarray,
    grid_vals: np.ndarray,
    true_optima: list,
    maximize: bool,
    title_base: str,
    needle_xy: np.ndarray | None = None,
    azim: float | None = None,  # unused (ternary is 2D); kept for a uniform signature
) -> np.ndarray:
    fig = Figure(figsize=(8, 7))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    draw_ternary_frame(ax)

    ax.scatter(gxy[:, 0], gxy[:, 1], c=grid_vals, cmap="viridis",
               s=6, alpha=0.72, zorder=2, rasterized=True)

    if n_visible > 0:
        ax.scatter(pxy[:n_visible, 0], pxy[:n_visible, 1],
                   c=y_vals[:n_visible], cmap="viridis",
                   vmin=grid_vals.min(), vmax=grid_vals.max(),
                   s=40, alpha=1.0, zorder=5,
                   edgecolors="black", linewidths=0.6)

    if true_optima:
        mxy = comp_to_xy(np.array(true_optima))
        label = "True maxima" if maximize else "True minima"
        ax.scatter(mxy[:, 0], mxy[:, 1], marker="*", s=360, c="blue", alpha=0.45,
                   zorder=11, edgecolors="navy", linewidths=1.3, label=label)

    if needle_xy is not None and len(needle_xy):
        ax.scatter(needle_xy[:, 0], needle_xy[:, 1], marker="*", s=360,
                   c="red", alpha=0.55, zorder=12, edgecolors="darkred",
                   linewidths=1.3, label="Found needles")

    if true_optima or (needle_xy is not None and len(needle_xy)):
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_title(f"Coverage: {title_base}  ({n_visible} points)", fontsize=12)
    fig.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    fig.clear()
    return img


def _render_coverage_frame_3d(
    n_visible: int,
    pxy: np.ndarray,
    y_vals: np.ndarray,
    gxy: np.ndarray,
    grid_vals: np.ndarray,
    true_optima: list,
    maximize: bool,
    title_base: str,
    needle_xy: np.ndarray | None = None,
    azim: float | None = None,
) -> np.ndarray:
    fig = Figure(figsize=(8, 7))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    draw_tetra_frame(ax)

    ax.scatter(gxy[:, 0], gxy[:, 1], gxy[:, 2], c=grid_vals, cmap="viridis",
               s=5, alpha=0.12, zorder=2, depthshade=False, rasterized=True)

    if n_visible > 0:
        ax.scatter(pxy[:n_visible, 0], pxy[:n_visible, 1], pxy[:n_visible, 2],
                   c=y_vals[:n_visible], cmap="viridis",
                   vmin=grid_vals.min(), vmax=grid_vals.max(),
                   s=40, alpha=1.0, zorder=5, depthshade=False,
                   edgecolors="black", linewidths=0.6)

    if true_optima:
        mxy = comp_to_xyz(np.array(true_optima))
        label = "True maxima" if maximize else "True minima"
        ax.scatter(mxy[:, 0], mxy[:, 1], mxy[:, 2], marker="*", s=360, c="blue",
                   alpha=0.45, zorder=11, edgecolors="navy", linewidths=1.3,
                   depthshade=False, label=label)

    if needle_xy is not None and len(needle_xy):
        ax.scatter(needle_xy[:, 0], needle_xy[:, 1], needle_xy[:, 2], marker="*",
                   s=360, c="red", alpha=0.55, zorder=12, edgecolors="darkred",
                   linewidths=1.3, depthshade=False, label="Found needles")

    if true_optima or (needle_xy is not None and len(needle_xy)):
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax.view_init(elev=18, azim=(-60 if azim is None else azim))
    ax.set_title(f"Coverage: {title_base}  ({n_visible} points)", fontsize=12)
    fig.tight_layout()

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[:, :, :3].copy()
    fig.clear()
    return img


POINTS_PER_LINE = 24


def _line_boundaries(n_total: int) -> list[int]:
    """Cumulative point counts at each line boundary (one frame per line)."""
    boundaries = list(range(POINTS_PER_LINE, n_total, POINTS_PER_LINE))
    if not boundaries or boundaries[-1] != n_total:
        boundaries.append(n_total)
    return boundaries


def make_coverage_video(
    out_path: str,
    pxy: np.ndarray,
    y_vals: np.ndarray,
    gxy: np.ndarray,
    grid_vals: np.ndarray,
    true_optima: list,
    maximize: bool,
    title_base: str,
    needle_xy: np.ndarray | None = None,
    render_frame=_render_coverage_frame,
    rotate: bool = False,
) -> None:
    n_total = len(pxy)
    boundaries = _line_boundaries(n_total)
    n_frames = len(boundaries)
    fps = max(VIDEO_MIN_FPS, min(VIDEO_MAX_FPS, n_frames / VIDEO_TARGET_DURATION_S))

    def _even(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        return img[: h - (h % 2), : w - (w % 2)]

    bar_width = 40
    frames = []
    for frame_idx, n_visible in enumerate(boundaries, 1):
        azim = -60 + 360 * (frame_idx - 1) / max(n_frames, 1) if rotate else None
        img = render_frame(
            n_visible, pxy, y_vals, gxy, grid_vals, true_optima, maximize, title_base,
            needle_xy=needle_xy, azim=azim)
        img = _stamp_counter(img, f"Points sampled: {n_visible}")
        frames.append(_even(img))
        filled = int(bar_width * frame_idx / n_frames)
        bar = "█" * filled + "░" * (bar_width - filled)
        pct = 100 * frame_idx / n_frames
        print(f"\r  Rendering: |{bar}| {pct:5.1f}%  ({frame_idx}/{n_frames})", end="", flush=True)
    print()

    print("  Encoding video ...", end="", flush=True)
    import imageio.v2 as iio  # lazy: only the video path needs imageio/ffmpeg
    iio.mimwrite(out_path, frames, fps=fps, codec="libx264", macro_block_size=None)
    if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        raise RuntimeError(f"Failed to write video: {out_path}")
    print(f"\r  [video] {out_path}  ({n_frames} frames @ {fps:.2f} fps)")


# ─── Data preparation + static figure (shared by the CLI and run_mobo) ─────────

def _prepare_coverage(trial_dir: str) -> dict:
    """Load points.csv + config + ground-truth grid for ``trial_dir``.

    Raises ``FileNotFoundError`` / ``ValueError`` on bad input rather than calling
    ``sys.exit`` so it is safe to call from inside a running optimisation (which
    catches exceptions but not ``SystemExit``). Returns everything the static
    figure and the video need.
    """
    points_path = os.path.join(trial_dir, "points.csv")
    if not os.path.isfile(points_path):
        raise FileNotFoundError(f"No points.csv found in {trial_dir}")

    cfg = _find_config(trial_dir)
    df = pd.read_csv(points_path)

    dataset = cfg.get("dataset", "RF")
    dim = cfg.get("dim", 3)
    if dim not in (3, 4):
        raise ValueError(f"Coverage plot only supports 3D (ternary) or 4D "
                         f"(tetrahedron) datasets (got dim={dim})")

    comp_cols = _detect_comp_cols(df, dim)
    map_comp = comp_to_xy if dim == 3 else comp_to_xyz

    if dataset == "RF":
        if dim != 3:
            raise ValueError(f"RF surrogate ground truth is only available for dim=3 "
                             f"(got dim={dim})")
        csv_path = cfg["csv_path"]
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Surrogate CSV not found: {csv_path}")
        grid_pts, grid_vals = _build_rf_ground_truth(csv_path)
    else:
        variant = cfg.get("ackley_variant", "realistic")
        grid_pts, grid_vals = _build_ackley_ground_truth(variant, dim)

    true_optima = cfg.get("true_optima", [])
    maximize = cfg.get("maximize", True)

    comps = df[comp_cols].values.astype(float)
    y_vals = df["Y"].values.astype(float)

    needles_path = os.path.join(trial_dir, "needles.csv")
    needle_xy = None
    if os.path.isfile(needles_path):
        ndf = pd.read_csv(needles_path)
        ncols = _detect_comp_cols(ndf, dim)
        needle_xy = map_comp(ndf[ncols].values.astype(float))

    return dict(
        dim=dim, map_comp=map_comp,
        gxy=map_comp(grid_pts), grid_vals=grid_vals,
        pxy=map_comp(comps), y_vals=y_vals,
        true_optima=true_optima, maximize=maximize,
        needle_xy=needle_xy, n_pts=len(df),
        title_base=os.path.basename(os.path.normpath(trial_dir)),
    )


def _build_static_figure(prep: dict):
    """Build (but do not show/save) the static coverage Figure from ``_prepare``."""
    dim        = prep["dim"]
    map_comp   = prep["map_comp"]
    gxy        = prep["gxy"]
    grid_vals  = prep["grid_vals"]
    pxy        = prep["pxy"]
    y_vals     = prep["y_vals"]
    true_optima = prep["true_optima"]
    maximize   = prep["maximize"]
    needle_xy  = prep["needle_xy"]
    n_pts      = prep["n_pts"]
    title_base = prep["title_base"]

    fig = plt.figure(figsize=(8, 7))
    if dim == 3:
        ax = fig.add_subplot(111)
        draw_ternary_frame(ax)
        sc_bg = ax.scatter(gxy[:, 0], gxy[:, 1], c=grid_vals, cmap="viridis",
                           s=6, alpha=0.72, zorder=2, rasterized=True)
        ax.scatter(pxy[:, 0], pxy[:, 1], c=y_vals, cmap="viridis",
                   vmin=grid_vals.min(), vmax=grid_vals.max(),
                   s=40, alpha=1.0, zorder=5, edgecolors="black", linewidths=0.6)
        if true_optima:
            mxy = map_comp(np.array(true_optima))
            label = "True maxima" if maximize else "True minima"
            ax.scatter(mxy[:, 0], mxy[:, 1], marker="*", s=360, c="blue", alpha=0.45,
                       zorder=11, edgecolors="navy", linewidths=1.3, label=label)
        if needle_xy is not None and len(needle_xy):
            ax.scatter(needle_xy[:, 0], needle_xy[:, 1], marker="*", s=360,
                       c="red", alpha=0.55, zorder=12, edgecolors="darkred",
                       linewidths=1.3, label="Found needles")
    else:
        ax = fig.add_subplot(111, projection="3d")
        draw_tetra_frame(ax)
        sc_bg = ax.scatter(gxy[:, 0], gxy[:, 1], gxy[:, 2], c=grid_vals, cmap="viridis",
                           s=5, alpha=0.12, zorder=2, depthshade=False, rasterized=True)
        ax.scatter(pxy[:, 0], pxy[:, 1], pxy[:, 2], c=y_vals, cmap="viridis",
                   vmin=grid_vals.min(), vmax=grid_vals.max(),
                   s=40, alpha=1.0, zorder=5, depthshade=False,
                   edgecolors="black", linewidths=0.6)
        if true_optima:
            mxy = map_comp(np.array(true_optima))
            label = "True maxima" if maximize else "True minima"
            ax.scatter(mxy[:, 0], mxy[:, 1], mxy[:, 2], marker="*", s=360, c="blue",
                       alpha=0.45, zorder=11, edgecolors="navy", linewidths=1.3,
                       depthshade=False, label=label)
        if needle_xy is not None and len(needle_xy):
            ax.scatter(needle_xy[:, 0], needle_xy[:, 1], needle_xy[:, 2], marker="*",
                       s=360, c="red", alpha=0.55, zorder=12, edgecolors="darkred",
                       linewidths=1.3, depthshade=False, label="Found needles")
        ax.view_init(elev=18, azim=-60)

    fig.colorbar(sc_bg, ax=ax, label="Objective", fraction=0.046, pad=0.04)
    if true_optima or (needle_xy is not None and len(needle_xy)):
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title(f"Coverage: {title_base}  ({n_pts} points)", fontsize=12)
    fig.tight_layout()
    return fig


def save_coverage_image(trial_dir: str, out_path: str | None = None) -> str:
    """Render the static coverage plot to a PNG — no video, no interactive window.

    This is the entry point ``run_mobo.py`` calls automatically at the end of a
    trial (3D ternary / 4D tetrahedron only). Returns the path written.
    """
    if out_path is None:
        out_path = os.path.join(trial_dir, "coverage.png")
    fig = _build_static_figure(_prepare_coverage(trial_dir))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ternary/tetrahedron coverage plot + video")
    parser.add_argument("trial_dir", help="Path to a trial_* folder (or any folder with points.csv)")
    parser.add_argument("--no-video", action="store_true",
                        help="Only show/save the static coverage plot; skip the mp4 video.")
    args = parser.parse_args()

    trial_dir = os.path.abspath(args.trial_dir)
    try:
        prep = _prepare_coverage(trial_dir)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))

    # ── Static plot (shown interactively) ──
    fig = _build_static_figure(prep)
    plt.show()

    if args.no_video:
        return

    # ── Video (saved to trial folder) ──
    dim = prep["dim"]
    video_path = os.path.join(trial_dir, "coverage.mp4")
    print(f"Rendering coverage video ({prep['n_pts']} points) ...")
    render_frame = _render_coverage_frame if dim == 3 else _render_coverage_frame_3d
    make_coverage_video(
        video_path, prep["pxy"], prep["y_vals"], prep["gxy"], prep["grid_vals"],
        prep["true_optima"], prep["maximize"], prep["title_base"],
        needle_xy=prep["needle_xy"], render_frame=render_frame, rotate=(dim == 4))


if __name__ == "__main__":
    main()
