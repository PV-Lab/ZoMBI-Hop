"""
visualization/pareto_animation.py
=================================
Animate a single MOBO run's Pareto front being built up one trial at a time.

Each trial (in trial order) is revealed one point at a time; on every frame the
Pareto-optimal / dominated split is recomputed over *only the points revealed so
far*, so a point that looked Pareto-optimal early can be greyed out once a later
point dominates it. Uses pareto.py's ``--pretty`` styling (full-title axis
labels, soft-grey dominated dots, bright-red Pareto stars).

The three pairwise objective panels (dist vs dup, dist vs time, dup vs time) are
laid out with one panel centered on the top row and the other two below it, so
the three consecutive plots stay compact. The output MP4 is a fixed 30 s.

Usage
-----
  python visualization/pareto_animation.py                 # default run below
  python visualization/pareto_animation.py <run_dir>       # a specific run dir
  python visualization/pareto_animation.py <run_dir> --out my.mp4 --seconds 30
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import imageio.v2 as iio

# Import the collection + Pareto machinery from optimize/pareto.py so the metric
# keys, filtering, and non-domination logic stay identical to the static plot.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OPTIMIZE_DIR = os.path.join(_REPO_ROOT, "optimize")
if _OPTIMIZE_DIR not in sys.path:
    sys.path.insert(0, _OPTIMIZE_DIR)

from pareto import (  # noqa: E402
    collect_trials,
    pareto_mask_min,
    _obj_pairs,
    _pretty_label,
    DIST_KEY,
    DUP_KEY,
    TIME_KEYS,
)

DEFAULT_RUN = os.path.join(_OPTIMIZE_DIR, "runs", "mobo_05_06_15_32")
DEFAULT_SECONDS = 30.0


def _build_matrix(records: list[dict]) -> tuple[np.ndarray, list[str]]:
    """(M, obj_labels) for the three live objectives, ordered by trial number.

    Rows follow trial order so the reveal proceeds in the order trials were run.
    The third (time) axis label tracks whichever time key the trials recorded
    (avg_time_per_iter_s / runtime_s), matching pareto.py.
    """
    records = sorted(records, key=lambda r: (r.get("trial") is None, r.get("trial", 0)))
    time_keys = {r["time_key"] for r in records}
    if time_keys == {"runtime_s"}:
        time_label = "runtime_s"
    elif time_keys == {"avg_time_per_iter_s"}:
        time_label = "avg_time_per_iter_s"
    else:
        time_label = "avg_time_per_iter_s | runtime_s (MIXED)"
    obj_labels = [DIST_KEY, DUP_KEY, time_label]
    M = np.array(
        [[r["metrics"][DIST_KEY], r["metrics"][DUP_KEY], r["time_value"]] for r in records],
        dtype=float,
    )
    return M, obj_labels


def _padded_limits(vals: np.ndarray, frac: float = 0.05) -> tuple[float, float]:
    """Fixed axis limits with a small margin so points never sit on the frame."""
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi == lo:
        pad = abs(hi) * frac or 1.0
        return lo - pad, hi + pad
    pad = (hi - lo) * frac
    return lo - pad, hi + pad


def render_animation(
    M: np.ndarray,
    obj_labels: list[str],
    out_path: str,
    seconds: float = DEFAULT_SECONDS,
) -> None:
    """Render the reveal animation to an MP4 of exactly ``seconds`` duration."""
    pairs = _obj_pairs(obj_labels)          # (0,1), (0,2), (1,2)
    n = len(M)

    # Fixed limits over the whole run so points appear in place and stay put.
    lims = [_padded_limits(M[:, i]) for i in range(M.shape[1])]

    # --- Static figure/axes; only the scatter data changes per frame. ---
    fig = plt.figure(figsize=(10, 10), dpi=100)
    # 2 rows x 4 cols: top panel centered over cols 1:3, bottom two over 0:2, 2:4.
    # Tight margins + small gaps keep the three panels roughly square with little
    # whitespace between the rows.
    gs = GridSpec(2, 4, figure=fig, hspace=0.16, wspace=0.42,
                  left=0.09, right=0.97, top=0.97, bottom=0.06)
    axes = [
        fig.add_subplot(gs[0, 1:3]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
    ]

    dom_artists, par_artists = [], []
    for ax, (ix, iy, xl, yl) in zip(axes, pairs):
        dom = ax.scatter([], [], c="#707070", alpha=0.45, s=70,
                         edgecolors="none", label="dominated")
        par = ax.scatter([], [], marker="*", s=420, c="red", zorder=5,
                         edgecolors="k", linewidths=0.5, label="Pareto")
        dom_artists.append(dom)
        par_artists.append(par)
        ax.set_xlim(*lims[ix])
        ax.set_ylim(*lims[iy])
        ax.set_xlabel(_pretty_label(xl))
        ax.set_ylabel(_pretty_label(yl))
        ax.legend(fontsize=8, loc="upper right")

    # One frame per revealed point; frame k shows points 0..k (k = 1..n).
    n_frames = n
    fps = n_frames / seconds

    def _frame(k: int) -> np.ndarray:
        """Render frame with the first ``k`` points revealed; return RGB array."""
        sub = M[:k]
        mask = pareto_mask_min(sub) if k else np.zeros(0, dtype=bool)
        n_par = int(mask.sum())
        for (ix, iy, _, _), dom, par in zip(pairs, dom_artists, par_artists):
            dom.set_offsets(sub[~mask][:, [ix, iy]] if k else np.empty((0, 2)))
            par.set_offsets(sub[mask][:, [ix, iy]] if n_par else np.empty((0, 2)))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        return buf[:, :, :3].copy()

    print(f"  Rendering {n_frames} frames @ {fps:.2f} fps  ->  {seconds:.0f}s video ...")
    frames = [_frame(k) for k in range(1, n_frames + 1)]

    # Even dims for libx264.
    h, w = frames[0].shape[:2]
    if h % 2 or w % 2:
        frames = [f[: h - (h % 2), : w - (w % 2)] for f in frames]

    iio.mimwrite(out_path, frames, fps=fps, codec="libx264", macro_block_size=None)
    plt.close(fig)
    if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
        raise RuntimeError("ffmpeg produced an empty file")
    print(f"  Pareto animation -> {out_path}  ({n_frames} frames @ {fps:.2f} fps)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Animate a MOBO run's Pareto front being built up one trial at a time.")
    parser.add_argument("run_dir", nargs="?", default=DEFAULT_RUN,
                        help=f"MOBO run directory (default: {DEFAULT_RUN}).")
    parser.add_argument("--out", default=None,
                        help="Output MP4 path (default: <run_dir>/pareto_animation.mp4).")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                        help="Video duration in seconds (default: 30).")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    out_path = os.path.abspath(args.out) if args.out else os.path.join(run_dir, "pareto_animation.mp4")

    print("=" * 70)
    print(f"Pareto animation  |  {run_dir}")
    print("=" * 70)

    records = collect_trials(run_dir)
    if not records:
        sys.exit(f"No usable trials found under {run_dir}.")
    M, obj_labels = _build_matrix(records)
    print(f"  {len(M)} trial(s) collected; objectives: {obj_labels}")

    render_animation(M, obj_labels, out_path, seconds=args.seconds)


if __name__ == "__main__":
    main()
