"""
optimize/make_videos.py
=======================
Compile per-iteration ternary plot frames (iter_*.png) into timelapse MP4
videos for ZoMBI-Hop MOBO trial folders.

Usage
-----
  python optimize/make_videos.py                         # newest run
  python optimize/make_videos.py <run_dir>               # specific run
  python optimize/make_videos.py <run_dir> --force       # rebuild all
  python optimize/make_videos.py <run_dir>/trial_5       # single trial
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

import imageio.v2 as iio
from PIL import Image as PILImage

VIDEO_TARGET_DURATION_S = 60.0
VIDEO_MIN_FPS           = 1.0
VIDEO_MAX_FPS           = 60.0


def resolve_run_dir(arg: str, runs_dir: str) -> str:
    """Resolve a positional argument to a run directory under runs/.

    arg == "__latest__"  -> newest mobo_* folder under runs_dir.
    otherwise            -> the given path (absolute, cwd-relative, or runs-relative).
    """
    if arg == "__latest__":
        cands = [c for c in glob.glob(os.path.join(runs_dir, "mobo_*")) if os.path.isdir(c)]
        if not cands:
            sys.exit(f"No mobo_* run found under {runs_dir}.")
        return max(cands, key=os.path.getmtime)
    for cand in (arg, os.path.join(runs_dir, arg), os.path.join(os.path.dirname(runs_dir), arg)):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    sys.exit(f"Run directory not found: {arg}")


def make_video_from_dir(plots_dir: str, out_path: str) -> bool:
    """Compile iter_*.png frames in plots_dir into a ~30s MP4 at out_path.

    Returns True on success. Tries imageio+ffmpeg (h264) then OpenCV; both paths
    verify the output is non-empty, since ffmpeg/cv2 can fail without raising.
    """
    frames = sorted(glob.glob(os.path.join(plots_dir, "iter_*.png")))
    if not frames:
        print(f"    [video] no frames in {plots_dir} — skipping.")
        return False
    fps = max(VIDEO_MIN_FPS, min(VIDEO_MAX_FPS, len(frames) / VIDEO_TARGET_DURATION_S))

    def _even(img):
        """Crop to even height/width (libx264 requirement)."""
        h, w = img.shape[:2]
        return img[: h - (h % 2), : w - (w % 2)]

    def _ok() -> bool:
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0

    imgs = [iio.imread(f)[:, :, :3] for f in frames]
    h, w = imgs[0].shape[:2]
    fixed = []
    for img in imgs:
        if img.shape[:2] != (h, w):
            img = np.array(PILImage.fromarray(img).resize((w, h), PILImage.LANCZOS))
        fixed.append(_even(img))
    iio.mimwrite(out_path, fixed, fps=fps, codec="libx264", macro_block_size=None)
    if not _ok():
        raise RuntimeError("imageio/ffmpeg produced an empty file")
    print(f"    [video] {out_path}  ({len(frames)} frames @ {fps:.2f} fps)")
    return True


def regenerate_videos(run_dir: str, force: bool = False) -> None:
    """Rebuild zombihop_timelapse.mp4 for every trial_* folder from its frames.

    Skips trials that already have a non-empty video unless force=True.
    """
    trial_dirs = sorted(
        glob.glob(os.path.join(run_dir, "trial_*")),
        key=lambda p: int(p.split("_")[-1]) if p.split("_")[-1].isdigit() else 0,
    )
    if not trial_dirs:
        print(f"No trial_* folders found in {run_dir}")
        return
    n_ok = n_skip = n_fail = 0
    for tdir in trial_dirs:
        plots_dir = os.path.join(tdir, "plots")
        out_path  = os.path.join(tdir, "zombihop_timelapse.mp4")
        name = os.path.basename(tdir)
        if not os.path.isdir(plots_dir) or not glob.glob(os.path.join(plots_dir, "iter_*.png")):
            print(f"  {name}: no frames — skipping."); continue
        if not force and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"  {name}: video already present — skipping (use --force to rebuild).")
            n_skip += 1; continue
        print(f"  {name}: building video …")
        if make_video_from_dir(plots_dir, out_path):
            n_ok += 1
        else:
            n_fail += 1
    print(f"\nVideos: {n_ok} written, {n_skip} skipped, {n_fail} failed.  ({run_dir})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile per-iteration frames into timelapse MP4 videos for MOBO trials.")
    parser.add_argument("run_dir", nargs="?", default="__latest__",
                        help="Path to a runs/mobo_* folder (default: newest run).")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild videos even if one already exists.")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir   = os.path.join(script_dir, "runs")

    run_dir = resolve_run_dir(args.run_dir, runs_dir)

    if os.path.basename(run_dir).startswith("trial_"):
        plots_dir = os.path.join(run_dir, "plots")
        out_path  = os.path.join(run_dir, "zombihop_timelapse.mp4")
        print(f"Building video for {run_dir}")
        make_video_from_dir(plots_dir, out_path)
    else:
        print(f"Regenerating videos for {run_dir}")
        regenerate_videos(run_dir, force=args.force)


if __name__ == "__main__":
    main()
