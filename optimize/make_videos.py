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
import csv
import glob
import os
import re
import shutil
import subprocess
import sys

import numpy as np

from PIL import Image as PILImage, ImageDraw, ImageFont

VIDEO_TARGET_DURATION_S = 60.0
VIDEO_MIN_FPS           = 1.0
VIDEO_MAX_FPS           = 60.0

POINTS_PER_ITERATION = 24
N_INIT_LINES         = 2
INIT_POINTS          = N_INIT_LINES * POINTS_PER_ITERATION


def _read_rgb(path: str) -> np.ndarray:
    """Load an RGB uint8 frame (imageio if present, else Pillow)."""
    try:
        import imageio.v2 as iio
        return iio.imread(path)[:, :, :3]
    except Exception:
        return np.asarray(PILImage.open(path).convert("RGB"))


def _write_mp4_imageio(frames: list[np.ndarray], out_path: str, fps: float) -> bool:
    try:
        import imageio.v2 as iio
    except ImportError:
        return False
    try:
        iio.mimwrite(out_path, frames, fps=fps, codec="libx264", macro_block_size=None)
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception as exc:
        print(f"    [video] imageio failed: {exc}")
        return False


def _write_mp4_cv2(frames: list[np.ndarray], out_path: str, fps: float) -> bool:
    try:
        import cv2
    except ImportError:
        return False
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        return False
    try:
        for img in frames:
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


def _write_mp4_ffmpeg(frames: list[np.ndarray], out_path: str, fps: float) -> bool:
    """Pipe raw RGB frames to system ffmpeg (no imageio needed)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    h, w = frames[0].shape[:2]
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", f"{fps:.4f}",
        "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        out_path,
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        assert proc.stdin is not None
        for img in frames:
            proc.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
        proc.stdin.close()
        rc = proc.wait(timeout=600)
    except Exception as exc:
        print(f"    [video] ffmpeg failed: {exc}")
        return False
    return rc == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0


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


def _iter_number(path: str) -> int | None:
    m = re.search(r"iter_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def _total_points_for_frame(iter_num: int | None) -> int:
    if iter_num is None:
        return 0
    return INIT_POINTS + (iter_num + 1) * POINTS_PER_ITERATION


def _stamp_counter(img: np.ndarray, text: str) -> np.ndarray:
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


def make_video_from_dir(plots_dir: str, out_path: str) -> bool:
    """Compile iter_*.png frames in plots_dir into a ~60s MP4 at out_path.

    Returns True on success. Tries imageio+ffmpeg, then OpenCV, then system
    ffmpeg CLI. Verifies the output is non-empty.
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

    imgs = [_read_rgb(f) for f in frames]
    h, w = imgs[0].shape[:2]
    fixed = []
    for i, img in enumerate(imgs):
        if img.shape[:2] != (h, w):
            img = np.array(PILImage.fromarray(img).resize((w, h), PILImage.LANCZOS))
        n_pts = _total_points_for_frame(_iter_number(frames[i]))
        if n_pts > 0:
            img = _stamp_counter(img, f"Points sampled: {n_pts}")
        fixed.append(_even(img))

    for writer, name in (
        (_write_mp4_imageio, "imageio"),
        (_write_mp4_cv2, "cv2"),
        (_write_mp4_ffmpeg, "ffmpeg"),
    ):
        if writer(fixed, out_path, fps):
            print(f"    [video] {out_path}  ({len(frames)} frames @ {fps:.2f} fps via {name})")
            return True
    print(f"    [video] FAILED to write {out_path} (tried imageio, cv2, ffmpeg)")
    return False


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

    plots_dir = os.path.join(run_dir, "plots")
    out_path  = os.path.join(run_dir, "zombihop_timelapse.mp4")
    flat_run  = (
        os.path.isdir(plots_dir)
        and glob.glob(os.path.join(plots_dir, "iter_*.png"))
    )

    if os.path.basename(run_dir).startswith("trial_") or flat_run:
        if not args.force and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"Video already present: {out_path}  (use --force to rebuild)")
        else:
            print(f"Building video for {run_dir}")
            make_video_from_dir(plots_dir, out_path)
    else:
        print(f"Regenerating videos for {run_dir}")
        regenerate_videos(run_dir, force=args.force)


if __name__ == "__main__":
    main()
