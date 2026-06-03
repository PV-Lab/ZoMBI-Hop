"""
make_video.py
=============
Compile all iter_XXXX.png frames from interactive_testing/plots/ into a
30-second MP4.  Frame rate is set to  n_frames / 30  so the video is
always 30 s regardless of how many iterations were saved.

Usage
-----
  conda activate zombi-hop
  python interactive_testing/make_video.py

Output
------
  interactive_testing/zombihop_timelapse.mp4

Dependencies
------------
  pip install imageio[ffmpeg]        # preferred
  -- or --
  pip install opencv-python          # fallback (cv2)
"""

from __future__ import annotations

import os
import sys
import glob

TARGET_DURATION_S = 30.0
MIN_FPS = 1.0
MAX_FPS = 60.0

script_dir = os.path.dirname(os.path.abspath(__file__))
plots_dir  = os.path.join(script_dir, "plots")
out_path   = os.path.join(script_dir, "zombihop_timelapse.mp4")


def _collect_frames(directory: str) -> list[str]:
    pattern = os.path.join(directory, "iter_*.png")
    frames = sorted(glob.glob(pattern))
    if not frames:
        # Also accept any PNG if the naming scheme changed.
        frames = sorted(glob.glob(os.path.join(directory, "*.png")))
    return frames


def _compute_fps(n: int) -> float:
    fps = n / TARGET_DURATION_S
    return max(MIN_FPS, min(MAX_FPS, fps))


def _write_imageio(frames: list[str], out: str, fps: float) -> None:
    import imageio.v3 as iio
    import numpy as np

    print(f"  reading {len(frames)} frames …")
    imgs = [iio.imread(f) for f in frames]
    # Ensure all frames are the same size (use first frame as reference).
    h, w = imgs[0].shape[:2]
    resized = []
    for img in imgs:
        if img.shape[:2] != (h, w):
            from PIL import Image as PILImage
            pil = PILImage.fromarray(img).resize((w, h), PILImage.LANCZOS)
            img = np.array(pil)
        resized.append(img[:, :, :3])   # drop alpha if present

    print(f"  writing MP4 at {fps:.2f} fps …")
    iio.imwrite(
        out, resized,
        plugin="pyav",
        fps=fps,
        codec="libx264",
        output_params=["-crf", "18", "-preset", "fast"],
    )


def _write_opencv(frames: list[str], out: str, fps: float) -> None:
    import cv2

    first = cv2.imread(frames[0])
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out, fourcc, fps, (w, h))

    print(f"  reading {len(frames)} frames …")
    for path in frames:
        img = cv2.imread(path)
        if img is None:
            print(f"  [warn] could not read {path}, skipping.")
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
    writer.release()


def main() -> None:
    if not os.path.isdir(plots_dir):
        sys.exit(f"[error] plots directory not found: {plots_dir}\n"
                 "Run interactive_test_zombi.py first to generate frames.")

    frames = _collect_frames(plots_dir)
    if not frames:
        sys.exit(f"[error] no PNG frames found in {plots_dir}")

    n = len(frames)
    fps = _compute_fps(n)
    print(f"Found {n} frames  →  {fps:.3f} fps  ({TARGET_DURATION_S:.0f}s video)")
    print(f"Output: {out_path}")

    # Try imageio (preferred — handles codec options cleanly).
    try:
        import imageio  # noqa: F401
        _write_imageio(frames, out_path, fps)
        print(f"Done.  Saved to {out_path}")
        return
    except ImportError:
        print("  imageio not found, trying opencv …")
    except Exception as exc:
        print(f"  imageio failed ({exc}), trying opencv …")

    # Fallback: opencv.
    try:
        import cv2  # noqa: F401
        _write_opencv(frames, out_path, fps)
        print(f"Done.  Saved to {out_path}")
        return
    except ImportError:
        pass

    sys.exit(
        "[error] Neither imageio[ffmpeg] nor opencv-python is installed.\n"
        "Install one of them:\n"
        "  pip install 'imageio[ffmpeg]'\n"
        "  pip install opencv-python"
    )


if __name__ == "__main__":
    main()
