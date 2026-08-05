#!/usr/bin/env python3
"""Re-run the 3 best pair-MOBO hparams on both ELA twins with full zoom-bound frames.

Sets ZOMBI_SAVE_ALL_FRAMES=1 so each iteration writes plots/iter_XXXX.png (with
trust-region bounds), then compiles zombihop_timelapse.mp4 via make_videos.

Usage
-----
  ZOMBI_SAVE_ALL_FRAMES=1 python optimize/rerun_pair_top3_zoom_videos.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "optimize"))

os.environ["ZOMBI_SAVE_ALL_FRAMES"] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import run_mobo as rm  # noqa: E402
from mobo_landscapes import build_ela_landscape  # noqa: E402

OUT = REPO / "optimize" / "runs" / "pair_lowest_dist_compare_edge01" / "zoom_videos"
STAGE = REPO / "optimize" / "runs" / "transfer_pair_top3_edge01" / "hparams_stage"
OPTIMA_DIR = REPO / "warm_start" / "optima_finder_ela_rf_edge_sweep" / "edge_0.01"

TRIALS = [
    ("job19365317_t167", 167, "mobo_ela_rf_g_interior20_pair_job19365317"),
    ("job19369437_t103", 103, "mobo_ela_rf_g_interior20_pair_job19369437"),
    ("job19369437_t179", 179, "mobo_ela_rf_g_interior20_pair_job19369437"),
]
TWINS = [
    ("run_1", "ela_3d_18535497"),
    ("run_2", "ela_3d_18503666"),
]

# Match the original pair MOBO wall budget (batch JSON time_limit_hours=0.2).
TIME_LIMIT_HOURS = 0.2


def load_optima(twin: str):
    data = json.loads((OPTIMA_DIR / f"{twin}_optima.json").read_text())
    import numpy as np
    return [np.asarray(x, float) for x in data["true_optima"]]


def load_hparams(trial_num: int) -> dict:
    path = STAGE / f"trial_{trial_num}" / "trial.json"
    if path.is_file():
        return json.loads(path.read_text())["hparams"]
    # fall back to source pair run
    for tag, tn, run in TRIALS:
        if tn == trial_num:
            src = REPO / "optimize" / "runs" / run / f"trial_{tn}" / "trial.json"
            return json.loads(src.read_text())["hparams"]
    raise FileNotFoundError(trial_num)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rm._configure_mpl_backend(headless=True)
    rm._apply_runtime_overrides(device=None, time_limit_hours=TIME_LIMIT_HOURS)

    summary = []
    for tag, tn, _src in TRIALS:
        hp = load_hparams(tn)
        print(f"\n======== {tag} (trial_{tn}) ========", flush=True)
        for run_name, twin in TWINS:
            out_dir = OUT / tag / run_name
            if out_dir.exists():
                shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True)

            # Stub run_config so coverage_plot / _auto_generate_plots don't sys.exit.
            (out_dir / "run_config.json").write_text(json.dumps({
                "dataset": "ela",
                "landscape": "ela",
                "dim": 3,
                "maximize": True,
                "ela_run": f"ela/runs/{twin}",
                "true_optima_source": "edge_min=0.01",
            }, indent=2) + "\n")

            optima = load_optima(twin)
            ls = build_ela_landscape(
                REPO / "ela" / "runs" / twin,
                maximize=True,
                true_optima=optima,
                time_limit_hours=TIME_LIMIT_HOURS,
                repo_root=str(REPO),
                use_rf_g=True,
            )
            print(f"\n--- {tag} {run_name} {twin}  n_optima={len(optima)} ---", flush=True)
            result = rm.run_single_trial(hp, ls, str(out_dir))
            plots = out_dir / "plots"
            n_frames = len(list(plots.glob("iter_*.png"))) if plots.is_dir() else 0
            mp4 = out_dir / "zombihop_timelapse.mp4"
            ok = False
            if n_frames:
                try:
                    import make_videos as mv
                    ok = mv.make_video_from_dir(str(plots), str(mp4))
                except Exception as exc:
                    print(f"  [video] assembly failed ({exc}); frames kept in {plots}",
                          flush=True)
            # also mirror into the comparison videos folder
            dest = OUT.parent / "videos" / tag / run_name
            dest.mkdir(parents=True, exist_ok=True)
            if ok and mp4.is_file():
                shutil.copy2(mp4, dest / "zombihop_timelapse.mp4")
            row = {
                "trial": tag,
                "run": run_name,
                "twin": twin,
                "dist": result.get("dist"),
                "n_iters": result.get("n_iters"),
                "n_frames": n_frames,
                "video": str((dest / "zombihop_timelapse.mp4").relative_to(REPO)) if ok else None,
                "out_dir": str(out_dir.relative_to(REPO)),
            }
            summary.append(row)
            print(
                f"  done dist={row['dist']:.4f} iters={row['n_iters']} "
                f"frames={n_frames} video={row['video']}",
                flush=True,
            )

    (OUT / "summary.json").write_text(json.dumps({"time_limit_hours": TIME_LIMIT_HOURS,
                                                   "runs": summary}, indent=2) + "\n")
    print("\nWrote", OUT / "summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
