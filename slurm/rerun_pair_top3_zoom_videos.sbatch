#!/bin/bash
# =============================================================================
# Re-run the 3 best pair-MOBO hparams on both ELA twins with full zoom-bound
# frames, then assemble zombihop_timelapse.mp4.
#
# Requires (synced to ORCD):
#   - optimize/rerun_pair_top3_zoom_videos.py
#   - optimize/run_mobo.py with ZOMBI_SAVE_ALL_FRAMES support
#   - optimize/runs/transfer_pair_top3_edge01/hparams_stage/
#   - warm_start/optima_finder_ela_rf_edge_sweep/edge_0.01/
#   - ela/runs/ela_3d_18535497 and ela/runs/ela_3d_18503666 (best oracle + RF train)
#
# Submit from repo root:
#   sbatch slurm/rerun_pair_top3_zoom_videos.sbatch
# =============================================================================

#SBATCH --job-name=pair_zoom_vid
#SBATCH --output=optimize/scripts/logs/pair_top3_zoom_videos_%j.out
#SBATCH --error=optimize/scripts/logs/pair_top3_zoom_videos_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO}"
mkdir -p optimize/runs optimize/scripts/logs \
  optimize/runs/pair_lowest_dist_compare_edge01/zoom_videos

# shellcheck source=/dev/null
source "${REPO}/slurm/mobo_job_common.sh"
_mobo_setup_env

export ZOMBI_SAVE_ALL_FRAMES=1
export MPLBACKEND=Agg

echo "=== pair top-3 zoom-bound timelapses ==="
echo "  host: $(hostname)"
echo "  start: $(date -Is)"
echo "  ZOMBI_SAVE_ALL_FRAMES=${ZOMBI_SAVE_ALL_FRAMES}"

python -u optimize/rerun_pair_top3_zoom_videos.py \
  2>&1 | tee optimize/runs/pair_lowest_dist_compare_edge01/zoom_videos/driver.log

echo "finished: $(date -Is)"
echo "summary:  optimize/runs/pair_lowest_dist_compare_edge01/zoom_videos/summary.json"
