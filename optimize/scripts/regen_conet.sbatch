#!/bin/bash
#SBATCH --job-name=regen_conet
#SBATCH --output=optimize/scripts/logs/regen_conet_%A_%a.out
#SBATCH --error=optimize/scripts/logs/regen_conet_%A_%a.err
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --partition=sched_mit_sloan_batch_r8
# =============================================================================
# Re-render the CoNet pair for a manifest of run directories, one array task per
# run, using visualization/paired_conet.py's per-landscape frame cache.
#
# Run it in TWO stages so tasks never race to fit the same landscape frame:
#
#   # stage 1 — fit + cache the 10 distinct landscape frames (one run each)
#   sbatch --export=MANIFEST=optimize/runs/regen_conet_frames.txt \
#          --array=0-9%5 optimize/scripts/regen_conet.sbatch
#
#   # stage 2 — re-render all 40 runs, every one reusing a cached frame
#   sbatch --dependency=afterok:<STAGE1> \
#          --export=MANIFEST=optimize/runs/regen_conet_all.txt \
#          --array=0-39%5 optimize/scripts/regen_conet.sbatch
#
# Stage 1 is not strictly required — the cache write is atomic and a race merely
# wastes a refit — but skipping it means up to 4 jobs redundantly fitting the same
# 15k-sample UMAP, which is the expensive part.
#
# No GPU: this is UMAP + matplotlib, all CPU.
# =============================================================================
set -u

REPO=/orcd/scratch/orcd/013/adewinmb/ZoMBI-Hop
cd "$REPO" || exit 1

: "${MANIFEST:?set MANIFEST=<file of run dirs> via --export}"

RUN_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
if [ -z "$RUN_DIR" ]; then
    echo "[$(date)] no manifest entry for task $SLURM_ARRAY_TASK_ID in $MANIFEST"
    exit 1
fi

echo "[$(date)] task $SLURM_ARRAY_TASK_ID -> $RUN_DIR"

MPLBACKEND=Agg uv run python visualization/paired_conet.py \
    --run "$RUN_DIR" \
    --save "$RUN_DIR/conet.png" \
    --save-uniform "$RUN_DIR/conet_uniform.png"
rc=$?

echo "[$(date)] task $SLURM_ARRAY_TASK_ID finished rc=$rc"
exit $rc
