#!/bin/bash
#SBATCH --job-name=render_accordion
#SBATCH --output=visualization/scripts/logs/render_accordion_%j.out
#SBATCH --error=visualization/scripts/logs/render_accordion_%j.err
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=mit_quicktest

# Headless render of a visualization/accordion.py panel, off the login node.
#
# Why a batch job: at --stride 1 the slice panel draws ~80 triangles and ~2k
# depth-sorted 3D scatter points onto a canvas capped at 34x9 inches, then reopens
# the multi-megapixel PNG in Pillow to trim it. That is a minute-plus of
# single-threaded matplotlib and a few GB of peak memory — fine for a compute node,
# rude on a shared login node.
#
# Why mit_quicktest: it caps jobs at 15 minutes and keeps a few hundred cores idle,
# so short jobs backfill immediately (the accordion run array started within
# seconds there while the GPU partition had a multi-hour pending queue). A render
# never needs a GPU — this is pure matplotlib/numpy/Pillow CPU work.
#
# Usage — every argument is passed straight through to accordion.py, so any flag
# the script accepts works here:
#
#   # the defaults below (run 3, every iteration, tightly packed)
#   sbatch visualization/scripts/render_accordion.sbatch
#
#   # a handful of slices chosen to land ON the needle iterations, which an even
#   # --stride almost always misses (the run prints its needle iterations)
#   sbatch visualization/scripts/render_accordion.sbatch \
#       --slices optimize/runs/figure_one_job19312951/trial_112/run_3 \
#       --at-iters 0,13,28,44,66,90 --out visualization/accordion_slices_6.png
#
#   # the landscape panel instead
#   sbatch visualization/scripts/render_accordion.sbatch --base
#
# Output paths are relative to the repo root (sbatch runs from your submit dir, so
# submit from the repo root). Progress and the saved path land in the .out log.

set -u

# Default: job 19312951 run_3 — best optima recovery of the five candidates (50% of
# its needles on a true optimum), every iteration, tightly packed.
DEFAULT_ARGS=(
    --slices optimize/runs/figure_one_job19312951/trial_112/run_3
    --stride 1
    --spacing 0.03
    --out visualization/accordion_slices.png
)

if [ "$#" -gt 0 ]; then
    ARGS=("$@")
else
    ARGS=("${DEFAULT_ARGS[@]}")
fi

echo "[$(date)] rendering: accordion.py ${ARGS[*]}"
# Keep matplotlib headless and stop numpy/BLAS from oversubscribing the 4 cores.
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

uv run python visualization/accordion.py "${ARGS[@]}"
rc=$?

echo "[$(date)] finished with rc=${rc}"
exit $rc
