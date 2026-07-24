#!/bin/bash
#SBATCH --job-name=ws_validate
#SBATCH --output=warm_start/scripts/logs/ws_validate_%j.out
#SBATCH --error=warm_start/scripts/logs/ws_validate_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1

# Cheap end-to-end check of the warm-start comparison before committing a GPU day
# to warm_start_compare_3d.sbatch. Runs ONE repetition of BOTH arms with
# deliberately cheap acquisition settings and a small budget, then asserts the
# things that would silently invalidate the real experiment:
#
#   * both arms measure the SAME number of points (equal-budget fairness)
#   * the warm arm's initial design is 4 lines = 96 points, the cold arm's 2 = 48
#   * the seed prior is actually active (seed_var > real_var > 0)
#   * every per-trial artifact run_mobo writes is present
#
# Run this first:  sbatch warm_start/scripts/warm_start_validate.sbatch

mkdir -p warm_start/scripts/logs
RUN_DIR="warm_start/runs/validate_job${SLURM_JOB_ID}"

uv run python -u -m warm_start.validate --run-dir "$RUN_DIR"
rc=$?

if [ "$rc" -eq 0 ]; then
    echo "[$(date)] validation PASSED — safe to submit:"
    echo "    sbatch warm_start/scripts/warm_start_compare_3d.sbatch"
else
    echo "[$(date)] validation FAILED (rc=$rc) — do NOT submit the full run yet."
fi
exit $rc
