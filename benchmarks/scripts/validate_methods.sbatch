#!/bin/bash
#SBATCH --job-name=validate_methods
#SBATCH --output=benchmarks/scripts/logs/validate_methods_%A_%a.out
#SBATCH --error=benchmarks/scripts/logs/validate_methods_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

# Validation of benchmarks/methods + the refactored benchmarks/sweeps, on compute
# nodes. Submit FROM THE REPO ROOT (the log paths above are relative to it):
#
#     sbatch benchmarks/scripts/validate_methods.sbatch
#
# A BOUNDED job — five array elements, no self-resubmission:
#   * element 0 first runs the needle-extractor calibration (known answers,
#     including needle-free landscapes) and an end-to-end smoke test of every
#     method on the GPU, then joins the worker pool;
#   * all five elements drain the timing campaign planned in
#     benchmarks/sweeps/runs/validate_20260926 (dims 3 and 10, n=10, b=6, one draw,
#     every method, the real 3000-point budget, 2 h ceiling per cell). Its purpose
#     is per-method cost per cell, to size the full sweep; a cell cut off by the
#     ceiling still records how many points it reached.
#
# Afterwards (login node is fine — it only reads CSVs and draws PNGs):
#     python -m benchmarks.sweeps status    --out benchmarks/sweeps/runs/validate_20260926
#     python -m benchmarks.sweeps summarize --out benchmarks/sweeps/runs/validate_20260926

CAMPAIGN="benchmarks/sweeps/runs/validate_20260926"
CHECKS="benchmarks/scripts/logs/validate_methods_${SLURM_ARRAY_JOB_ID}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo "[$(date)] element $SLURM_ARRAY_TASK_ID on $(hostname)"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    echo "[$(date)] extractor calibration"
    uv run python -m benchmarks.methods calibrate --device cuda --out "$CHECKS/calibration"
    echo "[$(date)] calibration rc=$?"
    echo "[$(date)] method smoke test (all methods, GPU)"
    uv run python -m benchmarks.methods smoke --device cuda --out "$CHECKS/smoke"
    echo "[$(date)] smoke rc=$?"
fi

uv run python -m benchmarks.sweeps run \
    --out "$CAMPAIGN" \
    --worker "$SLURM_ARRAY_TASK_ID" \
    --n-workers 5 \
    --device cuda < /dev/null
rc=$?
echo "[$(date)] worker $SLURM_ARRAY_TASK_ID exited rc=$rc"
exit $rc
