#!/bin/bash
#SBATCH --job-name=plot_10d_conet
#SBATCH --output=visualization/scripts/logs/plot_10d_conet_%j.out
#SBATCH --error=visualization/scripts/logs/plot_10d_conet_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=sched_mit_sloan_batch_r8

RUN_DIR="${1:-optimize/runs/mobo_ensemble_10d_job17775999/trial_13/run_2}"
SAVE_PNG="${2:-${RUN_DIR%/}/conet.png}"

echo "[$(date)] rendering CoNet"
echo "  run : $RUN_DIR"
echo "  save: $SAVE_PNG"

# Keep BLAS from oversubscribing beyond our cpus-per-task allocation.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
# Headless: no display on a compute node.
export MPLBACKEND=Agg

uv run python visualization/plot_10d.py --needles --save "$SAVE_PNG" --run "$RUN_DIR"
rc=$?

echo "[$(date)] plot_10d.py exited with rc=$rc"
exit $rc
