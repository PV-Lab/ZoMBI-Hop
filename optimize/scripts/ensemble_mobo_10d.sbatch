#!/bin/bash
#SBATCH --job-name=ensemble_mobo_10d
#SBATCH --output=optimize/scripts/logs/ensemble_mobo_10d_%j.out
#SBATCH --error=optimize/scripts/logs/ensemble_mobo_10d_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

# --- auto-continue across the wall-time limit ---------------------------------
# 120s before the --time limit SLURM sends USR1 to this batch script.
# Preferred path: requeue THIS job (same ID) so it resumes when scheduled.
# Fallback: if requeue is refused (e.g. requeue/age limit reached), submit a
# brand-new job so the chain never dies. Cancel with `scancel` to stop it.
requeued=0
on_time_limit() {
    echo "[$(date)] wall-time approaching; requeuing job $SLURM_JOB_ID"
    if scontrol requeue "$SLURM_JOB_ID"; then
        requeued=1
        echo "[$(date)] requeue accepted"
    else
        echo "[$(date)] requeue refused; submitting a fresh job"
        sbatch "$0"
    fi
    exit 0
}
trap on_time_limit USR1

uv run python optimize/run_mobo.py --dataset ensemble --dim 10 --time-limit 0.7 --share-history --start-from-best optimize/runs/mobo_3d_05_06_15_32/trial_112 &
wait $!

# Reached only if the run exits on its own (not killed by the wall-time signal).
if [ "$requeued" -eq 0 ]; then
    echo "[$(date)] run exited on its own; submitting next job"
    sbatch "$0"
fi
