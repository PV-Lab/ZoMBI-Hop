#!/bin/bash
#SBATCH --job-name=ws_compare_3d
#SBATCH --output=warm_start/scripts/logs/ws_compare_3d_%j.out
#SBATCH --error=warm_start/scripts/logs/ws_compare_3d_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

# Warm-start vs cold-start ZoMBI-Hop, 3d, 5 paired repetitions.
#
# Unlike optimize/scripts/ensemble_mobo_3d.sbatch this is a FIXED-SIZE experiment,
# not an open-ended MOBO chain, so it deliberately does NOT resubmit itself on a
# clean exit — that would re-run the same 10 trials forever. It only requeues when
# the wall clock runs out, and compare.py skips already-finished trials on the way
# back in, so the requeued job resumes rather than restarting.

# --- auto-continue across the wall-time limit ---------------------------------
# 120s before the --time limit SLURM sends USR1 to this batch script.
# Preferred path: requeue THIS job (same ID) so it resumes when scheduled.
# Fallback: if requeue is refused (e.g. requeue/age limit reached), submit a
# brand-new job so the experiment still finishes. Cancel with `scancel`.
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

mkdir -p warm_start/scripts/logs

# Stable run dir keyed on the job id: a `scontrol requeue` keeps the SAME
# SLURM_JOB_ID, so the relaunched job re-enters this exact folder and picks up the
# remaining trials instead of re-spending GPU time on the finished ones.
RUN_DIR="warm_start/runs/compare_3d_job${SLURM_JOB_ID}"

# --base-seed fixes the family of 5 landscapes. It is deliberately NOT keyed on the
# job id: the point of this run is a specific paired experiment, and keying the
# landscapes on the job id would silently change what a requeue is comparing.
# Pass a different --base-seed by hand for an independent replication.
uv run python -m warm_start.compare \
    --run-dir "$RUN_DIR" \
    --dim 3 \
    --reps 5 \
    --base-seed 0 &
wait $!
rc=$?

if [ "$requeued" -eq 0 ]; then
    if [ "$rc" -eq 0 ]; then
        echo "[$(date)] experiment complete (rc=0); summarising"
        uv run python -m warm_start.analyze --run-dir "$RUN_DIR" || true
        echo "[$(date)] results in $RUN_DIR"
    else
        echo "[$(date)] exited with rc=$rc — NOT resubmitting."
        echo "[$(date)] inspect the log, then resume with: sbatch $0"
        echo "[$(date)] (a resumed job re-uses finished trials in $RUN_DIR)"
    fi
fi
