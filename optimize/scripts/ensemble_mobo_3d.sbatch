#!/bin/bash
#SBATCH --job-name=ensemble_mobo_3d
#SBATCH --output=optimize/scripts/logs/ensemble_mobo_3d_%j.out
#SBATCH --error=optimize/scripts/logs/ensemble_mobo_3d_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
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

# Stable run dir keyed on the job id: a `scontrol requeue` keeps the SAME
# SLURM_JOB_ID, so the relaunched job re-enters this exact folder and resumes the
# interrupted ensemble trial (reusing already-finished run_<k> repeats) instead of
# starting a fresh run and re-spending that GPU time. A clean-exit chain or the
# fallback resubmit gets a new job id (new dir) — a genuinely fresh run, which is
# correct. Keep the `mobo_` prefix so shared-history's `mobo_*` glob still sees it.
RUN_DIR="optimize/runs/mobo_ensemble_3d_job${SLURM_JOB_ID}"

uv run python optimize/run_mobo.py --dataset ensemble --dim 3 --time-limit 0.2 --share-history --start-from-best optimize/runs/mobo_3d_05_06_15_32/trial_112 --run-dir "$RUN_DIR" &
wait $!
rc=$?

# Reached only if the run exits on its own (not killed by the wall-time signal).
# Only resubmit on a clean exit (rc=0). A non-zero rc means run_mobo.py hit a
# fatal abort (MAX_CONSEC_FAIL consecutive crashes) or the process died — in that
# case resubmitting just spawns another dead run_dir every couple of minutes, so
# stop the chain and let the user fix the root cause and resubmit manually.
if [ "$requeued" -eq 0 ]; then
    if [ "$rc" -eq 0 ]; then
        echo "[$(date)] run exited cleanly (rc=0); submitting next job"
        sbatch "$0"
    else
        echo "[$(date)] run exited with rc=$rc (fatal abort/crash) — NOT resubmitting."
        echo "[$(date)] fix the issue and resubmit manually with: sbatch $0"
    fi
fi
