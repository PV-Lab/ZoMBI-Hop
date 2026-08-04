#!/bin/bash
#SBATCH --job-name=warmgp_tune_4d
#SBATCH --output=warm_start/scripts/logs/warmgp_tune_4d_%j.out
#SBATCH --error=warm_start/scripts/logs/warmgp_tune_4d_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

# Warm-start GP hyperparameter tuning, d=4 (STATIC + DYNAMIC share this job).
# Tunes 100 MOBO trials against the FIXED warm-start GP landscape
# (warm_start/warm_gp_landscape.py). The best trial's hyperparameters are what
# BOTH warm-start deployments use — they differ only at runtime (static always
# uses them; dynamic alternates them with BEST_HPARAMS.md every 10 iterations),
# not in what is tuned here.

# --- auto-continue across the wall-time limit (same pattern as ensemble jobs) ---
requeued=0
on_time_limit() {
    echo "[$(date)] wall-time approaching; requeuing job $SLURM_JOB_ID"
    if scontrol requeue "$SLURM_JOB_ID"; then
        requeued=1; echo "[$(date)] requeue accepted"
    else
        echo "[$(date)] requeue refused; submitting a fresh job"; sbatch "$0"
    fi
    exit 0
}
trap on_time_limit USR1

# Stable, job-id-keyed run dir so a `scontrol requeue` (same id) resumes this exact
# folder — reloads completed trials + the persisted init design and continues at
# the next trial index — instead of restarting. Keep the mobo_ prefix so mobo_*
# tooling still sees it. NOT --share-history: this is an isolated experiment and
# must not pool with (or pollute) the ensemble tuning pools.
RUN_DIR="optimize/runs/mobo_warmgp_4d_job${SLURM_JOB_ID}"

# Seed the search with the current best hyperparameters (BEST_HPARAMS.md), written
# into the run dir so a requeue regenerates the identical seed. Capture ONLY the
# last stdout line (the path); earlier lines are progress notes.
SEED_DIR=$(uv run python warm_start/best_hparams_seed.py --dim 4 --out "${RUN_DIR}/best_seed" | tail -1)
echo "[$(date)] start-from-best seed dir: ${SEED_DIR}"

uv run python optimize/run_mobo.py --dataset warmgp --dim 4 \
    --max-trials 100 --time-limit 0.2 --warmgp-seed 0 \
    --start-from-best "${SEED_DIR}" \
    --run-dir "${RUN_DIR}" &
wait $!
rc=$?

# One-shot experiment: a clean exit (rc=0) means the 100-trial budget is done, so
# do NOT resubmit (that would spawn endless isolated replicate runs and there would
# be no well-defined "finish" for the deploy/eval dependency to hang on). A
# wall-time hit is handled above by `scontrol requeue` (same job id, resumes this
# run dir), so reaching here with requeued=0 is a genuine end-of-run.
if [ "$requeued" -eq 0 ]; then
    if [ "$rc" -eq 0 ]; then
        echo "[$(date)] run finished its 100-trial budget (rc=0). Done — not resubmitting."
    else
        echo "[$(date)] run exited with rc=$rc (fatal abort/crash). Not resubmitting."
        echo "[$(date)] fix the issue and resubmit manually with: sbatch $0"
    fi
fi
