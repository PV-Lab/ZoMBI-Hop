#!/bin/bash
#SBATCH --job-name=mixture_tune_3d
#SBATCH --output=warm_start/scripts/logs/mixture_tune_3d_%j.out
#SBATCH --error=warm_start/scripts/logs/mixture_tune_3d_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

# Mixture hyperparameter tuning, d=3. Each of the 100 MOBO trials is scored on
# BOTH a re-randomized ensemble landscape (the standard N-repeat average) AND the
# fixed warm-start GP landscape, blended 50/50 per metric — so the tuned
# hyperparameters are good on the warm start and on generic landscapes at once.
# Cost per trial = (ensemble-repeats + 1 warm) ZoMBI runs, so ensemble-repeats is
# kept at 3 (not the default 5) to bound wall time.

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
# folder, reusing already-finished ensemble repeats + the warm eval of an
# interrupted trial. NOT --share-history: isolated experiment.
RUN_DIR="optimize/runs/mobo_mixture_3d_job${SLURM_JOB_ID}"

# Re-randomize the ensemble half per job but keep it STABLE across a requeue by
# keying the seed on the (requeue-invariant) job id, exactly like the ensemble
# jobs. Modulo keeps it in NumPy's int32 RandomState range.
ENSEMBLE_SEED=$(( SLURM_JOB_ID % 2147483647 ))

# Seed the search with the current best hyperparameters (BEST_HPARAMS.md).
SEED_DIR=$(uv run python warm_start/best_hparams_seed.py --dim 3 --out "${RUN_DIR}/best_seed" | tail -1)
echo "[$(date)] start-from-best seed dir: ${SEED_DIR}   ensemble_seed: ${ENSEMBLE_SEED}"

uv run python optimize/run_mobo.py --dataset mixture --dim 3 \
    --max-trials 100 --time-limit 0.15 --ensemble-repeats 3 \
    --ensemble-seed "${ENSEMBLE_SEED}" --warmgp-seed 0 \
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
