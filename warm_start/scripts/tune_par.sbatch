#!/bin/bash
#SBATCH --job-name=tune_par
#SBATCH --output=warm_start/scripts/logs/tune_par_%A_%a.out
#SBATCH --error=warm_start/scripts/logs/tune_par_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --array=0-3
#
# PARALLEL, shared-history hyperparameter tuning (fix A: cuts tuning wall time ~Nx).
# ===============================================================================
# A job ARRAY of WORKERS tasks all tune the SAME objective with --share-history, so
# every worker's GP is seeded with ALL siblings' trials as they land, while each
# worker only runs (and is billed budget for) its OWN slice of the trials. Own trials
# drive the max-trials budget; siblings' pooled trials enter as X_prior and are never
# double-counted, so WORKERS x (TOTAL_TRIALS/WORKERS) == TOTAL_TRIALS distinct trials
# collectively — reached in ~1/WORKERS the wall time of the single-job scripts.
#
# This is the parallel replacement for warmgp_tune_Nd.sbatch / mixture_tune_Nd.sbatch.
# Pick the objective at submit time via --export:
#
#   sbatch --export=ALL,DATASET=warmgp,DIM=3  warm_start/scripts/tune_par.sbatch
#   sbatch --export=ALL,DATASET=warmgp,DIM=4  warm_start/scripts/tune_par.sbatch
#   sbatch --export=ALL,DATASET=mixture,DIM=3 warm_start/scripts/tune_par.sbatch
#   sbatch --export=ALL,DATASET=mixture,DIM=4 warm_start/scripts/tune_par.sbatch
#
# Default: 4 workers (--array=0-3), 100 trials total. To change worker count keep the
# two in sync, e.g. 5 workers: --array=0-4 --export=ALL,...,WORKERS=5
#
# NB shared history is signature-matched (dataset + dim + time-limit + ensemble-repeats),
# so besides its siblings a run ALSO pools any prior matching run in the runs dir. That
# only ever *seeds the GP* (never the budget or the selected config), so it speeds
# convergence; to tune fully clean-room, run in a runs dir with no prior matches.
#
# After ALL workers of one submission finish, select the deploy config across EVERY
# worker dir (shell-glob the worker family — select_best.py takes multiple dirs):
#   uv run python warm_start/select_best.py \
#       --run-dir optimize/runs/mobo_${DATASET}_${DIM}d_job${ARRAY_JOB_ID}_w* \
#       --out <cfg_dir>/best

set -uo pipefail
DATASET="${DATASET:?set DATASET=warmgp|mixture via --export}"
DIM="${DIM:?set DIM=3|4 via --export}"
WORKERS="${WORKERS:-4}"
TOTAL_TRIALS="${TOTAL_TRIALS:-100}"
ENSEMBLE_REPEATS="${ENSEMBLE_REPEATS:-3}"
# Per-dataset per-trial wall budget, matching the single-job scripts.
if [ "${DATASET}" = "mixture" ]; then TIME_LIMIT="${TIME_LIMIT:-0.15}"; else TIME_LIMIT="${TIME_LIMIT:-0.2}"; fi

# Split TOTAL_TRIALS across workers; the last worker takes any remainder.
PER=$(( TOTAL_TRIALS / WORKERS ))
if [ "${SLURM_ARRAY_TASK_ID}" -eq "$(( WORKERS - 1 ))" ]; then
    PER=$(( TOTAL_TRIALS - PER * (WORKERS - 1) ))
fi

# --- auto-continue across the wall-time limit (same pattern as the single-job jobs) --
requeued=0
on_time_limit() {
    echo "[$(date)] wall-time approaching; requeuing task $SLURM_ARRAY_JOB_ID:$SLURM_ARRAY_TASK_ID"
    if scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; then
        requeued=1; echo "[$(date)] requeue accepted"
    else
        echo "[$(date)] requeue refused"
    fi
    exit 0
}
trap on_time_limit USR1

# Worker-private, requeue-stable run dir keyed on the ARRAY job id + task id: all
# workers of one submission share the ${ARRAY_JOB_ID} family (so a glob gathers them)
# but never collide, and a requeue keeps the same id so it resumes this exact folder.
RUN_DIR="optimize/runs/mobo_${DATASET}_${DIM}d_job${SLURM_ARRAY_JOB_ID}_w${SLURM_ARRAY_TASK_ID}"

# Ensemble seed: re-randomized per worker (distinct landscapes broaden coverage) but
# STABLE across a requeue by keying on the (requeue-invariant) array job id + task.
ENSEMBLE_SEED=$(( (SLURM_ARRAY_JOB_ID + SLURM_ARRAY_TASK_ID) % 2147483647 ))

echo "[$(date)] worker ${SLURM_ARRAY_TASK_ID}/${WORKERS}: ${DATASET} d=${DIM}, ${PER} trials, run dir ${RUN_DIR}"

# Seed the search with the current best hyperparameters (BEST_HPARAMS.md), written
# into the run dir so a requeue regenerates the identical seed.
SEED_DIR=$(uv run python warm_start/best_hparams_seed.py --dim "${DIM}" --out "${RUN_DIR}/best_seed" | tail -1)

# mixture also scores an ensemble half (repeats) alongside the warm-start GP.
EXTRA=()
if [ "${DATASET}" = "mixture" ]; then EXTRA=(--ensemble-repeats "${ENSEMBLE_REPEATS}"); fi

uv run python optimize/run_mobo.py --dataset "${DATASET}" --dim "${DIM}" \
    --max-trials "${PER}" --time-limit "${TIME_LIMIT}" \
    --warmgp-seed 0 --ensemble-seed "${ENSEMBLE_SEED}" \
    --share-history \
    --start-from-best "${SEED_DIR}" \
    "${EXTRA[@]}" \
    --run-dir "${RUN_DIR}" &
wait $!
rc=$?

# A clean exit (rc=0) means this worker finished its ${PER}-trial slice; do NOT
# resubmit. A wall-time hit is handled above by scontrol requeue (same id, resumes).
if [ "$requeued" -eq 0 ]; then
    if [ "$rc" -eq 0 ]; then
        echo "[$(date)] worker done its ${PER}-trial slice (rc=0). Not resubmitting."
    else
        echo "[$(date)] worker exited rc=$rc (fatal). Not resubmitting; fix + resubmit."
    fi
fi
