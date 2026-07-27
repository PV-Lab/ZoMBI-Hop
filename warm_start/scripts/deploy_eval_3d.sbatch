#!/bin/bash
#SBATCH --job-name=deploy_eval_3d
#SBATCH --output=warm_start/scripts/logs/deploy_eval_3d_%j.out
#SBATCH --error=warm_start/scripts/logs/deploy_eval_3d_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

# Deploy + head-to-head evaluation of the three warm-start methods, d=3.
# Submitted with --dependency=afterok on warmgp_tune_3d + mixture_tune_3d, so it
# starts automatically once BOTH finish their 100-trial budget cleanly.
#
#   static   = warmgp-tuned hparams, held fixed.
#   dynamic  = warmgp-tuned hparams, alternated every 10 iters with BEST_HPARAMS.md.
#   mixture  = mixture-tuned hparams (good on warm + ensemble), held fixed.
#
# Each method is scored on the fixed warm-start GP landscape AND on re-randomized
# ensemble landscapes (--dataset warmgp,ensemble), 5 runs each, then aggregated
# into one comparison table.

DIM=3
NUM_RUNS=5
TIME_LIMIT_MIN=10
ENSEMBLE_SEED=0     # fixed so the ensemble landscapes are identical across methods

# --- resume-on-wall-time (eval restarts fresh; expected ~5h << 12h so rare) ------
requeued=0
on_time_limit() {
    echo "[$(date)] wall-time approaching; requeuing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID" && requeued=1
    exit 0
}
trap on_time_limit USR1

set -euo pipefail

RUN_ROOT="optimize/runs/deploy_${DIM}d_job${SLURM_JOB_ID}"
CFG_DIR="${RUN_ROOT}/configs"
EVAL_DIR="${RUN_ROOT}/eval"
mkdir -p "${CFG_DIR}" "${EVAL_DIR}"

# Locate the finished tuning run dirs. WARMGP_RUN_DIR / MIX_RUN_DIR may be exported
# at submit time (deterministic); otherwise fall back to the newest matching dir.
newest() { ls -dt optimize/runs/$1 2>/dev/null | head -1; }
WARM_DIR="${WARMGP_RUN_DIR:-$(newest "mobo_warmgp_${DIM}d_job*")}"
MIX_DIR="${MIX_RUN_DIR:-$(newest "mobo_mixture_${DIM}d_job*")}"
echo "[$(date)] warmgp run dir: ${WARM_DIR}"
echo "[$(date)] mixture run dir: ${MIX_DIR}"
[ -n "${WARM_DIR}" ] && [ -d "${WARM_DIR}" ] || { echo "no warmgp run dir"; exit 1; }
[ -n "${MIX_DIR}" ]  && [ -d "${MIX_DIR}" ]  || { echo "no mixture run dir"; exit 1; }

# 1) Pick the deploy config from each tuning run (lexicographic dist,dup,time).
WARM_HP=$(uv run python warm_start/select_best.py --run-dir "${WARM_DIR}" --out "${CFG_DIR}/warm_best" | tail -1)
MIX_HP=$(uv run python warm_start/select_best.py  --run-dir "${MIX_DIR}"  --out "${CFG_DIR}/mixture_best" | tail -1)
# The dynamic method's alternate set = current production hparams (BEST_HPARAMS.md).
BEST_HP=$(uv run python warm_start/best_hparams_seed.py --dim ${DIM} --out "${CFG_DIR}/best_hparams" | tail -1)/trial.json
echo "[$(date)] warm-tuned hparams: ${WARM_HP}"
echo "[$(date)] mixture hparams:    ${MIX_HP}"
echo "[$(date)] dynamic alt (BEST): ${BEST_HP}"

# 2) STATIC — warm-tuned hparams, fixed.
uv run python optimize/evaluate.py --hparams-json "${WARM_HP}" \
    --dataset warmgp,ensemble --dim ${DIM} \
    --num-runs ${NUM_RUNS} --time-limit-min ${TIME_LIMIT_MIN} \
    --warmgp-seed 0 --ensemble-seed ${ENSEMBLE_SEED} \
    --out-dir "${EVAL_DIR}/static"

# 3) DYNAMIC — warm-tuned hparams, alternated with BEST_HPARAMS every 10 iters.
uv run python optimize/evaluate.py --hparams-json "${WARM_HP}" \
    --dynamic-alt-hparams-json "${BEST_HP}" --dynamic-period 10 \
    --dataset warmgp,ensemble --dim ${DIM} \
    --num-runs ${NUM_RUNS} --time-limit-min ${TIME_LIMIT_MIN} \
    --warmgp-seed 0 --ensemble-seed ${ENSEMBLE_SEED} \
    --out-dir "${EVAL_DIR}/dynamic"

# 4) MIXTURE — mixture-tuned hparams, fixed.
uv run python optimize/evaluate.py --hparams-json "${MIX_HP}" \
    --dataset warmgp,ensemble --dim ${DIM} \
    --num-runs ${NUM_RUNS} --time-limit-min ${TIME_LIMIT_MIN} \
    --warmgp-seed 0 --ensemble-seed ${ENSEMBLE_SEED} \
    --out-dir "${EVAL_DIR}/mixture"

# 5) Head-to-head comparison table.
uv run python warm_start/compare_methods.py --base "${EVAL_DIR}"

echo "[$(date)] deploy+eval complete. Results in ${RUN_ROOT}"
