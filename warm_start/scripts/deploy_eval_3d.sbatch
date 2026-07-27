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

# Deploy + head-to-head evaluation of the warm-start methods against a baseline, d=3.
# Submitted with --dependency=afterok on warmgp_tune_3d + mixture_tune_3d, so it
# starts automatically once BOTH finish their 100-trial budget cleanly.
#
#   baseline = production BEST_HPARAMS.md, held fixed — NO warm-start influence.
#   static   = warmgp-tuned hparams, held fixed.
#   dynamic  = warmgp-tuned hparams, alternated every 10 iters with BEST_HPARAMS.md.
#   mixture  = mixture-tuned hparams (good on warm + ensemble), held fixed.
#
# Each method is scored on the FULL-run GP landscape (the GP fit to every point of
# the real campaign — the honest ground truth) AND on re-randomized ensemble
# landscapes (--dataset fullgp,ensemble), 5 runs each, then aggregated into one
# comparison table. NB the warm-start GP is only the TUNING surrogate; it is never
# an evaluation target.

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

# 1) Pick the deploy config from each tuning run. If WARMGP_TRIAL / MIX_TRIAL are
#    exported, pin those exact hand-picked trials (chosen off the Pareto plots);
#    otherwise select_best uses its weighted score (0.8*dist + 0.2*dup, normalised).
WARM_TRIAL_ARG=""; [ -n "${WARMGP_TRIAL:-}" ] && WARM_TRIAL_ARG="--trial ${WARMGP_TRIAL}"
MIX_TRIAL_ARG="";  [ -n "${MIX_TRIAL:-}" ]    && MIX_TRIAL_ARG="--trial ${MIX_TRIAL}"
WARM_HP=$(uv run python warm_start/select_best.py --run-dir "${WARM_DIR}" ${WARM_TRIAL_ARG} --out "${CFG_DIR}/warm_best" | tail -1)
MIX_HP=$(uv run python warm_start/select_best.py  --run-dir "${MIX_DIR}"  ${MIX_TRIAL_ARG}  --out "${CFG_DIR}/mixture_best" | tail -1)
# Production hparams (BEST_HPARAMS.md): the baseline config AND the dynamic method's
# alternate set.
BEST_HP=$(uv run python warm_start/best_hparams_seed.py --dim ${DIM} --out "${CFG_DIR}/best_hparams" | tail -1)/trial.json
echo "[$(date)] warm-tuned hparams:   ${WARM_HP}"
echo "[$(date)] mixture hparams:      ${MIX_HP}"
echo "[$(date)] baseline/dynamic-alt: ${BEST_HP}"

# 2) BASELINE — production BEST_HPARAMS, fixed, no warm-start influence at all.
uv run python optimize/evaluate.py --hparams-json "${BEST_HP}" \
    --dataset fullgp,ensemble --dim ${DIM} \
    --num-runs ${NUM_RUNS} --time-limit-min ${TIME_LIMIT_MIN} \
    --ensemble-seed ${ENSEMBLE_SEED} \
    --out-dir "${EVAL_DIR}/baseline"

# 3) STATIC — warm-tuned hparams, fixed.
uv run python optimize/evaluate.py --hparams-json "${WARM_HP}" \
    --dataset fullgp,ensemble --dim ${DIM} \
    --num-runs ${NUM_RUNS} --time-limit-min ${TIME_LIMIT_MIN} \
    --ensemble-seed ${ENSEMBLE_SEED} \
    --out-dir "${EVAL_DIR}/static"

# 4) DYNAMIC — warm-tuned hparams, alternated with BEST_HPARAMS every 10 iters.
uv run python optimize/evaluate.py --hparams-json "${WARM_HP}" \
    --dynamic-alt-hparams-json "${BEST_HP}" --dynamic-period 10 \
    --dataset fullgp,ensemble --dim ${DIM} \
    --num-runs ${NUM_RUNS} --time-limit-min ${TIME_LIMIT_MIN} \
    --ensemble-seed ${ENSEMBLE_SEED} \
    --out-dir "${EVAL_DIR}/dynamic"

# 5) MIXTURE — mixture-tuned hparams, fixed.
uv run python optimize/evaluate.py --hparams-json "${MIX_HP}" \
    --dataset fullgp,ensemble --dim ${DIM} \
    --num-runs ${NUM_RUNS} --time-limit-min ${TIME_LIMIT_MIN} \
    --ensemble-seed ${ENSEMBLE_SEED} \
    --out-dir "${EVAL_DIR}/mixture"

# 6) Head-to-head comparison table.
uv run python warm_start/compare_methods.py --base "${EVAL_DIR}"

echo "[$(date)] deploy+eval complete. Results in ${RUN_ROOT}"
