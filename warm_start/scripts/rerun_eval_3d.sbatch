#!/bin/bash
#SBATCH --job-name=rerun_eval_3d
#SBATCH --output=warm_start/scripts/logs/rerun_eval_3d_%j.out
#SBATCH --error=warm_start/scripts/logs/rerun_eval_3d_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

# Re-run of the d=3 deploy/eval with the CORRECTED evaluation landscape, doing ONLY
# the work that actually changed:
#
#   * The old deploy (deploy_3d_job18847323) evaluated on the warm-start GP; that was
#     wrong. Evaluation should be on the FULL-run GP (fit to every point of the real
#     campaign). So the fullgp landscape is (re)computed for every method.
#   * A brand-new BASELINE method (production BEST_HPARAMS, no warm-start influence)
#     is added, so it must run on BOTH landscapes.
#   * The warm-start methods' hyperparameters did NOT change, and swapping fullgp in
#     for warmgp does not touch the ensemble evaluation — so static/dynamic/mixture's
#     ENSEMBLE results are REUSED verbatim from the old run (symlinked), not re-run.
#
# Net new compute vs. a full redo: 3 fullgp evals (static/dynamic/mixture) + 1
# baseline on both landscapes; the 3 warm-start ensemble evals are reused.

DIM=3
NUM_RUNS=5
TIME_LIMIT_MIN=10
ENSEMBLE_SEED=0     # same seed the source run used, so ensemble is comparable

# --- resume-on-wall-time (eval restarts fresh; expected well under 8h) -----------
requeued=0
on_time_limit() {
    echo "[$(date)] wall-time approaching; requeuing job $SLURM_JOB_ID"
    scontrol requeue "$SLURM_JOB_ID" && requeued=1
    exit 0
}
trap on_time_limit USR1

set -euo pipefail

# Source deploy run: reuse its selected configs and its warm-start ensemble results.
SRC="${SRC_DEPLOY_DIR:-optimize/runs/deploy_3d_job18847323}"
SRC_CFG="${SRC}/configs"
SRC_EVAL="${SRC}/eval"
[ -d "${SRC_EVAL}" ] || { echo "source deploy eval dir not found: ${SRC_EVAL}"; exit 1; }

RUN_ROOT="optimize/runs/rerun_3d_fullgp_job${SLURM_JOB_ID}"
EVAL_DIR="${RUN_ROOT}/eval"
mkdir -p "${EVAL_DIR}"

WARM_HP="${SRC_CFG}/warm_best/hparams.json"
MIX_HP="${SRC_CFG}/mixture_best/hparams.json"
BEST_HP="${SRC_CFG}/best_hparams/trial.json"
for f in "${WARM_HP}" "${MIX_HP}" "${BEST_HP}"; do
    [ -f "$f" ] || { echo "missing reused config: $f"; exit 1; }
done
echo "[$(date)] reusing configs from ${SRC_CFG}"

# 1) Reuse the warm-start methods' ENSEMBLE results verbatim (absolute symlinks so
#    compare_methods reads them straight from the source run).
for m in static dynamic mixture; do
    mkdir -p "${EVAL_DIR}/${m}"
    src_ens="$(readlink -f "${SRC_EVAL}/${m}/ensemble")"
    [ -d "${src_ens}" ] || { echo "missing reused ensemble: ${SRC_EVAL}/${m}/ensemble"; exit 1; }
    ln -s "${src_ens}" "${EVAL_DIR}/${m}/ensemble"
done
echo "[$(date)] reused ensemble results for static / dynamic / mixture"

# 2) fullgp eval for the three warm-start methods (ONLY the changed landscape).
uv run python optimize/evaluate.py --hparams-json "${WARM_HP}" \
    --dataset fullgp --dim ${DIM} \
    --num-runs ${NUM_RUNS} --time-limit-min ${TIME_LIMIT_MIN} \
    --out-dir "${EVAL_DIR}/static"

uv run python optimize/evaluate.py --hparams-json "${WARM_HP}" \
    --dynamic-alt-hparams-json "${BEST_HP}" --dynamic-period 10 \
    --dataset fullgp --dim ${DIM} \
    --num-runs ${NUM_RUNS} --time-limit-min ${TIME_LIMIT_MIN} \
    --out-dir "${EVAL_DIR}/dynamic"

uv run python optimize/evaluate.py --hparams-json "${MIX_HP}" \
    --dataset fullgp --dim ${DIM} \
    --num-runs ${NUM_RUNS} --time-limit-min ${TIME_LIMIT_MIN} \
    --out-dir "${EVAL_DIR}/mixture"

# 3) BASELINE — brand-new method (production BEST_HPARAMS, no warm start): both
#    landscapes, since there is nothing to reuse for it.
uv run python optimize/evaluate.py --hparams-json "${BEST_HP}" \
    --dataset fullgp,ensemble --dim ${DIM} \
    --num-runs ${NUM_RUNS} --time-limit-min ${TIME_LIMIT_MIN} \
    --ensemble-seed ${ENSEMBLE_SEED} \
    --out-dir "${EVAL_DIR}/baseline"

# 4) Head-to-head comparison table (baseline + 3 warm-start methods x fullgp/ensemble).
uv run python warm_start/compare_methods.py --base "${EVAL_DIR}"

echo "[$(date)] rerun complete. Corrected results in ${RUN_ROOT}"
