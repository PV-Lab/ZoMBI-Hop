#!/bin/bash
# =============================================================================
# Joint MOBO hparam training on BOTH ELA RF(g) interior-20 twins:
#   ela_3d_18535497  +  ela_3d_18503666
#
# Each trial evaluates the same hparams on both landscapes (run_1 / run_2) and
# averages dist / dup / avg_time_per_iter — one shared Pareto front, one hparam set.
#
# Fresh run (new folder):
#   sbatch slurm/run_mobo_ela_pair.sbatch
#
# Continue one folder:
#   MOBO_RUN_DIR=optimize/runs/mobo_ela_rf_g_interior20_pair_jobXXXX \
#     sbatch --export=ALL,MOBO_RUN_DIR slurm/run_mobo_ela_pair.sbatch
#
# Concurrent jobs (share history live — use SEPARATE run dirs, not the same
# MOBO_RUN_DIR): submit twice without pinning, or pin each to its own folder.
# --share-history pools matching ela_multi trials across sibling pair runs.
#
# Cancel the auto-continue chain with: scancel <job_id>
# =============================================================================

#SBATCH --job-name=mobo_ela_pair
#SBATCH --output=optimize/scripts/logs/mobo_ela_pair_%j.out
#SBATCH --error=optimize/scripts/logs/mobo_ela_pair_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO}"
mkdir -p logs optimize/runs optimize/scripts/logs

# shellcheck source=/dev/null
source "${REPO}/slurm/mobo_job_common.sh"
_mobo_setup_env

CONFIG="${REPO}/optimize/mobo_batch_configs/ela_rf_g_interior20_pair_18535497_18503666.json"
for JOB in 18535497 18503666; do
  EXPR="${REPO}/ela/runs/ela_3d_${JOB}/best/expression.json"
  SAMPLES="${REPO}/ela/runs/ela_3d_${JOB}/samples.npz"
  if [[ ! -f "${EXPR}" ]]; then
    echo "ERROR: missing ELA expression at ${EXPR}" >&2
    exit 1
  fi
  if [[ ! -f "${SAMPLES}" ]]; then
    echo "ERROR: missing samples.npz at ${SAMPLES} (needed for RF(g))" >&2
    exit 1
  fi
done
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: missing batch config at ${CONFIG}" >&2
  exit 1
fi

JOB_TAG="${SLURM_JOB_ID:-local}"
RUN_DIR="${MOBO_RUN_DIR:-optimize/runs/mobo_ela_rf_g_interior20_pair_job${JOB_TAG}}"
export MOBO_RUN_DIR="${RUN_DIR}"

_chain_next() {
    sbatch --export=ALL,MOBO_RUN_DIR="${MOBO_RUN_DIR}" "$0"
}

requeued=0
on_time_limit() {
    echo "[$(date)] wall-time approaching; requeuing job ${SLURM_JOB_ID}"
    if scontrol requeue "${SLURM_JOB_ID}"; then
        requeued=1
        echo "[$(date)] requeue accepted"
    else
        echo "[$(date)] requeue refused; submitting a fresh job"
        _chain_next
    fi
    exit 0
}
trap on_time_limit USR1

MOBO_CMD=(
  python optimize/run_mobo.py
  --no-show
  --device cuda
  --batch
  --config optimize/mobo_batch_configs/ela_rf_g_interior20_pair_18535497_18503666.json
  --time-limit 0.2
  --share-history
  --share-dirs "${REPO}/optimize/runs"
  --start-from-best
  optimize/runs/archived_runs/mobo_3d_05_06_15_32/trial_112
  optimize/runs/archived_runs/mobo_ensemble_3d_job16948629/trial_33
  optimize/runs/archived_runs/mobo_ensemble_3d_job16861902/trial_49
  optimize/runs/archived_runs/mobo_ensemble_3d_job16954340/trial_63
  optimize/runs/mobo_ela_3d_18535497_rf_g_interior20_job18814588/trial_303
  optimize/runs/mobo_ela_3d_18503666_rf_g_interior20_job18814589/trial_123
  --run-dir "${RUN_DIR}"
)

echo "=== Joint MOBO on ELA 18535497 + 18503666 RF(g) interior20 ==="
printf '  %q' "${MOBO_CMD[@]}"
echo
echo "=== config: ${CONFIG} ==="
echo "=== run-dir: ${RUN_DIR} ==="

set +e
"${MOBO_CMD[@]}" &
wait $!
rc=$?
set -e

if [[ "${requeued}" -eq 0 ]]; then
    if [[ "${rc}" -eq 0 ]]; then
        echo "[$(date)] run exited cleanly; submitting next job"
        _chain_next
    else
        echo "[$(date)] run exited with rc=${rc}; not resubmitting." >&2
        exit "${rc}"
    fi
fi
