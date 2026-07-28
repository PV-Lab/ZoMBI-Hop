#!/bin/bash
# =============================================================================
# MOBO hparam training on ELA twin ela_3d_18535497 — RF(g) interior optima.
#
# 20 needles from warm_start/optima_finder_ela_rf_interior (edge_min=0.03).
# Objective is RF(g), not raw best/oracle.py g(z).
#
# Submit from repo root on ORCD (resumes existing run by default):
#   sbatch slurm/run_mobo_ela_18535497.sbatch
#
# Override run dir if needed:
#   MOBO_RUN_DIR=optimize/runs/mobo_ela_3d_18535497_rf_g_interior20_jobXXXX \
#     sbatch slurm/run_mobo_ela_18535497.sbatch
#
# Cancel the auto-continue chain with: scancel <job_id>
# =============================================================================

#SBATCH --job-name=mobo_ela_18535497
#SBATCH --output=optimize/scripts/logs/mobo_ela_18535497_%j.out
#SBATCH --error=optimize/scripts/logs/mobo_ela_18535497_%j.err
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

EXPR="${REPO}/ela/runs/ela_3d_18535497/best/expression.json"
SAMPLES="${REPO}/ela/runs/ela_3d_18535497/samples.npz"
CONFIG="${REPO}/optimize/mobo_batch_configs/ela_3d_18535497_rf_g_interior20.json"
if [[ ! -f "${EXPR}" ]]; then
  echo "ERROR: missing ELA expression at ${EXPR}" >&2
  exit 1
fi
if [[ ! -f "${SAMPLES}" ]]; then
  echo "ERROR: missing samples.npz at ${SAMPLES} (needed for RF(g))" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: missing batch config at ${CONFIG}" >&2
  exit 1
fi

# Pin to the existing MOBO run so requeue / clean resubmit continue trials
# (304+ so far) instead of opening a fresh jobXXXX folder.
RUN_DIR="${MOBO_RUN_DIR:-optimize/runs/mobo_ela_3d_18535497_rf_g_interior20_job18814588}"
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
  --config optimize/mobo_batch_configs/ela_3d_18535497_rf_g_interior20.json
  --time-limit 0.2
  --start-from-best
  optimize/runs/archived_runs/mobo_3d_05_06_15_32/trial_112
  optimize/runs/archived_runs/mobo_ensemble_3d_job16948629/trial_33
  optimize/runs/archived_runs/mobo_ensemble_3d_job16861902/trial_49
  optimize/runs/archived_runs/mobo_ensemble_3d_job16954340/trial_63
  --run-dir "${RUN_DIR}"
)

echo "=== MOBO on ELA 18535497 RF(g) interior20 ==="
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
