#!/bin/bash
# =============================================================================
# MOBO hparam training on campaign1a RF (same landscape/needles as mobo_05_06_15_32).
#
# Objectives (all minimized by MOBO):
#   1. dist_to_needles
#   2. dup_fraction
#   3. avg_time_per_iter_s   (wall-clock seconds per ZoMBI iteration)
#
# Config: optimize/mobo_batch_configs/campaign1a_mobo_05_06_15_32_repeat.json
#   · RF on data/campaign1a.csv
#   · 20 reference optima from mobo_05_06_15_32
#   · per-trial time_limit_hours=0.4 (override with --time-limit below)
#
# Submit from repo root on ORCD:
#   sbatch slurm/run_mobo_campaign1a.sbatch
#
# Resume a specific run dir:
#   MOBO_RUN_DIR=optimize/runs/mobo_campaign1a_jobXXXX \
#     sbatch --export=ALL,MOBO_RUN_DIR slurm/run_mobo_campaign1a.sbatch
#
# Cancel the auto-continue chain with: scancel <job_id>
# =============================================================================

#SBATCH --job-name=mobo_campaign1a
#SBATCH --output=optimize/scripts/logs/mobo_campaign1a_%j.out
#SBATCH --error=optimize/scripts/logs/mobo_campaign1a_%j.err
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

CSV="${REPO}/data/campaign1a.csv"
CONFIG="${REPO}/optimize/mobo_batch_configs/campaign1a_mobo_05_06_15_32_repeat.json"
if [[ ! -f "${CSV}" ]]; then
  echo "ERROR: missing campaign CSV at ${CSV}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: missing batch config at ${CONFIG}" >&2
  exit 1
fi

# Fresh submit → new jobXXXX folder. Requeue keeps SLURM_JOB_ID → same folder.
# Override with MOBO_RUN_DIR to continue an existing run across a new job id.
JOB_TAG="${SLURM_JOB_ID:-local}"
RUN_DIR="${MOBO_RUN_DIR:-optimize/runs/mobo_campaign1a_job${JOB_TAG}}"
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
  --config optimize/mobo_batch_configs/campaign1a_mobo_05_06_15_32_repeat.json
  --time-limit 0.4
  --start-from-best
  optimize/runs/archived_runs/mobo_3d_05_06_15_32/trial_112
  optimize/runs/archived_runs/mobo_ensemble_3d_job16948629/trial_33
  optimize/runs/archived_runs/mobo_ensemble_3d_job16861902/trial_49
  optimize/runs/archived_runs/mobo_ensemble_3d_job16954340/trial_63
  --run-dir "${RUN_DIR}"
)

echo "=== MOBO on campaign1a RF (mobo_05_06_15_32 needles) ==="
echo "  objectives: dist_to_needles, dup_fraction, avg_time_per_iter_s"
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
