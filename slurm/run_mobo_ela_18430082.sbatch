#!/bin/bash
# MOBO hparam training on ELA 18430082 using 10 RBF-matched catchments.
#
# Submit from the ORCD repository root after the ELA pilot finishes:
#   sbatch slurm/run_mobo_ela_18430082.sbatch
#
# Continuous GPU setup; no --share-history. Cancel with: scancel <job_id>

#SBATCH --job-name=mobo_ela_18430082
#SBATCH --output=optimize/scripts/logs/mobo_ela_18430082_%j.out
#SBATCH --error=optimize/scripts/logs/mobo_ela_18430082_%j.err
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

ORACLE="${REPO}/ela/runs/ela_3d_18430082/best/oracle.py"
CONFIG="${REPO}/optimize/mobo_batch_configs/ela_3d_18430082_rbf10.json"
if [[ ! -f "${ORACLE}" ]]; then
  echo "ERROR: run 18430082 has no final oracle at ${ORACLE}" >&2
  echo "Wait for the ELA pilot to finish, sync the completed run, then resubmit." >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: missing batch config at ${CONFIG}" >&2
  exit 1
fi

requeued=0
on_time_limit() {
    echo "[$(date)] wall-time approaching; requeuing job ${SLURM_JOB_ID}"
    if scontrol requeue "${SLURM_JOB_ID}"; then
        requeued=1
        echo "[$(date)] requeue accepted"
    else
        echo "[$(date)] requeue refused; submitting a fresh job"
        sbatch "$0"
    fi
    exit 0
}
trap on_time_limit USR1

JOB_TAG="${SLURM_JOB_ID:-local}"
RUN_DIR="optimize/runs/mobo_ela_3d_18430082_rbf10_job${JOB_TAG}"

MOBO_CMD=(
  python optimize/run_mobo.py
  --no-show
  --device cuda
  --batch
  --config optimize/mobo_batch_configs/ela_3d_18430082_rbf10.json
  --time-limit 0.2
  --start-from-best
  optimize/runs/archived_runs/mobo_3d_05_06_15_32/trial_112
  optimize/runs/archived_runs/mobo_ensemble_3d_job16948629/trial_33
  optimize/runs/archived_runs/mobo_ensemble_3d_job16861902/trial_49
  optimize/runs/archived_runs/mobo_ensemble_3d_job16954340/trial_63
  --run-dir "${RUN_DIR}"
)

echo "=== MOBO on ELA 18430082 (10 RBF-matched catchments) ==="
printf '  %q' "${MOBO_CMD[@]}"
echo
echo "=== oracle: ${ORACLE} ==="
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
        sbatch "$0"
    else
        echo "[$(date)] run exited with rc=${rc}; not resubmitting." >&2
        exit "${rc}"
    fi
fi
