#!/bin/bash
# =============================================================================
# MOBO hparam training on ELA twin ela_3d_18299175 (hand-picked needles).
#
# 1) Locally pick maxima and write the batch JSON:
#      python ela/pick_optima.py --job-id 18299175
#    → optimize/mobo_batch_configs/ela_3d_18299175.json
#
# 2) Sync that JSON + ela/runs/ela_3d_18299175 onto ORCD, then:
#      cd ~/orcd/scratch/ZoMBI-Hop
#      sbatch slurm/run_mobo_ela_18299175.sbatch
#
# Same GPU / requeue / seeding setup as slurm/run_mobo_ela_17849250.sbatch.
# No --share-history. Cancel chain: scancel <job_id>
# =============================================================================

#SBATCH --job-name=mobo_ela_18299175
#SBATCH --output=optimize/scripts/logs/mobo_ela_18299175_%j.out
#SBATCH --error=optimize/scripts/logs/mobo_ela_18299175_%j.err
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

ORACLE="${REPO}/ela/runs/ela_3d_18299175/best/oracle.py"
CONFIG="${REPO}/optimize/mobo_batch_configs/ela_3d_18299175.json"
if [[ ! -f "${ORACLE}" ]]; then
  echo "ERROR: missing ELA oracle at ${ORACLE}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: missing batch config at ${CONFIG}" >&2
  echo "Pick needles locally first:" >&2
  echo "  python ela/pick_optima.py --job-id 18299175" >&2
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
RUN_DIR="optimize/runs/mobo_ela_3d_18299175_job${JOB_TAG}"

MOBO_CMD=(
  python optimize/run_mobo.py
  --no-show
  --device cuda
  --batch
  --config optimize/mobo_batch_configs/ela_3d_18299175.json
  --time-limit 0.2
  --start-from-best
  optimize/runs/archived_runs/mobo_3d_05_06_15_32/trial_112
  optimize/runs/archived_runs/mobo_ensemble_3d_job16948629/trial_33
  optimize/runs/archived_runs/mobo_ensemble_3d_job16861902/trial_49
  optimize/runs/archived_runs/mobo_ensemble_3d_job16954340/trial_63
  --run-dir "${RUN_DIR}"
)

echo "=== MOBO on ELA twin 18299175 (hand-picked needles) ==="
printf '  %q' "${MOBO_CMD[@]}"
echo
echo "=== cwd: ${REPO} ==="
echo "=== oracle: ${ORACLE} ==="
echo "=== config: ${CONFIG} ==="
echo "=== run-dir: ${RUN_DIR} ==="
echo

"${MOBO_CMD[@]}" &
wait $!
rc=$?

if [ "${requeued}" -eq 0 ]; then
    if [ "${rc}" -eq 0 ]; then
        echo "[$(date)] run exited cleanly (rc=0); submitting next job"
        sbatch "$0"
    else
        echo "[$(date)] run exited with rc=${rc} (fatal abort/crash) — NOT resubmitting."
        echo "[$(date)] fix the issue and resubmit manually with: sbatch $0"
    fi
fi
