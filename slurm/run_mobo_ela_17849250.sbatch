#!/bin/bash
# =============================================================================
# MOBO hparam training on ELA twin ela_3d_17849250 (12 interior catchments).
#
# Mirrors slurm/run_mobo_manual_ensemble_3d.sbatch resource / requeue / seeding
# setup, but runs a fixed ELA landscape batch config and does NOT pass
# --share-history.
#
# Submit from repo root on ORCD:
#   cd ~/orcd/scratch/ZoMBI-Hop
#   sbatch slurm/run_mobo_ela_17849250.sbatch
#
# Cancel the auto-continue chain with: scancel <job_id>
# =============================================================================

#SBATCH --job-name=mobo_ela_17849250
#SBATCH --output=optimize/scripts/logs/mobo_ela_17849250_%j.out
#SBATCH --error=optimize/scripts/logs/mobo_ela_17849250_%j.err
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

ORACLE="${REPO}/ela/runs/ela_3d_17849250/best/oracle.py"
CONFIG="${REPO}/optimize/mobo_batch_configs/ela_3d_17849250_interior12.json"
if [[ ! -f "${ORACLE}" ]]; then
  echo "ERROR: missing ELA oracle at ${ORACLE}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: missing batch config at ${CONFIG}" >&2
  exit 1
fi

# --- auto-continue across the wall-time limit (same pattern as ensemble 3d) ---
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

# Stable run dir keyed on job id so a requeue resumes the same folder.
JOB_TAG="${SLURM_JOB_ID:-local}"
RUN_DIR="optimize/runs/mobo_ela_3d_17849250_interior12_job${JOB_TAG}"

# Typical 3d MOBO knobs: --time-limit 0.2, seed from archived best trials.
# No --share-history (fixed ELA twin; do not pull collaborator ensemble history).
MOBO_CMD=(
  python optimize/run_mobo.py
  --no-show
  --device cuda
  --batch
  --config optimize/mobo_batch_configs/ela_3d_17849250_interior12.json
  --time-limit 0.2
  --start-from-best
  optimize/runs/archived_runs/mobo_3d_05_06_15_32/trial_112
  optimize/runs/archived_runs/mobo_ensemble_3d_job16948629/trial_33
  optimize/runs/archived_runs/mobo_ensemble_3d_job16861902/trial_49
  optimize/runs/archived_runs/mobo_ensemble_3d_job16954340/trial_63
  --run-dir "${RUN_DIR}"
)

echo "=== MOBO on ELA twin 17849250 (12 interior catchments) ==="
printf '  %q' "${MOBO_CMD[@]}"
echo
echo "=== cwd: ${REPO} ==="
echo "=== oracle: ${ORACLE} ==="
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
