#!/bin/bash
# =============================================================================
# ela/scripts/run_pilot_3d.sbatch — 3D S1 GP landscape recreation
# =============================================================================
#
# CPU-only (sklearn ELA + symbolic GP). Submit from scratch repo root:
#
#   cd ~/orcd/scratch/ZoMBI-Hop
#   sbatch ela/scripts/run_pilot_3d.sbatch
#
# Or:
#   bash ela/scripts/submit_pilot_3d.sh
#
# Prerequisites:
#   data/2nd_real_run.db, data/2nd_real_run_ela_full.json
#   conda env zombi-hop-linebo (slurm/mobo_job_common.sh)
#
# Outputs:
#   ela/runs/pilot_3d_seed<SEED>_job<JOBID>/
#   ela/scripts/logs/pilot_3d_<JOBID>.{out,err}
# =============================================================================

#SBATCH --job-name=pilot_3d_s1
#SBATCH --output=ela/scripts/logs/pilot_3d_%j.out
#SBATCH --error=ela/scripts/logs/pilot_3d_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --ntasks=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/pilot_job_common.sh"
REPO="$(_pilot_resolve_repo "${SCRIPT_DIR}")"
cd "${REPO}"
_pilot_setup_env

PILOT_SEED="${PILOT_SEED:-0}"
PILOT_POPULATION="${PILOT_POPULATION:-150}"
PILOT_GENERATIONS="${PILOT_GENERATIONS:-80}"
PILOT_TIER1_GAMMA="${PILOT_TIER1_GAMMA:-8}"
PILOT_ALPHA="${PILOT_ALPHA:-2}"
PILOT_DB="${PILOT_DB:-data/2nd_real_run.db}"
PILOT_TARGET="${PILOT_TARGET:-data/2nd_real_run_ela_full.json}"
PILOT_LANDSCAPE_EVERY="${PILOT_LANDSCAPE_EVERY:-1}"
RUN_DIR="${PILOT_RUN_ROOT}/pilot_3d_seed${PILOT_SEED}_job${SLURM_JOB_ID}"

PILOT_CMD=(
  python ela/run_pilot_3d.py
  --db "${PILOT_DB}"
  --target "${PILOT_TARGET}"
  --out-dir "${RUN_DIR}"
  --seed "${PILOT_SEED}"
  --population "${PILOT_POPULATION}"
  --generations "${PILOT_GENERATIONS}"
  --tier1-gamma "${PILOT_TIER1_GAMMA}"
  --alpha "${PILOT_ALPHA}"
  --landscape-every "${PILOT_LANDSCAPE_EVERY}"
  --grid-n 200
  --landscape-grid-n 100
)

if [[ "${PILOT_QUICK:-0}" == "1" ]]; then
  PILOT_CMD+=(--quick)
fi
if [[ "${PILOT_NO_LANDSCAPE_VIZ:-0}" == "1" ]]; then
  PILOT_CMD+=(--no-landscape-viz)
fi
if [[ "${PILOT_NO_VIZ:-0}" == "1" ]]; then
  PILOT_CMD+=(--no-viz)
fi

{
  echo "=== ELA S1 pilot 3D ==="
  echo "started:    $(date -Is)"
  echo "job_id:     ${SLURM_JOB_ID}"
  echo "partition:  ${SLURM_JOB_PARTITION:-unknown}"
  echo "hostname:   $(hostname)"
  echo "repo:       ${REPO}"
  echo "run_dir:    ${RUN_DIR}"
  echo "cpus/task:  ${SLURM_CPUS_PER_TASK:-?}"
  echo ""
  printf '  %q' "${PILOT_CMD[@]}"
  echo
  echo ""
} | tee "${RUN_DIR}.submit.log" 2>/dev/null || true

WALL_START=$(date +%s)
"${PILOT_CMD[@]}"
rc=$?
WALL_END=$(date +%s)

{
  echo "wall_clock_s: $(( WALL_END - WALL_START ))"
  echo "exit_code:    ${rc}"
  echo "finished:     $(date -Is)"
  echo "artifacts:    ${RUN_DIR}"
  echo "  tail -f ${RUN_DIR}/pilot.log"
} | tee -a "${RUN_DIR}.submit.log" 2>/dev/null || true

exit "${rc}"
