#!/bin/bash
# =============================================================================
# ela/scripts/run_pilot_3d_array.sbatch — 5 independent pilots (seeds 0–4)
# =============================================================================
#
#   cd ~/orcd/scratch/ZoMBI-Hop
#   sbatch ela/scripts/run_pilot_3d_array.sbatch
#   bash ela/scripts/submit_pilot_3d.sh array
# =============================================================================

#SBATCH --job-name=pilot_3d_s1_arr
#SBATCH --output=ela/scripts/logs/pilot_3d_array_%A_%a.out
#SBATCH --error=ela/scripts/logs/pilot_3d_array_%A_%a.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --array=0-4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/pilot_job_common.sh"
REPO="$(_pilot_resolve_repo "${SCRIPT_DIR}")"
cd "${REPO}"
_pilot_setup_env

PILOT_SEED="${PILOT_SEED:-${SLURM_ARRAY_TASK_ID}}"
PILOT_POPULATION="${PILOT_POPULATION:-150}"
PILOT_GENERATIONS="${PILOT_GENERATIONS:-80}"
PILOT_TIER1_GAMMA="${PILOT_TIER1_GAMMA:-8}"
PILOT_ALPHA="${PILOT_ALPHA:-2}"
PILOT_DB="${PILOT_DB:-data/2nd_real_run.db}"
PILOT_TARGET="${PILOT_TARGET:-data/2nd_real_run_ela_full.json}"
PILOT_LANDSCAPE_EVERY="${PILOT_LANDSCAPE_EVERY:-5}"
RUN_DIR="${PILOT_RUN_ROOT}/pilot_3d_seed${PILOT_SEED}_arr${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

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

echo "=== array task ${SLURM_ARRAY_TASK_ID} seed=${PILOT_SEED} repo=${REPO} ==="
echo "run_dir=${RUN_DIR}"
printf '  %q' "${PILOT_CMD[@]}"
echo

"${PILOT_CMD[@]}"
