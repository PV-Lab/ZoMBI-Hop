#!/bin/bash
# =============================================================================
# ela/scripts/run_pilot_3d_array.sbatch — alias for slurm/run_pilot_3d_array.sbatch
# =============================================================================

#SBATCH --job-name=ela_3d_arr
#SBATCH --output=ela/scripts/logs/ela_3d_%j.out
#SBATCH --error=ela/scripts/logs/ela_3d_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --array=0-4

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"

# shellcheck source=/dev/null
source "${REPO}/slurm/mobo_job_common.sh"
_mobo_setup_env
# shellcheck source=/dev/null
source "${REPO}/slurm/pilot_job_common.sh"
_pilot_setup_dirs
_pilot_check_data

PILOT_SEED="${PILOT_SEED:-${SLURM_ARRAY_TASK_ID}}"
PILOT_POPULATION="${PILOT_POPULATION:-200}"
PILOT_GENERATIONS="${PILOT_GENERATIONS:-100}"
PILOT_DB="${PILOT_DB:-data/2nd_real_run.db}"
PILOT_TARGET="${PILOT_TARGET:-data/2nd_real_run_ela_full.json}"
PILOT_LANDSCAPE_EVERY="${PILOT_LANDSCAPE_EVERY:-5}"

RUN_DIR="$(_pilot_run_dir_array)"
mkdir -p "${RUN_DIR}"

PILOT_CMD=(
  python ela/run_pilot_3d.py
  --db "${PILOT_DB}"
  --target "${PILOT_TARGET}"
  --out-dir "${RUN_DIR}"
  --seed "${PILOT_SEED}"
  --population "${PILOT_POPULATION}"
  --generations "${PILOT_GENERATIONS}"
  --landscape-every "${PILOT_LANDSCAPE_EVERY}"
  --grid-n 200
  --landscape-grid-n 100
)

_pilot_append_mode_flags PILOT_CMD
_pilot_append_viz_flags PILOT_CMD

echo "=== ELA S1 pilot 3D (array) ==="
if [[ "${PILOT_CAMPAIGN_MODE:-0}" == "1" ]]; then
  echo "mode:       campaign-twin"
else
  echo "mode:       paper (Muñoz S1, RF λ_T)"
fi
echo "array_job:  ${SLURM_ARRAY_JOB_ID}"
echo "task_id:    ${SLURM_ARRAY_TASK_ID}"
echo "seed:       ${PILOT_SEED}"
echo "run_dir:    ${RUN_DIR}"
echo ""
printf '  %q' "${PILOT_CMD[@]}"
echo
echo "=== cwd: ${REPO} ==="
echo

"${PILOT_CMD[@]}" &
wait $!
rc=$?

echo "exit_code: ${rc}"
echo "artifacts: ${RUN_DIR}"
exit "${rc}"
