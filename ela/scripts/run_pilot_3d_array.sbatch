#!/bin/bash
# =============================================================================
# ela/scripts/run_pilot_3d_array.sbatch — parallel landscape jobs (array)
# =============================================================================
#
# Submit from repo root:
#
#   cd ~/orcd/scratch/ZoMBI-Hop
#   sbatch ela/scripts/run_pilot_3d_array.sbatch
#
# Or:
#   bash ela/scripts/submit_pilot_3d.sh array
#
# Override array range:
#   sbatch --array=0-9 ela/scripts/run_pilot_3d_array.sbatch
# =============================================================================

#SBATCH --job-name=ela_3d_arr
#SBATCH --output=ela/scripts/logs/ela_3d_%j.out
#SBATCH --error=ela/scripts/logs/ela_3d_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --array=0-4

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"

# shellcheck source=/dev/null
source "${REPO}/ela/scripts/pilot_job_common.sh"
_pilot_setup_env
_pilot_check_data

RUN_DIR="$(_pilot_run_dir_array)"
mkdir -p "${RUN_DIR}"

PILOT_CMD=(
  python ela/run_pilot_3d.py
  --out-dir "${RUN_DIR}"
)

_pilot_append_config_flags PILOT_CMD
_pilot_append_mode_flags PILOT_CMD
_pilot_append_population_flags PILOT_CMD
_pilot_append_viz_flags PILOT_CMD
_pilot_append_worker_flags PILOT_CMD
_pilot_append_seed_flags PILOT_CMD

echo "=== ELA S1 pilot 3D (array) ==="
_pilot_log_config_summary
echo "array_job:  ${SLURM_ARRAY_JOB_ID}"
echo "task_id:    ${SLURM_ARRAY_TASK_ID}"
if [[ -v PILOT_SEED ]]; then
  echo "gp_seed:    ${PILOT_SEED} (fixed via PILOT_SEED)"
else
  echo "gp_seed:    auto (SLURM_JOB_ID=${SLURM_JOB_ID})"
fi
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
