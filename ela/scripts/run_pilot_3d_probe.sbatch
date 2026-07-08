#!/bin/bash
# =============================================================================
# ela/scripts/run_pilot_3d_probe.sbatch — cluster smoke test (~5 min)
# =============================================================================
#
# Submit from repo root:
#
#   cd ~/orcd/scratch/ZoMBI-Hop
#   sbatch ela/scripts/run_pilot_3d_probe.sbatch
#
# Uses ela/pilot_config_probe.json by default.
# =============================================================================

#SBATCH --job-name=ela_3d_probe
#SBATCH --output=ela/scripts/logs/ela_3d_%j.out
#SBATCH --error=ela/scripts/logs/ela_3d_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --ntasks=1

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"

# shellcheck source=/dev/null
source "${REPO}/ela/scripts/pilot_job_common.sh"
export PILOT_CONFIG="${PILOT_CONFIG:-${REPO}/ela/pilot_config_probe.json}"
PILOT_EVAL_WORKERS="${PILOT_EVAL_WORKERS:-2}"
_pilot_setup_env
_pilot_check_data

RUN_DIR="$(_pilot_run_dir_slurm)"
mkdir -p "${RUN_DIR}"

PILOT_CMD=(
  python ela/run_pilot_3d.py
  --out-dir "${RUN_DIR}"
  --eval-workers "${PILOT_EVAL_WORKERS}"
)

_pilot_append_config_flags PILOT_CMD
_pilot_append_mode_flags PILOT_CMD
_pilot_append_seed_flags PILOT_CMD

echo "=== Pilot probe ==="
_pilot_log_config_summary
echo "job_id:     ${SLURM_JOB_ID}"
echo "repo:       ${REPO}"
echo "hostname:   $(hostname)"
echo "run_dir:    ${RUN_DIR}"
echo "python:     $(which python)"
python -c "import sklearn, scipy, numpy; print('numpy', numpy.__version__, 'sklearn', sklearn.__version__)"
echo ""
printf '  %q' "${PILOT_CMD[@]}"
echo
echo "=== cwd: ${REPO} ==="
echo

"${PILOT_CMD[@]}" &
wait $!
rc=$?

if [[ "${rc}" -eq 0 ]]; then
  echo "Probe OK -> ${RUN_DIR}"
else
  echo "Probe FAILED (rc=${rc})"
fi
exit "${rc}"
