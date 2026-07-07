#!/bin/bash
# =============================================================================
# slurm/run_pilot_3d_probe.sbatch — cluster smoke test (~5 min)
# =============================================================================
#
# Submit from repo root:
#
#   cd ~/orcd/scratch/ZoMBI-Hop
#   sbatch slurm/run_pilot_3d_probe.sbatch
#
# Or:
#   bash ela/scripts/submit_pilot_3d.sh probe
# =============================================================================

#SBATCH --job-name=pilot_3d_probe
#SBATCH --output=ela/scripts/logs/pilot_3d_probe_%j.out
#SBATCH --error=ela/scripts/logs/pilot_3d_probe_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --ntasks=1

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO}"

# shellcheck source=/dev/null
source "${REPO}/slurm/mobo_job_common.sh"
_mobo_setup_env
# shellcheck source=/dev/null
source "${REPO}/slurm/pilot_job_common.sh"
_pilot_setup_dirs
_pilot_check_data

RUN_DIR="$(_pilot_run_dir_slurm 0 1)"
mkdir -p "${RUN_DIR}"

PILOT_CMD=(
  python ela/run_pilot_3d.py
  --quick
  --early-reject-mult 0
  --no-landscape-viz
  --out-dir "${RUN_DIR}"
  --seed 0
  --population 12
  --generations 4
)

echo "=== Pilot probe ==="
echo "mode:       paper (Muñoz S1, RF λ_T)"
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
