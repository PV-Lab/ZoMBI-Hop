#!/bin/bash
# =============================================================================
# ela/scripts/run_pilot_3d_probe.sbatch — cluster smoke test (~5 min)
# =============================================================================
#
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/pilot_job_common.sh"
REPO="$(_pilot_resolve_repo "${SCRIPT_DIR}")"
cd "${REPO}"
_pilot_setup_env

RUN_DIR="${PILOT_RUN_ROOT}/pilot_3d_probe_job${SLURM_JOB_ID}"

{
  echo "=== Pilot probe ==="
  echo "job_id:   ${SLURM_JOB_ID}"
  echo "repo:     ${REPO}"
  echo "hostname: $(hostname)"
  echo "python:   $(which python)"
  python -c "import sklearn, scipy, numpy; print('numpy', numpy.__version__, 'sklearn', sklearn.__version__)"
  test -f "${REPO}/data/2nd_real_run.db" && echo "db: OK" || echo "db: MISSING"
  test -f "${REPO}/data/2nd_real_run_ela_full.json" && echo "target: OK" || echo "target: MISSING"
  echo ""
} | tee "${RUN_DIR}.probe.log"

python ela/run_pilot_3d.py \
  --quick \
  --early-reject-mult 0 \
  --no-landscape-viz \
  --out-dir "${RUN_DIR}" \
  --seed 0 \
  --population 12 \
  --generations 4

echo "Probe OK -> ${RUN_DIR}"
