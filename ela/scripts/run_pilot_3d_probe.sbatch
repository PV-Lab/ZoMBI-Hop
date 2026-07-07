#!/bin/bash
# =============================================================================
# ela/scripts/run_pilot_3d_probe.sbatch — cluster smoke test (~5 min)
# =============================================================================
#
#   cd ~/orcd/scratch/ZoMBI-Hop
#   sbatch ela/scripts/run_pilot_3d_probe.sbatch
#
# IMPORTANT: sbatch must be run from the repo root (SLURM_SUBMIT_DIR).
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

# Slurm executes a copy under /var/spool/slurmd/ — never use BASH_SOURCE for paths.
REPO="${SLURM_SUBMIT_DIR:-}"
_search="${REPO}"
while [[ -n "${_search}" && "${_search}" != "/" ]]; do
  if [[ -f "${_search}/ela/scripts/pilot_job_common.sh" ]]; then
    REPO="${_search}"
    break
  fi
  _search="$(dirname "${_search}")"
done
if [[ ! -f "${REPO}/ela/scripts/pilot_job_common.sh" ]]; then
  echo "FATAL: repo not found from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-}" >&2
  echo "  cd ~/orcd/scratch/ZoMBI-Hop && sbatch ela/scripts/run_pilot_3d_probe.sbatch" >&2
  exit 1
fi
cd "${REPO}"
# shellcheck source=/dev/null
source "${REPO}/ela/scripts/pilot_job_common.sh"
_pilot_setup_env
_pilot_check_data

RUN_DIR="$(_pilot_run_dir_slurm 0 1)"
mkdir -p "${RUN_DIR}"

{
  echo "=== Pilot probe ==="
  echo "job_id:   ${SLURM_JOB_ID}"
  echo "repo:     ${REPO}"
  echo "hostname: $(hostname)"
  echo "python:   $(which python)"
  python -c "import sklearn, scipy, numpy; print('numpy', numpy.__version__, 'sklearn', sklearn.__version__)"
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
