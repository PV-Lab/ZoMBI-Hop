#!/usr/bin/env bash
# Submit ELA S1 3D pilot jobs to ORCD / Engaging Slurm.
#
# Intended layout on ORCD scratch:
#   ~/orcd/scratch/ZoMBI-Hop/          ← repo clone (REPO)
#   ~/orcd/scratch/ZoMBI-Hop/ela/runs/ ← pilot outputs (default)
#
# Usage (from repo root on login node):
#   bash ela/scripts/submit_pilot_3d.sh
#   bash ela/scripts/submit_pilot_3d.sh probe
#   bash ela/scripts/submit_pilot_3d.sh array
#
# Or submit directly (same style as MOBO ensemble jobs):
#   sbatch slurm/run_pilot_3d.sbatch
#   sbatch slurm/run_pilot_3d_probe.sbatch
#   sbatch slurm/run_pilot_3d_array.sbatch
#
# Paper mode (default): Muñoz S1, ELA-only on raw g(z), RF λ_T target.
# Campaign-twin: PILOT_CAMPAIGN_MODE=1 bash ela/scripts/submit_pilot_3d.sh
#
# Environment overrides:
#   PILOT_SEED=0  PILOT_POPULATION=200  PILOT_GENERATIONS=100
#   PILOT_CAMPAIGN_MODE=1  PILOT_ALPHA=3  PILOT_TIER1_GAMMA=5
#   PILOT_QUICK=1  PILOT_NO_LANDSCAPE_VIZ=1  PILOT_NO_VIZ=1
#   PILOT_RUN_ROOT=...   (default: ${REPO}/ela/runs)
#   PILOT_PARTITION=mit_normal  PILOT_TIME=12:00:00

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-single}"
SLURM_DIR="${REPO}/slurm"

cd "${REPO}"
mkdir -p "${REPO}/ela/scripts/logs" "${REPO}/ela/runs"

# shellcheck source=/dev/null
source "${SLURM_DIR}/pilot_job_common.sh"
export REPO
_pilot_setup_dirs
_pilot_check_data

PARTITION="${PILOT_PARTITION:-mit_normal}"
TIME_LIMIT="${PILOT_TIME:-12:00:00}"
CPUS="${PILOT_CPUS:-32}"
MEM="${PILOT_MEM:-64G}"

sbatch_common=(
  --partition="${PARTITION}"
  --cpus-per-task="${CPUS}"
  --mem="${MEM}"
)

case "${MODE}" in
  single|"")
    SBATCH="${SLURM_DIR}/run_pilot_3d.sbatch"
    sbatch_common+=(--time="${TIME_LIMIT}" --job-name=pilot_3d_s1)
    ;;
  probe)
    SBATCH="${SLURM_DIR}/run_pilot_3d_probe.sbatch"
    sbatch_common+=(--time=00:30:00 --job-name=pilot_3d_probe --cpus-per-task=4 --mem=16G)
    ;;
  array)
    SBATCH="${SLURM_DIR}/run_pilot_3d_array.sbatch"
    sbatch_common+=(--time="${TIME_LIMIT}" --job-name=pilot_3d_s1_arr)
    if [[ -n "${PILOT_ARRAY:-}" ]]; then
      sbatch_common+=(--array="${PILOT_ARRAY}")
    fi
    ;;
  *)
    echo "Usage: bash ela/scripts/submit_pilot_3d.sh [single|probe|array]" >&2
    exit 1
    ;;
esac

echo "Submitting pilot (${MODE}) from ${REPO}"
if [[ "${PILOT_CAMPAIGN_MODE:-0}" == "1" ]]; then
  echo "  mode: campaign-twin"
else
  echo "  mode: paper (Muñoz S1)"
fi
echo "  sbatch: ${SBATCH}"

job_id=$(sbatch --export=ALL "${sbatch_common[@]}" "${SBATCH}" | awk '{print $NF}')
echo "Submitted job ${job_id}"
echo "  logs: ${REPO}/ela/scripts/logs/"
echo "  runs: ${PILOT_RUN_ROOT}/pilot_3d_seed*_job*"
echo "  squeue -j ${job_id}"
