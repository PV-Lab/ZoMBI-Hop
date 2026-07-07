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
# Environment overrides:
#   PILOT_SEED=0  PILOT_POPULATION=150  PILOT_GENERATIONS=80
#   PILOT_TIER1_GAMMA=8  PILOT_ALPHA=2  PILOT_QUICK=1
#   PILOT_RUN_ROOT=...   (default: ${REPO}/ela/runs)
#   PILOT_PARTITION=mit_normal  PILOT_TIME=08:00:00

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="${1:-single}"
SCRIPTS="${REPO}/ela/scripts"

cd "${REPO}"
mkdir -p "${SCRIPTS}/logs" "${REPO}/ela/runs"

# shellcheck source=/dev/null
source "${SCRIPTS}/pilot_job_common.sh"
_pilot_check_data

PARTITION="${PILOT_PARTITION:-mit_normal}"
TIME_LIMIT="${PILOT_TIME:-08:00:00}"
CPUS="${PILOT_CPUS:-16}"
MEM="${PILOT_MEM:-32G}"

sbatch_common=(
  --partition="${PARTITION}"
  --cpus-per-task="${CPUS}"
  --mem="${MEM}"
)

case "${MODE}" in
  single|"")
    SBATCH="${SCRIPTS}/run_pilot_3d.sbatch"
    sbatch_common+=(--time="${TIME_LIMIT}" --job-name=pilot_3d_s1)
    ;;
  probe)
    SBATCH="${SCRIPTS}/run_pilot_3d_probe.sbatch"
    sbatch_common+=(--time=00:30:00 --job-name=pilot_3d_probe --cpus-per-task=4 --mem=16G)
    ;;
  array)
    SBATCH="${SCRIPTS}/run_pilot_3d_array.sbatch"
    sbatch_common+=(--time="${TIME_LIMIT}" --job-name=pilot_3d_s1_arr)
    ;;
  *)
    echo "Usage: bash ela/scripts/submit_pilot_3d.sh [single|probe|array]" >&2
    exit 1
    ;;
esac

echo "Submitting pilot (${MODE}) from ${REPO}"
echo "  sbatch: ${SBATCH}"

job_id=$(sbatch --export=ALL "${sbatch_common[@]}" "${SBATCH}" | awk '{print $NF}')
echo "Submitted job ${job_id}"
echo "  logs: ${SCRIPTS}/logs/"
echo "  runs: ${PILOT_RUN_ROOT:-${REPO}/ela/runs}/pilot_3d_seed*_job*"
echo "  squeue -j ${job_id}"
