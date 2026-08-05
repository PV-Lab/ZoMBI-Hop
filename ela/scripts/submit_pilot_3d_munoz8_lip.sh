#!/usr/bin/env bash
# Submit Muñoz-8 + Lipschitz hybrid pilot (individual job).
#
# From repo root on ORCD login node:
#   bash ela/scripts/submit_pilot_3d_munoz8_lip.sh
#
# Config: ela/pilot_config_munoz8_lipschitz.json
# Outputs: ela/runs/ela_3d_<JOBID>/

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS="${REPO}/ela/scripts"
CONFIG="${REPO}/ela/pilot_config_munoz8_lipschitz.json"

export REPO
export PILOT_CONFIG="${CONFIG}"

cd "${REPO}"
mkdir -p "${SCRIPTS}/logs" "${REPO}/ela/runs"

# shellcheck source=/dev/null
source "${SCRIPTS}/pilot_job_common.sh"
_pilot_configure_parallelism
_pilot_setup_dirs
_pilot_check_data

PARTITION="${PILOT_PARTITION:-mit_normal}"
TIME_LIMIT="${PILOT_TIME:-12:00:00}"
CPUS="${PILOT_CPUS:-32}"
MEM="${PILOT_MEM:-128G}"

echo "Submitting munoz8-lipschitz pilot from ${REPO}"
_pilot_log_config_summary
echo "  eval_workers: ${PILOT_EVAL_WORKERS} (OMP=${OMP_NUM_THREADS}/worker)"

job_id=$(
  sbatch --export=ALL \
    --partition="${PARTITION}" \
    --cpus-per-task="${CPUS}" \
    --mem="${MEM}" \
    --time="${TIME_LIMIT}" \
    --job-name=ela_3d_m8lip \
    "${SCRIPTS}/run_pilot_3d.sbatch" | awk '{print $NF}'
)

echo "Submitted job ${job_id}"
echo "  logs: ${REPO}/ela/scripts/logs/ela_3d_${job_id}.{out,err}"
echo "  runs: ${PILOT_RUN_ROOT}/ela_3d_${job_id}"
echo "  squeue -j ${job_id}"
