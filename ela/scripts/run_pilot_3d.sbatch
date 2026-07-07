#!/bin/bash
# =============================================================================
# ela/scripts/run_pilot_3d.sbatch — Muñoz S1 3D GP landscape recreation
# =============================================================================
#
# Submit from repo root on ORCD / Engaging:
#
#   cd ~/orcd/scratch/ZoMBI-Hop
#   sbatch ela/scripts/run_pilot_3d.sbatch
#
# Or:
#   bash ela/scripts/submit_pilot_3d.sh
#
# Paper mode (default): ELA-only fitness on raw g(z), Muñoz 8 features, RF λ_T.
# Campaign-twin (RMSE anchor + calibration):
#   PILOT_CAMPAIGN_MODE=1 sbatch ela/scripts/run_pilot_3d.sbatch
# Fixed GP seed (optional — default is SLURM_JOB_ID on cluster, random locally):
#   PILOT_SEED=42 sbatch ela/scripts/run_pilot_3d.sbatch
#
# Probe smoke test (~5 min):
#   sbatch ela/scripts/run_pilot_3d_probe.sbatch
#
# Prerequisites:
#   data/2nd_real_run.db, data/2nd_real_run_ela_full.json
#   conda env zombi-hop-linebo (slurm/mobo_job_common.sh)
#
# Outputs:
#   ela/runs/ela_3d_<JOBID>/
#   ela/scripts/logs/ela_3d_<JOBID>.{out,err}
# =============================================================================

#SBATCH --job-name=ela_3d
#SBATCH --output=ela/scripts/logs/ela_3d_%j.out
#SBATCH --error=ela/scripts/logs/ela_3d_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"

# shellcheck source=/dev/null
source "${REPO}/ela/scripts/pilot_job_common.sh"
_pilot_setup_env
_pilot_check_data

PILOT_POPULATION="${PILOT_POPULATION:-400}"
PILOT_GENERATIONS="${PILOT_GENERATIONS:-100}"
PILOT_DB="${PILOT_DB:-data/2nd_real_run.db}"
PILOT_TARGET="${PILOT_TARGET:-data/2nd_real_run_ela_full.json}"
PILOT_LANDSCAPE_EVERY="${PILOT_LANDSCAPE_EVERY:-1}"

RUN_DIR="$(_pilot_run_dir_slurm)"
mkdir -p "${RUN_DIR}"

PILOT_CMD=(
  python ela/run_pilot_3d.py
  --db "${PILOT_DB}"
  --target "${PILOT_TARGET}"
  --out-dir "${RUN_DIR}"
  --population "${PILOT_POPULATION}"
  --generations "${PILOT_GENERATIONS}"
  --landscape-every "${PILOT_LANDSCAPE_EVERY}"
  --grid-n 200
  --landscape-grid-n 100
)

_pilot_append_mode_flags PILOT_CMD
_pilot_append_viz_flags PILOT_CMD
_pilot_append_worker_flags PILOT_CMD
_pilot_append_seed_flags PILOT_CMD

echo "=== ELA S1 pilot 3D ==="
if [[ "${PILOT_CAMPAIGN_MODE:-0}" == "1" ]]; then
  echo "mode:       campaign-twin (RMSE + calibration)"
else
  echo "mode:       paper (Muñoz S1, RF λ_T)"
fi
echo "started:    $(date -Is)"
echo "job_id:     ${SLURM_JOB_ID}"
echo "partition:  ${SLURM_JOB_PARTITION:-unknown}"
echo "hostname:   $(hostname)"
echo "repo:       ${REPO}"
echo "run_dir:    ${RUN_DIR}"
echo "cpus/task:  ${SLURM_CPUS_PER_TASK:-?}"
echo "eval_workers: ${PILOT_EVAL_WORKERS}"
echo "omp/worker: ${OMP_NUM_THREADS}"
if [[ -v PILOT_SEED ]]; then
  echo "gp_seed:    ${PILOT_SEED} (fixed via PILOT_SEED)"
else
  echo "gp_seed:    auto (SLURM_JOB_ID=${SLURM_JOB_ID})"
fi
echo ""
printf '  %q' "${PILOT_CMD[@]}"
echo
echo "=== cwd: ${REPO} ==="
echo

WALL_START=$(date +%s)
"${PILOT_CMD[@]}" &
wait $!
rc=$?
WALL_END=$(date +%s)

echo "wall_clock_s: $(( WALL_END - WALL_START ))"
echo "exit_code:    ${rc}"
echo "finished:     $(date -Is)"
echo "artifacts:    ${RUN_DIR}"
echo "  tail -f ${RUN_DIR}/pilot.log"

exit "${rc}"
