#!/bin/bash
# =============================================================================
# ela/scripts/run_pilot_structure_4d.sbatch — structure-hparams 4D ELA twin
# =============================================================================
#
# Compact structure fitness (PKS/FDC/DISP/Hmax/Lipschitz/…) on run_9dfe.
# Same fitness block as run_pilot_structure_3d.sbatch.
#
#   cd ~/orcd/scratch/ZoMBI-Hop
#   sbatch ela/scripts/run_pilot_structure_4d.sbatch
#
# Outputs: ela/runs/ela_4d_<JOBID>/
# =============================================================================

#SBATCH --job-name=ela_struct_4d
#SBATCH --output=ela/scripts/logs/ela_struct_4d_%j.out
#SBATCH --error=ela/scripts/logs/ela_struct_4d_%j.err
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO}"

export PILOT_DIM=4
export PILOT_CONFIG="${PILOT_CONFIG:-${REPO}/ela/pilot_config_structure_4d.json}"

# shellcheck source=/dev/null
source "${REPO}/ela/scripts/pilot_job_common.sh"
_pilot_setup_env
_pilot_check_data

RUN_DIR="$(_pilot_run_dir_slurm)"
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

echo "=== ELA structure-hparams pilot 4D ==="
_pilot_log_config_summary
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
echo "  cat ${RUN_DIR}/pilot_config.resolved.json"

exit "${rc}"
