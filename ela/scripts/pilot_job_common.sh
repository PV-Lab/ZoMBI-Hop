# Shared conda + thread env for ELA S1 pilot Slurm jobs.
# Source from ela/scripts/run_pilot_3d*.sbatch after REPO is set.

_pilot_resolve_repo() {
  local script_dir="$1"
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/ela/run_pilot_3d.py" ]]; then
    echo "${SLURM_SUBMIT_DIR}"
    return 0
  fi
  local root
  root="$(cd "${script_dir}/../.." && pwd)"
  if [[ -f "${root}/ela/run_pilot_3d.py" ]]; then
    echo "${root}"
    return 0
  fi
  echo "${SLURM_SUBMIT_DIR:-${root}}"
}

_pilot_setup_env() {
  # Reuse MOBO cluster env (conda zombi-hop-linebo, Agg matplotlib, thread caps).
  # shellcheck source=/dev/null
  source "${REPO}/slurm/mobo_job_common.sh"
  _mobo_setup_env

  ELA_SCRIPTS="${REPO}/ela/scripts"
  export PILOT_RUN_ROOT="${PILOT_RUN_ROOT:-${REPO}/ela/runs}"
  export PILOT_LOG_DIR="${PILOT_LOG_DIR:-${ELA_SCRIPTS}/logs}"
  mkdir -p "${PILOT_RUN_ROOT}" "${PILOT_LOG_DIR}"
}
