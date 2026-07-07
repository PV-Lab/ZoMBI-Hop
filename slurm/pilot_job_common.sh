# Shared helpers for ELA S1 pilot Slurm jobs.
# Source after cd to REPO:
#   source "${REPO}/slurm/mobo_job_common.sh" && _mobo_setup_env
#   source "${REPO}/slurm/pilot_job_common.sh"

_pilot_setup_dirs() {
  export PILOT_RUN_ROOT="${PILOT_RUN_ROOT:-${REPO}/ela/runs}"
  export PILOT_LOG_DIR="${PILOT_LOG_DIR:-${REPO}/ela/scripts/logs}"
  mkdir -p "${PILOT_RUN_ROOT}" "${PILOT_LOG_DIR}" "${REPO}/ela/runs"
}

# Run dirs — Slurm jobs use stable names keyed on SLURM_JOB_ID:
#   slurm:  ela/runs/ela_3d_<SLURM_JOB_ID>
#   local:  ela/runs/pilot_3d_seed0_DD_MM_HH_MM_SS_<pid>  (run_pilot_3d.py --out-dir)
_pilot_run_dir_slurm() {
  echo "${PILOT_RUN_ROOT}/ela_3d_${SLURM_JOB_ID}"
}

_pilot_run_dir_array() {
  echo "${PILOT_RUN_ROOT}/ela_3d_${SLURM_JOB_ID}"
}

_pilot_check_data() {
  local db="${1:-${REPO}/data/2nd_real_run.db}"
  local target="${2:-${REPO}/data/2nd_real_run_ela_full.json}"
  if [[ ! -f "${db}" ]]; then
    echo "FATAL: campaign DB missing: ${db}" >&2
    echo "  data/ is gitignored — rsync from your workstation:" >&2
    echo "  rsync -av data/2nd_real_run.db login:~/orcd/scratch/ZoMBI-Hop/data/" >&2
    return 1
  fi
  python - "${db}" <<'PY'
import sqlite3
import sys
from pathlib import Path

db = Path(sys.argv[1])
size_mb = db.stat().st_size / (1024 * 1024)
if size_mb < 1.0:
    raise SystemExit(
        f"FATAL: {db} is only {size_mb:.2f} MB (expected ~5.5 MB). "
        "Copy the real campaign DB from your workstation."
    )

con = sqlite3.connect(db)
tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
if "results" not in tables:
    raise SystemExit(
        f"FATAL: {db} has no 'results' table (found: {tables or 'none'}). "
        "Copy the real campaign DB from your workstation."
    )
n = con.execute(
    'SELECT COUNT(*) FROM results WHERE "FAPbI3" IS NOT NULL '
    'AND "MAPbI3" IS NOT NULL AND "MAPbBr3" IS NOT NULL AND "Objective" IS NOT NULL'
).fetchone()[0]
con.close()
if n < 100:
    raise SystemExit(f"FATAL: only {n} complete campaign rows in {db} (expected ~644)")
print(f"db OK: {db.name} rows={n} size={size_mb:.1f}MB")
PY
  if [[ ! -f "${target}" ]]; then
    echo "Tier-1 target missing; generating from ${db} (one-time, ~1-2 min)..."
    python "${REPO}/ela/compute_lambda_target.py" \
      --db "${db}" \
      --full \
      --out "${target}"
  fi
  python - "${target}" <<'PY'
import json
import sys
from pathlib import Path

target = Path(sys.argv[1])
data = json.loads(target.read_text())
if "feature_groups" not in data and "features" not in data:
    raise SystemExit(f"FATAL: {target} is not a valid ELA target JSON")
print(f"target OK: {target.name}")
PY
}

_pilot_append_mode_flags() {
  local -n _cmd=$1
  if [[ "${PILOT_CAMPAIGN_MODE:-0}" == "1" ]]; then
    _cmd+=(--campaign-mode)
    if [[ -n "${PILOT_ALPHA:-}" ]]; then
      _cmd+=(--alpha "${PILOT_ALPHA}")
    fi
    if [[ -n "${PILOT_TIER1_GAMMA:-}" ]]; then
      _cmd+=(--tier1-gamma "${PILOT_TIER1_GAMMA}")
    fi
  fi
}

_pilot_append_viz_flags() {
  local -n _cmd=$1
  if [[ "${PILOT_QUICK:-0}" == "1" ]]; then
    _cmd+=(--quick)
  fi
  if [[ "${PILOT_NO_LANDSCAPE_VIZ:-0}" == "1" ]]; then
    _cmd+=(--no-landscape-viz)
  fi
  if [[ "${PILOT_NO_VIZ:-0}" == "1" ]]; then
    _cmd+=(--no-viz)
  fi
}
