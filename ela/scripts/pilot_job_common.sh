# Shared helpers for ELA S1 pilot Slurm jobs.
# Source after cd to REPO:
#   source "${REPO}/ela/scripts/pilot_job_common.sh"
#   _pilot_setup_env

_pilot_default_config() {
  echo "${PILOT_CONFIG:-${REPO}/ela/pilot_config.json}"
}

_pilot_setup_dirs() {
  export PILOT_RUN_ROOT="${PILOT_RUN_ROOT:-${REPO}/ela/runs}"
  export PILOT_LOG_DIR="${PILOT_LOG_DIR:-${REPO}/ela/scripts/logs}"
  mkdir -p "${PILOT_RUN_ROOT}" "${PILOT_LOG_DIR}" "${REPO}/ela/runs"
}

_pilot_configure_parallelism() {
  local cpus="${SLURM_CPUS_PER_TASK:-${PILOT_CPUS:-$(nproc 2>/dev/null || echo 8)}}"
  local workers="${PILOT_EVAL_WORKERS:-8}"
  workers=$(( workers > 0 ? workers : 1 ))
  if (( workers > cpus )); then
    workers=$cpus
  fi
  local omp="${PILOT_OMP_THREADS:-$(( (cpus + workers - 1) / workers ))}"
  if (( omp < 1 )); then
    omp=1
  fi
  export PILOT_EVAL_WORKERS=$workers
  export OMP_NUM_THREADS=$omp
  export MKL_NUM_THREADS=$omp
  export OPENBLAS_NUM_THREADS=$omp
}

_pilot_setup_env() {
  # shellcheck source=/dev/null
  source "${REPO}/slurm/mobo_job_common.sh"
  _mobo_setup_env
  _pilot_configure_parallelism
  _pilot_setup_dirs
}

# Run dirs — Slurm jobs use stable names keyed on SLURM_JOB_ID:
#   ela/runs/ela_3d_<SLURM_JOB_ID>
_pilot_run_dir_slurm() {
  echo "${PILOT_RUN_ROOT}/ela_3d_${SLURM_JOB_ID}"
}

_pilot_run_dir_array() {
  echo "${PILOT_RUN_ROOT}/ela_3d_${SLURM_JOB_ID}"
}

_pilot_check_data() {
  local db="${1:-}"
  local target="${2:-}"
  local cfg
  cfg="$(_pilot_default_config)"

  # Prefer db/target from the active pilot config when not passed explicitly.
  if [[ (-z "${db}" || -z "${target}") && -f "${cfg}" ]]; then
    local resolved
    resolved="$(
      python - "${REPO}" "${cfg}" <<'PY'
import json
import sys
from pathlib import Path

repo, cfg_path = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.loads(cfg_path.read_text())
data = cfg.get("data") or {}
db = data.get("db") or "data/2nd_real_run.db"
target = data.get("target") or "data/2nd_real_run_ela_full.json"
db_p = Path(db)
tgt_p = Path(target)
if not db_p.is_absolute():
    db_p = repo / db_p
if not tgt_p.is_absolute():
    tgt_p = repo / tgt_p
print(db_p)
print(tgt_p)
PY
    )"
    if [[ -z "${db}" ]]; then
      db="$(printf '%s\n' "${resolved}" | sed -n '1p')"
    fi
    if [[ -z "${target}" ]]; then
      target="$(printf '%s\n' "${resolved}" | sed -n '2p')"
    fi
  fi
  db="${db:-${REPO}/data/2nd_real_run.db}"
  target="${target:-${REPO}/data/2nd_real_run_ela_full.json}"

  if [[ ! -f "${db}" ]]; then
    echo "FATAL: campaign data missing: ${db}" >&2
    echo "  data/ is gitignored — rsync from your workstation." >&2
    return 1
  fi

  case "${db}" in
    *.csv|*.CSV)
      python - "${db}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
size_mb = path.stat().st_size / (1024 * 1024)
with open(path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    if reader.fieldnames is None:
        raise SystemExit(f"FATAL: empty CSV: {path}")
    names = set(reader.fieldnames)
    need = {"FAPbI3", "MAPbI3", "MAPbBr3", "Objective"}
    missing = sorted(need - names)
    if missing:
        raise SystemExit(f"FATAL: {path} missing columns {missing}")
    n = 0
    for row in reader:
        try:
            vals = [float(row[c]) for c in ("FAPbI3", "MAPbI3", "MAPbBr3", "Objective")]
        except (TypeError, ValueError):
            continue
        if any(v != v for v in vals):
            continue
        n += 1
if n < 100:
    raise SystemExit(f"FATAL: only {n} complete campaign rows in {path} (expected ~639)")
print(f"csv OK: {path.name} rows={n} size={size_mb:.1f}MB")
PY
      ;;
    *)
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
      ;;
  esac

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

_pilot_append_config_flags() {
  local -n _cmd=$1
  local cfg
  cfg="$(_pilot_default_config)"
  if [[ ! -f "${cfg}" ]]; then
    echo "FATAL: pilot config missing: ${cfg}" >&2
    return 1
  fi
  _cmd+=(--config "${cfg}")
}

_pilot_append_mode_flags() {
  local -n _cmd=$1
  # Legacy env overrides still work; they beat config file mode at runtime.
  if [[ "${PILOT_CAMPAIGN_MODE:-0}" == "1" ]]; then
    _cmd+=(--campaign-mode)
  fi
  if [[ "${PILOT_PURE_PAPER:-0}" == "1" ]]; then
    _cmd+=(--pure-paper)
  fi
  if [[ -n "${PILOT_ALPHA:-}" ]]; then
    _cmd+=(--alpha "${PILOT_ALPHA}")
  fi
  if [[ -n "${PILOT_TIER1_GAMMA:-}" ]]; then
    _cmd+=(--tier1-gamma "${PILOT_TIER1_GAMMA}")
  fi
  if [[ -n "${PILOT_BETA:-}" ]]; then
    _cmd+=(--beta "${PILOT_BETA}")
  fi
  if [[ -n "${PILOT_LINEARITY_PENALTY:-}" ]]; then
    _cmd+=(--linearity-penalty "${PILOT_LINEARITY_PENALTY}")
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
  if [[ -n "${PILOT_LANDSCAPE_EVERY:-}" ]]; then
    _cmd+=(--landscape-every "${PILOT_LANDSCAPE_EVERY}")
  fi
}

_pilot_append_worker_flags() {
  local -n _cmd=$1
  _cmd+=(--eval-workers "${PILOT_EVAL_WORKERS:-8}")
}

_pilot_append_seed_flags() {
  local -n _cmd=$1
  if [[ -v PILOT_SEED ]]; then
    _cmd+=(--seed "${PILOT_SEED}")
  fi
}

_pilot_append_population_flags() {
  local -n _cmd=$1
  if [[ -n "${PILOT_POPULATION:-}" ]]; then
    _cmd+=(--population "${PILOT_POPULATION}")
  fi
  if [[ -n "${PILOT_GENERATIONS:-}" ]]; then
    _cmd+=(--generations "${PILOT_GENERATIONS}")
  fi
}

_pilot_log_config_summary() {
  local cfg
  cfg="$(_pilot_default_config)"
  echo "pilot_config: ${cfg}"
  if [[ -f "${cfg}" ]]; then
    python - "${cfg}" <<'PY'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
print(f"  name: {cfg.get('name', '?')}")
print(f"  mode: {cfg.get('mode', '?')}")
fit = cfg.get("fitness", {})
ga = cfg.get("ga", {})
features = fit.get("fitness_features")
if features:
    print(f"  fitness_features: {features}")
weights = fit.get("tier1_weights")
if weights:
    print(f"  tier1_weights: {weights}")
elif features:
    print("  tier1_weights: uniform (1.0)")
print(
    "  fitness: "
    f"α={fit.get('alpha_subspace')} "
    f"β={fit.get('beta_complexity')} "
    f"γ={fit.get('tier1_gamma')} "
    f"linearity={fit.get('linearity_penalty_gamma')} "
    f"calibrate={fit.get('linear_calibration')} "
    f"rmse_frac={fit.get('subspace_rmse_frac')}"
)
print(f"  ga: pop={ga.get('population')} gens={ga.get('generations')}")
PY
  fi
}
