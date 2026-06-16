#!/bin/bash
#SBATCH --job-name=zombi-mobo-batch
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/mobo-array-%A_%a.out
#SBATCH --error=logs/mobo-array-%A_%a.err

set -euo pipefail
# Slurm array: one MOBO run per manifest entry.
#
# Submit from repo root (array bounds + partition set by scripts/submit_mobo_batch.sh):
#   bash scripts/submit_mobo_batch.sh
#   MOBO_DEVICE=cuda MOBO_MANIFEST=optimize/mobo_batch_manifest_synthetic_3d.json bash scripts/submit_mobo_batch.sh
#
# Environment:
#   MOBO_MANIFEST=...      (default optimize/mobo_batch_manifest.json)
#   MOBO_DEVICE=cpu|cuda   (default cpu; submit script sets partition/gres)
#   MOBO_MAX_TRIALS=N      (optional CLI override per task)

REPO="${SLURM_SUBMIT_DIR}"
MANIFEST="${MOBO_MANIFEST:-optimize/mobo_batch_manifest.json}"
DEVICE="${MOBO_DEVICE:-cpu}"
MAX_TRIALS="${MOBO_MAX_TRIALS:-}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

cd "${REPO}"
mkdir -p logs optimize/runs

CONFIG=$(python - <<'PY'
import json, os, sys
manifest = os.environ["MOBO_MANIFEST"]
task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
with open(manifest) as f:
    data = json.load(f)
configs = data["configs"]
if task_id < 0 or task_id >= len(configs):
    sys.exit(f"array task {task_id} out of range for {len(configs)} configs")
print(configs[task_id])
PY
)

CONFIG_PATH="${REPO}/${CONFIG}"
BATCH_NAME=$(python - <<PY
import json
with open("${CONFIG_PATH}") as f:
    cfg = json.load(f)
print(cfg.get("name", "batch"))
PY
)
RUN_DIR="${REPO}/optimize/runs/mobo_batch_${BATCH_NAME}_job${SLURM_ARRAY_JOB_ID}_task${TASK_ID}"

# shellcheck source=/dev/null
source "${REPO}/slurm/mobo_job_common.sh"
_mobo_setup_env

EXTRA=(--run-dir "${RUN_DIR}")
if [[ -n "${MAX_TRIALS}" ]]; then
  EXTRA+=(--max-trials "${MAX_TRIALS}")
fi

echo "[array] task=${TASK_ID}  device=${DEVICE}  config=${CONFIG}  run_dir=${RUN_DIR}"

python optimize/run_mobo.py --batch --no-show \
  --config "${CONFIG_PATH}" \
  --device "${DEVICE}" \
  "${EXTRA[@]}"
