#!/bin/bash
#SBATCH --job-name=zombi-mobo-batch
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/mobo-array-%A_%a.out
#SBATCH --error=logs/mobo-array-%A_%a.err

set -euo pipefail
# Slurm array: one MOBO run per manifest entry (optimize/mobo_batch_manifest.json).
#
# Submit from repo root (array bounds set by scripts/submit_mobo_batch.sh):
#   cd ~/ZoMBI-Hop && bash scripts/submit_mobo_batch.sh
#
# Or manually after editing --array above:
#   sbatch slurm/run_mobo_array.sbatch

REPO="${SLURM_SUBMIT_DIR}"
MANIFEST="${MOBO_MANIFEST:-optimize/mobo_batch_manifest.json}"
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

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate zombi-hop-linebo

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

echo "[array] task=${TASK_ID}  config=${CONFIG}  run_dir=${RUN_DIR}"

python optimize/run_mobo.py --batch --no-show \
  --config "${CONFIG_PATH}" \
  --device cpu \
  --run-dir "${RUN_DIR}"
