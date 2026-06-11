#!/bin/bash
#SBATCH --job-name=zombi-mobo
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mobo-%j.out
#SBATCH --error=logs/mobo-%j.err

set -euo pipefail
# Single headless MOBO run (optimize/run_mobo.py).
#
# Submit from repo root:
#   cd ~/ZoMBI-Hop && sbatch slurm/run_mobo.sbatch
#
# Override config / trials:
#   MOBO_CONFIG=optimize/mobo_batch_configs/campaign1a_objective_min.json \
#   MOBO_MAX_TRIALS=30 sbatch slurm/run_mobo.sbatch

REPO="${SLURM_SUBMIT_DIR}"
CONFIG="${MOBO_CONFIG:-optimize/mobo_batch_configs/campaign1a_objective_min.json}"
MAX_TRIALS="${MOBO_MAX_TRIALS:-}"

cd "${REPO}"
mkdir -p logs optimize/runs

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate zombi-hop-linebo

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

EXTRA=()
if [[ -n "${MAX_TRIALS}" ]]; then
  EXTRA+=(--max-trials "${MAX_TRIALS}")
fi

python optimize/run_mobo.py --batch --no-show \
  --config "${REPO}/${CONFIG}" \
  --device cpu \
  "${EXTRA[@]}"
