#!/bin/bash
#SBATCH --job-name=zombi-mobo-gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/mobo-gpu-%j.out
#SBATCH --error=logs/mobo-gpu-%j.err

set -euo pipefail
# 3D campaign RF MOBO on GPU (optimize/run_mobo.py).
#
# Submit from repo root:
#   cd ~/ZoMBI-Hop && sbatch slurm/run_mobo_gpu.sbatch
#
# Quick probe (1 trial):
#   MOBO_MAX_TRIALS=1 sbatch slurm/run_mobo_gpu.sbatch
#   bash scripts/probe_cluster.sh gpu-campaign
#
# Override config / trials:
#   MOBO_CONFIG=optimize/mobo_batch_configs/campaign1a_objective_min.json \
#   MOBO_MAX_TRIALS=20 sbatch slurm/run_mobo_gpu.sbatch

REPO="${SLURM_SUBMIT_DIR}"
CONFIG="${MOBO_CONFIG:-optimize/mobo_batch_configs/campaign1a_objective_min.json}"
MAX_TRIALS="${MOBO_MAX_TRIALS:-}"
RUN_DIR="${MOBO_RUN_DIR:-}"

cd "${REPO}"
mkdir -p logs optimize/runs

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate zombi-hop-linebo

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

EXTRA=()
if [[ -n "${MAX_TRIALS}" ]]; then
  EXTRA+=(--max-trials "${MAX_TRIALS}")
fi
if [[ -n "${RUN_DIR}" ]]; then
  EXTRA+=(--run-dir "${REPO}/${RUN_DIR}")
fi

python optimize/run_mobo.py --batch --no-show \
  --config "${REPO}/${CONFIG}" \
  --device cuda \
  "${EXTRA[@]}"
