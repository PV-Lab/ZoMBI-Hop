#!/bin/bash
#SBATCH --job-name=zombi-mobo-10d
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/mobo-10d-%j.out
#SBATCH --error=logs/mobo-10d-%j.err

set -euo pipefail
# Unified 10D Multi-Ackley MOBO via optimize/run_mobo.py (same run layout as campaign MOBO).
#
# Submit from repo root:
#   cd ~/ZoMBI-Hop && sbatch slurm/mobo_10d.sbatch
#
# Quick probe (2 trials):
#   MOBO_MAX_TRIALS=2 sbatch slurm/mobo_10d.sbatch
#   bash scripts/probe_cluster.sh gpu
#
# Override config / trials / output dir:
#   MOBO_CONFIG=optimize/mobo_batch_configs/ackley_10d_layout1.json \
#   MOBO_MAX_TRIALS=2 \
#   MOBO_RUN_DIR=optimize/runs/mobo_ackley_10d_test sbatch slurm/mobo_10d.sbatch

REPO="${SLURM_SUBMIT_DIR}"
CONFIG="${MOBO_CONFIG:-optimize/mobo_batch_configs/ackley_10d_layout1.json}"
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
