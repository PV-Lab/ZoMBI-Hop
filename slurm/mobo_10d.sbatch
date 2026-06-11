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
# Submit from repo root:  cd ~/ZoMBI-Hop && sbatch slurm/mobo_10d.sbatch
REPO="${SLURM_SUBMIT_DIR}"
cd "${REPO}"
mkdir -p logs runs

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate zombi-hop-linebo

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

python optimize/run_mobo_10d.py \
  --batch \
  --dim 10 \
  --layout 1 \
  --n-init-trials 8 \
  --n-mobo-trials 20 \
  --device cuda \
  --no-show \
  --save-dir "${REPO}/runs/mobo_10d"
