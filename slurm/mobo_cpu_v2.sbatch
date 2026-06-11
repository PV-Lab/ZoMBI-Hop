#!/bin/bash
#SBATCH --job-name=zombi-mobo-cpu
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mobo-%j.out
#SBATCH --error=logs/mobo-%j.err

set -euo pipefail
# Submit from repo root:  cd ~/ZoMBI-Hop && sbatch slurm/mobo_cpu_v2.sbatch
REPO="${SLURM_SUBMIT_DIR}"
SCRATCH="${HOME}/orcd/scratch/zombi-mobo-${SLURM_JOB_ID}"
cd "${REPO}"
mkdir -p logs runs "${SCRATCH}"

# Warm RF cache / data on node-local scratch (optional; results still land in repo)
rsync -a "${REPO}/test_rfs/" "${SCRATCH}/test_rfs/" 2>/dev/null || true
rsync -a "${REPO}/data/" "${SCRATCH}/data/" 2>/dev/null || true

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate zombi-hop-linebo

WORKERS=4
export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / WORKERS ))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

python scripts/run_zombi_test_v2.py --mobo \
  --regions "${REPO}/scripts/max_min_regions.json" \
  --mobo-init 8 \
  --mobo-iters 10 \
  --batch 4 \
  --workers "${WORKERS}" \
  --device cpu \
  --results-json "${REPO}/runs/hyperparam_cpu_results_v2.json"
