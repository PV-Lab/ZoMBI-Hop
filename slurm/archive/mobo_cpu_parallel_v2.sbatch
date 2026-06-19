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

# Parallelize ZoMBI sub-runs inside each MOBO eval (not MOBO configs).
MOBO_WORKERS=1
BENCHMARK_WORKERS="${SLURM_CPUS_PER_TASK:-16}"
THREADS_PER_WORKER=$(( BENCHMARK_WORKERS > 0 ? SLURM_CPUS_PER_TASK / BENCHMARK_WORKERS : 1 ))
THREADS_PER_WORKER=$(( THREADS_PER_WORKER < 1 ? 1 : THREADS_PER_WORKER ))

export OMP_NUM_THREADS="${THREADS_PER_WORKER}"
export MKL_NUM_THREADS="${THREADS_PER_WORKER}"
export OPENBLAS_NUM_THREADS="${THREADS_PER_WORKER}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

python scripts/run_zombi_test_v2_parallel.py --mobo \
  --regions "${REPO}/scripts/max_min_regions.json" \
  --mobo-init 4 \
  --mobo-iters 1 \
  --batch 2 \
  --workers "${MOBO_WORKERS}" \
  --benchmark-workers "${BENCHMARK_WORKERS}" \
  --threads-per-worker "${THREADS_PER_WORKER}" \
  --device cpu \
  --results-json "${REPO}/runs/hyperparam_cpu_results_v2_parallel.json"
