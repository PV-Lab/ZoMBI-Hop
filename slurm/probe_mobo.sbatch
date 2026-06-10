#!/bin/bash
#SBATCH --job-name=zombi-probe
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/probe-%j.out
#SBATCH --error=logs/probe-%j.err

set -euo pipefail
# Short MOBO probe to measure per-trial runtime and log cluster allocation.
#
# Submit from repo root:
#   sbatch slurm/probe_mobo.sbatch                              # CPU, campaign, 1 trial
#   bash scripts/probe_cluster.sh gpu-campaign                  # GPU, 3D campaign (recommended)
#   bash scripts/probe_cluster.sh gpu-ackley                    # GPU, Ackley 10D
#   MOBO_MAX_TRIALS=2 bash scripts/probe_cluster.sh gpu-campaign
#
# Or use the helper:
#   bash scripts/probe_cluster.sh info
#   bash scripts/probe_cluster.sh cpu
#   bash scripts/probe_cluster.sh gpu-campaign
#   bash scripts/probe_cluster.sh gpu-ackley
#   bash scripts/probe_cluster.sh both-campaign               # CPU + GPU campaign probes

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROBE_TARGET="${PROBE_TARGET:-cpu}"
PROBE_LANDSCAPE="${PROBE_LANDSCAPE:-campaign}"
MAX_TRIALS="${MOBO_MAX_TRIALS:-1}"
REPORT="${REPO}/logs/probe-${SLURM_JOB_ID:-local}.txt"

cd "${REPO}"
mkdir -p logs optimize/runs

if [[ "${PROBE_TARGET}" == "gpu" ]]; then
  DEVICE="cuda"
  CPUS="${SLURM_CPUS_PER_TASK:-8}"
  if [[ "${PROBE_LANDSCAPE}" == "ackley" ]]; then
    CONFIG="${MOBO_CONFIG:-optimize/mobo_batch_configs/ackley_10d_layout1.json}"
  elif [[ "${PROBE_LANDSCAPE}" == "synthetic" ]]; then
    CONFIG="${MOBO_CONFIG:-optimize/mobo_batch_configs/probe_synthetic_messy_gpu.json}"
  else
    CONFIG="${MOBO_CONFIG:-optimize/mobo_batch_configs/probe_campaign_gpu.json}"
  fi
else
  CONFIG="${MOBO_CONFIG:-optimize/mobo_batch_configs/probe_campaign_gpu.json}"
  DEVICE="cpu"
  CPUS="${SLURM_CPUS_PER_TASK:-8}"
fi

RUN_DIR="${MOBO_RUN_DIR:-optimize/runs/probe_${SLURM_JOB_ID:-local}_${PROBE_TARGET}_${PROBE_LANDSCAPE}}"

{
  echo "=== ZoMBI-Hop MOBO cluster probe ==="
  echo "started:    $(date -Is)"
  echo "job_id:     ${SLURM_JOB_ID:-local}"
  echo "target:     ${PROBE_TARGET} (${PROBE_LANDSCAPE})"
  echo "partition:  ${SLURM_JOB_PARTITION:-unknown}"
  echo "config:     ${CONFIG}"
  echo "max_trials: ${MAX_TRIALS}"
  echo "run_dir:    ${RUN_DIR}"
  echo "device:     ${DEVICE}"
  echo ""
  echo "--- Slurm allocation ---"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job "${SLURM_JOB_ID}" 2>/dev/null | egrep "JobId|JobName|Partition|NumNodes|NumTasks|CPUs|TRES|TimeLimit|RunTime|Command|WorkDir" || true
  fi
  echo ""
  echo "--- Node / memory ---"
  echo "hostname:   $(hostname)"
  echo "cpus/task:  ${CPUS}"
  free -h 2>/dev/null || true
  echo ""
  if [[ "${DEVICE}" == "cuda" ]]; then
    echo "--- GPU ---"
    nvidia-smi -L 2>/dev/null || echo "(nvidia-smi unavailable)"
    nvidia-smi 2>/dev/null || true
    echo ""
  fi
} | tee "${REPORT}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate zombi-hop-linebo

export OMP_NUM_THREADS="${CPUS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

WALL_START=$(date +%s)

python optimize/run_mobo.py --batch --no-show \
  --config "${REPO}/${CONFIG}" \
  --device "${DEVICE}" \
  --max-trials "${MAX_TRIALS}" \
  --run-dir "${REPO}/${RUN_DIR}"

WALL_END=$(date +%s)
WALL_S=$(( WALL_END - WALL_START ))

{
  echo ""
  echo "--- MOBO timing summary ---"
  python "${REPO}/scripts/summarize_mobo_runs.py" "${REPO}/${RUN_DIR}" || true
  echo ""
  echo "wall_clock_s: ${WALL_S}  ($(( WALL_S / 60 )) min)"
  echo "finished:     $(date -Is)"
  echo ""
  echo "After the job ends, check Slurm accounting:"
  echo "  sacct -j ${SLURM_JOB_ID:-JOBID} --format=JobID,Partition,Elapsed,Timelimit,MaxRSS,State,ExitCode -P"
} | tee -a "${REPORT}"

echo "Probe report: ${REPORT}"
