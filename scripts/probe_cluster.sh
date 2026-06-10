#!/usr/bin/env bash
# Probe MOBO runtime and print ORCD cluster limits.
#
# Usage (from repo root on MIT ORCD):
#   bash scripts/probe_cluster.sh info              # partition/account limits
#   bash scripts/probe_cluster.sh cpu               # 1-trial CPU, 3D campaign
#   bash scripts/probe_cluster.sh gpu-campaign      # 1-trial GPU, 3D campaign RF
#   bash scripts/probe_cluster.sh gpu-synthetic       # 1-trial GPU, 3D messy synthetic RF
#   bash scripts/probe_cluster.sh gpu-ackley        # 1-trial GPU, Ackley 10D
#   bash scripts/probe_cluster.sh both-campaign     # CPU + GPU campaign probes
#   bash scripts/probe_cluster.sh summarize         # timing from optimize/runs
#
# Environment overrides (passed to sbatch):
#   MOBO_MAX_TRIALS=2        — trials per probe (default 1; overrides config)
#   MOBO_CONFIG=...          — batch JSON (default depends on probe type)
#   MOBO_RUN_DIR=...         — explicit output run directory

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH="${REPO}/slurm/probe_mobo.sbatch"
MODE="${1:-info}"

cd "${REPO}"
mkdir -p logs optimize/runs

print_cluster_info() {
  echo "=== Partition overview ==="
  if command -v sinfo >/dev/null 2>&1; then
    sinfo -o "%P %a %l %D %c %m %G %F" 2>/dev/null | head -1
    sinfo -o "%P %a %l %D %c %m %G %F" 2>/dev/null | grep -E "mit_normal" || true
  else
    echo "(sinfo not available — run on ORCD login node)"
  fi

  echo ""
  echo "=== Your jobs in queue ==="
  if command -v squeue >/dev/null 2>&1; then
    squeue -u "${USER}" -o "%.10i %.9P %.18j %.8u %.2t %.10M %.6D %R" 2>/dev/null || true
  fi

  echo ""
  echo "=== Account limits (if sacctmgr available) ==="
  if command -v sacctmgr >/dev/null 2>&1; then
    sacctmgr show assoc user="${USER}" format=Account,Partition,QOS,MaxJobs,MaxSubmit,MaxWall,GrpTRES -p 2>/dev/null \
      | head -20 || echo "(sacctmgr query failed)"
  else
    echo "(sacctmgr not available)"
  fi

  echo ""
  echo "=== MOBO sbatch scripts ==="
  echo "  slurm/run_mobo.sbatch       CPU campaign (mit_normal, 16 CPU, 12h)"
  echo "  slurm/run_mobo_gpu.sbatch   GPU campaign (mit_normal_gpu, 1 GPU, 12h)"
  echo "  slurm/mobo_10d.sbatch       GPU Ackley 10D (mit_normal_gpu, 1 GPU, 6h)"
  echo "  slurm/probe_mobo.sbatch     short probe, MOBO_MAX_TRIALS=${MOBO_MAX_TRIALS:-1}"
  echo ""
  echo "Probe configs:"
  echo "  probe_campaign_gpu.json   — 1 trial, 0.4 h/trial, 3D RF campaign"
  echo "  probe_synthetic_messy_gpu.json — 1 trial, 3D messy synthetic RF"
  echo "  ackley_10d_layout1.json   — Ackley 10D synthetic benchmark"
  echo ""
  echo "Synthetic data:"
  echo "  bash scripts/generate_synthetic_mobo_data.sh"
  echo "  MOBO_MANIFEST=optimize/mobo_batch_manifest_synthetic_3d.json bash scripts/submit_mobo_batch.sh"
}

submit_probe() {
  local target="$1"
  local landscape="${2:-campaign}"
  local extra=()

  if [[ "${target}" == "gpu" ]]; then
    extra+=(--partition=mit_normal_gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G)
  else
    extra+=(--partition=mit_normal --cpus-per-task=8 --mem=32G)
  fi

  echo "Submitting ${target}/${landscape} probe (MOBO_MAX_TRIALS=${MOBO_MAX_TRIALS:-1}) …"
  PROBE_TARGET="${target}" \
  PROBE_LANDSCAPE="${landscape}" \
  MOBO_MAX_TRIALS="${MOBO_MAX_TRIALS:-1}" \
  MOBO_CONFIG="${MOBO_CONFIG:-}" \
  sbatch "${extra[@]}" --export=ALL,PROBE_TARGET="${target}",PROBE_LANDSCAPE="${landscape}" "${SBATCH}"
}

case "${MODE}" in
  info)
    print_cluster_info
    ;;
  cpu)
    print_cluster_info
    echo ""
    submit_probe cpu campaign
    ;;
  gpu|gpu-campaign)
    print_cluster_info
    echo ""
    submit_probe gpu campaign
    ;;
  gpu-ackley)
    print_cluster_info
    echo ""
    submit_probe gpu ackley
    ;;
  gpu-synthetic)
    print_cluster_info
    echo ""
    MOBO_CONFIG="${MOBO_CONFIG:-optimize/mobo_batch_configs/probe_synthetic_messy_gpu.json}" \
      submit_probe gpu synthetic
    ;;
  both|both-campaign)
    print_cluster_info
    echo ""
    submit_probe cpu campaign
    submit_probe gpu campaign
    ;;
  summarize)
    python "${REPO}/scripts/summarize_mobo_runs.py" --latest
    echo ""
    python "${REPO}/scripts/summarize_mobo_runs.py"
    ;;
  *)
    echo "Usage: bash scripts/probe_cluster.sh [info|cpu|gpu-campaign|gpu-synthetic|gpu-ackley|both-campaign|summarize]" >&2
    echo "       (gpu is an alias for gpu-campaign)" >&2
    exit 1
    ;;
esac
