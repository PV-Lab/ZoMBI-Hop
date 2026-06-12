#!/usr/bin/env bash
# Probe MOBO runtime and print ORCD cluster limits.
#
# Usage (from repo root on MIT ORCD):
#   bash scripts/probe_cluster.sh info          # print partition/account limits only
#   bash scripts/probe_cluster.sh cpu           # submit 1-trial CPU probe
#   bash scripts/probe_cluster.sh gpu           # submit 1-trial GPU probe
#   bash scripts/probe_cluster.sh both          # submit CPU + GPU probes
#   bash scripts/probe_cluster.sh summarize     # summarize all runs under optimize/runs
#
# Environment overrides (passed to sbatch):
#   MOBO_MAX_TRIALS=2   — trials per probe job (default 1)
#   MOBO_CONFIG=...     — batch JSON config path

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
  echo "=== Current sbatch defaults in this repo ==="
  echo "  slurm/run_mobo.sbatch     mit_normal,     16 CPU, 32G,  12h wall, device=cpu"
  echo "  slurm/mobo_10d.sbatch     mit_normal_gpu,  8 CPU, 64G,   6h wall, 1 GPU"
  echo "  slurm/probe_mobo.sbatch   2h wall, MOBO_MAX_TRIALS=${MOBO_MAX_TRIALS:-1}"
  echo ""
  echo "Campaign config: 20 trials × 0.4 h/trial ≈ 8 h ZoMBI (12 h sbatch limit)"
  echo "Ackley config:   28 trials, max_activations/trial — measure with gpu probe first"
}

submit_probe() {
  local target="$1"
  local extra=()

  if [[ "${target}" == "gpu" ]]; then
    extra+=(--partition=mit_normal_gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G)
  else
    extra+=(--partition=mit_normal --cpus-per-task=16 --mem=32G)
  fi

  echo "Submitting ${target} probe (MOBO_MAX_TRIALS=${MOBO_MAX_TRIALS:-1}) …"
  PROBE_TARGET="${target}" \
  MOBO_MAX_TRIALS="${MOBO_MAX_TRIALS:-1}" \
  MOBO_CONFIG="${MOBO_CONFIG:-}" \
  sbatch "${extra[@]}" --export=ALL,PROBE_TARGET="${target}" "${SBATCH}"
}

case "${MODE}" in
  info)
    print_cluster_info
    ;;
  cpu)
    print_cluster_info
    echo ""
    submit_probe cpu
    ;;
  gpu)
    print_cluster_info
    echo ""
    submit_probe gpu
    ;;
  both)
    print_cluster_info
    echo ""
    submit_probe cpu
    submit_probe gpu
    ;;
  summarize)
    python "${REPO}/scripts/summarize_mobo_runs.py" --latest
    echo ""
    python "${REPO}/scripts/summarize_mobo_runs.py"
    ;;
  *)
    echo "Usage: bash scripts/probe_cluster.sh [info|cpu|gpu|both|summarize]" >&2
    exit 1
    ;;
esac
