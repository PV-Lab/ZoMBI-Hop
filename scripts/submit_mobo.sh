#!/usr/bin/env bash
# Submit a single MOBO Slurm job (optimize/run_mobo.py via slurm/run_mobo.sbatch).
#
# Usage (from repo root on MIT Engaging):
#   bash scripts/submit_mobo.sh
#   MOBO_DEVICE=cuda bash scripts/submit_mobo.sh
#   MOBO_DEVICE=cuda MOBO_CONFIG=optimize/mobo_batch_configs/synthetic_3d_messy.json bash scripts/submit_mobo.sh
#   MOBO_DEVICE=cuda MOBO_CONFIG=optimize/mobo_batch_configs/ackley_10d_realistic.json bash scripts/submit_mobo.sh
#   MOBO_MAX_TRIALS=1 MOBO_DEVICE=cuda bash scripts/submit_mobo.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SBATCH="${REPO}/slurm/run_mobo.sbatch"
DEVICE="${MOBO_DEVICE:-cpu}"

cd "${REPO}"
mkdir -p logs

extra=()
case "${DEVICE}" in
  cuda)
    extra+=(
      --job-name=zombi-mobo-gpu
      --partition=mit_normal_gpu
      --gres=gpu:1
      --cpus-per-task=8
      --mem=64G
      --time=6:00:00
    )
    ;;
  cpu)
    extra+=(
      --job-name=zombi-mobo
      --partition=mit_normal
      --cpus-per-task=16
      --mem=32G
      --time=12:00:00
    )
    ;;
  *)
    echo "MOBO_DEVICE must be cpu or cuda (got ${DEVICE})" >&2
    exit 1
    ;;
esac

echo "Submitting MOBO job: device=${DEVICE} config=${MOBO_CONFIG:-<default>}"

MOBO_DEVICE="${DEVICE}" sbatch \
  "${extra[@]}" \
  --export=ALL,MOBO_DEVICE="${DEVICE}" \
  "${SBATCH}"
