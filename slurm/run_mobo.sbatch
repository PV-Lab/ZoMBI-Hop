#!/bin/bash
#SBATCH --job-name=zombi-mobo
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mobo-%j.out
#SBATCH --error=logs/mobo-%j.err

set -euo pipefail
# Single headless MOBO run (optimize/run_mobo.py).
#
# Submit from repo root (recommended — sets partition from MOBO_DEVICE):
#   bash scripts/submit_mobo.sh
#   MOBO_DEVICE=cuda bash scripts/submit_mobo.sh
#   MOBO_DEVICE=cuda MOBO_CONFIG=optimize/mobo_batch_configs/synthetic_3d_messy.json bash scripts/submit_mobo.sh
#   MOBO_DEVICE=cuda MOBO_CONFIG=optimize/mobo_batch_configs/ackley_10d_layout1.json bash scripts/submit_mobo.sh
#
# Direct sbatch (CPU defaults in #SBATCH headers above):
#   cd ~/ZoMBI-Hop && sbatch slurm/run_mobo.sbatch
#
# Environment:
#   MOBO_DEVICE=cpu|cuda   (default cpu)
#   MOBO_CONFIG=...        (default campaign1a_objective_min.json)
#   MOBO_MAX_TRIALS=N
#   MOBO_RUN_DIR=...       (optional explicit output dir, relative to repo)
#
# Quick runtime probe:
#   bash scripts/probe_cluster.sh cpu
#   bash scripts/probe_cluster.sh gpu-campaign

REPO="${SLURM_SUBMIT_DIR}"
DEVICE="${MOBO_DEVICE:-cpu}"
CONFIG="${MOBO_CONFIG:-optimize/mobo_batch_configs/campaign1a_objective_min.json}"
MAX_TRIALS="${MOBO_MAX_TRIALS:-}"
RUN_DIR="${MOBO_RUN_DIR:-}"

cd "${REPO}"
mkdir -p logs optimize/runs

# shellcheck source=/dev/null
source "${REPO}/slurm/mobo_job_common.sh"
_mobo_setup_env

EXTRA=()
if [[ -n "${MAX_TRIALS}" ]]; then
  EXTRA+=(--max-trials "${MAX_TRIALS}")
fi
if [[ -n "${RUN_DIR}" ]]; then
  EXTRA+=(--run-dir "${REPO}/${RUN_DIR}")
fi

python optimize/run_mobo.py --batch --no-show \
  --config "${REPO}/${CONFIG}" \
  --device "${DEVICE}" \
  "${EXTRA[@]}"
