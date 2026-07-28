#!/bin/bash
# =============================================================================
# Re-run trial_112 hparams on campaign1a RF (same landscape/needles as
# mobo_05_06_15_32). Fixed hparams — not MOBO search.
#
# Objectives reported: dist_to_needles, dup_fraction, runtime_s
# (same ZoMBI metrics as the original trial_112 evaluation).
#
# Submit from repo root on ORCD:
#   sbatch slurm/run_eval_trial112_campaign1a.sbatch
#
# Optional overrides:
#   NUM_RUNS=3 TIME_LIMIT_MIN=24 OUT_DIR=optimize/runs/rerun_t112_c1a \
#     sbatch --export=ALL,NUM_RUNS,TIME_LIMIT_MIN,OUT_DIR \
#     slurm/run_eval_trial112_campaign1a.sbatch
# =============================================================================

#SBATCH --job-name=eval_t112_c1a
#SBATCH --output=optimize/scripts/logs/eval_t112_campaign1a_%j.out
#SBATCH --error=optimize/scripts/logs/eval_t112_campaign1a_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO}"
mkdir -p optimize/runs optimize/scripts/logs

# shellcheck source=/dev/null
source "${REPO}/slurm/mobo_job_common.sh"
_mobo_setup_env

# Prefer live run dir; fall back to archived copy on ORCD.
HPARAMS="${REPO}/optimize/runs/mobo_05_06_15_32/trial_112"
RUNS_PATH="${REPO}/optimize/runs/mobo_05_06_15_32"
if [[ ! -d "${HPARAMS}" ]]; then
  HPARAMS="${REPO}/optimize/runs/archived_runs/mobo_3d_05_06_15_32/trial_112"
  RUNS_PATH="${REPO}/optimize/runs/archived_runs/mobo_3d_05_06_15_32"
fi
if [[ ! -d "${HPARAMS}" ]]; then
  echo "ERROR: trial_112 not found under optimize/runs/ or archived_runs/" >&2
  exit 1
fi
if [[ ! -f "${RUNS_PATH}/run_config.json" ]]; then
  echo "ERROR: missing run_config.json at ${RUNS_PATH}" >&2
  exit 1
fi

NUM_RUNS="${NUM_RUNS:-1}"
TIME_LIMIT_MIN="${TIME_LIMIT_MIN:-10}"
OUT_DIR="${OUT_DIR:-optimize/runs/rerun_trial112_campaign1a_job${SLURM_JOB_ID:-local}}"

EVAL_CMD=(
  python optimize/evaluate.py
  --hparams "${HPARAMS}"
  --dataset RF
  --runs-path "${RUNS_PATH}"
  --num-runs "${NUM_RUNS}"
  --time-limit-min "${TIME_LIMIT_MIN}"
  --device cuda
  --out-dir "${OUT_DIR}"
)

echo "=== Re-run trial_112 hparams on campaign1a RF ==="
echo "  hparams:   ${HPARAMS}"
echo "  runs-path: ${RUNS_PATH}"
echo "  out-dir:   ${OUT_DIR}"
echo "  num-runs:  ${NUM_RUNS}"
echo "  time-limit-min: ${TIME_LIMIT_MIN}"
printf '  %q' "${EVAL_CMD[@]}"
echo

"${EVAL_CMD[@]}"
echo "finished: $(date -Is)"
echo "summary:  ${OUT_DIR}/rerun_summary.json"
