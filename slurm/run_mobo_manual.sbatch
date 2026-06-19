#!/bin/bash
# =============================================================================
# slurm/run_mobo_manual.sbatch — one-off MOBO job (edit flags, then sbatch)
# =============================================================================
#
# Submit from repo root on ORCD / Engaging:
#
#   cd ~/ZoMBI-Hop
#   sbatch slurm/run_mobo_manual.sbatch
#
# This file is a template: edit the #SBATCH headers and MOBO_CMD block below,
# exactly like typing ``python optimize/run_mobo.py ...`` locally. Always keep
# ``--no-show`` on the cluster (``mobo_job_common.sh`` sets MPLBACKEND=Agg).
#
# Flag cheat sheet (pick ONE run mode — do not combine incompatible flags):
#
#   Fresh RF / synthetic / Multi-Ackley from JSON (MUST include --batch):
#     --batch --config optimize/mobo_batch_configs/<name>.json
#     (Do NOT add --dataset RF — that opens the interactive picker instead.)
#
#   Fresh analytic Ackley (ackley3d / ackley4d / ackley10d):
#     --dataset ackley4d
#     --dataset ackley10d --ackley-variant realistic
#
#   Continue after crash / partial run:
#     --resume
#     --resume-from optimize/runs/mobo_DD_MM_HH_MM_SS_PID
#
#   Reuse saved settings, no prior trial data:
#     --copy-config optimize/runs/mobo_05_06_15_32
#
#   Re-evaluate specific trial hparams as init (adds to Sobol init):
#     --start-from-best optimize/runs/mobo_05_06_15_32/trial_1
#
#   Common extras (any mode):
#     --device cuda|cpu          (default: cuda if available else cpu)
#     --time-limit-hours 0.4     (or --time-limit 0.4)
#     --max-trials 50            (default: unbounded until job time limit)
#     --run-dir optimize/runs/my_explicit_name
#     --runs-dir optimize/runs   (override crawl root for --resume)
#
#   Do NOT combine:
#     --config  without  --batch          (interactive picker; hangs on HPC)
#     --dataset RF  with  --config         (same problem — omit both; use --batch)
#     --batch  with  --resume | --resume-from | --copy-config | --resume-scratch
#     more than one of --resume / --resume-from / --copy-config / --resume-scratch
#
# Slurm resources — edit for CPU vs GPU:
#
#   CPU (RF surrogate, long wall clock):
#     #SBATCH --partition=mit_normal
#     #SBATCH --cpus-per-task=16
#     #SBATCH --mem=32G
#     #SBATCH --time=12:00:00
#     MOBO: --device cpu
#
#   GPU (cuda ZoMBI trials):
#     #SBATCH --partition=mit_normal_gpu
#     #SBATCH --gres=gpu:1
#     #SBATCH --cpus-per-task=16
#     #SBATCH --mem=64G
#     #SBATCH --time=6:00:00
#     MOBO: --device cuda
#
# =============================================================================

#SBATCH --job-name=zombi-mobo
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/mobo-%j.out
#SBATCH --error=logs/mobo-%j.err

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO}"
mkdir -p logs optimize/runs

# shellcheck source=/dev/null
source "${REPO}/slurm/mobo_job_common.sh"
_mobo_setup_env

# =============================================================================
# EDIT THIS BLOCK — one command, same flags as on your laptop
# =============================================================================
#
# Active example: campaign1a RF repeat (batch JSON).

# MOBO_CMD=(
#   python optimize/run_mobo.py
#   --no-show
#   --device cuda
#   --batch
#   --config optimize/mobo_batch_configs/campaign1a_mobo_05_06_15_32_repeat.json
#   --time-limit-hours 0.4
# )

# --- Resume (crawl all optimize/runs/mobo_*, reuse latest run_config) --------
# MOBO_CMD=(
#   python optimize/run_mobo.py
#   --no-show
#   --device cuda
#   --resume
# )

# --- Resume from one run directory only --------------------------------------
MOBO_CMD=(
  python optimize/run_mobo.py
  --no-show
  --device cuda
  --resume-from optimize/runs/mobo_19_06_13_01_55_545039
)

# --- Copy config from a past run (fresh Sobol + BO, no prior trials) ---------
# MOBO_CMD=(
#   python optimize/run_mobo.py
#   --no-show
#   --device cuda
#   --copy-config optimize/runs/mobo_05_06_15_32
#   --time-limit-hours 0.4
# )

# --- Ackley benchmark (no batch JSON; uses --dataset) ------------------------
# MOBO_CMD=(
#   python optimize/run_mobo.py
#   --no-show
#   --device cuda
#   --dataset ackley4d
#   --ackley-variant realistic
#   --time-limit 0.5
# )

# --- Seed from best trial hparams (re-evaluated; Sobol still runs) -------------
# MOBO_CMD=(
#   python optimize/run_mobo.py
#   --no-show
#   --device cuda
#   --batch
#   --config optimize/mobo_batch_configs/campaign1a_mobo_05_06_15_32_repeat.json
#   --start-from-best optimize/runs/mobo_05_06_15_32/trial_315
# )

# =============================================================================

# MOBO_CMD=(
#   python optimize/run_mobo.py
#   --no-show
#   --device cuda
#   --batch
#   --config optimize/mobo_batch_configs/campaign1a_mobo_05_06_15_32_repeat.json
#   --time-limit-hours 0.4
# )

--resume-from
echo "=== MOBO command ==="
printf '  %q' "${MOBO_CMD[@]}"
echo
echo "=== cwd: ${REPO} ==="
echo

"${MOBO_CMD[@]}"
