#!/bin/bash
# =============================================================================
# slurm/run_mobo_manual.sbatch — one-off MOBO job (edit flags, then sbatch)
# =============================================================================
#
# For ensemble 3d/4d with auto-requeue chains, use instead:
#   sbatch slurm/run_mobo_manual_ensemble_3d.sbatch
#   sbatch slurm/run_mobo_manual_ensemble_4d.sbatch
#
# Submit from repo root on ORCD / Engaging:
#
#   cd ~/ZoMBI-Hop
#   sbatch slurm/run_mobo_manual.sbatch
#
# Ensemble MOBO with auto-requeue chain (wall-time + clean-exit resubmit):
#   sbatch slurm/run_mobo_manual_ensemble_3d.sbatch
#   sbatch slurm/run_mobo_manual_ensemble_4d.sbatch
#
# This file is a template: edit the #SBATCH headers and the single MOBO_CMD
# block below, exactly like typing ``python optimize/run_mobo.py ...`` locally.
# Always keep ``--no-show`` on the cluster (``mobo_job_common.sh`` sets
# MPLBACKEND=Agg).
#
# Flag cheat sheet (pick ONE run mode — do not combine incompatible flags):
#
#   Fresh RF / synthetic from JSON (MUST include --batch):
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
# Slurm resources — edit #SBATCH headers for CPU vs GPU:
#
#   CPU (RF surrogate, long wall clock):
#     #SBATCH --partition=mit_normal
#     #SBATCH --cpus-per-task=16
#     #SBATCH --mem=32G
#     #SBATCH --time=12:00:00
#     MOBO: --device cpu
#
#   GPU ensemble 3d (matches optimize/scripts/ensemble_mobo_3d.sbatch):
#     #SBATCH --job-name=ensemble_mobo_3d
#     #SBATCH --output=optimize/scripts/logs/ensemble_mobo_3d_%j.out
#     #SBATCH --error=optimize/scripts/logs/ensemble_mobo_3d_%j.err
#     #SBATCH --time=12:00:00
#     #SBATCH --ntasks=1
#     #SBATCH --cpus-per-task=1
#     #SBATCH --mem=32G
#     #SBATCH --partition=sched_mit_sloan_gpu_r8
#     #SBATCH --gres=gpu:1
#     MOBO: --dataset ensemble --dim 3 --time-limit 0.2 --share-history
#           --start-from-best optimize/runs/mobo_3d_05_06_15_32/trial_112
#
#   GPU ensemble 4d (matches optimize/scripts/ensemble_mobo_4d.sbatch):
#     #SBATCH --job-name=ensemble_mobo_4d
#     #SBATCH --output=optimize/scripts/logs/ensemble_mobo_4d_%j.out
#     #SBATCH --error=optimize/scripts/logs/ensemble_mobo_4d_%j.err
#     #SBATCH --time=12:00:00
#     #SBATCH --ntasks=1
#     #SBATCH --cpus-per-task=1
#     #SBATCH --mem=64G
#     #SBATCH --partition=sched_mit_sloan_gpu_r8
#     #SBATCH --gres=gpu:1
#     MOBO: --dataset ensemble --dim 4 --time-limit 0.3 --share-history
#           --start-from-best optimize/runs/mobo_3d_05_06_15_32/trial_112
#
# How to switch modes: replace the #SBATCH headers and MOBO_CMD block with one
# of the examples below (only one active MOBO_CMD= assignment in this file).
#
# =============================================================================
# ACTIVE #SBATCH — ensemble 3d (swap for the 4d block in the cheat sheet above)
# =============================================================================

#SBATCH --job-name=ensemble_mobo_3d
#SBATCH --output=optimize/scripts/logs/ensemble_mobo_3d_%j.out
#SBATCH --error=optimize/scripts/logs/ensemble_mobo_3d_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO}"
mkdir -p logs optimize/runs

# shellcheck source=/dev/null
source "${REPO}/slurm/mobo_job_common.sh"
_mobo_setup_env

# =============================================================================
# EXAMPLES
# =============================================================================
#
# --- Batch RF (campaign1a repeat) --------------------------------------------
# MOBO_CMD=(
#   python optimize/run_mobo.py
#   --no-show
#   --device cuda
#   --batch
#   --config optimize/mobo_batch_configs/campaign1a_mobo_05_06_15_32_repeat.json
#   --time-limit-hours 0.4
# )
#
# --- Resume (crawl all optimize/runs/mobo_*, reuse latest run_config) --------
# MOBO_CMD=(
#   python optimize/run_mobo.py
#   --no-show
#   --device cuda
#   --resume
# )
#
# --- Resume from one run directory only --------------------------------------
# MOBO_CMD=(
#   python optimize/run_mobo.py
#   --no-show
#   --device cuda
#   --resume-from optimize/runs/mobo_19_06_13_01_55_545039
# )
#
# --- Copy config from a past run (fresh Sobol + BO, no prior trials) ---------
# MOBO_CMD=(
#   python optimize/run_mobo.py
#   --no-show
#   --device cuda
#   --copy-config optimize/runs/mobo_05_06_15_32
#   --time-limit-hours 0.4
# )
#
# --- Ackley benchmark (no batch JSON; uses --dataset) ------------------------
# MOBO_CMD=(
#   python optimize/run_mobo.py
#   --no-show
#   --device cuda
#   --dataset ackley4d
#   --ackley-variant realistic
#   --time-limit 0.5
# )
#
# --- Seed from best trial hparams (re-evaluated; Sobol still runs) -----------
# MOBO_CMD=(
#   python optimize/run_mobo.py
#   --no-show
#   --device cuda
#   --batch
#   --config optimize/mobo_batch_configs/campaign1a_mobo_05_06_15_32_repeat.json
#   --start-from-best optimize/runs/mobo_05_06_15_32/trial_315
# )
#
# --- Ensemble 3d (matches optimize/scripts/ensemble_mobo_3d.sbatch) ------------
# RUN_DIR="optimize/runs/mobo_ensemble_3d_job${SLURM_JOB_ID}"
# MOBO_CMD=(
#   python optimize/run_mobo.py
#   --no-show
#   --dataset ensemble
#   --dim 3
#   --time-limit 0.2
#   --share-history
#   --start-from-best optimize/runs/mobo_3d_05_06_15_32/trial_112
#   --run-dir "${RUN_DIR}"
# )
#
# --- Ensemble 4d (matches optimize/scripts/ensemble_mobo_4d.sbatch) ------------
# RUN_DIR="optimize/runs/mobo_ensemble_4d_job${SLURM_JOB_ID}"
# MOBO_CMD=(
#   python optimize/run_mobo.py
#   --no-show
#   --dataset ensemble
#   --dim 4
#   --time-limit 0.3
#   --share-history
#   --start-from-best optimize/runs/mobo_3d_05_06_15_32/trial_112
#   --run-dir "${RUN_DIR}"
# )
#
# =============================================================================
# ACTIVE COMMAND — edit this block only (one MOBO_CMD= assignment in this file)
# =============================================================================

RUN_DIR="optimize/runs/mobo_ensemble_3d_job${SLURM_JOB_ID}"
ENSEMBLE_SEED=$(( SLURM_JOB_ID % 2147483647 ))

MOBO_CMD=(
  python optimize/run_mobo.py
  --no-show
  --dataset ensemble
  --dim 3
  --time-limit 0.2
  --share-history
  --ensemble-seed "${ENSEMBLE_SEED}"
  --start-from-best
  optimize/runs/archived_runs/mobo_3d_05_06_15_32/trial_112
  /home/adewinmb/orcd/scratch/ZoMBI-Hop/optimize/runs/archived_runs/mobo_ensemble_3d_job16948629/trial_33
  optimize/runs/archived_runs/mobo_ensemble_3d_job16861902/trial_49
  optimize/runs/archived_runs/mobo_ensemble_3d_job16954340/trial_63
  --run-dir "${RUN_DIR}"
)
# =============================================================================

echo "=== MOBO command ==="
printf '  %q' "${MOBO_CMD[@]}"
echo
echo "=== cwd: ${REPO} ==="
echo

"${MOBO_CMD[@]}"
