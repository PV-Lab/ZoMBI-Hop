#!/bin/bash
# =============================================================================
# slurm/run_mobo_manual_ensemble_4d.sbatch — ensemble 4d MOBO with requeue chain
# =============================================================================
#
# Submit from repo root:
#   cd ~/ZoMBI-Hop
#   sbatch slurm/run_mobo_manual_ensemble_4d.sbatch
#
# Auto-continue: 120s before wall time SLURM sends USR1 → requeue this job
# (or submit a fresh one). On a clean exit (rc=0) the script chains another job.
# Cancel the chain with `scancel <job_id>`. Fatal MOBO aborts (rc≠0) stop the chain.
#
# One-shot (no chain): use slurm/run_mobo_manual.sbatch instead.
# =============================================================================

#SBATCH --job-name=ensemble_mobo_4d
#SBATCH --output=optimize/scripts/logs/ensemble_mobo_4d_%j.out
#SBATCH --error=optimize/scripts/logs/ensemble_mobo_4d_%j.err
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO}"
mkdir -p logs optimize/runs optimize/scripts/logs

# shellcheck source=/dev/null
source "${REPO}/slurm/mobo_job_common.sh"
_mobo_setup_env

requeued=0
on_time_limit() {
    echo "[$(date)] wall-time approaching; requeuing job $SLURM_JOB_ID"
    if scontrol requeue "$SLURM_JOB_ID"; then
        requeued=1
        echo "[$(date)] requeue accepted"
    else
        echo "[$(date)] requeue refused; submitting a fresh job"
        sbatch "$0"
    fi
    exit 0
}
trap on_time_limit USR1

# Stable run dir keyed on the job id: a `scontrol requeue` keeps the SAME
# SLURM_JOB_ID, so the relaunched job re-enters this exact folder and resumes the
# interrupted ensemble trial (reusing already-finished run_<k> repeats) instead of
# starting a fresh run and re-spending that GPU time. A clean-exit chain or the
# fallback resubmit gets a new job id (new dir) — a genuinely fresh run, which is
# correct. Keep the `mobo_` prefix so shared-history's `mobo_*` glob still sees it.
RUN_DIR="optimize/runs/mobo_ensemble_4d_job${SLURM_JOB_ID}"

MOBO_CMD=(
  python optimize/run_mobo.py
  --no-show
  --dataset ensemble
  --dim 4
  --time-limit 0.3
  --share-history
  --start-from-best optimize/runs/mobo_3d_05_06_15_32/trial_112
  --run-dir "${RUN_DIR}"
)

echo "=== MOBO command ==="
printf '  %q' "${MOBO_CMD[@]}"
echo
echo "=== cwd: ${REPO} ==="
echo

"${MOBO_CMD[@]}" &
wait $!
rc=$!

if [ "$requeued" -eq 0 ]; then
    if [ "$rc" -eq 0 ]; then
        echo "[$(date)] run exited cleanly (rc=0); submitting next job"
        sbatch "$0"
    else
        echo "[$(date)] run exited with rc=$rc (fatal abort/crash) — NOT resubmitting."
        echo "[$(date)] fix the issue and resubmit manually with: sbatch $0"
    fi
fi
