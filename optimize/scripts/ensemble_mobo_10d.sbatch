#!/bin/bash
#SBATCH --job-name=ensemble_mobo_10d
#SBATCH --output=optimize/scripts/logs/ensemble_mobo_10d_%j.out
#SBATCH --error=optimize/scripts/logs/ensemble_mobo_10d_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

# --- auto-continue across the wall-time limit ---------------------------------
# 120s before the --time limit SLURM sends USR1 to this batch script.
# Preferred path: requeue THIS job (same ID) so it resumes when scheduled.
# Fallback: if requeue is refused (e.g. requeue/age limit reached), submit a
# brand-new job so the chain never dies. Cancel with `scancel` to stop it.
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
RUN_DIR="optimize/runs/mobo_ensemble_10d_job${SLURM_JOB_ID}"

# Randomize the ensemble landscape per job, but keep it STABLE across a requeue.
# Keying the seed on SLURM_JOB_ID (like RUN_DIR above) means a `scontrol requeue`
# — same job id — rebuilds the identical landscape so the resumed trials still
# match, while every fresh sbatch / clean-exit resubmit gets a new id and thus a
# new landscape. Modulo keeps it in the int32 range NumPy's RandomState accepts.
ENSEMBLE_SEED=$(( SLURM_JOB_ID % 2147483647 ))

# Seeds: the four 6d Pareto winners (2 best dist_to_needles, 2 best dup_fraction from
# the job192023xx pool) PLUS merge1, the hand-crafted config that led the showdown's
# gap-to-field on dist while declaring the most needles — all PROJECTED into the
# HPARAM_SPACE re-tightened on 2026-08-12. The projection is written to disk rather
# than left to hparams_to_norm's silent [0,1] clamp so the configuration that gets
# re-evaluated here is exactly the one the JSON describes; seeding the older
# optimize/hparams/clamped_6d/ files directly would now re-evaluate something neither
# file describes (dup1c's nat_grad_max_steps=400 would land on 80, dist2c's
# max_zooms=10 on 6, and so on). merge1 needed no projection at all — it already sat
# inside every new bound, which is a large part of why the new bounds are where they
# are. Regenerate with optimize/hparams/tight_6d/ if HPARAM_SPACE moves again.
#
# These are re-evaluated as initial trials IN ADDITION to the full Sobol init
# (load_seed_hparams ignores their stored 6d metrics), which is what makes a fresh
# 10d pool start from known-good regions without inheriting incomparable scores.
# The old 10d runs were moved to optimize/runs/archived_runs/ — they were tuned
# under the OLD bounds, where the same normalised coordinate means a different
# configuration (and the space has since lost an axis, so it is not even the same
# dimension), and --share-history only globs one level (runs/mobo_*), so archiving
# them is what keeps this pool clean. Do not move them back.
uv run python optimize/run_mobo.py --dataset ensemble --dim 10 --time-limit 0.7 --share-history \
    --ensemble-seed "$ENSEMBLE_SEED" \
    --start-from-best \
        optimize/hparams/tight_6d/dist1t.json \
        optimize/hparams/tight_6d/dist2t.json \
        optimize/hparams/tight_6d/dup1t.json \
        optimize/hparams/tight_6d/dup2t.json \
        optimize/hparams/tight_6d/merge1t.json \
    --run-dir "$RUN_DIR" &
wait $!
rc=$?

# Reached only if the run exits on its own (not killed by the wall-time signal).
# Only resubmit on a clean exit (rc=0). A non-zero rc means run_mobo.py hit a
# fatal abort (MAX_CONSEC_FAIL consecutive crashes) or the process died — in that
# case resubmitting just spawns another dead run_dir every couple of minutes, so
# stop the chain and let the user fix the root cause and resubmit manually.
if [ "$requeued" -eq 0 ]; then
    if [ "$rc" -eq 0 ]; then
        echo "[$(date)] run exited cleanly (rc=0); submitting next job"
        sbatch "$0"
    else
        echo "[$(date)] run exited with rc=$rc (fatal abort/crash) — NOT resubmitting."
        echo "[$(date)] fix the issue and resubmit manually with: sbatch $0"
    fi
fi
