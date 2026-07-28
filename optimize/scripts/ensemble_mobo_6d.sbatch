#!/bin/bash
#SBATCH --job-name=ensemble_mobo_6d
#SBATCH --output=optimize/scripts/logs/ensemble_mobo_6d_%j.out
#SBATCH --error=optimize/scripts/logs/ensemble_mobo_6d_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
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
# Submitting this script N times therefore gives N distinct concurrent run dirs.
RUN_DIR="optimize/runs/mobo_ensemble_6d_job${SLURM_JOB_ID}"

# Randomize the ensemble landscape per job, but keep it STABLE across a requeue.
# Keying the seed on SLURM_JOB_ID (like RUN_DIR above) means a `scontrol requeue`
# — same job id — rebuilds the identical landscape so the resumed trials still
# match, while every fresh sbatch / clean-exit resubmit gets a new id and thus a
# new landscape. Modulo keeps it in the int32 range NumPy's RandomState accepts.
ENSEMBLE_SEED=$(( SLURM_JOB_ID % 2147483647 ))

# History sharing between the three concurrent 6d jobs, and ONLY those.
# --share-history pools (X,Y) from every run whose run_config signature matches
# (dataset/dim/time-limit/direction/variant/ensemble margin+repeats). All three
# submissions of this script share that signature — only --ensemble-seed and the
# run dir differ, and neither is part of the signature — so each job's GP keeps
# ingesting the other two's finished trials.
# --share-dirs pins the scan to THIS account's runs dir, so eve_lal's runs are
# never pooled in even if collab_dirs.SHARE_COLLABORATOR_HISTORY is flipped back
# to True for some other run (it is False today, but this makes it explicit and
# local to the 6d campaign).
OWN_RUNS_DIR="/home/adewinmb/orcd/scratch/ZoMBI-Hop/optimize/runs"

# Seeds: hyperparameters that are RE-EVALUATED on this 6d objective as extra
# initial trials (stored metrics are ignored), alongside the full Sobol init.
# Chosen with pareto.py as the 2 best dist_to_needles trials in each of the
# ensemble 4d and ensemble 10d pools (own runs only):
#   4d  mobo_ensemble_4d_job17402635/trial_9   dist=3.184  dup=0.710  t/iter=5.79
#   4d  mobo_ensemble_4d_job17402635/trial_6   dist=4.011  dup=0.677  t/iter=3.81
#   10d mobo_ensemble_10d_job17776000/trial_34 dist=1.893  dup=0.553  t/iter=3.21
#   10d mobo_ensemble_10d_job17775999/trial_13 dist=2.165  dup=0.506  t/iter=3.22
# (10d trial_34 is the pool's global dist minimum; 17775999/trial_13 is next best.
# Both 4d picks are also Pareto-optimal in the 4d pool.) load_seed_hparams() skips
# any seed dir that has been cleaned off scratch with a warning rather than aborting,
# so re-derive replacements with pareto.py if these ever disappear.
#
# --time-limit 0.5 h per trial sits between the 4d (0.3) and 10d (0.7) budgets.
# Changing it changes the shared-history signature, so keep all three 6d jobs on
# this one script — a job launched with a different budget will not pool.
uv run python optimize/run_mobo.py --dataset ensemble --dim 6 --time-limit 0.5 --share-history \
    --share-dirs "$OWN_RUNS_DIR" \
    --ensemble-seed "$ENSEMBLE_SEED" \
    --start-from-best \
        optimize/runs/mobo_ensemble_4d_job17402635/trial_9 \
        optimize/runs/mobo_ensemble_4d_job17402635/trial_6 \
        optimize/runs/mobo_ensemble_10d_job17776000/trial_34 \
        optimize/runs/mobo_ensemble_10d_job17775999/trial_13 \
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
