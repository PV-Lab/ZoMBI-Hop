#!/bin/bash
#SBATCH --job-name=ensemble_mobo_10d
#SBATCH --output=optimize/scripts/logs/ensemble_mobo_10d_%j.out
#SBATCH --error=optimize/scripts/logs/ensemble_mobo_10d_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# =============================================================================
# 10d MOBO campaign "campaign_10d_20k_20261124" (started 2026-11-24).
#   * seeded with optimize/hparams/dup1c_custom.json + 3 random (Sobol) trials,
#     then MOBO (qLogNEHVI) on the pooled history;
#   * 2 ensemble landscapes per trial (--ensemble-repeats 2), 20k measured points
#     per landscape run (--budget-points 20000), no plots (data only);
#   * history pooled ONLY from this campaign's run dirs ($CAMPAIGN/mobo_*): it lives
#     outside optimize/runs/mobo_*, and --share-dirs points at it alone. (Its
#     signature -- budget 20000, 2 repeats, random_simplex_seeded -- matches no
#     older run anyway.)
# Up to 6 of these run at once (babysitter TARGET[10d]=6 plus the $CAMPAIGN/max_workers
# cap, which run_mobo checks at each trial boundary; see optimize/worker_cap.py -- the
# cap is how a running chain is shrunk, since a job's `sbatch "$0"` resubmits SLURM's
# spooled copy of this script and never sees edits to it). Was 8 until 2026-11-25.
# The previous 10d fleet script (3000 points, 5 repeats) is in git history.
#
# Roles: the first 4 jobs to start each claim one init trial (seed, sobol1..3) with
# an atomic mkdir under $CAMPAIGN/init_claims/; every later job is a BO worker. A BO
# worker that starts before the 4 init trials are done does not sit on a GPU: it
# resubmits itself to start an hour later and exits. Init workers run BO after their
# init trial, waiting (--min-history 4) until all 4 init trials are in.
# =============================================================================

CAMPAIGN="optimize/runs/campaign_10d_20k_20261124"
N_INIT=4                    # dup1c_custom seed + 3 Sobol
WALL_S=$(( 24 * 3600 ))     # keep equal to --time above
INIT_GRACE_H=14             # BO workers stop waiting for init this long after launch

# --- auto-continue across the wall-time limit ---------------------------------
# Normal path: run_mobo stops on its own before starting a landscape run that cannot
# finish by $DEADLINE (exit 75) and this job requeues itself under the SAME id, so the
# trial resumes at that landscape run. The USR1 trap (120 s before the kill) is the
# backstop if a run overruns. Requeue keeps the job id -> same RUN_DIR -> resume; if
# requeue is refused, a fresh job is submitted (new id, new run dir).
requeued=0
requeue_self() {
    if scontrol requeue "$SLURM_JOB_ID"; then
        requeued=1
        echo "[$(date)] requeue accepted"
    else
        echo "[$(date)] requeue refused; submitting a fresh job"
        sbatch "$0"
    fi
}
on_time_limit() {
    echo "[$(date)] wall-time approaching; requeuing job $SLURM_JOB_ID"
    requeue_self
    exit 0
}
trap on_time_limit USR1

DEADLINE=$(( $(date +%s) + WALL_S - 300 ))
mkdir -p "$CAMPAIGN/init_claims"
[ -f "$CAMPAIGN/started_at" ] || date +%s > "$CAMPAIGN/started_at"

# Stable run dir keyed on the job id (a requeue keeps the id and resumes in place).
# The mobo_ prefix is what --share-history globs.
RUN_DIR="$CAMPAIGN/mobo_ensemble_10d_job${SLURM_JOB_ID}"
# Landscape seed per job, stable across a requeue (int32 range for NumPy).
ENSEMBLE_SEED=$(( SLURM_JOB_ID % 2147483647 ))

n_done() { ls "$CAMPAIGN"/mobo_*/trial_*/trial.json 2>/dev/null | wc -l; }

# --- role (persisted in RUN_DIR/role so a requeue keeps it) --------------------
if [ -f "$RUN_DIR/role" ]; then
    ROLE=$(cat "$RUN_DIR/role")
else
    ROLE=bo
    for r in seed sobol1 sobol2 sobol3; do
        c="$CAMPAIGN/init_claims/$r"
        if [ -d "$c" ]; then
            # Reclaim an init slot whose owner is gone (not queued/running) without
            # having finished its init trial (a crash replaced by a fresh job id).
            owner=$(cat "$c/owner" 2>/dev/null)
            if [ -n "$owner" ] && [ -z "$(squeue -h -j "$owner" -o %T 2>/dev/null)" ] \
               && ! ls "$CAMPAIGN/mobo_ensemble_10d_job${owner}"/trial_1/trial.json >/dev/null 2>&1; then
                echo "[$(date)] init slot $r: owner job $owner is gone without a result; reclaiming"
                rm -rf "$c"
            fi
        fi
        if mkdir "$c" 2>/dev/null; then
            echo "$SLURM_JOB_ID" > "$c/owner"
            ROLE=$r
            break
        fi
    done
    if [ "$ROLE" = bo ]; then
        started=$(cat "$CAMPAIGN/started_at")
        if [ "$(n_done)" -lt "$N_INIT" ] && [ $(( $(date +%s) - started )) -lt $(( INIT_GRACE_H * 3600 )) ]; then
            echo "[$(date)] BO worker: $(n_done)/$N_INIT init trials done; resubmitting to start in 1 h"
            sbatch --begin=now+1hour "$0"
            exit 0
        fi
    fi
    mkdir -p "$RUN_DIR"
    echo "$ROLE" > "$RUN_DIR/role"
fi
echo "[$(date)] job $SLURM_JOB_ID on $(hostname): role=$ROLE run_dir=$RUN_DIR seed=$ENSEMBLE_SEED"

case "$ROLE" in
    seed)   INIT_ARGS=(--start-from-best optimize/hparams/dup1c_custom.json --n-init-trials 0) ;;
    sobol*) INIT_ARGS=(--n-init-trials 1 --force-init) ;;
    *)      INIT_ARGS=(--n-init-trials 0) ;;
esac

# --time-limit 9: per-landscape-run safety cap in hours (a dup1c_custom 10d run at
# 20k points takes ~4.8 h); runs normally end on the point budget.
uv run python optimize/run_mobo.py --dataset ensemble --dim 10 --init axial \
    --budget-points 20000 --ensemble-repeats 2 --time-limit 9 \
    --share-history --share-dirs "$CAMPAIGN" \
    --ensemble-seed "$ENSEMBLE_SEED" \
    "${INIT_ARGS[@]}" --min-history "$N_INIT" \
    --deadline-epoch "$DEADLINE" \
    --run-dir "$RUN_DIR" < /dev/null &
wait $!
rc=$?

if [ "$requeued" -eq 0 ]; then
    if [ "$rc" -eq 75 ]; then
        echo "[$(date)] run stopped before its deadline (rc=75); requeuing job $SLURM_JOB_ID"
        requeue_self
    elif [ "$rc" -eq 76 ]; then
        echo "[$(date)] worker retired under $CAMPAIGN/max_workers (rc=76); not resubmitting"
    elif [ "$rc" -eq 0 ]; then
        echo "[$(date)] run exited cleanly (rc=0); submitting next job"
        sbatch "$0"
    else
        # Marker the babysitter classifies crashes by -- keep the wording.
        echo "[$(date)] run exited with rc=$rc (fatal abort/crash) — NOT resubmitting."
        echo "[$(date)] fix the issue and resubmit manually with: sbatch $0"
    fi
fi
