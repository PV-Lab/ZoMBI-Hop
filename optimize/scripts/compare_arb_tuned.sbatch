#!/bin/bash
#SBATCH --job-name=compare_arb_tuned
#SBATCH --output=optimize/scripts/logs/compare_arb_tuned_%j.out
#SBATCH --error=optimize/scripts/logs/compare_arb_tuned_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1
#SBATCH --requeue

# Arbitrary-vs-tuned hyperparameter comparison (paired seeds) on the newRF fixed
# RF-surrogate landscape. This is a BOUNDED job (n_seeds x 2 runs x iterations),
# not an open-ended MOBO search, so unlike the ensemble_mobo_*.sbatch scripts it
# does NOT chain/requeue itself — it runs once and exits. 6h wall time (half the
# 12h the MOBO scripts request) is ample headroom for the default 5 seeds x 40
# iterations; --requeue only lets SLURM restart it after a node failure.

# Output dir keyed on the job id so a requeued job re-uses the same folder name.
RUN_DIR="optimize/runs/compare_arb_tuned_job${SLURM_JOB_ID}"

# Defaults: arbitrary = optimize/hparams/3d_llm_chosen.json,
#           tuned     = optimize/runs/mobo_ensemble_3d_job17560178/trial_13/trial.json,
#           landscape = newRF (fixed RF surrogate from the default results DB).
# Point --db at your results DB/CSV if the default 2nd_real_run.db is not resolvable
# under data/, and override --arb-hparams / --tuned-hparams to compare other sets.
uv run python optimize/compare_arb_tuned.py \
    --dataset newRF \
    --db 2nd_real_run.db \
    --n-seeds 5 \
    --iterations 40 \
    --out-dir "$RUN_DIR"

rc=$?
echo "[$(date)] compare_arb_tuned.py exited with rc=$rc; outputs in $RUN_DIR"
exit $rc
