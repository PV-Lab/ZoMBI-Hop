#!/bin/bash
#SBATCH --job-name=accordion_runs
#SBATCH --output=optimize/scripts/logs/accordion_runs_%A_%a.out
#SBATCH --error=optimize/scripts/logs/accordion_runs_%A_%a.err
#SBATCH --time=00:45:00
#SBATCH --array=1-5
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=sched_mit_sloan_batch_r8

# Five candidate ZoMBI-Hop runs on the accordion base landscape, for picking the
# trajectory that goes into the paper's summary figure (visualization/accordion.py
# --slices).
#
# --dataset accordion is the FIXED hand-placed landscape (three optima, one per
# ternary corner) rather than `ensemble`, which re-randomizes the surface per run.
# Because it is deterministic, the five runs differ only in the optimizer's own
# randomness — nothing seeds torch/numpy per run — which is what makes them
# comparable as candidates.
#
# PARALLEL: a 5-task job array, so the whole set finishes in one run's wall clock
# rather than five. (evaluate.py's own --num-runs 5 would give a tidier layout and a
# combined rerun_summary.json, but it loops SEQUENTIALLY — 5x the wall clock.)
#
# Wall clock budget, sized so nothing is lost to a kill:
#   * the optimizer gets 15 minutes (--time-limit-min 15). These are CPU nodes, not
#     the a100s the ensemble scripts use, so each iteration is slower and 15 min
#     lands around 150-175 LineBO iterations.
#   * points.csv / needles.csv and the per-iteration ternary frames are all written
#     AFTER the search stops, at roughly 3.2 s per frame — so ~175 frames is another
#     ~10 minutes.
#   * --time is 45 minutes: 15 + ~10 with a wide margin. An earlier 10-minute job
#     with an 8-minute run killed two of five runs mid-render (CSVs landed, plots/
#     had a third of its frames, no metrics.json). If you raise --time-limit-min,
#     raise --time by MORE, since the frame count grows with it.
#
# This is why the partition is sched_mit_sloan_batch_r8 (group CPU partition, 4-day
# limit, typically a near-empty queue) and not mit_quicktest, which caps jobs at 15
# minutes and cannot fit the render tail. No GPU: a d=3 run is small enough that CPU
# is not the bottleneck.
#
# Hyperparameters: archived trial_112, the 3d set the ensemble sbatch scripts seed
# from. --no-video skips only the MP4 assembly; the frames it would be built from
# are still written to plots/.
#
# Add --coverage if you want each run to also write coverage_ground_truth.npz;
# accordion.py's landscape slice prefers that file, though for this dataset it
# falls back to rebuilding the identical landscape, so it is optional here.

set -u

K="${SLURM_ARRAY_TASK_ID}"
PARENT="optimize/runs/accordion_job${SLURM_ARRAY_JOB_ID}"
# Each task must own its --out-dir: evaluate.py always names the run `run_1` under
# --num-runs 1 and has no run-index flag, so five tasks pointed at one directory
# would overwrite each other. Staging per task and moving the result into place
# afterwards keeps the parallelism AND gives the flat trial_112/run_<K> layout
# (rather than the redundant run_<K>/trial_112/run_1 nesting of job 19306237).
STAGE="${PARENT}/.staging/task_${K}"

uv run python optimize/evaluate.py \
    --hparams optimize/runs/archived_runs/mobo_3d_05_06_15_32/trial_112/trial.json \
    --dataset accordion \
    --num-runs 1 \
    --time-limit-min 15 \
    --no-video \
    --out-dir "$STAGE"
rc=$?

# Promote this task's run into the shared trial dir as run_<K>. mkdir -p is safe
# under concurrency, and every task writes byte-identical hparams.json /
# rerun_config.json, so the racing copies below cannot disagree.
if [ -d "${STAGE}/trial_112/run_1" ]; then
    mkdir -p "${PARENT}/trial_112"
    rm -rf "${PARENT}/trial_112/run_${K}"
    mv "${STAGE}/trial_112/run_1" "${PARENT}/trial_112/run_${K}"
    cp -f "${STAGE}/trial_112/hparams.json" "${PARENT}/trial_112/" 2>/dev/null
    cp -f "${STAGE}/rerun_config.json" "${PARENT}/" 2>/dev/null
    # Per-task summary kept under its own name: evaluate.py writes one summary per
    # invocation covering only that task's single run, so they cannot be merged into
    # one rerun_summary.json without racing. Read metrics.json per run instead.
    cp -f "${STAGE}/rerun_summary.json" "${PARENT}/rerun_summary_run_${K}.json" 2>/dev/null
    rm -rf "$STAGE"
    echo "[$(date)] task ${K} -> ${PARENT}/trial_112/run_${K} (rc=${rc})"
else
    echo "[$(date)] task ${K} FAILED (rc=${rc}); staging left at ${STAGE} for inspection"
fi

exit $rc
