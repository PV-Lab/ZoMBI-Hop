#!/bin/bash
#SBATCH --job-name=render_conet
#SBATCH --output=visualization/scripts/logs/render_conet_%A_%a.out
#SBATCH --error=visualization/scripts/logs/render_conet_%A_%a.err
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=sched_mit_sloan_batch_r8

# Headless render of BOTH CoNet artifacts for a finished ensemble run directory:
#   * conet.png          — the run's own samples + its discovered-needle stars
#   * conet_uniform.png  — a uniform-sampling baseline of the SAME landscape, with
#                          the landscape's true optima starred (the "no-optimizer"
#                          reference: how much of the space / how many optima a run
#                          would touch with unbiased coverage).
#
# Why a batch job and NOT the login node: each render embeds EVERY sample (~19k)
# and fits UMAP on a full N×N precomputed distance matrix — a ~10-15 GB,
# multi-minute, single-threaded job. Far too heavy for a shared login node.
# Why NO GPU: pure numpy/scipy/umap-learn/matplotlib CPU work.
#
# Two ways to run:
#
#   1) One run directory (no array):
#        sbatch visualization/scripts/render_conet.sbatch <RUN_DIR>
#
#   2) A batch of run directories via a manifest (one RUN_DIR per line), as a job
#      array where each task renders the run on its line:
#        sbatch --array=1-$(wc -l < visualization/scripts/conet_batch_manifest.txt) \
#               visualization/scripts/render_conet.sbatch \
#               visualization/scripts/conet_batch_manifest.txt

ARG="${1:-optimize/runs/mobo_ensemble_10d_job17775999/trial_13/run_1}"

if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    # Array mode: ARG is a manifest file; pick the line for this array index.
    RUN_DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$ARG")
    if [ -z "$RUN_DIR" ]; then
        echo "no manifest entry at line ${SLURM_ARRAY_TASK_ID} of $ARG" >&2
        exit 1
    fi
else
    RUN_DIR="$ARG"
fi

echo "[$(date)] rendering CoNet artifacts for: $RUN_DIR"

# Keep BLAS from oversubscribing beyond our cpus-per-task allocation.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
# Headless: no display on a compute node.
export MPLBACKEND=Agg

rc=0

echo "[$(date)] --- conet.png (run samples + discovered needles) ---"
uv run python visualization/plot_10d.py \
    --run "$RUN_DIR" --needles --save "${RUN_DIR%/}/conet.png"
rc1=$?; echo "[$(date)] conet.png rc=$rc1"; [ $rc1 -ne 0 ] && rc=$rc1

echo "[$(date)] --- conet_uniform.png (uniform baseline + true optima) ---"
uv run python visualization/uniform_baseline_conet.py \
    --run "$RUN_DIR" --save "${RUN_DIR%/}/conet_uniform.png"
rc2=$?; echo "[$(date)] conet_uniform.png rc=$rc2"; [ $rc2 -ne 0 ] && rc=$rc2

echo "[$(date)] done (rc=$rc)"
exit $rc
