#!/bin/bash
#SBATCH --job-name=zombi_tests
#SBATCH --output=optimize/scripts/logs/run_tests_gpu_%j.out
#SBATCH --error=optimize/scripts/logs/run_tests_gpu_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1

# Run the pytest suite on a GPU node — most tests require CUDA and the login
# nodes have none. Bounded (~1 min), runs once and exits.
#
#   sbatch optimize/scripts/run_tests_gpu.sbatch                         # whole suite
#   sbatch optimize/scripts/run_tests_gpu.sbatch tests/test_box_domain.py -x
#   COMPARE_HEAD=1 sbatch optimize/scripts/run_tests_gpu.sbatch          # + committed HEAD
#
# COMPARE_HEAD=1 also runs the suite on a `git archive` of HEAD (in a temp dir,
# same interpreter) so a failure can be attributed: present on both = already
# broken before your working-tree changes; working tree only = a regression.

REPO="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

echo "[$(date)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
nvidia-smi -L || true

PYTEST=(uv run --project "$REPO" --with pytest python -m pytest -p no:cacheprovider -q -rfEs)
ARGS=("$@")
[[ ${#ARGS[@]} -eq 0 ]] && ARGS=(tests)

echo "===== WORKING TREE ($(git rev-parse --short HEAD) + uncommitted) ====="
"${PYTEST[@]}" "${ARGS[@]}"
rc=$?

if [[ "${COMPARE_HEAD:-0}" == "1" ]]; then
    HEAD_DIR=$(mktemp -d)
    trap 'rm -rf "$HEAD_DIR"' EXIT
    git archive HEAD src tests scripts optimize benchmarks synthetic_data | tar -x -C "$HEAD_DIR"
    echo "===== HEAD $(git rev-parse --short HEAD) (committed only) ====="
    (cd "$HEAD_DIR" && "${PYTEST[@]}" "${ARGS[@]}") || true
fi

echo "[$(date)] working-tree pytest exited with rc=$rc"
exit $rc
