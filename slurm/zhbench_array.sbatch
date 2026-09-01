#!/bin/bash
#SBATCH --job-name=zhbench
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/zhbench-%A_%a.out
#SBATCH --error=logs/zhbench-%A_%a.err

set -euo pipefail
# One benchmark CELL per array task. Array bounds come from the manifest.
#
#   python -m benchmarks.zhbench.suite s1_real \
#       --suite-dir benchmarks/runs/s1_real_orcd \
#       --manifest  benchmarks/runs/s1_real_orcd/manifest.json
#   sbatch --array=0-179 slurm/zhbench_array.sbatch          # prints the exact line
#
# Then aggregate on the login node -- resume finds every finished cell, skips it,
# and writes aggregate.csv / curves.json / summary.md without re-running anything:
#
#   python -m benchmarks.zhbench.suite s1_real --suite-dir benchmarks/runs/s1_real_orcd
#   python -m benchmarks.zhbench.report benchmarks/runs/s1_real_orcd
#   python -m benchmarks.zhbench.stats  benchmarks/runs/s1_real_orcd
#
# Environment:
#   ZHBENCH_MANIFEST=...   (default benchmarks/runs/manifest.json)
#
# Why one task per cell rather than one task per suite: a cell is already fully
# isolated (its own process, its own metrics.json), a crash costs one task instead
# of the grid, and the scheduler backfills 4-CPU tasks far better than one
# long 24-CPU job. It also makes the per-cell timeout the SLURM --time, so an
# over-budget cell is reported by the scheduler rather than silently absent.

REPO="${SLURM_SUBMIT_DIR}"
MANIFEST="${ZHBENCH_MANIFEST:-benchmarks/runs/manifest.json}"
TASK_ID="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID not set}"

cd "${REPO}"
mkdir -p logs

source slurm/mobo_job_common.sh
_mobo_setup_env

# One cell per task, so one thread per cell would waste the allocation; give the
# cell its cpus-per-task. Unlike the laptop path (24 cores, ~6 concurrent cells,
# hence OMP_NUM_THREADS=1), here the parallelism is the array itself.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# Fail loudly if the environment still carries two OpenMP runtimes. On the laptop
# KMP_DUPLICATE_LIB_OK=TRUE papers over it; on the cluster we build the env from a
# single provider instead (see benchmarks/docs/ORCD.md), and silently reintroducing
# the duplicate would make results unreproducible rather than merely noisy.
python - <<'PY'
import ctypes.util, glob, os, sys
import sklearn, torch
libs = []
for pkg, pat in ((sklearn, ".libs/*omp*"), (torch, "lib/*omp*")):
    libs += glob.glob(os.path.join(os.path.dirname(pkg.__file__), pat))
if len(libs) > 1:
    sys.exit("two OpenMP runtimes present, refusing to run:\n  " + "\n  ".join(libs))
print("openmp providers:", libs or ["(system)"])
PY

python - "$MANIFEST" "$TASK_ID" <<'PY'
import json, subprocess, sys, os
manifest_path, task_id = sys.argv[1], int(sys.argv[2])
with open(manifest_path, encoding="utf-8") as fh:
    man = json.load(fh)
cells = man["cells"]
if not 0 <= task_id < len(cells):
    sys.exit(f"array task {task_id} out of range for {len(cells)} cells")
job = cells[task_id]

# Resume: a completed cell is a no-op, so requeuing the whole array after a
# partial failure costs nothing for the cells that already landed.
done = os.path.join(job["run_dir"], "metrics.json")
if os.path.exists(done):
    print(f"[skip] {job['cell']} already has metrics.json")
    raise SystemExit(0)

print(f"[{task_id}] {job['cell']}")
cmd = [sys.executable, "-m", "benchmarks.zhbench.runner",
       "--objective", json.dumps(job["objective"]),
       "--optimizer", json.dumps(job["optimizer"]),
       "--seed", str(job["seed"]),
       "--protocol", json.dumps(job["protocol"]),
       "--value-tol", str(job["value_tol"]),
       "--out", job["run_dir"]]
raise SystemExit(subprocess.call(cmd))
PY
