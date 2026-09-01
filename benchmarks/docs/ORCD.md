# Running zhbench on ORCD

Prep for the 6-D high-budget run. **Nothing here has been executed on the cluster
yet** — this is the spec and the template, written so the job is submit-ready.
The laptop numbers quoted are measured; the cluster numbers are estimates and are
labelled as such.

## Why the cluster at all

`real6d` is the only real landscape sharp enough to discriminate (peak rarity
0.0021, against 0.0349 for real3d), and it is the one landscape s1 could say
nothing about: `reached_ratio` was still climbing steeply at N=2000
(0.004 → 0.006 → 0.006 → 0.026, no plateau). N=2000 is 83 lines against the 109
lines / 2616 samples the real campaign used to find its 68 optima.

The decision rule is stated in advance, so the result is not chosen after the fact:

> If the `reached_ratio` curves plateau by N≈4000, 6-D discriminates and we report
> it. If they are still climbing at N=6000, 6-D is **budget-limited at printable
> scales** and we report that instead.

Both outcomes are publishable. Only the second one is embarrassing to discover
after quoting a 6-D comparison.

## The environment: one OpenMP provider, and that is the point

The laptop environment loads **two** OpenMP runtimes in every process that imports
`optimize/run_mobo.py` — `sklearn/.libs/vcomp140.dll` (Microsoft) and
`torch/lib/libiomp5md.dll` (Intel). `KMP_DUPLICATE_LIB_OK=TRUE` suppresses the
abort, and it is kept locally because it is harmless there.

Two things to be clear about:

* It was **not** the cause of the 0xC0000005 crashes. That was an unbounded
  allocation in the acquisition loop (DESIGN.md §19); a pure-BoTorch reproducer
  that never loads `vcomp140` crashed identically, and `KMP_DUPLICATE_LIB_OK` did
  not help.
* It is still wrong, and a cluster is where it stops being harmless: duplicate
  OpenMP runtimes give nondeterministic thread-pool behaviour, which on a
  multi-day array job means results that cannot be reproduced.

So the cluster env is built from a single provider, and
`slurm/zhbench_array.sbatch` **refuses to run** if it finds more than one rather
than papering over it. Do not add `KMP_DUPLICATE_LIB_OK` to the cluster env to get
past that check — fix the env.

```bash
conda env create -f slurm/environment.yml -n zombi-hop-linebo   # existing spec
conda activate zombi-hop-linebo
# pull torch and scikit-learn from ONE channel so one libgomp/libiomp is linked
conda install -c conda-forge pytorch scikit-learn        # not pip torch + conda sklearn
python -c "import sklearn, torch, glob, os; print(
    glob.glob(os.path.dirname(sklearn.__file__)+'/.libs/*omp*'),
    glob.glob(os.path.dirname(torch.__file__)+'/lib/*omp*'))"
```

Record the resolved set (`conda list --explicit`) into the run directory. The
published bundle cannot say which of two local interpreters produced it; the
cluster bundle must not repeat that.

Also required, from `slurm/mobo_job_common.sh`: conda's `libstdc++` ahead of the
system one on `LD_LIBRARY_PATH` (ORCD compute nodes ship an older `libstdc++` and
BoTorch's JIT `logei` extension needs `GLIBCXX_3.4.29+`), and `MPLBACKEND=Agg`,
without which `run_mobo`'s module-scope `pyplot` import kills the worker.

## Submitting

The manifest is the single source of truth for cell naming — array task `i` runs
cell `i`, and nothing re-derives a directory name in shell.

```bash
python -m benchmarks.zhbench.suite s6_highn \
    --suite-dir benchmarks/runs/s6_highn_orcd \
    --manifest  benchmarks/runs/s6_highn_orcd/manifest.json
# prints: sbatch --array=0-<N-1> slurm/zhbench_array.sbatch
ZHBENCH_MANIFEST=benchmarks/runs/s6_highn_orcd/manifest.json \
    sbatch --array=0-<N-1> slurm/zhbench_array.sbatch
```

Aggregate afterwards on the login node. Resume skips every finished cell, so this
re-runs nothing:

```bash
python -m benchmarks.zhbench.suite s6_highn --suite-dir benchmarks/runs/s6_highn_orcd
python -m benchmarks.zhbench.report benchmarks/runs/s6_highn_orcd
python -m benchmarks.zhbench.stats  benchmarks/runs/s6_highn_orcd
```

A partially failed array is re-submitted verbatim: completed cells exit
immediately on the `metrics.json` check.

## Sizing, and the trap that made this a prerequisite

The per-cell timeout **was hard-coded at 7200 s and reachable from nowhere**. At
N=2000 the slowest cell (`real6d` / `zombihop`) took 6374 s — 89% of the limit.
At N=6000 it would have blown straight through it, and a timeout is deliberately
*not* retried, so those cells would have been silently absent from the bundle with
an error string in place of a result. `timeout_s` is now a config field with a
per-objective override (DESIGN.md §26); the sbatch `--time` must be set above it.

Measured per-cell wall-clock at N=2000 on the laptop (CPU, 1 thread/cell):

| method | real3d | real4d | real6d |
|---|---|---|---|
| `random` | 1 s | 1 s | 1 s |
| `gp_ts` | 211 s | 299 s | 445 s |
| `gp_qucb` | 2374 s | 1819 s | 1755 s |
| `gp_qlogei` | 1960 s | 2593 s | 2509 s |
| `zombihop` | 509 s | 658 s | **4784 s** (max 6374 s) |
| `zombihop_nc5` | 465 s | 423 s | 2913 s |

**Do not extrapolate these linearly to N=6000.** ZoMBI-Hop's cost per decision
grows with the number of samples the GP carries (`max_gp_points` caps it, but the
zoom/activation count does not scale linearly), and the GP baselines refit on a
growing design. **Measure one cell before sizing the array** — the first thing to
run on the cluster is a single `real6d / zombihop` cell at N=6000 with a generous
`--time`, and the array is sized from what it actually takes.

Suggested starting point, to be replaced by that measurement:
`--time=24:00:00`, `timeout_s: 72000` for the 6-D objective, `--cpus-per-task=4`.

## GPU

Untested for this harness. `src/core/zombihop.py` sets a CUDA default device and
`float32` at **import time** when CUDA is present (UPSTREAM_REQUESTS item 1);
`zhbench/_repo.preserve_torch_defaults` contains that mutation, which is invisible
on the CPU-only laptop and would first bite here. Before requesting a GPU
partition, run the test suite on a GPU node and confirm
`test_core_pins.py` and `test_protocol.py` still pass — a silent dtype change from
float64 to float32 would move every number in the bundle.
