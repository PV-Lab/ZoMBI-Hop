# ZoMBI-Hop Benchmark Harness

This package provides the benchmark interface for comparing ZoMBI-Hop with
other optimization methods on simplex-valued materials objectives.

## Implemented

- Shared objective and optimizer protocols.
- NumPy simplex utilities and ILR/Aitchison distance metrics.
- Deterministic planted synthetic simplex objective.
- Point-mode random/simplex-uniform baseline.
- Point-mode `GP-ARD-EI` and `GP-ARD-UCB` baselines using BoTorch on ILR inputs.
- Point-mode `RF-BO` baseline using scikit-learn random forests on ILR inputs.
- Fair line-mode wrappers for `random_simplex`, `GP-ARD-EI`, `GP-ARD-UCB`, and `RF-BO`.
- Deterministic simplex line candidate generation with mean-acquisition and random line scoring.
- Fair line-budget ZoMBI-Hop adapter using internal LineBO.
- Real 3D perovskite RF-surrogate objective from `data/campaign1a.csv`.
- Optional HEBO adapter using ILR-space suggestions when the external `hebo`
  package is installed.
- Benchmark-local finite-pool TuRBO-1 adapter using BoTorch/GPyTorch in
  normalized ILR coordinates with valid raw-simplex candidate pools.
- Line-metadata sanity audit columns for endpoint validity, endpoint minima, and
  explicit raw-simplex vs ILR length coordinate systems.
- Reproducible CSV/JSON outputs.
- Suite runner with point-level and line-level aggregate CSV summaries.

## Still Stubbed

- SAASBO
- 4D/10D real RF-surrogate objectives
- Native line-mode acquisition for external optimizers such as HEBO/TuRBO/SAASBO

## Optional HEBO Baseline

`optimizer.kind: hebo` is registered as a real optional HEBO adapter. It uses
ILR coordinates internally, derives finite ILR bounds from a deterministic
simplex candidate pool by default, inverse-transforms HEBO suggestions back to
raw simplex compositions, and sends `-y` to HEBO when the benchmark objective is
maximized.

HEBO is not vendored into this repository. Install it in the active environment
before running HEBO scientific configs:

```powershell
.\.venv\Scripts\python -m pip install HEBO
```

Point-mode HEBO configs are:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_hebo_synthetic_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_hebo_real_rf_3d.yaml
```

HEBO line mode is enabled through a documented point-native adapter:

```text
line_adapter: hebo_anchor_chord
```

HEBO suggests one anchor point in ILR space, the benchmark maps it to the raw
simplex, generates deterministic zero-sum simplex chords through that anchor,
selects the longest valid raw-simplex chord, evaluates the full printable line,
and batch-observes all line points back into HEBO. This is not a native HEBO
line acquisition and should be interpreted as an interface-fair adapter, not as
LineBO-equivalent acquisition scoring.

## TuRBO Baseline

`optimizer.kind: turbo` is registered as a benchmark-local TuRBO-1 style
adapter. It does not require the old standalone TuRBO package. It uses the
existing `torch`, `botorch`, and `gpytorch` stack, transforms observed simplex
compositions to ILR coordinates, normalizes those ILR coordinates with
deterministic candidate-pool quantile bounds, and maintains a local trust region
around the current best observed composition.

TuRBO never sends unconstrained ILR optimizer outputs to the objective. Instead,
it generates finite valid raw-simplex candidate pools, transforms them to ILR,
filters or downweights candidates outside the current trust region, scores the
remaining candidates with the configured acquisition, and returns raw simplex
compositions.

Point-mode TuRBO smokes are:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_turbo_synthetic_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_turbo_real_rf_3d.yaml
```

TuRBO line mode uses the existing fair line wrapper:

```text
line_adapter: turbo_acq_line
```

The wrapper scores deterministic candidate lines by mean TuRBO acquisition
across the whole line, downweights points outside the current ILR trust region,
evaluates the selected line, and batch-observes all line points. This is a
point-native trust-region optimizer adapted to line printing through the
benchmark interface, not native LineBO.

Line-mode TuRBO smokes are:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_turbo_line_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_turbo_real_rf_line_3d.yaml
```

## Line Metadata Audit

Each new `line_metrics.csv` row includes endpoint sanity fields:

```text
line_endpoint_min
line_endpoint_sum_deviation
line_endpoints_valid_simplex
line_length_l2_coordinate_system
line_length_ilr_coordinate_system
line_length_l2_within_simplex_diameter
line_adapter
line_adapter_caveat
```

This is intended to diagnose large ILR lengths. For example, ZoMBI-Hop LineBO
can produce raw endpoints that are valid simplex points and have raw L2 length
within the simplex diameter, while one or more endpoint components are exactly
or nearly zero. In that case the large ILR length is expected near-boundary ILR
behavior rather than a raw-coordinate logging bug.

## Real RF Objective

`objective.kind: real_rf_surrogate` evaluates optimizer suggestions against a
real-data-derived random-forest surrogate, not the physical lab hardware. The
default local configs train a deterministic RF from `data/campaign1a.csv` using:

```text
components: FAPbI3, MAPbI3, MAPbBr3
target: Objective
mode: maximize
```

Existing reference optima are loaded from
`optimize/reference_optima/mobo_05_06_15_32_campaign1a.json`. If a future real
RF asset does not include stored optima, the objective can detect up to `top_k`
needles on a deterministic simplex grid and write them to `objective_needles.csv`.
Each run also writes `objective_metadata.json`.

## Line Mode

Line mode uses the budget convention:

```text
total evaluated compositions = n_init + n_lines * points_per_line
```

For the benchmark-local wrappers, one optimizer decision produces one full
simplex composition line. The objective evaluates every composition on that
line, then the wrapped optimizer receives the whole batch in a single
`observe(...)` call. The optimizer is not updated sequentially within the same
printed line.

Line-mode outputs keep the existing point-level files and add line metadata:

```text
points.csv             one row per evaluated composition, with line columns
metrics_over_time.csv  one row per evaluated composition trajectory
line_metrics.csv       one row per printed/evaluated line
summary.json           includes num_lines, points_per_line, and line summaries
```

ZoMBI-Hop uses its own internal candidate-anchor plus LineBO procedure, while
the baseline methods use benchmark-local line wrappers. The comparison is fair
at the experimental-interface level: every method receives the same initial
design, objective, seed convention, line budget, points per line, and batched
within-line update rule. For ZoMBI-Hop, one objective callback equals one
printed/evaluated LineBO line, and `ZoMBIHop.run(...)` is stopped gracefully by
`max_objective_calls`.

## Smoke Commands

Run from the repository root:

```powershell
.\.venv\Scripts\python -m pytest benchmarks/tests -q

.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_random_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_gp_ard_ei_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_gp_ard_ucb_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_rf_bo_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_zombihop_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_turbo_synthetic_3d.yaml
```

Run the first point-mode suite:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1a_synthetic_3d.yaml
```

Run line-mode smokes:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_random_line_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_gp_ard_ei_line_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_gp_ard_ucb_line_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_rf_bo_line_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_zombihop_line_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_turbo_line_3d.yaml
```

Run the first line-mode suite:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1a_synthetic_3d_line.yaml
```

Run the line-mode suite with ZoMBI-Hop:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1a_synthetic_3d_line_with_zombihop.yaml
```

Run real RF-surrogate smokes:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_random_real_rf_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_gp_ard_ei_real_rf_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_rf_bo_real_rf_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_turbo_real_rf_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_random_real_rf_line_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_gp_ard_ei_real_rf_line_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_zombihop_real_rf_line_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_turbo_real_rf_line_3d.yaml
```

Run real RF-surrogate suites:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1a_real_rf_3d_point.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1a_real_rf_3d_line_with_zombihop.yaml
```

Run milestone suites with HEBO and TuRBO:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1a_synthetic_3d_point_with_hebo_turbo.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1a_real_rf_3d_point_with_hebo_turbo.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1a_synthetic_3d_line_with_hebo_turbo.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1a_real_rf_3d_line_with_hebo_turbo.yaml
```

Generate the Milestone 1A synthetic-to-real report after the synthetic and real
RF point/line suites exist:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.report --config benchmarks/configs/report_milestone1a_synthetic_real.yaml
```

Generate the Milestone 1A report with HEBO and TuRBO after the corresponding
suite aggregate directories exist:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.report --config benchmarks/configs/report_milestone1a_with_external_baselines.yaml
```

The report writes a timestamped directory under
`benchmark_runs/reports/milestone1a_synthetic_real/` with Markdown, CSV tables,
synthetic-to-real rank deltas, AUC metrics, and plot PNGs. If `matplotlib` is
not installed in the active environment, CSV/Markdown reporting still runs and
placeholder plot PNGs are written with a note to install `matplotlib`.

Outputs are written under `benchmark_runs/<experiment_name>/`. Suite aggregate
files are written under `benchmark_runs/<suite_name>/aggregate/`.
