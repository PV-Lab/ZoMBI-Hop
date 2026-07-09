# ZoMBI-Hop Benchmark Harness

This package provides the benchmark interface for comparing ZoMBI-Hop with
other optimization methods on simplex-valued materials objectives.

## Implemented

- Shared objective and optimizer protocols.
- NumPy simplex utilities and ILR/Aitchison distance metrics.
- Deterministic planted synthetic simplex objective.
- Brianna-realistic Ackley simplex objective using `synthetic_data.ackley.Ackley("realistic")`.
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
- Optional SAASBO adapter using BoTorch's fully Bayesian SAAS GP on normalized
  ILR coordinates with finite valid raw-simplex candidate pools.
- Line-metadata sanity audit columns for endpoint validity, endpoint minima, and
  explicit raw-simplex vs ILR length coordinate systems.
- Explicit ILR and composition-L2 metric columns:
  `dist_to_needles_ilr`, `pct_matched_ilr`, `dup_fraction_ilr`,
  `dist_to_needles_comp`, `pct_matched_comp`, and `dup_fraction_comp`.
- Line-aware composition duplicate diagnostic: `dup_fraction_comp_cross_line`,
  which ignores duplicate pairs from the same printed line.
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

## SAASBO Baseline

`optimizer.kind: saasbo` is an optional SAASBO adapter for BoTorch's fully
Bayesian sparse-axis-aligned GP. It stores raw simplex observations, transforms
them to ILR coordinates, normalizes ILR coordinates to deterministic
candidate-pool quantile bounds, fits `SaasFullyBayesianSingleTaskGP` with
`fit_fully_bayesian_model_nuts`, and selects only from finite valid raw-simplex
candidate pools.

SAASBO is most scientifically useful for later 4D/10D work. In 3D simplex
benchmarks there are only two ILR axes, so Step 10 treats it mainly as a
compatibility and reporting baseline.

This adapter does not silently fall back to a standard GP. If BoTorch's fully
Bayesian optional dependencies are missing, `optimizer.kind: saasbo` fails with
a dependency message. In BoTorch 0.18.x those extras include `jax`, `jaxlib`,
and `numpyro`; install with:

```powershell
.\.venv\Scripts\python -m pip install "botorch[fully_bayesian]"
```

SAASBO line mode uses the existing fair line wrapper through
`score_candidates(...)` and labels line metadata with:

```text
line_adapter: saasbo_acq_line
```

Each SAASBO optimizer state records dependency status, HMC/NUTS settings,
candidate-pool size, normalized ILR bounds source, fit/acquisition timing, and
median lengthscale diagnostics.

The median lengthscales are reported in the normalized ILR coordinate system
used by the adapter. They are useful as SAASBO sparsity diagnostics, especially
for later 4D/10D suites, but should not be read as direct raw component
importance without mapping back through the composition transform.

SAASBO smoke configs are:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_saasbo_realistic_ackley_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_saasbo_real_rf_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_saasbo_realistic_ackley_line_3d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_saasbo_real_rf_line_3d.yaml
```

## Realistic Ackley Synthetic Objective

`objective.kind: realistic_ackley_simplex` is the Milestone 1A headline
synthetic objective. It wraps Brianna's `Ackley("realistic", dim=3)` generator
from `synthetic_data/ackley.py` with defaults from
`synthetic_data/defaults/ackley.json`:

```text
n_optima: 20
basin_width: 86.0
noise_freq: 9.0
noise_amp: 400.0
```

The older `synthetic_3d_planted` objective remains available as a lightweight
smoke fixture.

## Milestone 1B: 4D Realistic Ackley

Step 11 expands the headline synthetic benchmark to 4D before any 10D work.
The goal is to test whether optimizer rankings and behavior from realistic 3D
Ackley transfer to a modestly higher-dimensional realistic Ackley landscape.
Brianna's notes make 4D the safer next bridge: the 3D/4D synthetic scaling is
more plausible, while 10D remains manually tuned and scientifically uncertain.

The 4D objective uses the same `objective.kind: realistic_ackley_simplex`
wrapper with:

```text
n_components: 4
n_optima: 30
basin_width: 65.0
noise_freq: 9.0
noise_amp: 300.0
```

The 3D configs are unchanged. Each realistic Ackley run writes
`objective_metadata.json`, `objective_needles.csv`, and a lightweight
`objective_distribution_<dim>d.csv` diagnostic summarizing a deterministic
random sample of the objective values.

4D is still synthetic, not real hardware validation. Real 4D RF-surrogate
benchmarking is deferred until campaign data or a vetted surrogate is available;
do not fabricate a 4D real-data objective.

For line-mode duplicate behavior, prefer `dup_fraction_comp_cross_line` as the
headline redundancy metric. The all-points `dup_fraction_comp` remains useful as
a diagnostic, but it can be inflated by adjacent points along the same printed
line. Also note that `pct_matched` can drop from 3D to 4D because the known
optima count rises from 20 to 30.

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

Run Step 9 external-baseline suites with realistic Ackley and real RF:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1a_realistic_ackley_3d_point_external.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1a_realistic_ackley_3d_line_external.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1a_real_rf_3d_point_external.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1a_real_rf_3d_line_external.yaml
```

Run Step 10 SAASBO mini-suites after fully Bayesian dependencies are available:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1a_realistic_ackley_3d_point_with_saasbo_mini.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1a_realistic_ackley_3d_line_with_saasbo_mini.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1a_real_rf_3d_point_with_saasbo_mini.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1a_real_rf_3d_line_with_saasbo_mini.yaml
```

Run Step 11 4D realistic Ackley smokes:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_realistic_ackley_4d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_realistic_ackley_line_4d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_zombihop_realistic_ackley_line_4d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_saasbo_realistic_ackley_4d.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.runner --config benchmarks/configs/smoke_saasbo_realistic_ackley_line_4d.yaml
```

Run Step 11 4D mini-suites before full 10-seed suites:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1b_realistic_ackley_4d_point_external_mini.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1b_realistic_ackley_4d_line_external_mini.yaml
```

Run the full 4D suites only after mini-suite runtime and SAASBO diagnostics look
manageable:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1b_realistic_ackley_4d_point_external.yaml
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.suite --config benchmarks/configs/suite_milestone1b_realistic_ackley_4d_line_external.yaml
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

Generate the Step 9 external-baseline report:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.report --config benchmarks/configs/report_milestone1a_external_baselines.yaml
```

Generate the Step 10 SAASBO mini-suite report:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.report --config benchmarks/configs/report_milestone1a_with_saasbo.yaml
```

Generate the Step 11 3D-to-4D transfer report after the Step 10 3D realistic
Ackley SAASBO mini-suites and Step 11 4D mini-suites exist:

```powershell
.\.venv\Scripts\python -m benchmarks.zombihop_benchmark.report --config benchmarks/configs/report_milestone1b_3d_to_4d.yaml
```

The report writes a timestamped directory under
`benchmark_runs/reports/milestone1a_synthetic_real/` with Markdown, CSV tables,
synthetic-to-real rank deltas, AUC metrics, and plot PNGs. If `matplotlib` is
not installed in the active environment, CSV/Markdown reporting still runs and
placeholder plot PNGs are written with a note to install `matplotlib`.

The Step 11 report additionally writes `dimension_rank_delta.csv` and
`dimension_metric_delta.csv` for 3D-to-4D transfer analysis, plus SAASBO
fit/acquisition timing diagnostics when those runs are present.

Outputs are written under `benchmark_runs/<experiment_name>/`. Suite aggregate
files are written under `benchmark_runs/<suite_name>/aggregate/`.
