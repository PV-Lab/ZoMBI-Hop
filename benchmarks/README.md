# zhbench — the ZoMBI-Hop multi-optimum benchmark

Measures what ZoMBI-Hop is actually for: finding **many distinct local optima** of
a simplex-constrained objective inside a realistic SDL sample budget, against
standard BO baselines run exactly as their authors intended.

Nothing here edits `src/core`, `optimize/`, `synthetic_data/` or `warm_start/`. It
imports from them, so it stays in sync with whatever lands on `brianna` next.

## Run it

```bash
python -m benchmarks.zhbench.suite smoke
```

Smoke finishes on CPU in about a minute and proves the harness end to end. Its
ZoMBI-Hop hyperparameters are deliberately reduced so it stays fast — **never
quote a number from the smoke suite.**

```bash
python -m benchmarks.zhbench.suite s1_real      # the three real campaigns (~66 CPU-h)
python -m benchmarks.zhbench.suite s2_needles   # the needle-count hypothesis
python -m benchmarks.zhbench.report benchmarks/runs/<suite_dir>   # figures (separate step)
python -m benchmarks.zhbench.stats  benchmarks/runs/<suite_dir>   # paired statistics
pytest benchmarks/tests -q -m "not slow"
```

One trial directly:

```bash
python -m benchmarks.zhbench.runner --objective '{"kind":"ensemble","dim":3,"n_optima":20}' --optimizer '{"name":"zombihop"}' --seed 0 --protocol '{"n_samples":1000}' --out benchmarks/runs/_scratch
```

There is no `--n-samples` flag; every protocol setting goes inside `--protocol`.
The JSON arguments above work verbatim in Git Bash. **Windows PowerShell 5.1 strips
the inner double quotes** and `json.loads` then fails with `Expecting property name
enclosed in double quotes` — escape them there: `'{\"kind\":\"ensemble\"}'`.

Outputs land in `benchmarks/runs/<suite>_<timestamp>/`: one directory per
`(objective, optimizer, seed)` containing `points.csv`, `declared_optima.csv`,
`true_optima.csv`, `metrics.json`, `config_resolved.json`; plus `aggregate.csv`,
`curves.json` and `summary.md` for the suite. Re-running a suite resumes: cells
that already have a `metrics.json` are skipped.

## What is measured

The headline is the quality of the **declared** solution set `S` — the optima a
method claims to have found. ZoMBI-Hop declares needles; standard BO declares
nothing, so `S` is extracted post hoc from its samples by greedy value-ordered
selection with an exclusion radius (`metrics.posthoc_solution_set`), the same
courtesy ROBOT extends to single-solution baselines.

| metric | meaning |
|---|---|
| `peak_ratio` | fraction of true optima matched **one-to-one** within `r = 0.05` |
| `precision` | fraction of declared optima that are real — penalises needle spam |
| `f1` | harmonic mean of the two |
| `dist_to_needles` | `optimize.eval_metrics.metric_dist_to_needles` (Hungarian, capped 0.5) |
| `reached_ratio(t)` | true optima reached by sample `t`, requiring a sample both **near** and **high** |
| `t_first_optimum`, `t_half_optima` | samples to the k-th distinct optimum |
| `input_cost` | SnAKe composition distance travelled |
| `dup_fraction`, `best_y`, `wall_s` | secondary |
| `lift` | `peak_ratio / peak_ratio(random)` — the cross-dimension quantity. **Defined but not wired in**: no column in `aggregate.csv`. |

`dist_to_needles` is a Hungarian assignment **capped at 0.5**, so every true optimum
a method leaves unclaimed costs it the full cap. That makes it incomparable across
methods that declare different numbers of optima — read it only at matched `|S|`
(`stats.matched_curves`, plotted as `fig6`), never straight from `curves.json`.
See DESIGN.md §22.

Matching is one-to-one because the reference sets contain optima closer together
than `2r` (the 3-D campaign GP has a pair 0.067 apart); without it a single needle
dropped between two of them would "find" both, and a cluster of 50 needles on one
optimum would score like 50 spread out.

`reached` requires a value as well as a position. Proximity alone is what let
random sampling win the old `pct_matched`. The tolerance is a fraction of each
peak's **prominence above the landscape background**, not a fraction of `y`,
because the real campaign surrogates live in a narrow band (0.48–0.88 at 4-D)
where a 4.5%-of-`y` tolerance would cover 40% of the whole range and do nothing.

## Objectives

| name | source | notes |
|---|---|---|
| `{kind: real_gp, dim: 3}` | GP over `data/2nd_real_run.db` (41 lines / 953 rows) | n_true 14 after support + merge |
| `{kind: real_gp, dim: 4}` | GP over `data/3rd_real_run.db` (61 lines / 1358 rows) | n_true 27 |
| `{kind: real_gp, dim: 6}` | GP over `data/4th_real_run.db` (111 lines / 2423 rows) | n_true 68 |
| `{kind: ensemble, dim, n_optima, landscape}` | `synthetic_data.ensemble` | sharp, controllable needle count |

Both real objectives are gitignored data. `data/3rd_real_run.db` is the campaign DB
Colin supplied; `data/2nd_real_run.db` is recoverable from git:

```bash
git show origin/evelyn-compositional:data/2nd_real_run.db > data/2nd_real_run.db
```

**Read the real-campaign scores with care.** Their reference optima are peaks of a
*surrogate* fit to the campaign, not hardware-validated optima, and both
landscapes are shallow — see `frac_peaks_above_random_p99` from
`metrics.landscape_contrast`, reported per objective. The sharp multi-optimum test
is the ensemble suite.

## Protocol

| item | value |
|---|---|
| sample budget `N` | 1000 (~12 h of SDL) |
| batch `q` | 24 — one printed line |
| decisions | `(N − 48) / q`, identical for every method |
| initial design | 2 random chords, printed through the physics model, **byte-identical across methods at a given seed** |
| input noise | calibrated (see below) |
| output noise | multiplicative, `run_mobo.OUTPUT_NOISE_FRAC` = 0.045 |

Baselines get real joint `q`-batch acquisitions (sequential-greedy `qLogEI` /
`qUCB` with `X_pending`), not greedy top-k of a single-point acquisition. Greedy
top-k returns `q` near-identical points and would cripple the baselines — the
mirror image of the mistake the previous benchmark made with ZoMBI-Hop.

### The noise model, and why it is calibrated

ZoMBI-Hop asks for a printed **line**; the core realizes it through
`composition_prediction.physics_simulate_line`, a deterministic model of syringe
ramp lag/overshoot plus junction-volume diffusion mixing. A batch baseline asks for
24 unrelated points, for which no line model applies.

Handing those points `N(0, NOISE_LEVEL = 0.128)` — the obvious choice, and what the
original spec called for — would give the baselines roughly **3× the handicap
ZoMBI-Hop carries**, because `NOISE_LEVEL` was measured on real 6-D hardware
(mean L2 0.271) while the simulator is much gentler:

| d | 3 | 4 | 5 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| physics model, mean L2 | 0.066 | 0.079 | 0.083 | 0.091 | 0.103 | 0.113 |
| `N(0, 0.128)`, mean L2 | 0.222 | 0.256 | 0.286 | 0.314 | 0.362 | 0.405 |

So batch points instead get a perturbation bootstrapped from the *measured*
physics residual distribution at that dimension
(`benchmarks/zhbench/data/input_noise_calibration.json`, regenerate with
`python -m benchmarks.zhbench.protocol --calibrate`). Both sides then face the
same amount of composition error. `input_noise: gaussian` and `none` remain
available as ablations.

### The ten-module ceiling

`physics_simulate_line` raises above **10 components** — the printer has ten
syringe modules. This is a hardware limit, not a software one. Runs above 10-D are
therefore purely computational, have no printer to model, and record
`line_realization: "no_printer_model"`; above 10 components the line gets the same
calibrated point-wise perturbation the baselines get, extrapolated from d=10.

### Batches vs lines

Baselines are given free scattered batches, which they could not print in the real
lab. That is deliberate — Aleks: "we don't really want to be modifying the
benchmark approaches" — and it *handicaps* ZoMBI-Hop, so winning under it is a
strong claim. `input_cost` is what keeps this honest: it measures the physical
price of a scattered batch that the benchmark declines to charge.

## Layout

```
benchmarks/
  zhbench/
    protocol.py          budget, batching, init design, noise model
    objectives.py        objective registry
    metrics.py           the metric set
    optimizers/          random, gp (qLogEI / qUCB), + registry
    zombihop_runner.py   ZoMBI-Hop under a sample budget, no core changes
    runner.py            one (objective, optimizer, seed) -> run dir
    suite.py             a config grid -> aggregate.csv + summary.md
    spaces.py seeding.py salvaged from the old benchmarking branch
    configs/             smoke, s1_sanity, s1_real, s1_real_clean, s2_needles
    stats.py             paired-by-seed statistics -> STATS.md
    data/                hparams/*.json, input_noise_calibration.json
  tests/
```

`docs/DESIGN.md` records what changed from the original spec and why.
