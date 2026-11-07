# `benchmarks/sweeps` — ZoMBI-Hop vs. the baselines, across needle landscapes

A full-factorial sweep of **methods × landscapes**:

| axis | default values |
|---|---|
| method | `zombi_hop`, `random`, `gp_bo`, `turbo`, `hebo` (see `benchmarks/methods`) |
| number of needles `n` | 2, 10, 30, 50 |
| needle sharpness (basin width) `b` | 2.2, 6, 10, 15 |
| dimensionality `d` | 2, 3, 5, 9 — the 3/4/6/10-component simplex in free dimensions |

5 methods × 64 landscape configurations × `--n-draws` placements. Landscapes are
**bumps-only needles on the unit cube** (`needles.py`), and every method gets the
**byte-identical landscape and the same measurement-noise stream** for a given
`(d, n, b, draw)`, so method-vs-method differences are paired.

Every cell's **measurement budget is 100 × dim points, one point per call** (200 at
2d, 300 at 3d, 500 at 5d, 900 at 9d; `--budget-per-dim` changes the multiplier and
`--budget N` fixes one number for every dim), the same for every method at a given
dim and enforced by one shared `benchmarks.methods.Problem`. Every method is fully
sequential: ZoMBI-Hop measures the single candidate it proposes (`sampling="point"`,
no LineBO lines), and the baselines run at q = 1. All start with a 48-point Sobol'
initial design. ZoMBI-Hop's hyperparameters are used as tuned, so the ones that used
to count lines now count points; [`POINTWISE.md`](POINTWISE.md) lists every change and why. A wall-clock budget would hand fast methods and low
dimensions more experiments, so the budget is points, the quantity that costs money
on real hardware. `--cell-max-hours` is only a safety ceiling.

> **Changed 2026-09-26.** This package used to sweep ZoMBI-Hop alone on the
> *simplex*, with a line budget patched into `run_mobo`. It is now cube-only and
> multi-method. Campaigns planned by the old code (`runs/first`, `runs/second`)
> keep their `summary/`; the new code refuses them with a pointer to commit
> `285424f`, which can still re-summarise them.

---

## Quick start

```bash
python -m benchmarks.methods install-hebo              # once per checkout

# 1. Plan (writes files only; fine on the login node). Run on the cluster —
#    the generated sbatch bakes in absolute paths.
python -m benchmarks.sweeps plan --out benchmarks/sweeps/runs/full --n-draws 10

# 2. Submit: self-restarting workers, each cell in its own process.
sbatch benchmarks/sweeps/runs/full/sweep.sbatch

# 3. Look in whenever; both work on a partially drained campaign.
python -m benchmarks.sweeps status    --out benchmarks/sweeps/runs/full
python -m benchmarks.sweeps summarize --out benchmarks/sweeps/runs/full
```

`describe` prints the grid, the separation each configuration needs and every
method's resolved configuration without planning anything; `selftest` checks the
landscape's closed-form identities. Neither fits a model.

**Nothing that fits a model runs on the login node.**
`benchmarks/scripts/validate_methods.sbatch` is the template for a bounded
validation job (extractor calibration, method smoke test, and a small timing
campaign).

### Choosing methods and configs

```bash
--methods zombi_hop,turbo,path/to/mine.py:MyMethod   # any registered name or a ref
--method-set turbo.n_trust_regions=5                 # one key, value parsed as JSON
--method-config gp_bo=configs/gp_ucb.json            # a JSON object of keys
--hparams 9=optimize/hparams/10d_ensemble.json       # zombi_hop's per-dim file
```

Configs are validated when you plan (a typo in a key stops the plan) and the
manifest stores the full merged config for every (method, dim).

---

## The landscape: bumps on the unit cube

`CartesianEnsemble` with every feature family off but the true optima:

```
y(x) = max( 0.5 + 0.5 * E(x), 0.75 ),   E(x) = max_c exp( -b * ||x - c|| / sqrt(d) )
```

The plain sits at **0.75**, every optimum peaks at exactly **1.0**, and a basin
meets the plain at radius `sqrt(d) * ln2 / b`. `selftest` checks each of these
numerically.

**Why the cube.** The baselines are box-domain methods. On the simplex each would
need a reparameterisation, and the comparison would partly measure that. On the
cube every method runs natively, and ZoMBI-Hop runs through `BoxDomain`
(`src/utils/domain.py`).

**Resolvable needles.** Optima are *placed*, not drawn, at a minimum pairwise
separation `s* = max(sigma_x, s_prom(b, d))`. The first term is the input-noise
floor, 0.128 (also `Ensemble`'s paring distance, so the advertised count is exactly
`n`). The second, `s_prom = -2·ln(1 - 2σ_y)·sqrt(d)/b`, makes the saddle between
two adjacent needles dip at least one output-noise sd (0.045) below them. The
prominence is also *measured* on every built landscape and recorded. The cube has
far more room than the simplex (a 3-cube holds ~900 points at 0.128 where the
3-simplex held ~60), so every default configuration fits at its prominence target;
the lattice-packed dim-3 / n = 50 corner of the old simplex sweep does not arise.

## Configurations

`zombi_hop` runs the per-dimension map in `hparams.py` (MOBO winners, two
labelled stand-ins). **Those were tuned on the simplex.** ZoMBI-Hop's length scales
are fractions of a unit-extent domain on both, so they transfer, but they are not
cube-tuned. The baselines run their published defaults. Neither side was tuned on
these landscapes; `summary/index.md` says so next to the results.

## Scoring

See `benchmarks/methods/README.md`. Briefly: `dist_to_needles` scores each
method's own declared needles (ZoMBI-Hop's, or the `gp_peaks` extractor's for the
baselines); `dist_to_needles_extracted` applies the same extractor to every
method's samples; `frac_optima_visited` asks whether a method ever measured near
each needle.

## What a campaign produces

```
runs/<campaign>/
├── manifest.json        grid, methods (+ import refs), full per-(method, dim)
│                        configs, budget, noise, extractor, feasibility
├── tasks.tsv            the queue: tid, method, landscape, dim, n, b, draw
├── claims/              atomic mkdir claims, heartbeated; FAILED marks give-ups
├── logs/                worker logs, fail_<tid>.log, attempts_<tid>
├── sweep.sbatch         self-restarting SLURM array
├── runs/<method>/d03_n10_b6/draw001/
│   ├── points.csv, needles.csv, metrics_over_time.csv, method.json, metrics.json
│   ├── ensemble_config.json   the exact landscape (Ensemble(**config) rebuilds it)
│   └── sweep_cell.json        this sweep's record; written last = completion marker
└── summary/
    ├── index.md               headline table, paired-vs-zombi_hop table, figures
    ├── cells.csv, grid.csv, methods.csv, paired.csv
    ├── method_by_dim.png      each metric vs dim, one line per method  <- headline
    ├── <metric>_heatmap.png   rows = method, columns = dim, tile = n x b
    ├── dist_over_time.png     panel per dim, one line per method
    ├── dist_over_time_all.png every cell's trajectory, panel per method
    ├── dist_over_time_by_axis.png  row per method, column per swept axis (d, n, b);
    │                               x = fraction of budget, since budgets differ by dim
    └── dist_over_time_grid.png     row per dim, column per n, one line per method
```

Paired comparisons use the landscape as the unit: for each `(d, n, b, draw)`
where both a method and `zombi_hop` finished, the difference and who won.
Bootstrap intervals resample landscapes.

---

## Running unattended

- **One process per cell.** A worker runs each cell as
  `python -m benchmarks.sweeps cell --tid …` under a hard timeout (ceiling +
  `--cell-margin-hours`). Importing ZoMBI-Hop changes torch's global default
  device and dtype, and HEBO adds a vendored path; none of that may leak into the
  next method's cell.
- **The pool restarts itself.** Workers stop claiming when a cell's ceiling no
  longer fits and resubmit their array index while work remains; SLURM's `USR1`
  300 s before the wall-time does the same.
- **Claims heal themselves.** Heartbeat once a minute; a claim silent for
  `--reclaim-after-min` (30) is released by the next worker.
- **Bounded retries.** A cell that fails `--max-attempts` (3) times is marked
  FAILED, excluded from the pending count (so the chain can end), and listed by
  `status`. `reset-stale --failed` re-opens it after a fix.
- **Draw-major, landscape-grouped queue.** Order is `(draw, d, n, b, method)`, so
  a campaign cut short still has every configuration at draw 1, with every method
  on each landscape it reached.
- **Budget vs ceiling.** A cell stopped by `--cell-max-hours` records
  `budget_hit: false` and is listed in `summary/index.md`; its scores are not
  comparable on equal terms. Raise the ceiling rather than reading around it.
