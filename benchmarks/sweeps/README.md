# `benchmarks/sweeps` — how robust is ZoMBI-Hop to the landscape?

`benchmarks/ablations` varies the **optimiser** on one landscape family. This
package does the opposite: it holds the optimiser fixed and sweeps the
**landscape** across a full-factorial grid.

| axis | values |
|---|---|
| number of needles `n` | 2, 10, 30, 50 |
| needle sharpness (basin width) `b` | 2.2, 6, 10, 15 |
| dimensionality `d` | 3, 4, 6, 10 |

4 × 4 × 4 = **64 landscape configurations**, each run on `--n-draws` independent
placements of the optima.

Every cell gets the same **measurement budget: 125 LineBO lines = 3000 measured
compositions** (24 points per line), *not* the same wall-clock. That matters: the
cost of one iteration is dominated by the exact-GP refit and the acquisition
ascent, both of which grow with the accumulated point count and the dimension, so
a wall-clock budget would hand dim 3 several times as many experiments as dim 10
and the "dimensionality" axis would be plotting the GP's cost curve rather than
the landscape.

---

## Quick start

```bash
# 1. Plan. Run this ON THE CLUSTER — the generated sbatch bakes in absolute paths.
python -m benchmarks.sweeps plan --out benchmarks/sweeps/runs/first --n-draws 5

# 2. Submit. Five workers that restart themselves until the queue drains.
sbatch benchmarks/sweeps/runs/first/sweep.sbatch

# 3. Look in whenever. Both work on a partially drained campaign.
python -m benchmarks.sweeps status    --out benchmarks/sweeps/runs/first
python -m benchmarks.sweeps summarize --out benchmarks/sweeps/runs/first
```

`python -m benchmarks.sweeps describe` prints the grid, the separation each
configuration needs and the hyperparameter map, without planning anything.
`python -m benchmarks.sweeps selftest` checks the landscape module's closed-form
identities against constructed `Ensemble` objects.

---

## The landscape: bumps and nothing else

`synthetic_data.ensemble.Ensemble` stacks seven feature families. This sweep turns
off all of them but the true optima — no roughness, no ridges, no plateaus, no
weak-optima distractors, no anisotropy, no edge bias. With the background field
identically zero the objective collapses to a closed form:

```
y(x) = max( 0.5 + 0.5 * E(x), 0.75 ),   E(x) = max_c exp( -b * ||x - c|| / sqrt(d) )
```

so the plain sits at **0.75**, every optimum peaks at exactly **1.0**, and a
basin meets the plain at radius `sqrt(d) * ln2 / b`. `selftest` verifies each of
those numerically rather than asking you to trust the algebra.

### What counts as a resolvable needle

`METHODS.md` §1 defines the target set as the local maximisers "whose basins are
resolvable above the noise — wider than σ_x and more prominent than σ_y". Those
are two separate conditions, and this sweep enforces **both**, as a minimum
pairwise separation `s*` between optima.

**1. Wider than the input noise.** `s ≥ σ_x = 0.128` (`run_mobo.NOISE_LEVEL`,
measured on the deposition system). Closer than this and the apparatus cannot be
*asked* for one optimum rather than its neighbour. It is also the exact test
`Ensemble._tag_true_optima` applies — so meeting it makes the paring a no-op and
the advertised count is exactly `n` by construction, which `build_landscape`
asserts on every cell.

**2. More prominent than the output noise.** The saddle between two adjacent peaks
has to dip more than σ_y below them, or the pair reads as one broad hill under
metrology noise however far apart the tips are. Two peaks at separation `s` put
their saddle at the midpoint, where `E = exp(-b·s/(2·sqrt(d)))`, so

```
prominence = 1.0 - max(0.5 + 0.5·E, 0.75)
```

and the simulator's noise is multiplicative — `σ_y = 0.045·|y|`
(`run_mobo.OUTPUT_NOISE_FRAC`), i.e. `0.045` at a peak. Requiring
`prominence ≥ σ_y` gives

```
s ≥ s_prom(b, d) = -2·ln(1 - 2σ_y)·sqrt(d)/b  ≈  0.1886·sqrt(d)/b
```

**The target is `s* = max(σ_x, s_prom(b, d))`.** Condition 2 binds only at the
broadest sharpness in the sweep: at `b ≥ 6` a basin is narrow enough that σ_x is
the stricter of the two at every dimension, so 48 of the 64 configurations place at
a plain 0.128 and the 16 at `b = 2.2` spread out (0.149 in dim 3 up to 0.271 in
dim 10).

The pairwise rule is *necessary* but not quite sufficient — the field is a max over
all basins, so a third peak near a pair can only lift their saddle. So the
prominence is also **measured** on the built landscape: `prominence_report` walks
the segment from each optimum to its nearest neighbour, takes the true minimum
along it, and records how many optima clear σ_y. Across all 64 configurations that
count comes out at `n`, and the tightest cells land at a measured minimum
prominence of 0.0459 against the 0.045 threshold.

### Why the optima are placed rather than drawn

Stock uniform placement does not give a clean count axis. Because `Ensemble` pares
optima that land within `input_noise` of an already-tagged one, a request for 50
optima advertises about **21** in dim 3, **33** in dim 4, **45** in dim 6 and
**50** in dim 10 — the axis would collapse, and collapse *differently per
dimension*, confounding the two axes the sweep exists to separate.

So the centers are placed here and handed to `Ensemble(pinned_optima=…)` with
`n_optima=0`. Placement is dart-throwing first (the least structured way to hit a
separation constraint — the points stay an honest uniform sample conditioned on it)
and, where that saturates short, a repulsion relaxation that pushes overlapping
pairs apart and re-projects onto the simplex. The relaxation is what reaches counts
dart-throwing cannot: 50 optima in dim 3 at 0.128, which random sequential
adsorption saturates around 45.

**The cost, stated plainly:** near the packing limit the result approaches a
lattice rather than a random scatter, and neighbouring basins overlap heavily. In
dim 3 at `n = 50` the optima sit ~0.15 apart while a `b = 2.2` basin has a plain
radius of 0.55 — that cell is a ridged mesa with 50 resolvable tips, not 50
isolated needles. That is what "50 needles in a triangle" *has* to mean, not a bug,
but read that corner of the heatmap knowing it. Every cell records
`separation_achieved`, `separation_target`, `separation_own_target`,
`prominence_target_met` and `n_prominence_resolved` in `sweep_cell.json`, and
`basin_plain_radius` lands in `summary/cells.csv` next to them.

### One placement per row, shared across the sharpness axis

`s_prom` goes as `1/b`, so placing each cell at its own `s*` would let the
sharpness axis change the *layout* as well as the basin shape — and it splits the
axis unevenly: `b = 2.2` places at `s_prom` (0.1485 in dim 3 up to 0.2711 in dim
10) while `b = 6, 10, 15` all place at the plain 0.128 floor. The broadest column,
which is the one the sharp columns get read against, would be the only one laid out
differently.

So a whole `(dim, n_needles, draw)` row is placed **once** — at the strictest
width's target (`needles.placement_width`) — and all four sharpnesses reuse those
exact centers. `NeedleFactory.placement_seed` leaves `basin_width` out of its hash,
so they share a seed as well as a separation. Two consequences worth having:

* the sharpness axis varies sharpness alone, so a difference along it is
  attributable to the basin shape;
* the comparison is **paired** — the layout cancels between two widths at the same
  draw, instead of being one more source of variance the draws have to average out.
  This is what makes small effects decidable at five draws rather than twenty.

The strictest width is the safe standard: it is the only one no column has to be
relaxed below, so no cell ends up packed tighter than its own prominence rule
wanted. Each cell still records `separation_own_target` — what it would have asked
for alone — next to the shared `separation_target`, and `describe` prints both.

Note that this makes `placement_seed` a function of `(dim, n_needles, draw)` only.
Re-running a campaign planned before this change will build **different**
landscapes than it originally did; `runs/first` was placed per-cell and its stored
artifacts remain the record of what it actually ran. The optimiser's own seeding
(`campaign._cell_seed_base`) is still per-cell, so the search is independently
randomised across the axis even though the landscape is not.

---

## Hyperparameters

Held fixed within a dimension, so a difference between two cells at the same `d` is
attributable to `n` and `b` and nothing else.

| dim | config | provenance |
|---|---|---|
| 3 | `optimize/hparams/trial_112_composition.json` | archived 3d MOBO winner (`mobo_3d_05_06_15_32` trial 112), re-expressed for composition space. Seeds `ensemble_mobo_3d.sbatch` and `ensemble_mobo_4d.sbatch`, and is where `warm_start`'s `REFERENCE_HPARAMS` comes from. |
| 4 | `optimize/hparams/clamped_6d/dist1c.json` | **stand-in** — no tuned 4d config exists in the repo. |
| 6 | `optimize/hparams/clamped_6d/dist1c.json` | best `dist_to_needles` trial of the 6d ensemble pool (`mobo_ensemble_6d_job19202380` trial 23), clamped into `HPARAM_SPACE`. |
| 10 | `optimize/hparams/clamped_6d/dist1c.json` | **stand-in** — `optimize/hparams/10d_ensemble.json` records `"phase": "sobol"`, i.e. trial 3 of the initial quasi-random sweep, not a tuned winner. |

Across dimensions the configuration therefore changes, so **dim-to-dim differences
include the hyperparameters, not only the landscape**. The stand-ins are flagged in
the manifest, starred in the heatmap panel titles and tabulated in `summary/index.md`
so this never has to be remembered. Override with `--hparams 4=path/to/config.json`
(repeatable).

`optimize/hparams/tight_6d/` holds the same 6d configurations re-projected into the
`HPARAM_SPACE` re-tightened on 2026-08-12, and `ensemble_mobo_10d.sbatch` warns
against seeding a *search* from the older `clamped_6d/` files for that reason. It
does not apply here: this sweep re-evaluates a fixed configuration rather than
seeding a search, so no coordinate is mapped through the space's bounds and
`dist1c.json` runs as the numbers it literally contains.

---

## What a campaign produces

```
runs/first/
├── manifest.json                  the complete plan: grid, budget, hyperparameters,
│                                  per-configuration feasibility
├── tasks.tsv                      the queue, one line per cell
├── claims/                        atomic mkdir claims, heartbeated (see below)
├── sweep.sbatch                   self-restarting SLURM array
├── runs/d03_n50_b2.2/
│   ├── run_config.json            so the dim-3 coverage plot resolves its landscape
│   └── draw001/                   ONE CELL — the full run_mobo artifact set, plus:
│       ├── metrics.json           the run's scalar results (shared runner)
│       ├── arm.json               hyperparameters, landscape spec, seed
│       └── sweep_cell.json        THIS sweep's record: grid coordinates, verified
│                                  landscape (separation, measured prominence),
│                                  budget accounting, metrics
└── summary/
    ├── index.md                   headline, hyperparameter table, every figure
    ├── cells.csv                  one row per finished cell
    ├── grid.csv                   per-configuration means with bootstrap intervals
    ├── recall_heatmap.png         n x b, one panel per dim  <- the headline
    ├── recall_by_axis.png         main effects, marginalised
    ├── dist_to_needles_heatmap.png
    ├── n_needles_heatmap.png
    └── dup_fraction_heatmap.png
```

Every cell routes through `benchmarks.ablations.runner.run_ablation_trial` on the
unmodified baseline arm, which routes through `run_mobo.run_single_trial` — so a
cell directory is interchangeable with a MOBO trial directory and `plot_metrics`,
`coverage_plot` and `pareto` read it unchanged.

### Metrics

`recall` is the headline: the fraction of a landscape's true optima that got a
declared needle within `MATCH_RADIUS`. The deliverable is the *set* of optima, so
the first thing to know about a landscape is how much of it was found — and because
the placement guarantees the denominator is exactly `n`, recall is directly
comparable across every cell in the grid. `dist_to_needles` sits beside it because
it is sensitive to *how badly* a miss missed, which recall is not; `n_needles`
separates "found few" from "declared few"; `dup_fraction` says whether a landscape
drives the optimiser into re-measuring.

Confidence intervals are percentile bootstraps over **draws**, so a band answers
"what if we drew another set of optima placements". With five draws the interval is
wide by construction — it is there to stop a one-draw fluke being read as a trend,
not to make a significance claim.

---

## Running unattended

The campaign is built to survive multi-day wall-clock without anyone touching it.

**The pool restarts itself.** A worker stops claiming with less than one cell's
ceiling of wall-time left and exits cleanly; the tail of the sbatch then resubmits
*that array index* if the queue still has work, and submits nothing once it does
not — so the chain ends on its own when the campaign finishes. If wall-time arrives
anyway, SLURM sends `USR1` 300 s early and the trap resubmits before the kill. A
crashed worker resubmits too: one bad cell should not end a 320-cell campaign.

**Claims heal themselves.** A worker touches its claim's `heartbeat` file once a
minute on a daemon thread. A worker killed outright — node failure, OOM, a SIGKILL
past the grace period — leaves a claim that stops beating, and the next worker to
walk past it releases it automatically after `--reclaim-after-min` (default 30).
This is what removes the `reset-stale`-between-submissions step that
`optimize/showdown.py` and `benchmarks/ablations` both require, and it is safe to
do while other workers are live *because* it keys on the heartbeat: a claim silent
for half an hour is not one somebody is working on. Two workers racing to release
the same claim is harmless — the `mkdir` that follows is still atomic.

`reset-stale` is still there for the case where you want the queue reopened
immediately, and `reset-stale --all` releases every unfinished claim regardless of
heartbeat (only safe with no workers running).

**Cells are skipped, not repeated.** A cell is complete when it has both
`metrics.json` and `sweep_cell.json`; one interrupted between the two is re-run
rather than counted with half a record.

**Draw-major queue order.** Tasks are ordered `(draw, dim, n, b)`, so a campaign
cut short has every one of the 64 configurations at draw 1 rather than all five
draws of the first few and nothing for the rest. A partial sweep is a complete
picture of the grid, just a noisier one — and workers start at rotated offsets so
the pool spreads across the grid instead of grinding through one region.

### Budget vs ceiling

`--n-lines` (default 125) is the **budget**; `--cell-max-hours` (default 6) is a
**safety valve** so one pathological cell cannot hold a worker indefinitely. A cell
stopped by the ceiling rather than the budget records `budget_hit: false` and is
listed in `summary/index.md` under "Cells that did not spend their budget" — its
metrics are not comparable to a full-budget cell on equal terms. If that table is
ever long, raise the ceiling rather than reading around it.

---

## Known rough edges

- **CoNet renders fail on dim-3 cells.** `visualization/plot_10d.py` looks for
  `x0..xk` composition columns but dim 3 writes `FA`/`MA`/`Br`. Pre-existing
  `run_mobo` behaviour, caught and logged, affects no metric — the cell just has no
  `conet*.png`. Same rough edge the ablations harness documents.
- **Plan on the cluster.** The generated sbatch bakes in absolute paths, so
  planning from a Windows checkout emits Windows paths. (The script itself is
  written ASCII-only with LF endings, so a plan committed from Windows is at least
  not corrupt.)
- **Placement costs seconds, twice.** A cell builds its landscape once for
  verification and once inside the runner; at dim 3 / `n = 50` the relaxation takes
  ~9 s, so ~20 s per cell. Against a multi-hour cell that is noise, but it is why
  `plan` is not instant on a large grid.
