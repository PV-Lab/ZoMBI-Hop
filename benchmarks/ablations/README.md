# `benchmarks/ablations` — ZoMBI-Hop component ablations

Four questions about which parts of ZoMBI-Hop are doing the work:

| | Question | Baseline | Variant arm |
|---|---|---|---|
| **A1** | Do *k* independent ZoMBI restarts match ZoMBI-Hop on the same budget? | `zombi_hop` | `k_restarts` |
| **A2** | Does the contracting trust region find optima the global search misses? | `zombi_hop` | `no_zoom` |
| **A3** | Does acquisition-ranked line selection beat a random chord through the same candidate? | `zombi_hop` | `random_chords` |
| **A4** | Does shaping each needle's basin to the local curvature beat a volume-matched sphere? | `zombi_hop` | `isotropic_basins` |

This is an **API, not a fixed experiment**. Nothing here is tied to a dataset or a
dimension: a campaign names a landscape factory and a per-cell budget, and the arms,
the artifacts and the statistics are the same whatever it points at. Swap in the real
dataset when it exists — see [Bringing your own landscape](#bringing-your-own-landscape)
— without touching any other file.

---

## Quick start

```bash
# 1. Plan: writes the manifest, the task queue and a SLURM array script.
python -m benchmarks.ablations plan \
    --out benchmarks/ablations/runs/first \
    --dim 6 --n-landscapes 5 --n-repeats 3 --time-limit-min 30

# 2. Drain it — locally…
python -m benchmarks.ablations run --out benchmarks/ablations/runs/first --device cuda
#    …or as a pool of workers on the cluster.
sbatch benchmarks/ablations/runs/first/ablations.sbatch

# 3. Progress, requeue after killed workers, summarise.
python -m benchmarks.ablations status      --out benchmarks/ablations/runs/first
python -m benchmarks.ablations reset-stale --out benchmarks/ablations/runs/first
python -m benchmarks.ablations summarize   --out benchmarks/ablations/runs/first
```

`python -m benchmarks.ablations describe` prints the arm registry — what each variant
changes, which hyperparameters it overrides and which patches it applies — without
running anything.

`summarize` works on a partially drained campaign, so you can look at the figures
while the rest is still running.

---

## What a campaign produces

```
runs/first/
├── manifest.json              the complete plan: arms, landscape, budget, hyperparameters
├── tasks.tsv                  the queue, one line per cell
├── claims/                    atomic mkdir claims (see Resumability)
├── ablations.sbatch           SLURM array of persistent workers
├── runs/<arm>/
│   ├── run_config.json        so the dim-3 coverage plot resolves its landscape
│   └── ls00000_r001/          ONE CELL — the full run_mobo artifact set, below
└── summary/
    ├── index.md               headline table + every figure, across all ablations
    ├── cells.csv              one row per finished cell
    ├── headlines.json         the paired results, machine-readable
    └── A1/ … A4/
        ├── dist_to_needles_over_time.png
        ├── dup_fraction_over_time.png
        ├── curves_dist_to_needles.csv     the plotted numbers + n_active
        ├── curves_dup_fraction.csv
        ├── summary.csv                    per-arm end-of-run metrics with CIs
        ├── paired_<metric>_<variant>.csv  per-cell paired deltas
        └── summary.md
```

### Per cell — the same artifacts `run_mobo.py` writes

Every arm routes through `run_mobo.run_single_trial`, so a cell directory is
interchangeable with a MOBO trial directory and the existing tooling
(`plot_metrics`, `coverage_plot`, `pareto`) reads it unchanged:

| File | What it holds |
|---|---|
| `points.csv` | every sample: composition, `Y`, `penalized`, `activation`, `zoom` — plus a `zoom_size` column this harness adds |
| `needles.csv` | every declared needle, with the activation/iteration it was declared on |
| `metrics_over_time.csv` | `dist_to_needles` / `dup_fraction` / `n_needles` / … per LineBO line |
| `metrics_over_time.png`, `needle_values.png` | the plots of the above |
| `convergence.png` | all `Y`, running best (reset per activation), needle vlines |
| `dist_from_centre.png` | `Y` vs distance from the simplex centroid |
| `line_length_hist.png` | LineBO main-line length distribution |
| `coverage.png` | ternary coverage — dim 3 only |
| `point_cloud.html` | interactive simplex cloud — dim 4 only |
| `conet*.png` | co-occurrence network renders — ensemble landscapes |
| `ensemble_config.json`, `coverage_ground_truth.npz` | the exact landscape this cell ran on |

plus two files this harness adds:

- **`metrics.json`** — the cell's scalar results, under the same names
  `pareto.py`/`summary_table.py` use. Written **last and only on success**, so its
  presence is the completion marker the queue resumes from.
- **`arm.json`** — arm definition, exact hyperparameters, landscape spec and seed. A
  cell is reproducible from this file alone.

The `k_restarts` arm additionally keeps each restart's own full artifact set in
`restart_0/`, `restart_1/`, … alongside `restarts.json`, and the cell-level artifacts
are the **pooled** versions (see [A1](#a1--k-independent-restarts) below).

### Per ablation — the summary figures

`dist_to_needles_over_time.png` and `dup_fraction_over_time.png` show, for each arm,
the mean across every (landscape, repeat) cell with a 95 % bootstrap confidence band.
One metric per figure, never two y-scales on one plot. The tick where the first cell
in the comparison ended is marked, because past it some curves are being held (below).

---

## How the comparison is made fair

**Shared landscapes.** Every arm runs on the same landscape indices. A difference
between two arms should be attributable to the arms, not to one drawing an easier
surface.

**Common random numbers.** The RNG is seeded from `(landscape_index, repeat)` and
deliberately *not* from the arm, so every arm starts a given cell from the identical
initial design. The arms diverge immediately after — they consume the stream at
different rates — which is the intent: shared start, independent thereafter.

**Paired statistics.** `summary.md` matches cells on `(landscape, repeat)` and reports
the mean paired delta with a bootstrap CI and a two-sided sign-flip permutation
p-value. Because a matched pair shares its seed, the difference is measured *within* a
cell, which is where nearly all the statistical power in a campaign this size comes
from. Both metrics are minimised, so **a negative delta favours the variant**.

**One shared baseline.** `zombi_hop` appears in all four ablations, so it is queued
once and read into all four figures — running it per-ablation would spend a quarter of
the campaign re-measuring the same arm.

**Runs that end early are held, not dropped.** Cells are wall-clock budgeted, so they
end at different iteration counts. Dropping a finished cell from the average would
make the tail of a curve a different population from its head — arms that finish early
would silently leave the comparison and the curve would bend for reasons that have
nothing to do with the optimiser. Instead a finished cell's last value is carried
forward (it really did end there, with that score), the first such tick is marked on
the figure, and `n_active` in `curves_*.csv` gives the exact count at every iteration.

**Bootstrap CI of the mean, not a spread.** `dist_to_needles` is bounded and skewed, so
a symmetric mean ± s.d. band would run outside the metric's own range. The percentile
bootstrap resamples *cells*, so the band answers "what if we drew another set of
landscapes and repeats?" — the question a reader actually has. Median/p25/p75 are in
the CSVs for anyone who wants the spread.

---

## How each arm is implemented

The variants are **monkeypatches applied around a trial**, not flags inside
`src/core/`. An ablation is a claim about the *published* optimiser, so the harness's
code must not sit on the baseline's execution path: with no arm active, nothing in
`arms.py` runs and `src/core` is byte-identical. Each patch is a context manager that
restores the original on exit, so arms can run back-to-back in one process.

### A1 — `k_restarts`

The budget is split across *k* independent ZoMBI runs. "Independent" has to mean a
restart inherits *nothing* — not the measured points, the GP posterior, the needle
penalties or the trust region — which rules out reusing one `ZoMBIHop` with its history
cleared. So each restart is a separate `run_single_trial` with its own optimiser and
its own empty `DataHandler`. Each is capped at `--activations-per-restart` activations
(default 1), which is what makes it plain ZoMBI rather than a short ZoMBI-Hop: zoom in,
converge, declare one needle, stop.

The restarts are then **pooled** into one cell, because `dist_to_needles` scores a *set*
of discovered optima and `dup_fraction` scores a *pooled* sample cloud — reporting
restarts separately would answer "how good is one short ZoMBI run?", not the question
A1 asks. `metrics_over_time.csv` is recomputed rather than concatenated: at merged
iteration *t* the discovered set is (all needles from finished restarts) ∪ (this
restart's needles so far).

Two budget details keep it honest:

- `--no-fill-budget` turns off the default behaviour of launching extra restarts when
  the planned ones finish early. Leave it on unless you specifically want the arm to
  spend less than the baseline.
- The budget is counted in **optimiser time**, not wall-clock. Each restart renders a
  full artifact set when it finishes; charging that to the budget would starve the
  later restarts while the baseline, which renders once, paid nothing.

### A2 — `no_zoom`

`max_zooms=1` stops the zoom loop advancing, and `min_zoom_for_needle=0` re-permits
needle declaration at zoom 0 — without it the default of 1 makes needles unreachable
and the arm would measure "no needles", not "no zooming". A patch additionally neuters
`DataHandler.determine_new_bounds`, which `_handle_failure_retry` calls directly; that
failure path is common, so without the patch the arm would still spend much of an
activation inside a contracted box.

### A3 — `random_chords`

`LineBO.ranked_line_endpoints` is replaced. Every chord passes through `x_tell` (the
point the GP actually proposed) in a uniformly random direction, and the order is a
random permutation, so the acquisition has no say in orientation. Anchoring at
`x_tell` is what makes this an interesting control rather than a strictly worse one:
the arm still measures through the GP's chosen point, so a *small* gap to the baseline
says line orientation is doing little work and the candidate is carrying the search.

### A4 — `isotropic_basins`

Each needle's Hessian ellipsoid is replaced by the sphere of **equal volume** — every
eigenvalue of the tangent-space precision `M` swapped for their geometric mean, which
leaves `det(M)` and therefore the enclosed volume exactly unchanged. An arithmetic
mean, or the largest or smallest semi-axis, would confound shape with size and turn A4
into an accidental test of penalty radius. Patching
`GPSimplex.determine_penalty_ellipsoid` covers all three sites that mint an `M` (live
declaration, capped-activation exclusion zones, and the wholesale refit), and
`shrink_all_needle_radii` only rescales by a scalar, so nothing reintroduces anisotropy
later.

---

## Bringing your own landscape

`--landscape` takes a built-in name (`ensemble`, `synthetic`, `warmgp`, `fullgp`) or
`module:<path.py or dotted.module>:<attr>`. The named attribute is called with the
`--landscape-arg` values and must return anything satisfying `LandscapeFactory`:

```python
# my_landscapes.py
from dataclasses import dataclass

@dataclass
class MyFactory:
    dim: int
    time_limit_hours: float | None = 0.5
    kind: str = "my_dataset"
    n_available: int | None = 12        # None = unbounded

    def spec(self) -> dict:             # recorded in the manifest
        return {"kind": self.kind, "dim": self.dim}

    def build(self, index: int):        # -> (LandscapeSpec, ensemble_config | None)
        return make_my_landscape(index, self.dim, self.time_limit_hours), None

def build(dim: int = 6, time_limit_hours: float | None = 0.5, **kw):
    return MyFactory(dim=dim, time_limit_hours=time_limit_hours, **kw)
```

```bash
python -m benchmarks.ablations plan --out runs/mine \
    --landscape module:my_landscapes.py:build --dim 6 --n-landscapes 12
```

`n_available` matters: a fixed surface (`warmgp`, `fullgp`) reports 1, and `plan`
refuses to schedule N "different" landscapes that are all the same surface — that
would tighten the confidence bands without adding evidence. Use `--n-repeats` there
instead.

---

## Library use

```python
from benchmarks.ablations import ARMS, resolve_landscape, run_ablation_trial

factory = resolve_landscape("ensemble", dim=6, time_limit_hours=0.5)
metrics = run_ablation_trial(
    arm="no_zoom", factory=factory,
    landscape_index=0, repeat=1,
    trial_dir="scratch/one_cell", device="cuda",
)
```

Adding a fifth arm is a `Arm(...)` entry in `arms.py`'s registry plus, if it needs one,
a patch in `PATCHES`. An arm is fully reconstructible from its name — a worker gets
only the name off the queue, so nothing may live in a caller's closure. Note that the
figures cap at **three** arms per ablation: the categorical palette's first three slots
are validated colourblind-safe on all pairs, and a fourth would not be. Split into
separate figures rather than adding a hue.

---

## Resumability

`metrics.json` is a cell's completion marker and `claims/<tid>` (an atomic `mkdir` —
the only primitive that is reliably atomic on a shared filesystem) is its lock. Re-running
`run` skips finished cells, so a campaign can be drained across several submissions.

A worker killed by wall-time, a node failure or an OOM leaves a claim with no result,
and nothing clears it on its own — so the cell would be invisible to every later
submission while the campaign looked drained. `reset-stale` releases those claims.
Run it only when no workers are live: a claim on a *running* cell is indistinguishable
from an abandoned one.

Tasks are queued repeat-major (`(repeat, landscape, arm)`), so a campaign cut short has
every arm on every landscape at repeat 1 rather than all repeats of the first arm and
nothing for the last — a partial campaign is still a fair comparison, just a noisier
one. Workers start at rotated offsets in the queue for the same reason.

---

## Known rough edges

- **CoNet renders fail on dim-3 ensemble cells.** `visualization/plot_10d.py` looks for
  `x0..xk` composition columns, but dim 3 writes `FA`/`MA`/`Br`. This is pre-existing
  `run_mobo` behaviour, it is caught and logged, and it does not affect any metric —
  the cell just has no `conet*.png`.
- **Budgets are wall-clock, so cell lengths vary.** That is why curves are held rather
  than truncated; see above. If you need iteration-matched comparisons instead, cut the
  curves at `n_active == n_cells` using the `curves_*.csv` columns.
- **`k_restarts` needs a budget several times an iteration.** `ZoMBIHop.run` only
  checks its time limit at iteration boundaries, so a restart overruns its slice by up
  to one LineBO line — and *k* restarts can overrun by up to *k* lines where the
  baseline overruns by one. The optimiser-time accounting deducts each overrun from the
  remaining slices, so the cell total still lands near the budget, but at a budget of a
  few iterations the later restarts get squeezed. Keep `--time-limit-min` at least an
  order of magnitude above one iteration's cost (with default `--n-restarts 4`, 30 min
  is comfortable; 1 min is not).
