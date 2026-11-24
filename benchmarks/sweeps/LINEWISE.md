# Line-wise sampling in the sweeps (2026-11-12)

`plan --sampling line` is the counterpart of [`POINTWISE.md`](POINTWISE.md). In
point mode, every method measures one point per call, and no method uses LineBO.
In line mode, **every method measures LineBO lines**: one call = one chord of
`--batch-size` (default 24) evenly spaced points. Each method keeps its own model
and acquisition; only how a candidate becomes a measurement changes.
Point mode is still the default, and nothing about it changed (apart from the
zombi_hop needle-declaration switch below, which applies to every sweep).

```bash
python -m benchmarks.sweeps plan --out benchmarks/sweeps/runs/<name> \
    --sampling line --lines-per-dim 2=10,3=20,5=125 ...
```

`--lines-per-dim` sets the budget in lines per dimension, init lines included
(`budget = lines x batch_size`); `--budget` / `--budget-per-dim` still work.

## The shared line machinery (`benchmarks/methods/_lines.py`)

All of it uses ZoMBI-Hop's `LineBO` geometry on `BoxDomain()`:

| piece | what it is |
|---|---|
| initial design | `n_init_lines` (2) random chords through the unit box, drawn the way zombi_hop line mode always drew them. Every method on a cell has the same seed, so **every method measures the identical init lines**. |
| chord | uniform anchor in a box, isotropic direction, extended to the box faces (`random_chords_through_simplex` with `BoxDomain`) |
| one step | draw `linebo_num_lines` (10) chords, score each on `linebo_points_per_line` (100) points with the method's own acquisition, measure the best at `batch_size` points |

The 10 × 100 scoring points are model queries, not measurements. Only the measured
chord counts against the budget, the same for every method. The geometry is
re-stated in numpy rather than imported, because importing `src.core` switches
torch's global default device and dtype.

## Per method

| method | how a chord is scored | what is not used any more |
|---|---|---|
| zombi_hop | unchanged: its own `LineBO` (mean UCB over the chord, penalty-zone trimming) inside the current zoom box | — |
| gp_bo | same SingleTaskGP + LogNEI; chord score = **mean EI** along the chord (`logmeanexp` of LogNEI, so EI is averaged, not log-EI) | `optimize_acqf` |
| turbo | same region GP; chords through the region's ARD-shaped box; **one joint Thompson sample** over all 1000 chord points, chord score = its mean | the coordinate-perturbation candidate set |
| hebo | HEBO's own `suggest` preamble (power transform, GP, `best_y`, kappa) builds MACE. Each objective is averaged along the chord (LCB directly, EI and PI as EI and PI). The **non-dominated chords** are kept, and one is picked at random, as HEBO picks from its NSGA-II front | the NSGA-II acquisition optimiser |
| random | one uniform random chord per call, unranked (`sampler` ignored) | — |

Step-counting choices:

- **TuRBO:** a line is one step. `tau_fail` defaults to the sequential `max(4, d)`.
  The q-batch rule `ceil(max(4/q, d/q))` gives 1 at q = 24 and would halve the
  region after every line that doesn't improve. Region restarts start from fresh
  random init chords.
- **HEBO:** kappa's iteration count is `n_points // batch_size`, i.e. lines so far.
- **zombi_hop:** hyperparameters are used as tuned. The keys that count objective
  calls (`max_iterations`, `min_iters_per_zoom`, the derived activation cap) count
  lines again, as they did when MOBO tuned them.

## zombi_hop needle declaration in sweeps

Every sweep, point or line, now plans zombi_hop with:

- `needle_pick = "argmax"` (a `ZoMBIHop` constructor option; the shipped default is
  `"median"`). A declaration lands on the in-bounds best point, not the repeat
  closest to the cluster median.
- `hparam_overrides = {"needle_min_repeats": 0}`, which turns off the repeatability
  gate (5 other measurements within 0.10 before a declaration).

Both guard hardware against upward noise excursions. On these landscapes input
noise is 0, so they only delay declarations and pull the declared point off the
peak. They are benchmark-only: MOBO, the GUI and hardware keep the median pick and
the gate. Restore them in a sweep with
`--method-set zombi_hop.needle_pick='"median"' --method-set zombi_hop.hparam_overrides='{}'`.

## Comparability

- Line campaigns can't be mixed with point campaigns. The manifest records
  `sampling`, `batch_size` and `lines_per_dim`.
- With 24-point lines, short budgets leave few optimiser lines (8 at 2d with 10
  lines). zombi_hop needs at least `min_zoom_for_needle` zooms with
  `min_iters_per_zoom` lines each before its first declaration (6 lines at 2d), so
  at 2d it can declare at most one or two needles.
