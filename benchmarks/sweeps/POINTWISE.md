# Point-wise sampling in the sweeps (2026-09-26)

Before this change, the sweep gave every method 3000 points in **batches of 24**.
ZoMBI-Hop spent each batch as one LineBO line, and the baselines as a q = 24 batch.
Now **every method measures one point per call**. ZoMBI-Hop measures exactly the
candidate it proposes, and the baselines run at q = 1. The comparison is fully
sequential and like-for-like. This applies to `benchmarks/sweeps` only: the
`zombi_hop` method still defaults to lines everywhere else.

## What changed in the code

| file | change |
|---|---|
| `benchmarks/methods/zombihop.py` | New config `sampling` (`"line"` default, `"point"`). Point mode: the objective clips the candidate to the box and measures it alone. LineBO is not built. The initial design is `n_init_points` (48) scrambled-Sobol' points, measured one at a time, as the baselines do. Hyperparameters are passed through unchanged (below). Line mode now refuses `batch_size < 2`, since a one-point "line" is just its left endpoint. |
| `benchmarks/sweeps/configs.py` | Forces `zombi_hop.sampling = "point"` unless overridden. Records `resolved_hparams` (the values actually run) per dimension in the manifest. |
| `benchmarks/sweeps/campaign.py` | `DEFAULT_BATCH` 24 → **1**. `plan` warns if zombi_hop is in point mode while `--batch-size` is not 1. |
| `benchmarks/sweeps/__main__.py` | `--trace-every` default 10 → **240**. `describe` prints zombi_hop's sampling mode and resolved values. |
| `benchmarks/sweeps/README.md` | Budget paragraph updated. |

`src/core/zombihop.py` is **unchanged**. Its loop already accepts any number of
points back from the objective, and the space-filling fallback goes through the
same objective, so it also measures one point.

## ZoMBI-Hop hyperparameters

**Every hyperparameter is used exactly as tuned.** Nothing is rescaled for point
mode. The keys that count objective calls therefore now count single points
instead of 24-point lines:

| key | what it bounds | 3d file | 4/6/10d file (`dist1c`) | ctor default |
|---|---|---|---|---|
| `max_iterations` | objective calls per zoom level | 8 | 30 | — |
| `min_iters_per_zoom` | calls at a zoom before it may declare or zoom again | 3 | 3 | 3 |
| `max_lines_per_activation` | calls per activation before the region is penalised | 30 | 30 | 30 |

So a zoom level is at most 8 (3d) or 30 points, and an activation is at most 30
points before its region is penalised. In line mode those were 192/720 and 720
points. The rest are unchanged too:

| key | note |
|---|---|
| `n_consecutive_converged` (1 at 3d, 5 otherwise) | A streak of converged decisions, one per point. |
| `needle_min_repeats` (5), `needle_repeat_radius_frac` | Already counted in measured points: 5 other measurements within radius, which must now arrive inside the ≤ 30-point activation. |
| `max_zooms`, `min_zoom_for_needle`, `jaccard_window` | Count zoom levels, not measurements. |
| `top_m_points` | Already a point count (points in the zoom ellipsoid fit). |
| everything continuous (`ucb_beta`, noise multipliers, penalty radii, shrink factors, …) | No measurement unit. |
| `linebo_*` | Unused in point mode. |

### A behavioural change worth knowing

The convergence test's second gate ("the best Y from this call improves on the
previous best by less than the noise floor") used to take the maximum over a 24-point
line. Now it sees one noisy point, so that gate passes more often. In effect,
convergence is decided by the EI gate (EI at the candidate < noise floor), which is
unaffected. Expect ZoMBI-Hop to reach "converged" more readily per call, and to
zoom, declare and penalise after far fewer points than in line mode.

## Baselines at q = 1

No config changes were needed. Every default is the method's standard sequential
setting:

- **gp_bo**: `qLogNEI` with q = 1 is ordinary LogNEI. The model refits every point.
- **TuRBO**: `failure_tolerance = ceil(max(4/q, d/q))` becomes `max(4, d)`. That is
  the TuRBO paper's sequential setting.
- **HEBO**: `suggest(n_suggestions=1)`, its standard mode.
- **random**: unaffected. It only changes the row structure of the trajectory.

## Costs

- **Model fits go up about 24×.** Every method except random refits a GP about
  2950 times per cell instead of about 125, on up to 3000 points (gp_bo and HEBO
  use exact GPs with no cap by default). Expect cells to take much longer.
- **`--cell-max-hours` (6 h) is probably too low now**, especially for gp_bo, HEBO
  and zombi_hop at 10d. A cell that hits it is flagged `budget_hit: false` and left
  out of the equal-terms comparison.
- **Measure before planning the full campaign.** Run a small timing campaign
  through `benchmarks/scripts/validate_methods.sbatch`, then set `--cell-max-hours`
  and `--n-workers` from it. If gp_bo or HEBO are far too slow,
  `gp_bo.max_train_points` and `turbo.max_train_points` exist (HEBO has no cap).
  Note that capping them departs from published defaults.
- **Trajectories are finer.** `metrics_over_time.csv` has one row per point (about
  3000 rows). The extractor runs every 240 points, as before in point terms.

## Comparability

- **Old campaigns can't be mixed with new ones.** Campaigns planned before this
  change (`batch_size` 24, zombi_hop in line mode) are not comparable with new
  ones. Each manifest records `batch_size`, `sampling` and `resolved_hparams`, so a
  campaign states which regime it ran.
- **The tuning is still not like-for-like.** ZoMBI-Hop's hyperparameters were
  tuned on the simplex with lines. The conversion above is a principled
  translation, not a re-tuning.

To run the old regime:

```bash
python -m benchmarks.sweeps plan --out ... --batch-size 24 \
    --method-set zombi_hop.sampling=line --trace-every 10
```

## Verification status

- **Done:** `python -m benchmarks.sweeps describe` resolves every config and shows
  the resolved values, and the edited modules compile.
- **Not done:** no optimiser has been run in point mode. Anything that fits a
  model belongs in an sbatch job, not on the login node.
