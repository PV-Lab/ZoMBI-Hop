# benchmarking-v2 — design decisions

What changed from `benchmarking_v2_spec.md`, and why. Everything here was checked
against the code on `brianna@77054a9`, not assumed.

## Corrections to the spec

**1. `physics_simulate_line` is deterministic, not Gaussian noise.**
The spec's protocol table said baselines should get "per-point Gaussian with std
`run_mobo.NOISE_LEVEL` (0.128) then re-project to the simplex", described as "same
physics for everyone". It is not the same physics. `physics_simulate_line`
(`optimize/composition_prediction.py:992`) is a deterministic print model — syringe
ramp lag/overshoot plus junction-volume diffusion mixing — and its comment says so
explicitly: "instead of adding random input noise".

Measured over 200 random chords per dimension, it distorts a requested line by mean
L2 0.066 (d=3) to 0.113 (d=10). `N(0, 0.128)` per component gives mean L2 0.222 to
0.405 — **about 3× harsher at every dimension.** Following the spec would have
handicapped the baselines threefold and manufactured the result the study is
supposed to test.

*Resolution:* `protocol.calibrate_input_noise` measures the physics residual
distribution per dimension and baselines bootstrap magnitudes from it. Table
committed at `zhbench/data/input_noise_calibration.json`.

**2. The printer has ten syringe modules, so the dimension ladder cannot reach 12-D
with a hardware model.**
`physics_simulate_line` raises `ValueError: physics model supports up to 10
modules` for d > 10. The spec's ladder (3/4/6/10/12) and the project goal ("from
current 3d, 4d to 5d...12d") run past a hardware limit at d=11. `run_mobo.make_sim_obj`
calls the physics model unconditionally, so ZoMBI-Hop as wired in `run_mobo` cannot
run above 10-D at all.

*Resolution:* `zombihop_runner` supplies its own `sim_obj` with the same contract
(LineBO and the core are untouched). Above 10 components it falls back to the
calibrated point-wise perturbation and records
`line_realization: "no_printer_model"`. **11-D and 12-D are computational studies
with no printer behind them, and the artifacts say so.** Whether to run them at all
is a call for Aleks.

**3. `warm_gp_landscape_summary.json` holds WARM-start-GP peaks, not full-GP peaks.**
The spec treated its 10 (3-D) / 15 (4-D) peaks as the real4d reference "already
computed". They are not: that JSON is produced by `build_warm_landscape`, which fits
only the 4/8 warm lines. `fullgp_objective(d)` fits the whole campaign and detects
**24 peaks at both d=3 and d=4**. The benchmark uses `fullgp_objective`.

**4. `data/2nd_real_run.db` was never lost.**
The spec said it had to come from Dropbox. It is committed on
`origin/evelyn-compositional`:
`git show origin/evelyn-compositional:data/2nd_real_run.db > data/2nd_real_run.db`
(953 rows, 41 lines, 644 scored, Objective 0.278–0.860 — matches the loader
docstring exactly). That branch also carries `2nd_real_run_ela_full.json`,
`campaign1a_ela_full.json` and `run_9dfe.db`, all now restored to `data/`.

**5. `data/campaign1a.csv` is not a 3-D stand-in.**
Different run (766 rows / 36 iterations vs 953 / 41), a CSV with no `results`
table so `load_campaign` cannot read it, and a non-overlapping Objective range.
Not used for `real3d`.

**6. The spec's post-hoc solution set optimised spread, not value.**
It proposed top-5%-by-`y` then `maximin_subset`. Maximin maximises separation,
which is not what a practitioner would declare. Replaced with greedy
value-ordered selection under an exclusion radius of `2r` — standard niche
extraction, and the most generous procedure available to a single-solution method.

**7. `peak_ratio` needs one-to-one matching and merged references.**
The spec defined it as "fraction of true optima with a member of S within r". The
3-D campaign GP has two detected peaks 0.067 apart and `r = 0.05`, so one declared
point sits within `r` of both. Reference optima closer than `2r` are now merged, and
matching is a maximum bipartite matching on the `≤ r` threshold graph.

**8. The `reached` value tolerance was scale-blind.**
`δ = 2 × OUTPUT_NOISE_FRAC × y*` is 0.063 on landscapes whose entire range is
0.48–0.88 — about 40% of the range, so the condition would have done nothing.
Replaced with a fraction of each peak's prominence above the landscape background.

**9. Baselines must get real q-batch acquisitions.**
The salvageable `gp_botorch.py` takes greedy top-k of a single-point analytic
acquisition, and the old `runner.py:218` hard-coded `suggest(1)` anyway — so the
previous benchmark never ran a batch at all. Rewritten as sequential-greedy
`qLogEI` / `qUCB` with `X_pending`.

**10. `ObjectiveRun` really can be the single noise/budget point.**
The spec assumed ZoMBI-Hop had to apply its own noise via `run_mobo.make_sim_obj`.
Since the runner supplies its own `sim_obj` anyway (see 2), that `sim_obj` calls
`ObjectiveRun.evaluate_batch` — so one class applies noise, counts samples and
raises `BudgetExhausted` for every method alike.

## Decisions that stand

- Budget enforced by raising out of the objective, `never_terminate=True`, no core
  patch and no `max_objective_calls` hook. Mirrors `evaluate.run_single_eval`.
- `r = MATCH_RADIUS = 0.05`, imported from `eval_metrics` so benchmark and tuner
  cannot drift.
- `dist_to_needles` reused from the core (Hungarian, capped at 0.5).
- Baselines get free scattered batches (no chord adapters), per Aleks. `input_cost`
  measures the physical price they are not charged.
- Salvaged from the old branch: `spaces.py` (ILR, correct and dimension-general),
  `seeding.py`. Everything else rewritten or dropped.

## Open questions for the team

1. **`input_noise` told to ZoMBI-Hop.** `ZOMBI_FIXED` carries 0.128 (hardware-measured,
   current production). The tuned 4-D set in `BEST_HPARAMS.md` was tuned with 0.064,
   and the simulator's actual per-point error is smaller than either. The benchmark
   uses `ZOMBI_FIXED` so it matches production. Worth confirming.
2. **`dup_fraction` radius.** The core scales it by zoom size, which rewards
   zooming; baselines have no zoom. The benchmark uses one global radius for
   everyone so the number is comparable, which means a zooming method reads high by
   construction. It is reported as a diagnostic, not a score.
3. **Shallow real landscapes.** Measured with `metrics.landscape_contrast`
   (2000 uniform probes):

   | objective | true optima | after merge | peaks above random p99 | mean prominence |
   |---|---|---|---|---|
   | `real3d` | 24 | **20** | **0.15** | 0.50 |
   | `real4d` | 24 | 24 | **0.46** | 0.73 |
   | `ensemble 3d, n=20` | 20 | 18 | 1.00 | 1.00 |
   | `ensemble 4d, n=20` | 20 | 19 | 1.00 | 1.00 |

   Only 15% of the 3-D campaign's detected peaks stand above what uniform random
   sampling routinely hits, and four of its 24 reference peaks are closer together
   than `2r` and get merged. `real3d` is close to being unable to separate methods
   at all; `real4d` is usable but soft. The RF surrogate over the same campaigns is
   spikier and should be added as `real{3,4}d_rf` — this is the strongest argument
   for doing so.

   An early signal that the metric change matters: on `real4d` at N=1000, uniform
   random reaches `peak_ratio` 0.292 while `gp_qucb` reaches 0.208. Random beats a
   standard BO baseline at recovering *many* optima, which is exactly Aleks's
   prediction and exactly what cumulative best-y would have hidden.

5. **Compare at matched |S|, and report both.** First numbers, N=1000, **one seed
   each** — not a result, a calibration of what the harness says:

   | | real3d (20 true) | real4d (24 true) |
   |---|---|---|
   | ZoMBI-Hop needles declared | 9 | 7 |
   | ZoMBI-Hop recall / precision | 0.250 / 0.556 | 0.083 / 0.286 |
   | random post-hoc at the same \|S\| | 0.100 / 0.222 | 0.167 / 0.571 |
   | random post-hoc at \|S\| = n_true | 0.400 / 0.400 | 0.292 / 0.292 |
   | ZoMBI-Hop's own samples, post-hoc at the same \|S\| | 0.300 | 0.125 |
   | input_cost | 49.6 vs 483.9 | 29.3 vs 481.8 |

   At 3-D ZoMBI-Hop beats random 2.5x on both recall and precision **at matched
   declarations**, and pays a tenth of the input cost. At 4-D it does not beat
   random even at matched \|S\|. Two things to chase before drawing any conclusion:
   ten seeds instead of one, and why ZoMBI-Hop declares only 7 needles in 1000
   samples on a landscape with 24 optima. Note also that in both cases its own
   samples support slightly more optima than it declares (0.300 vs 0.250; 0.125 vs
   0.083), so declaration, not search, is part of what is limiting it.
4. **Public materials datasets — an unresolved disagreement, not a task.**
   Aleks asked us to expand to `awesome-matchem-datasets`. Brianna has already
   evaluated and rejected that route in `Narrowing Generalization Options.docx`:
   HTEM "mixes at most 4 elements at a time, and there are only ~44 data points per
   set of elements"; the rest are "too sparse, have too few dimensions, just
   theoretical (so probably overly smooth), or in a different domain"; and Olympus
   OER is "too sparse/smooth compared to our data. Also 5d and 6d interactions are
   all theoretical." Somebody has to settle this before adapters get built.

   What a survey of the list actually supports:

   | source | status |
   |---|---|
   | Olympus OER plates | **the only strong candidate.** 4 plates, 6-D simplex, 2119–2121 rows each, real measured overpotential. Needs no `olympus` install — the raw `data.csv` is enough (the package pins `tensorflow==1.15` and will not install). Caveat: compositions lie on a **discrete 0.1 lattice** (11 levels/component), so a continuous objective means fitting a surrogate over the grid, as the ZoMBI paper does. |
   | HTEM-DB | `htem.nrel.gov` and `htem-api.nrel.gov` are **dead** (the awesome-matchem README still links the dead host). Confirms Brianna's view. |
   | everything else in awesome-matchem | structure-, SMILES-, text- or categorical-reagent-based. No large dense continuous-composition set at dim 3–12. |
   | `data/poisson_RF_trained.pkl` | **will not load** — pickled with scikit-learn 1.1.1; the tree node dtype changed in 1.3. Needs a `scikit-learn<1.3` env or a node-array rewrite. |
   | `data/3D-6-final-GP-model` | needs GPy, which does not install on py3.12 / numpy 2.x. |
   | ROBOT (`nataliemaus/robot`) | benchmarks rover / lunar lander / stocks / GuacaMol — **nothing material-science**. Not pip-installable, one commit from 2022. Usable as an algorithm to reimplement, not as a dependency. |

   Recommendation: build `oer6d` from the Olympus CSVs (cheap, real, 6-D, and it
   answers Aleks directly), and drop HTEM. Do not spend time on the two orphaned
   ZoMBI-paper pickles unless someone wants the 6-D Poisson ladder rung badly
   enough to stand up a legacy-sklearn environment.

---

# Round 2 — protocol revisions

Five changes agreed after the first results. Numbering continues from above.

## 11. Hardware-matched noise is now the primary, not physics-matched

Round 1 calibrated the baselines' input noise to the *physics simulator*
(mean L2 0.066-0.113). That equalised the two sides, but at the wrong level: the
printer was measured at 0.27 L2 / 0.128 per component, ~2.5-4x noisier, and ~87%
of the real residual is **perpendicular to the requested line** — a component a
ramp-lag/diffusion model cannot produce at all. Meanwhile ZoMBI-Hop is *told*
`input_noise=0.128`, so testing it in a world three times quieter than that makes
it over-conservative for no reason.

`Protocol.noise` is now a three-way switch:

| mode | lines | batch points | role |
|---|---|---|---|
| `hardware` | print model + random residual | random | **primary** |
| `physics` | print model only | bootstrapped from the print model's residuals | sensitivity |
| `none` | clean segment | clean, and output noise off | control |

In `hardware` mode the perturbation scale is **solved**, not assumed:
`calibrate_hardware_noise` bisects until the *realized* per-component std of
`X_actual - X_requested` equals `NOISE_LEVEL`. Injecting `N(0, 0.128)` and
projecting back to the simplex lands well below 0.128 because projection clips, so
"we injected 0.128" and "the samples really are 0.128 off" are different claims and
only the second is what NOISE_LEVEL means. A test asserts the realized value, and
every run records `realized_noise_std`.

## 12. ZoMBI-Hop's needle count has a structural ceiling; report it as a curve

"Why only 7 needles" is mostly arithmetic. With `max_zooms=3, max_iterations=2,
min_zoom_for_needle=1, min_iters_per_zoom=2` an activation must spend 4-6 lines
before it may declare anything, so ~42 lines (N=1000) buys at most 7-10 needles
regardless of the landscape. Covering 24 optima needs ~100-150 lines. Reading a low
`peak_ratio` as a search failure when it is a declaration budget is the easiest
mistake this benchmark invites, so every ZoMBI-Hop run now emits a
`needles_declared(t)` curve, and the budget sweep (below) is what separates the two
explanations.

## 13. Hyperparameter default switched to the JSON, with the stale value as a sensitivity

`optimize/hparams/6d_ensemble.json` has `n_consecutive_converged=2`;
`src/default_hparams.py`, which says it is a verbatim copy, still has 5. The JSON is
the value the 6-D campaign actually ran, so it is the benchmark default
(`zombihop`); 5 runs as `zombihop_nc5`. Reported upstream in `UPSTREAM_REQUESTS.md`.

## 14. Reference optima must have measured support

`detect_peaks` keeps anything rising 12% from the GP median to the max, which is
why only 15% (3-D) / 46% (4-D) of "true optima" cleared random's p99 — most were GP
wiggles between measurements. `benchmarks/zhbench/landscapes.py` raises prominence
to 0.3 **and** requires at least one real campaign sample within `r` whose measured
objective is close to the peak's predicted value. A GP bump with no measurement
behind it is not a discovered material. `n_true` is reported at both prominence
levels so the change is visible rather than asserted.

This also gave the 6-D campaign a peak finder that works at all: the upstream
lattice search is O(n^(d-1)) and is already coarse at d=4.

## 15. Budget is a curve, not an endpoint

Metrics are evaluated at N in {250, 500, 1000, 2000} by prefix, from a single
2000-sample run — nothing about a prefix depends on what came after it. At N=1000
uniform random puts ~9 samples within r=0.05 of any point on the 2-simplex, so 3-D
is a saturated test at that budget and the small-N columns are where it
discriminates. Declared sets are snapshotted at each checkpoint, so ZoMBI-Hop is
scored on the needles it had actually committed to by then.

## 16. A diversity-friendly BO baseline

`gp_ts` — batch Thompson sampling via `MaxPosteriorSampling` on the same GP and
candidate pool. qUCB and qLogEI are greedy by construction, so without TS the
finding "standard BO recovers fewer optima" could be read as "qUCB is greedy".
TuRBO and ROBOT cover the same objection more thoroughly later.

## 17. real6d, on the full simplex

`data/4th_real_run.db` (2423 rows / 111 iterations, 2042 scored across 109 lines;
components FAPbI3, CsPbI3, FAPbBr3, MACl, MAPbI3, MAPbBr3). The apparent per-component
ceilings (MACl 0.412, FAPbI3 0.515) are **not** a search box: of 218 line endpoints
only 7 sit within 0.02 of a component max — exactly one per component, i.e. only the
point that defines it — while 92 endpoints have a component at zero. LineBO presses
against the zero faces constantly and never against an upper face. Built on the full
simplex; `landscapes.unsupported_mask` reports what fraction of a run's samples fell
in regions the campaign never measured, so a result driven by extrapolation is visible.


## 18. Corrected landscape contrast (rank metric)

The numbers in section 3 above were computed with the broken threshold metric (see
`metrics.landscape_contrast`). Re-measured with rank-based rarity, 4000 probes:

| objective | n_true | peak rarity | in top 1% | frac at max | mean prominence |
|---|---|---|---|---|---|
| `real3d` | 14 | **0.0349** | 0.21 | 0.0003 | 0.60 |
| `real4d` | 27 | **0.0110** | 0.48 | 0.0003 | 0.77 |
| `real6d` | 68 | **0.0021** | 0.81 | 0.0003 | 0.69 |

Rarity is the number to read: on `real3d` 3.5% of uniform draws score as well as a
typical reference optimum, so random search hits optimum-quality values constantly
and a good `peak_ratio` there means little. On `real6d` it is 0.2%, seventeen times
rarer. None of the three saturates (`frac_at_max` 0.0003), which is why the broken
metric still produced the right ordering -- but these are the values to quote.

## 19. The 0xC0000005 crashes: unbounded acquisition memory (closed)

`gp_qucb` cells died with a Windows access violation at N=2000. Three of my
diagnoses were wrong before the right one, and the wrong ones are recorded here
because the pattern is instructive.

| diagnosis | basis | verdict |
|---|---|---|
| thread oversubscription | 10 workers x 24 OMP threads | wrong -- pinning threads changed nothing |
| dual OpenMP runtimes | sklearn vcomp140 + torch libiomp5md both present | wrong -- a pure-BoTorch reproducer that never loads vcomp140 crashes identically, and KMP_DUPLICATE_LIB_OK did not help |
| startup race, first GP cell only | crashed at cell 11 twice | wrong -- an artifact of an uncontrolled box; cell 12 then crashed too |
| budget threshold at N=2000 | crashed at 2000, passed at 336 | wrong -- no threshold; N=2000 simply runs 81 decisions instead of 12, so it is exposure |

Every one of those was inferred from a correlation rather than measured. The
actual cause, found by capping process commit with a job object and making the
crash deterministic (cap 3000 MB: 3/3 crash; cap 5200 MB: 3/3 clean):

`optimizers/gp.py` evaluated the acquisition over the whole 2048-candidate pool in
one t-batch, so gpytorch materialised a `(pool, q_pending, n_train)` float64
cross-covariance -- 2048 x 24 x 1024 x 8 = 384 MiB per temporary, with more beside
it in `sq_dist`. Peak commit per cell 10.1 GB; several workers exceed the machine
commit limit, and a torch allocation that cannot be satisfied faults with
0xC0000005 rather than raising MemoryError, because the allocation that lands short
is on an unchecked path inside gpytorch. That is also why try/except could not
catch it, and why it looked nondeterministic.

Fixed by chunking the pool (peak 10.1 -> 3.9 GB). **Changes no number**: chunked and
unchunked runs produce a byte-identical `points.csv` and differ only in wall-clock.
The chunk is rounded down to a power of two so MKL blocks the batched matmul the
same way it does at full width; a non-power-of-two chunk shifts values by ~1 ulp.

The environment really does load two OpenMP runtimes and that is still worth fixing
(`UPSTREAM_REQUESTS.md` section 5) -- it just was not this.

## 20. The budget was not being spent exactly

Every cell reported `n_samples = 1992` against a nominal 2000. The baseline loop ran
a fixed `n_decisions = (N - n_init) // q = 81`, and `48 + 81*24 = 1992`.

Cosmetically this dropped the headline `peak_ratio@2000` column, since prefix
scoring skips checkpoints above the samples actually spent. Substantively it was
worse: ZoMBI-Hop never uses `n_decisions` -- it runs until `BudgetExhausted` -- so it
would have spent 2000 while every baseline spent 1992. Comparing methods at
different budgets is the exact failure this benchmark exists to prevent, and 0.4%
would have been invisible in every figure.

Both paths now run until the budget is spent, final partial batch truncated
identically for everyone. Test pins it at a ragged N=200.

The general lesson, and the reason the acceptance tests are written the way they
are: assert on the artifacts on disk, not on what the protocol intends. "Same
initial design" held when tested that way. "Exact budget" did not, and nothing
short of reading `n_samples` out of a written `metrics.json` would have caught it.
