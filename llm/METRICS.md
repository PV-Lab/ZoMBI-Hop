# Interpreting `sweep_summary.json`

Each object in `sweep_summary.json` (and each row of `sweep_summary.csv`) is **one injection point** in a sweep produced by `evaluate_llm_sweep.py`. At an injection point the experiment:

1. Loads a mid-campaign snapshot of the real campaign2 run (`run_7eb9`).
2. Asks the LLM (once) whether to retune ZoMBI-Hop's hyperparameters given the run so far.
3. Continues the run to the budget's end **two ways** — with the LLM's hyperparameters (`llm_*`) and with the original hyperparameters (`baseline_*`) — repeating each continuation several times and reporting the mean/std.
4. Reports the LLM-minus-baseline differences (`diff_*`).

The goal of the whole harness: does letting the LLM retune hyperparameters mid-run beat leaving them fixed? See [[llm-tuning-harness]].

---

## Where the injection happened

| Field | Meaning |
|---|---|
| `injection_iter` | The ZoMBI-Hop iteration at which the LLM was consulted. The sweep steps through these (5, 10, 15, …, 40). |
| `snapshot` | Name of the saved run_7eb9 snapshot used as the starting state (e.g. `0026_act8_z0_i1` = activation 8, zoom 0, inner iter 1). |
| `n_points_at_injection` | How many droplets (measurements) had already been collected when the LLM was called. Grows monotonically across injection points. |
| `budget_iterations` | ZoMBI-Hop iterations remaining after injection, i.e. how much runway both the baseline and LLM continuations get. Shrinks as `injection_iter` grows (0 means injection happened at the very end). |
| `n_baseline_repeats` / `n_llm_repeats` | Number of continuation runs pooled for each mean/std. Baseline = 1 real campaign run + (N−1) reruns with the original hyperparameters; LLM = N reruns with the LLM's hyperparameters. |

## The LLM's decision

| Field | Meaning |
|---|---|
| `model` / `effort` | Which model answered and at what reasoning effort. |
| `latency_s` | Wall-clock seconds the LLM took to respond. |
| `changed` | `true` if the LLM chose to modify any hyperparameter, `false` if it left them all alone. |
| `changes` | JSON of the hyperparameters actually applied (after validation/clamping), e.g. `{"n_consecutive_converged": 2, "nat_grad_step": 0.05}`. Only keys the LLM changed appear. |
| `reasoning` | The LLM's free-text justification for its decision — the qualitative record of *why* it tuned the way it did. |

Common levers you'll see in `changes` (full glossary lives in `evaluate_llm.py`):
- `n_consecutive_converged` — consecutive low-improvement iterations required to declare a needle. Higher = more reluctant to lock in an optimum (more refinement before hopping).
- `nat_grad_step` — step size of the natural-gradient acquisition optimizer. At its floor (0.001) candidates barely move off their random restarts, so acquisition maximization is effectively random; raising it lets it actually climb the acquisition surface.
- `output_noise_threshold_mult` — convergence gate; lower converges (declares needles) sooner.
- `max_penalty_radius` — size of the ellipsoid that excludes a found needle from future search; larger steers harder away from known optima.
- `ucb_beta` — exploration/exploitation balance of the UCB acquisition; lower = more exploitative.
- `max_zooms` — trust-region shrinks allowed per activation before declaring a needle.

## Outcome metrics (baseline vs. LLM)

Each of these appears as a `baseline_*` mean, and `best_objective` also carries a std. The `llm_*` counterparts are the same metrics for the LLM-tuned continuations. **Higher is better for `best`; interpretation of `needles`/`dup` depends on your goal (see below).**

| Metric | Definition | How to read it |
|---|---|---|
| `*_best_mean` | Mean across repeats of the **best unpenalized objective** found by the end of the continuation. This is the headline quality number — the score of the best composition discovered. | Higher is better. The primary "did tuning help?" signal. |
| `*_best_std` | Standard deviation of that best objective across repeats. | Measures run-to-run stability. A low LLM std with a comparable mean (e.g. inj_020, inj_040) means the LLM made the outcome more *reliable*, even when the mean barely moved. |
| `*_needles_mean` | Mean number of **needles** (distinct local optima ZoMBI-Hop declared and penalized) found by the run's end. | Not "more is better." More needles = broader coverage of the simplex; fewer can mean the run refined a few basins deeply instead of hopping eagerly. Read alongside `best` and the reasoning. |
| `*_dup_mean` | Mean **duplicate fraction** — the fraction of sampled points that are near-duplicates of earlier points (repeatedly re-measuring the same composition). | Lower is better (less wasted budget). High dup = the run kept re-sampling regions it had already mapped. |

Note: `dist_to_ref_optima` is computed internally (distance from discovered needles to the RF reference optima) but is **not** carried into the sweep summary — only `best`, `needles`, and `dup` are summarized here.

## The differences (`diff_*`)

These are **LLM mean minus baseline mean** at that injection point:

| Field | Sign convention |
|---|---|
| `diff_best` | `llm_best_mean − baseline_best_mean`. **Positive = the LLM's tuning produced a better final objective.** This is the metric that most directly answers whether the intervention helped. |
| `diff_needles` | `llm_needles_mean − baseline_needles_mean`. Negative means the LLM found fewer needles — usually because it deliberately traded eager hopping for deeper refinement (see the reasoning). |
| `diff_dup` | `llm_dup_mean − baseline_dup_mean`. Positive = the LLM run re-sampled *more*; negative = it wasted less budget on duplicates. |

### Reading a row end-to-end

Example (`injection_iter: 20`):
- The LLM raised `n_consecutive_converged`, enabled a real `nat_grad_step`, and widened `max_penalty_radius`.
- `diff_best = +0.032` → the LLM's final objective was ~0.03 higher on average.
- `llm_best_std = 0.007` vs `baseline_best_std = 0.068` → not only better on average but far more consistent.
- `diff_needles = −5.6` → it declared fewer needles, consistent with its stated intent to refine promising basins instead of hopping.
- `diff_dup ≈ −0.007` → marginally fewer duplicate re-measurements.

This is a clear "LLM helped" point. Contrast with `injection_iter: 5`, where `diff_best = −0.058`: there the same instinct (be more reluctant to declare needles) hurt, because early in the run broad exploration mattered more than refinement.

---

**Caveats when drawing conclusions**
- Means are over only 5 repeats, so small `diff_best` values (a few thousandths) are within noise — weight the larger, consistent moves.
- `best` is the metric to optimize; `needles` and `dup` are diagnostic context, not objectives in themselves.
- The `reasoning` field is essential for interpretation: the numeric diffs tell you *whether* it helped, the reasoning tells you *what the LLM was trying to do*.
