# s1_real -- results (SUPERSEDED, pre-merge archive)

> **Superseded by `benchmarks/results/s1_v2/`.** This bundle is retained as the
> pre-merge archive. Its `zombihop` / `zombihop_nc5` rows were produced against core
> `77054a9` and **must not be quoted**; its `random` / `gp_*` rows are still valid and
> are carried forward into s1_v2.
>
> The headline claim below -- `zombihop` > `random` at matched |S| on real3d,
> +0.0857, p(sign)=0.039 -- **does not survive**. On the merged core at 20 seeds it is
> **+0.0036, 8W/3T/9L, p=0.874**. Part core fix, part seed luck; see DESIGN.md 27.


180/180 cells, 0 failures. 3 real campaigns x 6 methods x 10 seeds, N=2000 samples,
q=24 per decision, hardware-matched noise. Generated from `benchmarks/zhbench/configs/s1_real.yaml`.

Reference optima are **supported surrogate peaks** (GP over the whole campaign,
prominence 0.3, each peak required to have a real measured sample within r=0.05
at a comparable value). They are NOT hardware-validated optima.

## Provenance — these numbers are pinned to a core version

> ⚠️ **The core has since been merged forward (`origin/brianna` @ `baa51de`), and
> the `zombihop` / `zombihop_nc5` rows below are STALE.** They are kept as the
> archived pre-merge bundle. Measured shift on real3d over 3 paired seeds:
> `n_declared` 10.67 → 5.00, `precision` 0.449 → 0.764, `peak_ratio` −0.095
> (DESIGN.md §27). **Do not compare these two rows against post-merge numbers.**
> The four `random` / `gp_*` rows are unaffected and remain current.

Run at `benchmarking-v2` commits `d304c411` (170 cells) / `cd568622` (10 cells),
both carrying the ZoMBI-Hop core as of `77054a9`. `dirty=false` on all 180 cells.

**The `zombihop` and `zombihop_nc5` rows depend on core behaviour that has since
changed on `origin/brianna`, and they do not survive a merge unchanged.** Three
verified differences:

| | 77054a9 (these numbers) | `origin/brianna` | `origin/brianna-v2` |
|---|---|---|---|
| `_declare_needle_at_best` | needle planted at the **global** unpenalized argmax | planted at the argmax **inside the zoom box** (a genuine bug fix) | same as brianna |
| `min_iters_per_zoom` default | 2 | **3** | 3 |
| `min_zoom_for_needle` default | 1 | 1 | **0** |

Moving the needle moves the penalty ellipsoid, so the whole subsequent trajectory
changes: `peak_ratio`, `precision`, `n_declared` and `reached_ratio` are all at
risk. The four `random` / `gp_*` arms are untouched and remain valid.

This matters most for the *causal* claim below. 496 of the 563 needles **logged** in
this bundle (88.1%) were declared at `zoom == 1` — exactly the `min_zoom_for_needle`
floor — and none at `zoom == 0`. `brianna-v2` sets that floor to 0, i.e. it changes
precisely the constraint this bundle attributes the low `peak_ratio` to, and this
bundle contains no observations of that regime.

(Denominators: **496/563 = 88.1% of logged needles, 496/569 = 87.2% of all declared**.
569 were declared — `optimizer_state.n_needles` summed over the 60 cells — against 563
rows in `needles.csv`, because `zombihop_runner`'s `obj_wrapper` appended to
`needle_log` only on the *next* objective call, so a needle declared by the
budget-exhausting line was never logged. Six cells were short by one, and the 6 missing
carry no zoom level. **Fixed** — the log is now drained in the `finally` as well, with
a regression test — but this bundle was produced before the fix, so its `@N` prefix
curves remain short in those 6 cells. Two headline numbers are affected:
`real4d/zombihop/s6` `peak_ratio@2000` 0.185 vs 0.222 final, and `real6d/nc5/s5`
0.000 vs 0.015. **fig1, fig3 and fig4 are built from the prefix curves and inherit
this**; the final-column metrics read `dh.needles` directly and are correct.)

`optimize/eval_metrics.py` did **not** drift: `MATCH_RADIUS = 0.05` and
`metric_dist_to_needles` are byte-identical across 77054a9, `brianna` and
`brianna-v2`, so the scoring side of the harness is safe across a merge.

Every comparison below is **paired by seed** -- a seed fixes the initial design, and
the first 48 samples are byte-identical across all six methods -- so the per-seed
difference removes the shared landscape/start variance. Full tables with CIs and
per-seed win/tie/loss counts are in `STATS.md`; regenerate with
`python -m benchmarks.zhbench.stats <run_dir>`.

## Read this first: what 10 seeds actually resolve

A comparison is called **resolved** only when a paired t test *and* an exact sign
test both clear p<0.05. Recall moves in steps of `1/n_true` -- 0.0714 on real3d,
0.0370 on real4d, 0.0147 on real6d -- so a 0.02 difference is not a small effect,
it is no effect plus rounding.

| claim | verdict |
|---|---|
| `zombihop` > `random` at matched \|S\|, real3d | **resolved** (+0.086, 8W/1T/1L, t=+3.34 p=0.009, sign p=0.039) |
| `n_consecutive_converged` 2 > 5, declared recall, real4d | **resolved** (+0.093, 9W/1T/0L, t=+5.51 p<0.001, sign p=0.004) |
| `random` > `gp_qlogei` at matched \|S\|, real4d | **resolved** (+0.089, 9W/1T/0L, t=+5.04 p<0.001, sign p=0.004) |
| input cost: ZoMBI-Hop cheaper than every baseline | **resolved** -- no overlap at any dimension |
| wall-clock: ZoMBI-Hop cheaper than the GP baselines at 3D/4D | **resolved** -- no overlap |
| `zombihop` > `gp_ts` / `gp_qlogei` / `gp_qucb` at matched \|S\|, real3d | **not resolved** (gp_ts: +0.021, 4W/3T/3L, p=0.54) |
| `zombihop` better than anything on real4d or real6d | **not resolved** -- and mostly the wrong sign |
| `random` > `gp_qucb` / `gp_ts` at matched \|S\|, real4d | **not resolved** (p_sign 0.070 / 0.180) -- consistent trend only |

**The defensible headline today:** at matched declarations ZoMBI-Hop's samples are
*at least as informative as the best baseline* and *clearly better than uniform
random*, at a tenth of the input cost and a quarter of the compute. It is not
established that it beats a well-run GP baseline at recovering many optima.

Note also that the one resolved method win sits on `real3d`, the **least**
discriminative landscape in the set (peak rarity 0.0349).

## How discriminative is each campaign

| objective | n_true | peak rarity | interpretation |
|---|---|---|---|
| `real3d` | 14 | 0.0349 | 3.5% of uniform draws score as well as a typical optimum - weak test |
| `real4d` | 27 | 0.0110 | usable |
| `real6d` | 68 | 0.0021 | sharpest real landscape we have |

## Headline: compare at MATCHED declarations

At |S| = n_true every method declares as many optima as exist, which scores
ZoMBI-Hop's ~6 declarations against a baseline's 14 guesses. The fair comparison
applies the identical post-hoc extractor to every method's samples at the same |S|.

### real3d (n_true=14), matched |S|=6

| method | post-hoc recall @\|S\| | declared recall | precision | input cost | s/cell |
|---|---|---|---|---|---|
| `random` | 0.257 | 0.557 ± 0.111 | 0.557 ± 0.111 | 1000 ± 15 | 1 |
| `gp_qucb` | 0.286 | 0.543 ± 0.113 | 0.543 ± 0.113 | 1473 ± 30 | 2374 ± 249 |
| `gp_qlogei` | 0.314 | 0.550 ± 0.107 | 0.550 ± 0.107 | 1089 ± 87 | 1960 ± 36 |
| `gp_ts` | 0.321 | 0.514 ± 0.100 | 0.514 ± 0.100 | 359 ± 39 | 211 ± 8 |
| `zombihop` | **0.343** | 0.271 ± 0.125 | 0.672 ± 0.196 | **112 ± 5** | 509 ± 138 |
| `zombihop_nc5` | 0.307 | 0.179 ± 0.091 | 0.835 ± 0.230 | 108 ± 5 | 465 ± 133 |

`zombihop` has the highest mean, but only the gap to `random` survives a paired
test. The gaps to the three GP baselines are 0.021-0.057, i.e. **under one optimum**
(1/14 = 0.071), and none is resolved.

### real4d (n_true=27), matched |S|=15

| method | post-hoc recall @\|S\| | declared recall | precision | input cost | s/cell |
|---|---|---|---|---|---|
| `random` | **0.237** | 0.311 ± 0.078 | 0.311 ± 0.078 | 988 ± 10 | 1 |
| `gp_qucb` | 0.178 | 0.241 ± 0.068 | 0.241 ± 0.068 | 1340 ± 24 | 1819 ± 32 |
| `gp_qlogei` | 0.148 | 0.215 ± 0.072 | 0.215 ± 0.072 | 1152 ± 40 | 2592 ± 227 |
| `gp_ts` | 0.181 | 0.285 ± 0.106 | 0.285 ± 0.106 | 727 ± 39 | 299 ± 35 |
| `zombihop` | 0.219 | 0.185 ± 0.043 | 0.333 ± 0.077 | **60 ± 5** | 658 ± 307 |
| `zombihop_nc5` | **0.237** | 0.093 ± 0.031 | 0.428 ± 0.136 | 43 ± 6 | 423 ± 171 |

`zombihop_nc5` vs `random` here is a mean difference of **exactly 0.0000**.

### real6d (n_true=68), matched |S|=15

| method | post-hoc recall @\|S\| | declared recall | precision | input cost | s/cell |
|---|---|---|---|---|---|
| `random` | 0.016 | 0.029 ± 0.017 | 0.029 ± 0.017 | 902 ± 7 | 1 |
| `gp_qucb` | 0.016 | 0.029 ± 0.016 | 0.029 ± 0.016 | 1114 ± 10 | 1755 ± 27 |
| `gp_qlogei` | **0.019** | 0.034 ± 0.027 | 0.034 ± 0.027 | 981 ± 36 | 2509 ± 286 |
| `gp_ts` | 0.012 | 0.015 ± 0.016 | 0.015 ± 0.016 | 816 ± 8 | 445 ± 46 |
| `zombihop` | 0.010 | 0.007 ± 0.010 | 0.034 ± 0.049 | **62 ± 2** | 4784 ± 1060 |
| `zombihop_nc5` | 0.010 | 0.007 ± 0.010 | 0.051 ± 0.081 | 60 ± 2 | 2913 ± 565 |

Everything here is within ~1 optimum out of 68 of everything else. **Nothing about
6-D is decided by this run** -- see the budget curves below.

## Aleks's prediction: standard BO recovers fewer optima than random

On `real4d`, `random` beats every GP baseline at matched |S|. Only the `gp_qlogei`
comparison resolves (+0.089, 9W/1T/0L, both p<0.005); `gp_qucb` (+0.059, p_sign
0.070) and `gp_ts` (+0.056, p_sign 0.180) are consistent trends. Cumulative best-y
would have hidden all three.

**The effect reverses at 3-D**: there `gp_ts` beats `random` by 0.064 (1W/3T/6L,
t=-2.38 p=0.041, sign p=0.125 -- not resolved). So "standard BO is worse than
random at recovering many optima" is a 4-D observation, not a general one, and
should not be stated without the dimension attached.

## Search vs declaration -- the clearest qualitative finding

| real3d | reached_ratio (samples land on optima) | peak_ratio (declared) |
|---|---|---|
| `random` | 0.993 ± 0.023 | 0.557 ± 0.111 |
| `zombihop` | 0.957 ± 0.060 | 0.271 ± 0.125 |

ZoMBI-Hop's samples reach 96% of the true optima against random's 99%, but it
declares only 27%. The search is finding them; the declaration budget is the
bottleneck (an activation needs 4-6 lines before it may declare anything).

## Cost -- the most consistent result in the suite

Input cost (SnAKe path length) at N=2000, ZoMBI-Hop vs the **cheapest** baseline:

| | real3d | real4d | real6d |
|---|---|---|---|
| `zombihop` / `zombihop_nc5` | 112 / 108 | 60 / 43 | 62 / 60 |
| cheapest baseline (`gp_ts`) | 359 | 727 | 816 |
| ratio vs cheapest | **3.2x** | **12.2x** | **13.1x** |
| ratio vs `random` | 9.0x | 16.6x | 14.5x |

No seed overlaps at any dimension. ZoMBI-Hop is also **4-5x cheaper in wall-clock**
than the GP baselines at 3-D and 4-D -- contrary to its reputation -- though it
becomes the most expensive method at 6-D (4784 s/cell).

## `n_consecutive_converged`: the tuned value beats 5

Resolved on `real4d` declared recall (+0.093, 9W/1T/0L, t=+5.51 p<0.001, sign
p=0.004). Same direction on real3d (+0.093, 7W/2T/1L) but not resolved; no
difference at all on real6d. `nc5` buys precision (0.835 vs 0.672 at 3-D) by
declaring roughly half as many needles. Brianna has confirmed she and Aleks changed
it live from 5 to 2 mid-campaign for exactly this reason.

**Careful with the label.** The `zombihop` arm does not run `nc = 2` everywhere: it
takes whatever the tuned JSON for that dimension carries, and those are

| arm | real3d | real4d | real6d |
|---|---|---|---|
| `zombihop` | **1** (`3d.json`) | 2 (`4d.json`) | 2 (`6d_ensemble.json`) |
| `zombihop_nc5` | 5 | 5 | 5 |

So the resolved 4-D result is genuinely "2 beats 5"; the 3-D comparison is "1 vs 5",
and the two should not be pooled into a single "2 beats 5" sentence. The registry
comment in `optimizers/__init__.py` previously implied a uniform 2 — corrected.

## Budget curves -- is 6D under-budgeted?

reached_ratio at each checkpoint:

| objective | method | N=250 | N=500 | N=1000 | N=2000 |
|---|---|---|---|---|---|
| real3d | `random` | 0.600 | 0.871 | 0.943 | 0.993 |
| real3d | `zombihop` | 0.571 | 0.793 | 0.900 | 0.957 |
| real4d | `random` | 0.107 | 0.230 | 0.363 | 0.581 |
| real4d | `zombihop` | 0.100 | 0.193 | 0.341 | 0.507 |
| real6d | `random` | 0.004 | 0.006 | 0.006 | 0.026 |
| real6d | `zombihop` | 0.001 | 0.003 | 0.003 | 0.007 |

real3d saturates by N=1000 (0.943 -> 0.993), so its discriminating columns are
N=250/500. real6d is still climbing steeply at N=2000 and needs a much larger
budget before any 6D claim is meaningful: 68 optima and 83 lines, against the 109
lines (2616 samples) the real campaign used.

## Figures

- `fig3_matched_declarations.png` -- **the one that matters.** Recall vs how many
  optima each method declares; star marks ZoMBI-Hop's own declared count.
- `fig1_reached_ratio.png` -- sample efficiency, optima reached vs budget, +/-1 std.
- `fig6_dist_to_needles.png` -- distance to the optima vs budget **at matched |S|**.
  Read only at matched |S|: `metric_dist_to_needles` charges 0.5 for every optimum a
  method never claimed, so comparing raw declared sets reproduces exactly the
  unfairness fig3 exists to remove. At matched |S| all six methods overlap within
  ~0.01 on every landscape -- distance does not discriminate here, which is itself
  consistent with how shallow these surrogates are.
- `fig4_needles_declared.png` -- ZoMBI-Hop's declaration budget over time.
- `fig2_endpoint.png` -- end-of-budget bars. **Its `dist_to_needles` panel is the
  unmatched-|S| view** and will make ZoMBI-Hop look far worse than it is; prefer fig6.

## Reproducing

```bash
python -m benchmarks.zhbench.report benchmarks/results/s1_real
python -m benchmarks.zhbench.stats  benchmarks/results/s1_real
```

`curves.json` and `matched_curves.json` are committed alongside `aggregate.csv`, so
both commands work against this directory. Per-cell artifacts (`points.csv`,
`declared_optima.csv`, `metrics.json`) are under `benchmarks/runs/`, which is
gitignored -- ask Colin if you want them. Regenerating `matched_curves.json` needs
those per-cell files; the committed copy means you do not have to.
