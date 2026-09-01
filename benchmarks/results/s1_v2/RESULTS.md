# s1_v2 — results on the merged core

260 cells, 0 failures, every cell spending exactly 2000 samples. 3 real campaigns ×
up to 7 methods, 10 seeds (20 for `random` / `gp_ts` / `zombihop` at real3d and
real4d). q=24 per decision, hardware-matched noise.

Reference optima are **supported surrogate peaks** (GP over the whole campaign,
prominence 0.3, each peak required to have a real measured sample within r=0.05 at a
comparable value). They are NOT hardware-validated optima.

Full paired tables with CIs and per-seed win/tie/loss counts: `STATS.md`.
Per-arm provenance: `provenance.json`. Regenerate with
`python -m benchmarks.zhbench.{stats,report} benchmarks/results/s1_v2`.

---

## Provenance — which core produced which arm

This bundle is deliberately mixed. The core merge at `baa51de` invalidated the
ZoMBI-Hop arms and left the baselines untouched, so re-running all 180 published
cells would have burned ~40 CPU-h to reproduce 120 of them exactly.

| arm group | cells | core | source |
|---|---|---|---|
| `random`, `gp_*` seeds 0–9 | 120 | **77054a9** (`d304c411`/`cd568622`) | `s1_real_20260824_221242` |
| `random`, `gp_ts` seeds 10–19 (3D/4D) | 40 | **baa51de** | `s1_topup_20260901_030257` |
| `zombihop`, `zombihop_nc5` seeds 0–9 | 60 | **baa51de** | `s1_zh_rerun_20260831_223450` |
| `zombihop` seeds 10–19 (3D/4D) | 20 | **baa51de** | `s1_topup_20260901_030257` |
| `zombihop_mz0` seeds 0–9 (3D/4D) | 20 | **baa51de** | `s1_mz0_20260901_022236` |

**Mixing baseline seeds across the two cores is safe, and that was measured rather
than assumed.** Re-running a published baseline cell (`real3d/random/s0`) on the
merged core reproduces every `x_req` and `x_act` value bit-for-bit; `y` differs by
2.2e-16 (one ulp of a double, and plausibly the interpreter rather than the merge);
**every reported metric is identical**, only wall-clock changes. See DESIGN.md §23
for the symbol-level argument this confirms.

---

## What 20 seeds resolve

A claim is **resolved** only when a paired t test *and* an exact sign test both
clear p<0.05. Recall moves in steps of `1/n_true` — 0.0714 at 3D — so a 0.02
difference is not a small effect, it is no effect plus rounding.

| claim | verdict |
|---|---|
| Standard BO recovers **fewer** optima than uniform random at 4D | **resolved for all three**: `gp_qlogei` +0.082 (8/2/0, p=0.002/0.008), `gp_qucb` +0.056 (7/3/0, p=0.007/0.016), `gp_ts` +0.048 (14/3/3, p=0.013/0.013) |
| ZoMBI-Hop's input cost is far below every baseline | **resolved** — no seed overlap at any dimension |
| ZoMBI-Hop is faster than `gp_qucb` / `gp_qlogei` at 3D/4D | **resolved** — but `gp_ts` is faster than ZoMBI-Hop everywhere |
| **ZoMBI-Hop recovers more optima than random at matched \|S\|, real3d** | **NOT SUPPORTED.** +0.0036, 8W/3T/9L, p=0.874/1.000 |
| ZoMBI-Hop vs any baseline, any landscape, matched \|S\| | **nothing resolves** |
| `n_consecutive_converged` tuned-vs-5 | **not resolved** anywhere (4D: 7/2/1, p(sign)=0.070) |

### The claim that did not survive

The previous bundle's headline — `zombihop` > `random` at matched |S| on real3d —
was +0.0857, 8W/1T/1L, p(t)=0.009, p(sign)=0.039, the only resolved
**ZoMBI-Hop-versus-baseline** result in the suite. (Three other method-vs-method
pairs resolved in that bundle, all of them baselines beating a ZoMBI-Hop arm or each
other: `random` > `gp_qlogei` at real4d, and `gp_qucb` / `gp_qlogei` > `zombihop_nc5`
at real4d.) It is now **+0.0036, 8W/3T/9L, p=0.874**.

It decayed in two steps, and both are attributable:

| | real3d, `zombihop` − `random` at matched \|S\| |
|---|---|
| published core, 10 seeds | **+0.0857**, 8/1/1, p(sign)=0.039 — resolved |
| merged core, 10 seeds | +0.0429, 6/1/3, p(sign)=0.508 |
| merged core, **20 seeds** | **+0.0036**, 8/3/9, p(sign)=1.000 |

The first step is the core, **not** the change in matched |S| (6→5): computing both
cores at both |S| gives identical results within each core (DESIGN.md §27). The
second step is simply seed count. So the original finding was part declaration bug,
part seed luck.

**On the fixed core, at 20 seeds, ZoMBI-Hop and uniform random are indistinguishable
at recovering many optima on real3d, and `random` leads on real4d** (−0.044,
5W/2T/13L, p(t)=0.059 — not resolved, but the sign is against us).

---

## What the benchmark does establish

### 1. Standard BO is worse than random at multi-optimum recovery (real4d)

The strongest result in the suite, and it is Aleks's prediction. At matched |S|=11,
`random` beats **all three** GP baselines, all resolved:

| vs `random` | mean diff | W/T/L | p(t) | p(sign) |
|---|---|---|---|---|
| `gp_qlogei` | +0.0815 | 8/2/0 | 0.002 | 0.008 |
| `gp_qucb` | +0.0556 | 7/3/0 | 0.007 | 0.016 |
| `gp_ts` | +0.0481 | 14/3/3 | 0.013 | 0.013 |

Cumulative best-y would have hidden this entirely. **It reverses at 3D**, where the
GP baselines lead random (`gp_qucb` −0.050, p(sign)=0.062, not resolved) — so the
claim needs the dimension attached, and real3d is the least discriminative landscape
we have (peak rarity 0.0349).

### 2. Input cost — the largest and most consistent gap

| | real3d | real4d | real6d |
|---|---|---|---|
| `zombihop` | **111 ± 6** | **62 ± 5** | **61 ± 3** |
| cheapest baseline (`gp_ts`) | 367 ± 48 | 723 ± 46 | 816 ± 8 |
| `random` | 1002 ± 14 | 988 ± 10 | 902 ± 7 |
| ratio vs cheapest / vs random | 3.3× / 9.0× | 11.6× / 15.9× | 13.3× / 14.7× |

No seed overlaps at any dimension.

**Wall-clock is not a uniform win and should not be quoted as one.** ZoMBI-Hop is
4–5× cheaper than `gp_qucb` and `gp_qlogei` at 3D and 4D (470 s vs 2374 s / 1960 s at
3D) — but `gp_ts` is a GP baseline too, and it is **faster than ZoMBI-Hop at every
dimension** (208 s / 284 s / 445 s against 470 s / 470 s / 2480 s). At 6D ZoMBI-Hop is
the slowest arm in the suite.

This is the result the project can defend without qualification, and it is the one
SnAKe framing was brought in for. It currently has **no figure**.

### 3. Search versus declaration, bounded

On real3d ZoMBI-Hop's samples reach 94% of the true optima while it declares 21%.
The gap is real. What the gate experiment shows is *where* it can be converted —
see below.

---

## The declaration gate — pre-registered, and split

`zombihop_mz0` opens the gate (`min_zoom_for_needle = 0`) and changes nothing else.
The three-clause prediction was written into `configs/s1_mz0.yaml` **before the run**:
declarations rise, **declared recall rises with them**, precision falls.

| | `zombihop` → `mz0` | W/T/L | p(sign) |
|---|---|---|---|
| real3d `n_declared` | 4.60 → 7.40 | 7/1/2 | 0.180 |
| real3d `peak_ratio` | 0.236 → **0.321** | 7/2/1 | 0.070 |
| real3d `precision` | 0.645 → 0.667 | 2/2/6 | 0.289 |
| real4d `n_declared` | 11.20 → 12.90 | 7/3/0 | **0.016** |
| real4d `peak_ratio` | 0.148 → **0.148** | 4/2/4 | **1.000** |

`zombihop_mz0` ran on seeds 0–9 only, so every row above is the paired test on those ten
shared seeds — the `zombihop` 3D mean is 0.236 there, against 0.207 over all 20 seeds.

At **3D** all three clauses hold directionally and `mz0` reaches the highest declared
recall of any ZoMBI-Hop arm (0.321, above even the pre-merge 0.271) at comparable
precision. Nothing resolves at 10 seeds.

At **4D** declarations rise (resolved) while declared recall moves **exactly zero**.
The extra declarations are not landing on optima.

**Read on clauses 1 and 3 alone, 4D would have looked like a clean confirmation.**
It is the opposite — which is why clause 2 was pre-registered.

The reading the reach numbers support: **the gate suppresses real finds only where
the samples have already covered the optima.** real3d reach is 0.94, so there is
discovered-but-undeclared structure to convert; real4d is 0.56, so opening the gate
mostly emits non-optima. At matched |S|, `mz0` and `zombihop` are indistinguishable
(−0.014, 5/1/4) — the difference is entirely in *declaring*, not searching, which is
the declaration-budget claim stated precisely.

**Actionable**: at 3D, `min_zoom_for_needle = 0` is the better setting. It is
untested at 6D, where reach is 0.013 and the prediction is that it is purely harmful.
`origin/brianna-v2` deletes this gate globally, so expect it to help at 3D and add
noise at 4D/6D — a dimension-dependent result, not instability.

---

## `n_consecutive_converged`

The `zombihop` arm does **not** run a uniform value: it takes the tuned JSON for that
dimension. Read from the artifacts, not assumed:

| | real3d | real4d | real6d |
|---|---|---|---|
| `zombihop` | **1** (`3d.json`) | 2 (`4d.json`) | 2 (`6d_ensemble.json`) |
| `zombihop_nc5` | 5 | 5 | 5 |

So it is **1-vs-5 at 3D** and **2-vs-5 at 4D/6D**; these must not be pooled. Neither
resolves on the merged core (3D +0.057, p(sign)=0.289; 4D +0.056, p(sign)=0.070; 6D
+0.003, p(sign)=0.625). On the published core the 4D comparison *was* resolved
(9/1/0, p=0.004) — another claim the core fix removed.

---

## real6d is still budget-limited

`reached_ratio` at N=250/500/1000/2000 for `random`: 0.004 → 0.006 → 0.006 → 0.026,
still climbing steeply with no plateau. N=2000 is 83 lines against the 109 lines /
2616 samples the real campaign used for its 68 optima. **No 6D claim is meaningful
yet.** The N≈6000 run is specified in `benchmarks/docs/ORCD.md` with its decision
rule fixed in advance.

Note the merge made 6D much cheaper: `zombihop` 4784 s → 2480 s per cell (0/0/10).

---

## Figures

- `fig3_matched_declarations.png` — **the one that matters.** Recall vs how many
  optima each method declares; star marks ZoMBI-Hop's own count.
- `fig1_reached_ratio.png` — sample efficiency, ±1 std.
- `fig6_dist_to_needles.png` — distance vs budget **at matched |S|**. Read only at
  matched |S|: the Hungarian cost is capped at 0.5, so comparing raw declared sets
  reproduces the unfairness fig3 exists to remove.
- `fig4_needles_declared.png` — declaration budget over time.
- `fig2_endpoint.png` — end-of-budget bars. **Its `dist_to_needles` panel is the
  unmatched-|S| view and will mislead**; prefer fig6.

## What this means for the write-up

The honest framing is no longer "ZoMBI-Hop finds more optima". It is:

> ZoMBI-Hop recovers optima **as well as** uniform random and standard BO at matched
> declarations, at **an order of magnitude lower input cost** and a quarter of the
> compute — while standard BO is measurably **worse than random** at 4D. Its
> declaration budget, not its search, limits what it reports, and opening that gate
> helps exactly where sampling has already covered the optima.

The previous bundle (`benchmarks/results/s1_real/`) is retained as the pre-merge
archive. **Its ZoMBI-Hop rows must not be quoted**; its baseline rows are still valid
and are carried forward here.
