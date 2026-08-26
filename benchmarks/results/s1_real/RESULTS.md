# s1_real -- results

180/180 cells, 0 failures. 3 real campaigns x 6 methods x 10 seeds, N=2000 samples,
q=24 per decision, hardware-matched noise. Generated from `benchmarks/zhbench/configs/s1_real.yaml`.

Reference optima are **supported surrogate peaks** (GP over the whole campaign,
prominence 0.3, each peak required to have a real measured sample within r=0.05
at a comparable value). They are NOT hardware-validated optima.

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

| method | post-hoc recall @|S| | declared recall | precision | input cost | s/cell |
|---|---|---|---|---|---|
| `random` | **0.257** | 0.557 | 0.557 | 1000 | 1 |
| `gp_qucb` | **0.286** | 0.543 | 0.543 | 1473 | 2374 |
| `gp_qlogei` | **0.314** | 0.550 | 0.550 | 1089 | 1960 |
| `gp_ts` | **0.321** | 0.514 | 0.514 | 359 | 211 |
| `zombihop` | **0.343** | 0.271 | 0.672 | 112 | 509 |
| `zombihop_nc5` | **0.307** | 0.179 | 0.835 | 108 | 465 |

### real4d (n_true=27), matched |S|=15

| method | post-hoc recall @|S| | declared recall | precision | input cost | s/cell |
|---|---|---|---|---|---|
| `random` | **0.237** | 0.311 | 0.311 | 988 | 1 |
| `gp_qucb` | **0.178** | 0.241 | 0.241 | 1340 | 1819 |
| `gp_qlogei` | **0.148** | 0.215 | 0.215 | 1152 | 2592 |
| `gp_ts` | **0.181** | 0.285 | 0.285 | 727 | 299 |
| `zombihop` | **0.219** | 0.185 | 0.333 | 60 | 658 |
| `zombihop_nc5` | **0.237** | 0.093 | 0.428 | 43 | 423 |

### real6d (n_true=68), matched |S|=15

| method | post-hoc recall @|S| | declared recall | precision | input cost | s/cell |
|---|---|---|---|---|---|
| `random` | **0.016** | 0.029 | 0.029 | 902 | 1 |
| `gp_qucb` | **0.016** | 0.029 | 0.029 | 1114 | 1755 |
| `gp_qlogei` | **0.019** | 0.034 | 0.034 | 981 | 2509 |
| `gp_ts` | **0.012** | 0.015 | 0.015 | 816 | 445 |
| `zombihop` | **0.010** | 0.007 | 0.034 | 62 | 4784 |
| `zombihop_nc5` | **0.010** | 0.007 | 0.051 | 60 | 2913 |

## Search vs declaration

| real3d | reached_ratio (samples land on optima) | peak_ratio (declared) |
|---|---|---|
| `random` | 0.993 | 0.557 |
| `zombihop` | 0.957 | 0.271 |

ZoMBI-Hop's samples reach 96% of the true optima against random's 99%, but it
declares only 27%. The search is finding them; the declaration budget is the
bottleneck (an activation needs 4-6 lines before it may declare anything).

## Budget curves -- is 6D under-budgeted?

peak_ratio / reached_ratio at each checkpoint:

| objective | method | N=250 | N=500 | N=1000 | N=2000 |
|---|---|---|---|---|---|
| real3d | `random` reached | 0.600 | 0.871 | 0.943 | 0.993 |
| real3d | `zombihop` reached | 0.571 | 0.793 | 0.900 | 0.957 |
| real4d | `random` reached | 0.107 | 0.230 | 0.363 | 0.581 |
| real4d | `zombihop` reached | 0.100 | 0.193 | 0.341 | 0.507 |
| real6d | `random` reached | 0.004 | 0.006 | 0.006 | 0.026 |
| real6d | `zombihop` reached | 0.001 | 0.003 | 0.003 | 0.007 |

real3d saturates by N=1000 (random reached 0.943 -> 0.993). real6d is still
climbing steeply at N=2000 and needs a much larger budget before any 6D claim
is meaningful: 68 optima, 83 lines, against the 109 lines the real campaign used.

## Figures

- `fig3_matched_declarations.png` -- **the one that matters.** Recall vs how many
  optima each method declares; star marks ZoMBI-Hop's own declared count.
- `fig1_reached_ratio.png` -- sample efficiency, optima reached vs budget.
- `fig4_needles_declared.png` -- ZoMBI-Hop's declaration budget over time.
- `fig2_endpoint.png` -- end-of-budget bars.

Raw per-run artifacts (points.csv, declared_optima.csv, metrics.json per cell) are
under `benchmarks/runs/`, which is gitignored -- ask Colin if you want them.
