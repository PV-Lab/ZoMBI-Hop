# Upstream requests

Things the benchmark needed that belong in the core, and things it found that
affect the core's other consumers. Nothing here is benchmark-only: each item was
worked around inside `benchmarks/` so the core stayed untouched, but each is worth
fixing at the source.

## 1. Import-time mutation of global torch state — affects everyone

`src/core/zombihop.py:29-35`

```python
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)
```

Importing the module changes global torch defaults for the whole process. Any
tensor built afterwards without an explicit `device=`/`dtype=` silently lands on
the GPU in **float32**, including in code that has nothing to do with ZoMBI-Hop.
`optimize/run_mobo.py` imports the core transitively, so this reaches every
consumer of `run_mobo` — including the MOBO GP in the hyperparameter tuner, which
otherwise works in float64.

It never fires on a CPU-only machine, which is why it is easy to miss: the
benchmark ran clean locally and would have changed numerical precision the first
time it touched ORCD.

Suggested fix: move the CUDA settings into `ZoMBIHop.__init__` (or an explicit
`configure_cuda()` the entry points call), so importing a module does not
reconfigure the process. Worked around at `benchmarks/zhbench/_repo.py`
(`preserve_torch_defaults`), which snapshots and restores around the import.

## 2. `default_hparams.py` has drifted from the file it says it copies

`src/default_hparams.py:12-14` states `DEFAULT_HPARAMS` is "copied verbatim from
`optimize/hparams/6d_ensemble.json`". They disagree:

| key | `optimize/hparams/6d_ensemble.json` | `src/default_hparams.py` |
|---|---|---|
| `n_consecutive_converged` | **2** | **5** |

Everything else matches. The JSON's 2 is the newer value (loosened after the 6-D
campaign, commit `c4a9358`, alongside the retroactive needle refit in
`src/core/retro.py`); `default_hparams.py` still carries the pre-campaign 5.

This matters because the two files feed different consumers: the GUI
(`interface/app.py`) and the hardware runner (`scripts/run_zombi_main.py`) read
`DEFAULT_HPARAMS`, so a deployed run and a tuned run currently disagree about when
a needle has converged.

The benchmark uses the JSON as `zombihop` and runs 5 as the `zombihop_nc5`
sensitivity, so it will report which one is actually better on these landscapes.

## 3. `CAMPAIGNS` has no d=6 entry

`warm_start/warm_gp_landscape.py:110-119` covers d=3 and d=4 only. The 6-D
campaign (`data/4th_real_run.db`, 2423 rows / 111 iterations, 2042 scored across
109 lines, components `FAPbI3, CsPbI3, FAPbBr3, MACl, MAPbI3, MAPbBr3`) has no
entry, so `fullgp_objective(6)` raises.

Adding it upstream also needs a peak finder that does not use `simplex_grid`: the
lattice is O(n^(d-1)) and is already coarse at d=4 (`GRID_4D = 42`). The benchmark
uses a probe-cloud local-maximum search instead
(`benchmarks/zhbench/landscapes.py:detect_supported_peaks`), which is
dimension-agnostic and reproduces the lattice result where both apply.

**Open question for Brianna:** was the 6-D campaign run with a per-component
`bounds` box? The data are consistent with the full simplex — of 218 line
endpoints only 7 sit within 0.02 of a component maximum (exactly one per
component, i.e. only the point that defines it), while 92 endpoints have a
component at zero. LineBO presses against the zero faces constantly and never
against an upper face, which a real box would have produced. The benchmark
therefore builds `real6d` on the full simplex; if a box was used, it is a one-line
change.

## 4. Peak detection admits GP artifacts as "true optima"

`warm_gp_landscape._PEAK_PROMINENCE_FRAC = 0.12` keeps any local maximum rising
12% of the way from the GP median to the max. On the real campaigns most survivors
are GP wiggle along measured lines rather than material optima: at 0.12 only 15%
(3-D) and 46% (4-D) of the detected peaks stand above the 99th percentile of
uniform random sampling.

Since these peaks are the `dist_to_needles` reference that deployed
hyperparameters are scored against, tuning is partly being steered toward
recovering artifacts. The benchmark uses `prominence_frac = 0.3` plus a
**measured-support** rule — a peak needs at least one real campaign sample within
`r` whose measured objective is close to the peak's predicted value — and reports
`n_true` under both settings so the effect is visible.
