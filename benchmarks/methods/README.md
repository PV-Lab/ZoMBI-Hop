# `benchmarks/methods` — optimisers behind one interface

Every optimiser the benchmarks compare — ZoMBI-Hop and the standard baselines —
implements one small interface and is measured through one shared `Problem`, so
they spend identical budgets under an identical noise model and are scored the
same way.

| name | method | notes |
|---|---|---|
| `zombi_hop` | ZoMBI-Hop (`src/`) on `BoxDomain()` | measures LineBO lines of `batch_size` points; declares its own needles |
| `random` | random sampling | `sampler="uniform"` (default) or `"sobol"` |
| `gp_bo` | standard BO | BoTorch `SingleTaskGP` (stock priors) + batch qLogNEI, sequential greedy |
| `turbo` | TuRBO-m, [Eriksson et al. 2019](https://arxiv.org/abs/1910.01739) | `n_trust_regions=1` (TuRBO-1) by default; paper constants throughout |
| `hebo` | HEBO, [Cowen-Rivers et al. 2022](https://arxiv.org/abs/2012.03826) | the authors' `hebo` 0.3.6 package, vendored (below) |

```bash
python -m benchmarks.methods list            # methods and every config key
python -m benchmarks.methods install-hebo    # once per checkout
# model-fitting commands: run on a compute node, e.g. via
# benchmarks/scripts/validate_methods.sbatch — not on the login node
python -m benchmarks.methods smoke --methods turbo,hebo --device cuda
python -m benchmarks.methods calibrate --device cuda --out some/dir
```

## The interface

**`Problem`** (`base.py`) is what a method sees: `dim`, the unit box, `budget`
(points, initial design included), the preferred `batch_size`, and
`evaluate(X) -> (X_actual, Y)`. It owns the measurement model

```
x_actual = clip(x_requested + N(0, input_noise^2), 0, 1)     # input_noise default 0
y        = f(x_actual) * (1 + N(0, output_noise_frac^2))      # 0.045, run_mobo's value
```

and the budget. `evaluate` raises `BudgetExhausted` / `TimeLimitReached`
(both `BaseException`, so no optimiser's `except Exception` can swallow them) before
measuring anything it may not. The noiseless objective and the true optima are not
reachable through it.

**`Method`** is an optimiser: list every tunable in `defaults`, implement
`run(problem)`. Most are ask/tell, so subclass `AskTellMethod` instead:

```python
import numpy as np
from benchmarks.methods import AskTellMethod, register

@register
class MyMethod(AskTellMethod):
    name = "my_method"                  # --methods my_method; also the cell directory
    description = "one line for `list`"
    defaults = {"n_init": 48}           # unknown keys in a config are a TypeError

    def setup(self, problem):           # once, before the first ask
        self.dim = problem.dim

    def ask(self, n) -> np.ndarray:     # (n, dim) points in [0, 1]^dim
        return self.rng.random((n, self.dim))

    def tell(self, X_requested, X_actual, Y):   # Y is noisy; MAXIMISE it
        ...
```

Put it in this package and import it at the bottom of `__init__.py`, or leave it
anywhere and pass it by path — `--methods path/to/file.py:MyMethod` works in the
sweep with no registration. `self.seed`, `self.rng` and `self.device` are set for
you. Optional hooks: `summary()` (scalars into `metrics.json`),
`write_artifacts(dir)` (extra files — TuRBO writes its trust-region log),
`point_columns()` (extra `points.csv` columns). A method that declares its own
optima sets `declares_needles = True` and implements `declared_needles()`.

## Scoring: needles and the extractor

The headline metric, `dist_to_needles` (`optimize/eval_metrics.py`, unchanged),
scores a *declared set* of optima one-to-one against the true set; over- and
under-declaring are charged alike. Only ZoMBI-Hop declares a set. For every other
method a **needle extractor** reads one out of its samples, using only what the
method measured (no landscape knowledge). The runner also applies the same
extractor to *every* method's samples, ZoMBI-Hop included
(`dist_to_needles_extracted`), so there is always one comparison in which the
methods differ only in where they sampled.

Alongside it, **`greedy_dist`** scores the *samples*: for each true optimum the
distance to the nearest point the method measured, averaged over the optima
(`eval_metrics.metric_greedy_dist`). No declaration and no extractor enter it, the
pairing is greedy rather than one-to-one (two optima may share a sample — a
measurement is not a claim), and there is no unmatched penalty, so it keeps ranking
runs that declared nothing useful. It is `frac_optima_visited`'s distance-valued
sibling: the same per-optimum minima, averaged instead of thresholded at
`MATCH_RADIUS`. Because samples accumulate it can only fall over a budget.

`gp_peaks` (default, `extract.py`) fits a GP, climbs its posterior mean to local
maxima, and keeps a maximum only if the model is confident it stands out from a
shell of probes one to two fitted lengthscales away. The noise scale is floored at
the known instrument noise. The docstring records its calibration (zero false
positives on needle-free landscapes at 48–3000 samples) and its blind spot: a
needle touched once and never revisited reads as a noise spike and is not declared,
which is the same call ZoMBI-Hop's repeatability gate makes. `nms` is a model-free
alternative.

## Files a run writes (`runner.py`)

`points.csv` (every sample, noisy `y` and noiseless `f`), `needles.csv` (what
the cell is scored on), `needles_extracted.csv` (declaring methods),
`metrics_over_time.csv` (one row per batch; extractor columns every `trace_every`
batches), `method.json` (full config, seed, extractor), `metrics.json` (written
last, atomically: the completion marker), and `error.log` if the method raised.

## HEBO is vendored

`pip install HEBO` would downgrade numpy 2.4 → 1.24 (and scipy, pandas,
matplotlib) in the shared `.venv` the MOBO fleet runs from. `install-hebo`
installs `hebo`, `pymoo` 0.6.1.3, `disjoint-set` and `alive-progress` with
`--no-deps` into `benchmarks/methods/_vendor/` (git-ignored), and deletes pymoo's
compiled extensions, which need a newer libstdc++ than the cluster's (pymoo falls
back to pure Python). `.venv` is untouched.
