# Dimension Scaling of ZoMBI-Hop Hyperparameters

## Why this exists

The ZoMBI-Hop hyperparameters were tuned by on a **3-component (3-simplex)**
problem. The geometry of the search space changes with the dimension, so several hyperparameters that are expressed in absolute units
(distances, radii, sample counts) silently change meaning as the dimension grows.

To make a 3-D-tuned configuration transfer to higher dimensions, we rescale the
dimension-sensitive hyperparameters by an explicit function of the problem
dimension `d`.

We divide the hyperparameters into 2 groups that scale differently with dimension.



## Group 1: distances & radii: scale by `√((d−1)/2)`

**Parameters:** `paring_spatial_halfnoise`, `max_penalty_radius`,
`min_axis_noise_mult`.

These are all **lengths in tangent (ILR) space**: a paring/deduplication radius, a
penalty-ellipsoid radius cap, and the semi-axis floor that decides when a needle
has shrunk "into the noise". Each is, directly or after multiplying by
`input_noise_ilr`, a radius measured in the `(d − 1)`-dimensional ILR space.

**First-principles argument.** Take an isotropic noise ball — independent
perturbations of scale `σ` along each of the `n = d − 1` tangent axes. The
expected squared distance from the centre is

```
E[‖z‖²] = Σ_{i=1}^{n} E[z_i²] = n · σ²
⇒ typical radius  ‖z‖ ≈ σ · √n.
```

So the natural length scale of the problem — how far "one noise unit" reaches —
grows like `√n = √(d − 1)`. A radius that captures, say, the local noise cloud
or a typical needle basin in 3-D will capture a **vanishingly thin shell** of the
same cloud in 10-D if left at its absolute 3-D value. To keep these radii at a
constant number of *noise units* (constant relative coverage), they must grow
like `√(d − 1)`.

**Normalisation to `d = 3`.** We divide by the `d = 3` value of `√(d−1)`, i.e.
`√2`, so the factor is 1.0 at the tuning dimension:

```
g₁(d) = √((d − 1) / 2)
```

| `d`  | `g₁ = √((d−1)/2)` |
|------|-------------------|
| 3    | 1.00              |
| 4    | 1.22              |
| 10   | 2.12              |

This is the "roughly 2× at 10-D" behaviour we want: penalty balls, paring
distances, and the needle-stop floor all widen by ~2× by the time we reach the
10-simplex, matching the ~2× growth in the ambient length scale.


## Group 2 — sampling budgets: scale linearly by `(d−1)/2`

**Parameters:** `n_restarts`, `raw` (acquisition-optimiser restart and raw-sample
counts), the per-iteration LineBO line count (`num_lines` / `NUM_LINES`), and the
initial-seeding line count (`N_INIT_LINES` / `NUM_INIT_DATA`).

These are **counts of samples/restarts** used to *cover* the search space and to
escape local optima during acquisition optimisation. The relevant quantity here
is not a length but a **volume**, and volume is where dimensionality bites
hardest.

**First-principles argument.** To maintain a fixed sampling *density* — a fixed
expected number of samples within any fixed-radius ball — the number of samples
needed to cover a region grows **exponentially** in dimension:

```
N_needed ∝ (1/ε)^n,      n = d − 1.
```

The honest answer is therefore *exponential* growth. In practice we cannot
afford to multiply our restart/sample/line budgets by `~5^(d−3)`; that would make
a 10-D run hundreds of times more expensive than a 3-D run and dominate the wall
clock. So we deliberately settle for a **linear** compromise: spend more budget
as the dimension grows, enough to keep the optimiser from collapsing, but bounded
by what is computationally tractable.

**Normalisation to `d = 3`.** We use the *square* of the Group-1 factor, which is
linear in `d` and (like Group 1) equals 1.0 at `d = 3`:

```
g₂(d) = g₁(d)² = (d − 1) / 2
```

| `d`  | `g₂ = (d−1)/2` |
|------|----------------|
| 3    | 1.0            |
| 4    | 1.5            |
| 10   | 4.5            |

Using `g₁²` is a deliberate, tidy choice: it ties the two groups together (area
scales as length²) and gives a single coherent story — *linear where we can
afford it, even though the geometry would prefer exponential.* Counts are rounded
to the nearest integer (and floored at 1 for line budgets).

> **Caveat, stated plainly.** Linear scaling under-provisions in high dimension
> relative to the true exponential requirement. It is a budget-bounded heuristic,
> not a guarantee of constant coverage. If high-`d` runs show degraded needle
> discovery, the sampling budgets (Group 2) are the first place to spend more.


## Summary table

| Parameter | Group | Meaning | Factor | ×@`d=10` | Scaled in |
|-----------|:-----:|---------|--------|:--------:|-----------|
| `paring_spatial_halfnoise` | 1 | ILR paring/dedup radius | `√((d−1)/2)` | 2.12 | `ZoMBIHop.__init__` |
| `max_penalty_radius`       | 1 | penalty-ellipsoid radius cap | `√((d−1)/2)` | 2.12 | `ZoMBIHop.__init__` |
| `min_axis_noise_mult`      | 1 | needle semi-axis noise floor | `√((d−1)/2)` | 2.12 | `ZoMBIHop.__init__` |
| `n_restarts`               | 2 | acquisition-opt restarts | `(d−1)/2` | 4.5 | `ZoMBIHop.__init__` |
| `raw`                      | 2 | acquisition raw samples  | `(d−1)/2` | 4.5 | `ZoMBIHop.__init__` |
| `num_lines` (`NUM_LINES`)  | 2 | LineBO lines per iteration | `(d−1)/2` | 4.5 | `LineBO.__init__` |
| `N_INIT_LINES` / `NUM_INIT_DATA` | 2 | initial seeding lines | `(d−1)/2` | 4.5 | each init generator |

All factors are **1.0 at `d = 3`**, so a 3-D-tuned configuration is reproduced
exactly at the dimension it was tuned on, and only departs from it as the problem
grows.


### Budget Scaling

| Budget | d=3 | d=10 | Change |
|--------|:---:|:----:|:------:|
| Initial seeding lines (`N_INIT_LINES` / `NUM_INIT_DATA`) | 2 | 9 | **4.5× (+7 lines)** |
| Acquisition raw samples (`raw`, at default 500) | 500 | 2250 | **4.5× (+1750 samples)** |
| Acquisition restarts (`n_restarts`, at default 30) | 30 | 135 | **4.5×** |
| LineBO lines per iteration (`num_lines`, base 10) | 10 | 45 | **4.5×** |

