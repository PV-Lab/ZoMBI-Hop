# ZoMBI-Hop Ellipsoid — Complete Technical Reference

> **Version:** ellipsoid branch (Edit 1 / 2A / 2B applied May 2026)  
> **Location:** `ZoMBI-Hop-LineBO-working/src/`  
> **Last updated:** May 2026 — reflects current code state

---

## Table of Contents

1. [The Problem](#1-the-problem)  
2. [The Algorithm — High-Level Story](#2-the-algorithm--high-level-story)  
3. [Operational Flow Diagram](#3-operational-flow-diagram)  
4. [All Hyperparameters & What They Control](#4-all-hyperparameters--what-they-control)  
5. [File-by-File Reference](#5-file-by-file-reference)  
   - [utils/simplex.py](#51-utilssimplexpy)  
   - [utils/dataclasses.py](#52-utilsdataclassespy)  
   - [utils/datahandler.py](#53-utilsdatahandlerpy)  
   - [utils/gp_simplex.py](#54-utilsgp_simplexpy)  
   - [core/linebo.py](#55-corelinebopy)  
   - [core/zombihop.py](#56-corezombihoppy)  
   - [utils/visualization.py](#57-utilisvisualizationpy)  
6. [Key Mathematical Derivations](#6-key-mathematical-derivations)  
7. [Benefits, Drawbacks, and Known Failure Modes](#7-benefits-drawbacks-and-known-failure-modes)  
8. [Recent Algorithm Edits (May 2026)](#8-recent-algorithm-edits-may-2026)  

---

## 1. The Problem

ZoMBI-Hop solves **multi-modal Bayesian optimization over a probability simplex**.

The canonical use case is **materials composition search**: given *d* components (e.g. Pb, Sn, Br, I in a perovskite), each composition is a point on the *d*-simplex Δ^d = { x ∈ R^d | Σ x_i = 1, x_i ≥ 0 }. The objective (e.g. solar cell efficiency, bandgap) may have **multiple local optima** — "needles in a haystack" — at very different compositions. Standard BO finds one optimum and parks there. ZoMBI-Hop systematically discovers *all* significant optima.

**Why the simplex is hard:**
- It is a curved, bounded manifold — Euclidean BO ignores both constraints.  
- Components near zero cause log-ratio coordinates to diverge.  
- Multiple optima may be separated by flat low-value regions, not just local barriers.  
- Noise in robotic synthesis means the actually-deposited composition (X_actual) deviates from the requested one (X_expected).

---

## 2. The Algorithm — High-Level Story

ZoMBI-Hop operates in **activations** (outer loop) and **zooms** (inner loop).

### The Core Idea

Each activation attempts to find and declare one new local optimum ("needle"). Once declared, that region is **permanently excluded** via an ellipsoidal penalty so the next activation explores elsewhere. This continues until the user-specified number of activations is reached or the entire simplex is covered.

### Within an Activation: Zoom-In Search

1. **Start at full simplex bounds** (or updated bounds after a failure-retry, see below).
2. Fit a **Gaussian Process** in ILR-transformed space. When the bounds are not global (i.e. we are inside a zoom), **only points within the current bounds** are used for GP training: first the pared (deduplicated) points within bounds fill the `max_gp_points` budget, then raw unpenalized points within bounds fill the remainder. This makes the GP posterior tight and locally accurate so that PI drops quickly on genuine convergence.
3. Run **LineBO** to propose a candidate: sample random chords through the current bounds, score each chord by integrating the acquisition function along it (mean, not max), pick the best chord, and have the physical experiment evaluate that line.
4. After each evaluation, check **convergence**: PI (probability of improvement) is computed against the **local best** (best unpenalized point within the current zoom bounds, and `best_f` from the local GP training set). If PI drops below threshold AND the Y improvement since the previous local best is within output noise, increment a consecutive-converged counter.
5. After `n_consecutive_converged` consecutive converged iterations, declare a needle.
6. Before the next zoom: **Jaccard-aware sliding-window bounds selection** (`determine_new_bounds`) slides over windows of top-M points ranked by Y, picking the first candidate AABB whose Jaccard overlap against the last `jaccard_window` entries in `bounds_history` is ≤ `jaccard_threshold`. If all windows exceed the threshold, the least-overlapping window is used.
7. **Secondary Jaccard guard (MC):** if the selected AABB still has Monte Carlo Jaccard > `zoom_jaccard_threshold` with any entry in `zoom_bounds_history`, the algorithm **force-declares a needle** at the current best unpenalized point — repeated zooms to the same region are taken as evidence the optimum is there even if PI hasn't formally converged.
8. Otherwise, zoom in on the new AABB and continue.

### Needle Declaration

On needle declaration (either via PI convergence or Jaccard force-declare):
1. Fit a **clean local GP** using `fit_with_locality`: selects only data within `max_radius` of the needle first, then fills to `max_gp_points` with remaining points. This gives a locally-accurate Hessian uncontaminated by far-away observations.
2. Compute the **Hessian of the clean base acquisition** (no repulsive penalties) at the needle in ILR space via `create_clean_acquisition`. The penalty-free acquisition gives the true local curvature rather than one distorted by proximity to other needles.
3. Derive an **ellipsoidal exclusion zone** M from the Hessian: the region where the acquisition has dropped by a specified fraction from its peak.
4. Record the needle, storing both the **peak value** and the **median Y of all raw observations within `paring_spatial_halfnoise × input_noise_ilr`** of the needle in ILR space. The median is a noise-robust measure of the local optimum's true value.
5. Update the penalty mask, reset to full simplex for the next activation.

### Post-Activation Pared Relabeling

After every activation completes (needle found or failed), each entry in the pared dataset has its Y value replaced by the **median of all Y_all values whose X_all falls within `paring_spatial_halfnoise × input_noise_ilr`** in ILR space. This smooths noise spikes so the GP in subsequent activations trains on a clean, representative signal rather than individual noisy measurements.

### Failure Handling (Three-Way Dispatch)

The zoom loop is a `while current_zoom < max_zooms` loop rather than a fixed `for zoom` loop, which allows retrying the **same zoom level** on failure without advancing. Two per-activation flags track state: `first_failure_handled` and `data_added_since_last_failure`.

When an activation fails (candidate is None, all points penalized, or force-declare returned None):

| Condition | Action |
|-----------|--------|
| **First failure** (`not first_failure_handled`) | Recompute all needle ellipsoids using the clean local GP (`recompute_all_ellipsoids`): for each needle, `fit_with_locality` → `create_clean_acquisition` → `determine_penalty_ellipsoid`. Update `dh.needle_M_list`. Retry same zoom. |
| **Subsequent failure, data was added** (`first_failure_handled AND data_added_since_last_failure`) | Recompute zoom bounds Jaccard-aware: `dh.determine_new_bounds(add_to_history=False)`. The `add_to_history=False` prevents failure-retry bounds from contaminating the history used by success-path zooms. Retry same zoom. |
| **Subsequent failure, no new data** (`first_failure_handled AND NOT data_added_since_last_failure`) | Shrink all ellipsoid semi-axes: `dh.shrink_all_needle_radii(bounds_shrink_factor)`. If `max_needle_radius() < min_axis_noise_mult × input_noise_ilr`, all exclusion zones are at noise scale → stop. Otherwise retry same zoom. |

`data_added_since_last_failure` is set `True` immediately after `_objective_wrapper` returns (regardless of whether points are penalized), and reset to `False` after each failure dispatch.

### Point Paring (Deduplication)

Before any point enters the GP training set, it is checked against the **pared dataset** — a noise-deduplicated subset. Two points are considered duplicates if their ILR-space distance is less than `paring_spatial_halfnoise × σ_ILR` AND their Y-values differ by less than `paring_y_noise_multiplier × σ_Y`. Duplicates are handled by a fair coin: keep the old or replace with the new. This prevents noise spikes from dominating the GP. After each activation, all pared Y values are further smoothed via local medians (see above).

---

## 3. Operational Flow Diagram

```
╔══════════════════════════════════════════════════════════════════════════╗
║                         ZoMBI-Hop: run()                                ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  INIT: load X_init, Y_init → build pared set → set bounds = full Δ     ║
║                                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │  OUTER LOOP: activation = 0 … max_activations                  │    ║
║  │                                                                  │    ║
║  │  first_failure_handled = False                                   │    ║
║  │  data_added_since_last_failure = False                           │    ║
║  │  zoom_bounds_history = []                                        │    ║
║  │                                                                  │    ║
║  │  ┌──────── ZOOM LOOP: while current_zoom < max_zooms ──────────┐ │    ║
║  │  │  zoom = current_zoom                                        │ │    ║
║  │  │                                                             │ │    ║
║  │  │  [1] if global bounds: get_gp_data()                       │ │    ║
║  │  │      else: get_zoom_gp_data(bounds)                        │ │    ║
║  │  │      → pared in bounds first, then raw fill                │ │    ║
║  │  │  [2] GPSimplex.fit(X_ilr / ilr_std, Y)                     │ │    ║
║  │  │      best_f_local = Y.max()                                │ │    ║
║  │  │                                                             │ │    ║
║  │  │  ┌──── ITERATION LOOP: iter = 0 … max_iterations ────────┐ │ │    ║
║  │  │  │                                                        │ │ │    ║
║  │  │  │  [3] get_candidate(bounds, best_f=best_f_local)       │ │ │    ║
║  │  │  │      • RepulsiveAcquisition(UCB/EI) on seeds          │ │ │    ║
║  │  │  │      • nat-grad ascent on top-n_restarts seeds        │ │ │    ║
║  │  │  │      → candidate (d,) on simplex                      │ │ │    ║
║  │  │  │                                                        │ │ │    ║
║  │  │  │  if candidate is None → activation_failed, break      │ │ │    ║
║  │  │  │                                                        │ │ │    ║
║  │  │  │  prev_best ← get_best_in_bounds(bounds)               │ │ │    ║
║  │  │  │    (or get_best_unpenalized if global bounds)          │ │ │    ║
║  │  │  │                                                        │ │ │    ║
║  │  │  │  [4] LineBO.sampler(candidate, bounds, acq_fn)        │ │ │    ║
║  │  │  │      • generate num_lines random chords               │ │ │    ║
║  │  │  │      • score by mean acq over 100 pts                 │ │ │    ║
║  │  │  │      → (X_expected, X_actual, Y)                      │ │ │    ║
║  │  │  │  data_added_since_last_failure = True  ← always       │ │ │    ║
║  │  │  │                                                        │ │ │    ║
║  │  │  │  [5] add_all_points() → update pared set,             │ │ │    ║
║  │  │  │       penalty mask, and snapshot                       │ │ │    ║
║  │  │  │                                                        │ │ │    ║
║  │  │  │  [6] refit GP (local data if zoomed)                  │ │ │    ║
║  │  │  │      best_f_local = Y.max()                           │ │ │    ║
║  │  │  │                                                        │ │ │    ║
║  │  │  │  [7] check_convergence(best_f_ref=best_f_local)       │ │ │    ║
║  │  │  │      PI vs local best; ΔY vs prev_best in bounds      │ │ │    ║
║  │  │  │      consecutive_converged++                           │ │ │    ║
║  │  │  │                                                        │ │ │    ║
║  │  │  │  if consecutive_converged >= n_consecutive:            │ │ │    ║
║  │  │  │    [8] _declare_needle_at_best()                      │ │ │    ║
║  │  │  │        • fit_with_locality(needle) → local clean GP   │ │ │    ║
║  │  │  │        • create_clean_acquisition() (no repulsion)    │ │ │    ║
║  │  │  │        • determine_penalty_ellipsoid → M              │ │ │    ║
║  │  │  │        • compute local median Y within σ_ILR          │ │ │    ║
║  │  │  │        • add_needle(value, median_value, M)           │ │ │    ║
║  │  │  │        • reset bounds to full Δ                       │ │ │    ║
║  │  │  │        → break zoom loop (needle found)               │ │ │    ║
║  │  │  └────────────────────────────────────────────────────── ┘ │ │    ║
║  │  │                                                             │ │    ║
║  │  │  ── AFTER ITERATION LOOP ──────────────────────────────    │ │    ║
║  │  │                                                             │ │    ║
║  │  │  if activation_failed:                                      │ │    ║
║  │  │    → _handle_failure_retry() three-way dispatch:           │ │    ║
║  │  │      Case 1 (first failure):                               │ │    ║
║  │  │        recompute_all_ellipsoids (clean local GP each)      │ │    ║
║  │  │        update needle_M_list  → retry same zoom             │ │    ║
║  │  │      Case 2 (subsequent, data added):                      │ │    ║
║  │  │        determine_new_bounds(add_to_history=False)          │ │    ║
║  │  │        update bounds  → retry same zoom                    │ │    ║
║  │  │      Case 3 (subsequent, no data):                         │ │    ║
║  │  │        shrink_all_needle_radii(bounds_shrink_factor)       │ │    ║
║  │  │        if max_radius < min_axis_noise_mult×σ → finished   │ │    ║
║  │  │        else → retry same zoom                              │ │    ║
║  │  │    data_added_since_last_failure = False                   │ │    ║
║  │  │    consecutive_converged = 0                               │ │    ║
║  │  │    continue  ← same current_zoom, retry zoom body         │ │    ║
║  │  │                                                             │ │    ║
║  │  │  if no needle AND no failure AND zoom < max_zooms-1:        │ │    ║
║  │  │    [9a] determine_new_bounds() ← Jaccard sliding window    │ │    ║
║  │  │         slides over Y-sorted unpenalized; picks AABB with  │ │    ║
║  │  │         Jaccard ≤ jaccard_threshold vs bounds_history      │ │    ║
║  │  │    [9b] MC Jaccard guard vs zoom_bounds_history:           │ │    ║
║  │  │         if Jaccard > zoom_jaccard_threshold:               │ │    ║
║  │  │           → _declare_needle_at_best() ← force-declare      │ │    ║
║  │  │           (if no unpenalized pts → activation_failed)      │ │    ║
║  │  │         else: append to zoom_bounds_history; zoom in       │ │    ║
║  │  │    current_zoom += 1  ← advance only on clean success      │ │    ║
║  │  │                                                             │ │    ║
║  │  └─────────────────────────────────────────────────────────── ┘ │    ║
║  │                                                                  │    ║
║  │  _relabel_pared_with_medians()  ← smooth pared Y after activation│    ║
║  │  activation++ → next activation                                  │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
║                                                                          ║
║  return (needle_results, needle_locations, needle_vals, X_all, Y_all)  ║
╚══════════════════════════════════════════════════════════════════════════╝

Data flow between key objects:
─────────────────────────────────────────────────────────────────────────
  ZoMBIHop
    │
    ├─ DataHandler ── owns ALL persistent state
    │     ├── X_all_actual, Y_all                (raw observations)
    │     ├── X_pared, Y_pared                   (deduped GP training set,
    │     │                                        relabeled with medians
    │     │                                        after each activation)
    │     ├── needles, needle_M_list, needle_B    (discovered optima + ellipsoids)
    │     ├── needles_results                     (value + median_value per needle)
    │     ├── _penalty_mask                       (True = not penalized)
    │     ├── bounds, current_zoom_bounds         (trust region)
    │     └── bounds_history                      (recent AABB tensors for Jaccard)
    │
    ├─ GPSimplex ── wraps BoTorch GP
    │     ├── fit(X, Y)              ILR + std-normalize → SingleTaskGP
    │     ├── fit_with_locality()    local-only GP around a needle
    │     ├── create_acquisition()   RepulsiveAcquisition (with needle penalties)
    │     ├── create_clean_acquisition()  penalty-free acquisition for Hessian
    │     ├── recompute_all_ellipsoids()  bulk ellipsoid refit (one local GP per needle)
    │     ├── get_candidate()        raw_samples → nat-grad ascent → best point
    │     └── determine_penalty_ellipsoid(acq_fn=...)  Hessian M in ILR space
    │
    └─ LineBO ── evaluates the physical objective
          ├── ranked_line_endpoints()  → score chords by mean acquisition
          └── sampler()               → call objective_function with ranked endpoints
```

---

## 4. All Hyperparameters & What They Control

### Iteration Structure

| Parameter | Default | Controls |
|-----------|---------|----------|
| `max_zooms` | 3 | Number of zoom levels per activation. More zooms = finer trust regions but more iterations before declaring a needle. Also used as the Jaccard history window size. |
| `max_iterations` | 10 | Max BO iterations per zoom level. Total per activation ≤ max_zooms × max_iterations. |
| `n_consecutive_converged` | 2 | How many successive converged iterations trigger needle declaration. Higher = more conservative (requires sustained convergence). |
| `top_m_points` | auto (d+1) | How many top-Y unpenalized points define the AABB zoom bounds. Small = tight zoom; large = wide. |

### Convergence Thresholds

| Parameter | Default | Controls |
|-----------|---------|----------|
| `convergence_pi_threshold` | 0.01 | PI below this value satisfies the first convergence condition. PI is computed against the **local best within the current zoom bounds** (not global), so this threshold is meaningful relative to the active search region. Lower = harder to converge. |
| `input_noise_threshold_mult` | 2.0 | Unused directly in convergence check (kept for backward compat); related to input noise scale. |
| `output_noise_threshold_mult` | 2.0 | Y improvement (over the **local** prev_best within bounds) must be < this × GP output noise. Higher = easier to converge. |

### GP & Acquisition

| Parameter | Default | Controls |
|-----------|---------|----------|
| `max_gp_points` | 3000 | Cap on GP training set size. When zoomed, this budget is filled first by pared points within bounds, then by raw unpenalized points within bounds. Prevents O(n³) GP cost blowing up. |
| `acquisition_type` | "ucb" | Base acquisition: `"ucb"` (Upper Confidence Bound) or `"ei"` (Log Expected Improvement). |
| `ucb_beta` | 0.1 | Exploration weight for UCB: α(x) = μ + β½σ. Higher = more exploration. |
| `repulsion_lambda` | None (auto) | Strength of ellipsoid repulsion penalty. `None` = auto-computed as max(10×median\|acq\|, 100) each call. |
| `n_restarts` | 30 | Number of nat-grad restarts for acquisition optimization (get_candidate). |
| `raw` | 500 | Initial random samples before selecting restart seeds. |
| `nat_grad_step` | 0.02 | Step size α for natural-gradient ascent: x ← normalize(x ⊙ exp(α(g − ḡ))). |
| `nat_grad_max_steps` | 50 | Max gradient steps per restart. |

### Ellipsoid Penalty

| Parameter | Default | Controls |
|-----------|---------|----------|
| `max_penalty_radius` | 1.0 | Caps the maximum ellipsoid semi-axis length (ILR units). Prevents a single needle from excluding too much of the simplex. |
| `ellipsoid_drop_fraction` | 0.25 | Fraction of peak acquisition value that defines the ellipsoid boundary. Larger = bigger exclusion zone. |
| `ellipsoid_eigenvalue_floor` | 1e-6 | Minimum eigenvalue of –H before computing M. Prevents degenerate flat-Hessian ellipsoids. |

### Point Paring (Deduplication)

| Parameter | Default | Controls |
|-----------|---------|----------|
| `paring_spatial_halfnoise` | 0.5 | Spatial duplicate threshold in ILR units: points within `factor × input_noise_ilr` are candidates for deduplication. Also controls the neighbourhood radius for median relabeling of pared Y values and the needle median value computation. |
| `paring_y_noise_multiplier` | 1.0 | Y-value duplicate threshold: `factor × GP_output_noise`. |
| `input_noise_ilr` | 0.03 | Known physical input noise σ in ILR space (instrument precision). Used for paring, median neighbourhood radius, needle shrink stop condition, and old-needle radius. |

### Needle Shrink & Stop

| Parameter | Default | Controls |
|-----------|---------|----------|
| `needle_shrink_factor` | 0.85 | Per-demotion-retry multiplicative shrink on all ellipsoid radii. M ← M / f² tightens the exclusion zone. |
| `needle_stop_noise_multiplier` | 3.0 | Stop shrinking when max ellipsoid semi-axis < this × `input_noise_ilr`. Prevents shrinking below measurement resolution. |

### Repeated-Zoom Detection (Jaccard Sliding Window)

| Parameter | Default | Controls |
|-----------|---------|----------|
| `jaccard_window` | 3 | Number of recent bounding boxes to compare against when selecting new zoom bounds. `determine_new_bounds` slides over windows of top-M points to find a box with Jaccard ≤ `jaccard_threshold` vs all recent boxes in this window. |
| `jaccard_threshold` | 0.9 | Maximum Jaccard overlap allowed between a candidate bounding box and any box in the recent history window. A higher value accepts more overlap before trying the next window. |

The secondary MC Jaccard guard (`_bounds_jaccard_simplex`, threshold 0.75) is still used at zoom transitions as a force-declare trigger when the sliding-window bounds selection nonetheless produces a box that overlaps with history.

### Failure Retry

| Parameter | Default | Controls |
|-----------|---------|----------|
| `bounds_shrink_factor` | 0.8 | Radius shrink factor applied to all needle ellipsoids on a persistent-failure (no new data) retry. M ← M / factor². |
| `min_axis_noise_mult` | 2.0 | Stop condition for shrinking: if `max_needle_radius() < min_axis_noise_mult × input_noise_ilr`, all ellipsoids are at noise scale and the run terminates. |

## 5. File-by-File Reference

---

### 5.1 `utils/simplex.py`

**Role:** Low-level geometry on the probability simplex. All other modules import from here.

---

#### `Ellipsoid` (namedtuple)

Named triple `(c, M, B)`:
- `c`: (d,) centre on Δ  
- `M`: (d−1, d−1) SPD precision matrix — a point x is inside when u⊤M u ≤ 1, u = B⊤(x−c)  
- `B`: (d, d−1) orthonormal tangent-space columns (zero-sum hyperplane basis)

---

#### `get_tangent_basis(d, device, dtype)`

Returns a (d, d−1) QR basis for the centred hyperplane {v : Σvᵢ = 0}.

**Math:** Compute P = I − (1/d) 11⊤ (centering projector), then QR-decompose P and take the first d−1 columns.

Used to represent tangent directions on the simplex without leaving it.

---

#### `full_simplex_ellipsoid(d, device, dtype, ellipsoid_init_radius)`

Returns a spherical `Ellipsoid` centred at the barycentre (1/d, …, 1/d) with M = (1/r²)I.

Used as an initial trust region in some experimental code paths.

---

#### `composition_to_ilr(x)` ← **critical transform**

**Forward ILR (Helmert contrasts):**

For i = 0, …, d−2:

$$z_i = \sqrt{\frac{i+1}{i+2}} \left( \frac{1}{i+1}\sum_{j=0}^{i} \log x_j - \log x_{i+1} \right)$$

Maps compositions (d,) → ILR coordinates (d−1,). ILR is an isometry: Euclidean distances in ILR space equal Aitchison distances in simplex space.

**Why it matters:** The GP is trained in ILR space, where the simplex constraint is lifted to all of R^{d−1} and the geometry is Euclidean. Kernel lengthscales have consistent meaning across the simplex (unlike raw compositions, where corners are geometrically compressed).

**Benefits:** Statistically principled, isometric, no boundary singularities in ILR space.  
**Drawback:** Not invertible exactly when any xᵢ → 0 (hence the `+ 1e-10` epsilon guard). Compositions near a corner map to extreme ILR values, but the GP kernel dampens these naturally.

---

#### `ilr_to_composition(ilr, d)`

**Inverse ILR** via the pseudoinverse of the Helmert contrast matrix:

$$\log x_j \mathrel{+}= \frac{\text{coef}}{i+1} \cdot z_i \quad (j \leq i), \qquad \log x_{i+1} \mathrel{-}= \text{coef} \cdot z_i$$

then softmax to normalize. Used by `determine_penalty_ellipsoid` to compute the Hessian round-trip.

---

#### `random_simplex(num_samples, a, b, S, ...)` ← **CFS sampler**

Samples uniformly from the bounded probability simplex:

$$\{ x \in \mathbb{R}^d \mid \textstyle\sum x_i = S,\ a_i \leq x_i \leq b_i \}$$

**Algorithm (Conditional Frechet Sampling):**

For each dimension k in order:
1. The remaining sum after choosing x_0, …, x_{k−1} is s_rem.
2. The feasible range for x_k is [t_low, t_high] (clipped by both the box constraint and the need for the tail to sum correctly).
3. Compute the polytope volume V(s_rem − t) using **inclusion-exclusion**:
   $$V(s) = \frac{1}{(m-1)!} \sum_{S \subseteq [m]} (-1)^{|S|} \max(0, s - \sum_{j\in S} \text{cap}_j)^{m-1}$$
4. Invert the CDF numerically using **Newton's method** (bracketed for safety).

**Complexity:** O(2^d) per precomputation step. Maximum d = 20.  
**Benefits:** Exact uniform distribution, handles arbitrary bounds, GPU-batched.  
**Drawback:** Exponential in d — infeasible beyond d ≈ 20.

---

#### `subset_sums_and_signs(caps)` / `polytope_volume(S, ...)` / `newton_in_bracket(...)`

Internal helpers for CFS. `subset_sums_and_signs` precomputes all 2^m subset sums using bitmask DP. `polytope_volume` evaluates the inclusion-exclusion formula. `newton_in_bracket` inverts the CDF with Newton + bisection fallback.

---

#### `proj_simplex(X)`

Projects arbitrary R^d points onto the probability simplex.

**Algorithm (Duchi et al., 2008):** Sort descending, find the "rho" threshold ρ = max{j : u_j − (Σᵢ≤ⱼ uᵢ − 1)/j > 0}, compute θ = (Σᵢ≤ρ uᵢ − 1)/(ρ+1), return max(x − θ, 0).

Differentiable — safe to use inside autograd.

---

#### `sample_ellipsoid(n, ell, scale, jitter)`

Samples n points via Gaussian proposals in tangent space: u ~ N(0, scale²M⁻¹), then x = Proj_Δ(c + Bu). The projection keeps the result on the simplex.

Used by `random_chords_through_ellipsoid` in LineBO.

---

#### `clamp_ellipsoid_scale(ell, margin)`

Widens the trust ellipsoid by dividing eigenvalues of M by (margin² + ε). Used after ellipsoid optimisation steps (not currently active in main path).

---

#### `random_simplex_direction(n_directions, d, ...)` / alias `random_zero_sum_directions`

Samples unit-norm zero-sum directions: v ~ N(0,I), v ← v − mean(v), v ← v/‖v‖.

Zero-sum directions are tangent to the simplex — adding a scaled zero-sum vector to any simplex point keeps the sum at 1.

---

#### `is_on_simplex(x, tol)` / `simplex_distance(x, y, metric)`

Utility: membership test; pairwise Euclidean, Aitchison (CLR-based), or KL-divergence distances.

---

### 5.2 `utils/dataclasses.py`

**Role:** Configuration dataclasses. `ZoMBIHopConfig` is the single source of truth for all hyperparameters; `Checkpoint` is a legacy metadata struct for backward compatibility.

---

#### `ZoMBIHopConfig` (dataclass)

All hyperparameters with defaults (see §4 for descriptions). Key methods:
- `to_dict()` / `from_dict(data)` — JSON serialization, with forward-compatibility: unknown keys are silently dropped in `from_dict`.
- `get_torch_dtype()` — maps string dtype name to `torch.dtype`.
- `__post_init__()` — validates all fields on construction (raises `AssertionError` on bad values).

---

#### `Checkpoint` (dataclass)

Legacy metadata struct mirroring `ZoMBIHopConfig` fields plus `run_uuid`, `d`, `timestamp`, `version`, `metadata`, `status`, `error`, `traceback`. Kept for backward compatibility with old saved runs.

---

### 5.3 `utils/datahandler.py`

**Role:** The single owner of all mutable algorithm state. Every tensor, counter, and needle record lives here. `ZoMBIHop` and `GPSimplex` both hold a reference and call these methods.

---

#### `DataHandler.__init__(...)`

Stores all hyperparameters as plain attributes. Calls `_init_storage()`. Sets up run directory and UUID. If `directory` is None, saving is disabled (in-memory only).

---

#### `_init_storage()`

Zeros all in-memory tensors and state:
- `X_all_actual`, `X_all_expected`, `Y_all` — full observation history
- `X_init_actual`, `Y_init` — initial data (predates all needles, used as last-resort GP fallback)
- `X_pared`, `Y_pared` — noise-deduplicated GP training set; Y values are replaced with local medians after each activation
- `needles`, `needle_vals`, `needle_M_list`, `needle_B` — discovered optima + their ellipsoids
- `_penalty_mask` — boolean tensor, True = not inside any exclusion zone
- `bounds_history` — `List[Tensor]`, recent (2,d) AABB tensors for Jaccard sliding-window comparison; capped at `jaccard_window` entries
- Iteration counters: `current_activation`, `current_zoom`, `current_iteration`

---

#### `save_init(X_init_actual, X_init_expected, Y_init, bounds)`

Called once at the start of a new run. Populates all tensors from initial data, builds the initial pared set, writes `config.json`, and takes a permanent "init" snapshot.

---

#### `take_snapshot(label, permanent, activation, zoom, iteration)` ← **checkpointing**

Saves the complete algorithm state to disk in `run_dir/snapshots/{seq}_{label}/`:
- `tensors.pt` — all tensors via `torch.save`, including:
  - needle M stack (None entries encoded as zeros + `needle_has_M` boolean mask)
  - `bounds_history_stack`: `(n, 2, d)` tensor stacking all `bounds_history` entries (empty if none)
- `needles.json` — needle results as JSON (coordinates, peak value, **median_value**, discovery metadata; `nan` medians serialized as `null`)
- `summary.json` — human-readable: n_points, n_needles, best_y, timestamp

`_load_tensors()` restores `bounds_history` from `bounds_history_stack`; absent in old checkpoints defaults to `[]`.

If `max_snapshots` is set, `_cleanup_old_snapshots()` removes the oldest non-permanent entries.

---

#### `load_state()` → `(activation, zoom, iteration, no_improvements)`

Reads `latest.txt`, loads `tensors.pt` and `summary.json` from the referenced snapshot, reconstructs all tensors including the M list, and returns iteration state. Falls back to legacy `states/` format for old checkpoints.

---

#### `add_all_points(new_X_actual, new_X_expected, new_Y)` → `penalty_mask`

1. Compute `_compute_penalty_mask(new_X_actual)` for the new batch.
2. Concatenate to `X_all_actual`, `X_all_expected`, `Y_all`, `_penalty_mask`.
3. Call `_update_pared(new_X_actual, new_Y)` — adds unpenalized points to pared set.
4. Returns per-point boolean mask (True = not penalized).

---

#### `add_needle(needle, needle_value, needle_penalty_radius, activation, zoom, iteration, M, B, needle_median_value)`

Appends the needle to all needle tensors and `needles_results`. Stores both the peak `needle_value` and the noise-robust `needle_median_value` (median Y of all raw observations within `paring_spatial_halfnoise × input_noise_ilr` of the needle). Stores M (ellipsoid precision) and B (tangent basis, `None` for ILR-mode ellipsoids). Calls `_update_penalty_mask()` which also purges any now-penalized points from the pared set.

---

#### `_compute_penalty_mask(X)` → `bool tensor`

For each point x:
- **ILR ellipsoid needles** (when `needle_B is None` and M is not None):  
  δz = ILR(x) − ILR(needle); penalized if δz⊤ M δz ≤ 1
- **Legacy tangent-space ellipsoid** (when `needle_B is not None`):  
  u = B⊤(x−needle); penalized if u⊤ M u ≤ 1
- **Sphere fallback** (M is None):  
  penalized if ‖x − needle‖ ≤ r

Returns True = not penalized (safe to use in GP / zoom bounds).

---

#### `_update_penalty_mask()`

Recomputes `_penalty_mask` for all of `X_all_actual`. Then **purges the pared set**: any pared point now inside a penalty ellipsoid (e.g. because an ellipsoid was just refitted larger) is removed. This prevents retroactive contamination of the GP training data.

---

#### `_update_pared(X_new, Y_new)` ← **point paring**

Adds new UNPENALIZED points to the pared dataset with deduplication:

1. Filter: drop any penalized points.
2. For each new point (x_i, y_i):
   - Convert to ILR: z_i = ILR(x_i)
   - Compute distances to all pared points: dists = ‖z_i − Z_pared‖₂
   - Duplicate condition: `(dists < spatial_thresh) & (|y_i − Y_pared| < y_thresh)`
   - If duplicates exist AND pared set is large enough: **fair coin** — 50% keep old (discard new), 50% replace oldest duplicate with new.
   - If no duplicates: add unconditionally.

**Spatial threshold:** `paring_spatial_halfnoise × input_noise_ilr`  
**Y threshold:** `paring_y_noise_multiplier × max(GP_output_noise, 1e-6)`  
**Minimum pared size before dedup starts:** `max(2(d−1), 5)`

---

#### `get_gp_data()` → `(X, Y)`

Returns the global GP training set (all unpenalized data, not restricted to any bounds) in priority order:
1. Unpenalized pared points (preferred — clean, deduplicated, median-relabeled).
2. If pared set is empty/all penalized: unpenalized raw data.
3. Last resort: init data (predates all needles, guaranteed unpenalized).

Also applies `max_gp_points` truncation (keeps top-N by Y value to control GP cost). Used for needle declaration (global Hessian estimate) and failure-handling refits.

---

#### `get_zoom_gp_data(bounds)` → `(X, Y)` ← **local GP for zoom**

Local variant of `get_gp_data()` restricted to points within the axis-aligned `bounds`. Used during zoomed iterations so the GP posterior is tight within the active region and PI drops quickly on genuine convergence.

Budget (`max_gp_points`) filled in two passes:
1. **Pared points within bounds** (unpenalized, sorted by Y desc) — deduplicated, clean signal.
2. **Raw unpenalized points within bounds** (sorted by Y desc) — higher-density local observations that the paring step may have thinned.

Falls back to `get_gp_data()` when fewer than 2 local points exist (e.g. first zoom before any local data).

**Why this matters for PI:** `best_f` passed to the convergence check comes from `Y.max()` of this local dataset. If `get_gp_data()` (global) were used, a high-value point in a different region would keep `best_f` elevated and PI near zero everywhere in the current zoom — causing spurious or missed convergence. Local `best_f` means PI is measured against the best *local* observation.

---

#### `get_best_in_bounds(bounds)` → `(X_best, Y_best, global_idx)`

Returns the best unpenalized point within the axis-aligned `bounds`. Falls back to `get_best_unpenalized()` if no unpenalized points lie within bounds.

Used as the `prev_best` reference in the convergence check when zoomed, so the Y-improvement criterion measures progress within the active region, not against a global best from a different part of the simplex.

---

#### `_relabel_pared_with_medians()` ← **post-activation smoothing**

Called once after every activation (success or failure). For each entry in `X_pared`, replaces `Y_pared[i]` with the **median of all Y_all values whose corresponding X_all falls within `paring_spatial_halfnoise × input_noise_ilr`** in ILR space.

**Effect:** Smooths individual noise spikes that survived the paring deduplication step. Subsequent activations train their GPs on a smoother, more representative signal. Increments `_pared_version` if any Y value changed.

**Why median not mean:** The median is robust to outliers (a single high spike doesn't shift it). Mean would pull the label toward rare noise events, defeating the purpose of noise-robust paring.

---

#### `determine_new_bounds(add_to_history=True)` → `(2, d) tensor`  ← **Jaccard sliding window**

Slides a window of `top_m_points` over the Y-ranked unpenalized set to find an AABB that differs from recent history:

```
unpenalized points sorted by Y descending
for start = 0, 1, 2, ... (sliding):
    window = points[start : start + top_m_points]
    candidate_box = (min(window), max(window))
    max_jac = max(Jaccard(candidate_box, h) for h in bounds_history[-jaccard_window:])
    if max_jac <= jaccard_threshold:
        return candidate_box     ← first window that differs enough
return window with lowest max_jac   ← least-similar fallback
```

When `add_to_history=True` (default), the chosen box is appended to `self.bounds_history` (capped at `jaccard_window` entries). The **failure-retry path** calls with `add_to_history=False` so failure-retry bounds do not contaminate the history used by success-path zoom transitions.

`_jaccard_box(a, b)` is a static method computing volume-Jaccard of two (2,d) axis-aligned boxes. Dimensions where **both** boxes have zero width (< 1e-12) are projected out (contribute factor 1); this avoids division-by-zero for degenerate bounds.

**Used for:** zoom-in trust region after each successful zoom level; also called (with `add_to_history=False`) inside `_handle_failure_retry` Case 2.

---

#### `get_best_unpenalized()` → `(X_best, Y_best, global_idx)`

Returns the argmax of `Y_all[_penalty_mask]` — the globally best observation not in any exclusion zone. Used during global-bounds iterations and for needle declaration.

---

#### `update_all_needle_radii(new_M_list)`
Replaces each entry in `needle_M_list` with the corresponding M from `new_M_list` and calls `_update_penalty_mask()`. Called by `_handle_failure_retry` Case 1 after `recompute_all_ellipsoids` returns fresh M matrices.

---

#### `shrink_all_needle_radii(factor)`
Multiplies every needle's M by `1/factor²` (semi-axis_new = factor × semi-axis_old). Calls `_update_penalty_mask()`. Called by `_handle_failure_retry` Case 3 when no new data has been observed since the last failure.

---

#### `max_needle_radius()` → `float`
Returns the largest semi-axis across all active needle ellipsoids: `max(1/√(min_eigenvalue(M)))`. Used by `_handle_failure_retry` to decide whether all exclusion zones have shrunk to within measurement noise.

---

#### `get_input_noise()` / `get_normalized_input_noise()`

Computes median Euclidean distance between `X_all_expected` and `X_all_actual` (over all points). This is the empirical measure of physical robotic noise.

---

#### `update_gp_noise(sigma_y)` / `get_pared_hash()`

`update_gp_noise`: stores the GP-fitted output noise σ_Y after each fit. Used as Y-threshold in paring.  
`get_pared_hash`: returns `_pared_version`, an incrementing integer that changes whenever the pared set changes. Used by ZoMBIHop to detect whether new data arrived between demotion retries.

---

### 5.4 `utils/gp_simplex.py`

**Role:** Wraps a BoTorch `SingleTaskGP` with ILR normalization and provides acquisition creation, candidate generation, and Hessian ellipsoid computation.

---

#### `ILRWrappedAcquisition` (nn.Module)

```
__init__(base: nn.Module, ilr_std: Optional[Tensor])
forward(Xq: Tensor) -> Tensor
```

Takes simplex inputs (..., d), transforms to ILR (..., d−1), optionally divides by `ilr_std`, then calls the underlying BoTorch acquisition.

**Why needed:** BoTorch acquisitions expect unconstrained R^{d−1} inputs; this wrapper handles the composition→ILR→normalize chain so all outer code (RepulsiveAcquisition, nat-grad, seeding) works in simplex space.

---

#### `RepulsiveAcquisition` (nn.Module)

```
__init__(base, proj_fn, needles, penalty_radii, repulsion_lambda, needle_M_list, needle_B)
forward(Xq: Tensor) -> Tensor
```

Combines base acquisition with smooth needle repulsion:

$$\alpha_{\text{rep}}(x) = \alpha_{\text{base}}(x) - \lambda \sum_{i} \max\!\left(0,\; 1 - \delta z_i^\top M_i \delta z_i\right)^2$$

where δzᵢ = ILR(x) − ILR(needleᵢ) for ellipsoid needles, or `max(0, r − ‖x − needle‖)²` for sphere fallback.

**Penalty shape:** Zero outside the ellipsoid (when δz⊤Mδz ≥ 1), rising quadratically toward the centre. This is a **soft** boundary — the GP can still evaluate inside, but the acquisition is strongly penalized.

**λ auto-computation:** If `repulsion_lambda` is None, `_compute_repulsion_lambda` samples 100 random simplex points, computes `10 × median|α(x)|`, and uses `max(that, 100)`.

---

#### `GPSimplex.__init__(...)`

Stores references to DataHandler and all hyperparameters. Initializes `self.gp = None`, `self.ilr_std = None`.

---

#### `GPSimplex.fit(X, Y)` ← **GP training**

1. X → ILR coordinates: `X_ilr = composition_to_ilr(X)`  
2. Compute per-dimension std: `ilr_std = X_ilr.std(dim=0).clamp(min=1e-3)`, store as `self.ilr_std`  
3. Fit BoTorch GP: `SingleTaskGP(X_ilr / ilr_std, Y)` + `fit_gpytorch_mll`  
4. Update DataHandler's stored output noise.

**Why std-normalize ILR:** The ILR dimensions may have very different dynamic ranges (especially near simplex corners), causing the GP kernel's single lengthscale (or even ARD lengthscales) to be poorly calibrated. Normalizing by the observed std puts all ILR dimensions on equal footing. The `ilr_std` is always computed from whatever dataset was passed to `fit()` — so when using `get_zoom_gp_data`, it reflects the local ILR spread within the zoom bounds. The M matrices (penalty ellipsoids) intentionally stay in raw ILR space.

**Important:** `ilr_std` is always updated with the data the GP was actually trained on. When using local zoom data, `ilr_std` reflects the local ILR spread; when using global data, it reflects the global spread. The Hessian ellipsoid computation in `determine_penalty_ellipsoid` always calls `get_gp_data()` (global) to refit before computing M, ensuring the ellipsoid is sized in global ILR units.

---

#### `GPSimplex.predict(X)` → `(mean, variance)`

Queries the GP posterior at new compositions X. Applies ILR transform + ilr_std normalization before calling `gp.posterior()`.

---

#### `GPSimplex.probability_of_improvement(x, best_f)` → `float`

PI = Φ((μ(x) − f_best) / σ(x)), using the normalized ILR GP posterior. Returns probability ∈ [0, 1].

**Scope note:** PI is computed against whatever `best_f` is passed in. During zoomed iterations, ZoMBIHop passes `best_f_local = Y.max()` from `get_zoom_gp_data(bounds)` — so PI measures "probability of beating the best observation *within the current zoom bounds*", not the global best. This prevents spurious convergence when a high-value point exists elsewhere on the simplex.

---

#### `GPSimplex.compute_log_ei_at_point(x, best_f)` → `float`

For UCB mode: returns UCB value (used for logging, not convergence).  
For EI mode: returns log-EI via BoTorch `LogExpectedImprovement`. Both use the normalized ILR pipeline.

---

#### `GPSimplex.create_acquisition(best_f, penalty_value)` → `nn.Module`

1. Create base BoTorch acquisition (UCB or LogEI).  
2. Wrap in `ILRWrappedAcquisition(base_botorch, ilr_std=self.ilr_std)`.  
3. Auto-compute λ if needed.  
4. Fetch `needle_M_list`, `needle_B` from DataHandler.  
5. Return `RepulsiveAcquisition(base_acq, ...)`.

---

#### `GPSimplex._optimize_acquisition(acq, bounds, initial_conditions)` → `(candidates, values)`

**Natural-gradient ascent on the simplex:**

For each restart seed xᵣ:
1. Compute gradient: g = ∇ₓ α(x)
2. Compute Riemannian gradient: ḡ = Σᵢ xᵢ gᵢ (Fisher information inner product on the simplex)
3. Update: x ← normalize(x ⊙ exp(α(g − ḡ))) — this is the **exponential map** on the probability simplex
4. Clamp to bounds [lo, hi] and renormalize.

**Why natural gradient:** Standard gradient ascent on the simplex ignores the simplex's Riemannian metric (Fisher information). Natural gradient ascent respects the geometry, so steps are invariant to re-parameterization and avoid boundary-hugging artifacts.

**Bounds clamping:** After the exponential update, coordinates are hard-clamped to [lo, hi] and renormalized. This is a projected step, not a true Riemannian projection — it may slightly violate the sum-to-1 constraint at corners, which is why `proj_fn` is applied to the initial seed.

---

#### `GPSimplex.get_candidate(bounds, best_f, ...)` → `Tensor or None`

Full pipeline:
1. `create_acquisition(best_f)` → RepulsiveAcquisition
2. Sample `raw_samples` random points from bounds, score by acquisition
3. Filter by penalty mask (drop penalized seeds)
4. If < 10% of `n_restarts` are unpenalized after 5 extra sampling attempts → return None
5. Sort unpenalized seeds by acquisition, take top `n_restarts`
6. Run `_optimize_acquisition` on seeds
7. Sort results by final acquisition value
8. If `exclude_near` set: skip candidates within `exclude_near_tol`
9. Final penalty check on winner → return or None

---

#### `GPSimplex.fit_with_locality(needle, X_all, Y_all, max_radius, max_gp_points)`
Refits the GP using only data near the needle:
1. Compute ILR distances from all `X_all` to `needle`.
2. Select all points within `max_radius` (in ILR units) as the local set.
3. If local set has fewer than `top_m_points`, fall back to all unpenalized data.
4. Apply `max_gp_points` cap, call `self.fit(X_local, Y_local)`.

Used before computing the Hessian ellipsoid at needle declaration time. The locally-trained GP reflects true local curvature without being distorted by far-away observations.

---

#### `GPSimplex.create_clean_acquisition(best_f=None)` → `nn.Module`
Returns an `ILRWrappedAcquisition` (UCB or LogEI) **with no `RepulsiveAcquisition` wrapper** — no ellipsoidal needle penalties. This is the acquisition passed to `determine_penalty_ellipsoid` during needle declaration, so the Hessian reflects only the local objective landscape curvature, not proximity to other needles.

---

#### `GPSimplex.recompute_all_ellipsoids(needles, X_all, Y_all, max_radius, max_gp_points, drop_fraction, eigenvalue_floor)` → `List[Tensor]`
For each needle (loop, one at a time):
1. `fit_with_locality(needle, X_all, Y_all, ...)` — local GP around this needle
2. `create_clean_acquisition()` — penalty-free acquisition
3. `determine_penalty_ellipsoid(needle, acq_fn=clean_acq)` → new M

Returns a list of (d-1, d-1) M tensors, one per needle. GPU memory is bounded by `max_gp_points` at a time (sequential, not batched). Falls back to `_refit_all_needle_ellipsoids` (global pared GP) if an exception occurs. Called by `_handle_failure_retry` Case 1 via `dh.update_all_needle_radii(new_M_list)`.

---

#### `GPSimplex.determine_penalty_ellipsoid(needle, drop_fraction, eigenvalue_floor, max_radius, acq_fn=None)` → `(M, None)`

**Core Hessian ellipsoid computation:**

1. If `acq_fn` is provided, uses it directly. Otherwise calls `create_acquisition()` (with repulsion). **In the needle-declaration path, `create_clean_acquisition()` is always passed here** so the Hessian reflects only local objective curvature.
2. Define `tilde_alpha_ilr(z)` = acq_fn(ILR⁻¹(z)) — acquisition as function of raw ILR coordinates z.
3. Compute H = ∇²z `tilde_alpha_ilr`(z_needle) via `torch.autograd.functional.hessian`.
4. Symmetrize: neg_H = −(H + H⊤)/2.
5. Eigendecompose: neg_H = VΛV⊤; clamp Λ ≥ `eigenvalue_floor`.
6. Compute scale Δ:
   - Δ_acq = `drop_fraction` × |α(needle)| (acquisition drop criterion)
   - Δ_noise = 0.5 × λ_max × (3σ_in)² (noise-based criterion)
   - Δ = max(Δ_acq, Δ_noise, 1e-12)
7. M = VΛV⊤ / (2Δ). Boundary is where acquisition drops by Δ from the peak.
8. Clamp minimum eigenvalues of M to 1/max_radius² (caps maximum semi-axis).
9. Returns (M, None) — B is not needed since M lives in ILR space.

**Geometry:** A point x lies inside the ellipsoid iff (ILR(x) − ILR(needle))⊤ M (ILR(x) − ILR(needle)) ≤ 1. Semi-axes are 1/√(eigenvalues of M), in ILR units.

---

#### `GPSimplex._get_tangent_basis(d)` / `GPSimplex._compute_repulsion_lambda(...)` / `GPSimplex.get_last_computed_lambda()` / `GPSimplex.get_output_noise()`

Support utilities: cached QR basis; λ auto-computation via median acquisition; accessors.

---

### 5.5 `core/linebo.py`

**Role:** Physical experiment interface. Given a candidate point, proposes a line to evaluate, calls the user-supplied `objective_function`, and returns observations.

---

#### Module-level constants

```python
SAMPLER_MAX_EXTRA_ATTEMPTS = 50   # extra direction sampling attempts if few valid lines
MIN_DIRECTION_NORM = 1e-10        # minimum direction magnitude to keep
```

---

#### `directions_through_simplex_points(x0, k, device, dtype)`

Generates k directions from x0 toward random simplex points: d_i = normalize(P_i − x0). Drops directions where ‖P_i − x0‖ < MIN_DIRECTION_NORM (P ≈ x0, would give degenerate line). Resamples once if fewer than k valid directions found.

---

#### `line_simplex_segment(x0, d)` → `(t_min, t_max, x_left, x_right) or None`

Finds the chord of line x0 + t·d that lies within the probability simplex:
- For positive components of d: t ≥ −x0ⱼ/dⱼ (keeps xⱼ ≥ 0 on the positive side)
- For negative components of d: t ≤ −x0ⱼ/dⱼ (keeps xⱼ ≥ 0)
- t_min = max of all lower bounds, t_max = min of all upper bounds

If t_min ≥ t_max, the direction doesn't pierce the simplex (returns None).

---

#### `batch_line_simplex_segments(x0, D)` → `(x_left, x_right, t_min, t_max, mask)`

Batched version of `line_simplex_segment` for k directions simultaneously. Returns a boolean `mask` indicating valid (non-degenerate) lines.

---

#### `batch_line_bounds_segments(x0, D, bounds)` → `(x_left, x_right, t_min, t_max, mask)`

Clips lines to an axis-aligned box [lo, hi] instead of the full simplex. Used during zoomed search: the trust region is a box inside the simplex, and zero-sum directions preserve the sum=1 constraint, so all clipped points remain on the simplex.

**Math:** For each dimension and sign of d[i]:
- d[i] > 0: t ≤ (hi[i]−x0[i])/d[i] and t ≥ (lo[i]−x0[i])/d[i]
- d[i] < 0: t ≤ (lo[i]−x0[i])/d[i] and t ≥ (hi[i]−x0[i])/d[i]

Take intersection of all per-dimension intervals.

---

#### `batch_line_ellipsoid_segments(x0, D, ell)` → `(x_left, x_right, t_min, t_max, mask)`

Clips lines to the intersection of the simplex AND an ellipsoid {x : (B⊤(x−c))⊤M(B⊤(x−c)) ≤ 1}.

**Math:** Substituting x = x0 + t·d:
- u(t) = B⊤(x0 + td − c) = u0 + t·v, where v = B⊤d
- Ellipsoid constraint: (u0+tv)⊤M(u0+tv) ≤ 1
- Expand: a·t² + b·t + c_shift ≤ 0, where a = v⊤Mv, b = 2v⊤Mu0, c_shift = u0⊤Mu0 − 1
- Solve quadratic for t ∈ [t_lo_e, t_hi_e]; intersect with simplex t-interval.

Handles degenerate cases (a≈0, discriminant<0, line inside/outside ellipsoid).

---

#### `_clip_x_to_ellipsoid(x, ell)`

If x is outside the ellipsoid, radially projects it to the boundary: u ← u/√(u⊤Mu), x ← proj_Δ(c + Bu_clipped).

---

#### `random_chords_through_ellipsoid(k, ell, device, dtype)` → `(lefts, rights)`

Generates up to k random chord endpoints inside ellipsoid ∩ simplex. Each chord passes through a random interior point (from `sample_ellipsoid`) in a random zero-sum direction. Using random interior anchors gives uniform chord coverage — all chords from a fixed anchor would strongly cluster.

---

#### `random_chords_through_simplex(k, bounds, device, dtype)` → `(lefts, rights)`

Same idea but constrained to the bounds box. Interior anchor points are sampled from within bounds using CFS.

---

#### `LineBO.__init__(objective_function, dimensions, num_points_per_line, num_lines, device)`

Stores objective and parameters. `num_lines` defaults to 10×d.

---

#### `LineBO._integrate_acquisition_along_lines(x_left, x_right, acquisition_function)` → `(k,) tensor`

1. Parameterize each line: points[i, j] = (1−t_j) × x_left[i] + t_j × x_right[i] for t_j ∈ {0, 1/(N−1), …, 1}. Shape: (k, N, d).
2. Batch-evaluate acquisition in blocks of 500 (OOM-safe).
3. **Score by mean** over the N points (not max).

**Why mean not max:** With max, any line whose best point grazes the unpenalized fringe of a needle ellipsoid scores as well as a fully unexplored line — the penalties on the rest of the line are invisible to argmax. Mean integrates penalties along the full chord, so a line that spends most of its length inside a penalized region scores very negatively.

---

#### `LineBO.ranked_line_endpoints(x_tell, bounds, acquisition_function)` → `(x_left, x_right)`

1. Generate `num_lines` random chords through simplex (within bounds).
2. Score by `_integrate_acquisition_along_lines`.
3. Return endpoints sorted best-first (or random if no acquisition provided).

---

#### `LineBO.sampler(x_tell, bounds, acquisition_function)` → `(x_requested, x_actual, y)`

1. `ranked_line_endpoints` → (x_left_ranked, x_right_ranked)
2. Stack as endpoints tensor (num_lines, 2, d) and call `objective_function(endpoints_ranked)`.
3. Build `x_requested` from the first principal direction of `x_actual` (via SVD) — a smooth parameterization of the measurement locations along the best line.

Returns (x_requested, x_actual, y). The `ZoMBIHop._objective_wrapper` uses x_requested as `X_expected` and x_actual as the ground-truth measured composition.

---

### 5.6 `core/zombihop.py`

**Role:** Top-level orchestrator. Owns the activation/zoom/iteration loops, needle declaration, failure handling, convergence checking, and Jaccard-based repeated-zoom detection. Delegates storage to DataHandler and GP operations to GPSimplex.

---

#### Module-level helpers

##### `_is_global_bounds(bounds, eps=0.01)` → `bool`

True when bounds cover essentially the entire simplex: `bounds[0].max() < eps` AND `bounds[1].min() > 1 − eps`. Global bounds trivially overlap everything, so they are excluded from Jaccard checks. Also used to decide between global vs local GP data selection throughout `run()`.

##### `_bounds_jaccard_simplex(bounds_a, bounds_b, n_samples, device, dtype)` → `float`

Monte Carlo Jaccard overlap restricted to the simplex:
1. Sample n=500 Dirichlet-uniform points on the simplex.
2. Count points in each box: n_A, n_B, n_AB.
3. Return n_AB / (n_A + n_B − n_AB).

**Why Monte Carlo on simplex (not full hypercube):** The simplex is a proper subset of [0,1]^d. Two boxes that look very similar in ambient space might overlap very little within the simplex (e.g. a box near a corner). Restricting samples to the simplex gives the physically correct overlap fraction.

---

#### `ZoMBIHop.__init__(...)`

Validates dimensions, sets defaults (e.g. `top_m_points = max(d+1, 4)` if not specified), constructs DataHandler and GPSimplex, saves init data and initial full-simplex bounds.

---

#### `ZoMBIHop._all_needle_axes_below_min(dh)` → `bool`

Returns True when `dh.max_needle_radius() < min_axis_noise_mult × dh.input_noise_ilr`. Used inside `_handle_failure_retry` Case 3 to decide whether to stop after shrinking.

---

#### `ZoMBIHop._handle_failure_retry(dh, first_failure_handled, data_added_since_last_failure)` → `(should_stop, first_failure_handled)`
Three-way dispatch called from the while-zoom loop whenever `activation_failed` is set:

- **Case 1** (`not first_failure_handled`): calls `gp_handler.recompute_all_ellipsoids(...)` then `dh.update_all_needle_radii(new_M_list)`. On exception, falls back to an inline pared-GP refit (fit global GP, create repulsive acquisition, recompute each needle's M). Returns `(False, True)`.
- **Case 2** (`first_failure_handled AND data_added_since_last_failure`): calls `dh.determine_new_bounds(add_to_history=False)` and updates `dh.bounds`/`self.bounds`. Returns `(False, True)`.
- **Case 3** (`first_failure_handled AND NOT data_added_since_last_failure`): calls `dh.shrink_all_needle_radii(bounds_shrink_factor)`. If `_all_needle_axes_below_min(dh)`, returns `(True, True)` (stop). Otherwise returns `(False, True)`.

After the call, the caller resets `data_added_since_last_failure = False`, `activation_failed = False`, `consecutive_converged = 0`, reloads GP data for the (possibly updated) bounds, and issues `continue` to retry the same `current_zoom` level.

---

#### `ZoMBIHop._declare_needle_at_best(dh, activation, zoom, iteration, reason)` → `Optional[Tensor]`

Shared needle declaration logic called by both the PI convergence path and the Jaccard force-declare path.

1. Get current best unpenalized point (globally via `get_best_unpenalized()`).
2. Compute **needle median value**: median Y of all `X_all_actual` points within `paring_spatial_halfnoise × input_noise_ilr` in ILR space of the needle.
3. Log: needle location, peak value, and local median.
4. **Fit a clean local GP** via `gp_handler.fit_with_locality(needle, dh.X_all_actual, dh.Y_all, ...)` — uses only data near the needle for a locally-accurate Hessian, without contamination from far-away observations.
5. **Create penalty-free acquisition** via `gp_handler.create_clean_acquisition()` — excludes RepulsiveAcquisition so the Hessian reflects only the local objective landscape, not proximity to other needles.
6. Compute Hessian ellipsoid M via `determine_penalty_ellipsoid(needle, acq_fn=clean_acq)`.
7. Call `dh.add_needle(..., needle_median_value=median)`.
8. Reset bounds to full simplex for the next activation.
9. Returns the needle tensor, or `None` if no unpenalized points exist.

The `reason` string (`"PI convergence"` or `"Jaccard convergence"`) appears in log output so the trigger path is visible.

---

#### `ZoMBIHop._check_convergence_to_needle(candidate, unpenalized_X, unpenalized_Y, prev_best_X, prev_best_Y, best_f_ref)` → `(converged, pi, log_ei)`

Convergence requires **both** conditions:
1. PI(candidate) = Φ((μ(candidate) − f_best) / σ(candidate)) < `convergence_pi_threshold`
2. (current batch best Y) − (prev_best_Y) < GP_output_noise × `output_noise_threshold_mult`

`best_f_ref` (optional): if provided, used as `f_best` instead of the global max from `get_gp_data()`. During zoomed iterations, this is `Y.max()` from `get_zoom_gp_data(bounds)`, so PI is measured against the best observation *within the current bounds* rather than the global best. This prevents spurious early convergence when an unrelated high-value point exists elsewhere on the simplex.

`prev_best_Y` is the best within bounds (from `get_best_in_bounds(bounds)`) so the Y-improvement criterion also measures local progress.

Also logs `log_ei` to `dh.log_ei_history` for diagnostics.

---

#### `ZoMBIHop._objective_wrapper(X, bounds, acquisition_function)` → `(unpenalized_X, unpenalized_Y)`

1. Calls `self.objective(X, bounds, acquisition_function)` → (X_expected, X_actual, Y).
2. Projects X_actual onto the simplex (handles robotic noise that overshoots the simplex).
3. Calls `dh.add_all_points(X_actual, X_expected, Y)` → updates all storage.
4. Returns only the unpenalized subset.

---

#### `ZoMBIHop.run(max_activations, time_limit_hours)` ← **main loop**

The nested loop structure (Edit 2B — no outer demotion-retry loop):

```python
while activation < max_activations:
    first_failure_handled = False
    data_added_since_last_failure = False
    zoom_bounds_history = []      # MC Jaccard guard history (per activation)
    current_zoom = start_zoom

    while current_zoom < max_zooms and not finished:
        zoom = current_zoom
        # Local GP when zoomed in
        if global bounds: X,Y = get_gp_data()
        else:             X,Y = get_zoom_gp_data(bounds)
        best_f_local = Y.max()
        fit GP on X,Y

        for iteration in 0..max_iterations:
            prev_best ← get_best_in_bounds(bounds)         # local ref
            candidate = get_candidate(best_f=best_f_local) → LineBO.sampler
            data_added_since_last_failure = True            # always set after objective call
            X,Y = get_zoom_gp_data(bounds)
            best_f_local = Y.max()
            refit GP on X,Y
            check_convergence(best_f_ref=best_f_local, prev_best=local)
            if converged × n_consecutive:
                _declare_needle_at_best(reason="PI convergence")
                  # uses fit_with_locality + create_clean_acquisition
                break

        if activation_failed:
            if no needles: finished = True; break
            should_stop, first_failure_handled = _handle_failure_retry(
                dh, first_failure_handled, data_added_since_last_failure)
            data_added_since_last_failure = False
            activation_failed = False
            if should_stop: finished = True; break
            reload GP; continue  # retry SAME current_zoom — do NOT increment

        if needle: break zoom loop (advance activation)

        if zoom < max_zooms-1:  # advance to next zoom level
            new_bounds = determine_new_bounds()   # Jaccard sliding window (Edit 2A)
            # Secondary MC Jaccard guard
            if not global and MC_Jaccard(new_bounds, zoom_bounds_history) > threshold:
                _declare_needle_at_best(reason="Jaccard convergence")
            else:
                append new_bounds to zoom_bounds_history
                current_zoom += 1          # advance only on clean success

    _relabel_pared_with_medians()   # smooth pared Y after activation
    activation++
```

**Key structural changes vs the old code:**
- The outer `while True` demotion-retry loop is **gone**. Failure is handled in-place at the current zoom level via `_handle_failure_retry` + `continue`.
- `current_zoom` is **only incremented on clean success** (no failure, no force-declare). Retries repeat the exact same zoom body with updated bounds/ellipsoids.
- `zoom_bounds_history` is initialized **per activation** (not per demotion retry), since there is no longer an outer retry loop to reset it.
- `determine_new_bounds()` uses the **Jaccard sliding window** (Edit 2A) to choose a dissimilar AABB before the MC guard runs, so the MC guard only fires when the sliding window has exhausted all windows.

**Over-penalization guard (after needle found, before advancing activation):**  
If > 90% of the simplex is penalized: either reset to full simplex (infinite-activation mode) or stop.

---

### 5.7 `utils/visualization.py`

**Role:** Plotting utilities for 3-component (ternary) and 4-component (tetrahedral) problems, plus progress tracking.

---

#### `plot_optimization_progress(history, figsize, save_path)`

Three-panel figure: best Y over iterations, needle count over iterations, total points over iterations. Takes a list of snapshot summary dicts.

---

#### `plot_simplex_2d(X, Y, needles, ...)`

Ternary diagram for d=3 problems. Converts simplex coordinates to 2D via:
- x_coord = (2b + c) / 2  
- y_coord = (√3/2) × c  

Overlays colormap scatter of all observations plus red stars at needle locations.

---

#### `plot_simplex_3d(X, Y, needles, ...)`

Tetrahedral 3D plot for d=4 problems. Projects 4-simplex coordinates to 3D via `x_3d = X @ vertices` where `vertices` is a regular tetrahedron.

---

#### `plot_needles_summary(needles_results, ...)`

Bar chart of needle values + scatter timeline colored by activation. Also prints a formatted summary table with needle index, value, median_value (if available), activation, zoom, and iteration of discovery.

---

## 6. Key Mathematical Derivations

### 6.1 ILR as an Isometry

The ILR transform is an isometry from (Δ^d, d_A) to (R^{d−1}, ‖·‖₂) where d_A is the Aitchison distance. This means:

$$d_A(x, y) = \|ILR(x) - ILR(y)\|_2$$

The SE kernel in ILR space is therefore an Aitchison-stationary kernel on the simplex — exactly the right prior for composition-valued data.

### 6.2 Natural Gradient on the Simplex

The Fisher information metric on the probability simplex at point x is:

$$g_{ij}(x) = \frac{\delta_{ij}}{x_i}$$

The natural gradient ∇̃f = G⁻¹∇f, and the exponential map gives the multiplicative update:

$$x \leftarrow \frac{x \odot \exp(\alpha(g - \bar{g}))}{\sum_i x_i \exp(\alpha(g_i - \bar{g}))}$$

where $\bar{g} = \sum_i x_i g_i$ is the weighted mean gradient. This is exactly the `_optimize_acquisition` update rule.

### 6.3 Hessian Ellipsoid Geometry

At the needle z* = ILR(x*), the acquisition has a local maximum. Taylor expansion:

$$\alpha(z) \approx \alpha(z^*) + \frac{1}{2}(z-z^*)^\top H (z-z^*), \quad H = \nabla^2 \alpha(z^*)$$

Since z* is a maximum, H is negative semi-definite. Define neg_H = −H (positive semi-definite). The iso-acquisition surface where α drops by Δ from the peak is:

$$\frac{1}{2}(z-z^*)^\top \text{neg\_H} (z-z^*) = \Delta \iff \delta z^\top M \delta z = 1$$

with M = neg_H / (2Δ). The semi-axes of this ellipsoid are $r_i = 1/\sqrt{\lambda_i(M)} = \sqrt{2\Delta / \lambda_i(\text{neg\_H})}$.

### 6.4 CFS Polytope Volume

The volume of the polytope {y ∈ R^m | y_i ≥ 0, Σy_i ≤ s, y_i ≤ cap_i} is computed by inclusion-exclusion:

$$V(s) = \frac{1}{m!} \sum_{S \subseteq [m]} (-1)^{|S|} \max\!\left(0, s - \sum_{j \in S} \text{cap}_j\right)^m$$

The CDF inversion uses Newton's method on V(s_rem − t), which is the probability of the remaining tail summing to at most s_rem − t.

### 6.5 Local vs Global GP and PI Scope

When zoomed into bounds B ⊂ Δ, the GP is trained on `get_zoom_gp_data(B)` — only unpenalized data within B. The posterior mean and variance are therefore locally conditioned on the observations within B, and posterior variance within B is much lower than a globally-trained GP (which must explain variation everywhere).

PI is defined as:

$$\text{PI}(x) = \Phi\!\left(\frac{\mu(x) - f^*}{\sigma(x)}\right)$$

where f* = `best_f_local` = max Y from the local GP training set. This has two locality effects:
1. σ(x) is smaller within B (more local data) → PI drops sooner as we near the optimum.
2. f* is the local best within B, not the global best elsewhere → PI is not trivially near-zero when another activation found a higher value in a different region.

Without locality (global GP, global f*), PI can stay elevated indefinitely (global uncertainty) or drop spuriously (global f* >> local values).

---

## 7. Benefits, Drawbacks, and Known Failure Modes

### Benefits

| Aspect | Benefit |
|--------|---------|
| **Multi-modal discovery** | Guaranteed-different optima via hard ellipsoidal exclusion zones — each activation finds a new basin. |
| **Simplex fidelity** | ILR + CFS + natural gradient ascent all respect the simplex geometry exactly. |
| **Noise-robustness** | Point paring prevents noise spikes from poisoning the GP. Pared Y values are relabeled with local medians after each activation for additional smoothing. Needle records store both peak and median values. |
| **Adaptive exclusion zones** | Hessian ellipsoids adapt to the local curvature — narrow along flat directions, tight along steep ones. |
| **Failure recovery** | Three-way in-place failure dispatch (clean ellipsoid refit → Jaccard-aware bounds update → radius shrink) handles both transient and persistent failures without restarting the zoom sequence. |
| **Reproducibility** | Full state snapshots at every iteration enable exact restart. |
| **ILR std normalization** | Prevents poorly-scaled ILR dimensions from dominating GP kernel fitting. std computed from actual training data, reflecting the local ILR spread when using zoom-local data. |
| **Local GP for zoom** | Fitting only within-bounds data makes the posterior tight in the active region, causing PI to drop quickly on genuine convergence rather than remaining elevated due to global uncertainty. |
| **Local PI reference** | `best_f_ref` and `prev_best` are computed within zoom bounds, preventing spurious convergence when a high-value point exists in a different region. |
| **Jaccard sliding window** | `determine_new_bounds` slides over Y-ranked windows to find a genuinely different AABB before the MC Jaccard guard runs. Repeated-zoom force-declare is now a fallback, not the first resort. |
| **Clean local GP for ellipsoids** | Needle ellipsoids are computed from a local, penalty-free GP fit (`fit_with_locality` + `create_clean_acquisition`), giving curvature estimates that reflect only the local landscape without distortion from other needles or distant data. |

### Drawbacks

| Aspect | Drawback |
|--------|---------|
| **GP cost** | O(n³) GP fitting. Mitigated by `max_gp_points` truncation and local data restriction, but accuracy degrades when important points are excluded. |
| **CFS dimension limit** | `random_simplex` requires d ≤ 20 (2^d inclusion-exclusion). |
| **Hessian accuracy** | Hessian computed at one point; if the acquisition has a flat/curved top, the ellipsoid may be poorly sized. |
| **AABB zoom bounds** | Axis-aligned bounding box ignores simplex geometry — can straddle inactive corners or enclose empty space. |
| **Natural gradient pathology** | The multiplicative update can get stuck at simplex vertices (x_i = 0 makes coordinate frozen). |
| **Single-point needle** | Each activation converges to one needle; if the landscape has two very close optima, one may be missed. |
| **Jaccard is approximate** | Monte Carlo Jaccard (n=500) has sampling noise; the 0.75 threshold is a heuristic. A very tight zoom may legitimately have high Jaccard with its parent zoom — force-declare fires somewhat conservatively in this case. |
| **Pared relabeling cost** | O(n_pared × n_all) ILR distance computation per activation. Scales quadratically with dataset size. |

### Known Failure Modes

**Over-clustering at one corner:**  
*Cause:* `max` line scoring (old behavior) let lines grazing the unpenalized fringe of a needle score at peak UCB.  
*Fix:* `mean` line scoring integrates penalties along the full chord.

**Pared set contamination:**  
*Cause:* Ellipsoid refit after more data could cover previously-unpenalized pared points.  
*Fix:* `_update_penalty_mask()` now purges the pared set after every needle addition/refit.

**Zoom loop stagnation (repeated AABB):**  
*Cause:* Top-M points converge to the same region across successive zooms, so naively selected AABB overlaps prior bounds heavily.  
*Fix (layered):*  
  1. `determine_new_bounds` (Edit 2A) slides over Y-ranked windows to find a dissimilar AABB before the MC guard runs. Only if all windows exceed the threshold does the guard see an overlapping box.  
  2. MC Jaccard guard force-declares the needle rather than triggering `activation_failed`. Repeated zooms to the same region are taken as convergence evidence.  
  3. Edit 2B failure retry handles the residual case where even force-declare fails (no unpenalized points): clean ellipsoid refit → bounds update → shrink, each retrying at the same zoom level.

**Spurious PI convergence (global GP, global f*):**  
*Cause:* With global GP training data, a high-value observation from a previous activation (in a different region) keeps `f*` high and makes PI near zero everywhere in the current zoom, triggering premature needle declaration or preventing convergence detection at the right point.  
*Fix:* Local GP (`get_zoom_gp_data`) + local `best_f_ref` + local `prev_best` (from `get_best_in_bounds`). PI is now computed against the best observation *within the current zoom bounds*.

**Simplex edge ILR instability:**  
*Cause:* Compositions near x_i ≈ 0 give large ILR coordinates; the `+ 1e-10` epsilon shifts them but doesn't eliminate the issue.  
*Mitigation:* ILR std normalization dampens extreme values. The GP's Matérn/SE kernel provides implicit regularization. Explicit corner exclusion is not implemented.

**GP with too few unpenalized points:**  
*Cause:* Aggressive ellipsoids cover most of the data, leaving only init points.  
*Fix:* `get_gp_data()` fallback chain: pared → raw unpenalized → init data. `get_zoom_gp_data()` falls back to `get_gp_data()` when fewer than 2 local points exist.

---

## 8. Recent Algorithm Edits (May 2026)

This section documents three algorithm improvements applied in May 2026.

---

### Edit 1 — Clean Local GP for Needle Ellipsoid Fitting (`gp_simplex.py`)

**Problem:** When a needle is declared, the penalty ellipsoid M is derived from the Hessian of the acquisition function at the needle location. If the GP used for this Hessian computation was trained on local (zoom-restricted) data, the Hessian reflects the *local* landscape curvature accurately. But if the acquisition was built from the standard RepulsiveAcquisition (which includes existing needle penalties), the Hessian has contributions from those penalties that distort the curvature estimate.

**Fix:** `gp_simplex.py` now provides two additional methods used during needle declaration:

- **`fit_with_locality(X, Y, bounds=None, radius_mult=3.0)`** — refits the GP using only points whose ILR coordinates fall within `radius_mult × input_noise_ilr` of the best point. Falls back to all unpenalized data when fewer than `top_m_points` local points exist. This produces a tight, local-only GP whose Hessian reflects only the local landscape.

- **`create_clean_acquisition()`** — builds a base acquisition (UCB or EI) **without** ellipsoidal repulsion penalties. The Hessian at the needle from this penalty-free acquisition gives the true local curvature.

- **`recompute_all_ellipsoids(X_pared, Y_pared, needles, top_m)`** — for every needle in the list, calls `fit_with_locality` centered on that needle, then `create_clean_acquisition`, then computes `determine_penalty_ellipsoid` to refit M. Used by the failure-retry handler (Edit 2B) to recompute all ellipsoids when new data arrives.

These methods are called in `_declare_needle_at_best()` instead of the standard `fit` + `create_acquisition` path, ensuring the needle ellipsoid size reflects only local curvature, not proximity to other needles.

---

### Edit 2A — Jaccard Sliding-Window Bounds Selection (`datahandler.py`)

**Problem:** The old `determine_new_bounds()` always returned the AABB of the top-M unpenalized points by Y value. Over successive zooms in the same activation, this repeatedly produced nearly-identical bounding boxes, triggering the Jaccard force-declare immediately and causing premature needle declarations in some landscapes.

**Fix:** `determine_new_bounds` now uses a **sliding-window search** over the ranked unpenalized points to find bounds that are genuinely different from recent history.

#### New Parameters

| Parameter | Default | Controls |
|-----------|---------|----------|
| `jaccard_window` | 3 | How many recent bounding boxes to compare against. |
| `jaccard_threshold` | 0.9 | Maximum allowed Jaccard overlap with *all* boxes in the window before trying the next window. |

#### Algorithm

```
unpenalized points sorted by Y descending
for start = 0, 1, 2, ... (sliding over ranked points):
    window_pts = points[start : start + top_m_points]
    candidate_box = AABB(window_pts)
    max_jac = max(Jaccard(candidate_box, h) for h in bounds_history[-jaccard_window:])
    if max_jac <= jaccard_threshold:
        pick this window → return candidate_box
if no window qualifies:
    return the window with lowest max_jac (least similar to history)
```

When `add_to_history=True` (default), the chosen box is appended to `self.bounds_history` so future calls compare against it.

#### `_jaccard_box(a, b)` (static method)

Computes the **volume-Jaccard** of two axis-aligned bounding boxes in ILR space:

```
J = Vol(intersection) / Vol(union)
```

Dimensions where **both** boxes have zero width (width < 1e-12) are projected out (they contribute a factor of 1 to volume). This prevents division by zero in degenerate cases (e.g. all top-M points share a component value).

#### Serialization

`bounds_history` is serialized in `take_snapshot` as a stacked `(n, 2, d)` tensor `bounds_history_stack` and restored in `_load_tensors`.

#### New DataHandler Helpers

| Method | Description |
|--------|-------------|
| `update_all_needle_radii(new_M_list)` | Replaces each needle's M matrix with the corresponding entry from `new_M_list`. Called after `recompute_all_ellipsoids`. |
| `shrink_all_needle_radii(factor)` | Multiplies each needle M by `1/factor²` (equivalent to scaling each semi-axis by `factor`). Used when failure persists with no new data. |
| `max_needle_radius()` | Returns the maximum semi-axis length across all needle ellipsoids (= 1/√(min eigenvalue of M)). Used to check the stop condition. |

---

### Edit 2B — Failure Retry Within Activation (`zombihop.py`)

**Problem:** The old failure handling used an outer `while True` demotion-retry loop that wrapped the entire zoom loop. On failure, it rewound to zoom 0 with the same or shrunken ellipsoids, restarting the full zoom sequence. This was expensive and could loop indefinitely when the failure was transient.

**Fix:** The zoom loop is restructured to retry at the **same zoom level** on failure, with a three-way dispatch based on failure history.

#### New Parameters

| Parameter | Default | Controls |
|-----------|---------|----------|
| `bounds_shrink_factor` | 0.8 | Radius shrink factor applied to all needle ellipsoids on a no-new-data failure. |
| `min_axis_noise_mult` | 2.0 | Stop condition: if `max_needle_radius() < min_axis_noise_mult × input_noise_ilr`, all ellipsoids are at noise scale and shrinking further is pointless. |

#### Loop Restructure

```python
# OLD: for zoom in range(start_zoom, dh.max_zooms):
# NEW:
current_zoom = start_zoom
while current_zoom < dh.max_zooms and not finished:
    zoom = current_zoom
    ...
    if activation_failed:
        should_stop, first_failure_handled = _handle_failure_retry(...)
        data_added_since_last_failure = False
        activation_failed = False
        consecutive_converged = 0
        if should_stop:
            finished = True
            break
        continue           # retry SAME zoom level — don't advance current_zoom
    ...
    current_zoom += 1      # advance only on success
```

`data_added_since_last_failure` is set to `True` immediately after the objective function returns a new observation. It resets to `False` after each failure dispatch.

#### `_handle_failure_retry` Dispatch

```
Case 1 (first failure):
    not first_failure_handled
    → gp_handler.recompute_all_ellipsoids(...)
    → dh.update_all_needle_radii(new_M_list)
    → first_failure_handled = True
    → return (should_stop=False, first_failure_handled=True)

Case 2 (repeated failure, new data since last failure):
    first_failure_handled AND data_added_since_last_failure
    → dh.determine_new_bounds(add_to_history=False)
                                         ↑ don't contaminate history
    → update dh.bounds with new AABB
    → return (should_stop=False, first_failure_handled=True)

Case 3 (repeated failure, no new data):
    first_failure_handled AND NOT data_added_since_last_failure
    → dh.shrink_all_needle_radii(bounds_shrink_factor)
    → if dh.max_needle_radius() < min_axis_noise_mult × input_noise_ilr:
          return (should_stop=True, ...)   ← all ellipsoids at noise scale
    → return (should_stop=False, first_failure_handled=True)
```

#### `_all_needle_axes_below_min(dh)`

Helper that returns `True` when `dh.max_needle_radius() < self.min_axis_noise_mult × self.config.input_noise_ilr`. Called inside `_handle_failure_retry` to decide whether to stop.

#### Secondary Jaccard Guard

The MC Jaccard check (`_bounds_jaccard_simplex`) is retained at zoom transition (before advancing `current_zoom`) as a secondary force-declare guard: if the new AABB has Jaccard > threshold with recent history, the algorithm declares a needle at the current best unpenalized point rather than entering another zoom.

---

### GUI: Live Log Panel (`interface/app.py`)

The main application window now includes a **live run log panel** in the bottom-right area, visible whenever an experiment is running.

#### Layout Change

The right pane is now split vertically using a `ttk.PanedWindow`:
- **Top (weight=3):** SnapshotSlider + Notebook tabs (Convergence, Distance, Points, Needles, GP Query) — unchanged
- **Bottom (weight=1):** `LiveLogPanel`

#### `LiveLogPanel(ttk.LabelFrame)`

A self-contained widget with:
- **Static status line** — a `tk.StringVar`-backed `ttk.Label` (navy, bold, sunken relief) that shows the last parsed `[A/Z/I]` activation/zoom/iteration, current bounds, proposed candidate, and top-2 line endpoints.
- **Scrollable log** — a `scrolledtext.ScrolledText` with colour tags:
  - `info` — default foreground
  - `warn` — orange text
  - `error` — red text
  - `success` — green text
  - `failure` — dark red text

`LiveLogPanel.log(msg, tag=None)` parses the message for known patterns, updates the status line if a match is found, and appends the message to the log.

`LiveLogPanel.clear()` resets all state and clears the log.

#### Log Routing

`NewRunDialog._log_msg(msg, tag)` now calls `self._app.log_to_main(msg, tag)` in addition to writing to the dialog's own log widget. `ZoMBIApp.log_to_main(msg, tag)` delegates to `self._live_log.log(msg, tag)`.

#### Parsed Log Patterns

| Pattern | Example | Status field updated |
|---------|---------|----------------------|
| `[A1/Z2/I3]` | `[A0/Z1/I5]` | Activation, zoom, iteration |
| `--- zoom N/M` | `--- zoom 2/3 ---` | Zoom level |
| `candidate: [...]` | `candidate: [0.3, 0.4, 0.3]` | Candidate composition |
| `line_0 left: [...]` | `line_0 left: [0.1, 0.8, 0.1]` | Line left endpoint |
| `line_0 right: [...]` | `line_0 right: [0.5, 0.3, 0.2]` | Line right endpoint |
| `[FAILURE]` / `activation failed` | — | Status tag → `failure` |

#### Hardware Script Default

The Hardware tab in `NewRunDialog` now defaults to `scripts/run_zombi_main.py` (previously `scripts/main.py`).

---

### `scripts/run_zombi_main.py` — API Fixes

The hardware runner script had stale constructor calls. Fixed:

| Old | New |
|-----|-----|
| `bounds=bounds` kwarg | Removed (not a valid `ZoMBIHop` param) |
| `penalization_threshold`, `penalty_num_directions`, `penalty_radius_step` | Removed (old penalty API) |
| `penalty_max_radius=...` | `max_penalty_radius=...` |
| Resume: `X_init_actual=None, bounds=bounds_resumed` | Resume: dummy `torch.zeros(0, d)` tensors for X_init_actual, X_init_expected, Y_init; `run_uuid=resume_uuid` |
