# Meeting Notes: 10D Digital Twin via ELA Fingerprint Matching (Strategy S1)

**Date:** July 2026  
**Project:** ZoMBI-Hop — compositional Bayesian optimization on the probability simplex  
**Attendees / context:** Campaign modeling and MOBO benchmarking workstream  
**Reference:** Muñoz & Smith-Miles (2019), *Generating New Space-Filling Test Instances for Continuous Black-Box Optimization* ([doi:10.1162/evco_a_00262](https://doi.org/10.1162/evco_a_00262))  
**Data:** `data/2nd_real_run.db` (644 measured 3D compositions)  
**ELA output:** `data/2nd_real_run_ela_full.json`  
**Code:** `ela/features.py`, `ela/pflacco_port.py`, `ela/compute_lambda_target.py`

---

## Purpose of the meeting

We met to decide how to extend ZoMBI-Hop benchmarking from real 3D (soon 4D) campaign data to **10D** before 10D hardware exists. The goal is not a physical twin of the perovskite system, but a **landscape twin**: a synthetic 10D objective whose exploratory structure matches our real campaign, so we can tune MOBO hyperparameters on a representative oracle.

We agreed to follow **Muñoz & Smith-Miles Strategy S1** — evolve a symbolic landscape whose ELA fingerprint matches a target — but **not** to pick from existing synthetic libraries (`Ensemble`, `ackley`, etc.). The generator will be genetic programming (GP) on the 10-simplex in ILR coordinates.

---

## Problem we are solving

**Question:** Can we build a 10D synthetic landscape whose exploratory structure matches our real 3D (and later 4D) campaign, so ZoMBI-Hop can be benchmarked before 10D data exists?

Muñoz validated S1 by targeting **COCO benchmark functions** and checking whether GP could recreate them from ELA features alone. We invert that logic:

- **Their target:** a known analytic COCO function. **Ours:** a **500-tree random forest surrogate** of the real campaign (`random_state=42`).
- **Their generator:** symbolic GP on `[-5, 5]^D`. **Ours:** symbolic GP on **Δ⁹** (the 10-component simplex).
- **Their setting:** same input dimension (2D→2D, 10D→10D). **Ours:** cross-dimension — 3D/4D fingerprint → 10D output.
- **Their validation:** surface plots. **Ours:** feature recovery plus MOBO transfer experiments.

The output is anchored to measured data on the known 3D (and later 4D) subspace via an explicit subspace RMSE penalty in the GP fitness, not ELA alone.

---

## Agreed method

**Pipeline:**

1. Load real campaign → train RF surrogate `f₃`.
2. Compute target fingerprint `λ_T = ELA(f₃)`.
3. Evolve `g₁₀` on Δ⁹ with GP; fitness balances ELA match, subspace fidelity, and expression complexity.
4. Export a callable 10D oracle for fixed MOBO benchmarking.

**Planned fitness function:**

`J = ‖W(λ(g₁₀) − λ_T)‖₂ + α · RMSE₃D(g₁₀, f₃) + β · |tree nodes|`

- `λ_T` is the Tier-1 feature vector from the real campaign (see below).
- `RMSE₃D` enforces `g₁₀(E₃(u)) ≈ f₃(u)` on the embedded 3D face.
- `α` should be large enough that we do not sacrifice subspace fidelity for a better ELA match.

---

## Sampling protocol (two samples — do not conflate)

We use **two distinct samples**. Mixing them invalidates interpretation.

### Sample A — measured campaign (sparse)

- **Source:** `2nd_real_run.db`, `results` table.
- **Columns:** `FAPbI3`, `MAPbI3`, `MAPbBr3`, `Objective`.
- **Rows:** 644 (non-null on all four); each row renormalized to sum to 1.
- **Design:** experimental (line-biased, not uniform on the simplex).

**Used for:** training the RF; computing **`oob_r2`** only among Tier-1 features.

### Sample B — dense ELA sample (surrogate evaluations)

- **Size:** 4096 points = 2¹² (next power of 2 ≥ D×1000 = 3000).
- **Design:** scrambled Sobol in the 2D ILR box → inverse Helmert ILR → Δ²; seed 42.
- **ILR box:** from 5000 `Dirichlet(1,1,1)` probes with ±0.25 margin per axis.
- **Objective:** `yᵢ = f_RF(xᵢ)`.

**Used for:** all other ELA features.

**Coordinate choices:**

- Meta-model, classifiers, FDC, dispersion, ξ → **ILR** (matches ZoMBI `GPSimplex`).
- H(Y), γ(Y), PKS → objective values only.
- median_lipschitz → compositional Euclidean L₂, radius **0.064** (deposition noise scale).

**Reproduce:**

```bash
python ela/compute_lambda_target.py --db data/2nd_real_run.db
python ela/compute_lambda_target.py --db data/2nd_real_run.db --full
```

---

## ELA feature selection

Muñoz computed 33 features and reduced to 8 via correlation clustering. For ZoMBI-Hop we use **10 Tier-1 features**: Muñoz's 8 plus **OOB R²** and **median Lipschitz**. These are the only features recommended for GP fitness. Full mode computes 93 features for diagnostics.

### Tier-1 target vector (`2nd_real_run`, maximize=True, n_dense=4096, seed=42)

| Feature | Value | Interpretation for this campaign |
|---------|------:|----------------------------------|
| R²_Q | 0.584 | Moderate global structure; not a simple quadratic bowl |
| CN | 1.027 | Nearly isotropic curvature in the quadratic fit |
| H(Y) | 3.052 | Moderate entropy; mixed objective levels across the simplex |
| ξ(1) | 0.223 | Modest main effects; interactions also matter (ξ(2) ≈ 0.32) |
| γ(Y) | −0.094 | Slight negative skew — few very good compositions |
| EL25 | 0.892 | Top 25% occupy a coherent region in ILR space |
| LQ25 | 0.952 | Linear structure nearly sufficient for good/bad separation |
| PKS | 3 | Moderately multimodal RF surface (KDE peak count) |
| oob_r2 | 0.288 | Sparse 644-point campaign; surrogate is hard to fit |
| median_lipschitz | 1.142 | Moderate local slope at noise scale 0.064 |

**Surrogate range:** Y_dense ∈ [0.307, 0.789]; measured Y_campaign ∈ [0.278, 0.860].

**Important consistency notes:**

- An early Tier-1-only run gave R²_Q ≈ 0.11 and PKS ≈ 18 (histogram peaks). The **full pipeline** uses flacco-aligned meta-model fitting and **KDE-based PKS** (PKS = 3). Use values from `2nd_real_run_ela_full.json` going forward and keep the method fixed for all evolution runs.
- R²_Q and CN differ between minimal and full implementations because the full run uses flacco-aligned `quad_simple` (linear + squared terms).

### Diagnostic features (compute but do not optimize)

- **FDC = −0.671** — negative fitness-distance correlation; globally non-deceptive (fitness improves toward the best sample).
- **DISP1% = 1.474** — spread of near-optimum points.
- **R²_L = 0.54, R²_LI = 0.70** — interactions contribute beyond pure quadratic structure.
- **R²_QI = 0.829** — strong interaction curvature.
- **κ(Y) = −1.32** — light-tailed cost distribution.
- **Hmax = 0.63, M0 = 0.42** — information-content ruggedness along NN tour.

### Landscape characterization (consensus)

The 3D campaign RF fingerprint describes a **multimodal haystack with coherent good regions**:

- Multimodal but not extreme (PKS = 3).
- Globally non-trivial (R²_Q ≈ 0.58) with clear separability of top performers (EL25 ≈ 0.89).
- Strong funnel toward the optimum (FDC ≈ −0.67).
- Sparse and noisy in practice (oob_r2 ≈ 0.29).
- Locally moderately rough at instrument scale (lipschitz ≈ 1.14 at 0.064).

This aligns with ZoMBI-Hop's design: multiple needles, input noise 0.064, simplex geometry.

---

## Four-dimensional extension (when data arrives)

1. Compute λ(f₄) with the same protocol (same `sample_seed`, same RF settings).
2. Compare ‖λ(f₄) − λ(f₃)‖:
   - **Small shift** → average targets: λ_T = ½(λ₃ + λ₄).
   - **Large shift** → 3D-only target is unreliable; use 4D only or bracket with two twins.
3. Add a **4D subspace penalty** to GP fitness: g₁₀(E₄(v)) ≈ f₄(v).
4. Optionally discard 3D-only λ_T if the extra element materially changes the fingerprint.

Do not assume 3D ELA parameters remain valid at 10D without this check.

---

## Cross-dimension matching (3D → 10D)

ELA alone does not uniquely define a 10D function. We discussed mitigations:

- **PKS scales with dimension** — down-weight in fitness or rank-transform within a library.
- **ξ and CN are dimension-sensitive** — consider a 2D PCA of Tier-1 features across instances as a primary target.
- **Optima in the 7 extra dimensions are unknown** — subspace RMSE to f₃ (and later f₄) is essential.
- **Many 10D functions can share the same λ** — acceptable for benchmarking; this is not a physical extrapolation claim.

---

## Runtime and MOBO strategy

| Task | Estimated wall-clock |
|------|---------------------|
| Compute λ_T from 3D DB | ~1–5 min |
| GP evolve one 10D landscape | ~10–12 h |
| GP evolve 3D landscape (pipeline test) | ~2–4 h |

**Decision:** Recreating a new randomized 10D landscape for every MOBO trial is infeasible.

**Agreed plan:**

1. Evolve ~5 independent 10D landscapes (different GP seeds).
2. Keep those passing acceptance criteria (subspace RMSE + Tier-1 feature RMSE).
3. Run MOBO hyperparameter search on this **fixed small suite** — same oracle reused across trials.
4. Recompute λ_T only when campaign data or protocol changes.

---

## Acceptance criteria for evolved 10D twins

**Must pass (Tier 1):**

- Subspace RMSE: RMSE(g₁₀(E₃), f₃) < 0.02 × range(Y).
- Tier-1 feature RMSE: median relative error per feature < 10% (normalized by σⱼ).
- Stability: same expression + seed → identical λ.

**Should pass (Tier 2):**

- Lipschitz within 25% of campaign value.
- Landscape should not be trivially smooth (oob_r2 match not required — 10D analytic ≠ RF).

**Downstream validation (Tier 3):**

- MOBO hyperparameters tuned on the twin transfer better to 3D RF than hyperparameters from generic `ackley10d`.

---

## Action items

- [ ] Save fixed `X_dense.npy` / `Z_dense.npy` for evolution (seed=42).
- [ ] Export `tier1_target.json` with values and feature weights W.
- [ ] Implement `ela/evolve.py` (DEAP/GP + Tier-1 fitness + 3D subspace term).
- [ ] Pilot on **3D** GP recreation (~2–4 h) before committing to 10D (~12 h each).
- [ ] Evolve **5** 10D landscapes; wire the best as fixed MOBO oracles (`real_twin_10d` in `run_mobo.py` / `evaluate.py`).
- [ ] On 4D data: recompute λ_T, compare to 3D, update targets and subspace masks.

---

## References

- Muñoz, M. A., & Smith-Miles, K. (2019). Generating new space-filling test instances for continuous black-box optimization. *Evolutionary Computation*.
- Muñoz, M. A., & Smith-Miles, K. (2017). Performance analysis of continuous black-box optimization algorithms via footprints in instance space. *Evolutionary Computation*.
- Mersmann, O., et al. (2011). Exploratory landscape analysis. *GECCO*.
- Muñoz, M. A., Kirley, M., & Halgamuge, S. K. (2015). Information content of fitness landscapes.

---

*Notes reflect `2nd_real_run_ela_full.json` (644 campaign rows, 4096 dense Sobol sample, seed 42).*
