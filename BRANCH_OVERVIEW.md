# ZoMBI-Hop Optimization & Synthetic Data — Branch Overview

## First-Time Setup: Copy the `runs/` Folder

The `optimize/runs/` folder is **not tracked by git** (it contains large binary artifacts and auto-generated data). Before running anything, manually copy it from the shared Dropbox location into your local repo:

```
Source:       MIT Dropbox/Buonassisi-Group/Projects - Active/ZoMBI-Hop/06_Code/ZoMBI-Hop/optimize/runs/
Destination:  <your-local-repo>/optimize/runs/
```

Without this folder, `pareto.py` and `evaluate.py` will have nothing to analyze, and `run_mobo.py` will start fresh with no prior data.

---

## Directory Overview

### `optimize/` — Hyperparameter Tuning via MOBO

This directory contains the machinery for finding good ZoMBI-Hop hyperparameters using **multi-objective Bayesian optimization** (MOBO). Three objectives are jointly minimized:

- `dist_to_needles` — how close discovered peaks are to the true optima
- `dup_fraction` — fraction of redundant (duplicate) samples
- `runtime_s` — wall-clock seconds per trial

The optimization engine is **qLogNEHVI** (BoTorch), and the surrogate landscape is a Random Forest trained on campaign1a.csv (ternary composition → objective).

#### Key Scripts

| Script | What It Does |
|---|---|
| `run_mobo.py` | **Main entry point.** Runs the MOBO loop to search for hyperparameters. Starts with an interactive extrema picker, then runs Sobol init + BO trials indefinitely. |
| `pareto.py` | Computes the global Pareto front across all MOBO runs in `runs/`. Produces `pareto.json` and an interactive plot (hover to compare, click to open trial images). |
| `evaluate.py` | Re-evaluates specific hyperparameter sets (from a MOBO run) multiple times on one or more datasets to check reproducibility and cross-dataset transfer. |
| `make_videos.py` | Assembles per-iteration PNG frames into MP4 timelapse videos. |
| `plot_metrics.py` | Plots `metrics_over_time.csv` for a given trial. |
| `analyze_variance.py` | Ranks hyperparameter sets by reproducibility using pre-populated `variance_results.json`. |

#### `optimize/runs/` Layout

```
optimize/runs/
├── mobo_DD_MM_HH_MM/          # One MOBO search run
│   ├── run_config.json        # Static config (CSV path, optima, maximize flag)
│   ├── mobo_progress.json     # Updated each trial
│   └── trial_<n>/
│       ├── trial.json         # Metrics + hyperparameters for this trial
│       ├── points.csv / needles.csv / metrics_over_time.csv
│       ├── *.png              # Static diagnostic plots
│       ├── plots/iter_*.png   # Per-iteration landscape frames
│       └── zombihop_timelapse.mp4
├── IGNORE_mobo_DD_MM_HH_MM/   # Archived runs (excluded from pareto.py by default)
├── rerun_DD_MM_HH_MM/         # Evaluation run (from evaluate.py)
│   ├── rerun_config.json
│   └── <dataset>/trial_<n>/run_<k>/
│       ├── metrics.json
│       ├── points.csv / needles.csv / metrics_over_time.csv
│       ├── *.png
│       └── point_cloud.html   # 4D only
└── pareto.json
```

Runs prefixed with `IGNORE_` are treated as archived and skipped by `pareto.py` unless you pass `--with-old`.

---

### `synthetic_data/` — Analytic Benchmark Objectives

This directory provides synthetic test functions and interactive visualization tools. The primary use is benchmarking ZoMBI-Hop on analytic objectives (so ground truth is known exactly) and tuning those objectives interactively.

#### Key Scripts

| Script | What It Does |
|---|---|
| `ackley.py` | Defines the `Ackley` class — a negated Ackley function on probability simplices in arbitrary dimensions. Supports variants: `centroid`, `edge`, `vertex`, `multimodal`, `realistic`. The `realistic` variant uses Dirichlet-sampled peaks + Perlin-style noise, configurable via `ackley/defaults.json`. |
| `plot_3d.py` | Interactive Dash app for tuning the 3-simplex `realistic` Ackley. Adjust n\_optima, basin\_width, noise\_freq, noise\_amp with sliders; click "Save as Default" to persist to `defaults.json`. Run with `python synthetic_data/plot_3d.py` → [http://127.0.0.1:8050](http://127.0.0.1:8050) |
| `plot_4d.py` | Interactive 3D point-cloud viewer for the 4-simplex Ackley (tetrahedron embedding). Also provides `add_simplex_overlays()` used by `evaluate.py` to render ZoMBI-Hop artifacts (lines, needles, penalization ellipsoids) on top of the landscape. |
| `analyze_basin_vol.py` | Diagnostic script: samples 200K points uniformly on each simplex and histograms objective values for 3D, 4D, and 10D — illustrates how near-optimal basins shrink with dimensionality. |

#### Config

- `ackley/defaults.json` — global defaults for the `realistic` Ackley variant (n\_optima, basin\_width, noise\_freq, noise\_amp). Updated via the Dash "Save as Default" button.

---

## How to Use

At the top of each file should be a "Usage" section that describes how to use each file with its flags.
