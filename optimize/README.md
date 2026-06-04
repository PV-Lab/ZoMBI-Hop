# optimize/

Scripts for multi-objective Bayesian optimisation (MOBO) of ZoMBI-Hop hyperparameters
and post-run analysis.

---

## run_mobo.py

Runs MOBO of ZoMBI-Hop hyperparameters against a Random Forest surrogate built from
`campaign1a.csv`. Optimises three objectives simultaneously (all minimised):

- `dist_to_needles` — mean greedy distance from discovered needles to the nearest true optimum
- `dup_fraction` — fraction of sampled points whose nearest neighbour is within the noise radius
- `runtime` — wall-clock seconds for the ZoMBI-Hop run

Uses **qLogNEHVI** (BoTorch) with a Sobol initialisation phase followed by an unbounded BO
loop. Each trial runs ZoMBI-Hop until the per-trial wall-clock budget (`TIME_LIMIT_HOURS`)
expires. Press `Ctrl+C` to stop at any time; results are written after every trial.

**On a fresh run**, an interactive ternary plot appears for you to click the reference
extrema (maxima or minima) of the RF landscape.

```
conda activate zombi-hop
python optimize/run_mobo.py
```

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--resume` | flag | off | Crawl every `runs/mobo_*/mobo_progress.json`, collect all prior (X,Y) pairs, and seed a **new** `runs/mobo_*` run from the full landscape. Reuses the latest run's saved RF settings and reference optima — fully non-interactive. |
| `--max-trials` | int | unbounded | Stop after this many trials. Without this flag the loop runs until `Ctrl+C`. |

### Examples

```bash
# Fresh run (interactive: pick maximize/minimize and click reference optima)
python optimize/run_mobo.py

# Fresh run, stop after 20 trials
python optimize/run_mobo.py --max-trials 20

# Resume from all prior runs, non-interactive
python optimize/run_mobo.py --resume

# Resume and cap at 10 additional trials
python optimize/run_mobo.py --resume --max-trials 10
```

### Output layout

Each run creates `runs/mobo_DD_MM_HH_MM/` containing:

```
runs/mobo_DD_MM_HH_MM/
├── mobo_progress.json          running summary (updated after every trial)
├── mobo_results.json           same content as mobo_progress.json (written on exit)
├── mobo_results.pt             PyTorch checkpoint (used for resume)
├── run_config.json             static run config (RF path, optima, etc.)
├── sobol_design.pt             persisted Sobol init design
├── pareto_front.png            written on exit
├── errors.log                  appended on trial failures (if any)
└── trial_<n>/
    ├── trial.json              phase / pareto flag / metrics / hparams
    ├── points.csv
    ├── needles.csv
    ├── metrics_over_time.csv
    ├── dist_from_centre.png
    ├── line_length_hist.png
    ├── hparam_edge_proximity.png
    ├── plots/
    │   ├── iter_0000.png
    │   └── iter_NNNN.png …
    └── zombihop_timelapse.mp4
```

#### `mobo_progress.json` / `mobo_results.json`

Running summary of all trials in this run, updated atomically after every completed trial.
Contains trial-level metrics, hyperparameters, Pareto membership flags, and aggregate
statistics. `mobo_results.json` is written on exit with the same content.

#### `trial_<n>/points.csv`

All sampled points from this trial.

| Column | Description |
|--------|-------------|
| `sample_idx` | Sequential sample index |
| `FA` | FAPbI3 composition fraction |
| `MA` | MAPbI3 composition fraction |
| `Br` | MAPbBr3 composition fraction |
| `Y` | Observed objective value |
| `penalized` | 1 if this point is inside a needle's penalisation ellipsoid, 0 otherwise |
| `activation` | ZoMBI activation number when this point was sampled |
| `zoom` | Zoom level when this point was sampled |

#### `trial_<n>/needles.csv`

All needles (discovered local optima) from this trial.

| Column | Description |
|--------|-------------|
| `needle_idx` | Sequential needle index |
| `FA` | FAPbI3 composition fraction |
| `MA` | MAPbI3 composition fraction |
| `Br` | MAPbBr3 composition fraction |
| `value` | Objective value at the needle |
| `median_value` | Median objective value in the neighbourhood of the needle |
| `activation` | ZoMBI activation when the needle was confirmed |
| `zoom` | Zoom level when the needle was confirmed |
| `iteration` | Iteration number when the needle was confirmed |
| `dist_to_centre` | Euclidean distance from the needle to the simplex centroid (1/3, 1/3, 1/3) |

#### `trial_<n>/metrics_over_time.csv`

One row per ZoMBI-Hop iteration, tracking optimisation quality over time.

| Column | Description |
|--------|-------------|
| `iteration` | ZoMBI iteration number |
| `dist_to_needles` | Mean greedy distance from discovered needles to the nearest true optimum; unmatched true optima each add a fixed penalty |
| `dup_fraction` | Fraction of sampled points whose nearest neighbour in input space is within `noise/2` |
| `pct_matched` | Percentage of true optima that have at least one needle within `MATCH_RADIUS` |
| `avg_pairwise_dist` | Average pairwise Euclidean distance between all discovered needles |

#### `trial_<n>/dist_from_centre.png`

Scatter plot of each sampled point's Euclidean distance from the simplex centroid (x-axis)
vs. its objective value (y-axis), with needles highlighted. Mirrors the "Distance from
simplex centre" panel in the interactive GUI (`interface/app.py`).

#### `trial_<n>/line_length_hist.png`

Histogram of the composition-space L2 lengths of the main LineBO line proposed at each
iteration, with the mean marked. Shows how the trust region tightens or expands over time.

#### `trial_<n>/hparam_edge_proximity.png`

Horizontal bar chart showing how close each chosen hyperparameter value is to the nearest
boundary of its search range. Proximity = min(v, 1−v) where v is the normalised [0, 1]
value; 0 means sitting at `lo` or `hi`, 0.5 means dead centre. Bars in red are within
0.05 of an edge.

#### `trial_<n>/plots/`

Directory containing one PNG frame per ZoMBI-Hop iteration (`iter_0000.png`, `iter_0001.png`, …).
Each frame is a two-panel ternary plot: the RF reference landscape on the left and the
current ZoMBI-Hop exploration state (sampled points, trust region, needles, LineBO lines)
on the right.

#### `trial_<n>/zombihop_timelapse.mp4`

Timelapse video compiled from the `plots/` frames, targeting ~30 s duration with FPS
scaled to the number of frames. Built by `make_videos.py` (see below).

---

## make_videos.py

Compiles the per-iteration `iter_*.png` frames produced by `run_mobo.py` into
`zombihop_timelapse.mp4` timelapse videos. Videos are ~30 s long with FPS scaled
automatically to the number of frames (clamped to 1–60 fps). Tries `imageio`+`ffmpeg`
first, then falls back to OpenCV.

```bash
# Newest run (default)
python optimize/make_videos.py

# Specific run directory
python optimize/make_videos.py runs/mobo_04_06_11_47

# Rebuild all videos even if they already exist
python optimize/make_videos.py runs/mobo_04_06_11_47 --force
```

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `run_dir` | positional (optional) | newest `mobo_*` run | Path to a `runs/mobo_*` folder. Accepts an absolute path, a path relative to cwd, or just the folder name (resolved under `runs/`). |
| `--force` | flag | off | Rebuild videos even when `zombihop_timelapse.mp4` already exists and is non-empty. |

---

## plot_metrics.py

Plots the four time-series metrics from a `metrics_over_time.csv` file produced by
`run_mobo.py`. Displays a 2×2 grid of line plots (one per metric) using matplotlib.

```bash
python optimize/plot_metrics.py <csv_path> [--log-x] [--log-y]
```

### Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `csv_path` | positional (required) | — | Path to a `metrics_over_time.csv` file (e.g. `runs/mobo_04_06_11_47/trial_2/metrics_over_time.csv`). |
| `--log-x` | flag | off | Use log scale on the x-axis (iteration). |
| `--log-y` | flag | off | Use log scale on the y-axis (metric values). |

### Examples

```bash
python optimize/plot_metrics.py optimize/runs/mobo_04_06_11_47/trial_2/metrics_over_time.csv
python optimize/plot_metrics.py optimize/runs/mobo_04_06_11_47/trial_2/metrics_over_time.csv --log-y
python optimize/plot_metrics.py optimize/runs/mobo_04_06_11_47/trial_2/metrics_over_time.csv --log-x --log-y
```

### Metrics plotted

| Column | Description |
|--------|-------------|
| `dist_to_needles` | Mean distance from discovered needles to the nearest true optimum |
| `dup_fraction` | Fraction of sampled points with a near-duplicate |
| `pct_matched` | Percentage of true optima matched by at least one needle within `MATCH_RADIUS` |
| `avg_pairwise_dist` | Average pairwise Euclidean distance between discovered needles |
