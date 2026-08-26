# `benchmarks/public_db` — public experimental datasets

Pulls real experimental datasets from **Olympus**
([the-matter-lab/olympus](https://github.com/the-matter-lab/olympus)) without
taking on Olympus itself as a dependency. Olympus stores each dataset as three
plain files, and those are fetched verbatim and cached under `data/<name>/`, so
every load after the first is offline and byte-identical to upstream.

## Fetch

```bash
.venv/Scripts/python.exe benchmarks/public_db/olympus.py --fetch all
.venv/Scripts/python.exe benchmarks/public_db/olympus.py --info all
.venv/Scripts/python.exe benchmarks/public_db/olympus.py --describe hplc
```

## Load

```python
from benchmarks.public_db import load

ds = load("hplc")
ds.X            # (1386, 6) in physical units (ml, ml/min, Hz, s)
ds.unit_X       # the same, min-max scaled to [0,1] — use this for anything metric
ds.Y            # (1386,) peak area
ds.labels       # readable component names
ds.simplex      # False — the parameters are a box, not a composition
ds.goal         # "maximize"
ds.bounds       # (6, 2) declared [low, high] per parameter
```

## The four curated datasets

| name | d | constraint | goal | measures |
|---|---|---|---|---|
| `photo_pce10` | 4 | simplex | minimize | photodegradation of a PCE10/P3HT/PCBM/oIDTBR OPV blend |
| `photo_wf3` | 4 | simplex | minimize | the same, with WF3 in place of PCE10 |
| `hplc` | 6 | none | maximize | HPLC peak area vs six process parameters |
| `crossed_barrel` | 4 | none | maximize | toughness of a 3D-printed crossed-barrel structure vs four geometry parameters — [Gongora et al. 2020](https://doi.org/10.1126/sciadv.aaz1708) |

## Things worth knowing before you plot these

These are the details that decide which diagram each one lands on in
`visualization/plot_run.py`, and the ones most likely to surprise:

- **Both `photo_*` sets are 4-component simplices, not 3.** Rows sum to exactly
  1.0 and the design matrix has rank 3, so they are *quaternary* blends and draw
  as a **tetrahedron**, never as a ternary triangle. The source paper is
  literally titled "Beyond Ternary OPV" — quaternary is the point of it. Only
  839 of the 1040 points are interior; the other 201 sit on faces, edges or
  vertices, and the compositions are quantised to a 0.02 grid. `mat_3` (PCBM)
  never exceeds 0.9, so that corner of the tetrahedron is never reached pure.
- **`photo_pce10` and `photo_wf3` share an identical design grid.** The same
  1040 X rows in the same order; only the measured degradation differs (65 rows
  happen to coincide). That makes them a clean paired comparison — the same
  design, one material swapped.
- **`hplc` is the only non-simplex set, and its columns span four orders of
  magnitude** — `sample_loop` 0–0.08 ml against `push_speed` 80–150 Hz. Row-
  normalise the raw columns and `push_speed` is **94%** of every row, so any
  distance-based view (UMAP/CoNet, a GP with one shared length scale) built on
  raw units describes that one column and nothing else. Use `unit_X`.
- **`hplc` has 229 exact-zero peak areas** (failed injections) out of 1386, and
  **379 replicate rows** (1007 unique X, at most 2 measurements each).
- **`crossed_barrel` is `d=4` *without* a simplex, which is the whole point of
  carrying the constraint separately from the dimensionality.** It has the same
  four columns as the `photo_*` pair and lands on a completely different diagram
  — a **scatter-plot matrix**, not a tetrahedron — because its parameters are
  independent geometry (number of hollow columns, twist angle, outer radius,
  thickness), not a composition. It is the only curated dataset that exercises
  the SPLOM branch.
- **`crossed_barrel` is a lattice, not a continuum.** All four columns are
  declared `continuous` but take only 4 / 9 / 11 / 3 distinct values; its 600
  rows are roughly half of that 1188-point full factorial, with **no
  replicates**. A surrogate over it is interpolating a grid. `theta` (0–200)
  is 77% of every raw row, so it wants `unit_X` for the same reason `hplc` does.
- **The goals differ.** `photo_*` minimises; `hplc` and `crossed_barrel`
  maximise. A best-so-far envelope hard-coded to `max` is wrong for two of the
  four, which is why `goal` is carried through to the plotting layer.

## Adding another Olympus dataset

Nothing here is hard-coded to the four. Any dataset directory that exists
upstream at `src/olympus/datasets/dataset_<name>/` works:

```bash
.venv/Scripts/python.exe benchmarks/public_db/olympus.py --fetch suzuki
```

It will be picked up by `available()` and appear in the `plot_run.py` app's
**Public dataset** dropdown. Add an entry to `PRETTY_LABELS` in `olympus.py` if
its `config.json` names the parameters with placeholders rather than real
component names.
