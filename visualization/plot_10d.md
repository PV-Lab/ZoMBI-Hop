# How to Read the CoNet Plot (`plot_10d.py`)

A quick guide to interpreting the **CoNet** (Composition Co-occurrence Network) view of a
ZoMBI-Hop run. This is the single-panel, run-directory version of the DiSCO live viewer's
CoNet — one map, coloured by the campaign's **Objective**.

---

## What the plot *is*

Every **node** is one **measured sample** (one composition the optimizer actually made and
characterized). The nodes are laid out by a **UMAP embedding** of composition similarity — a
t-SNE-style map. There are **no meaningful axes**: position encodes *how similar samples are
to each other in composition*, nothing else. Two nodes near each other have similar recipes;
two far apart have very different ones.

The little **UMAP-1 / UMAP-2** L-arrows off the bottom-left corner are just there to say
"this is an embedding" — the numbers on those directions carry no physical units.

---

## The four things a node encodes

| Visual channel | Meaning |
|---|---|
| **Position** | Composition similarity (UMAP). Clusters = families of similar recipes. |
| **Colour** | The measured **Objective** value (viridis: dark purple = low, yellow = high). |
| **Size** | **Local sampling density** — how many similar samples surround it. Bigger = a composition region the campaign sampled heavily. |
| **Red rim** | Belongs to the **most recent acquisition batch** (drawn on top of everything). |

> Colour bounds are the **10th–90th percentile** of the Objective, so the extremes don't wash
> out the middle. The colourbar on the right is labelled "Objective"; the ▲/▼ end-caps mean
> values run beyond the shown range.

Nodes with **no measured value** appear as small hollow grey dots.

---

## The coloured background (dominance zones)

The soft, blended colour wash behind the nodes is the **composition field**: at each point in
the map it shows *which component dominates the recipes there*. Each component has its own
hue (see the legend). Where one component clearly dominates you get a fairly pure hue; where
recipes are mixtures you get a blend of hues.

Dashed / solid **outlines** enclose **regions**:

- **Solid outline + single label** = a **pure region**: one composition dominates that
  neighbourhood (its field is strongly above threshold).
- **Dashed concentric outlines (multi-colour) + stacked label** = a **blend region**: two or
  more compositions co-exist there. Each dashed ring is coloured for one co-existing
  component.

**Labels** sit in empty space and are tied back to their region by a thin **leader line**
(labels are placed away from clutter, so the leader tells you which region a label belongs
to). Not every region is labelled — only the larger ones get a label, but every region keeps
its boundary. Labels use bracketed chemistry names, e.g. `[MAPbI₃]`.

> Region reach is keyed to the *core* of the data (10–90th percentile extent), so a lone
> far-flung cluster can't balloon its zone across the empty parts of the frame.

---

## The stars ⭐ (best compositions / "needles")

Gold **stars** mark the **best compositions**:

- If ZoMBI-Hop **needle** data is available: one star per ranked optimum, brightest for rank 0
  (best) and fading for lower ranks. These are the optimizer's *proposed* optima.
- Otherwise: a single star on the **best measured Objective so far** (the incumbent).

A **selected** star (clicked) gets a red edge and a star-shaped glow.

---

## Reading it at a glance

1. **Find the yellow.** Bright yellow nodes and stars are your high-Objective samples — the
   good recipes.
2. **Look at where they cluster.** The background zone(s) under the yellow tell you *which
   composition family* is winning.
3. **Big vs. small nodes.** Large nodes = the campaign spent a lot of effort in that recipe
   region (it was promising or being refined); small isolated nodes = lightly explored recipes.
4. **Red rims = the newest batch.** Where the optimizer just looked. If the red rims sit near
   yellow, it's honing in; if they're off in dark regions, it's still exploring.
5. **Blend vs. pure regions.** Dashed multi-colour rings mean the good recipes there are
   *mixtures*, not a single pure compound.

---

## Interactive use

```bash
# interactive window (from repo root, using the repo uv venv):
python visualization/plot_10d.py --run run_9dfe

# headless render to PNG:
python visualization/plot_10d.py --run run_9dfe --save conet.png
```

- **Click a node** → the right-hand legend is replaced by an **inspector** showing that
  sample's number, the full **composition formula**, each component's **fraction**, and its
  measured **Objective**.
- **Click empty space** or the **✕** → deselect (legend returns).
- A tight click on a dense stack selects the **highest-Objective** node under the cursor.

### Useful flags

| Flag | Effect |
|---|---|
| `--run` | Run directory or bare run name (under `runs/`). |
| `--snapshot` | Reconstruct up to a specific snapshot (default: `latest.txt`). |
| `--save PNG` | Render once to a PNG and exit (no window). |
| `--purity` | Purity-layout threshold. `0` reverts to the plain plurality layout + regions. |
| `--spread` | UMAP `min_dist` — how spread out the points are (needs a UMAP re-fit). |
| `--gap-reach` | Squash factor pulling far-flung satellite clusters back toward the core. |

---

## Layout knobs, briefly (why the map looks the way it does)

These don't change the data, only how it's *arranged* for legibility:

- **Purity layout** (`--purity`, default 0.3): samples *purer* than the threshold gravitate to
  their cluster core; near-equal *mixtures* get flung outward, so mixtures visually separate
  from true majority compositions.
- **Gap reach** (`--gap-reach`): when the campaign resamples one composition over and over
  (e.g. near-pure CsPbI₃), UMAP can exile that clique far away, leaving dead space. This
  squashes those distances back in.
- **Spread** (`--spread`): UMAP `min_dist` — bigger spreads points out more within/between
  clusters.

---

## What is *not* shown (vs. the live DiSCO viewer)

This run-directory version intentionally drops: the Ternary view, GP-ternary landscape,
bottom line strips, XRD / IV / PL data, per-iteration time-travel, and within-iteration
smoothing. It shows a **single Objective response**, so points are ordered by **acquisition
order** (used only for draw order and the latest-batch red rim), not by a real iteration index.
