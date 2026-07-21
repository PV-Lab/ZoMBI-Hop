"""
visualization/plot_10d.py
=========================
Self-contained, interactive **CoNet** (composition co-occurrence network) view of
a ZoMBI-Hop *run directory*. The CoNet is a UMAP embedding — a t-SNE-style map —
where every node is one measured sample, positioned by composition similarity,
coloured by the measured Objective, sized by local sampling density, drawn over
faded per-component "dominance zone" backgrounds. Click a node to inspect it.

This is a trimmed, dependency-free merge of the DiSCO ``visualization.py``: ONLY
the CoNet is kept — no Ternary view, GP-ternary background, bottom line strips,
XRD needle strips, or Tkinter live GUI — and the data comes from a finished run's
delta snapshots (``reconstruct_snapshot_tensors``) instead of a live results.db.
The ranked-needle optima ARE drawn as CoNet stars (from a snapshot ``needles.json``
or a ``needles.csv``), toggled by the top-left button / 'n' key, off by default.
It reproduces the CoNet drawing 1:1 by reusing the original drawing helpers
verbatim; ``visualization.py`` is no longer imported and can be removed.

Usage
-----
  # interactive matplotlib window (click a node to inspect it):
  python visualization/plot_10d.py --run run_9dfe

  # headless render to PNG:
  python visualization/plot_10d.py --run run_9dfe --save conet.png

Because a run reconstructs a single Objective (no separate Bandgap /
Photoconductance / Stability responses), a single CoNet panel is drawn, coloured
by Objective. There is no per-point iteration in the reconstruction, so points
are ordered by acquisition (drives draw order + the latest-sample red rim); the
within-iteration smoothing and time-travel of the live viewer are dropped.

The 10-module hardware index -> real material map is ``DIM_LABELS``; a run's
active composition columns are its ``optimizing_dims`` (read from
live_plot_state.json, falling back to hw_config.json).

NOTE: the bulk of this file (UI theme + all CN_* constants + the CoNet builder /
dominance fields / drawing helpers) is copied VERBATIM from the DiSCO
``visualization.py`` so the CoNet renders identically. New, run-specific code
(data loading, single-panel layout, click-to-inspect tooltip, the interactive
viewer, and main) lives at the very bottom.
"""
from __future__ import annotations

import argparse
import itertools
import re  # noqa: F401  (used by the copied build_formula chemistry helpers)
import sys
import threading
import time
from pathlib import Path

import numpy as np


class _Progress:
    """Tiny threaded spinner so a long, blocking stage (S-matrix build, UMAP fit,
    dominance fields) shows it is *thinking*, not *stuck*. Prints a rotating glyph +
    label + elapsed seconds to stderr, updating ~5x/sec, and clears the line on exit.
    No-ops (stays silent) when stderr is not a TTY, so piped/redirected output and the
    headless PNG path stay clean. Use as a context manager:

        with _Progress("building UMAP embedding"):
            ...heavy work...
    """
    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label, stream=sys.stderr):
        self.label = label
        self.stream = stream
        self.enabled = bool(getattr(stream, "isatty", lambda: False)())
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        t0 = time.monotonic()
        for frame in itertools.cycle(self.FRAMES):
            if self._stop.is_set():
                break
            self.stream.write(f"\r  {frame} {self.label} … {time.monotonic() - t0:5.1f}s")
            self.stream.flush()
            self._stop.wait(0.2)

    def __enter__(self):
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        else:
            # non-TTY: still leave a one-line breadcrumb so logs show the stage started
            self.stream.write(f"  … {self.label}\n"); self.stream.flush()
        self._t0 = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            dt = time.monotonic() - self._t0
            mark = "✗" if exc_type else "✓"
            self.stream.write(f"\r  {mark} {self.label} … {dt:5.1f}s\n")
            self.stream.flush()
        return False

# project root on sys.path so `src` imports resolve
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.utils.datahandler import reconstruct_snapshot_tensors  # noqa: E402

# 10-module hardware index -> real material (labels the run's optimizing dims).
DIM_LABELS: dict[int, str] = {0: "FAPbI3", 9: "MAPbBr3", 8: "MAPbI3", 2: "CsPbI3"}

RUNS_DIR = _HERE.parent / "runs"
DEFAULT_RUN = "run_9dfe"

# ---- constants the copied CoNet code expects (were module-level in visualization.py) ----
VALUE_COLUMNS = ["Objective"]        # a run reconstructs a single objective response
RESP_LABEL = {"Bandgap": "Band Gap"}
CMAP_NAME = "viridis"
BANDGAP_CMAP = "viridis_r"
TARGET_EG_DEFAULT = 1.4              # only used by Bandgap contours (never drawn for Objective)


# ===========================================================================
# VERBATIM CoNet code from visualization.py (UI theme, CN_* config, builder,
# dominance fields, and all drawing helpers through draw_conet_legend).
# ===========================================================================


# ---- UI theme: minimal dark "lab console" ----------------------------------
# Everything below drives both the Tk chrome and the matplotlib figures so the
# whole window reads as one surface. Restore snapshots/visualization_*.py to
# get the classic light look back.
UI_BG        = "#0e1116"                # window + figure background (deep charcoal)
UI_PANEL     = "#151b24"                # popdown lists / separators
UI_FIELD     = "#1b2230"                # inputs, buttons, slider trough
UI_FIELD_HI  = "#273245"                # hover
UI_TEXT      = "#e8eaed"                # primary text
UI_MUTED     = "#98a0ad"                # secondary text
UI_FAINT     = "#5c6470"                # captions, coordinates, tertiary text
UI_ACCENT    = "#22d3ee"                # primary action (cyan)
UI_ACCENT_HI = "#67e8f9"
UI_RED       = "#ff4757"                # current-iteration edges + target-Eg contours
UI_EDGE_OLD  = (1.0, 1.0, 1.0, 0.35)    # hairline edges on prior-iteration points
UI_TRIANGLE  = (1.0, 1.0, 1.0, 0.30)    # ternary frame
UI_FONT      = "Segoe UI"
UI_WINDOW_ALPHA = 0.97                  # subtle whole-window translucency
UI_STATUS = {"green": "#34d399", "red": "#f87171", "blue": "#60a5fa"}
NODE_EDGE = "#aab3c0"                    # light rim so dark CoNet nodes read on the dark ground
EDGE_COL  = "#8b95a6"                    # CoNet composition links (neutral light on dark)
# brighter metric line colours that pop on the dark ground (one per response)
MLINE = {"Bandgap": "#c39bff", "Photoconductance": "#5ad1e6",
         "Stability": "#7fd88a", "Objective": "#ffb454"}

# Figure palettes: the live window renders on the dark theme; figures exported by
# the "Save Figures" button render on WHITE (for papers / slides).
FIG_DARK = dict(bg=UI_BG, text=UI_TEXT, muted=UI_MUTED, faint=UI_FAINT,
                triangle=UI_TRIANGLE, edge_old=UI_EDGE_OLD, red=UI_RED)
FIG_LIGHT = dict(bg="#ffffff", text="#1f2430", muted="#5c6470", faint="#98a0ad",
                 triangle=(0.0, 0.0, 0.0, 0.45), edge_old=(0.0, 0.0, 0.0, 0.35),
                 red="#e0304a")

# CoNet panel palettes: the live window draws DARK; "Save Figures" exports draw with the LIGHT palette
# so the images read cleanly on white (dark node rims, white label halos, darkened comp hues, stronger
# region alphas -- the on-dark alphas wash out on paper).
CN_PAL_DARK = dict(mode="dark", node_edge="#aab3c0", stroke=None, text="#e8eaed", muted="#98a0ad",
                   faint="#5c6470", red="#ff4757", a_pure=None, a_blend=None)
CN_PAL_LIGHT = dict(mode="light", node_edge="#20242c", stroke="#ffffff", text="#1f2430",
                    muted="#5c6470", faint="#98a0ad", red="#e0304a", a_pure=0.9, a_blend=0.45)

# ---- CoNet configuration (self-contained port of pub_conetwork) ------------
CN_FLOOR = 0.07              # a component is "active" in a sample above this fraction
CN_KNN = 5                  # edges: link each sample to its 5 most compositionally-similar neighbours
CN_BETA = 1.5
# min_dist default 0.99: maximal within/between-cluster spread (user preference), still exposed live as
# the UI "SPREAD" knob. n_neighbors 60 preserves the global composition relationships.
CN_UMAP_NN, CN_UMAP_MD, CN_UMAP_SPREAD = 60, 0.99, 1.0
CN_SEED = 42
CN_GRID = 240
CN_H_FACTOR = 6.0           # dominance-field bandwidth (larger -> regions cover more)
CN_DENS_THR = 0.012
CN_A_MAX = 0.30             # peak opacity of the blended composition background (dark theme)
CN_SHARP = 4.0             # power sharpening of the colour blend (larger -> more distinct hues)
CN_FILL_R = 0.125          # fill reaches this far (frac of view span) from any sample
CN_FILL_TAPER = 0.10       # soft outer fade of the fill
CN_FILL_BLUR = 5.0         # extra Gaussian blur (grid cells) on the fill alpha
CN_MIN_CELLS, CN_MIN_NODES = 30, 4
CN_PCT = (10.0, 90.0)      # CoNet colour bounds: 10th-90th percentile of the response
CN_STAR = "#ff2d2d"        # discovered-needle / incumbent (best Objective) marker (red)
# ---- two LIVE, user-tunable layout knobs (persist across new samples / timeline; see the UI) --------
# CLUSTER DIST (self.gap_reach): when the campaign repeatedly samples one composition (e.g. near-pure
# CsPbI3), that clique's UMAP neighbourhoods become self-contained and repulsion exiles it many main-mass
# radii away, leaving dead space. Radii beyond CN_GAP_KNEE_FRAC x r90 (r90 = 90th-pct radius of the main
# mass) are squashed (tanh) so the FARTHEST sample sits at ~gap_reach x r90. LOWER gap_reach pulls far
# clusters IN (less dead space); HIGHER lets them spread to the natural UMAP distance. Cheap to re-apply
# (no UMAP re-fit) so the UI can reshape the map live from the cached raw embedding.
CN_GAP_KNEE_FRAC = 2.0     # start squashing beyond this x the MEDIAN radius (leave the core untouched)
CN_GAP_REACH = 2.3         # farthest sample sits at ~this x the median radius (fixed; no longer a UI knob)
# ---- purity-driven layout (UI "PURITY") -------------------------------------------------------------
# Most samples are low-purity mixtures (median ~0.48), so plain plurality muddies the regions. With
# CN_PURITY_THR > 0: a sample whose dominant fraction is ABOVE the threshold gravitates to its cluster
# CORE; one BELOW it is flung radially OUTWARD, past the region boundary, so near-equal mixtures spread
# out and separate from true majority compositions. And a dashed ZONE is only drawn where the smoothed
# composition field exceeds CN_REGION_THR (not mere plurality). CN_PURITY_THR = 0 reverts to the old
# plurality layout + regions (the "revert" switch; PURITY spinbox = 0).
CN_PURITY_THR = 0.3
CN_PURITY_GAMMA = 1.2      # >1 pulls pure cores tighter and flings mixtures further
CN_PURITY_CLIP = (0.15, 2.3)   # radial factor bounds: small pure-core floor / mixture-fly-out cap
CN_REGION_THR = 0.45       # field.max ABOVE this in a cell -> a PURE core (solid outline); at/below -> a
                           # BLEND cell (dashed concentric outlines for its co-dominant compositions)
CN_BLEND_FLOOR = 0.22      # a composition counts as "co-existing" in a blend region if its mean field > this
CN_REGION_MIN_CELLS = 60   # draw a BOUNDARY around any region at least this big (so even small blends are
CN_REGION_MIN_NODES = 5    #   enclosed -> "every region has a boundary")
CN_LABEL_MIN_CELLS = 150   # ... but only LABEL regions at least this big, so the map isn't buried in tiny
CN_LABEL_MIN_NODES = 10    #   labels (small blends still read as blends by their dashed multi-colour ring)
CN_LABEL_MAX = 4           # cap on BLEND labels: label the biggest few UNIQUE blends (deduped by set)
CN_PURE_ALPHA = 0.75       # pure-region borders + labels: 25% transparent
CN_BLEND_ALPHA = 0.15      # blend-region borders + labels: 85% transparent (very subtle)
CN_LABEL_MAX_OVERLAP = 0.10  # a label box may cover at most this fraction of points/borders (else move)

# ---- ZoMBI-Hop needles (ranked optima proposed by the optimizer) ------------------------------------
# needles.db lives in the ZoMBI-Hop Dropbox; it is refreshed by the optimizer (possibly on another
# machine), so re-reading it only makes sense when the internet (-> Dropbox sync) is up. The viewer
# reads the local copy at startup, then re-checks it whenever new compositions arrive AND we are online.
NEEDLES_DB = (r"C:\Users\alexs\MIT Dropbox\Buonassisi-Group\Projects - Active\ZoMBI-Hop\06_Code"
              r"\ZoMBI-Hop\sql\needles.db")
NEEDLE_MIN_ALPHA = 0.40    # last-ranked needle star opacity (rank 0 = 1.0, linear in between)

# Red dashed target lines on the bottom line strips: Bandgap uses the campaign target_eg; the other
# responses aim at their ideal score of 1.0.
LINE_TARGETS = {"Photoconductance": 1.0, "Stability": 1.0, "Objective": 1.0}
# SPREAD (self.umap_md): UMAP min_dist -> how spread out the scatter points are within/between clusters.
# Higher = more spread. Changing it needs a full UMAP re-fit (a few seconds).  (UI "SPREAD")

# category colour scheme (mirrors the tsne/CoNet publication figures): 3D = blue gradient, 2D = yellow,
# additives = red. Colours are assigned over the FULL 10-component roster so a component's shade never
# depends on which others happen to be present.
CN_CATEGORY_ORDER = ["3D perovskite", "2D perovskite", "Additive"]
CN_LEGEND_NAMES = ["MAPbI3", "MAPbBr3", "FAPbI3", "FAPbBr3", "CsPbI3",
                   "(PEA)2PbI4", "BDAPbI4", "DMAI", "MACl", "CsI"]
CN_CATEGORY_OF = {"MAPbI3": "3D perovskite", "MAPbBr3": "3D perovskite", "FAPbI3": "3D perovskite",
                  "FAPbBr3": "3D perovskite", "CsPbI3": "3D perovskite",
                  "(PEA)2PbI4": "2D perovskite", "BDAPbI4": "2D perovskite",
                  "DMAI": "Additive", "MACl": "Additive", "CsI": "Additive"}
CN_PRETTY = {"MAPbI3": r"MAPbI$_3$", "MAPbBr3": r"MAPbBr$_3$", "FAPbI3": r"FAPbI$_3$",
             "FAPbBr3": r"FAPbBr$_3$", "CsPbI3": r"CsPbI$_3$",
             "(PEA)2PbI4": r"(PEA)$_2$PbI$_4$", "BDAPbI4": r"BDAPbI$_4$",
             "DMAI": "DMAI", "MACl": "MACl", "CsI": "CsI"}


def _apply_mpl_theme():
    """Point matplotlib at the dark UI theme (idempotent; called before any figure)."""
    import matplotlib as mpl
    from matplotlib import font_manager
    # Only keep the primary UI font if it's actually installed. On headless HPC
    # nodes Segoe UI (a Windows font) is absent, and matplotlib otherwise emits a
    # "findfont: Font family 'Segoe UI' not found." warning for every text
    # element, flooding stderr. Drop it there and fall straight to DejaVu Sans.
    installed = {f.name for f in font_manager.fontManager.ttflist}
    font_family = ([UI_FONT] if UI_FONT in installed else []) + ["DejaVu Sans"]
    mpl.rcParams.update({
        "font.family": font_family,
        "text.color": UI_TEXT,
        "axes.edgecolor": UI_FAINT,
        "axes.labelcolor": UI_TEXT,
        "xtick.color": UI_MUTED,
        "ytick.color": UI_MUTED,
        "figure.facecolor": UI_BG,
        "savefig.facecolor": UI_BG,
        "axes.facecolor": "none",
    })


# Matern-GP interpolation for the ternary background landscape (tuned to spread points while keeping
# sharp high-value features). The length scale is fixed (not optimized) and the whole grid is predicted
# in ONE batched call, so it recomputes fast on every update.
GP_LENGTH_SCALE = 1.0
GP_NU = 0.1            # Matern smoothness; low keeps sharp high-value points
GP_N_RESTARTS = 0     # 0 since the kernel is fixed (no hyperparameter optimization)
_H = np.sqrt(3.0) / 2.0  # height of a unit-base equilateral triangle


# ---------------------------------------------------------------------------
# small helpers (colours, smoothing)
# ---------------------------------------------------------------------------
def _bright(rgb, target=0.62):
    """Lift a colour toward white until it reads on the dark ground (keeps hue)."""
    import matplotlib as mpl
    r, g, b = mpl.colors.to_rgb(rgb)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if lum >= target:
        return (r, g, b)
    t = (target - lum) / (1 - lum)
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)


def _dark_comp_color(rgb):
    """Dark-mode composition hue for the blended background + region borders/labels.

    _bright() lifts every shade to the SAME luminance, which collapses a single-category ramp (all
    four real components are 3D perovskites on one blue ramp) into a flat grey wash. Instead we keep
    the hue, BOOST saturation, and lift value while PRESERVING the ramp's ordering (dark comps stay a
    little darker) -- so the composition gradient stays legible AND vivid on the dark ground."""
    import colorsys
    import matplotlib as mpl
    r, g, b = mpl.colors.to_rgb(rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    s2 = min(1.0, s * 1.1 + 0.18)          # vivid, but low-saturation light ends stay soft
    v2 = 0.58 + 0.40 * v                   # bright enough to read on dark; keeps dark<-->light order
    return colorsys.hsv_to_rgb(h, s2, v2)


# Distinct base hue PER component. The old scheme stepped each category along a single ramp, so
# adjacent 3D perovskites (all on one blue ramp) were nearly indistinguishable (e.g. MAPbI3 vs MAPbBr3).
# Here every component gets its own hue — _dark_comp_color preserves hue, so they stay distinct on the
# dark ground. Category temperature is kept loosely (3D = cool blue/teal/violet/cyan/steel, 2D = gold,
# additives = warm red/orange/pink). Fixed over the FULL roster so a colour never depends on which
# others are present.
CN_COMP_BASE = {
    "MAPbI3":  "#2f6fe0",    # blue
    "MAPbBr3": "#16b8a6",    # teal
    "FAPbI3":  "#8b5cf6",    # violet
    "FAPbBr3": "#38bdf8",    # sky / cyan
    "CsPbI3":  "#a7b4c9",    # pale steel (light, low-saturation -> clearly apart from the vivid blues)
    "(PEA)2PbI4": "#c98a1a",  # amber
    "BDAPbI4":    "#f2d94a",  # yellow
    "DMAI": "#d1352b",        # red
    "MACl": "#f57f3a",        # orange
    "CsI":  "#e94f9a",        # pink
}


def _cn_comp_colors(names):
    """component -> distinct base RGB (see CN_COMP_BASE); unknown names fall back to neutral slate.
    Fixed over the full roster so a component's colour never depends on which others are present."""
    import matplotlib as mpl
    return {nm: mpl.colors.to_rgb(CN_COMP_BASE.get(nm, "#8a94a6")) for nm in names}


def _assign_colors_by_space(names, E, dom):
    """Reassign the ACTIVE components' palette (same colours, different owners) so that colour
    similarity mirrors EMBEDDING proximity: components whose regions sit close together get the most
    similar colours (e.g. violet next to lavender) and far-apart components get the most different
    ones (e.g. teal vs violet). Brute-force permutation minimising the mismatch between the normalized
    spatial-distance and colour-distance matrices (n! is tiny for the <=8 active components; falls back
    to the fixed roster colours above that)."""
    import itertools
    import matplotlib as mpl
    n = len(names)
    base = [np.array(mpl.colors.to_rgb(CN_COMP_BASE.get(nm, "#8a94a6"))) for nm in names]
    if n < 3 or n > 8:
        return {nm: tuple(c) for nm, c in zip(names, base)}
    cents = np.array([E[dom == c].mean(0) if np.any(dom == c) else E.mean(0) for c in range(n)])
    Ds = np.linalg.norm(cents[:, None, :] - cents[None, :, :], axis=2)
    Ds /= (Ds.max() + 1e-12)
    disp = np.array([_dark_comp_color(tuple(c)) for c in base])   # distance in DISPLAYED colours
    Dc = np.linalg.norm(disp[:, None, :] - disp[None, :, :], axis=2)
    Dc /= (Dc.max() + 1e-12)
    iu, ju = np.triu_indices(n, 1)
    best, best_cost = tuple(range(n)), np.inf
    for perm in itertools.permutations(range(n)):
        p = np.array(perm)
        cost = float(np.abs(Ds[iu, ju] - Dc[p[iu], p[ju]]).sum())
        if cost < best_cost:
            best_cost, best = cost, perm
    return {names[i]: tuple(base[best[i]]) for i in range(n)}


_CN_ALL_COLORS = None                               # full-roster colours (built lazily; see below)


def _cn_all_colors():
    """Memoized full-roster component->colour map. Deferred (not computed at module scope) so that
    merely `import visualization` never pulls in matplotlib — main.py imports this module at top level
    in the orchestration process even when plotting is disabled."""
    global _CN_ALL_COLORS
    if _CN_ALL_COLORS is None:
        _CN_ALL_COLORS = _cn_comp_colors(CN_LEGEND_NAMES)
    return _CN_ALL_COLORS


def _cn_node_size(sN):
    return 14 + 105 * np.asarray(sN, float) ** 0.8


def _cn_view_limits(E, pad=0.05):
    """View limits that ENCLOSE EVERY point (full min/max + pad) so nothing ever clips, at the FIXED panel
    aspect CONET_FRAME_ASPECT (never data-driven, so the chart shape can't 'randomly' change). The full
    bounding box is padded, then whichever axis is SHORT is expanded until the box aspect == the fixed
    frame aspect. Result: all points inside, isotropic (never stretched), only extra margin (dead space)
    added on the shorter axis -- never a crop. Dominance-region reach keys off core_span, not this span,
    so widening here does NOT balloon the regions."""
    lo = E.min(0); hi = E.max(0)                    # FULL extent -> no point can fall outside
    m = pad * np.maximum(hi - lo, 1e-6)
    x0, x1 = lo[0] - m[0], hi[0] + m[0]
    y0, y1 = lo[1] - m[1], hi[1] + m[1]
    if x1 - x0 < 1e-6:
        x0 -= 0.5; x1 += 0.5
    if y1 - y0 < 1e-6:
        y0 -= 0.5; y1 += 0.5
    target = CONET_FRAME_ASPECT
    xs, ys = x1 - x0, y1 - y0
    if xs / ys < target:                            # too tall -> widen x (keeps every x-extreme inside)
        e = (target * ys - xs) / 2.0; x0 -= e; x1 += e
    else:                                           # too wide -> heighten y
        e = (xs / target - ys) / 2.0; y0 -= e; y1 += e
    return (x0, x1), (y0, y1)


def _cn_smoothstep(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def _cmap_for(value_name):
    """viridis_r for Bandgap; viridis for the other values (with transparent 'bad' for NaNs)."""
    import matplotlib
    cmap = matplotlib.colormaps[BANDGAP_CMAP if value_name == "Bandgap" else CMAP_NAME].copy()
    cmap.set_bad(alpha=0.0)
    return cmap


def smooth_seq(vals, iters, frac):
    """Within-iteration Gaussian smoothing of a 1-D value array (originally-NaN stay NaN)."""
    from scipy.ndimage import gaussian_filter1d
    out = np.asarray(vals, float).copy()
    if frac <= 0:
        return out
    for i in np.unique(iters):
        m = np.where(iters == i)[0]
        seg = out[m]; fin = np.isfinite(seg)
        if fin.sum() < 2:
            continue
        xs = np.arange(len(seg)); filled = np.interp(xs, xs[fin], seg[fin])
        sigma = min(frac / 100.0 * len(seg), float(len(seg)))   # cap: smoothing past the segment is moot
        sm = gaussian_filter1d(filled, sigma=sigma, mode="nearest")
        out[m] = np.where(fin, sm, np.nan)
    return out


# ---------------------------------------------------------------------------
# computed composition formula (ported from XRD_Database_Generator.py -- that module force-switches the
# matplotlib backend and imports pandas/openpyxl at import time, so the live viewer keeps its own copy
# of the small pure-string chemistry functions instead of importing it)
# ---------------------------------------------------------------------------
_F_A_ORDER = ["Cs", "FA", "MA", "DMA"]     # 3D A-site display order
_F_SP_ORDER = ["PEA", "BDA"]               # 2D spacer display order
_F_X_ORDER = ["I", "Br", "Cl"]             # halide display order


def _parse_component(name):
    """Classify a component column name -> dict(name, kind, A, X, A_count, X_count)."""
    name = str(name)
    if "Pb" not in name:                          # salt: MACl, CsI, DMAI, ...
        return dict(name=name, kind="additive")
    left, right = name.split("Pb", 1)
    mx = re.match(r"([A-Za-z]+)(\d+)$", right)
    if not mx:
        return dict(name=name, kind="additive")
    Xsym, Xn = mx.group(1), int(mx.group(2))
    kind = "3D" if Xn == 3 else "2D"
    ma = re.match(r"\(?([A-Za-z]+)\)?(\d*)$", left)
    Asym = ma.group(1) if ma else left
    Acount = int(ma.group(2)) if (ma and ma.group(2)) else 1
    return dict(name=name, kind=kind, A=Asym, X=Xsym, A_count=Acount, X_count=Xn)


def _f_num(v):
    r = round(float(v), 2)
    return "" if abs(r - 1.0) < 5e-3 else f"{r:.2f}"


def _f_frac(v):
    return f"{round(float(v), 2):.2f}"


def _f_count(c):
    r = round(float(c), 2)
    if abs(r - 1.0) < 5e-3:
        return ""
    if abs(r - round(r)) < 5e-3:
        return str(int(round(r)))
    return f"{r:.2f}"


def _f_xsite(weights, n):
    present = [(s, weights[s]) for s in _F_X_ORDER if round(weights.get(s, 0.0), 2) > 0]
    if len(present) == 1:
        return f"{present[0][0]}{n}"
    return "(" + "".join(f"{s}{_f_frac(w)}" for s, w in present) + f"){n}"


def build_formula(comp):
    """comp: {column_name: fraction} -> the nested composition string, e.g.
    {[Cs0.17FA0.60MA0.23Pb(I0.67Br0.33)3]0.71 + [(PEA)2PbI4]0.29}0.85 + {DMAI}0.15"""
    parsed = [(_parse_component(k), float(v)) for k, v in comp.items()
              if v is not None and np.isfinite(float(v)) and float(v) > 1e-9]
    p3d = [(p, w) for p, w in parsed if p["kind"] == "3D"]
    p2d = [(p, w) for p, w in parsed if p["kind"] == "2D"]
    adds = [(p, w) for p, w in parsed if p["kind"] == "additive"]
    S3 = sum(w for _, w in p3d)
    S2 = sum(w for _, w in p2d)
    Gper = S3 + S2

    def three_d():
        cats, ans = {}, {}
        for p, w in p3d:
            cats[p["A"]] = cats.get(p["A"], 0.0) + w / S3
            ans[p["X"]] = ans.get(p["X"], 0.0) + w / S3
        a = "".join(f"{s}{_f_num(cats.get(s, 0.0))}" for s in _F_A_ORDER
                    if round(cats.get(s, 0.0), 2) > 0)
        return f"{a}Pb{_f_xsite(ans, 3)}"

    def two_d():
        cnt, ans = {}, {}
        for p, w in p2d:
            cnt[p["A"]] = cnt.get(p["A"], 0.0) + p["A_count"] * w / S2
            ans[p["X"]] = ans.get(p["X"], 0.0) + w / S2
        sp = "".join(f"({s}){_f_count(cnt.get(s, 0.0))}" for s in _F_SP_ORDER
                     if round(cnt.get(s, 0.0), 2) > 0)
        return f"{sp}Pb{_f_xsite(ans, 4)}"

    if S3 > 0 and S2 > 0:
        perov = f"[{three_d()}]{_f_frac(S3 / Gper)} + [{two_d()}]{_f_frac(S2 / Gper)}"
    elif S3 > 0:
        perov = three_d()
    elif S2 > 0:
        perov = two_d()
    else:
        perov = ""
    if not adds:
        return perov or "(no perovskite)"
    add_str = " + ".join(f"{{{p['name']}}}{_f_frac(w)}" for p, w in adds)
    if not perov:
        return add_str
    return f"{{{perov}}}{_f_frac(Gper)} + {add_str}"


# ---------------------------------------------------------------------------
# CoNet structure (composition-only -> cached; rebuilt on new data)
# ---------------------------------------------------------------------------
class CoNetUnavailable(Exception):
    """Raised when the CoNet cannot be built (missing umap-learn / networkx, or too little data)."""


def _apply_gap_layout(E_raw, gap_reach, knee_frac=CN_GAP_KNEE_FRAC):
    """Reshape a raw UMAP embedding: squash radii beyond knee_frac*r50 toward gap_reach*r50 (smooth tanh
    rolloff), then PCA-orient (major axis -> x, deterministic sign). Pure post-process on the embedding —
    NO UMAP re-fit — so the UI 'CLUSTER DIST' knob can reshape the map live. The scale is the MEDIAN
    radius r50 of the main mass (robust: NOT inflated by the satellite population the way r90 is), so
    gap_reach is in intuitive 'core-radius' units: smaller pulls far satellites (e.g. resampled CsPbI3)
    in tight, larger lets them spread out to the natural UMAP distance."""
    E = np.asarray(E_raw, float).copy()
    ctr = np.median(E, axis=0)
    d = E - ctr
    r = np.hypot(d[:, 0], d[:, 1])
    scale = float(np.median(r)) or float(np.percentile(r, 90)) or 1.0   # robust core radius
    knee = knee_frac * scale
    cap = max(float(gap_reach), knee_frac + 0.1) * scale      # cap >= knee so there is room to squash into
    far = r > knee
    if scale > 0 and far.any() and cap > knee:
        span = cap - knee
        rn = knee + span * np.tanh((r[far] - knee) / span)     # excess -> saturates at (cap - knee)
        E[far] = ctr + d[far] * (rn / r[far])[:, None]
    _, R = np.linalg.eigh(np.cov(E.T))                         # canonical orientation (major axis -> x)
    E = E @ R[:, ::-1]
    for k in (0, 1):
        if np.sum(E[:, k] ** 3) < 0:
            E[:, k] = -E[:, k]
    return E - E.mean(0)


def _apply_purity_layout(E, comp, thr=CN_PURITY_THR, gamma=CN_PURITY_GAMMA, clip=CN_PURITY_CLIP):
    """Warp each node's radius about its component's pure-core anchor by composition PURITY: samples
    purer than `thr` gravitate to the core (radial factor -> small), mixtures fly outward past the
    region boundary (factor > 1) so near-equal mixtures spread out and separate from true majority
    compositions. Cheap post-process; thr <= 0 returns E unchanged (revert to the plurality layout)."""
    if thr is None or thr <= 0:
        return np.asarray(E, float)
    E = np.asarray(E, float).copy()
    dom = comp.argmax(1); purity = comp.max(1)
    A = np.zeros((comp.shape[1], 2))
    for c in range(comp.shape[1]):
        sel = dom == c
        if not sel.any():
            continue
        w = np.clip(purity[sel], 1e-3, None) ** 2          # weight toward the PURE members -> core anchor
        A[c] = np.average(E[sel], axis=0, weights=w)
    v = E - A[dom]
    factor = ((1.0 - purity) / max(1.0 - thr, 1e-6)) ** gamma   # p=thr -> 1; p=1 -> 0; p<thr -> >1
    factor = np.clip(factor, clip[0], clip[1])
    return A[dom] + v * factor[:, None]


def build_conet_structure(comp, comp_names, resp, iters, resp_names,
                          min_dist=CN_UMAP_MD, gap_reach=CN_GAP_REACH,
                          purity_thr=CN_PURITY_THR):
    """Build the co-occurrence-network structure from composition only.

    comp        (N, C) row-normalized composition fractions of the ACTIVE components
    comp_names  list of C component names (used only for colours/labels)
    resp        (N, R) response values (may contain NaN); coloured per panel
    Returns an M dict consumed by the CoNet drawing helpers. Raises CoNetUnavailable if the optional
    heavy dependencies are absent or there is too little data to embed."""
    try:
        import umap
        import networkx as nx
        from scipy.spatial import cKDTree  # noqa: F401  (import-check; used in dominance fields)
    except Exception as e:
        raise CoNetUnavailable(f"CoNet needs umap-learn + networkx: {e}")
    N = len(comp)
    if N < 5 or comp.shape[1] < 2:
        raise CoNetUnavailable("not enough data / components for a CoNet yet")
    dom = comp.argmax(1)

    active = comp > CN_FLOOR
    with _Progress(f"co-occurrence matrix ({N}×{N})"):
        # Zero out inactive fractions up front: min(mc_i, mc_j) is then already 0 whenever either
        # sample is inactive in component c, so the explicit active-outer + where() masking (which
        # allocated two extra N×N temporaries per component) is redundant. Bit-identical result.
        mc = np.where(active, comp, 0.0)
        S = np.zeros((N, N))
        for c in range(comp.shape[1]):
            S += np.minimum.outer(mc[:, c], mc[:, c])
        np.fill_diagonal(S, 0.0)

    A = (S / (S.max() + 1e-12)) ** CN_BETA
    Dmat = 1.0 - A; np.fill_diagonal(Dmat, 0.0)
    nn = min(CN_UMAP_NN, max(2, N - 1))
    import warnings
    with warnings.catch_warnings():
        # expected umap-learn chatter, silenced narrowly: we pass a precomputed distance matrix on
        # purpose (inverse_transform is never used), and n_jobs=1 matches what seeding forces anyway.
        warnings.filterwarnings("ignore", message="using precomputed metric.*")
        warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1")
        with _Progress(f"UMAP embedding ({N} nodes)"):
            E = umap.UMAP(n_neighbors=nn, min_dist=min_dist, spread=CN_UMAP_SPREAD,
                          random_state=CN_SEED, metric="precomputed", n_jobs=1).fit_transform(Dmat)
    # Cache the RAW embedding + the gap-squashed/PCA-oriented layout so the PURITY knob can re-warp the
    # map without a UMAP re-fit; then apply the purity warp for the current threshold.
    E_raw = E - E.mean(0)
    E_gap = _apply_gap_layout(E_raw, gap_reach)
    E = _apply_purity_layout(E_gap, comp, purity_thr)
    # colours assigned from the FINAL layout: nearby components get similar hues, far ones dissimilar
    ccolor = _assign_colors_by_space(list(comp_names), E, dom)

    edges = {}
    with _Progress(f"k-NN graph edges ({N} nodes)"):
        k = min(CN_KNN, N - 1)
        for i in range(N):
            row = S[i]
            # argpartition finds the top-k neighbours in O(N) without fully sorting the row; the
            # edges dict is keyed by pair with weight S[i,j], so the order within the top-k is
            # irrelevant and the resulting graph is identical to the argsort version.
            for j in np.argpartition(-row, k)[:k]:
                if row[j] <= 0:
                    continue
                a, b = (i, j) if i < j else (j, i)
                edges[(a, b)] = row[j]
    if edges:
        ei = np.array([k[0] for k in edges]); ej = np.array([k[1] for k in edges])
        ew = np.array(list(edges.values()))
    else:
        ei = np.array([], int); ej = np.array([], int); ew = np.array([], float)

    strength = np.zeros(N)
    if len(ei):
        np.add.at(strength, ei, ew); np.add.at(strength, ej, ew)
    sN = (strength - strength.min()) / (np.ptp(strength) + 1e-9)

    oidx = resp_names.index("Objective") if "Objective" in resp_names else 0
    ocol = resp[:, oidx]
    best_obj = int(np.nanargmax(ocol)) if np.isfinite(ocol).any() else 0
    return dict(comp=comp, names=list(comp_names), N=N, E=E, E_raw=E_raw, E_gap=E_gap, dom=dom,
                ccolor=ccolor, ei=ei, ej=ej, ew=ew, strengthN=sN, best_obj=best_obj,
                purity_thr=purity_thr, resp=resp.copy(), resp_names=list(resp_names))


def conet_dominance_fields(M):
    """Continuous per-component composition field over the view (for the blended background + zones)."""
    from scipy.spatial.distance import cdist
    from scipy.spatial import cKDTree, Delaunay
    E, comp = M["E"], M["comp"]
    (x0, x1), (y0, y1) = _cn_view_limits(E)
    # core scale = 10th..90th-percentile extent, which EXCLUDES a far satellite (e.g. a resampled
    # CsPbI3 clique). The composition fill/zone REACH keys off this instead of the full view span, so a
    # distant cluster can't balloon the dominance regions across the whole (mostly empty) frame.
    clo, chi = np.percentile(E, [10.0, 90.0], axis=0)
    core_span = float(max(chi[0] - clo[0], chi[1] - clo[1], 1e-9))
    gx, gy = np.meshgrid(np.linspace(x0, x1, CN_GRID), np.linspace(y0, y1, CN_GRID))
    G = np.column_stack([gx.ravel(), gy.ravel()])
    h = np.median(cKDTree(E).query(E, k=2)[0][:, 1]) * CN_H_FACTOR
    field = np.zeros((G.shape[0], comp.shape[1])); dens = np.zeros(G.shape[0])
    for s in range(0, G.shape[0], 6000):
        W = np.exp(-cdist(G[s:s + 6000], E, "sqeuclidean") / (2 * h * h))
        field[s:s + 6000] = (W @ comp) / (W.sum(1)[:, None] + 1e-12)
        dens[s:s + 6000] = W.sum(1)
    dens /= dens.max()
    dnn = cKDTree(E).query(G)[0]
    argf = field.argmax(1)
    sf = np.sort(field, 1); margin = sf[:, -1] - sf[:, -2]
    sh = (CN_GRID, CN_GRID)
    return dict(gx=gx, gy=gy, ext=[x0, x1, y0, y1], x0=x0, x1=x1, y0=y0, y1=y1,
                core_span=core_span,
                field=field.reshape(sh + (comp.shape[1],)), argf=argf.reshape(sh),
                margin=margin.reshape(sh), dens=dens.reshape(sh), dnn=dnn.reshape(sh),
                mask=(dens > CN_DENS_THR).reshape(sh))


def _cn_place_labels(ax, cands, names, F, iy, ix, border_mask):
    """Place every region label in EMPTY plot space: scan the dominance grid for the spot closest to the
    region whose label BOX covers < CN_LABEL_MAX_OVERLAP of occupied cells (scatter points, any region
    border, previously placed labels). Labels may end up far from their region -- a thin leader line
    (drawn by the caller) ties them back. Stores lx/ly (label centre) and hw/hh (box half-size, data
    units) on each cand."""
    from scipy.ndimage import binary_dilation
    G = CN_GRID
    MX, MXR = 26, 2   # spill allowance OUTSIDE the panel: generous on the LEFT (figure margin / column
    #                   gap), nearly none on the RIGHT -- the colourbar sits immediately right of every
    #                   panel and labels must never print on top of it
    W = MX + G + MXR                                     # padded occupancy width (x only; never vertical)
    xspan = F["x1"] - F["x0"]; yspan = F["y1"] - F["y0"]
    cellw, cellh = xspan / G, yspan / G
    # true label-box size: points -> data units via the panel's on-screen size
    fig = ax.figure; pos = ax.get_position(); fw, fh = fig.get_size_inches()
    ppx = xspan / max(pos.width * fw * 72.0, 1e-9)      # data units per typographic point (x)
    ppy = yspan / max(pos.height * fh * 72.0, 1e-9)
    occ0 = np.zeros((G, G), bool)
    occ0[iy, ix] = True                                  # scatter points ...
    occ0 = binary_dilation(occ0, iterations=3)           # ... with their marker radius + breathing room
    occ0 |= binary_dilation(border_mask, iterations=3)   # borders, widened so labels sit clear of them
    occ = np.zeros((G, W), bool)                         # x-padded: margins count as EMPTY space
    occ[:, MX:MX + G] = occ0
    lead_mask = np.zeros_like(occ)                       # placed-leader corridors: boxes may NEVER touch
    box_mask = np.zeros_like(occ)                        # placed label boxes: boxes may NEVER overlap
    cross_mask = binary_dilation(border_mask, iterations=1)   # for counting borders a leader crosses

    def crossings(rcy, rcx, yy, xx):
        """# of distinct border lines / label boxes / earlier leaders the straight leader path would
        cross (rising edges along a DENSE sample -- sparse sampling stepped clean over thin lines and
        undercounted, which is how leaders ended up slicing through regions)."""
        n = max(64, int(2.0 * np.hypot(yy - rcy, xx - rcx)))     # ~2 samples per cell of path length
        ys = np.clip(np.round(np.linspace(rcy, yy, n)).astype(int), 0, G - 1)
        xs = np.round(np.linspace(rcx, xx, n)).astype(int)
        inb = (xs >= 0) & (xs < G)
        hit = np.zeros(n, bool)
        hit[inb] = cross_mask[ys[inb], np.clip(xs[inb], 0, G - 1)]
        return int(np.sum(hit[1:] & ~hit[:-1]))

    def _ccw(ax_, ay_, bx_, by_, cx_, cy_):
        return (cy_ - ay_) * (bx_ - ax_) - (by_ - ay_) * (cx_ - ax_)

    placed_segs = []                                     # placed leaders, EXACT segments (cell coords)

    def leader_hits(x0, y0, x1, y1):
        """# of already-placed leader lines this leader would STRICTLY cross (exact segment test --
        raster approximations let X-crossings slip through)."""
        n = 0
        for (sx0, sy0, sx1, sy1) in placed_segs:
            d1 = _ccw(x0, y0, x1, y1, sx0, sy0); d2 = _ccw(x0, y0, x1, y1, sx1, sy1)
            d3 = _ccw(sx0, sy0, sx1, sy1, x0, y0); d4 = _ccw(sx0, sy0, sx1, sy1, x1, y1)
            if d1 * d2 < 0 and d3 * d4 < 0:
                n += 1
        return n

    # map centre (median of the nodes): every label prefers the spot OUTWARD from its region along the
    # centre->region ray, so leaders form radiating spokes that naturally never cross each other
    Cy, Cx = float(np.median(iy)), float(np.median(ix))
    # ---- per-label geometry (precomputed so a label can be RE-placed during the repulsion pass) ----
    for r in cands:
        fs = 27.0                                         # match _draw_blend_label: 3x-scaled labels
        w_pt = max(len(str(names[c])) for c in r["comps"]) * 0.60 * fs + 40   # text + brackets
        h_pt = len(r["comps"]) * fs * 1.5 + 12
        r["hw"], r["hh"] = 0.5 * w_pt * ppx, 0.5 * h_pt * ppy
        # test a PADDED box (+2 cells) so labels keep a margin from points/borders/each other
        r["_cw"] = max(int(np.ceil(r["hw"] / cellw)), 1) + 2
        r["_ch"] = max(int(np.ceil(r["hh"] / cellh)), 1) + 2
        r["_rcx"] = (r["cx"] - F["x0"]) / cellw          # anchor (density peak), unpadded cells
        r["_rcy"] = (r["cy"] - F["y0"]) / cellh
        r["_spot"] = None; r["_cost"] = 0.0
        r["lx"], r["ly"] = r["cx"], r["cy"]              # safe defaults
    static_occ = occ.copy()                              # nodes + borders only (labels re-added on top)
    static_cross = cross_mask.copy()

    def add_contrib(q):
        """Stamp label q's box + leader into every obstacle structure."""
        yy, xxp = q["_spot"]; cw, ch = q["_cw"], q["_ch"]
        occ[max(yy - ch - 1, 0):yy + ch + 2, max(xxp - cw - 1, 0):xxp + cw + 2] = True
        box_mask[max(yy - ch - 1, 0):yy + ch + 2, max(xxp - cw - 1, 0):xxp + cw + 2] = True
        gx0, gx1 = max(xxp - cw - 1 - MX, 0), min(xxp + cw + 2 - MX, G)
        if gx1 > gx0:
            cross_mask[max(yy - ch - 1, 0):min(yy + ch + 2, G), gx0:gx1] = True
        rcx, rcy = q["_rcx"], q["_rcy"]
        placed_segs.append((rcx, rcy, float(xxp - MX), float(yy)))
        lys = np.clip(np.round(np.linspace(rcy, yy, 64)).astype(int), 0, G - 1)
        lxs = np.round(np.linspace(rcx, xxp - MX, 64)).astype(int)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy_d = np.clip(lys + dy, 0, G - 1); xx_d = lxs + dx
                ok_s = (xx_d >= 0) & (xx_d < G)
                occ[yy_d[ok_s], xx_d[ok_s] + MX] = True
                lead_mask[yy_d[ok_s], xx_d[ok_s] + MX] = True

    def rebuild(exclude=()):
        """Reset masks to the static base and re-stamp every placed label except `exclude`."""
        occ[:] = static_occ; cross_mask[:] = static_cross
        box_mask[:] = False; lead_mask[:] = False
        del placed_segs[:]
        for q in cands:
            # identity test (`in` would ==-compare dicts holding numpy arrays -> ambiguous truth)
            if q["_spot"] is not None and all(q is not e for e in exclude):
                add_contrib(q)

    def place(r, forced_spot=None):
        """(Re)place label r against the CURRENT masks; commits the spot + contributions and returns
        the cost (None if forced_spot is infeasible). The search = NEAR band + RADIAL spokes."""
        cw, ch, rcx, rcy = r["_cw"], r["_ch"], r["_rcx"], r["_rcy"]
        sat = np.pad(occ, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        satL = np.pad(lead_mask, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        satB = np.pad(box_mask, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

        def boxfrac(yy, xxp, allow_leads=False):         # xxp in PADDED coords; hard vetoes: corridors+boxes
            if not (ch + 1 <= yy < G - ch - 1 and cw + 1 <= xxp < W - cw - 1):
                return None
            y0, y1 = yy - ch, yy + ch + 1; x0, x1 = xxp - cw, xxp + cw + 1
            if not allow_leads:
                if satL[y1, x1] - satL[y0, x1] - satL[y1, x0] + satL[y0, x0] > 0:
                    return None
                if satB[y1, x1] - satB[y0, x1] - satB[y1, x0] + satB[y0, x0] > 0:
                    return None
            tot = sat[y1, x1] - sat[y0, x1] - sat[y1, x0] + sat[y0, x0]
            return tot / ((y1 - y0) * (x1 - x0))

        def spot_cost(kind, fr_, dist, yy, xxp):
            ll = leader_hits(rcx, rcy, xxp - MX, yy)
            cr = crossings(rcy, rcx, yy, xxp - MX)
            return 600.0 * ll + 80.0 * cr + 150.0 * fr_ + (1.0 if kind == "near" else 0.55) * dist

        if forced_spot is not None:                      # repulsion: r takes another label's spot
            yy, xxp = forced_spot
            fr_ = boxfrac(yy, xxp)
            if fr_ is None or fr_ >= CN_LABEL_MAX_OVERLAP:
                return None
            cost = spot_cost("near", fr_, np.hypot(yy - rcy, xxp - MX - rcx), yy, xxp)
        else:
            cand_spots = []
            for yy in range(max(ch + 2, int(rcy) - 48), min(G - ch - 2, int(rcy) + 49), 3):
                for xxp in range(max(cw + 1, int(rcx) + MX - 48), min(W - cw - 1, int(rcx) + MX + 49), 3):
                    fr_ = boxfrac(yy, xxp)
                    if fr_ is not None and fr_ < CN_LABEL_MAX_OVERLAP:
                        cand_spots.append(("near", fr_, np.hypot(yy - rcy, xxp - MX - rcx), yy, xxp))
            rad = np.hypot(rcy - Cy, rcx - Cx)
            if rad > 12:
                base = np.arctan2(rcy - Cy, rcx - Cx)
                angles = [base + d for d in (0.0, 0.18, -0.18, 0.38, -0.38, 0.60, -0.60,
                                             0.85, -0.85, 1.15, -1.15, 1.45, -1.45)]
            else:
                angles = list(np.linspace(0, 2 * np.pi, 12, endpoint=False))
            for ang in angles:
                ca_, sa_ = np.cos(ang), np.sin(ang)
                for t in range(5, 3 * G, 3):
                    yy = int(round(rcy + sa_ * t)); xxp = int(round(rcx + ca_ * t)) + MX
                    if not (0 <= yy < G and 0 <= xxp < W):
                        break
                    fr_ = boxfrac(yy, xxp)
                    if fr_ is not None and fr_ < CN_LABEL_MAX_OVERLAP:
                        cand_spots.append(("radial", fr_, float(t), yy, xxp))
                        break
            if cand_spots:
                near_sorted = sorted([c_ for c_ in cand_spots if c_[0] == "near"],
                                     key=lambda c_: (c_[1], c_[2]))
                pool = near_sorted[:90] + [c_ for c_ in cand_spots if c_[0] == "radial"]
                best_cost, best = None, None
                for kind, fr_, dist, yy, xxp in pool:
                    c_ = spot_cost(kind, fr_, dist, yy, xxp)
                    if best_cost is None or c_ < best_cost:
                        best_cost, best = c_, (yy, xxp)
                (yy, xxp), cost = best, best_cost
            else:                                        # nowhere clean -> least-covered spot
                fb, fb_key = None, None
                for yy in range(ch + 2, G - ch - 2, 4):
                    for xxp in range(cw + 1, W - cw - 1, 4):
                        fr_ = boxfrac(yy, xxp, allow_leads=True)
                        if fr_ is not None:
                            k_ = (fr_, np.hypot(yy - rcy, xxp - MX - rcx))
                            if fb_key is None or k_ < fb_key:
                                fb_key, fb = k_, (yy, xxp)
                if fb is None:
                    r["_spot"], r["_cost"] = None, 0.0
                    r["lx"], r["ly"] = r["cx"], r["cy"]
                    return 0.0
                yy, xxp = fb
                cost = 300.0 + np.hypot(yy - rcy, xxp - MX - rcx)
        r["lx"] = F["x0"] + (xxp - MX + 0.5) * cellw
        r["ly"] = F["y0"] + (yy + 0.5) * cellh
        r["_spot"], r["_cost"] = (yy, xxp), float(cost)
        add_contrib(r)
        return float(cost)

    # ---- pass 1: greedy, OUTER regions first (they claim the margin at their own angle) ----
    for r in sorted(cands, key=lambda q: -np.hypot(q["_rcy"] - Cy, q["_rcx"] - Cx)):
        place(r)
    # ---- pass 2: REPULSION -- a label stuck with an expensive spot (far away / crossing things) may
    # TAKE a well-placed neighbour's spot near its own region; the neighbour is then re-placed on the
    # now-updated map (pushed away), committing only when the TOTAL cost strictly improves ----
    for _sweep in range(2):
        changed = False
        for r in sorted(cands, key=lambda q: -q["_cost"]):
            if r["_spot"] is None or r["_cost"] <= 55.0:
                continue
            others = sorted((q for q in cands if q is not r and q["_spot"] is not None
                             and not q.get("_evicted")),
                            key=lambda q: np.hypot(q["_spot"][0] - r["_rcy"],
                                                   q["_spot"][1] - MX - r["_rcx"]))
            for b in others:
                d_rb = np.hypot(b["_spot"][0] - r["_rcy"], b["_spot"][1] - MX - r["_rcx"])
                if d_rb + 30.0 >= r["_cost"]:
                    break                                # even the nearest spot isn't worth stealing
                saved = [(q, q["_spot"], q["_cost"], q["lx"], q["ly"]) for q in (r, b)]
                old_total = r["_cost"] + b["_cost"]
                target = b["_spot"]
                rebuild(exclude=(r, b))
                cost_r = place(r, forced_spot=target)
                if cost_r is None:                       # r's box doesn't fit there -> restore + next b
                    rebuild()
                    continue
                cost_b = place(b)                        # b gets pushed to its best remaining spot
                if cost_r + (1e9 if cost_b is None else cost_b) + 15.0 < old_total:
                    b["_evicted"] = True                 # each label is pushed at most once
                    changed = True
                    break
                for q, sp, co, lx_, ly_ in saved:        # not an improvement -> full rollback
                    q["_spot"], q["_cost"], q["lx"], q["ly"] = sp, co, lx_, ly_
                rebuild()
        if not changed:
            break


# ---------------------------------------------------------------------------
# CoNet drawing (dark, re-themed from pub_conetwork)
# ---------------------------------------------------------------------------
def _draw_blend_label(ax, x, y, pretty_names, color, alpha=1.0, stroke_col=None):
    """Region label: composition name(s) STACKED VERTICALLY (tight) inside a tall [ ... ] bracket (both
    pure single names and blends get the bracket), all in `color` at `alpha`. Added as a FIGURE artist
    (clip off, high zorder) so it is never hidden under a colourbar even at the panel's right edge.
    stroke_col = halo colour behind the text (default: the dark UI background; exports pass white)."""
    import matplotlib as mpl
    import matplotlib.patheffects as pe
    from matplotlib.offsetbox import TextArea, VPacker, HPacker, DrawingArea, AnnotationBbox
    from matplotlib.lines import Line2D
    # WHITE text + brackets with a BLACK outline for maximum readability over the coloured map. The
    # region `color`/`stroke_col` args are kept in the signature (callers still pass them) but the label
    # itself no longer tints with the region hue -- only the leader line does.
    rgba = mpl.colors.to_rgba("white", alpha)
    stroke = [pe.withStroke(linewidth=4.0, foreground=mpl.colors.to_rgba("black", alpha))]
    fs = 27.0                                              # ~3x the old 9.0 pt: big, bold region labels
    segs = [TextArea(nm, textprops=dict(color=rgba, fontsize=fs, fontweight="bold", path_effects=stroke))
            for nm in pretty_names]
    col = VPacker(children=segs, align="center", pad=0, sep=3)
    h = len(pretty_names) * (fs * 1.42) + 6                 # approx stack height (points)
    bw, serif = 9.6, 9.0                                   # bracket stem-to-serif width / serif length (3x)

    def bracket(left):
        da = DrawingArea(bw, h, 0, 0)
        xs = ([serif, 0, 0, serif] if left else [0, serif, serif, 0])
        da.add_artist(Line2D(xs, [0, 0, h, h], color=rgba, lw=3.2, solid_capstyle="round",
                             solid_joinstyle="miter", path_effects=stroke))
        return da

    box = HPacker(children=[bracket(True), col, bracket(False)], align="center", pad=0, sep=5)
    ax.figure.add_artist(AnnotationBbox(box, (x, y), xycoords=ax.transData, frameon=False, pad=0,
                                        box_alignment=(0.5, 0.5), annotation_clip=False, zorder=30))


def _conet_underlay(ax, M, F, pal=None, labels=True):
    import matplotlib as mpl
    import matplotlib.patheffects as pe
    from scipy.ndimage import gaussian_filter, label as cc_label, binary_erosion
    pal = pal or CN_PAL_DARK
    names = M["names"]
    # REACH keys off the core scale (10-90 pct extent), NOT the full view span, so a far satellite
    # can't inflate the fill/zones across the empty frame (see conet_dominance_fields core_span).
    span = F.get("core_span", max(F["x1"] - F["x0"], F["y1"] - F["y0"]))
    reach, taper = CN_FILL_R * span, CN_FILL_TAPER * span
    if pal["mode"] == "dark":
        bcol = {nm: _dark_comp_color(M["ccolor"][nm]) for nm in names}
    else:                                                # light/export: darkened hues read on white
        bcol = {nm: tuple(np.clip(np.array(mpl.colors.to_rgb(M["ccolor"][nm])) * 0.78, 0, 1))
                for nm in names}
    cols = np.array([bcol[names[c]] for c in range(len(names))])
    w = np.clip(F["field"], 0, None) ** CN_SHARP
    w = w / (w.sum(2, keepdims=True) + 1e-12)
    rgb = np.tensordot(w, cols, axes=([2], [0]))
    alpha = CN_A_MAX * _cn_smoothstep((reach - F["dnn"]) / taper)
    alpha = gaussian_filter(alpha, sigma=CN_FILL_BLUR)
    ax.imshow(np.dstack([np.clip(rgb, 0, 1), alpha]), extent=F["ext"], origin="lower",
              interpolation="bilinear", aspect="auto", zorder=0)
    iy = np.clip(((M["E"][:, 1] - F["y0"]) / (F["y1"] - F["y0"]) * (CN_GRID - 1)).round().astype(int), 0, CN_GRID - 1)
    ix = np.clip(((M["E"][:, 0] - F["x0"]) / (F["x1"] - F["x0"]) * (CN_GRID - 1)).round().astype(int), 0, CN_GRID - 1)
    field, argf, gx, gy, margin = F["field"], F["argf"], F["gx"], F["gy"], F["margin"]
    cover = F["dnn"] < reach
    purity_on = bool(M.get("purity_thr", 0.0)) and M["purity_thr"] > 0

    # ADAPTIVE thresholds: a fresh campaign starts with an (almost) empty results.db and grows in ~24-
    # sample batches, so the region/label gates scale with N -- small early batches still form regions
    # and the compositional map tightens as understanding accumulates (full strictness by ~500 samples).
    N_pts = len(M["E"])
    scaleN = float(np.clip(N_pts / 500.0, 0.25, 1.0))
    min_cells_r = max(12, int(CN_REGION_MIN_CELLS * scaleN))
    min_nodes_r = max(3, int(CN_REGION_MIN_NODES * scaleN))
    min_cells_l = max(25, int(CN_LABEL_MIN_CELLS * scaleN))
    min_nodes_l = max(4, int(CN_LABEL_MIN_NODES * scaleN))
    min_cells_p = max(10, int(CN_MIN_CELLS * scaleN))    # plurality (PURITY=0) revert thresholds
    min_nodes_p = max(2, int(CN_MIN_NODES * scaleN))

    regions = []                                        # every bounded region (pure AND blend)
    if not purity_on:
        # REVERT: dashed plurality regions (the old look, when PURITY = 0)
        for c in range(len(names)):
            lab, nlab = cc_label((argf == c) & cover); node_cc = lab[iy, ix]
            for k in range(1, nlab + 1):
                cells = lab == k
                nn = int(np.sum((M["dom"] == c) & (node_cc == k)))
                if cells.sum() < min_cells_p or nn < min_nodes_p:
                    continue
                regions.append(dict(cells=cells, comps=[c], kind="pure", dashed=True,
                                    color=bcol[names[c]], size=int(cells.sum()), nodes=nn,
                                    labelled=True))
    else:
        # TILE the whole covered area: every cell is keyed by WHICH compositions co-exist there (field >
        # CN_BLEND_FLOOR; at least the argmax so the key is never empty). Each connected region of a key
        # gets a boundary -> nothing is left unbounded. |key|==1 -> PURE (solid); |key|>=2 -> BLEND
        # (dashed concentric, one colour/outline per co-existing composition).
        present = field > CN_BLEND_FLOOR
        bit = np.where(cover, (present * (1 << np.arange(len(names)))).sum(2), 0)
        need = cover & (bit == 0)                       # guard: a covered cell with nothing over the floor
        if need.any():
            bit = np.where(need, (1 << argf), bit)
        for km in np.unique(bit[cover]):
            if km == 0:
                continue
            comps_in = [c for c in range(len(names)) if int(km) & (1 << c)]
            lab, nlab = cc_label(bit == km); node_in = lab[iy, ix]
            for k in range(1, nlab + 1):
                cells = lab == k
                nc, nn = int(cells.sum()), int(np.sum(node_in == k))
                if nc < min_cells_r or nn < min_nodes_r:
                    continue
                labelled = nc >= min_cells_l and nn >= min_nodes_l
                if len(comps_in) == 1:
                    regions.append(dict(cells=cells, comps=comps_in, kind="pure", dashed=False,
                                        color=bcol[names[comps_in[0]]], size=nc, nodes=nn,
                                        labelled=labelled))
                else:
                    meanf = field[cells].mean(0)
                    comps = sorted(comps_in, key=lambda c: -meanf[c])
                    wsum = float(sum(meanf[c] for c in comps)) + 1e-9       # shared BLEND colour
                    brgb = np.sum([meanf[c] * np.asarray(bcol[names[c]]) for c in comps], axis=0) / wsum
                    regions.append(dict(cells=cells, comps=comps, kind="blend", dashed=True,
                                        color=_bright(tuple(np.clip(brgb, 0, 1))), size=nc, nodes=nn,
                                        labelled=True))
    if not regions:
        return
    # ---- borders (pure 15% transparent / blend 50%) + the combined border mask for label placement ----
    border_all = np.zeros((CN_GRID, CN_GRID), bool)
    for r in regions:
        cells = r["cells"]
        r["alpha"] = ((pal["a_pure"] or CN_PURE_ALPHA) if r["kind"] == "pure"
                      else (pal["a_blend"] or CN_BLEND_ALPHA))   # exports use stronger alphas on white
        ind = gaussian_filter(cells.astype(float), sigma=2.0)

        def _outline(cs, black_lw):
            """WHITE region border with a BLACK outline (readability over the coloured map)."""
            eff = [pe.withStroke(linewidth=black_lw, foreground="black")]
            try:
                cs.set_path_effects(eff)                  # mpl>=3.8: ContourSet is a Collection
            except AttributeError:                        # older mpl: per-LineCollection
                for c in cs.collections:
                    c.set_path_effects(eff)

        if r["kind"] == "pure":
            lw = 1.8 if r["dashed"] else 2.2
            cs = ax.contour(gx, gy, ind, [0.5], colors="white",
                            linewidths=lw, alpha=r["alpha"],
                            linestyles="dashed" if r["dashed"] else "solid", zorder=6)
            _outline(cs, lw + 2.2)
        else:
            for comp, lev in zip(r["comps"], np.linspace(0.5, 0.85, len(r["comps"]))):
                cs = ax.contour(gx, gy, ind, [lev], colors="white", linewidths=1.6,
                                alpha=r["alpha"], linestyles="dashed", zorder=6)
                _outline(cs, 3.8)
        r["border"] = cells ^ binary_erosion(cells)
        border_all |= r["border"]
        # leader ANCHOR = the CENTROID of the region's cells, verified to lie INSIDE the boundary: for a
        # crescent/irregular region the raw centroid can fall outside, so it is snapped to the nearest
        # in-region cell in that case.
        yy_r, xx_r = np.where(cells)
        cyc, cxc = float(yy_r.mean()), float(xx_r.mean())
        jy, jx = int(round(cyc)), int(round(cxc))
        if not (0 <= jy < CN_GRID and 0 <= jx < CN_GRID and cells[jy, jx]):
            j_np = int(np.argmin((yy_r - cyc) ** 2 + (xx_r - cxc) ** 2))
            jy, jx = int(yy_r[j_np]), int(xx_r[j_np])
        r["cx"], r["cy"] = float(gx[jy, jx]), float(gy[jy, jx])
    if not labels:                                       # export variant: borders only, no labels/leaders
        return
    # ---- labels: bracket style for ALL (pure = single line), placed in EMPTY space + thin leaders.
    # Only the top HALF of label-worthy regions get a label (all regions keep their borders), ranked:
    # regions CONTAINING A NEEDLE first, then by point count, then the rest.
    needle_rows = [nd[0] for nd in M.get("needles", []) if 0 <= nd[0] < len(M["E"])]
    if needle_rows:
        n_iy, n_ix = iy[needle_rows], ix[needle_rows]
        for r in regions:
            r["has_needle"] = bool(r["cells"][n_iy, n_ix].any())
    else:
        for r in regions:
            r["has_needle"] = False
    cands = [r for r in regions if r["labelled"]]
    cands.sort(key=lambda r: (-int(r["has_needle"]), -r.get("nodes", 0)))
    cands = cands[:max(1, (len(cands) + 1) // 2)]
    if not cands:
        return
    _cn_place_labels(ax, cands, names, F, iy, ix, border_all)
    for r in cands:
        # leader: label box edge -> the CENTER of the region (not its edge), stopping a little short so
        # the tip doesn't sit on top of the nodes there
        axd, ayd = r["cx"], r["cy"]
        ddx, ddy = axd - r["lx"], ayd - r["ly"]
        if abs(ddx) > r["hw"] or abs(ddy) > r["hh"]:                # centre beyond the label box
            tt = [r["hw"] / abs(ddx) if abs(ddx) > 1e-12 else np.inf,
                  r["hh"] / abs(ddy) if abs(ddy) > 1e-12 else np.inf]
            t0 = min(min(tt), 1.0)
            ax.plot([r["lx"] + ddx * t0, r["lx"] + ddx * 0.92], [r["ly"] + ddy * t0, r["ly"] + ddy * 0.92],
                    color="white", lw=3.0, alpha=0.9, zorder=5.5,       # leader: white, thick
                    clip_on=False,                                      # labels may sit outside the panel
                    path_effects=[pe.withStroke(linewidth=5.4, foreground="black")])  # black outline
        _draw_blend_label(ax, r["lx"], r["ly"],
                          [CN_PRETTY.get(names[c], names[c]) for c in r["comps"]],
                          r["color"], alpha=r["alpha"], stroke_col=pal["stroke"])


def _conet_panel(fig, rect, cbrect, M, F, resp, bounds, view, target_eg, pal=None, labels=True):
    import matplotlib as mpl
    import matplotlib.patheffects as pe
    pal = pal or CN_PAL_DARK
    ax = fig.add_axes(rect); ax.set_facecolor("none")
    _conet_underlay(ax, M, F, pal=pal, labels=labels)
    E = M["E"]; iters = M["iters"]
    vis = np.ones(len(iters), bool) if view is None else (iters <= view + 0.5)   # time-travel
    y = M["resp"][:, M["resp_names"].index(resp)]
    size = _cn_node_size(M["strengthN"]); lo, hi = bounds[resp]; cmap = _cmap_for(resp)
    m = np.isfinite(y) & vis
    if view is not None:
        cur_it = view
    elif m.any():
        cur_it = int(np.nanmax(iters[m]))       # latest iteration MEASURED for this response
    else:
        cur_it = None
    cur = m & (np.abs(iters - cur_it) < 0.5) if cur_it is not None else np.zeros(len(iters), bool)
    non = m & ~cur
    ax.scatter(E[(~np.isfinite(y)) & vis, 0], E[(~np.isfinite(y)) & vis, 1], s=6,
               facecolors="none", edgecolors=pal["faint"], linewidths=0.5, zorder=3, alpha=0.5)
    # Draw order = which points sit ON TOP where they overlap. Highest value on top for every response
    # EXCEPT Band Gap, where the lowest (target-side / yellow) value is drawn on top instead. The current
    # iteration (red edge) is drawn last of all, so it always sits above every prior-iteration point.
    nv = y[non]
    order = (np.argsort(-nv) if resp == "Bandgap" else np.argsort(nv)) if nv.size else np.array([], int)
    En = E[non]
    ax.scatter(En[order, 0], En[order, 1], c=nv[order], cmap=cmap, vmin=lo, vmax=hi,
               s=size[non][order], edgecolors=pal["node_edge"], linewidths=0.18, zorder=4)  # thin outline
    if cur.any():                                # current iteration -> RED rims (like the ternary)
        ax.scatter(E[cur, 0], E[cur, 1], c=y[cur], cmap=cmap, vmin=lo, vmax=hi,
                   s=size[cur] * 1.2, edgecolors=pal["red"], linewidths=1.0, zorder=6)
    # Discovered Needles (ZoMBI-Hop ranked optima): one star per needle, rank 0 fully opaque then
    # fading to NEEDLE_MIN_ALPHA. Falls back to the single best-measured-Objective star when no
    # needles.db data is available. A SELECTED star gets a RED edge + star-shaped glow (below).
    si = M.get("selected")
    star_rows = set()
    # Needle stars are opt-in (UI toggle / --needles): when off, fall through to the single incumbent star.
    needles = [nd for nd in M.get("needles", []) if vis[nd[0]]] if M.get("show_needles") else []
    star_halo = pal["stroke"] or UI_BG
    if needles:
        for ni, (ridx, a) in enumerate(needles):
            star_rows.add(ridx)
            sel = (si == ridx)
            ax.scatter([E[ridx, 0]], [E[ridx, 1]], marker="*", s=430 if ni == 0 else 330, c=CN_STAR,
                       edgecolors=pal["red"] if sel else "#1a1200", linewidths=1.6 if sel else 1.0,
                       alpha=max(a, 0.9) if sel else a, zorder=8,
                       path_effects=[pe.withStroke(linewidth=2.0, foreground=star_halo)] if ni == 0 else None)
    else:
        ocol = M.get("resp_raw", M["resp"])[:, M["resp_names"].index("Objective")]  # RAW -> true incumbent
        ov = np.where(vis & np.isfinite(ocol), ocol, -np.inf)  # best MEASURED Objective SO FAR
        if np.max(ov) > -np.inf or len(E):
            bo = int(np.argmax(ov)) if np.max(ov) > -np.inf else M["best_obj"]
            star_rows.add(bo)
            sel = (si == bo)
            ax.scatter([E[bo, 0]], [E[bo, 1]], marker="*", s=430, c=CN_STAR,
                       edgecolors=pal["red"] if sel else "#1a1200", linewidths=1.6 if sel else 1.0,
                       alpha=0.9 if sel else 0.85, zorder=8,
                       path_effects=[pe.withStroke(linewidth=2.0, foreground=star_halo)])
    # click-to-inspect selection glow. For a STAR the glow RADIATES IN THE STAR SHAPE (scaled-up star
    # markers underneath, no circle) and the red lives on the star's own edge; nodes keep the circular
    # glow + ring.
    if si is not None and 0 <= si < len(E) and vis[si]:
        if si in star_rows:
            ax.scatter([E[si, 0]], [E[si, 1]], marker="*", s=430 * 5.5, c=UI_RED, alpha=0.10,
                       linewidths=0, zorder=7.4)
            ax.scatter([E[si, 0]], [E[si, 1]], marker="*", s=430 * 2.6, c=UI_RED, alpha=0.20,
                       linewidths=0, zorder=7.4)
        else:
            ss = float(size[si])
            ax.scatter([E[si, 0]], [E[si, 1]], s=ss * 9.0, c=UI_RED, alpha=0.10, linewidths=0, zorder=5)
            ax.scatter([E[si, 0]], [E[si, 1]], s=ss * 4.0, c=UI_RED, alpha=0.18, linewidths=0, zorder=5)
            ax.scatter([E[si, 0]], [E[si, 1]], s=ss * 1.7, facecolors="none", edgecolors=UI_RED,
                       linewidths=1.4, alpha=0.9, zorder=6.5)
    ax._conet_pick = True                        # tag: click-to-inspect hit-testing (see _on_canvas_click)
    # limits come from the dominance-field dict so the background extent and the axes always coincide;
    # draw_conet sized the panel FRAME to this data's aspect, so aspect="auto" fills the frame
    # near-isotropically (zoomed in, tight regions) without the window shape stretching the embedding.
    ax.set_xlim(F["x0"], F["x1"]); ax.set_ylim(F["y0"], F["y1"]); ax.set_aspect("auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(pal["faint"])
    # n = ... rides just ABOVE the panel's top-right corner (outside the plot area). The response name
    # is intentionally NOT drawn inside the panel any more — it lives only on the colourbar label.
    ax.text(1.0, 1.012, f"n = {int(m.sum())}", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, color=pal["muted"])
    cax = fig.add_axes(cbrect)
    cb = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.Normalize(lo, hi), cmap), cax=cax, extend="both")
    cb.set_label(RESP_LABEL.get(resp, resp), color=pal["muted"], fontsize=12)   # matches the tick colour
    cb.ax.tick_params(labelsize=9, colors=pal["muted"], length=2); cb.outline.set_visible(False)
    if resp == "Bandgap" and lo <= target_eg <= hi:
        cb.ax.axhline(target_eg, color=pal["red"], lw=2)


def _draw_umap_axes(fig, panel):
    """L-shaped UMAP-1/UMAP-2 orientation arrows whose 90-deg corner sits diagonally off the bottom-left
    corner of the panel — offset by the SAME distance LEFT and DOWN so the little axis reads symmetric
    (not 'more left than down'). Figure-space (not a shrink-to-fit sub-axes) so the arms stay
    perpendicular and equal-length, and the labels never ride on the shafts at any window aspect."""
    from matplotlib.patches import FancyArrowPatch
    fw, fh = fig.get_size_inches()
    arm = 0.40                                 # arm length (inches), equal for both arrows
    off = 0.18                                 # diagonal offset from the panel corner (inches), EQUAL x/y
    ox = max(0.006, panel[0] - off / fw)       # corner: `off` inches LEFT of the panel corner ...
    oy = max(0.006, panel[1] - off / fh)       # ... and the SAME `off` inches BELOW it -> symmetric
    for (x1, y1) in [(ox + arm / fw, oy), (ox, oy + arm / fh)]:   # ->right (UMAP-1), ^up (UMAP-2)
        p = FancyArrowPatch((ox, oy), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                            lw=2.0, color=UI_MUTED, shrinkA=0, shrinkB=0)
        p.set_transform(fig.transFigure); p.set_clip_on(False); fig.add_artist(p)
    fig.text(ox + 0.5 * arm / fw, oy - 0.09 / fh, "UMAP-1", ha="center", va="top",
             fontsize=8, color=UI_MUTED)
    fig.text(ox - 0.09 / fw, oy + 0.5 * arm / fh, "UMAP-2", ha="right", va="center",
             rotation=90, fontsize=8, color=UI_MUTED)


# CoNet panel frames are LANDSCAPE and sized to the embedding's own aspect (see draw_conet), clamped
# to this range so a far satellite widening the map can't make the frames an extreme sliver.
# FIXED CoNet panel aspect (width:height). Not data-driven, so the chart-area shape never changes with
# the embedding; the panels stay at this WIDE shape (near-max width) and the data fills them via
# _cn_view_limits (dead space is added on the short axis; points never spill).
CONET_FRAME_ASPECT = 1.82


def _grid2x2(fig, data_aspect, y0=0.23, y1=0.985, row_gap=0.033, cb_w=0.011, cb_pad=0.004,
             col_gap=0.05, max_frac=0.95, x_lo=0.0, x_hi=1.0):
    """Frames for a 2x2 grid of panels + right-hand colourbars, computed from the figure's LIVE size
    (the Tk canvas resizes the figure to fill the window, so fixed fractions would distort with the
    window shape). Each panel frame's ON-SCREEN aspect equals data_aspect exactly, as large as the
    [y0, y1] band / figure width allow, and the whole grid is centred horizontally WITHIN the band
    [x_lo, x_hi] (pass x_hi < 1 to reserve a right strip for the CoNet legend).
    Returns (panel_rects, cbar_rects) in figure fractions."""
    fw, fh = fig.get_size_inches()
    if fw <= 0 or fh <= 0:
        fw, fh = 16.0, 9.0
    avail = max(x_hi - x_lo, 0.1)
    max_total = max_frac * avail                      # cap grid+colourbar width to the available band
    h = (y1 - y0 - row_gap) / 2.0
    w = data_aspect * h * fh / fw                     # width fraction giving the target on-screen aspect
    total = 2.0 * (w + cb_pad + cb_w) + col_gap
    if total > max_total:                             # width-limited (tall window) -> shrink, keep aspect
        w = (max_total - col_gap) / 2.0 - cb_pad - cb_w
        h_new = w * fw / (data_aspect * fh)
        y0 = y0 + (y1 - y0 - (2.0 * h_new + row_gap)) / 2.0   # centre the rows in the band
        h = h_new
        total = 2.0 * (w + cb_pad + cb_w) + col_gap
    x0 = x_lo + (avail - total) / 2.0                 # centre within [x_lo, x_hi]
    xs = [x0, x0 + w + cb_pad + cb_w + col_gap]
    rects, cbs = [], []
    for yy in [y0 + h + row_gap, y0]:                 # top row first (reading order)
        for xx in xs:
            rects.append([xx, yy, w, h])
            cbs.append([xx + w + cb_pad, yy, cb_w, h])
    return rects, cbs


CONET_LEGEND_X = 0.865   # grid occupies [~0, CONET_LEGEND_X]; the (narrow) legend sits to its right


def draw_conet_legend(fig, rect, ccolor=None):
    """Vertical Compositions + Node-Size legend to the RIGHT of the CoNet grid (dark-theme port of the
    pub_conetwork legend). Taller than wide. Composition swatches use the SAME dark hues as the on-plot
    regions -- including the spatial colour re-assignment (`ccolor` from the live structure) -- so the
    legend always matches the map; the node-size circles mirror _cn_node_size."""
    from matplotlib.patches import FancyBboxPatch
    ax = fig.add_axes(rect); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.04, 0.01), 0.92, 0.98, boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor=UI_PANEL, edgecolor=UI_FAINT, linewidth=1.0, zorder=0))
    allc = dict(_cn_all_colors())
    if ccolor:
        allc.update(ccolor)                              # live spatial assignment wins for active comps
    sx, tx = 0.22, 0.40                                  # swatch centre x, label left x (block ~centred)
    # --- Compositions (full 10-component roster, so a shade never depends on which are present) ---
    ax.text(0.5, 0.965, "Compositions", ha="center", va="center", color=UI_TEXT,
            fontsize=10, fontweight="bold", zorder=3)
    ys = np.linspace(0.915, 0.50, len(CN_LEGEND_NAMES))
    for nm, yy in zip(CN_LEGEND_NAMES, ys):
        col = _dark_comp_color(allc.get(nm, (0.55, 0.6, 0.68)))
        ax.scatter([sx], [yy], marker="s", s=110, c=[col], edgecolors="#20242c", linewidths=0.5, zorder=2)
        ax.text(tx, yy, CN_PRETTY.get(nm, nm), ha="left", va="center", color=UI_TEXT, fontsize=8.5, zorder=3)
    ax.plot([0.12, 0.88], [0.455, 0.455], color=UI_FAINT, lw=0.8, zorder=1)
    # --- Node size = local sampling density (+ incumbent star) ---
    ax.text(0.5, 0.415, "Node Size", ha="center", va="center", color=UI_TEXT,
            fontsize=10, fontweight="bold", zorder=3)
    ax.text(0.5, 0.38, "Sampling Density", ha="center", va="center", color=UI_MUTED,
            fontsize=7.5, zorder=3)
    for (f, lab), yy in zip([(0.05, "Sparse"), (0.45, "Medium"), (0.9, "Dense")], [0.315, 0.235, 0.155]):
        ax.scatter([sx], [yy], s=_cn_node_size(f), c="#c2cad6", edgecolors="#20242c",
                   linewidths=0.4, zorder=2)
        ax.text(tx, yy, lab, ha="left", va="center", color=UI_TEXT, fontsize=8.5, zorder=3)
    ax.scatter([sx], [0.065], marker="*", s=260, c=CN_STAR, edgecolors="#1a1200", linewidths=0.8, zorder=2)
    ax.text(tx, 0.065, "Discovered\nNeedles", ha="left", va="center", color=UI_TEXT, fontsize=8.5,
            zorder=3, linespacing=1.1)



# ===========================================================================
# run-specific code (NOT from visualization.py): data loading, single-panel
# layout, click-to-inspect tooltip, interactive viewer, main.
# ===========================================================================

# ---------------------------------------------------------------------------
# run-directory data loading (replaces the live results.db reader)
# ---------------------------------------------------------------------------
def _resolve_run_dir(run_arg):
    """Accept a full path or a bare run name resolved against runs/."""
    p = Path(run_arg)
    if p.is_dir():
        return p.resolve()
    cand = RUNS_DIR / run_arg
    if cand.is_dir():
        return cand.resolve()
    raise FileNotFoundError(f"Run directory not found: {run_arg}")


def _default_snapshot(run_dir):
    """The snapshot named in latest.txt, else the last snapshot directory."""
    latest = run_dir / "latest.txt"
    if latest.exists():
        name = latest.read_text().strip()
        if name:
            return name
    snaps = sorted(s.name for s in (run_dir / "snapshots").iterdir() if s.is_dir())
    if not snaps:
        raise FileNotFoundError(f"No snapshots under {run_dir}")
    return snaps[-1]


def _optimizing_dims(run_dir):
    """The run's optimizing dims, in measured-column order (live_plot_state.json, then hw_config.json)."""
    import json
    lp = run_dir / "live_plot_state.json"
    if lp.exists():
        try:
            dims = json.loads(lp.read_text()).get("optimizing_dims")
            if dims:
                return [int(d) for d in dims]
        except Exception:
            pass
    hw = run_dir / "hw_config.json"
    if hw.exists():
        try:
            raw = json.loads(hw.read_text()).get("dims", "")
            dims = [int(x) for x in str(raw).split(",") if x.strip() != ""]
            if dims:
                return dims
        except Exception:
            pass
    return None


def _is_csv_run(run_dir):
    """A ``points.csv``-style run directory (e.g. optimize/runs/trial_*/run_*): a flat CSV of samples
    instead of reconstructable snapshot tensors."""
    return (Path(run_dir) / "points.csv").exists()


def _reduce_comp(X):
    """Row-normalize X to composition fractions and drop all-zero columns.
    Returns (comp, active_mask) so the SAME column selection can be applied to needle points."""
    s = X.sum(1, keepdims=True)
    comp = X / np.where(s == 0, 1.0, s)
    active = comp.sum(0) > 0
    if 2 <= int(active.sum()) < comp.shape[1]:
        return comp[:, active], active
    return comp, np.ones(comp.shape[1], bool)


def _reduce_needles(P, active):
    """Row-normalize ranked needle points (K, C_full) and select the SAME active columns as the samples,
    so needle compositions live in the same reduced space as ``comp``. None/empty/width-mismatch -> None."""
    if P is None:
        return None
    P = np.asarray(P, float)
    if P.ndim != 2 or len(P) == 0 or P.shape[1] != active.shape[0]:
        return None
    s = P.sum(1, keepdims=True)
    return (P / np.where(s == 0, 1.0, s))[:, active]


def _read_needles_csv(run_dir):
    """Ranked needle composition rows (K, C_full) from ``needles.csv`` (in ``needle_idx`` rank order),
    or None if the file is absent / has no x-columns."""
    import csv
    p = Path(run_dir) / "needles.csv"
    if not p.exists():
        return None
    with open(p, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    xcols = sorted((c for c in rows[0] if c and re.fullmatch(r"x\d+", c)), key=lambda c: int(c[1:]))
    if not xcols:
        return None
    if "needle_idx" in rows[0]:
        rows = sorted(rows, key=lambda r: float(r["needle_idx"]))   # needle_idx == rank
    return np.array([[float(r[c]) for c in xcols] for r in rows], float)


def _read_needles_json(run_dir, snapshot):
    """Ranked needle composition rows from ``snapshots/<snapshot>/needles.json`` (best ``value`` first),
    or None if the file is absent / empty."""
    import json
    p = Path(run_dir) / "snapshots" / str(snapshot) / "needles.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    items = [d for d in data if isinstance(d, dict) and d.get("point")]
    if not items:
        return None
    items.sort(key=lambda d: -(float(d["value"]) if d.get("value") is not None else -np.inf))
    return np.array([list(map(float, d["point"])) for d in items], float)


def _needle_rows(comp, needles_comp):
    """Map each ranked needle to the nearest measured sample (composition L2), dedupe keeping the best
    rank, and assign a fading star opacity (rank 0 -> 1.0 down to NEEDLE_MIN_ALPHA). Returns the
    ``[(row_index, alpha), ...]`` list the CoNet panel consumes as ``M['needles']``."""
    if needles_comp is None or len(needles_comp) == 0 or len(comp) == 0:
        return []
    K = len(needles_comp)
    rows, seen = [], set()
    for rank, nc in enumerate(needles_comp):
        ridx = int(np.argmin(((comp - nc) ** 2).sum(1)))
        if ridx in seen:
            continue
        seen.add(ridx)
        a = 1.0 - (1.0 - NEEDLE_MIN_ALPHA) * (rank / max(K - 1, 1))
        rows.append((ridx, float(a)))
    return rows


def load_csv_run_data(run_dir):
    """Reconstruct ``(comp, names, resp, iters, title, needles_comp)`` from a ``points.csv`` run dir.

    These directories (optimize/runs/trial_*/run_*) store one row per sample in ``points.csv`` with
    ``x0..xk`` simplex-coordinate columns and a ``Y`` objective column, rather than the delta snapshots
    a live campaign directory carries. Row order is the acquisition order (used as the pseudo-iteration).
    Ranked optima are read from a sibling ``needles.csv`` when present.
    """
    import csv
    run_dir = Path(run_dir)
    with open(run_dir / "points.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No datapoints in {run_dir}/points.csv")
    xcols = sorted((c for c in rows[0] if c and re.fullmatch(r"x\d+", c)), key=lambda c: int(c[1:]))
    if not xcols:
        raise RuntimeError(f"No x0..xk composition columns in {run_dir}/points.csv")
    ycol = "Y" if "Y" in rows[0] else next((c for c in rows[0] if c and c.lower() == "y"), None)
    if ycol is None:
        raise RuntimeError(f"No Y objective column in {run_dir}/points.csv")
    X = np.array([[float(r[c]) for c in xcols] for r in rows], dtype=float)
    Y = np.array([float(r[ycol]) for r in rows], dtype=float)

    comp, active = _reduce_comp(X)
    names = [n for n, a in zip(xcols, active) if a]   # synthetic benchmark dims: no real-material labels
    needles_comp = _reduce_needles(_read_needles_csv(run_dir), active)

    resp = Y.reshape(-1, 1)
    iters = np.arange(len(comp), dtype=float)
    title = f"{run_dir.name} — CoNet · {len(comp)} samples · coloured by Objective  (points.csv)"
    return comp, names, resp, iters, title, needles_comp


def load_run_conet_data(run_dir, snapshot=None):
    """Reconstruct ``(comp, names, resp, iters, title, needles_comp)`` for a run directory.

    comp          (N, C) row-normalized ACTIVE composition fractions
    names         the C component labels (real materials via DIM_LABELS, in optimizing-dim order)
    resp          (N, 1) measured Objective
    iters         (N,)  acquisition order used as a pseudo-iteration (draw order + latest-sample rim)
    needles_comp  (K, C) ranked needle compositions in the same reduced space as comp, or None

    A ``points.csv``-style directory (any dir with a points.csv/needles.csv, e.g.
    optimize/runs/trial_*/run_*) is loaded flat from CSV; otherwise the run's delta snapshots are
    reconstructed.
    """
    if _is_csv_run(run_dir):
        return load_csv_run_data(run_dir)
    snapshot = snapshot or _default_snapshot(run_dir)
    tensors = reconstruct_snapshot_tensors(run_dir, snapshot, device="cpu")
    X = tensors.get("X_all_actual")
    Y = tensors.get("Y_all")
    if X is None or Y is None or X.shape[0] == 0:
        raise RuntimeError(f"No datapoints reconstructed from {run_dir}/{snapshot}")
    X = X.detach().cpu().numpy().astype(float)
    Y = Y.detach().cpu().numpy().astype(float).ravel()

    dims = _optimizing_dims(run_dir)
    if dims is None or len(dims) != X.shape[1]:
        dims = list(range(X.shape[1]))          # fall back to raw column order
    names = [DIM_LABELS.get(d, f"dim {d}") for d in dims]

    comp, active = _reduce_comp(X)
    names = [n for n, a in zip(names, active) if a]
    needles_comp = _reduce_needles(_read_needles_json(run_dir, snapshot), active)

    resp = Y.reshape(-1, 1)
    iters = np.arange(len(comp), dtype=float)
    title = f"{run_dir.name} — CoNet · {len(comp)} samples · coloured by Objective  ({snapshot})"
    return comp, names, resp, iters, title, needles_comp


def build_conet(comp, names, resp, iters, umap_md=CN_UMAP_MD, gap_reach=CN_GAP_REACH,
                purity_thr=CN_PURITY_THR, needles_comp=None, show_needles=False):
    """Build the CoNet structure (UMAP) + dominance fields for a single-objective run. Ranked needle
    optima (if any) are anchored to their nearest sample node and stored as ``M['needles']``; whether
    their stars are drawn is gated by ``M['show_needles']`` (the UI toggle / --needles flag)."""
    M = build_conet_structure(comp, names, resp, iters, list(VALUE_COLUMNS),
                              min_dist=umap_md, gap_reach=gap_reach, purity_thr=purity_thr)
    M["iters"] = np.asarray(iters, float)
    M["needles"] = _needle_rows(M["comp"], needles_comp)
    M["show_needles"] = bool(show_needles)
    with _Progress("dominance fields"):
        F = conet_dominance_fields(M)
    return M, F


def conet_bounds(resp, pct=CN_PCT):
    """10th..90th-percentile Objective colour bounds (raw), matching the live viewer's CoNet bounds."""
    v = resp[:, VALUE_COLUMNS.index("Objective")]
    v = v[np.isfinite(v)]
    if len(v):
        lo, hi = np.nanpercentile(v, pct)
        if hi <= lo:
            lo, hi = float(np.nanmin(v)), float(np.nanmax(v)) + 1e-6
        return {"Objective": (float(lo), float(hi))}
    return {"Objective": (0.0, 1.0)}


# ---------------------------------------------------------------------------
# single-panel CoNet layout (the live viewer's draw_conet is a 2x2 grid; a run
# has a single Objective response, so we draw one large panel + legend/tooltip)
# ---------------------------------------------------------------------------
def _single_rects(fig, data_aspect=CONET_FRAME_ASPECT, x_lo=0.045, x_hi=0.99,
                  y0=0.075, y1=0.94, cb_w=0.014, cb_pad=0.006):
    """One CoNet panel + right colourbar, sized to data_aspect within [x_lo,x_hi] x [y0,y1]."""
    fw, fh = fig.get_size_inches()
    if fw <= 0 or fh <= 0:
        fw, fh = 16.0, 9.0
    avail_w = max(x_hi - x_lo, 0.1)
    band_h = max(y1 - y0, 0.1)
    h = band_h
    w = data_aspect * h * fh / fw               # width fraction giving the target on-screen aspect
    if w + cb_pad + cb_w > avail_w * 0.98:       # width-limited -> shrink, keep aspect
        w = avail_w * 0.98 - cb_pad - cb_w
        h = w * fw / (data_aspect * fh)
    x0 = x_lo + (avail_w - (w + cb_pad + cb_w)) / 2.0
    yy = y0 + (band_h - h) / 2.0
    return [x0, yy, w, h], [x0 + w + cb_pad, yy, cb_w, h]


def draw_conet_single(fig, M, F, bounds, view=None, target_eg=TARGET_EG_DEFAULT, tooltip=None):
    """Draw the single Objective CoNet panel + UMAP axes + (legend | selected-sample tooltip)."""
    panel, cb = _single_rects(fig)
    _conet_panel(fig, panel, cb, M, F, "Objective", bounds, view, target_eg)
    _draw_umap_axes(fig, panel)                  # orientation L off the panel's bottom-left corner
    # The Compositions / Node-Size legend has been removed so the CoNet panel can occupy the full width.
    # The selected-sample tooltip (interactive click-to-inspect) still overlays a right strip when active.
    if tooltip is not None:
        strip = [0.875, 0.095, 0.115, 0.84]
        return draw_tooltip(fig, strip, tooltip)
    return None


# ---------------------------------------------------------------------------
# click-to-inspect: selected-sample payload + a simplified tooltip (the live
# viewer's tooltip also shows environment / IV / PL / needle data pulled from
# save_all.db, none of which a run directory has)
# ---------------------------------------------------------------------------
def selected_payload(M, idx, names):
    """Tooltip data for the clicked sample: composition formula + fractions + Objective."""
    comp = M["comp"][idx]
    compd = {names[i]: float(comp[i]) for i in range(len(names))}
    obj = float(M["resp"][idx, VALUE_COLUMNS.index("Objective")])
    return dict(idx=int(idx), iter=int(M["iters"][idx]), comp=compd,
                formula=build_formula(compd), objective=obj)


def draw_tooltip(fig, rect, D):
    """Selected-sample inspector drawn in place of the legend. Returns the ✕ close-button rect
    (figure fractions) for click hit-testing."""
    import textwrap
    from matplotlib.patches import FancyBboxPatch
    x0, y0, w, h = rect
    ax = fig.add_axes(rect); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.005), 0.96, 0.99, boxstyle="round,pad=0,rounding_size=0.04",
                                facecolor=UI_PANEL, edgecolor=UI_FAINT, linewidth=1.0, zorder=0))
    ax.text(0.09, 0.985, f"Selected · sample {D['idx']}", ha="left", va="top", color=UI_TEXT,
            fontsize=10.5, fontweight="bold", zorder=3)
    ax.text(0.91, 0.987, "✕", ha="center", va="top", color=UI_MUTED, fontsize=12,
            fontweight="bold", zorder=4)
    close = [x0 + 0.80 * w, y0 + 0.955 * h, 0.20 * w, 0.045 * h]      # generous hit box round the ✕
    ty = 0.94

    def header(txt):
        nonlocal ty
        ax.plot([0.07, 0.93], [ty + 0.004, ty + 0.004], color=UI_FAINT, lw=0.7, zorder=2)
        ax.text(0.07, ty - 0.004, txt.upper(), ha="left", va="top", color=UI_ACCENT,
                fontsize=8.0, fontweight="bold", zorder=3)
        ty -= 0.032

    def rowkv(k_, v_):
        nonlocal ty
        ax.text(0.09, ty, k_, ha="left", va="top", color=UI_MUTED, fontsize=8.4, zorder=3)
        ax.text(0.93, ty, v_, ha="right", va="top", color=UI_TEXT, fontsize=8.4, zorder=3)
        ty -= 0.034

    header("Composition")
    frm_lines = textwrap.wrap(D.get("formula") or "—", 22)[:6]
    ax.text(0.09, ty, "\n".join(frm_lines), ha="left", va="top", color=UI_TEXT, fontsize=8.2,
            zorder=3, linespacing=1.35)
    ty -= 0.03 * len(frm_lines) + 0.014
    for c_, v in D["comp"].items():
        rowkv(CN_PRETTY.get(c_, c_), f"{v:.3f}" if np.isfinite(v) else "—")
    ty -= 0.012
    header("Characterization")
    rowkv("Objective", f"{D['objective']:.4g}")
    return close


# ---------------------------------------------------------------------------
# interactive matplotlib viewer (ports the live viewer's click hit-testing,
# minus time-travel)
# ---------------------------------------------------------------------------
def _draw_needle_toggle(fig, on, rect=(0.012, 0.944, 0.115, 0.04)):
    """Small clickable 'Needles: ON/OFF' button (top-left). Returns its rect in figure fractions for
    click hit-testing; the star + accent light up when needles are shown."""
    from matplotlib.patches import FancyBboxPatch
    x0, y0, w, h = rect
    ax = fig.add_axes(rect); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    accent = UI_ACCENT if on else UI_FAINT
    ax.add_patch(FancyBboxPatch((0.02, 0.08), 0.96, 0.84, boxstyle="round,pad=0,rounding_size=0.28",
                                facecolor=UI_FIELD_HI if on else UI_FIELD, edgecolor=accent,
                                linewidth=1.2, zorder=0))
    ax.scatter([0.14], [0.5], marker="*", s=150, c=CN_STAR if on else UI_FAINT,
               edgecolors="#1a1200" if on else "none", linewidths=0.8, zorder=2)
    ax.text(0.30, 0.5, f"Needles: {'ON' if on else 'OFF'}", ha="left", va="center",
            color=UI_TEXT if on else UI_MUTED, fontsize=9.5, fontweight="bold", zorder=2)
    return [x0, y0, w, h]


class ConetViewer:
    """Interactive CoNet window: click a node to inspect it, click empty space or the ✕ to deselect;
    toggle the ranked-needle stars with the top-left button or the 'n' key (off by default)."""

    def __init__(self, M, F, bounds, names, title, target_eg=TARGET_EG_DEFAULT, show_needles=False):
        self.M, self.F, self.bounds, self.names, self.title = M, F, bounds, names, title
        self.target_eg = target_eg
        self.sel = None
        self.detail = None
        self.close_rect = None
        self.needle_rect = None
        self.show_needles = bool(show_needles)
        self.has_needles = bool(M.get("needles"))
        self.fig = None

    def _draw(self):
        self.fig.clear()
        self.M["selected"] = self.sel
        self.M["show_needles"] = self.show_needles
        tooltip = self.detail if self.sel is not None else None
        self.close_rect = draw_conet_single(self.fig, self.M, self.F, self.bounds,
                                             target_eg=self.target_eg, tooltip=tooltip)
        self.needle_rect = (_draw_needle_toggle(self.fig, self.show_needles)
                            if self.has_needles else None)
        self.fig.text(0.5, 0.985, self.title, ha="center", va="top", color=UI_TEXT, fontsize=12.5)

    def _redraw(self):
        self._draw()
        self.fig.canvas.draw_idle()

    def _on_click(self, event):
        """Select the sample under a left-click (tight radius -> highest Objective in a stack, else the
        nearest node within a looser radius). Clicking the ✕ or empty space deselects."""
        try:
            if getattr(event, "button", None) != 1:
                return
            tb = getattr(self.fig.canvas, "toolbar", None)
            if tb is not None and getattr(tb, "mode", ""):
                return                                    # pan/zoom active -> not a selection click
            if event.x is not None:
                fw, fh = self.fig.bbox.width, self.fig.bbox.height
                fx, fy = event.x / max(fw, 1), event.y / max(fh, 1)

                def _in(rect):
                    return rect is not None and rect[0] <= fx <= rect[0] + rect[2] \
                        and rect[1] <= fy <= rect[1] + rect[3]
                if _in(self.needle_rect):                 # flip the ranked-needle stars
                    self.show_needles = not self.show_needles; self._redraw()
                    return
                if _in(self.close_rect):
                    self.sel = None; self.detail = None; self._redraw()
                    return
            ax = event.inaxes
            if ax is None or not getattr(ax, "_conet_pick", False) or event.xdata is None:
                return
            E = self.M["E"]
            span = self.F.get("core_span", 1.0)
            d2 = (E[:, 0] - event.xdata) ** 2 + (E[:, 1] - event.ydata) ** 2
            obj = self.M["resp"][:, VALUE_COLUMNS.index("Objective")]
            r_tight, r_loose = 0.012 * span, 0.03 * span
            tight = np.where(d2 <= r_tight ** 2)[0]
            if len(tight):
                sc = np.where(np.isfinite(obj[tight]), obj[tight], -np.inf)
                idx = int(tight[int(np.argmax(sc))]) if np.max(sc) > -np.inf \
                    else int(tight[int(np.argmin(d2[tight]))])
            else:
                cand = np.where(d2 <= r_loose ** 2)[0]
                if not len(cand):
                    if self.sel is not None:              # empty-space click -> deselect
                        self.sel = None; self.detail = None; self._redraw()
                    return
                idx = int(cand[int(np.argmin(d2[cand]))])
            self.sel = idx
            self.detail = selected_payload(self.M, idx, self.names)
            self._redraw()
        except Exception:
            import traceback; traceback.print_exc()

    def _on_key(self, event):
        """'n' toggles the ranked-needle stars (same as the top-left button)."""
        if getattr(event, "key", None) == "n" and self.has_needles:
            self.show_needles = not self.show_needles; self._redraw()

    def show(self):
        import matplotlib.pyplot as plt
        _apply_mpl_theme()
        self.fig = plt.figure(figsize=(15.5, 9.0), facecolor=UI_BG)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        try:
            self.fig.canvas.manager.set_window_title("ZoMBI-Hop CoNet")
        except Exception:
            pass
        self._draw()
        plt.show()


# ---------------------------------------------------------------------------
# headless PNG render
# ---------------------------------------------------------------------------
def save_png(M, F, bounds, title, out_path, target_eg=TARGET_EG_DEFAULT, dpi=130):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    _apply_mpl_theme()
    fig = Figure(figsize=(15.5, 9.0), facecolor=UI_BG)
    FigureCanvasAgg(fig)
    M["selected"] = None
    draw_conet_single(fig, M, F, bounds, target_eg=target_eg, tooltip=None)
    fig.text(0.5, 0.985, title, ha="center", va="top", color=UI_TEXT, fontsize=12.5)
    fig.savefig(out_path, dpi=dpi, facecolor=UI_BG)
    return out_path


def render_run_conet(run_dir, save_path=None, *, snapshot=None, show_needles=True,
                     umap_md=CN_UMAP_MD, gap_reach=CN_GAP_REACH, purity_thr=CN_PURITY_THR):
    """Render a run's CoNet PNG headlessly (the same view ``main --save`` produces).

    Loads the run's own samples + discovered needles, builds the CoNet and writes
    the PNG to ``save_path`` (default: ``<run_dir>/conet.png``). Returns
    ``(png_path, n_stars_drawn)``.
    """
    run_dir = Path(run_dir)
    save_path = Path(save_path) if save_path else run_dir / "conet.png"
    comp, names, resp, iters, title, needles_comp = load_run_conet_data(run_dir, snapshot)
    M, F = build_conet(comp, names, resp, iters, umap_md=umap_md, gap_reach=gap_reach,
                       purity_thr=purity_thr, needles_comp=needles_comp, show_needles=show_needles)
    bounds = conet_bounds(resp)
    out = save_png(M, F, bounds, title, str(save_path))
    return out, len(M["needles"])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Interactive CoNet (UMAP composition network) view of a ZoMBI-Hop run directory.")
    ap.add_argument("--run", default=DEFAULT_RUN,
                    help="run directory or bare run name (default: %(default)s)")
    ap.add_argument("--snapshot", default=None,
                    help="snapshot to reconstruct up to (default: latest.txt)")
    ap.add_argument("--save", default=None, metavar="PNG",
                    help="render once to this PNG and exit (headless, no window)")
    ap.add_argument("--purity", type=float, default=CN_PURITY_THR,
                    help="PURITY layout threshold (0 = plurality layout; default: %(default)s)")
    ap.add_argument("--spread", type=float, default=CN_UMAP_MD,
                    help="UMAP min_dist / point spread (default: %(default)s)")
    ap.add_argument("--gap-reach", type=float, default=CN_GAP_REACH,
                    help="far-satellite squash factor (default: %(default)s)")
    ap.add_argument("--needles", action="store_true",
                    help="show the ranked-needle stars (off by default; toggleable live in the window)")
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run)
    comp, names, resp, iters, title, needles_comp = load_run_conet_data(run_dir, args.snapshot)
    print(f"Loaded {len(comp)} samples from {run_dir.name} · components = {names}")
    M, F = build_conet(comp, names, resp, iters, umap_md=args.spread, gap_reach=args.gap_reach,
                       purity_thr=args.purity, needles_comp=needles_comp, show_needles=args.needles)
    print(f"Needles: {len(M['needles'])} loaded"
          + ("" if M["needles"] else " (none found)"))
    bounds = conet_bounds(resp)

    if args.save:
        out = save_png(M, F, bounds, title, args.save)
        print(f"wrote {out}")
    else:
        print("Opening interactive CoNet — click a node to inspect it; click empty space or the ✕ "
              "to deselect; toggle needle stars with the top-left button or the 'n' key.")
        ConetViewer(M, F, bounds, names, title, show_needles=args.needles).show()


if __name__ == "__main__":
    main()
