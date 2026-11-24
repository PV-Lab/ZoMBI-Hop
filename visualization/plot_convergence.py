"""
visualization/plot_convergence.py
=================================
Static-PNG export of the ZoMBI-Hop convergence plot for a *real* run directory.

This is the standalone/offline twin of the GUI's ``ConvergencePlotFrame`` in
``interface/app.py``: objective Y vs. sample index, a gray band over the
initialization samples and orange bands over each zoomed-in stretch, darker
for deeper zooms (``draw_search_bands``), the running-best envelope,
penalized-vs-valid point colouring, and a lollipop-and-leash marker per needle
(``draw_needle_leashes``): a numbered star on the sampled point, joined by a
horizontal leash to a diamond at the sample index where it was declared. It reconstructs the dataset from the run's delta
snapshots (``reconstruct_snapshot_tensors``) — the same source the GUI and
``visualization/plot_run.py`` use — so no live Tk session is needed.

Each result packet (the batch of samples one objective call returned) is drawn
with its higher-objective end last (``orient_packets``). This is display only:
the run's stored order is untouched, and packet boundaries do not move.

Usage
-----
  conda activate zombi-hop
  python visualization/plot_convergence.py runs/run_7eb9 runs/run_9dfe
  python visualization/plot_convergence.py run_7eb9 --out my_conv.png
  python visualization/plot_convergence.py run_9dfe --snapshot 0030_act9_z0_i0

Flags
-----
  RUN [RUN ...]     One or more run directories (or bare run names under runs/).
  --snapshot NAME   Snapshot to reconstruct up to (default: latest.txt).
                    Only meaningful when a single run is given.
  --out PATH        Output PNG path (single run only; default:
                    <run_dir>/convergence.png).
  --show            Display each figure as well as saving it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

# ── project root on sys.path so `src` imports resolve ──────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from src.utils.datahandler import reconstruct_snapshot_tensors  # noqa: E402

_ACT_RE = re.compile(r"_act(\d+)")
_ACT_ZOOM_RE = re.compile(r"_act(\d+)_z(\d+)")

ROOT = _HERE.parent
RUNS_DIR = ROOT / "runs"


# ── run/snapshot resolution (mirrors plot_run.py) ──────────────────────────────

def _resolve_run_dir(run_arg: str) -> Path:
    """Accept a full path, or a bare run name resolved against runs/."""
    p = Path(run_arg)
    if p.is_dir():
        return p.resolve()
    candidate = RUNS_DIR / run_arg
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"Run directory not found: {run_arg}")


def _default_snapshot(run_dir: Path) -> str:
    """The snapshot named in latest.txt, or the last snapshot directory."""
    latest = run_dir / "latest.txt"
    if latest.exists():
        name = latest.read_text().strip()
        if name:
            return name
    snaps = sorted(s.name for s in (run_dir / "snapshots").iterdir() if s.is_dir())
    if not snaps:
        raise FileNotFoundError(f"No snapshots found under {run_dir}")
    return snaps[-1]


# ── data loading ───────────────────────────────────────────────────────────────

class _ConvData:
    """The minimal slice of a run needed to draw the convergence plot."""

    def __init__(self, run_id: str, snapshot: str):
        self.run_id = run_id
        self.snapshot = snapshot
        self.Y: np.ndarray | None = None           # (n,)
        self.penalty_mask: np.ndarray | None = None  # (n,) bool
        self.needle_indices: np.ndarray | None = None  # (k,) int
        self.activation_starts: np.ndarray | None = None  # (a,) int sample idx
        self.declared_samples: np.ndarray | None = None   # (k,) sample idx per needle declaration
        self.n_init = 0                                     # init-design samples (before activation 0)
        self.zoom_starts: np.ndarray | None = None  # (z,) sample idx where each (activation, zoom) stretch begins
        self.zoom_levels: np.ndarray | None = None  # (z,) that stretch's zoom level (0 = full search box)
        self.packet_starts: np.ndarray | None = None  # (p,) sample idx where each result packet begins
        self.n_points = 0
        self.n_needles = 0


def init_packet_sizes(run_dir: Path, n_init: int) -> list[int] | None:
    """Sizes of the init design's result packets, in the order they were stacked
    into the init data; None when that is unknown.

    The init design is saved as ONE snapshot but arrives as several objective
    packets, one per axial trace. ``init_state.json`` keeps every usable block
    with its arrival time, and ``run_axial_initialization`` concatenates them in
    arrival order (``InitQueue.measured``), so sorting the blocks by time
    recovers the boundaries. Runs without that file, or whose blocks do not add
    up to ``n_init``, get None and keep their init in stored order."""
    path = run_dir / "init_state.json"
    if n_init <= 0 or not path.exists():
        return None
    try:
        queue = json.loads(path.read_text(encoding="utf-8"))["queue"]
        blocks = [b for trace in queue["data"] for b in trace if len(b["y"])]
        sizes = [len(b["y"]) for b in sorted(blocks, key=lambda b: b["t"])]
    except Exception:
        return None
    return sizes if sum(sizes) == n_init else None


def load_activation_timeline(
        run_dir: Path, snapshot: str,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    """Activation starts, needle-declaration samples, init size, zoom stretches,
    result packets.

    Replays the delta snapshots up to ``snapshot`` (the same traversal
    ``reconstruct_snapshot_tensors`` uses); a port of the GUI's
    ``interface/app.py:_activation_timeline``.

    ``n_init``: the number of initialization samples — everything measured
    before activation 0's first sample. The init snapshot carries no ``actN``
    tag, so this is the cumulative point count at the first ``actN`` snapshot
    that adds points; a run still in its init phase (no such snapshot yet) is
    all init. Taken from the replay rather than the deltas' ``n_init_points``
    marker, which older resumes reset to 0 in every later snapshot.

    ``starts``: the cumulative point count each time the ``actN`` number in the
    snapshot-dir name changes, after activation 0 — i.e. the interior reset
    boundaries only. Activation 0's own start is ``n_init``.

    ``declared``: one entry per needle, in needle-stack order (so entry k pairs
    with ``needle_indices[k]``) — the last sample index measured before the
    snapshot whose needle stack first grew. When the stack is rebuilt rather than
    appended to, earlier declarations are dropped and every rebuilt needle counts
    as declared at that snapshot.

    ``zoom_starts`` / ``zoom_levels``: one entry per stretch of samples measured
    at a single (activation, zoom) — the cumulative point count each time the
    ``actN_zM`` pair in the snapshot-dir name changes, and that stretch's zoom
    level M (0 = the full search box, 1 = the first zoom-in, …). A stretch runs
    to the next one's start. Untagged snapshots (``final``, retro / corrected
    needles) continue the current stretch, and init has none, so the first
    stretch starts at ``n_init``.

    ``packet_starts``: where each result packet begins — the batch of samples
    one objective call returned (``orient_packets`` flips them for display).
    After init that is every snapshot that adds points; the init packets come
    from ``init_packet_sizes`` and are omitted when it cannot tell them apart.

    All empty / 0 for legacy full-copy runs with no per-snapshot deltas."""
    snap_dir = run_dir / "snapshots"
    if not snap_dir.is_dir():
        empty = np.array([], dtype=int)
        return empty, empty, 0, empty, empty, empty

    starts: list[int] = []
    declared: list[int] = []
    zoom_starts: list[int] = []
    zoom_levels: list[int] = []
    packets: list[int] = []   # first sample of every snapshot that adds points
    n_seen = 0
    stack: list[int] = []   # needle_indices of the current needle stack
    cur_act: int | None = None
    cur_act_zoom: tuple[int, int] | None = None
    for sdir in sorted(s for s in snap_dir.iterdir() if s.is_dir()):
        delta_path = sdir / "delta.pt"
        if delta_path.exists():
            d = torch.load(str(delta_path), map_location="cpu", weights_only=False)
            x_new = d.get("X_new")
            n_new = int(x_new.shape[0]) if isinstance(x_new, torch.Tensor) else 0
            ni = d.get("needle_indices")
            new_stack = (ni.long().ravel().tolist() if isinstance(ni, torch.Tensor)
                         else stack)
        else:
            n_new, new_stack = 0, stack
        m = _ACT_RE.search(sdir.name)
        act = int(m.group(1)) if m else cur_act
        if act is not None and act != cur_act and n_new > 0:
            starts.append(n_seen)
            cur_act = act
        mz = _ACT_ZOOM_RE.search(sdir.name)
        act_zoom = (int(mz.group(1)), int(mz.group(2))) if mz else cur_act_zoom
        if act_zoom is not None and act_zoom != cur_act_zoom and n_new > 0:
            zoom_starts.append(n_seen)
            zoom_levels.append(act_zoom[1])
            cur_act_zoom = act_zoom
        if n_new > 0:
            packets.append(n_seen)
        n_seen += n_new
        if new_stack != stack:
            if new_stack[:len(stack)] != stack:   # rebuilt, not appended to
                declared = []
            declared.extend([n_seen - 1] * (len(new_stack) - len(declared)))
            stack = new_stack
        if sdir.name == snapshot:
            break

    n_init = starts[0] if starts else n_seen
    sizes = init_packet_sizes(run_dir, n_init) or []
    init_starts = np.cumsum([0] + sizes[:-1]).tolist() if sizes else []
    packet_starts = sorted(set(init_starts) | {p for p in packets if p >= n_init})
    return (np.array(sorted({s for s in starts if s > n_init}), dtype=int),
            np.array(declared, dtype=int), n_init,
            np.array(zoom_starts, dtype=int), np.array(zoom_levels, dtype=int),
            np.array(packet_starts, dtype=int))


def load_conv_data(run_dir: Path, snapshot: str | None) -> _ConvData:
    """Reconstruct Y / penalty-mask / needle-indices for a run at ``snapshot``."""
    snapshot = snapshot or _default_snapshot(run_dir)
    s = reconstruct_snapshot_tensors(run_dir, snapshot, device="cpu")

    cd = _ConvData(run_dir.name, snapshot)

    y = s.get("Y_all")
    if y is None or y.numel() == 0:
        raise RuntimeError(f"No datapoints reconstructed from {run_dir}/{snapshot}")
    cd.Y = y.float().numpy().ravel()
    cd.n_points = len(cd.Y)

    pm = s.get("penalty_mask")
    if pm is not None:
        cd.penalty_mask = pm.bool().numpy().ravel()

    ni = s.get("needle_indices")
    if ni is not None and ni.numel() > 0:
        cd.needle_indices = ni.long().numpy().ravel()
        cd.n_needles = len(cd.needle_indices)

    (cd.activation_starts, cd.declared_samples, cd.n_init,
     cd.zoom_starts, cd.zoom_levels,
     cd.packet_starts) = load_activation_timeline(run_dir, snapshot)

    return cd


# ── plotting (mirrors interface/app.py ConvergencePlotFrame.update) ────────────

# Search-phase bands (shared with the GUI): gray over the init design, orange
# over each zoomed-in stretch. Zoom 0 is the full search box, so it is left
# unshaded; zoom M >= 1 (the M-th zoom-in) takes ZOOM_BAND_ALPHAS[M - 1], and
# levels past the end of the tuple — only reached when max_zooms was raised
# above the usual 3 — reuse its last entry.
INIT_BAND_COLOR, INIT_BAND_ALPHA = "0.6", 0.25
ZOOM_BAND_COLOR = "orange"
ZOOM_BAND_ALPHAS = (0.15, 0.45, 0.62, 0.72)


def _ordinal(k: int) -> str:
    """1 -> '1st', 2 -> '2nd', 3 -> '3rd', 11 -> '11th', …"""
    suffix = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"


def draw_search_bands(ax, n_points, n_init=0, zoom_starts=None,
                      zoom_levels=None) -> None:
    """Background bands marking each sample's search phase, shared with the
    GUI's convergence plot.

    The ``n_init`` initialization samples get a gray band. After that, each
    stretch an activation spends zoomed in gets an orange band shaded by depth:
    ``zoom_starts[j]`` is where stretch j begins and ``zoom_levels[j]`` its zoom
    level; a stretch runs to the next one's start, the last to ``n_points``.
    Zoom 0 searches the full box and stays unshaded — its stretches only mark
    where the zoomed stretch before them ends. Sample i spans [i - 0.5, i + 0.5],
    so band edges fall between samples. One legend entry per phase; bands sit
    at zorder 0, behind every mark."""
    n_init = min(max(int(n_init), 0), n_points)
    if n_init > 0:
        ax.axvspan(-0.5, n_init - 0.5, color=INIT_BAND_COLOR, alpha=INIT_BAND_ALPHA,
                   lw=0, zorder=0, label=f"initialization ({n_init} pts)")
    if zoom_starts is None or zoom_levels is None or len(zoom_starts) == 0:
        return
    starts = np.asarray(zoom_starts, dtype=int)
    ends = np.minimum(np.append(starts[1:], n_points), n_points)
    labeled: set[int] = set()
    for start, end, level in zip(starts, ends, np.asarray(zoom_levels, dtype=int)):
        level = int(level)
        if level < 1 or end <= start:
            continue
        label = None
        if level not in labeled:
            labeled.add(level)
            label = f"{_ordinal(level)} zoom"
        ax.axvspan(start - 0.5, end - 0.5, color=ZOOM_BAND_COLOR,
                   alpha=ZOOM_BAND_ALPHAS[min(level, len(ZOOM_BAND_ALPHAS)) - 1],
                   lw=0, zorder=0, label=label)


def draw_needle_leashes(ax, Y, needle_indices, declared_samples=None,
                        activation_starts=None) -> None:
    """Lollipop-and-leash needle markers, shared with the GUI's convergence plot.

    Needle k gets a numbered crimson star on its sampled point
    ``(needle_indices[k], Y[needle_indices[k]])`` and, when its declaration is
    known, a horizontal leash at that height out to a green diamond at
    ``declared_samples[k] + 0.5`` (+0.5: the declaration happens after the last
    sample of its line). The leash length is how long the point sat measured
    before the search confirmed it. The leash is lifted a few points so it stays
    visible where it coincides with the running-best envelope (the needle is
    usually its activation's best point). A needle declared in a later activation
    than the one that sampled it (a retro declaration on resume, or a rebuilt
    stack) gets a dashed leash and a hollow diamond."""
    if needle_indices is None or len(needle_indices) == 0:
        return
    from matplotlib.transforms import offset_copy

    lifted = offset_copy(ax.transData, fig=ax.figure, y=3, units="points")
    starts = (np.asarray(activation_starts) if activation_starts is not None
              else np.array([], dtype=int))
    n_decl = len(declared_samples) if declared_samples is not None else 0

    labeled: set[str] = set()

    def _label(name):
        if name in labeled:
            return None
        labeled.add(name)
        return name

    for k, ni in enumerate(needle_indices):
        if not 0 <= ni < len(Y):
            continue
        y = float(Y[ni])
        ax.scatter([ni], [y], marker="*", s=110, color="crimson", edgecolors="black",
                   linewidths=0.4, zorder=7, label=_label("needle point sampled"))
        ax.annotate(f"N{k + 1}", (ni, y), xytext=(0, 7), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color="crimson", zorder=8)
        if k >= n_decl:
            continue
        ds = int(declared_samples[k])
        retro = (np.searchsorted(starts, ds, side="right")
                 != np.searchsorted(starts, ni, side="right"))
        x_end = ds + 0.5
        ax.plot([ni, x_end], [y, y], transform=lifted, color="crimson", lw=1.1,
                ls="--" if retro else "-", alpha=0.8, zorder=6)
        ax.scatter([x_end], [y], transform=lifted, marker="D", s=26, zorder=7,
                   facecolors="none" if retro else "seagreen", edgecolors="seagreen",
                   linewidths=1.1,
                   label=_label("declared in a later activation" if retro
                                else "needle declared"))


def orient_packets(Y, packet_starts, penalty_mask=None, needle_indices=None):
    """Draw order for the convergence plot: each result packet with its
    higher-objective end last. Display only — returns new arrays, never touches
    the run.

    A packet is the run of samples one objective call returned: the rows from
    one ``packet_starts`` entry to the next, the last to the end of ``Y``. Its
    rows follow the measured line, so which end comes first depends only on the
    line's direction. A packet whose first Y is above its last Y is reversed,
    so the running best climbs toward each packet's end. Per-point attributes
    move with their points: ``penalty_mask`` is permuted and ``needle_indices``
    are remapped to where their points are drawn (out-of-range ones pass
    through). Everything placed on packet boundaries — bands, activation and
    zoom starts, needle-declaration samples — is unchanged, and rows before the
    first packet start stay where they are.

    Returns ``(Y, penalty_mask, needle_indices)``; the last two are None when
    not given."""
    Y = np.asarray(Y)
    n = len(Y)
    order = np.arange(n)
    starts = np.unique(np.asarray([] if packet_starts is None else packet_starts,
                                  dtype=int))
    starts = starts[(starts >= 0) & (starts < n)]
    for start, end in zip(starts, np.append(starts[1:], n)):
        if end - start >= 2 and Y[start] > Y[end - 1]:
            order[start:end] = order[start:end][::-1]

    pm = None
    if penalty_mask is not None:
        pm = np.asarray(penalty_mask)
        pm = pm[order] if len(pm) == n else pm.copy()
    ni = None
    if needle_indices is not None:
        drawn_at = np.empty(n, dtype=int)
        drawn_at[order] = np.arange(n)
        ni = np.asarray(needle_indices).astype(int, copy=True)
        ok = (ni >= 0) & (ni < n)
        ni[ok] = drawn_at[ni[ok]]
    return Y[order], pm, ni


def sample_axis_label(packet_starts) -> str:
    """The convergence plot's x label, noting when packets are drawn oriented."""
    if packet_starts is not None and len(packet_starts) > 0:
        return "Sample index (each packet drawn with its higher-Y end last)"
    return "Sample index"


def build_convergence_figure(cd: _ConvData, plt):
    """Draw the convergence plot for one run; returns the matplotlib Figure."""
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    # Each result packet drawn with its higher-Y end last (display only); the
    # needle stars move with their points.
    Y, _, needle_indices = orient_packets(cd.Y, cd.packet_starts,
                                          needle_indices=cd.needle_indices)
    idx = np.arange(len(Y))

    ax.scatter(idx, Y, s=10, alpha=0.65, color="steelblue", label="obs", zorder=2)

    # Search-phase bands: the init design in gray, then each zoomed-in stretch in
    # orange by depth (zoom 0 is the full search box, unshaded). Init is not a
    # search phase, so it gets no running-best envelope: activation 0 starts at
    # the first post-init sample.
    n_init = min(max(int(cd.n_init), 0), len(Y))
    draw_search_bands(ax, len(Y), n_init, cd.zoom_starts, cd.zoom_levels)

    # Running best, reset at every activation: each activation is a fresh ZoMBI
    # search phase, so the envelope accumulates only within the current phase
    # rather than over the whole run. Activation 0 starts at the first post-init
    # sample; later boundaries are the activations' first sample indices. Within
    # each [start, end) slice the running best is an independent cumulative max,
    # drawn as its own orange segment.
    act_starts = np.array([], dtype=int)
    if cd.activation_starts is not None and len(cd.activation_starts) > 0:
        act_starts = cd.activation_starts[
            (cd.activation_starts > n_init) & (cd.activation_starts < len(Y))
        ]
    bounds = np.concatenate(([n_init], act_starts))
    ends = np.concatenate((act_starts, [len(Y)]))
    labeled = False
    for start, end in zip(bounds, ends):
        if end <= start:
            continue
        seg_best = np.maximum.accumulate(Y[start:end])
        kw = dict(color="darkorange", lw=1.8, zorder=4)
        if not labeled:
            kw["label"] = "running best (reset per activation)"
            labeled = True
        ax.plot(idx[start:end], seg_best, **kw)

    draw_needle_leashes(ax, Y, needle_indices, cd.declared_samples, act_starts)

    ax.set_xlabel(sample_axis_label(cd.packet_starts))
    ax.set_ylabel("Objective Y")
    ax.set_title(f"{cd.run_id} — Convergence  "
                 f"(snap: {cd.snapshot},  {cd.n_points} pts, {cd.n_needles} needles)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def export_convergence(run_dir: Path, snapshot: str | None, out: Path | None,
                       show: bool, plt) -> Path:
    """Render and save one run's convergence PNG; returns the output path."""
    cd = load_conv_data(run_dir, snapshot)
    fig = build_convergence_figure(cd, plt)
    out = out or (run_dir / "convergence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"{cd.run_id}: {cd.n_points} pts, {cd.n_needles} needles "
          f"(snap {cd.snapshot})  ->  {out}")
    if not show:
        plt.close(fig)
    return out


# ── main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static-PNG export of a ZoMBI-Hop run's convergence plot "
                    "(the offline twin of the GUI's ConvergencePlotFrame)."
    )
    parser.add_argument("runs", nargs="+",
                        help="One or more run directories or bare run names.")
    parser.add_argument("--snapshot", default=None,
                        help="Snapshot to reconstruct up to (default: latest.txt). "
                             "Only used when a single run is given.")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (single run only; default: "
                             "<run_dir>/convergence.png).")
    parser.add_argument("--show", action="store_true",
                        help="Display each figure as well as saving it.")
    args = parser.parse_args()

    if len(args.runs) > 1 and (args.out or args.snapshot):
        parser.error("--out / --snapshot only apply when a single run is given.")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for run_arg in args.runs:
        run_dir = _resolve_run_dir(run_arg)
        out = Path(args.out) if args.out else None
        export_convergence(run_dir, args.snapshot, out, args.show, plt)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
