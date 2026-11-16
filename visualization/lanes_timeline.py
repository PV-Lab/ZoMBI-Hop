"""ONE convergence plot of a lanes hardware run: both lanes in a single axes, one block per dive.

A lanes run (``scripts/lanes_hw.py``, the default hardware optimizer) keeps its results in
``<run>/lanes/`` rather than in ZoMBIHop snapshots. This module reads that folder (read-only) and draws the
GUI's Convergence tab for it; ``visualization/render_lanes_convergence.py`` renders the same figure to PNG.

    rows.jsonl      one 'row' record per measured row, in arrival order: qid, wid, lane, act, zoom_sent,
                    engine_rid, pid, x_actual, x_expected, y_opt ('raw_rows' after a rows error)
    journal.jsonl   'start' per process start, 'issue' per recipe id (wid -> lane / act / zoom),
                    'decision' per lane decision (the lane's act / zoom after it), 'ingested' per packet
    init.pt         the initial design (X_actual, X_expected, Y, ...); else ``<run>/init_state.json``
    engine_ckpt.pt  engine checkpoint: zones (every needle / exclusion zone of this process), events (last
                    500: activation / needle / zone / dive_end ...), lane_state, Y
    engine/run_*/   the engine's DataHandler snapshots; the latest carries the penalty mask of every row
    <run>/run.log   '[lanes] <lane> act <A>: needle zoom 2' ... lines: the fallback for needles and dive
                    ends when the checkpoint is missing or predates a restart
    <run>/init_state.json  the axial initial design (queue.traces / status / data, optimizing_dims): the
                    whole plot while the run is still initialising (``run_kind`` 'init')

What a run folder gets (``run_kind``): 'lanes' (lanes/rows.jsonl or lanes/init.pt) = the dive plot below;
'init' (init_state.json, no lanes data, no top-level snapshots: a run still in its axial initialization)
= the INIT VIEW: every measured trace's result packet in arrival order, its points by the trace's own
dimension's fraction (0 -> 1 left to right), coloured and labelled by the trace's real dimension
(``optimizing_dims[component]``), a running best, traces that have not reported yet (sent / re-queued /
queued) as hatched slots on the right and in the legend; 'legacy_init' (init_state.json + top-level snapshots, a
snapshot-optimizer run past init) = the same init view for the PNG tool only (the GUI keeps its snapshot
plot for it); None = nothing to draw.

Layout (the user's rules for this plot):

* x: the initialization rows first (grey band), then one BLOCK per dive = one activation of one lane
  (zoom 0 -> 1 -> 2), blocks in the order the activations started (the engine's activation counter;
  a restarted ZoMBI process starts its own count, so blocks sort by (process start, activation)). Inside
  a block its rows in arrival order, grouped by the zoom each line was SENT at (a stable sort, so a late
  zoom-0 result stays in zoom 0's band). Labels F1 / B1 / F2 ... = the front / back lane's #-th dive.
* the init band is the init data the engine used (lanes/init.pt), followed by late init results: rows of
  packets the router gave disposition 'init' (or class 'legacy', or the engine classed 'init'). Packets
  the journal only logs (disposition 'log') carry no rows and are not drawn. Rows no dive owns otherwise
  (unattributed 'shared' data) go to a trailing 'other' block.
* a dive that has started (activation / decision record) but has no rows yet is a narrow hatched block
  with a dashed outline, labelled like the others ('B2', start time, '…' while it runs). A running dive
  too narrow for its 'F4 …' label on this canvas is widened with blank room after its rows.
* zoom bands (orange, by SENT zoom), running best reset per dive, needle star (point sampled) leashed to
  a diamond (declared), as the snapshot plot draws them (``visualization/plot_convergence.py``).
* per (dive, SENT zoom) a bar at the MEASURED top-m mean (m = 8; 11 in 10-D: ``top_m_for_dim``). DIP
  RULE: a zoom's bar is compared with the NEAREST EARLIER zoom of the same dive that has rows (a finite
  top-m; zoom 1 when it has rows, else zoom 0): black when within 0.01 of it or higher, red when more
  than 0.01 below. A dive's first zoom with rows never flags. No noise-free level is drawn: on hardware
  the landscape is unknown.
* a dive that ended without a needle: grey (dashed running best by end reason, grey strip block, a tick
  where it ended). The dive still running is drawn like a needle dive.

Every reader tolerates missing files and a partially written last line. Nothing here writes anything.
CPU only: torch files load with ``map_location='cpu'``; importing this module imports neither torch nor
CUDA (torch is imported on the first .pt read), and drawing imports plot_convergence without src's
import-time CUDA setup (``_plot_convergence``). ``zoom_topm_levels`` is pure numpy (unit-tested).
"""
from __future__ import annotations

import bisect
import json
import math
import re
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["LANES", "LANE_C", "DIP_EPS", "END_STYLES", "TRACE_C", "LanesRun", "Layout", "is_lanes_run",
           "run_kind", "has_top_level_snapshots", "lanes_signature", "read_jsonl", "top_m_for_dim",
           "zoom_topm_levels", "load_init_state", "load_lanes_run", "dive_layout", "draw_lanes_convergence"]

LANES = ("front", "back")
LANE_LETTER = {"front": "F", "back": "B"}
# lane identity (as benchmarks/zoom_fix/lanes/report/combined_plot.py): violet / dark aqua, hues the app's
# plot does not use (steelblue / grey points, orange bands and running best, crimson needles, green diamonds)
LANE_C = {"front": "#d6e8f8", "back": "#a9cbec"}       # user, 14 Nov: subtle blues, pale / a bit darker
LANE_TXT = {"front": "#1f3a5f", "back": "#1f3a5f"}     # dark navy label text on the pale strip blocks
LANE_EDGE = {"front": "#8db4dc", "back": "#4f86c0"}    # separators / dashed outlines (a pale fill would vanish)
INIT_C = "#d6d6d6"
NO_NEEDLE_C = "#8f8e8a"            # strip fill of a dive that ended without a needle
OTHER_C = "#ecebe7"
INK2 = "#52514e"
GATE_COLOR = "dimgrey"             # running best of a dive that ended without a needle
LEVEL_C, DIP_C = "black", "#d62728"
DIP_EPS = 0.01                     # the user's tolerance: a zoom may sit at most 0.01 below the previous one
_DIP_SLACK = 1e-9                  # float slack so a step of exactly 0.01 stays black
STRIP_PT, STRIP_GAP_PT = 20.0, 2.0  # the dive strip: height and gap above the axes frame (points)
SLOT_PT = 34.0                     # width of an empty dive / pending init trace slot (points, label fits)
NARROW_IN = 10.0                   # figure width (in) below which the title's reading note is the short one
                                   # (the GUI tab opens at 9 in; the PNG tool draws at 11 in)
# init traces, by trace index (the entity, never the arrival rank): the reference categorical hues without
# orange and yellow (both read as the darkorange running best), violet (= the front lane) last: six hues, so
# every trace of a 6-D run has its own. A marker per trace is the second cue; traces past the sixth (10-D)
# reuse the hues with a dark marker edge as a third cue (``_trace_style``)
TRACE_C = ("#2a78d6", "#1baf7a", "#e87ba4", "#008300", "#e34948", "#6250d6")
TRACE_MK = ("o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "p")
INK = "#2b2a28"

# reason a dive ended -> (legend label, dash pattern); every no-needle ending is grey
END_STYLES = {
    "covered_by_other": ("no needle: box covered by the other lane", (0, (4, 2))),
    "overlaps_other_box": ("no needle: box overlaps the other lane's", (0, (1.5, 1.5))),
    "no_own_unpen": ("no needle: no unpenalised own rows left", (0, (6, 1.5, 1.5, 1.5, 1.5, 1.5))),
    "zooms_exhausted": ("no needle: zooms exhausted (exclusion zone)", (0, (8, 2))),
    "activation_budget": ("no needle: activation budget (exclusion zone)", (0, (8, 2, 2, 2))),
    "no_results": ("no needle: no results in the box (zone)", (0, (2, 1, 6, 1))),
    "needle_no_points": ("no needle: no zoom-2 points (zone)", (0, (5, 1.5, 1, 1.5))),
    "floor_blocks_zoom": ("no needle: width floor blocks the zoom (zone)", (0, (3, 1, 1, 1))),
    "no_box_rows": ("no needle: no rows in the box", (0, (2, 2))),
    "gate": ("no needle: zoom gate fired (exclusion zone)", (0, (6, 2, 1.5, 2))),
    "restart": ("no needle: cut off by a ZoMBI restart", (0, (1, 1.6))),
    "no_needle": ("no needle: dive ended without one", (0, (4, 2))),
}
TICK_LABEL = "where a no-needle dive ended"


# ── small readers ───────────────────────────────────────────────────────────────────────────────

def _i(v, default: int = -1) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _f(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _np(t) -> np.ndarray:
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.asarray(t)


def _read_bytes(path: Path) -> Optional[bytes]:
    """File bytes, retried briefly (the run folder lives in Dropbox, which can hold a lock); None if absent."""
    for attempt in range(3):
        try:
            return Path(path).read_bytes()
        except FileNotFoundError:
            return None
        except OSError:
            time.sleep(0.05 * (attempt + 1))
    return None


def _read_jsonl(path) -> Tuple[List[dict], int, bool]:
    """``read_jsonl`` + whether the LAST non-blank line was the (or one) unreadable one."""
    raw = _read_bytes(Path(path))
    if raw is None:
        return [], 0, False
    out: List[dict] = []
    bad = 0
    last_bad = False
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            rec = None
        last_bad = not isinstance(rec, dict)
        if last_bad:
            bad += 1
        else:
            out.append(rec)
    return out, bad, last_bad


def read_jsonl(path) -> Tuple[List[dict], int]:
    """(records, n_bad): every parseable JSON-object line; a torn or partially written line (the writer
    appends while we read) is skipped and counted. A missing file is ([], 0)."""
    recs, bad, _ = _read_jsonl(path)
    return recs, bad


def _torch_load(path: Path):
    """torch.load on CPU (our own run files: full unpickling), retried once: engine_ckpt.pt is rewritten in
    place after every result, so a read can catch it half written."""
    import torch
    err: Optional[BaseException] = None
    for attempt in range(2):
        try:
            try:
                return torch.load(str(path), map_location="cpu", weights_only=False)
            except TypeError:                                   # torch without weights_only
                return torch.load(str(path), map_location="cpu")
        except Exception as e:                                  # noqa: BLE001 - reported by the caller
            err = e
            time.sleep(0.15)
    raise err  # type: ignore[misc]


_CACHE: Dict[str, tuple] = {}        # path -> ((size, mtime_ns), parsed value): GUI redraws re-read nothing


def _cached(path: Path, parse):
    """``parse(path)`` memoised on the file's size + mtime (errors are not cached)."""
    st = Path(path).stat()
    sig = (st.st_size, st.st_mtime_ns)
    hit = _CACHE.get(str(path))
    if hit is not None and hit[0] == sig:
        return hit[1]
    val = parse(Path(path))
    if len(_CACHE) > 32:
        _CACHE.clear()
    _CACHE[str(path)] = (sig, val)
    return val


def _ckpt_essentials(path: Path) -> dict:
    """What the plot uses from engine_ckpt.pt (the rest of it - decisions, router - is dropped)."""
    ck = _torch_load(path)
    if not isinstance(ck, dict):
        raise TypeError(f"engine_ckpt.pt holds a {type(ck).__name__}, not a dict")
    Y = ck.get("Y")
    return dict(t=ck.get("t"), Y=None if Y is None else _np(Y).astype(float).reshape(-1),
                events=[e for e in (ck.get("events") or []) if isinstance(e, dict)],
                zones=[z for z in (ck.get("zones") or []) if isinstance(z, dict)],
                lane_state=ck.get("lane_state") or {})


def is_lanes_run(run_dir) -> bool:
    """True when ``run_dir`` holds a lanes run: ``lanes/rows.jsonl`` (results) or ``lanes/init.pt`` (the
    lanes phase started, no result yet). Legacy ZoMBIHop runs have no ``lanes/`` folder."""
    if run_dir is None:
        return False
    d = Path(run_dir) / "lanes"
    return (d / "rows.jsonl").is_file() or (d / "init.pt").is_file()


def has_top_level_snapshots(run_dir) -> bool:
    """True when ``run_dir`` has ZoMBIHop snapshots of its own (``latest.txt`` or a ``snapshots/<name>/``
    folder): a snapshot-optimizer run. A lanes run keeps its engine snapshots under lanes/engine/."""
    if run_dir is None:
        return False
    d = Path(run_dir)
    if (d / "latest.txt").is_file():
        return True
    try:
        return any(p.is_dir() for p in (d / "snapshots").iterdir())
    except OSError:
        return False


def run_kind(run_dir) -> Optional[str]:
    """What this module draws for ``run_dir``: 'lanes' (a lanes run: ``is_lanes_run``), 'init' (an
    ``init_state.json``, no lanes data, no top-level snapshots: a run still in its axial initialization),
    'legacy_init' (``init_state.json`` + top-level snapshots: a snapshot-optimizer run past its init; the
    GUI keeps the snapshot plot for it, the PNG tool draws its init view) or None (nothing to draw)."""
    if run_dir is None:
        return None
    if is_lanes_run(run_dir):
        return "lanes"
    if (Path(run_dir) / "init_state.json").is_file():
        return "legacy_init" if has_top_level_snapshots(run_dir) else "init"
    return None


_SIG_FILES = ("lanes/rows.jsonl", "lanes/engine_ckpt.pt", "lanes/init.pt", "lanes/journal.jsonl",
              "init_state.json")


def lanes_signature(run_dir) -> Optional[tuple]:
    """Size / mtime of the files the plot reads that change with new results (rows, checkpoint, init, the
    journal's activations, init_state.json during the initialization); None for a run the GUI does not draw
    with this module (``run_kind`` not 'lanes' / 'init'). A changed signature means the plot is stale."""
    if run_kind(run_dir) not in ("lanes", "init"):
        return None
    d = Path(run_dir)
    sig = []
    try:                                                    # side files of a blocked init_state.json save
        side = sorted(p.name for p in d.glob("init_state.json*.tmp"))
    except OSError:
        side = []
    for name in (*_SIG_FILES, *side):
        try:
            st = (d / name).stat()
            sig.append((name, st.st_size, st.st_mtime_ns))
        except OSError:
            sig.append((name, None, None))
    return tuple(sig)


def top_m_for_dim(d: int) -> int:
    """m of the measured top-m level: the box rule's top_m_points (8; raised to d + 1 above 7 dims, i.e.
    11 in 10-D, as ``src.hparams.apply_dim_rules``)."""
    return max(8, int(d) + 1) if d and int(d) > 0 else 8


def _norm_reason(r) -> str:
    r = str(r or "no_needle")
    return r[len("dive_end:"):] if r.startswith("dive_end:") else r


# ── the pure statistic ──────────────────────────────────────────────────────────────────────────

def zoom_topm_levels(y, act_key: Sequence, zoom, m: int = 8, eps: float = DIP_EPS) -> List[dict]:
    """Per (activation, SENT zoom): the MEASURED top-m level and its dip flag. Pure numpy.

    ``y``: (n,) measured values; ``act_key``: (n,) hashable activation key per row (None = not part of a
    dive, skipped); ``zoom``: (n,) the zoom each row's line was SENT at (< 0 = skipped). The level of a group
    is the mean of its ``m`` highest finite values (all of them when it has fewer). ``dip`` is True when this
    level is MORE than ``eps`` below the level of the NEAREST EARLIER zoom of the same activation that has
    rows with a finite level (zoom k-1, else k-2, ...); within ``eps`` or higher is not a dip, and an
    activation's first zoom with rows never flags. Order: activations by first appearance, zooms ascending.
    Each record: key, zoom, n (rows), m (values averaged), level, prev (the reference level or None),
    prev_zoom (its zoom or None), dip, rows (indices into ``y``, ascending)."""
    y = np.asarray(y, dtype=float).reshape(-1)
    zoom = np.asarray(zoom).astype(int).reshape(-1)
    keys = list(act_key)
    if not (len(keys) == len(y) == len(zoom)):
        raise ValueError(f"zoom_topm_levels: lengths differ (y {len(y)}, act_key {len(keys)}, zoom {len(zoom)})")
    groups: Dict[tuple, List[int]] = {}
    first: Dict[Any, int] = {}
    for i, (k, z) in enumerate(zip(keys, zoom)):
        if k is None or z < 0:
            continue
        first.setdefault(k, i)
        groups.setdefault((k, int(z)), []).append(i)
    level: Dict[tuple, Tuple[float, int]] = {}
    for kz, rows in groups.items():
        v = y[rows]
        v = np.sort(v[np.isfinite(v)])[::-1][:max(1, int(m))]
        level[kz] = (float(v.mean()) if v.size else float("nan"), int(v.size))
    out = []
    ref: Dict[Any, Tuple[float, int]] = {}      # activation -> (level, zoom) of its latest finite zoom so far
    for kz in sorted(groups, key=lambda kz: (first[kz[0]], kz[1])):
        k, z = kz
        lev, used = level[kz]
        prev, prev_zoom = ref.get(k, (None, None))
        dip = (prev is not None and np.isfinite(lev) and lev < prev - float(eps) - _DIP_SLACK)
        if np.isfinite(lev):
            ref[k] = (lev, z)
        out.append(dict(key=k, zoom=z, n=len(groups[kz]), m=used, level=lev, prev=prev, prev_zoom=prev_zoom,
                        dip=bool(dip), rows=np.asarray(groups[kz], dtype=int)))
    return out


# ── loading a run ───────────────────────────────────────────────────────────────────────────────

@dataclass
class LanesRun:
    """Everything the plot needs from a lanes run folder. Row arrays are in arrival (rows.jsonl) order;
    the GLOBAL row index (needle picks, penalty mask, 'n_rows' of events) is ``n_init + i``."""
    run_dir: Path
    run_id: str
    d: int = 0
    y_init: np.ndarray = field(default_factory=lambda: np.zeros(0))
    init_source: Optional[str] = None
    init_sizes: Optional[List[int]] = None        # init result packets (init_state.json), or None
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    lane: List[Optional[str]] = field(default_factory=list)
    act: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    zoom: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))      # SENT zoom
    session: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))   # ZoMBI process start index
    qid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    wid: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))                    # arrival (wall clock)
    valid: Optional[np.ndarray] = None            # (n_init + n,) unpenalised, from the engine snapshot
    acts: Dict[tuple, dict] = field(default_factory=dict)       # (session, act) -> lane, t_start, reason ...
    needles: List[dict] = field(default_factory=list)           # declaration order
    n_sessions: int = 1
    warnings: List[str] = field(default_factory=list)
    sources: Dict[str, Any] = field(default_factory=dict)
    kind: Optional[str] = "lanes"                 # run_kind(): 'lanes' / 'init' / 'legacy_init' / None
    late_init: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))   # (n,) row -> init band
    init_state: Optional[dict] = None             # load_init_state() (init view; None in a lanes run)
    optimizer: Optional[str] = None               # run.log '[Main] Optimizer: <name>' (init view wording)

    @property
    def n_init(self) -> int:
        return int(len(self.y_init))


INIT_STATE = "init_state.json"


def _init_state_files(run_dir: Path) -> List[Path]:
    """init_state.json and the side files a save leaves when Dropbox blocks the replace
    (``scripts/run_zombi_main.py:_save_init_state``)."""
    p = Path(run_dir) / INIT_STATE
    try:
        side = sorted(Path(run_dir).glob(INIT_STATE + "*.tmp"))
    except OSError:
        side = []
    return [p, *side]


def _read_init_state_json(run_dir: Path, warn: List[str]) -> Optional[dict]:
    """The init progress the run itself resumes from (``run_zombi_main._load_init_state``): init_state.json,
    or a side file holding more progress. Each file is retried briefly when it does not parse (it is
    rewritten while the run initialises). None when there is none / none parses."""
    best, best_key, err = None, None, None
    for f in _init_state_files(run_dir):
        st = None
        for attempt in range(3):
            raw = _read_bytes(f)
            if raw is None:
                if f.is_file():                                  # there, but every read failed (a lock)
                    err = OSError(f"{f.name} could not be read (locked?)")
                break
            try:
                st = json.loads(raw.decode("utf-8"))
                break
            except (UnicodeDecodeError, ValueError) as e:
                err = e
                time.sleep(0.1 * (attempt + 1))
        if not isinstance(st, dict):
            continue
        q = st.get("queue") if isinstance(st.get("queue"), dict) else {}
        key = (len(q.get("writes") or []) + len(q.get("result_times") or []), f.name == INIT_STATE)
        if best_key is None or key > best_key:
            best, best_key = st, key
    if best is None and err is not None:
        warn.append(f"init_state.json unreadable ({err!r})")
    return best


def load_init_state(run_dir, warn: Optional[List[str]] = None) -> Optional[dict]:
    """The axial initial design of ``<run>/init_state.json`` (read-only; None when absent or unreadable,
    recorded in ``warn``). Keys:

    y, trace          every measured value in arrival order (result blocks sorted by their time ``t``, the
                      order ``InitQueue.measured`` stacks them into the init data) and its trace index
    own               per value: the fraction of its trace's OWN optimizing dim in the measured x (NaN when
                      unknown); the init view sorts each trace's points by it (0 -> 1 left to right)
    sizes, block_t, block_trace   per result block (one objective packet)
    traces            per trace i: component (optimizing-dim index), dim (the REAL dimension,
                      ``optimizing_dims[component]``, None without optimizing_dims), label ('dim 8'; else
                      'component 4'), status (queued / sent / done / abandoned), sends, requeued (queued again
                      after a send: presumed lost), n_points, last_write (write number of its latest send, or None)
    complete, n_traces, d, first_write_t, points_per_trace, design"""
    warn = [] if warn is None else warn
    st = _read_init_state_json(Path(run_dir), warn)
    if st is None:
        return None
    try:
        q = st.get("queue") if isinstance(st.get("queue"), dict) else {}
        data = list(q.get("data") or [])
        tdefs = list(q.get("traces") or [])
        status = list(q.get("status") or [])
        sent_in = list(q.get("sent_in") or [])
        sends = list(q.get("sends") or [])
        opt = st.get("optimizing_dims")
        comps = [_i(tdefs[i].get("component")) if i < len(tdefs) and isinstance(tdefs[i], dict) else -1
                 for i in range(len(data))]
        blocks = [(i, b) for i, tr in enumerate(data) for b in (tr or [])
                  if isinstance(b, dict) and len(b.get("y") or [])]
        blocks.sort(key=lambda ib: _f(ib[1].get("t")))               # stable: InitQueue.measured's order
        y = [_f(v) for _, b in blocks for v in b["y"]]
        trace = [i for i, b in blocks for _ in b["y"]]
        own = []
        for i, b in blocks:
            xs = b.get("x") if isinstance(b.get("x"), list) else []
            c = comps[i]
            for k in range(len(b["y"])):
                xk = xs[k] if k < len(xs) else None
                own.append(_f(xk[c]) if isinstance(xk, (list, tuple)) and 0 <= c < len(xk) else float("nan"))
        d = 0
        for _, b in blocks:
            x = b.get("x")
            if x and isinstance(x[0], (list, tuple)):
                d = len(x[0])
                break
        if not d and tdefs and isinstance(tdefs[0], dict):
            d = len(tdefs[0].get("start") or [])
        traces = []
        for i in range(max(len(tdefs), len(data), len(status))):
            td = tdefs[i] if i < len(tdefs) and isinstance(tdefs[i], dict) else {}
            comp = _i(td.get("component"))
            dim = _i(opt[comp], None) if isinstance(opt, list) and 0 <= comp < len(opt) else None
            # without optimizing_dims the component is an optimizer index, never shown as a real dimension
            label = f"dim {dim}" if dim is not None else (f"component {comp}" if comp >= 0 else f"trace {i + 1}")
            n_pts = sum(len(b.get("y") or []) for b in ((data[i] or []) if i < len(data) else [])
                        if isinstance(b, dict))
            si = [w for w in (sent_in[i] if i < len(sent_in) and isinstance(sent_in[i], list) else [])
                  if isinstance(w, (int, float))]
            stat = str(status[i]) if i < len(status) else ("done" if n_pts else "unknown")
            n_sends = max(_i(sends[i], 0) if i < len(sends) else 0, len(si))
            traces.append(dict(i=i, component=comp, dim=dim, label=label, status=stat, sends=int(n_sends),
                               requeued=stat == "queued" and n_sends > 0,
                               n_points=int(n_pts), last_write=int(max(si)) if si else None))
    except Exception as e:  # noqa: BLE001
        warn.append(f"init_state.json unreadable ({e!r})")
        return None
    return dict(y=np.asarray(y, dtype=float), trace=np.asarray(trace, dtype=int),
                own=np.asarray(own, dtype=float), sizes=[len(b["y"]) for _, b in blocks], block_t=[_f(b.get("t")) for _, b in blocks],
                block_trace=[i for i, _ in blocks], traces=traces, complete=bool(st.get("complete")),
                n_traces=len(traces), d=int(d), first_write_t=_f(q.get("first_write_t")),
                points_per_trace=_i(q.get("points_per_trace")), design=st.get("design"))


def _load_init(run_dir: Path, lanes_dir: Path, warn: List[str]
               ) -> Tuple[np.ndarray, int, Optional[str], Optional[dict]]:
    """(y_init, d, source, init_state): lanes/init.pt (the init data the lanes engine used) when it is
    readable, else init_state.json (whose parsed state is returned too)."""
    p = lanes_dir / "init.pt"
    if p.is_file():
        def parse(path):
            st = _torch_load(path)
            xa = st.get("X_actual")
            d = int(_np(xa).shape[1]) if xa is not None and _np(xa).ndim == 2 else 0
            return _np(st["Y"]).astype(float).reshape(-1), d
        try:
            y, d = _cached(p, parse)
            return y.copy(), d, "lanes/init.pt", None
        except Exception as e:  # noqa: BLE001
            warn.append(f"lanes/init.pt unreadable ({e!r}); init from init_state.json")
    if (run_dir / INIT_STATE).is_file():
        S = load_init_state(run_dir, warn)
        if S is not None:
            return S["y"].copy(), S["d"], INIT_STATE, S
    return np.zeros(0), 0, None, None


def _init_sizes(run_dir: Path, n_init: int, S: Optional[dict] = None) -> Optional[List[int]]:
    """Init result-packet sizes in stacking order (``plot_convergence.init_packet_sizes``' rule: the
    init_state.json blocks by arrival time), or None when they do not add up to ``n_init``."""
    if n_init <= 0:
        return None
    if S is None and (Path(run_dir) / INIT_STATE).is_file():
        S = load_init_state(run_dir, [])                     # display nicety: its warnings are not shown
    sizes = list(S["sizes"]) if S else []
    return sizes if sizes and sum(sizes) == n_init else None


def _load_journal(lanes_dir: Path) -> dict:
    recs, bad = read_jsonl(lanes_dir / "journal.jsonl")
    J = dict(starts=[], issues={}, decisions=[], ingested={}, ingested_cls={}, routes={}, resumes=[], bad=bad,
             n=len(recs))
    for r in recs:
        ev, t = r.get("ev"), _f(r.get("t"))
        if ev == "start":
            J["starts"].append(dict(t=t, n_init=_i(r.get("n_init")), n_prior=_i(r.get("n_prior_rows")),
                                    epoch=_f(r.get("epoch"))))
        elif ev == "issue":
            wid = _i(r.get("wid"))
            if wid >= 0:
                J["issues"].setdefault(wid, (t, r.get("lane"), _i(r.get("act")), _i(r.get("zoom"))))
        elif ev == "decision" and r.get("act") is not None and r.get("lane") in LANES:
            J["decisions"].append((t, r.get("lane"), _i(r.get("act")), _i(r.get("zoom"))))
        elif ev == "ingested":
            J["ingested"][_i(r.get("qid"))] = (r.get("lane"), _i(r.get("act")), _i(r.get("zoom")))
            J["ingested_cls"][_i(r.get("qid"))] = r.get("cls")
        elif ev == "route":                                     # router class + disposition per packet
            J["routes"][_i(r.get("qid"))] = (r.get("cls"), r.get("disposition"))
        elif ev == "resume_dives":                              # a dive-exact resume: these acts CONTINUE
            J["resumes"].append(dict(t=t, lanes=r.get("lanes") if isinstance(r.get("lanes"), dict) else {}))
    return J


def _init_packet_qids(J: dict) -> set:
    """Packets whose rows belong to the init band: router disposition 'init' (a late init result), class
    'legacy' (an init-protocol packet ingested as shared data) or an engine packet classed 'init'. Packets
    the journal only logs (disposition 'log') have no rows, so they never reach the plot."""
    q = {qid for qid, (cls, disp) in J["routes"].items() if disp == "init" or (cls == "legacy" and disp != "log")}
    q |= {qid for qid, cls in J["ingested_cls"].items() if cls == "init"}
    q.discard(-1)
    return q


def _load_rows(lanes_dir: Path, ingested: dict) -> Tuple[dict, int, int, bool]:
    recs, bad, last_bad = _read_jsonl(lanes_dir / "rows.jsonl")
    cols: Dict[str, list] = {k: [] for k in ("y", "lane", "act", "zoom", "qid", "wid", "t")}
    d = 0

    def add(y, lane, act, zoom, qid, wid, t):
        if lane not in LANES or act < 0 or zoom < 0:
            lane, act, zoom = None, -1, -1                       # not attributable to a dive
        for k, v in zip(("y", "lane", "act", "zoom", "qid", "wid", "t"), (y, lane, act, zoom, qid, wid, t)):
            cols[k].append(v)

    for r in recs:
        ev = r.get("ev")
        if ev == "row":
            add(_f(r.get("y_opt")), r.get("lane"), _i(r.get("act")), _i(r.get("zoom_sent")), _i(r.get("qid")),
                _i(r.get("wid")), _f(r.get("t")))
            if not d and isinstance(r.get("x_actual"), list):
                d = len(r["x_actual"])
        elif ev == "raw_rows":                                   # lanes_hw._save_rows' fallback record
            qid = _i(r.get("qid"))
            lane, act, zoom = ingested.get(qid, (None, -1, -1))
            for v in r.get("y_opt") or []:
                add(_f(v), lane, _i(act), _i(zoom), qid, _i(r.get("wid")), _f(r.get("t")))
    return cols, bad, d, last_bad


_LOG_OPT = re.compile(rb"\[Main\] Optimizer:\s*(\S+)")


def _run_optimizer(run_dir: Path) -> Optional[str]:
    """The optimizer run.log names ('[Main] Optimizer: lanes', scripts/main.py; the last launch wins), or None."""
    raw = _read_bytes(Path(run_dir) / "run.log")
    found = _LOG_OPT.findall(raw) if raw else []
    return found[-1].decode("utf-8", errors="replace") if found else None


_LOG_EV = re.compile(r"\[lanes\]\s+(front|back)\s+act\s+(\d+):\s+(\w+)(?:\s+zoom\s+(\d+))?(.*)$")
_LOG_KV = re.compile(r"(\w+)=(\S+)")
_LOG_START = "STARTING LANES OPTIMIZATION"


def _parse_run_log(run_dir: Path) -> List[List[dict]]:
    """'[lanes] <lane> act <A>: <kind> [zoom k] [key=value ...]' events, one list per lanes process start
    (the 'STARTING LANES OPTIMIZATION' banner)."""
    raw = _read_bytes(run_dir / "run.log")
    if raw is None:
        return []
    sessions: List[List[dict]] = []
    for i, line in enumerate(raw.decode("utf-8", errors="replace").splitlines()):
        if _LOG_START in line:
            sessions.append([])
            continue
        if "[lanes]" not in line:
            continue
        m = _LOG_EV.search(line)
        if m is None:
            continue
        if not sessions:
            sessions.append([])
        kv = dict(_LOG_KV.findall(m.group(5) or ""))
        sessions[-1].append(dict(lane=m.group(1), act=int(m.group(2)), kind=m.group(3),
                                 zoom=_i(m.group(4)) if m.group(4) is not None else None, kv=kv, line=i))
    return sessions


def _engine_valid_mask(lanes_dir: Path, warn: List[str]) -> Optional[np.ndarray]:
    """Penalty mask (True = unpenalised) of every engine row, from the newest engine snapshot."""
    best = None
    for lt in (lanes_dir / "engine").glob("run_*/latest.txt"):
        try:
            mt = lt.stat().st_mtime
        except OSError:
            continue
        if best is None or mt > best[0]:
            best = (mt, lt)
    if best is None:
        return None
    try:
        name = best[1].read_text(encoding="utf-8").strip()
        delta = best[1].parent / "snapshots" / name / "delta.pt"
        if not delta.is_file():
            return None

        def parse(path):
            pm = _torch_load(path).get("penalty_mask")
            return None if pm is None else _np(pm).astype(bool).reshape(-1)
        return _cached(delta, parse)
    except Exception as e:  # noqa: BLE001
        warn.append(f"engine snapshot penalty mask unreadable ({e!r})")
        return None


def load_lanes_run(run_dir) -> LanesRun:
    """Read a lanes run folder (read-only; see the module doc). Missing / unreadable files only lose what
    they carry (recorded in ``warnings``); a partially written last line is skipped."""
    run_dir = Path(run_dir)
    lanes_dir = run_dir / "lanes"
    R = LanesRun(run_dir=run_dir, run_id=run_dir.name, kind=run_kind(run_dir))
    warn = R.warnings

    R.y_init, d_init, R.init_source, S = _load_init(run_dir, lanes_dir, warn)
    R.init_sizes = _init_sizes(run_dir, R.n_init, S) if R.n_init else None
    if R.kind in ("init", "legacy_init"):
        R.init_state = S                                     # the init view's traces (None: unreadable)
        R.optimizer = _run_optimizer(run_dir)

    J = _load_journal(lanes_dir)
    cols, bad_rows, d_rows, last_bad = _load_rows(lanes_dir, J["ingested"])
    R.d = int(d_init or d_rows or 0)
    n = len(cols["y"])
    R.y = np.asarray(cols["y"], dtype=float)
    R.lane = list(cols["lane"])
    R.act = np.asarray(cols["act"], dtype=int).reshape(-1)
    R.zoom = np.asarray(cols["zoom"], dtype=int).reshape(-1)
    R.qid = np.asarray(cols["qid"], dtype=int).reshape(-1)
    R.wid = np.asarray(cols["wid"], dtype=int).reshape(-1)
    R.t = np.asarray(cols["t"], dtype=float).reshape(-1)
    # late init results (router disposition 'init' ...): the init band, never a dive
    init_q = _init_packet_qids(J)
    R.late_init = np.asarray([int(q) in init_q for q in R.qid], dtype=bool).reshape(-1)
    for i in np.flatnonzero(R.late_init):
        R.lane[i], R.act[i], R.zoom[i] = None, -1, -1
    R.sources.update(rows=n, rows_bad_lines=bad_rows, journal_records=J["n"], journal_bad_lines=J["bad"],
                     init=R.init_source, late_init_rows=int(R.late_init.sum()))
    if bad_rows == 1 and last_bad:
        warn.append("rows.jsonl: 1 unreadable line skipped (a partial last line while writing)")
    elif bad_rows:
        warn.append(f"rows.jsonl: {bad_rows} unreadable line(s) skipped (a corrupt file?)")

    # ── process starts (a restarted ZoMBI process numbers its activations from 0 again) ──
    starts_t = sorted(s["t"] for s in J["starts"] if np.isfinite(s["t"]))
    R.n_sessions = max(1, len(starts_t))

    def session_of(t: float) -> Optional[int]:
        if not starts_t or not np.isfinite(t):
            return None
        return max(0, bisect.bisect_right(starts_t, t) - 1)

    # a dive-exact resume (journal 'resume_dives', scripts/lanes_hw.py) CONTINUES the listed activations: the
    # new process's (session, act) is the previous process's dive, not a new one
    cont: Dict[tuple, tuple] = {}
    for rz in J["resumes"]:
        s_new = session_of(rz["t"])
        if s_new is None or s_new < 1:
            continue
        for _lane, st in rz["lanes"].items():
            a = _i((st or {}).get("act"))
            if a >= 0:
                cont[(s_new, a)] = (s_new - 1, a)

    def canon(s, a):
        k = (int(s), int(a))
        while k in cont:
            k = cont[k]
        return k

    last_sess: Dict[tuple, int] = {}                 # canonical dive -> the last process it ran in
    for k_new in cont:
        c = canon(*k_new)
        last_sess[c] = max(last_sess.get(c, c[0]), k_new[0])

    sess = np.zeros(n, dtype=int)
    for i in range(n):
        iss = J["issues"].get(int(R.wid[i]))
        s = session_of(iss[0]) if iss is not None else None      # the line's issue: late results stay put
        if s is None:
            s = session_of(R.t[i])
        sess[i] = 0 if s is None else s
        if R.lane[i] is not None:
            sess[i] = canon(sess[i], int(R.act[i]))[0]
    R.session = sess
    epoch = None
    for s in J["starts"]:
        if np.isfinite(s.get("epoch", float("nan"))):
            epoch = s["epoch"]
    if epoch is None:
        try:
            epoch = _f(json.loads((lanes_dir / "meta.json").read_text(encoding="utf-8")).get("epoch"))
        except Exception:  # noqa: BLE001
            epoch = None

    # ── activations: lane, start time (first decision of the act, else its first row's arrival) ──
    acts: Dict[tuple, dict] = {}

    def act_rec(key, lane=None) -> dict:
        rec = acts.setdefault(key, dict(lane=lane, t_start=float("inf"), t_first_row=float("inf"),
                                        reason=None, end_n_rows=None, src=None))
        if rec["lane"] is None and lane in LANES:
            rec["lane"] = lane
        return rec

    for i in range(n):
        if R.lane[i] is not None:
            rec = act_rec((int(sess[i]), int(R.act[i])), R.lane[i])
            if np.isfinite(R.t[i]):
                rec["t_first_row"] = min(rec["t_first_row"], float(R.t[i]))
    for t, lane, a, _z in J["decisions"]:
        s = session_of(t)
        if s is None or a < 0:
            continue
        rec = act_rec(canon(s, a), lane)
        if np.isfinite(t):
            rec["t_start"] = min(rec["t_start"], t)

    def set_end(key, reason, n_rows=None, src=None, lane=None, force=False):
        key = canon(*key)
        rec = act_rec(key, lane)
        if rec["reason"] is None or force:
            rec["reason"] = reason
            rec["src"] = src
        if n_rows is not None and rec["end_n_rows"] is None:
            rec["end_n_rows"] = int(n_rows)

    needles: Dict[tuple, dict] = {}

    # ── run.log: every process start (aligned to the journal's starts from the newest) ──
    log_sessions = _parse_run_log(run_dir)
    offset = R.n_sessions - len(log_sessions)
    R.sources["run_log_sessions"] = len(log_sessions)
    for j, evs in enumerate(log_sessions):
        s = j + offset
        if s < 0:
            continue
        last_act: Dict[str, int] = {}
        for e in evs:
            key = canon(s, e["act"])
            k = e["kind"]
            if k == "activation":
                prev = last_act.get(e["lane"])
                if prev is not None and prev != e["act"] and e["kv"].get("reason"):
                    set_end((s, prev), _norm_reason(e["kv"]["reason"]), src="run.log", lane=e["lane"])
            elif k == "needle":
                set_end(key, "needle", src="run.log", lane=e["lane"], force=True)
                needles.setdefault(key, dict(session=key[0], act=e["act"], lane=e["lane"], zoom=e["zoom"], row=None,
                                             n_rows=None, value=None, order=(s, 1, e["line"]), src="run.log"))
            elif k in ("zone", "dive_end"):
                set_end(key, _norm_reason(e["kv"].get("reason", k)), src="run.log", lane=e["lane"])
            last_act[e["lane"]] = e["act"]

    # ── engine checkpoint: exact needles (zones, events) of the process that wrote it ──
    ck = None
    ck_path = lanes_dir / "engine_ckpt.pt"
    if ck_path.is_file():
        try:
            ck = _cached(ck_path, _ckpt_essentials)
        except Exception as e:  # noqa: BLE001
            warn.append(f"engine_ckpt.pt unreadable ({e!r}); needles / dive ends from run.log")
    ck_Y = None
    if isinstance(ck, dict):
        s_ck = session_of(_f(ck.get("t")))
        s_ck = R.n_sessions - 1 if s_ck is None else s_ck
        R.sources["ckpt_session"] = s_ck
        ck_Y = ck.get("Y")
        for e in ck["events"]:
            if e.get("restored"):                 # history copied over by a resume: placed by its own process
                continue
            kind, lane, a = e.get("kind"), e.get("lane"), _i(e.get("act"))
            nr = e.get("n_rows")
            if kind == "activation" and e.get("prev_act") is not None:
                set_end((s_ck, _i(e["prev_act"])), _norm_reason(e.get("reason")), n_rows=nr, src="ckpt",
                        lane=lane, force=True)
            elif kind == "needle" and a >= 0:
                set_end((s_ck, a), "needle", n_rows=nr, src="ckpt", lane=lane, force=True)
                kc = canon(s_ck, a)
                nd = needles.get(kc)
                if nd is None or nd["src"] != "ckpt":           # replaces a run.log entry of the same dive
                    nd = needles[kc] = dict(session=kc[0], act=a, lane=lane, src="ckpt",
                                                   order=(s_ck, 0, 10 ** 6 + _i(e.get("seq"), 0)))
                nd.update(row=_i(e.get("pick")), n_rows=None if nr is None else _i(nr),
                          zoom=_i(e.get("zoom")), value=_f(e.get("value")))
            elif kind in ("zone", "dive_end") and a >= 0:
                set_end((s_ck, a), _norm_reason(e.get("reason")), n_rows=nr, src="ckpt", lane=lane)
        # zones: every needle of the process (events keep only the last 500); zone seq = declaration order.
        # A zone restored by a resume (scripts/lanes_hw.py: restored=True) keeps the act id of the process that
        # declared it; t_proc (a time inside that process) maps it there, never to the process that wrote the
        # checkpoint (whose own act ids restart / continue differently).
        for z in ck["zones"]:
            if z.get("kind") != "needle":
                continue
            a = _i(z.get("act"))
            if a < 0:
                continue
            s_z = s_ck
            if z.get("restored"):
                s_z = session_of(_f(z.get("t_proc")))
                if s_z is None:
                    continue
            set_end((s_z, a), "needle", src="ckpt", lane=z.get("lane"), force=True)
            kc = canon(s_z, a)
            nd = needles.get(kc)
            if nd is None or nd["src"] != "ckpt":
                nd = needles[kc] = dict(session=kc[0], act=a, lane=z.get("lane"), src="ckpt", row=None,
                                               n_rows=None, zoom=None, value=None)
            if nd.get("row") is None or nd["row"] < 0:
                nd["row"] = _i(z.get("pick"))
            if z.get("restored"):
                # the declaration's row count / value, carried by the restored record (its needle event is in
                # an earlier process's checkpoint): the diamond stays where it was declared
                if nd.get("n_rows") is None and z.get("n_rows") is not None:
                    nd["n_rows"] = _i(z["n_rows"])
                if nd.get("value") is None and z.get("value") is not None:
                    nd["value"] = _f(z["value"])
            nd["order"] = (s_z, 0, _i(z.get("seq"), 0))
        ls = ck["lane_state"] if isinstance(ck["lane_state"], dict) else {}
        R.sources["lane_state"] = {k: dict(v) for k, v in ls.items() if isinstance(v, dict)}

    # ── inferred ends: an earlier activation of a lane is over; a process restart cut the last ones off ──
    by_lane: Dict[tuple, List[int]] = {}
    for (s, a), rec in acts.items():
        if rec["lane"] in LANES:
            by_lane.setdefault((s, rec["lane"]), []).append(a)
    for (s, lane), alist in by_lane.items():
        alist = sorted(set(alist))
        for a in alist[:-1]:
            set_end((s, a), "no_needle", src="inferred")
        if last_sess.get((s, alist[-1]), s) < R.n_sessions - 1:   # continued by a dive-exact resume: still open
            set_end((s, alist[-1]), "restart", src="inferred")
    for rec in acts.values():
        rec["open"] = rec["reason"] is None
        if not np.isfinite(rec["t_start"]):
            rec["t_start"] = rec["t_first_row"]
    R.acts = acts

    # ── needle rows: exact (checkpoint pick, checked against the rows) or approximate (run.log) ──
    n_init = R.n_init
    y_all = np.concatenate([R.y_init, R.y])
    for key, nd in needles.items():
        nd["approx"] = False
        row = nd.get("row")
        ok = row is not None and 0 <= row < len(y_all)
        if ok and ck_Y is not None and row < len(ck_Y) and abs(float(y_all[row]) - float(ck_Y[row])) > 1e-6:
            ok = False                                           # rows.jsonl and the engine disagree
            warn.append(f"needle of act {key}: pick row {row} does not match the checkpoint; approximated")
        if ok and row >= n_init and (int(sess[row - n_init]), int(R.act[row - n_init])) != key:
            ok = False
        if not ok:
            own = [i for i in range(n) if (int(sess[i]), int(R.act[i])) == key and R.lane[i] is not None]
            if not own:
                nd["row"] = None
                continue
            zk = nd.get("zoom")
            zk = int(max(R.zoom[own])) if zk is None or zk < 0 else int(zk)
            pool = [i for i in own if R.zoom[i] == zk] or own
            if nd.get("value") is not None and np.isfinite(nd["value"]):
                i_best = min(pool, key=lambda i: abs(float(R.y[i]) - float(nd["value"])))
            else:
                i_best = max(pool, key=lambda i: float(R.y[i]) if np.isfinite(R.y[i]) else -np.inf)
            nd["row"] = n_init + i_best
            nd["approx"] = True
        if nd.get("n_rows") is None:
            nd["n_rows"] = acts.get(key, {}).get("end_n_rows")
    R.needles = sorted((nd for nd in needles.values() if nd.get("row") is not None), key=lambda nd: nd["order"])
    R.sources["needles"] = len(R.needles)
    R.sources["needles_approx"] = sum(bool(nd["approx"]) for nd in R.needles)

    # ── penalty mask (display colouring only) ──
    pm = _engine_valid_mask(lanes_dir, warn)
    if pm is not None and len(pm) >= n_init:
        v = np.ones(n_init + n, dtype=bool)
        k = min(len(pm), n_init + n)
        v[:k] = pm[:k]
        R.valid = v
    R.sources["epoch"] = epoch
    return R


# ── dive layout (pure numpy over a LanesRun) ────────────────────────────────────────────────────

@dataclass
class Layout:
    n_init: int
    order: np.ndarray            # display position -> global row index (-1 = an empty dive's slot)
    y: np.ndarray                # display-ordered measured y (NaN in an empty slot)
    key: List[Optional[tuple]]   # display-ordered dive key (session, act); None = init / unattributed
    zoom: np.ndarray             # display-ordered SENT zoom (-1 = init / unattributed / empty slot)
    valid: Optional[np.ndarray]  # display-ordered penalty mask
    blocks: List[dict]
    packet_starts: np.ndarray
    zoom_starts: np.ndarray
    zoom_levels: np.ndarray
    levels: List[dict]           # zoom_topm_levels + display start / end
    needles: List[dict]          # star / declared display positions
    m: int
    n_band: int = 0              # width of the init band: init.pt rows + late init rows
    n_late: int = 0              # late init rows (router disposition 'init' ...) at the end of the band
    gap: int = 0                 # display positions of an empty dive's slot
    min_open: int = 0            # minimum display width of a running dive (its label fits); 0 = none


def dive_layout(R: LanesRun, m: Optional[int] = None, gap: int = 4, min_open: int = 0) -> Layout:
    """Display order of a lanes run: init rows, then late init rows (``R.late_init``, arrival order), then one
    block per dive in start order (session, act), each block's rows in arrival order grouped (stable) by
    SENT zoom, then rows no dive owns ('other'). A dive that started (``R.acts``) without rows yet gets an
    empty slot of ``gap`` positions (order -1, y NaN, ``block['empty']``). A RUNNING dive with fewer than
    ``min_open`` rows is padded to ``min_open`` positions after its rows (order -1, y NaN, ``block['pad']``),
    so its strip label fits: room for its next results."""
    m = int(m) if m else top_m_for_dim(R.d)
    gap = max(1, int(gap))
    min_open = max(0, int(min_open))
    n_init, n = R.n_init, len(R.y)
    late = R.late_init if len(R.late_init) == n else np.zeros(n, dtype=bool)
    keys: List[Optional[tuple]] = [(int(R.session[i]), int(R.act[i])) if R.lane[i] is not None else None
                                   for i in range(n)]
    by_key: Dict[tuple, List[int]] = {}
    other: List[int] = []
    late_rows: List[int] = []
    for i, k in enumerate(keys):
        if late[i]:
            late_rows.append(i)
        elif k is None:
            other.append(i)
        else:
            by_key.setdefault(k, []).append(i)
    for k, info in R.acts.items():                     # started, nothing back yet: an empty slot
        if k not in by_key and info.get("lane") in LANES:
            by_key[k] = []

    order = list(range(n_init)) + [n_init + i for i in late_rows]
    n_band = len(order)
    blocks: List[dict] = []
    counts = {ln: 0 for ln in LANES}
    for k in sorted(by_key):
        idx = sorted(by_key[k], key=lambda i: int(R.zoom[i]))           # stable: arrival order per zoom
        info = R.acts.get(k, {})
        if idx:
            lanes_here = [R.lane[i] for i in idx]
            lane = max(set(lanes_here), key=lanes_here.count)
        else:
            lane = info.get("lane")
        counts[lane] += 1
        start = len(order)
        reason = info.get("reason")
        pad = max(0, min_open - len(idx)) if idx and reason is None else 0
        order += ([n_init + i for i in idx] + [-1] * pad) if idx else [-1] * gap
        t_start = info.get("t_start", float("inf"))
        if not np.isfinite(t_start):
            ts = R.t[idx] if idx else np.zeros(0)
            t_start = float(np.nanmin(ts)) if np.isfinite(ts).any() else float("nan")
        blocks.append(dict(key=k, lane=lane, label=f"{LANE_LETTER[lane]}{counts[lane]}", start=start,
                           end=len(order), t_start=t_start, reason=reason, open=reason is None,
                           needle=reason == "needle", end_n_rows=info.get("end_n_rows"),
                           zooms=sorted({int(R.zoom[i]) for i in idx}), session=k[0], act=k[1],
                           empty=not idx, n=len(idx), pad=pad))
    if other:
        start = len(order)
        order += [n_init + i for i in other]
        blocks.append(dict(key=None, lane=None, label="other", start=start, end=len(order), t_start=float("nan"),
                           reason=None, open=False, needle=False, end_n_rows=None, zooms=[], session=None,
                           act=None, empty=False, n=len(other), pad=0))
    order_a = np.asarray(order, dtype=int)
    real = order_a >= 0
    g = np.where(real, order_a, 0)
    n_all = n_init + n
    y_all = np.concatenate([R.y_init, R.y]).astype(float)
    key_all: List[Optional[tuple]] = [None] * n_init + keys
    zoom_all = np.concatenate([np.full(n_init, -1, dtype=int), R.zoom.astype(int)])
    qid_all = np.concatenate([np.full(n_init, -1, dtype=int), R.qid.astype(int)])

    def take(a, fill):
        out = np.full(len(order_a), fill, dtype=np.asarray(a).dtype if len(a) else type(fill))
        if n_all and len(order_a):
            out[real] = np.asarray(a)[g[real]]
        return out

    yd = take(y_all, np.nan).astype(float)
    kd: List[Optional[tuple]] = [key_all[int(p)] if p >= 0 else None for p in order_a]
    for b in blocks:                                   # an empty slot / a running dive's padding: its dive
        if b["empty"] or b["pad"]:
            p0 = b["start"] if b["empty"] else b["end"] - b["pad"]
            kd[p0:b["end"]] = [b["key"]] * (b["end"] - p0)
    zd = take(zoom_all, -1).astype(int)
    qd = take(qid_all, -1).astype(int)
    vd = (take(R.valid.astype(bool), True).astype(bool)
          if (R.valid is not None and len(R.valid) == n_all) else None)

    # result packets (display only: orient_packets) and SENT-zoom stretches (bands); an empty slot is its own
    # packet and an unshaded stretch, so neither the orientation nor a zoom band runs across it
    pstarts: List[int] = []
    if R.init_sizes and sum(R.init_sizes) == n_init:
        pstarts += np.cumsum([0] + list(R.init_sizes[:-1])).astype(int).tolist()
    zs: List[int] = []
    zl: List[int] = []
    prev = None
    for p in range(n_init, len(order_a)):
        cur = (kd[p], int(zd[p]), int(qd[p]))
        if cur != prev:
            pstarts.append(p)
        if p >= n_band and (p == n_band or prev is None or (kd[p], int(zd[p])) != prev[:2]):
            zs.append(p)
            zl.append(max(int(zd[p]), 0))
        prev = cur

    levels = zoom_topm_levels(yd[n_init:], kd[n_init:], zd[n_init:], m=m) if len(yd) > n_init else []
    for lv in levels:
        lv["start"] = n_init + int(lv["rows"].min())
        lv["end"] = n_init + int(lv["rows"].max()) + 1

    # needles: star at the pick's display position; diamond after the last row of its own block that had
    # arrived when it was declared (global index < n_rows at the declaration)
    pos = np.full(n_all, -1, dtype=int)
    pos[order_a[real]] = np.flatnonzero(real)
    blk_of = {b["key"]: b for b in blocks if b["key"] is not None}
    nds = []
    for k, nd in enumerate(R.needles):
        gr = int(nd["row"])
        if not 0 <= gr < n_all or pos[gr] < 0:
            continue
        star = int(pos[gr])
        b = blk_of.get((nd["session"], nd["act"]))
        declared = None
        if b is not None:
            seg = order_a[b["start"]:b["end"]]
            nr = nd.get("n_rows")
            if nr is not None and nr >= 0:
                before = np.flatnonzero((seg >= 0) & (seg < int(nr)))
                declared = b["start"] + int(before.max()) if before.size else None
            if declared is None:
                declared = b["end"] - b["pad"] - 1
            declared = max(declared, star) if b["start"] <= star < b["end"] else declared
        nds.append(dict(label=f"N{k + 1}", star=star, declared=declared, approx=bool(nd.get("approx")),
                        lane=nd.get("lane"), session=nd["session"], act=nd["act"], row=gr,
                        block=None if b is None else b["label"]))
    return Layout(n_init=n_init, order=order_a, y=yd, key=kd, zoom=zd, valid=vd, blocks=blocks,
                  packet_starts=np.asarray(sorted(set(pstarts)), dtype=int), zoom_starts=np.asarray(zs, dtype=int),
                  zoom_levels=np.asarray(zl, dtype=int), levels=levels, needles=nds, m=m, n_band=n_band,
                  n_late=len(late_rows), gap=gap, min_open=min_open)


# ── drawing ─────────────────────────────────────────────────────────────────────────────────────

def _clock(t: float, t0: Optional[float] = None) -> str:
    """Local 'HH:MM' of ``t``, prefixed '+Nd ' when it falls N calendar days after ``t0``'s day."""
    try:
        if not np.isfinite(t):
            return ""
        dt = datetime.fromtimestamp(float(t))
        days = (dt.date() - datetime.fromtimestamp(float(t0)).date()).days if t0 is not None else 0
        return (f"+{days}d " if days > 0 else "") + dt.strftime("%H:%M")
    except (OverflowError, OSError, ValueError, TypeError):
        return ""


def _renderer(fig):
    get = getattr(fig.canvas, "get_renderer", None)
    if get is None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        FigureCanvasAgg(fig)
        get = fig.canvas.get_renderer
    try:
        return get()
    except Exception:  # noqa: BLE001
        return None


def _ordinal(k: int) -> str:
    return f"{k}{'th' if 10 <= k % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')}"


def _ax_width_pt(fig) -> float:
    return fig.get_figwidth() * 72.0 * 0.86          # the axes' share of the figure width (approx.)


def _title_width_ch(fig, fontsize: float = 8.0) -> int:
    """Characters per title line: the title is centred on the axes (right of the figure's centre, past the
    y label), so a line may use about 0.9 of the figure width; ~0.6 em per character."""
    return max(60, int(fig.get_figwidth() * 0.9 * 72 / (fontsize * 0.6)))


def _wrap(text: str, width: int) -> List[str]:
    """``textwrap.wrap`` into as few lines as at ``width``, but balanced (no orphan word on a last line)."""
    lines = textwrap.wrap(text, width)
    if len(lines) <= 1:
        return lines
    for w in range(max(1, -(-len(text) // len(lines))), width + 1):
        cand = textwrap.wrap(text, w, break_long_words=False)
        if len(cand) <= len(lines):
            return cand
    return lines


def _text_pt(txt: str, fs: float, bold: bool = False) -> float:
    """Approximate width (points) of a strip label: ~0.58 em per character (0.66 bold) + 3 pt margin."""
    return len(txt) * fs * (0.66 if bold else 0.58) + 3


def _fit_widths(L: "Layout", fig) -> Tuple[int, int]:
    """(gap, min_open) for ``dive_layout`` on this canvas: an empty dive's slot wide enough for its label and
    start time (``_slot_units``), and the minimum width of a RUNNING dive so its 'F4 …' label fits (0 when
    every running dive is wide enough already, or the running dives would need more than half the axes)."""
    dv = [b for b in L.blocks if b["key"] is not None]
    n_empty = sum(bool(b["empty"]) for b in dv)
    n_pos = len(L.y)
    gap = L.gap
    if n_empty:
        gap = _slot_units(n_pos - n_empty * L.gap, n_empty, fig)
        n_pos += n_empty * (gap - L.gap)
    opn = [b for b in dv if b["open"] and not b["empty"]]
    if not opn:
        return gap, 0
    need = max(_text_pt(b["label"] + " …", 6.5, True) for b in opn) + 1.0
    w = _ax_width_pt(fig)
    if need * len(opn) > 0.5 * w:
        return gap, 0
    units = int(math.ceil(need * (n_pos + 2.0 * max(1.0, 0.01 * n_pos)) / w))
    return gap, (units if any(b["n"] < units for b in opn) else 0)


def _slot_units(n_pos: int, n_slots: int, fig, need_pt: float = SLOT_PT) -> int:
    """Width in x units of one empty slot (a dive without rows, a pending init trace) so that its strip label
    fits: ``n_pos`` data positions and ``n_slots`` slots share the axes' width. Slots take at most 65 % of
    it on a narrow canvas."""
    if n_slots <= 0:
        return 1
    w = _ax_width_pt(fig)
    avail = w - n_slots * need_pt
    if avail < 0.35 * w:
        need_pt, avail = 0.65 * w / n_slots, 0.35 * w
    pad = 2.0 * max(1.0, 0.01 * n_pos)
    return max(2, int(math.ceil(need_pt * (max(n_pos, 1) + pad) / avail)))


class _Strip:
    """The labelled strip just above the axes frame (children of the one axes): x = data, y = points above
    the frame, so it keeps its height at any canvas size and the layout engine reserves room for it;
    clipped to the axes' width when the toolbar zooms in. Build it after the x limits are set."""

    def __init__(self, ax, fig):
        from matplotlib.transforms import Affine2D, Bbox, ScaledTranslation, TransformedBbox, blended_transform_factory
        ytr = Affine2D().scale(1.0, 1.0 / 72.0) + fig.dpi_scale_trans + ScaledTranslation(0, 1, ax.transAxes)
        self.ax = ax
        self.btr = blended_transform_factory(ax.transData, ytr)
        self.clip = TransformedBbox(Bbox.from_extents(0, 0, 1, STRIP_GAP_PT + STRIP_PT + 1),
                                    blended_transform_factory(ax.transAxes, ytr))
        self.ax_w_pt = _ax_width_pt(fig)
        lo, hi = ax.get_xlim()
        self.span = (hi - lo) or 1.0

    def fits(self, x0, x1, txt, fs, bold=False) -> bool:
        return (x1 - x0) / self.span * self.ax_w_pt >= _text_pt(txt, fs, bold)

    def _put(self, artist):
        artist.set_clip_box(self.clip)
        return artist

    def block(self, x0, x1, fc, label, sub, txt_c="white", ec="white", ls="-", short=None) -> bool:
        """One strip block from x0 to x1 (data); bold label (+ a smaller sub line when both fit; else
        ``short``, e.g. 'F4 …', when it fits). False when not even the label fits (then the block is drawn
        unlabelled)."""
        from matplotlib.patches import Rectangle
        y0, h = STRIP_GAP_PT, STRIP_PT
        self._put(self.ax.add_patch(Rectangle((x0, y0), x1 - x0, h, transform=self.btr, fc=fc, ec=ec, lw=1.0,
                                              ls=ls, zorder=9)))
        xm = (x0 + x1) / 2
        kw = dict(transform=self.btr, ha="center", va="center", color=txt_c, zorder=10)
        if sub and self.fits(x0, x1, label, 7.0, True) and self.fits(x0, x1, sub, 6.0):
            self._put(self.ax.text(xm, y0 + 0.69 * h, label, fontsize=7.0, weight="bold", **kw))
            self._put(self.ax.text(xm, y0 + 0.27 * h, sub, fontsize=6.0, **kw))
            return True
        for txt in ((short, label) if short else (label,)):
            if self.fits(x0, x1, txt, 6.5, True):
                self._put(self.ax.text(xm, y0 + 0.5 * h, txt, fontsize=6.5, weight="bold", **kw))
                return True
        return False


def _set_summary(ax, chk: dict, lines: List[str]) -> None:
    """No visible title (user, 14 Nov). The text it carried stays available as ``ax.lanes_summary`` and
    ``chk['summary']`` (tests, logs). A blank title line keeps room above the axes for the dive strip."""
    text = "\n".join(lines)
    ax.lanes_summary = text
    chk["summary"] = text
    ax.set_title(" ", fontsize=6, pad=STRIP_GAP_PT + STRIP_PT + 2)


def _empty_slot(ax, x0: float, x1: float, color, text: str) -> None:
    """A hatched, dashed-outline slot over the plot's full height (an empty dive / a pending init trace),
    with ``text`` written vertically inside when the slot is wide enough."""
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory
    bt = blended_transform_factory(ax.transData, ax.transAxes)
    ax.add_patch(Rectangle((x0, 0.0), x1 - x0, 1.0, transform=bt, fill=False, hatch="////", ec=color, lw=0.9,
                           ls="--", alpha=0.55, zorder=1))
    if text:
        ax.text((x0 + x1) / 2, 0.5, text, rotation=90, transform=bt, ha="center", va="center", fontsize=6.5,
                color=INK2, zorder=6, clip_on=True)


def _legend_below(ax, fig, pairs) -> None:
    """One legend under the axes (below the ticks and x label), as many columns as fit the figure width."""
    from matplotlib.transforms import offset_copy
    seen, H, Lb = set(), [], []
    for h, lab in pairs:
        if lab and not lab.startswith("_") and lab not in seen:
            seen.add(lab)
            H.append(h)
            Lb.append(lab)
    if not H:
        return
    below = offset_copy(ax.transAxes, fig=fig, y=-32, units="points")   # under the ticks + x label
    rend = _renderer(fig)
    for ncol in (5, 4, 3, 2):
        leg = ax.legend(H, Lb, loc="upper center", bbox_to_anchor=(0.5, 0.0), bbox_transform=below, ncol=ncol,
                        fontsize=6.8, framealpha=0.85, handlelength=2.2, columnspacing=1.2, borderaxespad=0.0)
        if rend is None:
            break
        try:
            w_in = leg.get_window_extent(rend).width / fig.dpi
        except Exception:  # noqa: BLE001
            break
        if w_in <= fig.get_figwidth() - 0.2 or ncol == 2:
            break
        leg.remove()


def _plot_convergence():
    """``visualization.plot_convergence`` (bands, needle leashes, packet orientation), imported WITHOUT
    initialising CUDA. It imports ``src``, whose src/core/zombihop.py claims the GPU at import time
    (``torch.cuda.set_per_process_memory_fraction``) when CUDA is available. The GUI imported it (and src) at
    start-up, so it is simply reused there; anywhere else (the PNG tool, tests) it is imported with
    ``torch.cuda.is_available`` reporting False for the duration of the import, so that block is skipped and
    drawing never touches the GPU. (On Windows an empty CUDA_VISIBLE_DEVICES does not hide the GPU.) A
    process that imports this before ``src`` and then optimises on CUDA would miss src's CUDA defaults."""
    import importlib
    import sys
    mod = sys.modules.get("visualization.plot_convergence")
    if mod is not None:
        return mod
    import torch
    real = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        return importlib.import_module("visualization.plot_convergence")
    finally:
        torch.cuda.is_available = real


def _target(target):
    from matplotlib.figure import Figure
    if isinstance(target, Figure):
        return target, target.add_subplot(111)
    return target.figure, target


def draw_lanes_convergence(target, data: LanesRun, *, m: Optional[int] = None, title: Optional[str] = None,
                           legend: bool = True) -> dict:
    """Draw the lanes convergence plot of ``data`` into ``target`` (a Figure: one subplot is added; or an
    Axes) — ONE axes, both lanes; a run still in its initialization (``data.kind`` 'init' / 'legacy_init')
    gets the init view. Returns checks: blocks, bars (level / dip per dive and SENT zoom), needles, clean /
    finished dive counts, warnings (init view: traces, n_done, pending)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    pc = _plot_convergence()
    draw_needle_leashes, draw_search_bands, orient_packets = (pc.draw_needle_leashes, pc.draw_search_bands,
                                                              pc.orient_packets)

    fig, ax = _target(target)
    if data.kind in ("init", "legacy_init"):
        return _draw_init_view(fig, ax, data, title=title, legend=legend)
    L = dive_layout(data, m)
    for _ in range(4):                           # size empty slots / running dives so their labels fit
        gap, min_open = _fit_widths(L, fig)
        if (gap, min_open) == (L.gap, L.min_open):
            break
        L = dive_layout(data, m, gap=gap, min_open=min_open)
    n = len(L.y)
    n_rows = len(data.y)
    dives = [b for b in L.blocks if b["key"] is not None]
    finished = [b for b in dives if not b["open"]]
    dip_keys = {lv["key"] for lv in L.levels if lv["dip"]}
    clean = [b for b in finished if b["needle"] and b["key"] not in dip_keys]
    chk = dict(run=data.run_id, kind=data.kind, n_init=L.n_init, n_rows=n_rows, n_late_init=L.n_late, m=L.m,
               blocks=[{k: v for k, v in b.items() if k != "zooms"} for b in L.blocks],
               bars=[dict(block=next((b["label"] for b in dives if b["key"] == lv["key"]), None), zoom=lv["zoom"],
                          n=lv["n"], level=lv["level"], prev=lv["prev"], prev_zoom=lv["prev_zoom"], dip=lv["dip"])
                     for lv in L.levels],
               needles=L.needles, finished=len(finished), clean=len(clean),
               red_bars=sum(lv["dip"] for lv in L.levels), warnings=list(data.warnings), sources=dict(data.sources))
    head = title or data.run_id
    if n == 0:
        ax.text(0.5, 0.5, f"{head}: lanes run, no rows yet", ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color=INK2)
        ax.set_xticks([])
        ax.set_yticks([])
        return chk

    star = np.asarray([nd["star"] for nd in L.needles], dtype=int)
    Yd, pm, star_d = orient_packets(L.y, L.packet_starts, L.valid, star if star.size else None)
    Yd = np.asarray(Yd, dtype=float)
    idx = np.arange(n)
    fin = np.isfinite(Yd)

    # points: the app's colouring (penalised grey, valid steelblue) when the engine's mask is readable
    if pm is not None and (~pm[fin]).any():
        ax.scatter(idx[fin & ~pm], Yd[fin & ~pm], s=10, alpha=0.35, color="#aaaaaa", label="penalized", zorder=2)
        ax.scatter(idx[fin & pm], Yd[fin & pm], s=10, alpha=0.65, color="steelblue", label="valid", zorder=3)
    else:
        ax.scatter(idx[fin], Yd[fin], s=10, alpha=0.65, color="steelblue", label="measured", zorder=2)

    # bands: init grey (init.pt rows + late init rows), then each (dive, SENT zoom) stretch by depth
    draw_search_bands(ax, n, L.n_band, L.zoom_starts, L.zoom_levels)
    for b in L.blocks[1:]:                       # a thin lane-coloured rule at every later block start
        ax.axvline(b["start"] - 0.5, color=LANE_EDGE.get(b["lane"], INK2), lw=0.8, alpha=0.7, zorder=1)

    # running best, reset per dive; grey (dashed by end reason) for a dive that ended without a needle;
    # a dive with no rows yet: a hatched slot
    reasons_drawn: List[str] = []
    ticks = 0
    rb_labeled = False
    empty_open = empty_done = 0
    for b in dives:
        s, e = b["start"], b["end"] - b["pad"]   # its rows (a running dive's padding stays blank)
        if b["empty"]:
            if b["open"]:
                empty_open += 1
                _empty_slot(ax, s - 0.5, e - 0.5, LANE_EDGE[b["lane"]], "no results yet")
            else:
                empty_done += 1
                _empty_slot(ax, s - 0.5, e - 0.5, NO_NEEDLE_C, "ended, no results")
            continue
        best = np.fmax.accumulate(Yd[s:e])
        grey = not b["open"] and not b["needle"]
        if grey:
            r = b["reason"] or "no_needle"
            ax.plot(idx[s:e], best, color=GATE_COLOR, lw=1.0, ls=END_STYLES.get(r, END_STYLES["no_needle"])[1],
                    zorder=4)
            if r not in reasons_drawn:
                reasons_drawn.append(r)
            # tick where it ended: before the first of its rows that arrived after the end
            seg = L.order[s:e]
            late = np.flatnonzero(seg >= b["end_n_rows"]) if b.get("end_n_rows") is not None else np.zeros(0, int)
            if late.size:
                p = s + int(late.min())
                tx, ty = p - 0.5, float(best[p - s - 1]) if p > s else float(Yd[p])
            else:
                tx, ty = e - 0.5, float(best[-1])
            ax.plot([tx], [ty], ls="none", marker="|", ms=9, mew=1.4, color=GATE_COLOR, zorder=5)
            ticks += 1
        else:
            kw = dict(color="darkorange", lw=1.8, zorder=4)
            if not rb_labeled:
                kw["label"] = "running best (reset per dive)"
                rb_labeled = True
            ax.plot(idx[s:e], best, **kw)

    # target 1: measured top-m per (dive, SENT zoom)
    for lv in L.levels:
        if np.isfinite(lv["level"]):
            ax.hlines(lv["level"], lv["start"] - 0.5, lv["end"] - 0.5, color=DIP_C if lv["dip"] else LEVEL_C,
                      lw=2.4, zorder=7)

    # needles: numbered star on the point sampled, leashed to the diamond where it was declared
    if star.size:
        decl = np.asarray([nd["declared"] if nd["declared"] is not None else nd["star"] for nd in L.needles],
                          dtype=int)
        draw_needle_leashes(ax, Yd, star_d, decl, np.asarray([b["start"] for b in L.blocks], dtype=int))
        approx = {nd["label"] for nd in L.needles if nd["approx"]}
        if approx:
            for t in ax.texts:
                if t.get_text() in approx:
                    t.set_text(t.get_text() + "≈")

    # limits: data + bars, room for the needle labels
    vals = np.concatenate([Yd[fin], [lv["level"] for lv in L.levels if np.isfinite(lv["level"])]])
    lo, hi = (float(vals.min()), float(vals.max())) if vals.size else (0.0, 1.0)
    rng = (hi - lo) or max(abs(hi), 1.0) * 0.1
    ax.set_ylim(lo - 0.05 * rng, hi + 0.10 * rng)
    pad = max(1.0, 0.01 * n)
    ax.set_xlim(-0.5 - pad, n - 0.5 + pad)

    # dive strip, just above the axes frame: init, then F# / B# blocks in their lane's colour (grey when the
    # dive ended without a needle; white with a dashed outline while it has no rows), start time under the
    # label when it fits ('…' while the dive runs)
    strip = _Strip(ax, fig)
    hidden = 0
    first_t = [b["t_start"] for b in dives if np.isfinite(b["t_start"])]
    t_first = min(first_t) if first_t else None
    if L.n_band:
        strip.block(-0.5, L.n_band - 0.5, INIT_C, "init", "", txt_c=INK2)
    for b in L.blocks:
        x0, x1 = b["start"] - 0.5, b["end"] - 0.5
        if b["key"] is None:
            shown = strip.block(x0, x1, OTHER_C, "other", "", txt_c=INK2)
        else:
            sub = (_clock(b["t_start"], t_first) + (" …" if b["open"] else "")).strip()
            if b["empty"]:                       # 'B2 …' (+ its start time when it fits), dashed outline
                c = LANE_EDGE[b["lane"]] if b["open"] else NO_NEEDLE_C
                shown = strip.block(x0, x1, "white", b["label"] + (" …" if b["open"] else ""),
                                    _clock(b["t_start"], t_first), txt_c=LANE_TXT[b["lane"]] if b["open"] else c,
                                    ec=c, ls="--")
            else:                                # a running dive keeps its '…' when the time does not fit
                grey = not b["open"] and not b["needle"]
                shown = strip.block(x0, x1, NO_NEEDLE_C if grey else LANE_C[b["lane"]], b["label"], sub,
                                    txt_c="white" if grey else LANE_TXT[b["lane"]],
                                    ec=LANE_EDGE[b["lane"]] if grey else "white",
                                    short=b["label"] + " …" if b["open"] else None)
        hidden += int(not shown)
    chk["labels_hidden"] = hidden

    ax.set_xlabel("Dive-ordered sample index (each packet drawn with its higher-Y end last)", fontsize=8)
    ax.set_ylabel("Objective Y (measured)", fontsize=8)
    ax.tick_params(labelsize=7.5)

    # title: counts + the reading rule
    nf = sum(b["lane"] == "front" for b in dives)
    nb = sum(b["lane"] == "back" for b in dives)
    n_open = sum(b["open"] for b in dives)
    day = datetime.fromtimestamp(t_first).strftime("%d %b") if t_first is not None else ""
    counts = (f"({L.n_init} init" + (f" + {L.n_late} late init" if L.n_late else "")
              + f" + {n_rows - L.n_late} measured pts; {nf} front + {nb} back dives"
              + (f", {n_open} running" if n_open else "") + f"; {len(L.needles)} needles; "
              f"clean: {len(clean)} of {len(finished)} finished dives)")
    clean_rule = f"clean = needle and no zoom more than {DIP_EPS:g} below the previous zoom with rows"
    if fig.get_figwidth() < NARROW_IN:           # the GUI's small tab: the legend carries the rest
        note = (f"x: init, then dives F#/B# in start order{(' (strip: start time, ' + day + ')') if day else ''}; "
                + clean_rule)
    else:
        note = (f"x: init, then one block per dive (F#/B# = the front/back lane's #-th activation, zoom 0→1→2) "
                f"in start order{(' (strip: start time, ' + day + ')') if day else ''}, rows in arrival order; "
                f"bands, top-{L.m} bars, running-best resets by SENT zoom; grey: ended without a needle; "
                + ("hatched: no results yet; " if empty_open else "") + clean_rule)
    width_ch = _title_width_ch(fig)
    lines = _wrap(f"{head} — Convergence, both lanes  {counts}", width_ch) + _wrap(note, width_ch)
    if any(nd["approx"] for nd in L.needles):
        lines += _wrap("N#≈: needle row approximated from run.log (engine checkpoint missing or from an "
                       "earlier ZoMBI process)", width_ch)
    _set_summary(ax, chk, lines)

    if legend:
        hs, labs = ax.get_legend_handles_labels()
        pairs = list(zip(hs, labs))
        for r in reasons_drawn:
            lab, ls = END_STYLES.get(r, (f"no needle: {r}", END_STYLES["no_needle"][1]))
            pairs.append((Line2D([], [], color=GATE_COLOR, ls=ls, lw=1.2), lab))
        if ticks:
            pairs.append((Line2D([], [], ls="none", marker="|", ms=9, mew=1.4, color=GATE_COLOR), TICK_LABEL))
        if any(np.isfinite(lv["level"]) and not lv["dip"] for lv in L.levels):       # only bars drawn
            pairs.append((Line2D([], [], color=LEVEL_C, lw=2.4), f"measured top-{L.m} of the zoom"))
        if any(lv["dip"] for lv in L.levels):
            pairs.append((Line2D([], [], color=DIP_C, lw=2.4), f"top-{L.m} > {DIP_EPS:g} below previous zoom"))
        pairs += [(Patch(fc=LANE_C["front"], ec=LANE_EDGE["front"], lw=0.8), "front lane dive F#"),
                  (Patch(fc=LANE_C["back"], ec=LANE_EDGE["back"], lw=0.8), "back lane dive B#")]
        if any(not b["open"] and not b["needle"] for b in dives):
            pairs.append((Patch(fc=NO_NEEDLE_C, ec="none"), "dive ended without a needle"))
        if empty_open:
            pairs.append((Patch(fill=False, hatch="////", ec=INK2, ls="--", lw=0.9), "dive started, no results yet"))
        if empty_done:
            pairs.append((Patch(fill=False, hatch="////", ec=NO_NEEDLE_C, ls="--", lw=0.9),
                          "dive ended before any result"))
        _legend_below(ax, fig, pairs)
    return chk


# ── the init view: a run still in its axial initialization ──────────────────────────────────────

# a trace that has not reported yet, by kind (``_pending_kind``): legend text, slot text, strip sub line
_PENDING = {"sent": ("sent, awaiting result", "awaiting result", "…"),
            "requeued": ("re-queued (sent {k}×, presumed lost)", "re-queued", "re-queued"),
            "queued": ("queued, not sent yet", "queued", "queued")}
_PENDING_RANK = {"sent": 0, "requeued": 1, "queued": 2}


def _pending_kind(t: dict) -> Optional[str]:
    """'sent' (awaiting its result), 'requeued' (sent before, presumed lost, queued again), 'queued' (never
    sent) or None (done / abandoned / unknown)."""
    if t.get("status") == "sent":
        return "sent"
    if t.get("status") == "queued":
        return "requeued" if t.get("requeued") else "queued"
    return None


def _trace_style(i: int) -> Tuple[str, str, str]:
    """Colour, marker and marker edge of init trace ``i`` (by trace index, so a trace keeps them as results
    arrive). Past ``len(TRACE_C)`` traces the hues repeat, with a dark marker edge and their own marker."""
    i = max(int(i), 0)
    return TRACE_C[i % len(TRACE_C)], TRACE_MK[i % len(TRACE_MK)], ("white" if i < len(TRACE_C) else INK)


def _own_order(own: np.ndarray, starts: Sequence[int], sizes: Sequence[int]) -> np.ndarray:
    """Display order of the init points: result packets stay in arrival order, each packet's points sorted
    by their trace's own-dimension fraction (ascending: 0 -> 1 left to right). A packet with an unknown
    fraction keeps its measured order."""
    order = np.arange(len(own))
    for s0, sz in zip(starts, sizes):
        seg = own[s0:s0 + sz]
        if sz >= 2 and len(seg) == sz and np.isfinite(seg).all():
            order[s0:s0 + sz] = s0 + np.argsort(seg, kind="stable")
    return order


def _draw_init_view(fig, ax, R: LanesRun, *, title: Optional[str] = None, legend: bool = True) -> dict:
    """The init view (``run_kind`` 'init' / 'legacy_init'): every measured init point, result packets in
    arrival order and each packet's points by their trace's own-dimension fraction (0 -> 1 left to right, so
    Y against the labelled dimension reads the right way round), coloured and labelled by the trace's real
    dimension, a running best, and the traces that have not reported yet as hatched slots on the right
    (strip label 'dim 3' + '…' / 're-queued' / 'queued') and in the legend."""
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    draw_search_bands = _plot_convergence().draw_search_bands

    S = R.init_state or {}
    traces = list(S.get("traces") or [])
    y = np.asarray(R.y_init, dtype=float).reshape(-1)
    n = len(y)
    tr = np.asarray(S.get("trace", np.zeros(0, dtype=int)), dtype=int).reshape(-1)
    if len(tr) != n:
        tr = np.full(n, -1, dtype=int)
    own = np.asarray(S.get("own", np.zeros(0)), dtype=float).reshape(-1)
    if len(own) != n:
        own = np.full(n, np.nan)
    sizes = list(S.get("sizes") or [])
    if sum(sizes) != n:
        sizes = []
    pstarts = np.cumsum([0] + sizes[:-1]).astype(int) if sizes else np.zeros(0, dtype=int)
    order = _own_order(own, pstarts.tolist(), sizes)
    Yd, trd = y[order], tr[order]                       # trace ids move with their points
    done = [t for t in traces if t["status"] == "done"]
    n_empty = sum(1 for t in done if not t["n_points"])
    pending = sorted((t for t in traces if _pending_kind(t)),
                     key=lambda t: (_PENDING_RANK[_pending_kind(t)],
                                    10 ** 9 if t["last_write"] is None else t["last_write"], t["i"]))
    n_kind = {k: sum(_pending_kind(t) == k for t in pending) for k in _PENDING_RANK}
    abandoned = [t for t in traces if t["status"] == "abandoned"]
    gap = _slot_units(n, len(pending), fig, need_pt=SLOT_PT + 4) if pending else 0
    n_pos = n + gap * len(pending)
    head = title or R.run_id
    chk = dict(run=R.run_id, kind=R.kind, n_init=n, n_rows=0, n_late_init=0, m=top_m_for_dim(R.d), blocks=[],
               bars=[], needles=[], finished=0, clean=0, red_bars=0, warnings=list(R.warnings),
               sources=dict(R.sources), traces=[dict(t) for t in traces], n_traces=len(traces),
               n_done=len(done), n_empty=n_empty, n_sent=n_kind["sent"], n_requeued=n_kind["requeued"],
               n_queued=n_kind["queued"], pending=[t["label"] for t in pending], complete=bool(S.get("complete")),
               optimizer=R.optimizer, labels_hidden=0)

    if not traces and n == 0:
        ax.text(0.5, 0.5, f"{head}: initialization, init_state.json "
                f"{'unreadable' if R.warnings else 'has no traces yet'}", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color=INK2)
        ax.set_xticks([])
        ax.set_yticks([])
        return chk

    idx = np.arange(n)
    if n:
        draw_search_bands(ax, n, n)                      # the grey 'initialization (n pts)' band
        for t in traces:
            sel = trd == t["i"]
            if sel.any():
                c, mk, edge = _trace_style(t["i"])
                ax.scatter(idx[sel], Yd[sel], s=18, marker=mk, color=c, alpha=0.9, edgecolors=edge,
                           linewidths=0.4 if edge == "white" else 0.8, zorder=3,
                           label=f"{t['label']} trace: {int(sel.sum())} pts")
        if (trd < 0).any():
            ax.scatter(idx[trd < 0], Yd[trd < 0], s=12, alpha=0.65, color="steelblue", label="measured", zorder=3)
        ax.plot(idx, np.fmax.accumulate(Yd), color="darkorange", lw=1.8, zorder=4, label="running best")
    else:
        ax.text(0.5, 0.92, "waiting for the first init result", transform=ax.transAxes, ha="center",
                va="center", fontsize=8.5, color=INK2, zorder=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.9))
    for k, t in enumerate(pending):
        x0 = n - 0.5 + k * gap
        _empty_slot(ax, x0, x0 + gap, _trace_style(t["i"])[0], _PENDING[_pending_kind(t)][1])

    fin = np.isfinite(Yd)
    lo, hi = (float(Yd[fin].min()), float(Yd[fin].max())) if fin.any() else (0.0, 1.0)
    rng = (hi - lo) or max(abs(hi), 1.0) * 0.1
    ax.set_ylim(lo - 0.05 * rng, hi + 0.10 * rng)
    pad = max(1.0, 0.01 * n_pos)
    ax.set_xlim(-0.5 - pad, n_pos - 0.5 + pad)

    # strip: one block per result packet (the trace's colour, tinted; dim label + arrival time), then the
    # pending traces' slots (white, dashed outline, 'dim 3' + '…' / 're-queued' / 'queued')
    strip = _Strip(ax, fig)
    hidden = 0
    bt = [t for t in (S.get("block_t") or []) if np.isfinite(t)]
    t0 = min(bt) if bt else None
    if sizes:
        for s0, sz, tb, ti in zip(pstarts.tolist(), sizes, S.get("block_t") or [], S.get("block_trace") or []):
            c = _trace_style(ti)[0]
            fc = tuple(1.0 - 0.38 * (1.0 - v) for v in to_rgb(c))
            lab = next((t["label"] for t in traces if t["i"] == ti), f"trace {ti + 1}")
            hidden += int(not strip.block(s0 - 0.5, s0 + sz - 0.5, fc, lab, _clock(tb, t0), txt_c=INK, ec=c))
    for k, t in enumerate(pending):
        x0 = n - 0.5 + k * gap
        c = _trace_style(t["i"])[0]
        shown = strip.block(x0, x0 + gap, "white", t["label"], _PENDING[_pending_kind(t)][2], txt_c=INK, ec=c,
                            ls="--")
        hidden += int(not shown)
    chk["labels_hidden"] = hidden

    ax.set_xlabel("Init sample index: traces in arrival order, each by its own dimension's fraction 0 → 1",
                  fontsize=8)
    ax.set_ylabel("Objective Y (measured)", fontsize=8)
    ax.tick_params(labelsize=7.5)

    n_tr = len(traces)
    counted = (f"{len(done)} of {n_tr} traces {'reported' if n_empty else 'measured'} ("
               + (f"{n_empty} empty, " if n_empty else "") + f"{n} points)")
    state = ("initialization complete: " if S.get("complete") else "initialization: ") + counted
    if R.kind != "init":
        tail = "a snapshot-optimizer run past init (its later rows are in its snapshots, not drawn here)"
    elif R.optimizer and R.optimizer != "lanes":
        tail = f"optimizer '{R.optimizer}': its snapshot plot follows the init"
    else:
        tail = "lanes not started"
    t_sent = _f(S.get("first_write_t"))
    t_ref = t_sent if np.isfinite(t_sent) else t0
    when = []
    if np.isfinite(t_sent):
        when.append(f"first sent {_clock(t_sent)}")
    if bt:
        when.append(f"last result {_clock(max(bt), t_ref)}")
    waiting = ([f"{n_kind['sent']} trace{'s' if n_kind['sent'] != 1 else ''} awaiting results"]
               if n_kind["sent"] else [])
    if n_kind["requeued"]:
        waiting.append(f"{n_kind['requeued']} re-queued (presumed lost)")
    if n_kind["queued"]:
        waiting.append(f"{n_kind['queued']} queued")
    note = ("x: one packet per trace result, in arrival order, its points by the trace's own fraction 0 → 1; "
            "colour + marker = the trace's real dimension (strip: arrival time)"
            + (f"; hatched: {', '.join(waiting)}" if waiting else "")
            + (f"; abandoned: {', '.join(t['label'] for t in abandoned)}" if abandoned else "")
            + (f"; {', '.join(when)}" if when else ""))
    width_ch = _title_width_ch(fig)
    lines = _wrap(f"{head} — {state}; {tail}", width_ch) + _wrap(note, width_ch)
    _set_summary(ax, chk, lines)

    if legend:
        hs, labs = ax.get_legend_handles_labels()
        pairs = list(zip(hs, labs))
        for t in traces:
            c, mk, _ = _trace_style(t["i"])
            kind = _pending_kind(t)
            if kind:
                pairs.append((Line2D([], [], ls="none", marker=mk, ms=5, mfc="white", mec=c, mew=1.1),
                              f"{t['label']} trace: {_PENDING[kind][0].format(k=t.get('sends', 0))}"))
            elif t["status"] == "abandoned":
                pairs.append((Line2D([], [], ls="none", marker=mk, ms=5, mfc="white", mec=NO_NEEDLE_C, mew=1.1),
                              f"{t['label']} trace: abandoned"))
            elif t["status"] == "done" and not (trd == t["i"]).any():
                pairs.append((Line2D([], [], ls="none", marker=mk, ms=5, mfc="white", mec=c, mew=1.1),
                              f"{t['label']} trace: empty result"))
        if pending:
            pairs.append((Patch(fill=False, hatch="////", ec=INK2, ls="--", lw=0.9), "trace awaiting its result"))
        _legend_below(ax, fig, pairs)
    return chk
