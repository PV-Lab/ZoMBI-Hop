"""
Retroactive needle-trigger replay for ZoMBI-Hop
===============================================

Reconstructs the per-iteration convergence record stream of an existing run —
preferring the ``convergence_history.jsonl`` sidecar written by the run loop,
falling back to parsing ``run.log`` — and identifies activations that WOULD
have declared a needle under the *current* (possibly loosened) criteria.
Consumed by ``ZoMBIHop.retro_declare_needles``.

Evidence standard
-----------------
``find_retro_triggers`` replays the RECORDED convergence counter, which
already includes every reset that actually happened (non-converged
iterations, too-shallow zoom-in resets, failure dispatches). It deliberately
does NOT simulate the counterfactual resets a lower ``n_consecutive`` would
have caused at shallow zooms: at a declarable position (``zoom >=
min_zoom_for_needle`` and ``iters_this_zoom >= min_iters_per_zoom``) a
recorded ``counter >= n`` means exactly "the last n measured lines all
converged consecutively", which is the intended standard of evidence.

Record schema (all integers 0-based, matching the loop variables):
  {ts, activation, zoom, event: "zoom_entry"}            zoom-loop (re)entry
  {ts, activation, zoom, iteration, measured: false,
   event: "candidate_none"}                              no candidate produced
  {ts, activation, zoom, iteration, measured: true,
   event: "all_penalized"}                               measured, no check ran
  {ts, activation, zoom, iteration, measured: true,
   converged, counter, ei}                               normal measured line
"""

import json
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

# run.log markers. Every ``_log`` message appears twice (a plain print and a
# "[HH:MM:SS] " tee copy, in either order); _dedupe_log_lines collapses the
# pairs before these are applied. Print-only lines (status, "Loaded state:")
# appear once, never timestamped.
_TS_PREFIX = re.compile(r"^\[\d{2}:\d{2}:\d{2}\] ?")
_ACTIVATION = re.compile(r"^ACTIVATION (\d+)/(\S+)$")
_ZOOM = re.compile(r"^--- Zoom (\d+)/(\d+) ---$")
_ITER = re.compile(r"^\s*· iter (\d+)/(\d+)\s+\(activation lines (\d+)/(\d+)\)$")
_MEASURED = re.compile(r"^\s*\[ZoMBIHop\] Objective returned (\d+) points?,")
_ALL_PENALIZED = re.compile(r"^No unpenalized Y values, breaking")
_STATUS = re.compile(r"^\[A(\d+)/Z(\d+)/I(\d+)\] Candidate: (.*)$")
_CONV_COUNT = re.compile(r"^Convergence count: (\d+)/(\d+)$")
_CONV_DETAIL = re.compile(r"^Converged: EI=(\S+), improvement=")
_LOADED_STATE = re.compile(r"^Loaded state: activation=(\d+), zoom=(\d+), iteration=(\d+)$")
_LAUNCH_SEAM = re.compile(r"^===== (hardware run launched|process exited)")

_ACT_LABEL = re.compile(r"^act(\d+)")


def _read_jsonl(path: Path) -> List[dict]:
    """Read a convergence_history.jsonl into a record list (never raises)."""
    records: List[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line from a mid-write kill
                if isinstance(rec, dict):
                    records.append(rec)
    except OSError:
        return []
    return records


def _by_activation(records: List[dict]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = {}
    for rec in records:
        act = rec.get("activation")
        if isinstance(act, int):
            grouped.setdefault(act, []).append(rec)
    return grouped


def _n_measured(records: List[dict]) -> int:
    return sum(1 for r in records if r.get("measured"))


def load_convergence_history(run_dir: Union[str, Path]) -> Tuple[List[dict], str]:
    """
    Load the convergence record stream for a run.

    Merges the ``convergence_history.jsonl`` sidecar with the stream
    reconstructed from ``run.log``, choosing PER ACTIVATION whichever source
    carries more measured lines for that activation (ties go to the sidecar,
    which is structured rather than parsed). Returns ``(records, source)`` —
    or ``([], reason)`` when neither file exists.

    Per-activation rather than whole-file selection because a run that
    predates the sidecar accumulates a partial one as soon as it is resumed:
    preferring the file wholesale would let a couple of rows from an aborted
    resume shadow the entire logged history. Merging per activation is sound
    because triggers are per-activation and ``find_retro_triggers`` carries no
    state across activation boundaries.
    """
    run_dir = Path(run_dir)
    jsonl_path = run_dir / "convergence_history.jsonl"
    log_path = run_dir / "run.log"

    jsonl_records = _read_jsonl(jsonl_path) if jsonl_path.exists() else []
    log_records = parse_run_log(log_path) if log_path.exists() else []

    if not jsonl_records and not log_records:
        return [], f"no convergence_history.jsonl or run.log in {run_dir}"
    if not jsonl_records:
        return log_records, "run.log"
    if not log_records:
        return jsonl_records, "convergence_history.jsonl"

    from_jsonl = _by_activation(jsonl_records)
    from_log = _by_activation(log_records)
    merged: List[dict] = []
    used_jsonl = used_log = 0
    for act in sorted(set(from_jsonl) | set(from_log)):
        j = from_jsonl.get(act, [])
        l = from_log.get(act, [])
        if _n_measured(j) >= _n_measured(l) and j:
            merged.extend(j)
            used_jsonl += 1
        else:
            merged.extend(l)
            used_log += 1
    return merged, (f"run.log + convergence_history.jsonl "
                    f"({used_log} activation(s) from the log, "
                    f"{used_jsonl} from the sidecar)")


def _dedupe_log_lines(path: Union[str, Path]) -> List[str]:
    """
    Collapse the duplicated plain/timestamped line pairs of a run.log.

    Per line: strip a leading "[HH:MM:SS] " prefix, drop the line if the
    result is empty (the tee copy of a message starting with "\\n" renders as
    a bare prefix line), and drop it if identical to the previous kept line.
    Order-agnostic — the plain and timestamped copies interleave in either
    order depending on tee buffering.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read().splitlines()
    kept: List[str] = []
    for line in raw:
        stripped = _TS_PREFIX.sub("", line, count=1)
        if not stripped.strip():
            continue
        if kept and stripped == kept[-1]:
            continue
        kept.append(stripped)
    return kept


def parse_run_log(path: Union[str, Path]) -> List[dict]:
    """
    Reconstruct the convergence record stream from a run.log.

    Handles: duplicated plain/timestamped lines; "No unpenalized Y" blocks
    (measured, no counter line); candidate-None failure iterations; zoom
    re-entries; kill/resume seams (a "Loaded state:" line mid-file followed by
    repeated activation/zoom headers); measured lines whose status line was
    lost to a kill (recorded with counter=0). All positions converted to
    0-based. Records carry no ``ts`` field — the log timestamps are
    wall-clock-of-day only.
    """
    lines = _dedupe_log_lines(path)

    records: List[dict] = []
    activation = 0
    zoom = 0
    iteration = 0
    pending: Optional[dict] = None   # measured record awaiting a possible counter line
    measured_open = False            # "Objective returned" seen, outcome line not yet
    saw_converged_detail = False

    def _flush():
        # Close out any half-parsed measured line at a structural boundary.
        nonlocal pending, measured_open, saw_converged_detail
        if pending is not None:
            records.append(pending)
        elif measured_open:
            # Measured, snapshotted, but the process died before the status
            # line: counts as a line at this zoom with a reset counter.
            records.append({
                "activation": activation, "zoom": zoom, "iteration": iteration,
                "measured": True, "converged": False, "counter": 0,
                "event": "measured_no_status",
            })
        pending = None
        measured_open = False
        saw_converged_detail = False

    for line in lines:
        m = _CONV_COUNT.match(line)
        if m:
            if pending is not None:
                pending["counter"] = int(m.group(1))
                pending["converged"] = True
                records.append(pending)
                pending = None
            continue

        if _CONV_DETAIL.match(line):
            saw_converged_detail = True
            continue

        m = _STATUS.match(line)
        if m:
            if pending is not None:  # previous measured line had counter 0
                records.append(pending)
                pending = None
            activation, zoom, iteration = (int(m.group(1)) - 1,
                                           int(m.group(2)) - 1,
                                           int(m.group(3)) - 1)
            if m.group(4).strip().startswith("None"):
                records.append({
                    "activation": activation, "zoom": zoom, "iteration": iteration,
                    "measured": False, "event": "candidate_none",
                })
            else:
                pending = {
                    "activation": activation, "zoom": zoom, "iteration": iteration,
                    "measured": True, "converged": saw_converged_detail, "counter": 0,
                }
            measured_open = False
            saw_converged_detail = False
            continue

        if _ALL_PENALIZED.match(line):
            if pending is not None:  # stale record from a previous iteration
                records.append(pending)
                pending = None
            records.append({
                "activation": activation, "zoom": zoom, "iteration": iteration,
                "measured": True, "event": "all_penalized",
            })
            measured_open = False
            saw_converged_detail = False
            continue

        if _MEASURED.match(line):
            if pending is not None or measured_open:
                _flush()
            measured_open = True
            continue

        m = _ITER.match(line)
        if m:
            _flush()
            iteration = int(m.group(1)) - 1
            continue

        m = _ZOOM.match(line)
        if m:
            _flush()
            zoom = int(m.group(1)) - 1
            records.append({"activation": activation, "zoom": zoom,
                            "event": "zoom_entry"})
            continue

        m = _ACTIVATION.match(line)
        if m:
            _flush()
            activation = int(m.group(1)) - 1
            zoom = 0
            iteration = 0
            continue

        m = _LOADED_STATE.match(line)
        if m:
            _flush()
            activation, zoom, iteration = (int(m.group(1)), int(m.group(2)),
                                           int(m.group(3)))
            continue

        if _LAUNCH_SEAM.match(line):
            _flush()
            continue

    _flush()
    return records


def find_retro_triggers(
    records: List[dict],
    n_consecutive: int,
    min_zoom_for_needle: int,
    min_iters_per_zoom: int,
    skip_activations: Optional[Set[int]] = None,
) -> List[dict]:
    """
    Walk records chronologically and return at most one trigger per
    activation: the FIRST measured record satisfying

        counter >= n_consecutive
        and zoom >= min_zoom_for_needle
        and iters_this_zoom >= min_iters_per_zoom

    ``iters_this_zoom`` counts measured records since the last zoom_entry,
    mirroring the live loop (which resets it on every zoom entry, including
    failure-retry re-entries). Activations in ``skip_activations`` — those
    that already declared a needle — never trigger. Trigger dicts:
    ``{activation, zoom, iteration, counter}``, in chronological order.
    """
    skip = {int(a) for a in (skip_activations or set())}
    triggers: List[dict] = []
    fired: Set[int] = set()
    iters_this_zoom = 0
    for rec in records:
        if rec.get("event") == "zoom_entry":
            iters_this_zoom = 0
            continue
        if not rec.get("measured"):
            continue
        iters_this_zoom += 1
        act = rec.get("activation")
        if act is None or int(act) in skip or int(act) in fired:
            continue
        counter = int(rec.get("counter") or 0)
        zoom = int(rec.get("zoom") or 0)
        if (counter >= n_consecutive
                and zoom >= min_zoom_for_needle
                and iters_this_zoom >= min_iters_per_zoom):
            fired.add(int(act))
            triggers.append({
                "activation": int(act),
                "zoom": zoom,
                "iteration": int(rec.get("iteration") or 0),
                "counter": counter,
            })
    return triggers


def _iter_snapshot_summaries(run_dir: Union[str, Path]) -> Iterator[dict]:
    """Yield each snapshot's summary.json dict in sequence (name) order."""
    snap_dir = Path(run_dir) / "snapshots"
    if not snap_dir.exists():
        return
    for name in sorted(p.name for p in snap_dir.iterdir() if p.is_dir()):
        summary_path = snap_dir / name / "summary.json"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(summary, dict):
            yield summary


def needle_discovery_activations(run_dir: Union[str, Path]) -> List[int]:
    """
    Discovery activation of each needle, in needle order.

    Derived from the snapshot summaries — needles.json does not record the
    activation, and the stored per-needle ``iteration`` resets every run() —
    by attributing each increment of ``n_needles`` to the activation of the
    summary that first reported it.
    """
    out: List[int] = []
    prev = 0
    for summary in _iter_snapshot_summaries(run_dir):
        n = summary.get("n_needles")
        if not isinstance(n, int):
            continue
        if n > prev:
            out.extend([int(summary.get("activation", 0))] * (n - prev))
        prev = max(prev, n)
    return out


def activation_point_ranges(run_dir: Union[str, Path]) -> Dict[int, List[Tuple[int, int]]]:
    """
    Half-open row ranges of X_all_actual attributable to each activation.

    Snapshot summaries record the cumulative point count at snapshot time and
    the run loop snapshots once per measured line, so rows ``[prev, n)``
    belong to the snapshot that first reported ``n`` points — the same order
    delta replay reconstructs X_all in on load. Rows are attributed when the
    snapshot label starts with ``act{A}`` (measured lines, spacefill,
    timeouts); init rows stay unattributed.
    """
    ranges: Dict[int, List[Tuple[int, int]]] = {}
    prev = 0
    for summary in _iter_snapshot_summaries(run_dir):
        n = summary.get("n_points")
        if not isinstance(n, int):
            continue
        if n > prev:
            m = _ACT_LABEL.match(str(summary.get("label") or ""))
            if m:
                ranges.setdefault(int(m.group(1)), []).append((prev, n))
        prev = max(prev, n)
    return ranges
