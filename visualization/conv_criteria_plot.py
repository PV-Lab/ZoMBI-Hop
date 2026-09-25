"""
visualization/conv_criteria_plot.py
===================================
Convergence plot in the style of ``optimize/run_mobo.py``'s ``plot_convergence``
— every observed Y against sample index, a running-best envelope that **resets at
each activation**, and a crimson dashed rule at every declared needle — plus one
extra mark this view exists for:

  * **green dotted rules at every point in time the convergence criteria were
    satisfied**, i.e. each line where ``ZoMBIHop._check_convergence_to_needle``
    returned True (EI below the output-noise floor *and* the best-Y improvement
    below that same floor).

Those events are recovered from what the run logged, since they are not stored in
the snapshots:

  * ``run.log`` — ``_check_convergence_to_needle`` emits
    ``Converged: EI=…, improvement=…, input_dist=…, logEI=…`` exactly when both
    gates pass (``converged = ei_low and output_within_noise``), so every such
    line is one satisfied-criteria event.
  * ``convergence_history.jsonl`` — the sidecar's ``{"converged": true}`` records,
    where present, carry the same events with a unix timestamp.

Neither source records a sample index, so each event is placed in time: snapshot
``summary.json`` files carry ``(timestamp, n_points)``, and a snapshot is written
immediately after every objective call, so the event is attributed to the last
snapshot at or before it — i.e. the line whose measurement the check ran on. The
``run.log`` timestamps are wall-clock ``[HH:MM:SS]`` only, so the date comes from
the ``===== hardware run launched YYYY-MM-DD … =====`` headers and is advanced on
each midnight rollover. Events from both sources are unioned and de-duplicated by
sample index.

Note that a green rule marks the criteria being **satisfied**, not a needle being
declared: declaration additionally requires ``n_consecutive_converged`` in a row,
``min_zoom_for_needle`` and ``min_iters_per_zoom``. Those extra gates are
deliberately ignored here — greens without a following crimson are precisely the
convergences the search-discipline constraints threw away.

Counterfactual ``output_noise_threshold_mult``
----------------------------------------------
Both gates compare against the *same* quantity,
``floor = get_output_noise() * output_noise_threshold_mult``, so an event survives
a counterfactual multiplier iff ``max(EI, improvement) < floor``. Both of those
per-event numbers are in the log, which makes the counterfactual an exact replay —
no GP refit — over a single parameter, the floor.

Two caveats, and they matter:

  * ``get_output_noise()`` (the GP likelihood noise) is **never logged and never
    snapshotted** — it lives only in ``DataHandler._gp_output_noise`` in memory —
    so the absolute floor of each event is not recoverable. What *is* recoverable
    is a lower bound: every logged event passed at the run's actual multiplier, so
    ``output_noise > max_events(max(EI, improvement)) / actual_mult``. That bound
    is the default assumed noise here, which makes the reported counts at a given
    multiplier the **most permissive** consistent with the run. Pass a measured
    ``--output-noise`` (or a direct ``--floor``) to replace the assumption.
  * The counterfactual can only ever **remove** events, never add them. The log
    records a line only when both gates passed, so checks that failed left no
    trace and cannot be resurrected by raising the multiplier. And the replay is
    static: really changing the multiplier would change which needles are
    declared and therefore the whole sampling trajectory downstream.

Usage
-----
  python visualization/conv_criteria_plot.py --run runs/run_39af
  python visualization/conv_criteria_plot.py --run runs/run_39af --mult 0.5
  python visualization/conv_criteria_plot.py --run runs/run_39af --mult 1.0 --output-noise 0.02
  python visualization/conv_criteria_plot.py --run runs/run_39af --floor 5e-3
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── project root (and this dir, for `plot_run`) on sys.path ───────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from plot_run import (  # noqa: E402
    _default_snapshot,
    _resolve_run_dir,
    load_run_dataset,
    run_activations,
    run_needles,
)

# `Converged: EI=4.81e-03, improvement=-4.75e-02, …`, timestamped by the log writer.
# Only the timestamped copy is matched: every such message is written twice (once
# from the bare print, once through write_log with a clock prefix), and matching
# both would double-count each event.
_CONVERGED_RE = re.compile(
    r"^\[(\d{2}):(\d{2}):(\d{2})\]\s*Converged: EI=([-\d.eE+]+), improvement=([-\d.eE+]+)"
)
_LAUNCH_RE = re.compile(r"^=+ hardware run launched (\d{4}-\d{2}-\d{2}) (\d{2}):(\d{2}):(\d{2})")
_STAMP_RE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]")

# Slack when attributing an event to a snapshot: the log clock is second-resolution
# and the convergence check is logged a moment *after* the snapshot it belongs to,
# so an event may read as marginally earlier than its own snapshot's timestamp.
_TS_SLACK_S = 2.0

# Fallback multiplier when the run recorded no hparams_effective.json.
_FALLBACK_MULT = 2.0


@dataclass(frozen=True)
class ConvEvent:
    """One logged satisfied-criteria check, placed on the sample axis.

    ``improvement`` is NaN for sidecar-only events, which record the EI but not the
    realized best-Y change; ``requirement`` degrades to the EI alone for those.
    """
    sample: int
    ts: float
    ei: float
    improvement: float
    source: str

    @property
    def requirement(self) -> float:
        """Smallest floor that would still admit this event.

        The check is ``ei < floor and improvement < floor``, so both survive iff the
        larger of the two is below the floor.
        """
        vals = [v for v in (self.ei, self.improvement) if np.isfinite(v)]
        return max(vals) if vals else float("nan")


def snapshot_timeline(run_dir: Path, snapshot: str | None = None) -> list[tuple[float, int, str]]:
    """Chronological ``(timestamp, n_points, label)`` for a run's snapshots.

    Stops after ``snapshot`` when given, so the timeline covers exactly the slice
    of the run the plotted dataset was reconstructed from.
    """
    snaps_dir = run_dir / "snapshots"
    out: list[tuple[float, int, str]] = []
    if not snaps_dir.is_dir():
        return out
    for snap in sorted(p for p in snaps_dir.iterdir() if p.is_dir()):
        try:
            s = json.loads((snap / "summary.json").read_text())
            out.append((float(s["timestamp"]), int(s["n_points"]), snap.name))
        except Exception:
            continue                      # a partial/unreadable snapshot is skipped
        if snapshot is not None and snap.name == snapshot:
            break
    out.sort(key=lambda r: r[0])
    return out


def _sample_index_at(ts: float, timeline: list[tuple[float, int, str]]) -> int | None:
    """Index of the last sample collected at or before wall-clock ``ts``.

    ``n_points`` is the cumulative count *after* that snapshot's objective call, so
    the line the check ran on ends at sample ``n_points - 1``.
    """
    n = None
    for t, n_points, _ in timeline:
        if t <= ts + _TS_SLACK_S:
            n = n_points
        else:
            break
    return None if n is None or n <= 0 else n - 1


def converged_events_from_log(run_dir: Path) -> list[tuple[float, float, float]]:
    """``(unix ts, EI, improvement)`` for every satisfied-criteria event in run.log.

    ``run.log`` timestamps are wall-clock only. The date is taken from the launch
    banner of the process that wrote the following lines, and advanced whenever the
    clock goes backwards (midnight rollover), so a multi-day run resolves correctly.
    """
    log = run_dir / "run.log"
    if not log.exists():
        return []
    events: list[tuple[float, float, float]] = []
    date: _dt.date | None = None
    prev: tuple[int, int, int] | None = None

    def _advance(tod: tuple[int, int, int]) -> _dt.date:
        nonlocal date, prev
        if prev is not None and tod < prev:
            date = date + _dt.timedelta(days=1)
        prev = tod
        return date

    with log.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = _LAUNCH_RE.match(line)
            if m:                          # a restart re-anchors the date exactly
                date = _dt.date.fromisoformat(m.group(1))
                prev = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
                continue
            if date is None:               # lines before any banner have no date
                continue
            m = _CONVERGED_RE.match(line)
            if m:
                tod = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                d = _advance(tod)
                ts = _dt.datetime.combine(d, _dt.time(*tod)).timestamp()
                events.append((ts, float(m.group(4)), float(m.group(5))))
                continue
            m = _STAMP_RE.match(line)
            if m:                          # keep the rollover tracker current
                _advance((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return events


def converged_events_from_sidecar(run_dir: Path) -> list[tuple[float, float, float]]:
    """``(unix ts, EI, nan)`` for each ``converged`` record in convergence_history.jsonl.

    The sidecar does not record the realized improvement, only the EI — the record
    exists at all only because both gates passed.
    """
    path = run_dir / "convergence_history.jsonl"
    if not path.exists():
        return []
    events: list[tuple[float, float, float]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("converged") and rec.get("ts") is not None:
                ei = rec.get("ei")
                events.append((float(rec["ts"]),
                               float(ei) if ei is not None else float("nan"),
                               float("nan")))
    return events


def convergence_events(run_dir: Path, timeline: list[tuple[float, int, str]],
                       n_samples: int) -> list[ConvEvent]:
    """Satisfied-criteria events placed on the sample axis, de-duplicated.

    Both sources describe the same underlying checks where they overlap, so the
    union is taken over sample index. ``run.log`` records win ties: they are the
    only ones carrying ``improvement``, which the counterfactual needs.
    """
    by_sample: dict[int, ConvEvent] = {}
    for source, raw in (("sidecar", converged_events_from_sidecar(run_dir)),
                        ("log", converged_events_from_log(run_dir))):
        for ts, ei, imp in raw:
            i = _sample_index_at(ts, timeline)
            if i is None or not (0 <= i < n_samples):
                continue
            prev = by_sample.get(i)
            if prev is None or (prev.source == "sidecar" and source == "log"):
                by_sample[i] = ConvEvent(sample=i, ts=ts, ei=ei,
                                         improvement=imp, source=source)
    return [by_sample[i] for i in sorted(by_sample)]


def convergence_samples(run_dir: Path, timeline: list[tuple[float, int, str]],
                        n_samples: int) -> np.ndarray:
    """Sorted, de-duplicated sample indices where the convergence criteria held."""
    return np.array([e.sample for e in convergence_events(run_dir, timeline, n_samples)],
                    dtype=int)


# ── counterfactual over the noise floor ───────────────────────────────────────
def run_output_noise_mult(run_dir: Path) -> float:
    """The ``output_noise_threshold_mult`` the run actually used."""
    path = run_dir / "hparams_effective.json"
    if path.exists():
        try:
            return float(json.loads(path.read_text())["output_noise_threshold_mult"])
        except Exception:
            pass
    return _FALLBACK_MULT


def implied_output_noise(events: list[ConvEvent], actual_mult: float) -> float:
    """Lower bound on the run's GP output noise, from the events it accepted.

    The GP noise is not persisted anywhere, but every logged event cleared
    ``max(EI, improvement) < output_noise * actual_mult``, so the worst event pins
    the noise from below. Using that bound as the assumed noise makes counterfactual
    counts the most permissive consistent with what the run did.
    """
    reqs = [e.requirement for e in events if np.isfinite(e.requirement)]
    if not reqs or actual_mult <= 0:
        return float("nan")
    return max(reqs) / actual_mult


def counterfactual_samples(events: list[ConvEvent], floor: float) -> np.ndarray:
    """Sample indices that would still converge with the noise floor at ``floor``."""
    return np.array([e.sample for e in events
                     if np.isfinite(e.requirement) and e.requirement < floor], dtype=int)


def plot_conv_criteria(run_dir: Path, snapshot: str, out_path: Path,
                       mult: float | None = None,
                       output_noise: float | None = None,
                       floor: float | None = None) -> Path:
    """Render the convergence plot with satisfied-criteria rules to ``out_path``.

    ``mult`` / ``output_noise`` / ``floor`` select the counterfactual noise floor;
    each defaults to the run's own value (with ``output_noise`` falling back to the
    lower bound implied by the accepted events, since it is not recorded).
    """
    X, Y = load_run_dataset(run_dir, snapshot)
    Y = np.asarray(Y, dtype=float).ravel()
    n = Y.size
    if n == 0:
        raise SystemExit(f"No datapoints reconstructed from {run_dir} @ {snapshot}")

    acts = run_activations(run_dir, snapshot, n)
    needles = run_needles(run_dir, snapshot, X, Y)
    timeline = snapshot_timeline(run_dir, snapshot)
    events = convergence_events(run_dir, timeline, n)

    actual_mult = run_output_noise_mult(run_dir)
    mult = actual_mult if mult is None else float(mult)
    noise = implied_output_noise(events, actual_mult) if output_noise is None \
        else float(output_noise)
    sel_floor = mult * noise if floor is None else float(floor)
    # The run itself is the identity case: at its own floor nothing is filtered out.
    is_baseline = floor is None and output_noise is None and mult == actual_mult
    conv_idx = (np.array([e.sample for e in events], dtype=int) if is_baseline
                else counterfactual_samples(events, sel_floor))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    idx = np.arange(n)
    ax.scatter(idx, Y, s=10, alpha=0.65, color="steelblue", label="observed", zorder=3)

    # Running best, reset at every activation boundary — each activation is its own
    # disconnected segment, matching run_mobo.plot_convergence(activations=...).
    if acts is not None and len(acts) == n:
        acts = np.asarray(acts).ravel()
        start, labeled = 0, False
        for i in range(1, n + 1):
            if i == n or acts[i] != acts[start]:
                ax.plot(idx[start:i], np.maximum.accumulate(Y[start:i]),
                        color="darkorange", lw=1.8, drawstyle="steps-post", zorder=5,
                        label=(None if labeled else "running best (reset/activation)"))
                labeled = True
                if i < n:
                    ax.axvline(float(i) - 0.5, color="#888888", alpha=0.35,
                               lw=0.7, ls=":", zorder=1)
                start = i
    else:
        ax.plot(idx, np.maximum.accumulate(Y), color="darkorange", lw=1.8,
                drawstyle="steps-post", label="running best", zorder=5)

    # Events the counterfactual drops, drawn faintly so the comparison is visible.
    if not is_baseline:
        dropped = sorted({e.sample for e in events} - set(conv_idx.tolist()))
        for k, di in enumerate(dropped):
            ax.axvline(float(di), color="#bbbbbb", alpha=0.9, lw=1.0, ls=":", zorder=2,
                       label="dropped by counterfactual" if k == 0 else None)

    # Convergence criteria satisfied (EI < noise floor AND Δbest-Y < noise floor).
    for k, ci in enumerate(conv_idx):
        ax.axvline(float(ci), color="green", alpha=0.75, lw=1.1, ls=":", zorder=4,
                   label="convergence criteria met" if k == 0 else None)

    if needles is not None and len(needles):
        for k, (ni, _) in enumerate(np.asarray(needles).reshape(-1, 2)):
            ax.axvline(float(ni), color="crimson", alpha=0.55, lw=0.9, ls="--", zorder=6,
                       label="needle found" if k == 0 else None)

    n_needles = 0 if needles is None else len(needles)
    n_acts = len(np.unique(acts)) if acts is not None else 0
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Objective Y")
    title = (f"{run_dir.name} @ {snapshot}  —  {n} pts, {n_acts} activations, "
             f"{n_needles} needles, {len(conv_idx)}/{len(events)} convergence-criteria events")
    if not is_baseline:
        title += f"  |  mult={mult:g}, floor={sel_floor:.3g}"
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"{len(events)} logged convergence events; "
          f"{len(conv_idx)} survive floor={sel_floor:.4g} "
          f"(mult={mult:g} x output_noise={noise:.4g})")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/run_39af",
                    help="run directory, or a bare run name resolved against runs/")
    ap.add_argument("--snapshot", default=None,
                    help="snapshot to reconstruct (default: the run's latest)")
    ap.add_argument("--out", default=None,
                    help="output PNG (default: <run_dir>/conv_criteria.png)")
    ap.add_argument("--mult", type=float, default=None,
                    help="counterfactual output_noise_threshold_mult "
                         "(default: the run's own value)")
    ap.add_argument("--output-noise", type=float, default=None,
                    help="GP output noise to assume; it is not recorded by the run, so "
                         "the default is the lower bound implied by the accepted events")
    ap.add_argument("--floor", type=float, default=None,
                    help="set the noise floor directly, bypassing mult x output_noise")
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run)
    snapshot = args.snapshot or _default_snapshot(run_dir)
    out = Path(args.out) if args.out else run_dir / "conv_criteria.png"
    path = plot_conv_criteria(run_dir, snapshot, out, mult=args.mult,
                              output_noise=args.output_noise, floor=args.floor)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
