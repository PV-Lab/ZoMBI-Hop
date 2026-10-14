"""
Result-driven print queue for the initial design on the two-transporter hardware.
=================================================================================

The optimisation loop reacts to every objective result by writing a new (main, cache)
pair to compositions.db. During initialization there is nothing to predict, so this
queue supplies the pair instead: each result sends the next queued traces. Nothing
ever waits for a particular transporter.

What ZoMBI can and cannot see (DiSCO code, Full_System_Integration/communication.py):
  * A transporter takes `main` (else `cache`) only when it reaches the printer. A new
    write replaces BOTH slots, so a slot still unread at that moment is lost.
  * There is no read receipt. The only signal is an objective packet per measured
    slide, arriving long after the print (stability runs on the Prism PC).
  * `objective_sender` strips NaN rows and never sends an empty packet, so a slide
    that fails completely is SILENT -- no result ever arrives for it.

So the queue infers what happened from the results:
  * each packet is matched to the trace its measured compositions lie on (mean
    distance only: DiSCO drops failed rows, and a print's rows bunch up unevenly along
    its line, so how much of the trace a result covers says nothing reliable). A packet
    that lies on no sent trace, or arrives sooner after that trace was sent than any
    print could (a leftover from an earlier run), is ignored;
  * prints follow write order, so once a trace from a later write reports, any trace
    from an earlier write still without a result is presumed lost (overwritten unread,
    or failed silently) and is re-queued at the front, up to `max_sends` sends;
  * a stall watchdog covers the case where nothing reports at all: after a timeout
    sized from the observed result spacing, every outstanding trace is presumed lost.
A packet with no usable rows (if DiSCO ever sends one) completes the oldest
outstanding trace with no data rather than holding the queue.

Pure logic, no I/O: `scripts/run_zombi_main.py` drives it against the databases and
`benchmarks/axial_init/disco_sim.py` against a simulated DiSCO.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .axial_init import AxialTrace

#: Mean point-to-trace distance above which a packet is not from any init trace.
#: Real prints sit 0.058 (median) / 0.22 (p99) from the line they were sent on
#: (runs/*/composition_log.jsonl, 467 packets).
FOREIGN_DISTANCE = 0.25


def _segment_distance(X: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    dd = float(d @ d)
    s = np.zeros(len(X)) if dd == 0 else np.clip((X - a) @ d / dd, 0.0, 1.0)
    return np.linalg.norm(X - (a + s[:, None] * d), axis=1)


class InitQueue:
    """Bookkeeping for the initial design. Trace status: queued -> sent -> done, or
    sent -> queued again (presumed lost) -> ... -> abandoned after `max_sends` sends.
    Times are seconds on whatever clock the caller passes in."""

    def __init__(self, traces: Sequence[AxialTrace], max_sends: int = 3,
                 points_per_trace: int = 24, min_latency: float = 0.0):
        self.traces: List[AxialTrace] = list(traces)
        n = len(self.traces)
        self.max_sends = int(max_sends)
        self.points_per_trace = int(points_per_trace)
        #: no result can come back sooner than this after its trace was sent (seconds)
        self.min_latency = float(min_latency)
        self.status: List[str] = ["queued"] * n
        self.sends: List[int] = [0] * n
        self.sent_in: List[List[int]] = [[] for _ in range(n)]   # write numbers
        self.data: List[List[Dict[str, Any]]] = [[] for _ in range(n)]
        self.queue: deque = deque(range(n))
        self.writes: List[Dict[str, Any]] = []
        self.result_times: List[float] = []
        self.latencies: List[float] = []     # send -> result, matched traces
        self.first_write_t: Optional[float] = None
        self.last_activity_t: Optional[float] = None
        self.log: List[Dict[str, Any]] = []

    # ── queries ──────────────────────────────────────────────────────────────

    def outstanding(self) -> List[int]:
        """Sent traces without a result, oldest write first."""
        out = [i for i, s in enumerate(self.status) if s == "sent"]
        return sorted(out, key=lambda i: (self.sent_in[i][-1], i))

    def complete(self) -> bool:
        return not self.queue and not self.outstanding()

    def measured(self) -> List[Dict[str, Any]]:
        """Every block of usable measurements, in arrival order."""
        blocks = [b for i in range(len(self.traces)) for b in self.data[i] if len(b["y"])]
        return sorted(blocks, key=lambda b: b["t"])

    # ── actions ──────────────────────────────────────────────────────────────

    def next_pair(self, now: float) -> Optional[tuple]:
        """Pop the next (main, cache) trace indices to write, or None if the queue is
        empty. A lone last trace fills both slots (the second copy is a replicate)."""
        while self.queue and self.status[self.queue[0]] == "done":
            self.queue.popleft()
        if not self.queue:
            return None
        main = self.queue.popleft()
        while self.queue and self.status[self.queue[0]] == "done":
            self.queue.popleft()
        cache = self.queue.popleft() if self.queue else main
        no = len(self.writes)
        self.writes.append({"no": no, "main": main, "cache": cache, "t": now})
        for i in {main, cache}:
            self.status[i] = "sent"
            self.sends[i] += 1
            self.sent_in[i].append(no)
        if self.first_write_t is None:
            self.first_write_t = now
        self.last_activity_t = now
        return main, cache

    def on_result(self, x, y, now: float) -> Dict[str, Any]:
        """Register one objective packet (measured x in optimizer dims, measured y).

        Returns {"kind": "matched"|"empty"|"foreign", "trace", "distance",
        "requeued", "abandoned"}. Only matched/empty results should trigger a write.
        """
        x = np.asarray(x, float).reshape(-1, len(self.traces[0].start)) if np.size(x) else \
            np.zeros((0, len(self.traces[0].start)))
        y = np.asarray(y, float).ravel()
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        ok = np.isfinite(y) & np.isfinite(x).all(axis=1) & (np.abs(x).sum(axis=1) > 1e-12)
        x, y = x[ok], y[ok]
        ev: Dict[str, Any] = {"kind": None, "trace": None, "distance": None,
                              "requeued": [], "abandoned": [], "t": now, "n": int(len(y))}
        if len(y) == 0:
            out = self.outstanding()
            if not out:
                ev["kind"] = "foreign"
                self.log.append(ev)
                return ev
            i = out[0]
            ev.update(kind="empty", trace=i)
            self.data[i].append({"x": [], "y": [], "t": now})
        else:
            dist = [float(_segment_distance(x, t.start, t.end).mean()) for t in self.traces]
            i = int(np.argmin(dist))
            ev["distance"] = dist[i]
            too_soon = (bool(self.sent_in[i])
                        and now - self.writes[self.sent_in[i][0]]["t"] < self.min_latency)
            if dist[i] > FOREIGN_DISTANCE or not self.sent_in[i] or too_soon:
                ev["kind"] = "foreign"
                self.log.append(ev)
                return ev
            ev.update(kind="matched", trace=i)
            self.data[i].append({"x": x.tolist(), "y": y.tolist(), "t": now})
            if self.status[i] == "sent":
                self.latencies.append(now - self.writes[self.sent_in[i][-1]]["t"])
        self.result_times.append(now)
        self.last_activity_t = now
        was = self.status[i]
        self.status[i] = "done"
        if was == "queued":
            self.queue = deque(j for j in self.queue if j != i)
        if ev["kind"] == "matched":
            # Prints follow write order: anything sent in an earlier write (or the main
            # of the same write, when this was its cache) that is still unreported was
            # never printed or failed silently.
            k = self.sent_in[i][0]
            same_write_main = self.writes[k]["main"] if self.writes[k]["cache"] == i else None
            lost = [o for o in self.outstanding()
                    if self.sent_in[o][-1] < k or (o == same_write_main and self.sent_in[o][-1] == k)]
            self._presume_lost(lost, ev)
        self.log.append(ev)
        return ev

    def on_stall(self, now: float) -> Dict[str, Any]:
        """Nothing reported for too long: presume every outstanding trace lost."""
        ev = {"kind": "stall", "t": now, "requeued": [], "abandoned": []}
        self._presume_lost(self.outstanding(), ev)
        self.last_activity_t = now
        self.log.append(ev)
        return ev

    def _presume_lost(self, idx: List[int], ev: Dict[str, Any]) -> None:
        requeue = []
        for o in idx:
            if self.sends[o] < self.max_sends:
                self.status[o] = "queued"
                requeue.append(o)
            else:
                self.status[o] = "abandoned"
                ev["abandoned"].append(o)
        # re-send the oldest first, ahead of traces never sent
        self.queue = deque(requeue + [j for j in self.queue if j not in requeue])
        ev["requeued"].extend(requeue)

    def stall_timeout(self, first_timeout: float, floor: float, mult: float) -> float:
        """Seconds of silence after the last write/result before `on_stall`.

        Sized from the observed send -> result latency of matched traces (a full
        print -> anneal -> measure -> stability round trip), not from the spacing
        between results: the two transporters report in bursts minutes apart, while a
        freshly sent pair needs the whole round trip. `mult` x the largest latency
        seen (90th percentile once 3 are known), never below `floor`; before any
        result, `first_timeout`."""
        if not self.latencies:
            return float(first_timeout)
        lat = self.latencies
        ref = float(np.percentile(lat, 90)) if len(lat) >= 3 else max(lat)
        return max(float(floor), float(mult) * ref)

    # ── persistence ──────────────────────────────────────────────────────────

    def to_state(self) -> Dict[str, Any]:
        return {
            "traces": [t.to_dict() for t in self.traces],
            "max_sends": self.max_sends,
            "points_per_trace": self.points_per_trace,
            "min_latency": self.min_latency,
            "status": list(self.status),
            "sends": list(self.sends),
            "sent_in": [list(s) for s in self.sent_in],
            "data": self.data,
            "queue": list(self.queue),
            "writes": self.writes,
            "result_times": list(self.result_times),
            "latencies": list(self.latencies),
            "first_write_t": self.first_write_t,
        }

    @classmethod
    def from_state(cls, st: Dict[str, Any], resume: bool = True) -> "InitQueue":
        """Rebuild from `to_state`. With `resume`, abandoned traces get a fresh set of
        sends; sent-but-unreported ones stay outstanding (their prints may still
        report) and are re-queued by the normal rules if they don't."""
        q = cls([AxialTrace.from_dict(t) for t in st["traces"]], st.get("max_sends", 3),
                st.get("points_per_trace", 24), st.get("min_latency", 0.0))
        q.status = list(st["status"])
        q.sends = list(st["sends"])
        q.sent_in = [list(s) for s in st["sent_in"]]
        q.data = st["data"]
        q.queue = deque(st["queue"])
        q.writes = st["writes"]
        q.result_times = list(st["result_times"])
        q.latencies = list(st.get("latencies", []))
        q.first_write_t = st.get("first_write_t")
        if resume:
            for i, s in enumerate(q.status):
                if s == "abandoned":
                    q.status[i] = "queued"
                    q.sends[i] = 0
                    q.queue.append(i)
        return q
