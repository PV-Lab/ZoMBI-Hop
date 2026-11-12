"""Two-head mode's hardware side: one DiSCO rail per head, routed by slot.

Head 0 owns the ``main`` slot, head 1 the ``cache`` slot. ``RailBroker.measure``
writes ONE slot (``communication.write_slot``: a fresh per-slot timestamp, the
other rail untouched) and blocks until that line's result is back, then returns
it. Results are read from the append-only ``objective_queue`` in objective.db
(written by ``communication.objective_receiver``) and routed to the line they
belong to:

1. by ``slot_ts`` when DiSCO echoes it (rails protocol v2,
   docs/disco_independent_rails.md) - exact, independent of timing;
2. otherwise by measured composition: the outstanding line the slide's
   compositions lie closest to (as the axial initialization does), with a warning,
   because a v1 DiSCO re-arms both rails on every write and so re-prints the other
   head's line.

A result for a line its head has stopped waiting for (it arrived after a re-send
or after the head gave up) is not lost: it goes to ``on_extra_data`` (the two-head
hub hands it to both heads as GP data). A line with no result after the stall
timeout is re-sent with a new slot ts (up to ``max_sends``); after that the head
gets an empty result, which ZoMBI-Hop treats like a fully dropped line.

Any thread may pump the queue; a lock makes each row routed exactly once.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from scripts import communication
from src.utils.init_queue import FOREIGN_DISTANCE, _segment_distance

SLOT_OF_HEAD = ("main", "cache")
_ZERO_COMP_TOL = 1e-12


class BrokerStopped(Exception):
    """The run is stopping (another head failed, or the operator stopped it)."""


@dataclass
class RailLine:
    head: int
    slot: str
    left: np.ndarray           # (d,) optimizer-dim endpoints as requested
    right: np.ndarray
    payload: Tuple[np.ndarray, np.ndarray, np.ndarray]   # (start10, end10, array10)
    slot_ts: List[float] = field(default_factory=list)   # one per send
    sent_t: List[float] = field(default_factory=list)
    result: Optional[Tuple[np.ndarray, np.ndarray]] = None
    abandoned: bool = False

    @property
    def done(self) -> bool:
        return self.result is not None or self.abandoned


class RailBroker:
    def __init__(self, optimizing_dims: List[int], *,
                 obj_db: str = "./sql/objective.db",
                 comp_db: str = "./sql/compositions.db",
                 first_timeout_s: float = 240 * 60.0,
                 stall_floor_s: float = 45 * 60.0,
                 stall_mult: float = 3.0,
                 max_sends: int = 3,
                 poll_s: float = 1.0,
                 stop_event: Optional[threading.Event] = None,
                 on_extra_data: Optional[Callable[[int, np.ndarray, np.ndarray], None]] = None,
                 log_path: Optional[Path] = None,
                 log: Callable[[str], None] = print) -> None:
        self.dims = list(optimizing_dims)
        self.obj_db, self.comp_db = obj_db, comp_db
        self.first_timeout_s = float(first_timeout_s)
        self.stall_floor_s = float(stall_floor_s)
        self.stall_mult = float(stall_mult)
        self.max_sends = int(max_sends)
        self.poll_s = float(poll_s)
        self.stop_event = stop_event or threading.Event()
        self.on_extra_data = on_extra_data
        self.log_path = Path(log_path) if log_path is not None else None
        self.log = log
        self._lock = threading.RLock()
        self._by_ts: Dict[float, RailLine] = {}
        self._lines: List[RailLine] = []
        self._last_ts = 0.0
        self.latencies: List[float] = []
        self._warned_v1 = False
        self.n_foreign = 0

    # ── public ───────────────────────────────────────────────────────────────
    def start(self) -> None:
        """Drop results left over from before this session (nothing of ours is
        outstanding yet, so they cannot be attributed)."""
        n = communication.clear_objective_queue(self.obj_db)
        if n:
            self.log(f"[rails] discarded {n} queued result(s) from before this session")

    def abort(self) -> None:
        self.stop_event.set()

    def measure(self, head: int, left: np.ndarray, right: np.ndarray,
                payload: Tuple[np.ndarray, np.ndarray, np.ndarray]
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Print one line on ``head``'s rail and wait for its result.

        ``left``/``right`` are the line endpoints in optimizer dims (for composition
        matching); ``payload`` is (start10, end10, array10) exactly as written.
        Returns (x_meas (k, d) in optimizer dims, y_measured (k,)); k may be 0.
        """
        line = RailLine(head=head, slot=SLOT_OF_HEAD[head],
                        left=np.asarray(left, float), right=np.asarray(right, float),
                        payload=payload)
        with self._lock:
            self._lines.append(line)
            self._send(line)
        while True:
            if self.stop_event.is_set():
                with self._lock:
                    line.abandoned = True
                raise BrokerStopped()
            self.pump()
            with self._lock:
                if line.result is not None:
                    return line.result
                now = time.time()
                if now - line.sent_t[-1] >= self.stall_timeout():
                    if len(line.slot_ts) < self.max_sends:
                        self.log(f"[rails] {line.slot}: no result {self._mins(now - line.sent_t[-1])} "
                                 f"after sending; re-sending (send {len(line.slot_ts) + 1}/"
                                 f"{self.max_sends})")
                        self._event("stall_resend", line)
                        self._send(line)
                    else:
                        line.abandoned = True
                        self._event("abandoned", line)
                        self.log(f"[rails] {line.slot}: no result after {self.max_sends} "
                                 f"send(s); giving up on this line (head gets no points)")
                        d = len(self.dims)
                        return np.zeros((0, d)), np.zeros(0)
            self.stop_event.wait(self.poll_s)

    def stall_timeout(self) -> float:
        """As the init queue: `stall_mult` x the largest observed send->result
        latency (90th percentile once 3 are known), never below the floor; before
        any result, `first_timeout_s`."""
        lat = self.latencies
        if not lat:
            return self.first_timeout_s
        ref = float(np.percentile(lat, 90)) if len(lat) >= 3 else max(lat)
        return max(self.stall_floor_s, self.stall_mult * ref)

    def pump(self) -> None:
        """Route every new queue row to its line."""
        with self._lock:
            rows = communication.read_objective_queue(self.obj_db)
            if not rows:
                return
            communication.consume_objective_queue([r["id"] for r in rows], self.obj_db)
            for r in rows:
                self._route(r)

    # ── internals ────────────────────────────────────────────────────────────
    def _send(self, line: RailLine) -> None:
        ts = max(time.time(), self._last_ts + 1e-3)
        self._last_ts = ts
        s10, e10, a10 = line.payload
        communication.write_slot(line.slot, s10, e10, a10, slot_ts=ts, db_path=self.comp_db)
        line.slot_ts.append(ts)
        line.sent_t.append(time.time())
        self._by_ts[round(ts, 6)] = line
        self._event("send", line, slot_ts=ts)

    def _route(self, row: dict) -> None:
        x, y = self._parse(row)
        line: Optional[RailLine] = None
        how = "slot_ts"
        sent_at = None
        if row.get("slot_ts") is not None:
            line = self._by_ts.get(round(float(row["slot_ts"]), 6))
            if line is not None:
                # latency from THIS copy's send (the original may answer after a re-send)
                k = [round(t, 6) for t in line.slot_ts].index(round(float(row["slot_ts"]), 6))
                sent_at = line.sent_t[k]
        if line is None and row.get("slot_ts") is None:
            if not self._warned_v1:
                self._warned_v1 = True
                self.log("[rails] WARNING DiSCO results carry no slot/slot_ts: DiSCO is "
                         "not on rails protocol v2 (docs/disco_independent_rails.md). "
                         "Routing by composition, but a v1 DiSCO re-prints the other "
                         "rail's line on every write.")
            line, how = self._match_by_composition(x), "composition"
        if line is None:
            self.n_foreign += 1
            self.log(f"[rails] result ({len(y)} points, slot={row.get('slot')}, "
                     f"slot_ts={row.get('slot_ts')}) matches no line sent this session; ignored")
            self._event("foreign", None, row_slot=row.get("slot"), row_slot_ts=row.get("slot_ts"),
                        n=len(y))
            return
        now = time.time()
        if line.result is None and not line.abandoned:
            line.result = (x, y)
            self.latencies.append(now - (sent_at if sent_at is not None else line.sent_t[-1]))
            self._event("result", line, n=len(y), how=how,
                        y=[round(float(v), 6) for v in y])
            self.log(f"[rails] {line.slot} result for head{line.head}: {len(y)} point(s) "
                     f"(routed by {how})")
        else:
            # A second result for the same line (original + re-send both printed)
            # or one that arrived after the head gave up: still real data.
            self._event("extra", line, n=len(y), how=how)
            self.log(f"[rails] late/extra {line.slot} result for head{line.head} "
                     f"({len(y)} points): given to both heads as GP data")
            if self.on_extra_data is not None and len(y):
                self.on_extra_data(line.head, x, y)

    def _parse(self, row: dict) -> Tuple[np.ndarray, np.ndarray]:
        d = len(self.dims)
        y = np.asarray(row["values"], float).ravel()
        comps = row.get("comps")
        if comps is None or np.size(comps) == 0:
            return np.zeros((0, d)), np.zeros(0)
        comps = np.atleast_2d(np.asarray(comps, float))
        n = min(len(y), comps.shape[0])
        y, comps = y[:n], comps[:n]
        ok = np.isfinite(y) & np.isfinite(comps).all(axis=1) & \
            (np.abs(comps).sum(axis=1) > _ZERO_COMP_TOL)
        return comps[ok][:, self.dims], y[ok]

    def _match_by_composition(self, x: np.ndarray) -> Optional[RailLine]:
        cands = [ln for ln in self._lines if not ln.done]
        if not cands or len(x) == 0:
            return cands[0] if (cands and len(x) == 0) else None
        dist = [float(_segment_distance(x, ln.left, ln.right).mean()) for ln in cands]
        i = int(np.argmin(dist))
        return cands[i] if dist[i] <= FOREIGN_DISTANCE else None

    @staticmethod
    def _mins(s: float) -> str:
        return f"{s / 60:.1f} min"

    def _event(self, kind: str, line: Optional[RailLine], **extra) -> None:
        if self.log_path is None:
            return
        rec = {"t": time.time(), "event": kind}
        if line is not None:
            rec.update(head=line.head, slot=line.slot, slot_ts=line.slot_ts[-1] if line.slot_ts else None,
                       sends=len(line.slot_ts),
                       left=[round(float(v), 6) for v in line.left],
                       right=[round(float(v), 6) for v in line.right])
        rec.update(extra)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass
