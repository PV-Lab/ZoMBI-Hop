"""Two-head mode: two independent ZoMBI-Hop instances on one landscape.

Each head is an ordinary ``ZoMBIHop`` with its own DataHandler (own run
directory, activation/zoom state, needle list and penalty mask) and its own
hardware rail. The heads interact in exactly two ways:

* **Points (any time).** Every line a head measures is handed to the other
  head, which adds it to its data as a *foreign* row the next time it finishes
  a measurement (``DataHandler.add_all_points(..., foreign=True)``). Foreign rows
  feed paring and the GP, as the GP does not care which activation a point came
  from, but they never count as the receiving head's activation data: zoom
  windows and needle declarations see only the head's own rows
  (``DataHandler.current_activation_mask``).

* **Penalty regions (between activations only).** A needle or exclusion zone
  declared by one head is published to the hub but adopted by the other head
  only when that head starts its next activation, so a head is never thrown off
  course mid-activation by a region the other head just placed. Both heads may
  therefore declare the same needle; both are kept (each needle records its
  ``origin_head``).

Nothing else is shared. In particular there is no divergence mechanism; with the
same GP the heads can chase the same peak until one of them declares it.

The hub is hardware-agnostic: the objective each head calls decides where its
line is measured (scripts/rail_broker.py routes each head to its own DiSCO rail).
All DataHandler mutation happens on the owning head's thread; the hub only moves
CPU copies of tensors between heads under a lock.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

HEAD_TAGS = ("head0/main", "head1/cache")


def _cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", torch.float64).clone()


class HeadLink:
    """One head's connection to the hub (installed as ``zombi.two_head_link``)."""

    def __init__(self, hub: "TwoHeadHub", head_id: int, zombi) -> None:
        self.hub = hub
        self.head_id = head_id
        self.zombi = zombi
        self.tag = HEAD_TAGS[head_id] if head_id < len(HEAD_TAGS) else f"head{head_id}"
        self.n_foreign_points = 0
        self.n_imported_regions = 0

    # Called by ZoMBIHop._objective_wrapper right after its own points are stored.
    def after_measurement(self, X_actual: torch.Tensor, X_expected: torch.Tensor,
                          Y: torch.Tensor) -> None:
        self.hub.publish_points(self.head_id, X_actual, X_expected, Y)
        self.drain_points()

    # Called by ZoMBIHop.run at the top of every activation.
    def at_activation_boundary(self, sync_regions: bool = True) -> None:
        self.drain_points()
        if sync_regions:
            self.import_regions()

    def drain_points(self) -> int:
        dh = self.zombi.data_handler
        n = 0
        for X_a, X_e, Y in self.hub.take_points(self.head_id):
            dev = dict(device=dh.device, dtype=dh.dtype)
            dh.add_all_points(X_a.to(**dev), X_e.to(**dev), Y.to(**dev).reshape(-1, 1),
                              foreign=True)
            n += X_a.shape[0]
        if n:
            self.n_foreign_points += n
            self.zombi._log(f"  [two-head] took in {n} point(s) measured by the other "
                            f"head (GP data only; {self.n_foreign_points} so far)")
        return n

    def import_regions(self) -> int:
        dh = self.zombi.data_handler
        n = 0
        for origin, rec in self.hub.take_regions(self.head_id):
            if _has_region(dh, rec["center"]):
                continue
            dh.import_region(rec, origin)
            n += 1
            self.zombi._log(
                f"  [two-head] adopted a {rec['kind']} from head{origin} at "
                f"{[round(v, 4) for v in rec['center'].tolist()]}"
                + (f" (Y={rec['value']:.4f})" if rec.get("value") is not None else ""))
        if n:
            self.n_imported_regions += n
            dh._update_penalty_mask()
        return n


def _has_region(dh, center: torch.Tensor, tol: float = 1e-9) -> bool:
    """True if ``dh`` already has a needle or exclusion centred at ``center``."""
    c = center.to(dh.device, dh.dtype).reshape(1, -1)
    for T in (dh.needles, dh.exclusions):
        if T is not None and T.shape[0] > 0 and bool(
                (torch.norm(T - c, dim=1) <= tol).any()):
            return True
    return False


class TwoHeadHub:
    """Shared mailbox between the heads of one run (thread-safe).

    ``on_regions_changed(records)`` is called (from the declaring head's thread)
    with every region published so far, e.g. to mirror the union of both heads'
    needles into sql/needles.db.
    """

    def __init__(self, n_heads: int = 2,
                 on_regions_changed: Optional[Callable[[List[Tuple[int, Dict[str, Any]]]], None]] = None,
                 log: Callable[[str], None] = print) -> None:
        self.n_heads = n_heads
        self._lock = threading.Lock()
        self._inbox: List[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [
            [] for _ in range(n_heads)]
        self._regions: List[Tuple[int, Dict[str, Any]]] = []
        self._synced = [0] * n_heads
        self.on_regions_changed = on_regions_changed
        self.log = log
        self.links: List[Optional[HeadLink]] = [None] * n_heads
        self.stop_event = threading.Event()

    # ── wiring ──────────────────────────────────────────────────────────────
    def attach(self, head_id: int, zombi) -> HeadLink:
        link = HeadLink(self, head_id, zombi)
        self.links[head_id] = link
        zombi.two_head_link = link
        dh = zombi.data_handler
        dh.region_listener = lambda rec, h=head_id: self.publish_region(h, rec)
        # A head's own rows: everything not foreign. Mark the mask as live so new
        # rows get an explicit False/True entry from here on.
        if dh._foreign_rows is None:
            dh._foreign_rows = dh.foreign_mask()
        return link

    # ── points ──────────────────────────────────────────────────────────────
    def publish_points(self, origin: Optional[int], X_actual: torch.Tensor,
                       X_expected: torch.Tensor, Y: torch.Tensor) -> None:
        """Queue a batch for every head except ``origin`` (None = every head, e.g.
        a late result whose head has moved on)."""
        if X_actual.shape[0] == 0:
            return
        item = (_cpu(X_actual), _cpu(X_expected), _cpu(Y).reshape(-1))
        with self._lock:
            for h in range(self.n_heads):
                if h != origin:
                    self._inbox[h].append(item)

    def take_points(self, head_id: int):
        with self._lock:
            items, self._inbox[head_id] = self._inbox[head_id], []
        return items

    # ── regions ─────────────────────────────────────────────────────────────
    def publish_region(self, origin: int, rec: Dict[str, Any]) -> None:
        with self._lock:
            self._regions.append((origin, rec))
            snapshot = list(self._regions)
        if self.links[origin] is not None:
            self.links[origin].zombi._log(
                f"  [two-head] published {rec['kind']} to the other head (adopted at "
                f"its next activation boundary)")
        self._notify(snapshot)

    def take_regions(self, head_id: int) -> List[Tuple[int, Dict[str, Any]]]:
        with self._lock:
            new = self._regions[self._synced[head_id]:]
            self._synced[head_id] = len(self._regions)
        return [(o, r) for o, r in new if o != head_id]

    def regions(self) -> List[Tuple[int, Dict[str, Any]]]:
        with self._lock:
            return list(self._regions)

    def _notify(self, snapshot) -> None:
        if self.on_regions_changed is None:
            return
        try:
            self.on_regions_changed(snapshot)
        except Exception as e:      # noqa: BLE001 — mirroring must not kill a head
            self.log(f"[two-head] on_regions_changed failed: {e!r}")

    # ── resume ──────────────────────────────────────────────────────────────
    def reconcile(self, heads: Sequence) -> None:
        """After both heads are loaded from their checkpoints: re-offer each head's
        own rows to the other (``add_all_points`` drops exact duplicates, so rows it
        already has are skipped) and rebuild the region list from both heads'
        penalty sets (importing is idempotent by centre). Regions are adopted at
        each head's next activation boundary, as usual."""
        for h, z in enumerate(heads):
            dh = z.data_handler
            n_init = int(getattr(dh, "_n_init_points", 0) or 0)
            own = ~dh.foreign_mask()
            own[:n_init] = False
            if own.any():
                self.publish_points(h, dh.X_all_actual[own], dh.X_all_expected[own],
                                    dh.Y_all[own])
        for h, z in enumerate(heads):
            for rec in region_records(z.data_handler):
                origin = rec.pop("_origin", None)
                self._regions.append((h if origin is None else origin, rec))
        self._notify(self.regions())


def region_records(dh) -> List[Dict[str, Any]]:
    """Every penalty region in ``dh`` as hub records (needles first)."""
    out: List[Dict[str, Any]] = []
    B = None if dh.needle_B is None else _cpu(dh.needle_B)
    n = dh.needles.shape[0] if dh.needles is not None else 0
    for i in range(n):
        res = dh.needles_results[i] if i < len(dh.needles_results) else {}
        M = dh.needle_M_list[i] if i < len(dh.needle_M_list) else None
        out.append({
            "kind": "needle", "center": _cpu(dh.needles[i]),
            "radius": float(dh.needle_penalty_radii[i].item()),
            "M": None if M is None else _cpu(M), "B": B,
            "value": float(dh.needle_vals[i].item()),
            "median_value": res.get("median_value"),
            "zoom": res.get("zoom", -1), "iteration": res.get("iteration", -1),
            "reason": res.get("reason"), "_origin": res.get("origin_head"),
        })
    m = dh.exclusions.shape[0] if dh.exclusions is not None else 0
    for i in range(m):
        M = dh.exclusion_M_list[i] if i < len(dh.exclusion_M_list) else None
        out.append({"kind": "exclusion", "center": _cpu(dh.exclusions[i]),
                    "radius": float(dh.exclusion_radii[i].item()),
                    "M": None if M is None else _cpu(M), "B": B})
    return out


def run_heads(heads: Sequence, hub: TwoHeadHub, run_kwargs: Dict[str, Any],
              on_abort: Optional[Callable[[], None]] = None) -> None:
    """Run every head's ``ZoMBIHop.run(**run_kwargs)`` on its own thread.

    Returns when every head has returned. If a head raises, the hub's
    ``stop_event`` is set and ``on_abort`` is called (the rail broker uses both
    to release the other head from its hardware wait), and the first error is
    re-raised here once the threads are gone.
    """
    errors: List[Tuple[int, BaseException]] = []

    def _target(h: int, z) -> None:
        try:
            z.run(**run_kwargs)
            hub.log(f"[two-head] head{h} finished")
        except BaseException as e:  # noqa: BLE001 — report on the main thread
            errors.append((h, e))
            hub.stop_event.set()
            if on_abort is not None:
                try:
                    on_abort()
                except Exception:
                    pass

    threads = [threading.Thread(target=_target, args=(h, z), name=f"zombi-head{h}",
                                daemon=True) for h, z in enumerate(heads)]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.5)
            if errors:
                # Give the other head a moment to unwind from its wait.
                for t in threads:
                    t.join(timeout=30.0)
                break
    except KeyboardInterrupt:
        hub.stop_event.set()
        if on_abort is not None:
            on_abort()
        raise
    if errors:
        h, e = errors[0]
        raise RuntimeError(f"two-head run: head{h} failed: {e!r}") from e
