"""The shared store of a lanes engine: one dataset, two lanes (DESIGN.md 1.2, 1.3.1).

``LaneEngine`` is the sandbox's ``LaneStore`` (``benchmarks/zoom_fix/lanes/lane_zombihop.py`` = LZ
238-262, 265-592) moved into production. One ``LaneZoMBIHop`` (``engine.z``) owns the shared
``DataHandler`` (every measured row, needles, exclusion zones) and the GP handler; each lane
(``src/lanes/lane.py``) keeps its own activation / zoom / box state. Only one lane computes at a time;
``engine.context(lane)`` exports the lane's box / activation / zoom into the DataHandler and makes
``engine.active`` the lane whose own rows ``current_activation_mask`` returns.

Changes against the sandbox, all from DESIGN 1.3.1 (every other line is LZ verbatim):
  * imports from ``src.lanes.*`` (no ``benchmarks.*``);
  * ``PF.fix_rng()`` (LZ 274) -> ``self._rng = SeededSimplex.install()`` at the same place (before the
    ZoMBIHop is built); ``close()`` uninstalls it (Fix 4 scoped to the engine's lifetime, N7);
  * the ``opts.input_noise`` override (LZ 276-278) is gone (input_noise is a core hparam);
  * ``ZoMBIHop(objective=_no_objective, ...)`` + the instance patches (LZ 289-296, 322-324, 333) ->
    ``LaneZoMBIHop(engine=self, ...)`` whose ``attach`` swaps the DataHandler / GPSimplex to the lane
    subclasses (``src/lanes/handlers.py``); the LineBO (LZ 325-332) -> ``LaneLineBO`` (Fix 8 + the line
    gate + the "lines" timer, ``src/lanes/linebo.py``);
  * removed: ``needle_pending`` (LZ 308), ``penalty_mask_excluding`` (LZ 401-407), ``_install_timers``
    (LZ 429-460), the ``LaneSystem`` alias (LZ 1427);
  * added: ``timers`` (``comp`` is ``timers.comp``), ``stats`` + ``count()`` (the explore framework's
    ``ctx.count``), ``zb_transitions`` (the zbudget transition log, ZB 257-261), ``commit()`` (one
    DataHandler snapshot per ingested packet, at the END of ``ingest``, and one per lane decision, N6),
    ``summary()`` also reports ``fit_rng_consumed`` / ``fit_skip_after_rng``.

The attribute and method names the sandbox harness (``lane_system.py``), the auditor
(``src/lanes/audit.py`` = ``lane_selftest.audit``) and the logs read are kept (DESIGN 1.1, REVIEW F11):
``lanes, dh, z, gp, d, full, n_init, hp, opts, rows (.a / .t()), packets, events, audit, zones,
line_by_rid, acts, seed, predict, stats, fit_stats, region_index`` and ``ingest, on_result, tags_of,
summary, conservation, close, event, audit_rows, add_zone, coverage, context, new_act, new_line_id,
other``.

Nothing here draws RNG outside the sandbox's own draws (the cover rule under ``fork_rng``, LZ 386-399).
No import-time side effects (REVIEW F12).
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import time
import weakref
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.lanes import labels
from src.lanes.geometry import DEV, DT, LANE_CODE, LANES, x_requested
from src.lanes.handlers import LaneZoMBIHop
from src.lanes.instrument import Timers
from src.lanes.linebo import LaneLineBO
from src.lanes.lines import LineRec
from src.lanes.options import LaneOptions
from src.lanes.rng import SeededSimplex

__all__ = ["RowTags", "LaneEngine"]


# ── per-row provenance (LZ 238-262 verbatim) ─────────────────────────────────────────────────

class RowTags:
    """Per-row provenance of dh.X_all_actual (aligned; init rows lane -1)."""
    FIELDS = ("lane", "act", "zoom", "eff", "line", "bridge", "pkt")

    def __init__(self, n0: int):
        self.a = {k: np.full(n0, -1, dtype=np.int64) for k in self.FIELDS}
        self.a["bridge"][:] = 0
        self._cache: dict = {}

    def __len__(self):
        return int(self.a["lane"].shape[0])

    def append(self, n: int, **vals):
        for k in self.FIELDS:
            self.a[k] = np.concatenate([self.a[k], np.full(n, int(vals.get(k, -1)), dtype=np.int64)])
        self._cache.clear()

    def set(self, field_, idx, val):
        self.a[field_][np.asarray(idx, dtype=np.int64)] = int(val)
        self._cache.clear()

    def t(self, field_) -> torch.Tensor:
        if field_ not in self._cache:
            self._cache[field_] = torch.as_tensor(self.a[field_], device=DEV)
        return self._cache[field_]


_RowTags = RowTags            # the sandbox name (LZ 238)


def _no_rmtree_error(path):
    shutil.rmtree(path, True)


# ── the engine (= sandbox LaneStore, LZ 265-592) ─────────────────────────────────────────────

class LaneEngine:
    """The shared dataset + both lanes. ``engine.lanes['front']`` / ``engine.lanes['back']``.

    Built by ``src.lanes.engine.build_lane_engine`` (DESIGN 2.1), which resolves the options and the
    core hyperparameters and passes ``lane_cls=LeaderLane``. ``hparams=None`` -> ``engine_hparams(d)``;
    ``lane_cls=None`` -> ``src.lanes.engine.LeaderLane`` (the validated leader).

    ``defer_decision_commit``: False (Phase 1 / emulator) -> each lane decision commits its own snapshot
    in ``DiveLane._finish``; True (Phase-2 hardware / DES controller) -> the controller commits after the
    wire ids are issued (DESIGN 6.5).
    """

    def __init__(self, X_init_actual, X_init_expected, Y_init, bounds, *, options: LaneOptions,
                 hparams: Optional[dict] = None, seed: int = 0, checkpoint_dir: Optional[str] = None,
                 run_uuid: Optional[str] = None, resume: bool = False, predict=None, verbose: bool = False,
                 lanes=LANES, defer_decision_commit: bool = False, lane_cls=None):
        if not isinstance(options, LaneOptions):
            raise TypeError(f"LaneEngine: options must be a src.lanes.options.LaneOptions, got {type(options)!r}")
        self.opts = options
        self.seed = seed
        self.predict = predict                      # noise-free landscape (logging only)
        self.defer_decision_commit = bool(defer_decision_commit)
        # instrumentation and counters exist before z is built (handlers.attach looks them up)
        self.timers = Timers()
        self.comp = self.timers.comp                # LZ 318 (exclusive seconds per component)
        self.fit_stats: dict = {}                   # LZ 319
        self.stats: dict = {}                       # ctx.count of the explore methods; line_gate (LaneLineBO)
        self.zb_transitions: List[dict] = []        # ZB 257-261 (capped at budget.TRANSITION_LOG_CAP)
        self.active = None                          # LZ 317 (set early: the mask reads it)
        self.build_info: dict = {}
        self._snapshot_warned: Optional[str] = None
        self._tmp = None
        self._rng = SeededSimplex.install()         # Fix 4 (LZ 274: PF.fix_rng()), scoped (N7)
        weakref.finalize(self, self._rng.uninstall)  # a never-closed engine still releases Fix 4 at GC
        try:
            self._build(X_init_actual, X_init_expected, Y_init, bounds, hparams=hparams,
                        checkpoint_dir=checkpoint_dir, run_uuid=run_uuid, resume=resume, verbose=verbose,
                        lanes=lanes, lane_cls=lane_cls)
        except BaseException:
            self.close()
            raise

    def _build(self, X_init_actual, X_init_expected, Y_init, bounds, *, hparams, checkpoint_dir, run_uuid,
               resume, verbose, lanes, lane_cls):
        d = int(X_init_actual.shape[1])
        if hparams is None:
            from src.lanes.engine import engine_hparams
            hparams = engine_hparams(d)
        hp = dict(hparams)
        self.hp = hp
        if checkpoint_dir is None and self.opts.snapshots:
            # snapshots as run() takes them (compute parity), in a scratch dir outside the repo that
            # is removed by close() or, at the latest, when the engine is garbage-collected / at exit
            self._tmp = tempfile.mkdtemp(prefix="lanes_ckpt_")
            weakref.finalize(self, _no_rmtree_error, self._tmp)
            checkpoint_dir = self._tmp

        bounds = torch.as_tensor(bounds, device=DEV, dtype=DT)
        # LZ 289-296: ZoMBIHop with an objective that raises; LaneZoMBIHop.__init__ then attaches the
        # DataHandler / GPSimplex to this engine (replaces LZ 322-324 and the dh / gp / z timers of 333)
        self.z = LaneZoMBIHop(engine=self, X_init_actual=X_init_actual.to(DEV, DT),
                              X_init_expected=X_init_expected.to(DEV, DT), Y_init=Y_init.to(DEV, DT).reshape(-1, 1),
                              bounds=bounds, device=str(DEV), dtype=DT, checkpoint_dir=checkpoint_dir,
                              run_uuid=run_uuid, resume=resume, verbose=verbose, **hp)
        self.dh = self.z.data_handler
        self.gp = self.z.gp_handler
        self.d = d
        self.full = self.z.full_bounds.clone()
        self.n_init = int(self.dh.X_all_actual.shape[0])
        self.rows = RowTags(self.n_init)
        self.rows_at_version = [self.n_init]
        self.version = 0                  # +1 per ingested packet
        self.penalty_version = 0          # +1 per needle / zone added, moved or shrunk
        self.zones: List[dict] = []       # needles and exclusion zones with their owner
        self.zone_seq = 0
        self.packets: List[dict] = []
        self._pkt_by_key: Dict[str, int] = {}
        self.audit: List[dict] = []
        self.events: List[dict] = []
        self.line_by_rid: Dict[str, LineRec] = {}
        self._line_counter = 0
        self._act_counter = 0
        self.acts: Dict[int, dict] = {}

        # LZ 325-332 (+ the line gate XR 87-134 and the "lines" timer LZ 453, both inside LaneLineBO)
        zref = [self.z]
        self.linebo = LaneLineBO(None, d, num_points_per_line=100, num_lines=self.opts.num_lines,
                                 device=str(DEV), zombi_ref=zref, engine=self, incumbent=self.opts.incmean)
        # the gate counters exist from the build on, as the sandbox's build.gate_stats did (XR 95-96)
        self.stats["line_gate"] = self.linebo.gate_stats
        if lane_cls is None:
            from src.lanes.engine import LeaderLane
            lane_cls = LeaderLane
        self.lane_cls = lane_cls
        self.lanes: Dict[str, object] = {}
        for name in lanes:                # front then back: activation ids front 0, back 1 (XR 143-150)
            self.lanes[name] = lane_cls(self, name)

    # ── bookkeeping ──
    def close(self):
        """Remove the scratch snapshot dir (LZ 339-342) and release Fix 4 (idempotent)."""
        if self._tmp:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None
        rng = getattr(self, "_rng", None)
        if rng is not None:
            rng.uninstall()

    def count(self, key: str, n=1) -> None:
        """A method counter (the explore framework's ``ctx.count``, FW 626-627)."""
        self.stats[key] = self.stats.get(key, 0) + n

    def new_act(self, lane: str) -> int:
        a = self._act_counter
        self._act_counter += 1
        self.acts[a] = dict(lane=lane, row0=int(self.dh.X_all_actual.shape[0]), zone_seq0=self.zone_seq)
        return a

    def new_line_id(self) -> int:
        self._line_counter += 1
        return self._line_counter

    def other(self, lane) -> Optional[object]:
        for n, ln in self.lanes.items():
            if ln is not lane:
                return ln
        return None

    def event(self, **ev):
        ev.setdefault("seq", len(self.events))
        ev.setdefault("n_rows", int(self.dh.X_all_actual.shape[0]))
        ev.setdefault("pkt_seen", len(self.packets))
        self.events.append(ev)
        return ev

    def audit_rows(self, kind: str, lane, rows, **extra):
        rows = [int(r) for r in (rows.tolist() if hasattr(rows, "tolist") else rows)]
        self.audit.append(dict(kind=kind, lane=lane.name, act=lane.act, zoom=lane.zoom, rows=rows,
                               seq=len(self.events), **extra))

    def add_zone(self, kind: str, lane, index: int, **extra) -> dict:
        self.zone_seq += 1
        self.penalty_version += 1
        z = dict(seq=self.zone_seq, kind=kind, lane=lane.name, act=lane.act, index=int(index), **extra)
        self.zones.append(z)
        return z

    def region_index(self, zone: dict) -> int:
        """Index of a zone record in dh._penalty_regions() (needles first, then exclusions)."""
        if zone["kind"] == "needle":
            return zone["index"]
        n_needles = int(self.dh.needles.shape[0]) if self.dh.needles is not None else 0
        return n_needles + zone["index"]

    def coverage(self, box: torch.Tensor, zones: List[dict], n: int = 1024) -> float:
        """Share of the box (uniform on box ∩ simplex) inside any of ``zones``."""
        if not zones:
            return 0.0
        # its own RNG stream: how often the cover rule is evaluated must not shift the decisions' draws
        devs = [torch.cuda.current_device()] if DEV.type == "cuda" else []
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(1_000_003 * (self.seed + 1) + 7919 * self.penalty_version + len(zones))
            S = self.z.random_sampler(n, box[0], box[1], device=str(DEV), torch_dtype=DT).to(DEV, DT)
        centers, radii, M_list = self.dh._penalty_regions()
        inside = torch.zeros(S.shape[0], dtype=torch.bool, device=DEV)
        for zr in zones:
            inside |= self.dh._inside_region(S, centers, radii, M_list, self.region_index(zr))
        return float(inside.double().mean())

    def _current_activation_mask(self) -> torch.Tensor:
        # LZ 409-414 (reached through LaneDataHandler.current_activation_mask)
        n = int(self.dh.X_all_actual.shape[0])
        ln = self.active
        if ln is None or len(self.rows) != n:
            return torch.zeros(n, dtype=torch.bool, device=DEV)
        return ln.own_act_mask()

    @contextmanager
    def context(self, lane):
        prev = self.active
        self.active = lane
        lane._sync_dh()
        try:
            yield
        finally:
            self.active = prev
            if prev is not None:
                prev._sync_dh()

    # ── commits (N6; Phase 2 replaces the body with the commit protocol of DESIGN 6.5) ──
    def commit(self, kind: str, label: str, info=None) -> None:
        """One engine transition is complete in memory: an ingested packet (``kind="packet"``, called as the
        last step of ``ingest``) or a lane decision (``kind="decision"``, from ``DiveLane._finish``).

        Phase 1: one DataHandler snapshot named ``label`` (``src.lanes.labels``) when ``opts.snapshots``,
        with NO activation / zoom / iteration kwargs (decision-neutral: ``take_snapshot`` draws no RNG and,
        without kwargs, changes nothing a decision reads). Never a store event (REVIEW F1)."""
        if not self.opts.snapshots:
            return
        self.dh.take_snapshot(label)
        self._check_snapshot_limit()

    def _check_snapshot_limit(self) -> None:
        # REVIEW F10: snapshot dirs sort by name; warn from 9000 on (stderr, once per message class)
        msg = labels.snapshot_limit_warning(int(getattr(self.dh, "_snapshot_count", 0) or 0))
        if msg is not None and (self._snapshot_warned is None or msg[:5] != self._snapshot_warned[:5]):
            self._snapshot_warned = msg
            print(f"[lanes] {msg}", file=sys.stderr, flush=True)

    # ── packets ──
    def _norm(self, packet) -> dict:
        get = (lambda k, dflt=None: packet.get(k, dflt)) if isinstance(packet, dict) else \
            (lambda k, dflt=None: getattr(packet, k, dflt))
        x, y, y_true = get("x"), get("y"), get("y_true")
        if isinstance(x, dict):              # disco_des measure payload (zombi_adapter style)
            y_true = x.get("y_true", y_true)
            x = x["x"]
        x = torch.as_tensor(x, device=DEV, dtype=DT).reshape(-1, self.d)
        y = torch.as_tensor(y, device=DEV, dtype=DT).reshape(-1)
        assert x.shape[0] == y.shape[0], "packet x / y length mismatch"
        key = get("pid")
        if key is None:
            h = hashlib.sha1()
            h.update(np.round(x.detach().cpu().numpy(), 12).tobytes())
            h.update(np.round(y.detach().cpu().numpy(), 12).tobytes())
            key = h.hexdigest()
        t = get("t_zombi", None)
        t = get("t", None) if t is None else t
        return dict(x=x, y=y, y_true=None if y_true is None else np.asarray(y_true, float).reshape(-1),
                    key=str(key), t=t)

    def ingest(self, packet, tags: Optional[dict] = None, now=None) -> Tuple[dict, bool]:
        """Add a packet's rows to the shared data once (idempotent); the owning lane classifies it.
        LZ 484-546 verbatim, then the packet commit (DESIGN 1.3.1 item 8) as the LAST step."""
        t0 = time.perf_counter()
        pk = self._norm(packet)
        if pk["key"] in self._pkt_by_key:
            return self.packets[self._pkt_by_key[pk["key"]]], False
        dh = self.dh
        tags = dict(tags) if tags else None
        owner = self.lanes.get(tags.get("lane")) if tags else None
        line = owner.lines.get(tags.get("line")) if (owner is not None and tags.get("line") is not None) else None
        if owner is not None and line is None and tags.get("rid") in self.line_by_rid:
            line = self.line_by_rid[tags["rid"]]
        X = self.z.proj_fn(pk["x"])
        Y = pk["y"]
        Xe = x_requested(X)
        n_in = int(X.shape[0])
        novel = torch.ones(n_in, dtype=torch.bool, device=DEV)
        if dh.X_all_actual.shape[0] > 0 and n_in > 0:
            dist = torch.cdist(X.to(torch.float64), dh.X_all_actual.to(torch.float64))
            dy = torch.abs(Y.reshape(-1, 1) - dh.Y_all.reshape(1, -1).to(torch.float64))
            novel = ~((dist < 1e-5) & (dy < 1e-5)).any(dim=1)
        n0 = int(dh.X_all_actual.shape[0])
        saved = dh.current_zoom_bounds
        if owner is not None:                 # paring radius: the owner's zoom box
            dh.current_zoom_bounds = owner.box.clone()
        try:
            if bool(novel.any()):
                dh.add_all_points(X[novel], Xe[novel], Y[novel].reshape(-1, 1))
        finally:
            dh.current_zoom_bounds = saved
        n1 = int(dh.X_all_actual.shape[0])
        n_new = n1 - n0
        assert n_new == int(novel.sum()), "DataHandler dropped rows the store did not expect"
        pid = len(self.packets)
        lane_code = LANE_CODE[owner.name] if owner is not None else -1
        act = line.act if line is not None else -1
        zoom = line.zoom if line is not None else -1
        self.rows.append(n_new, lane=lane_code, act=act, zoom=zoom, eff=zoom,
                         line=(line.line_id if line is not None else -1),
                         bridge=int(bool(line.bridge)) if line is not None else 0, pkt=pid)
        unpen = dh._penalty_mask[n0:n1]
        rec = dict(pid=pid, key=pk["key"], t=(now if now is not None else pk["t"]),
                   lane=(owner.name if owner is not None else None), act=act, zoom=zoom,
                   line=(line.line_id if line is not None else None),
                   rid=(line.rid if line is not None else (tags or {}).get("rid")),
                   bridge=bool(line.bridge) if line is not None else False,
                   kind=(tags or {}).get("kind", "regular" if line is not None else "unattributed"),
                   row0=n0, n=n_new, n_in=n_in, n_dup=n_in - n_new,
                   y=[round(float(v), 6) for v in Y[novel].tolist()],
                   y_true=(None if pk["y_true"] is None else [float(v) for v in pk["y_true"][novel.cpu().numpy()]]),
                   unpen_at_meas=int(unpen.sum()), cls=None, counted_zoom=None, processed_seq=None)
        self.packets.append(rec)
        self._pkt_by_key[pk["key"]] = pid
        self.version += 1
        self.rows_at_version.append(n1)
        self.comp["ingest"] += time.perf_counter() - t0
        if owner is not None and line is not None:
            owner._on_own_packet(rec, line)
        elif owner is not None:
            rec["cls"] = "unknown_line"
        else:
            rec["cls"] = "init" if (tags or {}).get("kind") == "init" else "unattributed"
        # N6: one snapshot per ingested packet (any class), labelled by the line's act / SENT zoom / lane
        self.commit("packet", labels.packet_label(rec), rec)
        return rec, True

    def on_result(self, packet, tags: Optional[dict] = None, now=None) -> str:
        rec, _ = self.ingest(packet, tags, now)
        return rec["cls"]

    def tags_of(self, rid: str) -> Optional[dict]:
        r = self.line_by_rid.get(rid)
        return None if r is None else r.tags()

    # ── reports ──
    def conservation(self) -> dict:
        dh = self.dh
        rows = int(dh.X_all_actual.shape[0])
        Y = dh.Y_all.reshape(-1).detach().cpu().numpy()
        bad = 0
        for p in self.packets:
            r0, n = p["row0"], p["n"]
            if n and float(np.max(np.abs(Y[r0:r0 + n] - np.asarray(p["y"])))) > 1e-5:
                bad += 1
        pk_rows = sum(p["n"] for p in self.packets)
        dup = sum(p["n_dup"] for p in self.packets)
        out = dict(rows=rows, n_init=self.n_init, packet_rows=pk_rows, rows_dup_skipped=dup,
                   rows_discarded=self.n_init + pk_rows - rows, packets_not_in_dh=bad,
                   tags_aligned=len(self.rows) == rows)
        out["ok"] = bool(out["rows_discarded"] == 0 and bad == 0 and out["tags_aligned"] and dup == 0)
        return out

    def summary(self) -> dict:
        dh = self.dh
        nd = []
        for i, r in enumerate(dh.needles_results):
            x = r["point"].detach().cpu().numpy().reshape(1, -1)
            z = next((z for z in self.zones if z["kind"] == "needle" and z["index"] == i), {})
            nd.append(dict(value=float(r["value"]), median=r.get("median_value"),
                           y_true=(float(self.predict(x)[0]) if self.predict is not None else None),
                           lane=z.get("lane"), act=z.get("act"), refined=bool(r.get("refined", False))))
        dec = [d for ln in self.lanes.values() for d in ln.decisions]
        secs = np.array([d["seconds"] for d in dec]) if dec else np.zeros(0)
        return dict(options=self.opts.to_dict(), label=self.opts.label(), rows=int(dh.X_all_actual.shape[0]),
                    packets=len(self.packets), needles=nd,
                    exclusions=int(dh.exclusions.shape[0]) if dh.exclusions is not None else 0,
                    activations=self._act_counter, decisions=len(dec),
                    decide_s_median=float(np.median(secs)) if secs.size else None,
                    decide_s_p90=float(np.percentile(secs, 90)) if secs.size else None,
                    comp={k: round(v, 3) for k, v in self.comp.items()},
                    fit_reused=self.fit_stats.get("fit_reused"), fit_calls=self.fit_stats.get("fit_calls"),
                    # validation 4(a): the leader logs never recorded these two
                    fit_rng_consumed=self.fit_stats.get("fit_rng_consumed"),
                    fit_skip_after_rng=self.fit_stats.get("fit_skip_after_rng"))
