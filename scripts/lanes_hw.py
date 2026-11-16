"""Hardware driver of the two-lane engine ("lanes + catch-up + zoom-2 floor 3", src/lanes).

    run_lanes_hardware(run_dir, X_init_actual, X_init_expected, Y_init, bounds, hp, *, preset="leader", log=print)

Runs until the process is stopped. It drives ``src.lanes.build_lane_engine`` with the call order of the T3 fake
lag-1 driver (tests/test_lanes_engine_smoke.py ``run_fake``, pickup="after_other"):

    start   each lane decides (main + backup) -> its two slots; a second start decision per lane follows once
            its rail has read the first main (or after ``SECOND_DECIDE_S``)
    pickup  a "k" read ack names a recipe id -> lane.on_pickup(engine rid)
    result  routed by "r" (wire.ResultRouter) -> on_result(packet, tags, now) for BOTH lanes (ingested once),
            then the owner lane decides (main + backup -> its slots), then the other lane refreshes if its
            main is still unread (new lines -> its slots)
    refill  a lane whose two slots were both read (or which got no result for ``REPLENISH_S``) decides again,
            so a rail is never left with nothing to read while the lane can produce a line
    restart a 'disco_restart' link event (communication.DiscoRestartLink: DiSCO's startup self-test ping,
            i.e. DiSCO restarted and wiped its compositions.db) -> router.on_disco_restart for every recipe
            issued before the restart (read -> lost, unread -> unprinted), then every slot issued before it
            is retired: a lane whose main is still unread gets its unread lines again, same content, under
            NEW ids; a lane whose main was read decides its next main + backup; one full map, new ts
            (``_on_disco_restart``). Results of pre-restart slides still route by their own id ('late').

Every slot write is the FULL four-slot map (DESIGN R1) through ``communication.write_slots4``; new recipe ids
are persisted to ``<run>/lanes/ids.json`` by the router hook before the slots carrying them are written (R4);
results whose id this run never issued are logged, never routed to a line (R2).

Files (all in ``<run>/lanes/``): ``journal.jsonl`` (every packet, routing decision, slot write and engine
decision, appended with flush + fsync BEFORE acting on it), ``rows.jsonl`` (measured rows), ``ids.json``
(router state), ``init.pt`` (initial design, for a resume), ``engine_ckpt.pt`` (best-effort data checkpoint,
every result, start, timer decision and needle), ``meta.json`` (run epoch), ``needles_state.pt`` (the needle /
exclusion-zone set with its exact geometry and zone records, rewritten whenever it changes), ``engine/`` (the
engine's DataHandler snapshots).

Resume: when ``ids.json`` / ``rows.jsonl`` exist, the router is reloaded (no id is re-issued) and the engine is
rebuilt from ``init.pt`` + every row of ``rows.jsonl`` (the data is kept; each lane continues inside its dive,
``_restore_dives``), and the previous needles are restored with their exact penalty geometry.

needles.db: with ``needles_db=True`` (run_zombi_main) the needles are mirrored into <repo>/sql/needles.db (or
ZOMBI_NEEDLES_DB), which DiSCO's visualization.py draws as stars. Off by default: tests and harnesses never
touch the DiSCO-facing file. ZOMBI_NEEDLES_DB is honoured ONLY by this lanes mirror (``_needles_db_target``):
scripts/main.py's needles.db reset / preload on a hardware start and the legacy optimizer always use
<repo>/sql/needles.db. It is meant for tests and restart rehearsals, not for moving the file DiSCO reads.
"""
from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from scripts import communication
from src.lanes import wire

LANES = ("front", "back")


def in_box_check(inner, outer, tol: float = 1e-9) -> bool:
    """A child zoom box lies inside its parent (dive-exact resume self-check)."""
    return bool((inner[0] >= outer[0] - tol).all() and (inner[1] <= outer[1] + tol).all())

# The needles.db DiSCO's visualization.py reads (src/utils/needles_db.DEFAULT_NEEDLES_DB_PATH before any patch)
LIVE_NEEDLES_DB = Path(__file__).resolve().parent.parent / "sql" / "needles.db"


def _same_path(a, b) -> bool:
    """``a`` and ``b`` name the same file by any spelling (case, 8.3 short name, junction / symlink):
    os.path.samefile when both exist, else the normalised real paths. Raises on a bad argument."""
    a, b = os.fspath(a), os.fspath(b)
    if os.path.exists(a) and os.path.exists(b):
        return os.path.samefile(a, b)
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def live_needles_db_refused(path) -> bool:
    """A test (ZOMBIHOP_TESTING=1, tests/conftest.py) never mirrors into the live sql/needles.db: on 2026-11-15
    two lanes driver tests emptied the running campaign's needle table and DiSCO's stars vanished. Any spelling
    of the live file is refused; an error while comparing refuses too (fail closed)."""
    if os.environ.get("ZOMBIHOP_TESTING") != "1":
        return False
    try:
        return _same_path(path, LIVE_NEEDLES_DB)
    except Exception:
        return True


POLL_S = 1.0
SECOND_DECIDE_S = float(os.environ.get("ZOMBI_LANES_SECOND_S", 25 * 60.0))   # protocols.LanesPerRail4 second_write_after
REPLENISH_S = float(os.environ.get("ZOMBI_LANES_REPLENISH_S", 2 * 50 * 60.0 + 5 * 60.0))  # 2 periods + delay
MAX_REPLENISH = 4
ZERO_COMP_TOL = 1e-12


# ── durable append-only logs ─────────────────────────────────────────────────────────
def _jsonable(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


class _Appender:
    """Append one JSON line per record, flush + fsync, retried (the run dir is in Dropbox). Never raises."""

    def __init__(self, path: Path, log: Callable = print):
        self.path = Path(path)
        self.log = log

    def __call__(self, ev: str, **rec) -> bool:
        line = json.dumps(_jsonable(dict(t=time.time(), ev=ev, **rec)), default=str) + "\n"
        for attempt in range(20):
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line)
                    fh.flush()
                    os.fsync(fh.fileno())
                return True
            except OSError:
                time.sleep(0.1 * (attempt + 1))
        try:
            self.log(f"[lanes] WARNING could not append to {self.path}: {line[:200]}")
        except Exception:
            pass
        return False


def _read_jsonl(path: Path) -> List[dict]:
    out = []
    if not Path(path).exists():
        return out
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue                         # a torn last line after a crash
    return out


def _to_wire10(v_opt: np.ndarray, opt_dims: List[int]) -> List[float]:
    """The legacy _write_line_pair arithmetic: round(., 3), normalise (numpy), pad to the 10 hardware columns."""
    a = np.round(np.asarray(v_opt, dtype=float), 3)
    s = a.sum()
    a = a / s if s != 0 else np.full_like(a, 1.0 / a.shape[0])
    if not np.isfinite(a).all():
        a = np.full_like(a, 1.0 / a.shape[0])
    out = np.zeros(10, dtype=float)
    out[list(opt_dims)] = a
    return [float(x) for x in out]


class _Slot:
    __slots__ = ("wid", "line", "lane", "role", "state", "t_write", "start", "end", "t_issue", "retired")

    def __init__(self, wid, line, lane, role, start, end, t_issue=None):
        self.wid, self.line, self.lane, self.role = int(wid), line, lane, role
        self.state = "unread"
        self.t_write = time.time()
        self.t_issue = self.t_write if t_issue is None else float(t_issue)   # the router's issue time
        self.start, self.end = start, end
        self.retired = False            # set by a DiSCO restart: this object is never written again


class LanesHardwareDriver:
    def __init__(self, run_dir, X_init_actual, X_init_expected, Y_init, bounds, hp, *, preset="leader",
                 log=print, opt_dims=None, y_to_optimizer=None, seed: int = 0, needles_db=None):
        """``needles_db``: mirror the needles into a needles.db (DiSCO's visualization.py draws them as stars).
        None / False (default): no mirror, so a test or harness never touches the DiSCO-facing file. True: the
        ZOMBI_NEEDLES_DB path when set, else <repo>/sql/needles.db (the hardware run, run_zombi_main). A str /
        Path: that file. The needle restore on a resume and lanes/needles_state.pt do not depend on it."""
        self.run_dir = Path(run_dir)
        self.dir = self.run_dir / "lanes"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.journal = _Appender(self.dir / "journal.jsonl", log)
        self.rows_log = _Appender(self.dir / "rows.jsonl", log)
        if opt_dims is None or y_to_optimizer is None:
            from scripts import run_zombi_main as _rzm
            opt_dims = list(_rzm.OPTIMIZING_DIMS) if opt_dims is None else opt_dims
            y_to_optimizer = _rzm.y_measured_to_optimizer if y_to_optimizer is None else y_to_optimizer
        self.opt_dims = [int(k) for k in opt_dims]
        self.y_to_opt = y_to_optimizer
        self.preset = preset
        self.hp = None if hp is None else dict(hp)

        # ── run epoch (engine clock = seconds since the lanes phase started; survives a resume) ──
        meta_p = self.dir / "meta.json"
        meta = {}
        if meta_p.exists():
            try:
                meta = json.loads(meta_p.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        self.epoch = float(meta.get("epoch", time.time()))
        self.resumed = bool(meta)
        meta.update(epoch=self.epoch, preset=preset, opt_dims=self.opt_dims, n_starts=int(meta.get("n_starts", 0)) + 1)
        wire.save_json(meta_p, meta)

        # ── initial data (+ every measured row on a resume) ──
        init_p = self.dir / "init.pt"
        if not init_p.exists():
            try:
                torch.save(dict(X_actual=X_init_actual.detach().cpu(), X_expected=X_init_expected.detach().cpu(),
                                Y=Y_init.detach().cpu(), bounds=torch.as_tensor(bounds).detach().cpu(),
                                opt_dims=self.opt_dims), init_p)
            except Exception as e:
                self.log(f"[lanes] WARNING could not save init.pt: {e!r}")
        Xa, Xe, Y = X_init_actual, X_init_expected, Y_init.reshape(-1, 1)
        prior_all = _read_jsonl(self.dir / "rows.jsonl")
        # only measured rows: a 'raw_rows' record (the _save_rows fallback) has no composition to rebuild from
        prior = [r for r in prior_all if r.get("ev", "row") == "row" and r.get("x_actual") is not None
                 and r.get("x_expected") is not None and r.get("y_opt") is not None]
        if len(prior) != len(prior_all):
            self.log(f"[lanes] WARNING resume: {len(prior_all) - len(prior)} rows.jsonl record(s) without a "
                     f"composition skipped (raw_rows fallback)")
        if prior:
            pa = torch.tensor([r["x_actual"] for r in prior], dtype=Xa.dtype, device=Xa.device)
            pe = torch.tensor([r["x_expected"] for r in prior], dtype=Xa.dtype, device=Xa.device)
            py = torch.tensor([[r["y_opt"]] for r in prior], dtype=Xa.dtype, device=Xa.device)
            Xa, Xe, Y = torch.cat([Xa, pa]), torch.cat([Xe, pe]), torch.cat([Y, py])
            self.log(f"[lanes] resume: {len(prior)} measured row(s) from rows.jsonl added to the initial data")

        # ── recipe ids / router (R2, R4) ──
        ids_p = self.dir / "ids.json"
        self.router = None
        if ids_p.exists():
            try:
                self.router = wire.ResultRouter.load(ids_p)
                self.log(f"[lanes] resume: recipe-id state loaded from {ids_p} (rev {self.router.rev})")
            except Exception as e:
                self.log(f"[lanes] WARNING could not load {ids_p}: {e!r}; rebuilding ids from the journal")
        if self.router is None:
            used = {}
            for rec in _read_jsonl(self.dir / "journal.jsonl"):
                if rec.get("ev") == "slot_write":
                    for s in rec.get("slots") or []:
                        used[int(s[1])] = float(rec.get("t", time.time()))
            ids = wire.RecipeIds.recover(used, time.time()) if used else wire.RecipeIds.for_run()
            self.router = wire.ResultRouter(ids)
        self.router.persist_to(str(ids_p))
        self.router.save(str(ids_p), time.time())

        self.journal("start", resumed=self.resumed, n_init=int(Y.shape[0]), n_prior_rows=len(prior),
                     preset=preset, opt_dims=self.opt_dims, epoch=self.epoch)

        # ── needles of the previous process (read BEFORE the new engine writes its own first snapshot); a
        # failure here must never stop the run from starting ──
        self._prev_needles = None
        self._needles_read_error = None
        self._needle_state_unreadable = None   # set by _read_needle_state when needles_state.pt exists but fails
        if self.resumed:
            try:
                self._prev_needles = self._read_needle_state()
            except Exception as e:
                self._prev_needles = None
                self._needles_read_error = repr(e)
                self.journal("needles_read_failed", error=repr(e), tb=traceback.format_exc())
                try:
                    self.log(f"[lanes] WARNING could not read the previous needles ({e!r}); the run starts "
                             f"without them")
                except Exception:
                    pass

        # ── the engine (the hardware driver seeds; build_lane_engine never does, REVIEW F3) ──
        torch.manual_seed(int(seed))
        from src.lanes import build_lane_engine
        self.engine = build_lane_engine(
            Xa, Xe, Y, bounds, preset=preset, hparams=self.hp, seed=int(seed),
            checkpoint_dir=str(self.dir / "engine"), run_uuid=self.run_dir.name.replace("run_", "", 1),
            verbose=False)
        self.lanes = {n: self.engine.lanes[n] for n in LANES}
        self.slots: Dict[int, Optional[_Slot]] = {k: None for k in range(4)}
        self.n_decide = {n: 0 for n in LANES}
        self.t_last_result = {n: time.time() for n in LANES}
        self.t_last_write = {n: time.time() for n in LANES}
        self.n_replenish = {n: 0 for n in LANES}
        self.t_start = time.time()
        self.after_id = 0
        self.n_results = 0
        # link events (DiSCO restarts). A new run skips the ones already in the queue: they predate every id
        # it will issue (a resume replays from its journal cursor instead, resume_cursor).
        self.after_event_id = 0 if self.resumed else communication.max_link_event_id()
        self.t_restart = None           # t_ping of the newest DiSCO restart handled
        self.n_restarts = 0

        # ── dive-exact resume: continue each lane inside its dive (same act / zoom / boxes / counters) ──
        self.restored = None
        if self.resumed and prior and os.environ.get("ZOMBI_LANES_RESUME", "dive") != "fresh":
            try:
                self.restored = self._restore_dives(prior)
                self.log("[lanes] resume: dive-exact - " + "; ".join(
                    f"{n} act {s['act']} zoom {s['zoom']} (results/zoom {s['results']}, sends/zoom {s['sends']})"
                    for n, s in self.restored.items()))
            except Exception as e:
                self.log(f"[lanes] WARNING dive-exact resume not possible ({e!r}); the lanes start fresh "
                         f"activations (all measured data kept)")
                self.journal("resume_fallback", error=repr(e), tb=traceback.format_exc())
                try:
                    self.engine.close()
                except Exception:
                    pass
                torch.manual_seed(int(seed))
                self.engine = build_lane_engine(
                    Xa, Xe, Y, bounds, preset=preset, hparams=self.hp, seed=int(seed),
                    checkpoint_dir=str(self.dir / "engine"), run_uuid=self.run_dir.name.replace("run_", "", 1),
                    verbose=False)
                self.lanes = {n: self.engine.lanes[n] for n in LANES}
                self.n_decide = {n: 0 for n in LANES}
                self.restored = None
        if self.restored is not None:
            self.journal("resume_dives", lanes=self.restored)

        # ── needles: restore the previous process's needles (exact penalty geometry), persist them at once
        # (lanes/needles_state.pt) and, when asked (needles_db), mirror them into the needles.db DiSCO's
        # visualization.py draws as stars (2026-11-15: lanes never mirrored them) ──
        self.n_needles_restored = 0
        if self._prev_needles is not None:
            restore_error = None
            try:
                self.n_needles_restored = self._apply_needle_state(self._prev_needles)
            except Exception as e:
                restore_error = repr(e)
                self.log(f"[lanes] WARNING could not restore the previous needles ({e!r}); the optimizer runs "
                         f"WITHOUT their penalty; new needles still mirror to needles.db")
                self.journal("needles_restore_failed", error=repr(e), tb=traceback.format_exc())
            n_prev = self._count(self._prev_needles.get("tensors") or {}, "needles")
            if restore_error is None and n_prev and not self.n_needles_restored:
                restore_error = f"{n_prev} needle(s) read, 0 restored"
            if restore_error is not None:
                # before anything can overwrite needles_state.pt with the smaller set (the next needle would)
                self._save_unrestored_needles(self._prev_needles, restore_error)
            else:
                src = str(self._prev_needles.get("snapshot") or "")
                if self._needle_state_unreadable is not None and src != str(self.dir / self.NEEDLE_STATE):
                    # restored from a snapshot because needles_state.pt could not be read (a Dropbox lock, a
                    # torn file): it may hold MORE than the snapshot - keep a copy before it is rewritten below
                    self._save_unrestored_needles(
                        None, f"needles_state.pt unreadable at restart ({self._needle_state_unreadable}); "
                              f"{self.n_needles_restored} needle(s) restored from {src}",
                        partial=True)
                self._save_needle_state()
        elif self._needles_read_error is not None or (self.resumed and os.path.exists(self.dir / self.NEEDLE_STATE)):
            # unreadable now (a Dropbox lock, a torn file): keep a copy of what the previous process saved
            self._save_unrestored_needles(None, self._needles_read_error or "needles_state.pt not readable")
        self.needles_db_path = self._needles_db_target(needles_db)
        if self.needles_db_path is not None:
            self._enable_needle_mirror(self.needles_db_path)
        # DataHandler.add_needle ends with _sync_needles_db: persist needles_state.pt right there too, so a
        # kill between a needle's declaration and its decision's snapshot (~5 s later) loses nothing
        dh = self.engine.dh
        self._dh_sync_needles_db = getattr(dh, "_sync_needles_db", None)
        if self._dh_sync_needles_db is not None:
            dh._sync_needles_db = self._persist_needles
        self._needle_sig_last = self._needle_sig()

    # ── dive-exact resume (2026-11-14) ──
    def _restore_dives(self, prior: List[dict]) -> dict:
        """Put each lane back INSIDE its dive after a ZoMBI restart: same activation, zoom, zoom boxes,
        per-zoom result/send counters and row ownership, instead of a fresh activation.

        Sources (all written by the previous process): engine_ckpt.pt (lane act/zoom, the zoom_in events with
        their boxes, extend events), journal.jsonl ('ingested' = which result was counted at which zoom,
        'issue' + 'pickup' = lines read at the current zoom, 'decision' counts) and rows.jsonl (row owner).
        Raises on anything inconsistent; the caller then rebuilds a fresh engine (the previous behaviour)."""
        from src.lanes.geometry import LANE_CODE
        eng = self.engine
        ck = torch.load(self.dir / "engine_ckpt.pt", map_location="cpu", weights_only=False)
        lane_state = ck.get("lane_state") or {}
        events = [e for e in (ck.get("events") or []) if isinstance(e, dict)]
        jr = _read_jsonl(self.dir / "journal.jsonl")
        n_rows = int(eng.dh.X_all_actual.shape[0])
        n0 = n_rows - len(prior)                           # rows of the initial design
        if n0 <= 0 or n_rows != int(eng.n_init) or len(eng.rows) != n_rows:
            raise RuntimeError(f"row alignment: dh {n_rows} rows, n_init {eng.n_init}, prior {len(prior)}")
        # 1. row ownership: lane / act / sent zoom per measured row (init rows stay -1)
        for i, r in enumerate(prior):
            code = LANE_CODE.get(r.get("lane"))
            if code is None:
                continue
            idx = n0 + i
            z = int(r.get("zoom_sent", -1))
            eng.rows.a["lane"][idx] = code
            eng.rows.a["act"][idx] = int(r.get("act", -1))
            eng.rows.a["zoom"][idx] = z
            eng.rows.a["eff"][idx] = z
        eng.rows._cache.clear()
        # 2. journal facts
        issued = {}                                         # wid -> issue record (latest)
        for rec in jr:
            if rec.get("ev") == "issue":
                issued[int(rec["wid"])] = rec
        restored = {}
        acts_max = -1
        for name in LANES:
            st = lane_state.get(name)
            if not st:
                raise RuntimeError(f"engine_ckpt.pt has no lane_state for {name}")
            ln = self.lanes[name]
            act, zoom = int(st["act"]), int(st["zoom"])
            mz = int(eng.dh.max_zooms)
            if not (0 <= zoom < mz):
                raise RuntimeError(f"{name}: zoom {zoom} outside 0..{mz - 1}")
            zins = {int(e["to_zoom"]): e for e in events
                    if e.get("kind") == "zoom_in" and e.get("lane") == name and int(e.get("act", -9)) == act}
            boxes = [eng.full.clone()]
            info = [dict(kind="full")]
            for k in range(1, zoom + 1):
                e = zins.get(k)
                if e is None or e.get("box") is None:
                    raise RuntimeError(f"{name} act {act}: no zoom_in event with the zoom-{k} box")
                b = torch.as_tensor(e["box"], dtype=eng.full.dtype, device=eng.full.device)
                if b.shape != eng.full.shape or bool((b[1] < b[0]).any()):
                    raise RuntimeError(f"{name} act {act}: bad zoom-{k} box {tuple(b.shape)}")
                boxes.append(b)
                info.append(dict(e.get("info") or {}, kind="zoom", from_zoom=k - 1, restored=True))
            results = [0] * (mz + 1)
            sends = [0] * (mz + 1)
            ext = [0] * (mz + 1)
            # results counted at each zoom = the engine's own classification ('own_current', counted_zoom) of
            # the packets in engine_ckpt.pt, on top of the counts carried over by an earlier restore
            consec = 0
            for e in events:
                if e.get("kind") == "resume_baseline" and e.get("lane") == name and int(e.get("act", -9)) == act:
                    for k, v in enumerate(e.get("results") or []):
                        if k <= mz:
                            results[k] = int(v)
                    if int(e.get("zoom", -9)) == zoom:
                        consec = int(e.get("consec", 0))
            for p in ck.get("packets") or []:
                if (p.get("lane") == name and p.get("act") == act and p.get("cls") == "own_current"
                        and p.get("counted_zoom") is not None):
                    k = int(p["counted_zoom"])
                    results[k] += 1
                    v = p.get("verdict") or {}
                    if k == zoom and isinstance(v, dict) and v.get("consec") is not None:
                        consec = int(v["consec"])        # the verdict records the counter after itself
            for k in range(zoom):                            # finished zooms: sends at the zoom-in
                sends[k] = int(zins[k + 1].get("sends", 0))
            # current zoom: the state as of the LAST RECEIVED OBJECTIVE - only lines that returned data count as
            # sends. A line read but never measured (e.g. a print that failed, or anything in flight when ZoMBI
            # stopped) produced nothing; the new engine does not know it, so it does not use up the zoom budget.
            measured = set()
            for rec in jr:
                if rec.get("ev") == "ingested" and rec.get("lane") == name:
                    iss = issued.get(int(rec.get("wid", -1)))
                    if iss and iss.get("lane") == name and int(iss.get("act", -9)) == act \
                            and int(iss.get("zoom", -9)) == zoom:
                        measured.add(str(iss.get("engine_rid")))
            sends[zoom] = len(measured)
            for e in events:
                if e.get("kind") == "extend" and e.get("lane") == name and int(e.get("act", -9)) == act:
                    ext[int(e["zoom"])] = max(ext[int(e["zoom"])], int(e.get("ext", 0)))
            if not results[zoom]:
                consec = 0                                   # reset at the zoom-in, no verdict since
            ln.act, ln.zoom = act, zoom
            ln.boxes, ln.box_info = boxes, info
            ln.results, ln.sends, ln.ext = results, sends, ext
            ln.consec = consec
            ln.lines_act = int(sum(sends))
            ln.pending_confirm = None
            ln.zone_seq0 = 0
            ln._cover_cache = None
            ln.n_dec = sum(1 for r in jr if r.get("ev") == "decision" and r.get("lane") == name
                           and r.get("result") != "none")
            acts_max = max(acts_max, act)
            ev_act = next((e for e in events if e.get("kind") == "activation" and e.get("lane") == name
                           and int(e.get("act", -9)) == act), None)
            eng.acts[act] = dict(lane=name, row0=int(ev_act.get("n_rows", n0)) if ev_act else n0, zone_seq0=0)
            eng.events.append(dict(kind="resume_baseline", lane=name, act=act, zoom=zoom, results=list(results),
                                   consec=int(consec), restored=True))
            restored[name] = dict(act=act, zoom=zoom, results=results[:mz], sends=sends[:mz], consec=consec,
                                  lines_act=ln.lines_act, box=[[round(float(v), 4) for v in row] for row in boxes[-1].tolist()])
        eng._act_counter = max(int(eng._act_counter), acts_max + 1)
        # engine line ids ('F49', 'B52', ...) continue after the previous process's: a late result of an old line
        # can never be taken for a new line of the same name
        old_ids = [0]
        for rec in jr:
            for key in ("engine_rid",):
                v = rec.get(key)
                if isinstance(v, str) and v[1:].isdigit():
                    old_ids.append(int(v[1:]))
            for key in ("main", "backup"):
                v = rec.get(key)
                if isinstance(v, dict) and isinstance(v.get("rid"), str) and v["rid"][1:].isdigit():
                    old_ids.append(int(v["rid"][1:]))
        eng._line_counter = max(int(eng._line_counter), max(old_ids))
        for e in events:                                     # keep the box history in the next engine_ckpt.pt
            if e.get("kind") in ("zoom_in", "extend"):
                eng.events.append(dict(e, restored=True))
        if eng.active is not None:
            eng.active._sync_dh()
        for name in LANES:                                   # self-check: the dive's own rows and box are usable
            ln = self.lanes[name]
            ln._box_rows(ln.zoom)
            if ln.zoom >= 1 and not bool(in_box_check(ln.boxes[ln.zoom], ln.boxes[ln.zoom - 1])):
                raise RuntimeError(f"{name}: zoom-{ln.zoom} box is not inside its parent")
        for name in LANES:
            self.n_decide[name] = max(2, self.lanes[name].n_dec)
        return restored

    # ── needles (restore on resume + live mirror to sql/needles.db) ──
    def _engine_run_dir(self) -> Path:
        return self.dir / "engine" / f"run_{self.run_dir.name.replace('run_', '', 1)}"

    NEEDLE_STATE = "needles_state.pt"
    _NEEDLE_KEYS = ("needles", "needle_vals", "needle_indices", "needle_penalty_radii", "needle_M_stack",
                    "needle_has_M", "needle_shrink_scale", "needle_B", "exclusions", "exclusion_radii",
                    "exclusion_M_stack", "exclusion_has_M", "penalty_mask")

    @staticmethod
    def _retry(fn, tries: int = 4):
        """fn() with a few retries (the run dir is in Dropbox: a sync can hold a file for a moment). A missing
        file is final at once."""
        for k in range(tries):
            try:
                return fn()
            except FileNotFoundError:
                raise
            except Exception:
                if k == tries - 1:
                    raise
                time.sleep(0.25 * (k + 1))

    @staticmethod
    def _count(d: dict, key: str) -> int:
        v = d.get(key)
        return int(v.shape[0]) if isinstance(v, torch.Tensor) and v.dim() >= 1 else 0

    @staticmethod
    def _n_needle_zones(zones: List[dict]) -> int:
        return sum(1 for z in zones if isinstance(z, dict) and z.get("kind") == "needle")

    def _read_needle_state(self) -> Optional[dict]:
        """The previous processes' needle state, read BEFORE the new engine writes its own first snapshot.

        Sources (each read with retries; a failure is a WARNING, never an exception):
          * ``lanes/needles_state.pt``: written by this driver right after a restore and whenever the needle set
            changes (tensors incl. the ellipsoid matrices, per-needle records, zone records);
          * the engine DataHandler's latest snapshot (``delta.pt`` + ``needles.json``; the only source of a
            process that predates needles_state.pt) with the zone records of ``engine_ckpt.pt``;
          * always, the richest snapshot of this run's engine dir by its summary.json (most needles, then
            newest): covers a torn latest.txt, a restart before the first decision (latest = 0001_init), a
            missing needles_state.pt and stale zone records at once (~40 ms for ~240 snapshots).
        The richest state wins (most needles, then most exclusion zones, then newest): needles and zones are
        never removed in a lanes run. None when there is nothing to restore (a WARNING when a source says there
        were needles). Needle zone records get the declaration's ``n_rows`` / ``value`` from the matching
        engine_ckpt.pt needle event (the timeline's 'declared' diamond)."""
        ed = self._engine_run_dir()
        cands, zone_lists = [], []
        ev_needle = {}                       # (lane, act, needle index) -> the checkpoint's needle event
        st_p = self.dir / self.NEEDLE_STATE
        if st_p.exists():
            try:
                s = self._retry(lambda: torch.load(st_p, map_location="cpu", weights_only=False))
                zs = [dict(z) for z in (s.get("zones") or []) if isinstance(z, dict)]
                t_s = float(s.get("t", 0.0))
                zone_lists.append((zs, t_s, 1))
                cands.append(dict(source=str(st_p), tensors=dict(s["tensors"]), t=t_s,
                                  records=[r for r in (s.get("records") or []) if isinstance(r, dict)]))
            except Exception as e:
                self._needle_state_unreadable = repr(e)
                self.log(f"[lanes] WARNING could not read {st_p} ({e!r}); falling back to the engine snapshots")
        ck_p = self.dir / "engine_ckpt.pt"
        if ck_p.exists():
            try:
                ck = self._retry(lambda: torch.load(ck_p, map_location="cpu", weights_only=False))
                zone_lists.append(([dict(z) for z in (ck.get("zones") or []) if isinstance(z, dict)],
                                   float(ck.get("t") or 0.0), 0))
                for e in ck.get("events") or []:
                    try:
                        if isinstance(e, dict) and e.get("kind") == "needle":
                            ev_needle[(e.get("lane"), int(e["act"]), int(e.get("needle", -1)))] = e
                    except Exception:
                        continue
            except Exception as e:
                self.log(f"[lanes] WARNING could not read the zone records in {ck_p} ({e!r})")

        def snap_cand(snap: Path, t_hint: Optional[float] = None) -> dict:
            d = self._retry(lambda: torch.load(snap / "delta.pt", map_location="cpu", weights_only=False))
            recs = []
            if (snap / "needles.json").exists():
                recs = self._retry(lambda: json.loads((snap / "needles.json").read_text(encoding="utf-8")))
            t = t_hint
            if t is None:
                try:
                    t = float(json.loads((snap / "summary.json").read_text(encoding="utf-8")).get("timestamp"))
                except Exception:
                    t = (snap / "delta.pt").stat().st_mtime
            return dict(source=str(snap), tensors=d, t=float(t),
                        records=[r for r in recs if isinstance(r, dict)] if isinstance(recs, list) else [])

        latest_dir = None
        if (ed / "latest.txt").exists():
            try:
                name = self._retry(lambda: (ed / "latest.txt").read_text(encoding="utf-8").strip())
                if not name:
                    raise RuntimeError("latest.txt is empty")
                cands.append(snap_cand(ed / "snapshots" / name))
                latest_dir = ed / "snapshots" / name
            except Exception as e:
                self.log(f"[lanes] WARNING could not read the engine's latest snapshot ({e!r}); using the "
                         f"snapshot scan")
        n_expected = max([self._n_needle_zones(zs) for zs, _, _ in zone_lists] or [0])
        n_have = max([self._count(c["tensors"], "needles") for c in cands] or [0])
        # always: the richest snapshot of this run's engine dir (only ``ed``, never another run's) is a candidate
        n_scan = 0
        try:
            best = None
            if (ed / "snapshots").is_dir():
                for sd in (ed / "snapshots").iterdir():
                    try:
                        sm = json.loads((sd / "summary.json").read_text(encoding="utf-8"))
                        key = (int(sm.get("n_needles") or 0), float(sm.get("timestamp") or 0.0))
                    except Exception:
                        continue
                    if key[0] > 0 and (best is None or key > best[0]):
                        best = (key, sd)
            if best is not None:
                n_scan = best[0][0]
                if best[1] != latest_dir:                      # the latest snapshot is a candidate already
                    try:
                        cands.append(snap_cand(best[1], t_hint=best[0][1]))
                        if best[0][0] > n_have:
                            self.log(f"[lanes] WARNING needles: the latest state holds {n_have} needle(s), the "
                                     f"zone records {n_expected}; snapshot {best[1].name} holds {best[0][0]}")
                    except Exception as e:
                        self.log(f"[lanes] WARNING could not read snapshot {best[1]} ({e!r})")
        except Exception as e:
            self.log(f"[lanes] WARNING could not scan the engine snapshots in {ed} ({e!r})")
            self.journal("needles_scan_failed", error=repr(e), tb=traceback.format_exc())
        cands = [c for c in cands if self._count(c["tensors"], "needles") + self._count(c["tensors"], "exclusions")]
        if not cands:
            why = []
            if n_expected:
                why.append(f"the zone records name {n_expected} needle(s)")
            if os.path.exists(st_p):
                why.append(f"{st_p.name} exists")
            if n_scan:
                why.append(f"a snapshot summary names {n_scan} needle(s)")
            if why:
                self.log(f"[lanes] WARNING needles NOT restored: {'; '.join(why)}, but no readable needle state "
                         f"holds them")
                self.journal("needles_not_restored", why=why)
            return None
        c = max(cands, key=lambda c: (self._count(c["tensors"], "needles"), self._count(c["tensors"], "exclusions"),
                                      c["t"]))
        n_c = self._count(c["tensors"], "needles")
        if n_c < max(n_expected, n_scan):
            self.log(f"[lanes] WARNING only {n_c} of the {max(n_expected, n_scan)} declared needle(s) are "
                     f"restorable ({c['source']})")
        # zone records: the richest list (needles_state.pt first on a tie); each one keeps a time inside the
        # process that declared it (t_proc) so the timeline puts it in that process (restored zones)
        zones, t_z = [], 0.0
        if zone_lists:
            zones, t_z, _ = max(zone_lists, key=lambda zl: (len(zl[0]), zl[2], zl[1]))
        zones = [dict(z, t_proc=z.get("t_proc", t_z)) for z in zones]
        # the declaration's row count (+ value) from its needle event: the timeline's 'declared' diamond stays at
        # its place after a restart (events of an earlier process are not in the next checkpoint; the zone record
        # carries them on through needles_state.pt and the next engine_ckpt.pt)
        for z in zones:
            try:
                if z.get("kind") != "needle":
                    continue
                e = ev_needle.get((z.get("lane"), int(z.get("act", -1)), int(z.get("index", -1))))
                if e is None:
                    continue
                if z.get("n_rows") is None and e.get("n_rows") is not None:
                    z["n_rows"] = int(e["n_rows"])
                if z.get("value") is None and e.get("value") is not None:
                    z["value"] = float(e["value"])
            except Exception:
                continue
        return dict(snapshot=c["source"], tensors=c["tensors"], records=c["records"], zones=zones)

    def _apply_needle_state(self, st: dict) -> int:
        """Put the previous needles (and exclusion zones) into the new engine's DataHandler with their EXACT
        saved geometry, rebuild its penalty mask and check it against the saved one; restore the lanes zone
        records (tagged ``restored``). Everything is computed first; the commit is rolled back if the mask
        rebuild fails, so the DataHandler never holds new needles with a stale mask.

        Row indices: the penalty geometry depends only on the points / ellipsoids, not on the row index. When a
        saved index no longer points at its needle's row (rows.jsonl lost a line), the index is re-derived as
        the nearest row (as add_needle does) and the geometry is still restored exactly."""
        dh, eng, d = self.engine.dh, self.engine, st["tensors"]
        dev, dt = dh.device, dh.dtype
        X = dh.X_all_actual
        D = int(X.shape[1])

        def T(k, shape1):
            v = d.get(k)
            return v.to(device=dev, dtype=dt) if isinstance(v, torch.Tensor) else torch.empty(0, shape1, device=dev, dtype=dt)

        needles = T("needles", D)
        n = int(needles.shape[0])
        if n and int(needles.shape[1]) != D:
            raise RuntimeError(f"needles have {int(needles.shape[1])} dims, the engine {D}")
        vals = T("needle_vals", 1).reshape(-1, 1)
        radii = T("needle_penalty_radii", 1).reshape(-1, 1)
        if int(vals.shape[0]) != n or int(radii.shape[0]) != n:
            raise RuntimeError(f"{n} needles but {int(vals.shape[0])} values / {int(radii.shape[0])} radii")
        idx = d.get("needle_indices")
        idx = (idx.to(device=dev, dtype=torch.int64).reshape(-1, 1) if isinstance(idx, torch.Tensor)
               else torch.empty(0, 1, device=dev, dtype=torch.int64))
        if n:
            flat = idx.reshape(-1)
            aligned = (int(flat.numel()) == n and int(flat.min()) >= 0 and int(flat.max()) < int(X.shape[0])
                       and float(torch.norm(X[flat] - needles, dim=1).max()) <= 1e-6)
            if not aligned:
                dmin, amin = torch.cdist(needles, X).min(dim=1)
                idx = amin.reshape(-1, 1).to(torch.int64)
                worst = float(dmin.max())
                absent = int((dmin > 1e-6).sum())
                self.log(f"[lanes] WARNING restored needles: saved row indices do not match the rebuilt rows; "
                         f"re-derived as the nearest rows" + (f" ({absent} needle point(s) not among the rows, "
                         f"max distance {worst:.3g}; their penalty geometry is restored exactly)" if absent else ""))
                self.journal("needles_reindexed", n=n, absent=absent, max_distance=worst)
        has_M = d.get("needle_has_M")
        M_stack = d.get("needle_M_stack")
        M_list = ([M_stack[i].to(dev, dt) if bool(has_M[i]) else None for i in range(int(has_M.shape[0]))]
                  if isinstance(has_M, torch.Tensor) and isinstance(M_stack, torch.Tensor) and has_M.shape[0] > 0
                  else [None] * n)
        if len(M_list) != n:
            raise RuntimeError(f"{len(M_list)} ellipsoids for {n} needles")
        sc = d.get("needle_shrink_scale")
        scales = [float(v) for v in sc.reshape(-1).tolist()] if isinstance(sc, torch.Tensor) else []
        scales = (scales + [1.0] * n)[:n]
        nb = d.get("needle_B")
        needle_B = nb.to(dev, dt) if isinstance(nb, torch.Tensor) else dh.needle_B
        excl = T("exclusions", D)
        excl_r = T("exclusion_radii", 1).reshape(-1, 1)
        ne = int(excl.shape[0])
        if int(excl_r.shape[0]) != ne:
            raise RuntimeError(f"{ne} exclusion zones but {int(excl_r.shape[0])} radii")
        excl_M = dh._unstack_optional_M(d, "exclusion_M_stack", "exclusion_has_M")
        if excl_M and len(excl_M) != ne:
            raise RuntimeError(f"{len(excl_M)} exclusion ellipsoids for {ne} zones")
        zones = [z for z in st["zones"]
                 if (z.get("kind") == "needle" and 0 <= int(z.get("index", -1)) < n)
                 or (z.get("kind") != "needle" and 0 <= int(z.get("index", -1)) < ne)]
        if len(zones) != len(st["zones"]):
            self.log(f"[lanes] WARNING {len(st['zones']) - len(zones)} zone record(s) without a restorable "
                     f"needle / exclusion zone dropped")
        lane_of = {int(z.get("index", -1)): z.get("lane") for z in zones if z.get("kind") == "needle"}
        recs = st["records"]
        results = []
        for i in range(n):
            r = recs[i] if i < len(recs) and isinstance(recs[i], dict) else {}
            results.append(dict(point=needles[i].clone(), value=r.get("value", float(vals[i].item())),
                                median_value=r.get("median_value"), zoom=r.get("zoom"),
                                iteration=r.get("iteration"), reason=r.get("reason"),
                                lane=lane_of.get(i, r.get("lane"))))
        # ── commit (every check passed); rolled back if the mask rebuild fails ──
        keep = ("needles", "needle_vals", "needle_indices", "needle_penalty_radii", "needle_M_list",
                "needle_shrink_scale", "needle_B", "exclusions", "exclusion_radii", "exclusion_M_list",
                "needles_results", "_penalty_mask", "X_pared", "Y_pared", "_pared_version")
        prev = {k: getattr(dh, k, None) for k in keep}
        try:
            dh.needles = needles
            dh.needle_vals = vals
            dh.needle_indices = idx
            dh.needle_penalty_radii = radii
            dh.needle_M_list = M_list
            dh.needle_shrink_scale = scales
            dh.needle_B = needle_B
            dh.exclusions = excl
            dh.exclusion_radii = excl_r
            dh.exclusion_M_list = excl_M
            dh.needles_results = results
            dh._update_penalty_mask()
        except Exception:
            for k, v in prev.items():
                setattr(dh, k, v)
            raise
        old = d.get("penalty_mask")
        new = dh._penalty_mask
        if isinstance(old, torch.Tensor) and new is not None and old.numel() == new.numel():
            diff = int((old.reshape(-1).bool().cpu() != new.reshape(-1).bool().cpu()).sum())
            if diff:
                self.log(f"[lanes] WARNING restored needles: penalty mask differs from the saved one on {diff} row(s)")
        if zones:
            # restored=True: the record (and its act id) belongs to the process that declared it; the timeline
            # (visualization/lanes_timeline.py) places it by t_proc, never in this process
            eng.zones = [dict(z, restored=True) for z in zones]
            eng.zone_seq = max(int(z.get("seq", 0)) for z in eng.zones)
            eng.penalty_version += len(eng.zones)
        for ln in self.lanes.values():
            ln.zone_seq0 = eng.zone_seq
            ln.seen = (eng.version, eng.penalty_version)
        self.log(f"[lanes] resume: {n} needle(s) and {ne} exclusion zone(s) restored "
                 f"(penalised rows {int((~dh._penalty_mask.reshape(-1).bool()).sum()) if dh._penalty_mask is not None else '?'})")
        self.journal("needles_restored", n=n, exclusions=ne, snapshot=st["snapshot"], zones=len(zones))
        return n

    def _save_needle_state(self, tries: int = 3) -> bool:
        """lanes/needles_state.pt: the needle set as it is now (the DataHandler's needle / exclusion tensors as
        its snapshots store them, the per-needle records, the lanes zone records with t_proc), written in place
        (Dropbox: no os.replace) with retries. Written right after a restore and whenever the set changes, so
        a restart never depends on the engine's latest snapshot alone. Never raises.

        True when the file holds the current set (written now, or there is no needle / zone to write); False
        when it could not be written: up to ``tries`` tries (3: at most 0.3 s of waiting; it runs in the
        decision path). torch.save on a file Dropbox holds raises RuntimeError (its zip writer), an OSError
        otherwise."""
        try:
            dh, now = self.engine.dh, time.time()
            n = int(dh.needles.shape[0]) if dh.needles is not None else 0
            ne = int(dh.exclusions.shape[0]) if dh.exclusions is not None else 0
            if n == 0 and ne == 0:
                return True
            has_M, M_stack = dh._stack_optional_M(list(dh.needle_M_list))
            eh, es = dh._stack_optional_M(list(dh.exclusion_M_list))
            cpu = (lambda v: v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            tensors = dict(needles=dh.needles, needle_vals=dh.needle_vals, needle_indices=dh.needle_indices,
                           needle_penalty_radii=dh.needle_penalty_radii, needle_M_stack=M_stack,
                           needle_has_M=has_M,
                           needle_shrink_scale=torch.tensor(dh._needle_scales(), dtype=torch.float64),
                           needle_B=dh.needle_B, exclusions=dh.exclusions, exclusion_radii=dh.exclusion_radii,
                           exclusion_M_stack=es, exclusion_has_M=eh, penalty_mask=dh._penalty_mask)
            recs = []
            for r in dh.needles_results:
                pt = r.get("point")
                recs.append(dict(point=pt.detach().cpu().tolist() if isinstance(pt, torch.Tensor) else pt,
                                 **{k: _jsonable(r.get(k)) for k in ("value", "median_value", "zoom", "iteration",
                                                                     "reason", "lane")}))
            state = dict(version=1, t=now, n_rows=int(dh.X_all_actual.shape[0]),
                         tensors={k: cpu(v) for k, v in tensors.items()}, records=recs,
                         zones=_jsonable([dict(z, t_proc=z.get("t_proc", now)) for z in self.engine.zones]))
            p = self.dir / self.NEEDLE_STATE
            err = None
            tries = max(1, int(tries))
            for attempt in range(tries):
                try:
                    torch.save(state, p)
                    return True
                except (OSError, RuntimeError) as e:
                    err = e
                    if attempt < tries - 1:
                        time.sleep(0.1 * (attempt + 1))
            self.log(f"[lanes] WARNING could not write {p} ({err!r}); retried at the next decision")
            self.journal("needle_state_write_failed", path=str(p), error=repr(err))
        except Exception as e:
            try:
                self.log(f"[lanes] WARNING could not save the needle state ({e!r})")
                self.journal("needle_state_write_failed", error=repr(e), tb=traceback.format_exc())
            except Exception:
                pass
        return False

    def _save_unrestored_needles(self, st: Optional[dict], reason: str, partial: bool = False) -> List[str]:
        """After a FAILED restore (``st`` read but not restored) or an unreadable needle state: keep it in NEW
        files before anything can overwrite lanes/needles_state.pt with a smaller set (the next needle would):
        ``needles_state.unrestored_<YYYYmmdd-HHMMSS>.pt`` (torch.save of ``st``) and
        ``needles_state.unrestored_<YYYYmmdd-HHMMSS>.copy.pt`` (a copy of the existing needles_state.pt, never
        renamed). Both have the needles_state.pt layout (tensors / records / zones / t). Logs CRITICAL with the
        paths and the count; never raises. Returns the paths written.

        ``partial``: the needles WERE restored (from a snapshot) but needles_state.pt could not be read; only
        its copy is kept, logged as a WARNING (it may hold a needle the snapshot does not)."""
        if partial:
            written = []
            try:
                src = self.dir / self.NEEDLE_STATE
                if os.path.exists(src):
                    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    base, k = f"needles_state.unrestored_{stamp}", 0
                    while (self.dir / f"{base}.copy.pt").exists():
                        k += 1
                        base = f"needles_state.unrestored_{stamp}_{k}"
                    q = self.dir / f"{base}.copy.pt"
                    for attempt in range(3):
                        try:
                            shutil.copyfile(src, q)
                            written.append(str(q))
                            break
                        except OSError:
                            time.sleep(0.2 * (attempt + 1))
                self.log(f"[lanes] WARNING {reason}; " + (f"its contents kept in {written[0]}" if written else
                                                          "and it could not be copied"))
                self.journal("needles_unrestored_saved", path=written[0] if written else None, paths=written,
                             n=None, reason=reason, partial=True)
            except Exception as e:
                try:
                    self.journal("needles_unrestored_save_failed", error=repr(e), tb=traceback.format_exc())
                except Exception:
                    pass
            return written
        written = []
        try:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            base = f"needles_state.unrestored_{stamp}"
            k = 0
            while any((self.dir / f"{base}{x}").exists() for x in (".pt", ".copy.pt")):
                k += 1
                base = f"needles_state.unrestored_{stamp}_{k}"
            n = None
            if st is not None:
                n = self._count(st.get("tensors") or {}, "needles")
                p = self.dir / f"{base}.pt"
                for attempt in range(3):
                    try:
                        torch.save(dict(st, version=1, t=time.time(), reason=reason), p)
                        written.append(str(p))
                        break
                    except (OSError, RuntimeError):
                        time.sleep(0.2 * (attempt + 1))
            src = self.dir / self.NEEDLE_STATE
            if os.path.exists(src):
                q = self.dir / f"{base}.copy.pt"
                for attempt in range(3):
                    try:
                        shutil.copyfile(src, q)
                        written.append(str(q))
                        break
                    except OSError:
                        time.sleep(0.2 * (attempt + 1))
            what = "?" if n is None else str(n)
            if written:
                self.log(f"[lanes] CRITICAL the previous needles were NOT restored ({reason}); {what} needle(s) "
                         f"kept in {', '.join(written)} (needles_state.pt layout; recover them from there)")
            elif st is None and not os.path.exists(src):
                self.log(f"[lanes] CRITICAL the previous needles were NOT restored ({reason}); there is no "
                         f"needles_state.pt to keep (the engine snapshots in {self._engine_run_dir()} hold them)")
            else:
                self.log(f"[lanes] CRITICAL the previous needles were NOT restored ({reason}) and could not be "
                         f"saved to a new file")
            self.journal("needles_unrestored_saved", path=written[0] if written else None, paths=written, n=n,
                         reason=reason)
        except Exception as e:
            try:
                self.journal("needles_unrestored_save_failed", error=repr(e), tb=traceback.format_exc())
                self.log(f"[lanes] CRITICAL the previous needles were NOT restored ({reason}) and could not be "
                         f"saved ({e!r})")
            except Exception:
                pass
        return written

    def _needle_sig(self) -> tuple:
        """What a restart must not lose: needle / exclusion values, radii, shrinks, ellipsoids, medians, zones."""
        try:
            dh = self.engine.dh

            def r(v):
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    return None
                return round(v, 12) if np.isfinite(v) else None

            def tl(x):
                return () if x is None else tuple(r(v) for v in x.reshape(-1).tolist())
            m_sum = sum(float(m.sum()) for m in dh.needle_M_list if m is not None)
            return (tl(dh.needles), tl(dh.needle_vals), tl(dh.needle_penalty_radii),
                    tuple(r(s) for s in dh._needle_scales()), r(m_sum), tl(dh.exclusions), tl(dh.exclusion_radii),
                    tuple(r(x.get("median_value")) for x in dh.needles_results), len(self.engine.zones))
        except Exception:
            return ()

    def _on_needles_maybe_changed(self) -> None:
        """After every decision: when the needle set changed, resync needles.db (the lane overrides the newest
        needle's median with its own-rows median AFTER add_needle's sync, src/lanes/lane.py), write
        lanes/needles_state.pt and engine_ckpt.pt (the zone records). The set counts as persisted only once
        needles_state.pt was written: a failed write is retried at the next decision. Never raises."""
        try:
            sig = self._needle_sig()
            if sig == getattr(self, "_needle_sig_last", None):
                return
            saved = self._persist_needles(tries=3)       # the decision's only retry budget (<= 0.3 s)
            if saved:
                self._needle_sig_last = sig
            self._checkpoint()
        except Exception as e:
            self.journal("needle_state_error", error=repr(e), tb=traceback.format_exc())

    def _persist_needles(self, tries: int = 1) -> bool:
        """The DataHandler's own needles.db sync (a no-op unless the mirror is on) + lanes/needles_state.pt.
        Installed as the engine DataHandler's _sync_needles_db (add_needle's last step, called with no
        arguments: ONE try, no waiting - the same decision's _on_needles_maybe_changed retries, so a locked file
        costs <= 0.3 s per decision, not twice that). Never raises. True when needles_state.pt holds the current
        set (``_save_needle_state``)."""
        try:
            if getattr(self, "_dh_sync_needles_db", None) is not None:
                self._dh_sync_needles_db()
        except Exception:
            pass
        try:
            return bool(self._save_needle_state(tries=tries))
        except Exception:
            return False

    def _needles_db_target(self, needles_db) -> Optional[str]:
        """The needles.db the mirror writes: None (no mirror) for None / False; for True the ZOMBI_NEEDLES_DB
        path when set, else src.utils.needles_db.DEFAULT_NEEDLES_DB_PATH (<repo>/sql/needles.db, DiSCO's file);
        a str / Path as given. ZOMBI_NEEDLES_DB is honoured ONLY here: scripts/main.py's reset / preload on a
        hardware start and the legacy optimizer ignore it, so it is for tests and restart rehearsals only."""
        if needles_db is None or needles_db is False:
            return None
        if needles_db is True:
            env = os.environ.get("ZOMBI_NEEDLES_DB")
            if env:
                return env
            from src.utils import needles_db as ndb
            return str(ndb.DEFAULT_NEEDLES_DB_PATH)
        return str(needles_db)

    def _enable_needle_mirror(self, path: str) -> None:
        """Mirror the engine's needles into ``path`` now and on every new needle (DataHandler's own live sync,
        as the legacy optimizer used). Refused for the live <repo>/sql/needles.db under pytest."""
        try:
            if live_needles_db_refused(path):
                self.log(f"[lanes] WARNING needles.db mirror NOT enabled: {path} is the live DiSCO file and "
                         f"this is a test (ZOMBIHOP_TESTING=1)")
                return
            self.engine.dh._enable_needles_db(self.opt_dims, path=path)
            n = int(self.engine.dh.needles.shape[0]) if self.engine.dh.needles is not None else 0
            self.log(f"[lanes] needles.db mirror on ({n} needle(s) now): {path}")
        except Exception as e:
            self.log(f"[lanes] WARNING needles.db mirror not enabled ({e!r})")

    # ── clocks ──
    def now_engine(self) -> float:
        return time.time() - self.epoch

    # ── slots ──
    def _issue_and_place(self, lane: str, role: str, line, kind: str) -> bool:
        slot = wire.MAIN_SLOT[lane] if role == "main" else wire.BACKUP_SLOT[lane]
        t = line.tags
        meta = wire.make_line("lane", lane=lane, rid=t["rid"], hrid=f"{lane[0].upper()}{self.n_decide[lane]}"
                              + ("b" if role == "backup" else ""), role=role, act=int(t["act"]),
                              zoom=int(t["zoom"]), send=kind)
        meta["ends"] = np.asarray(line.ends, float).tolist()
        now = time.time()
        wid = self.router.try_issue(slot, meta, now)          # persisted to ids.json before it returns (R4)
        if wid is None:
            self.journal("issue_failed", lane=lane, role=role, engine_rid=t["rid"], slot=slot)
            self.log(f"[lanes] CRITICAL no free recipe id for {lane}-{role}; the slot keeps its recipe")
            return False
        old = self.slots[slot]
        if old is not None and old.state == "unread":
            self.router.supersede(old.wid, now)
        ends = np.asarray(line.ends, float)
        self.slots[slot] = _Slot(wid, line, lane, role, _to_wire10(ends[0], self.opt_dims),
                                 _to_wire10(ends[1], self.opt_dims), t_issue=now)
        self.journal("issue", lane=lane, role=role, slot=slot, wid=wid, engine_rid=t["rid"], act=t["act"],
                     zoom=t["zoom"], kind=kind, supersedes=None if old is None else old.wid,
                     ends=ends.tolist())
        return True

    def _slots_text(self) -> str:
        r = {k: ("-" if self.slots[k] is None else self.slots[k].wid) for k in range(4)}
        F, B = "front", "back"
        return (f"front main r={r[wire.MAIN_SLOT[F]]} backup r={r[wire.BACKUP_SLOT[F]]}"
                f" | back main r={r[wire.MAIN_SLOT[B]]} backup r={r[wire.BACKUP_SLOT[B]]}")

    def _write_full(self, reason: str, quiet: bool = False) -> None:
        for k in range(4):                     # G2 guard: a slot retired by a DiSCO restart is never re-sent
            s = self.slots[k]
            if s is not None and s.retired:
                self.journal("retired_slot_dropped", slot=k, wid=s.wid, reason=reason)
                self.log(f"[lanes] CRITICAL slot {k} still held r={s.wid} from before the DiSCO restart; "
                         f"withdrawn instead of re-sent")
                self.slots[k] = None
        entries = []
        for k in range(4):
            s = self.slots[k]
            if s is not None:
                entries.append((k, s.wid, list(s.start), list(s.end)))
        self.journal("slot_write", reason=reason, slots=[list(e) for e in entries],
                     states={k: (None if self.slots[k] is None else self.slots[k].state) for k in range(4)})
        communication.write_slots4(entries)
        if not quiet:
            self.log(f"[lanes] slots ({reason}): {self._slots_text()}")

    def _decide(self, lane: str, kind: str, log: bool = True) -> bool:
        ln = self.lanes[lane]
        now = self.now_engine()
        t0 = time.time()
        try:
            if kind == "refresh":
                r = ln.refresh(now)
            else:
                r = ln.decide(now)
        finally:
            self._on_needles_maybe_changed()     # a needle / zone of this decision is on disk before anything else
        if r is None:
            self.journal("decision", lane=lane, kind=kind, result="none", seconds=round(time.time() - t0, 3))
            return False
        main, backup, tags = r
        self.n_decide[lane] += 1
        self.journal("decision", lane=lane, kind=kind, main=tags["main"], backup=tags["backup"],
                     main_ends=np.asarray(main.ends, float).tolist(),
                     backup_ends=np.asarray(backup.ends, float).tolist(),
                     act=int(ln.act), zoom=int(ln.zoom), seconds=round(time.time() - t0, 3))
        if log:
            self.log(f"[lanes] {lane} decided ({kind}): act {int(ln.act)} zoom {int(ln.zoom)} "
                     f"in {time.time() - t0:.0f} s")
        ok_m = self._issue_and_place(lane, "main", main, kind)
        ok_b = self._issue_and_place(lane, "backup", backup, kind)
        self.t_last_write[lane] = time.time()
        return ok_m or ok_b

    def start(self) -> None:
        for lane in LANES:
            self._decide(lane, "start")
        self._write_full("start")
        self._checkpoint()
        self.log("[lanes] start: four slots written (front main/cache, back main/cache)")

    # ── packets ──
    def _apply_acks(self, last_read) -> List[str]:
        """'k' = (front last read id, back last read id): mark the slot read, tell the lane. Returns the lanes
        whose two slots are now both consumed."""
        empty = []
        if not last_read:
            return empty
        for rail, rid in zip(wire.RAILS, last_read):
            if rid is None:
                continue
            for k in (wire.MAIN_SLOT[rail], wire.BACKUP_SLOT[rail]):
                s = self.slots[k]
                if s is not None and s.wid == int(rid) and s.state == "unread":
                    s.state = "read"
                    self.journal("pickup", rail=rail, slot=k, wid=s.wid, engine_rid=s.line.rid, role=s.role)
                    self.log(f"[lanes] {rail} rail read r={s.wid} ({s.role})")
                    self.lanes[s.lane].on_pickup(s.line.rid, self.now_engine())
                    if s.role == "backup":                 # a backup is read only after its main
                        m = self.slots[wire.MAIN_SLOT[rail]]
                        if m is not None and m.state == "unread" and m.wid != s.wid:
                            m.state = "read"
            ms, bs = self.slots[wire.MAIN_SLOT[rail]], self.slots[wire.BACKUP_SLOT[rail]]
            if all(x is None or x.state == "read" for x in (ms, bs)):
                empty.append(rail)
        return empty

    def _packet_rows(self, decoded):
        vals = np.array([np.nan if v is None else v for v in (decoded.get("values") or [])], dtype=float).reshape(-1)
        comps = decoded.get("comps")
        if comps is None or len(comps) != len(vals) or len(vals) == 0:
            return None, None, f"values/comps mismatch ({len(vals)} vs {None if comps is None else len(comps)})"
        C = np.array(comps, dtype=float).reshape(len(vals), -1)
        keep = np.isfinite(vals) & (np.abs(C).sum(axis=1) > ZERO_COMP_TOL) & np.isfinite(C).all(axis=1)
        if not keep.any():
            return None, None, "no usable row (NaN y or all-zero compositions)"
        x = C[keep][:, self.opt_dims]
        y = np.asarray(self.y_to_opt(vals[keep]), dtype=float).reshape(-1)
        return x, y, None

    def on_packet(self, qid: int, pkt: dict, t_recv: float) -> None:
        self.journal("packet", qid=qid, raw=pkt, t_recv=t_recv)            # durable BEFORE acting
        decoded = wire.decode_objective(json.dumps(pkt))
        if decoded is None:
            self.journal("ignored", qid=qid, why="not an objective packet")
            return
        consumed = self._apply_acks(decoded.get("last_read"))
        now = time.time()
        routed = self.router.on_objective(decoded, now)
        disp = wire.disposition(routed)
        line = routed.get("line") if isinstance(routed.get("line"), dict) else {}
        self.journal("route", qid=qid, cls=routed.get("cls"), wid=routed.get("rid"), disposition=disp,
                     lane=line.get("lane"), engine_rid=line.get("rid"), lost=routed.get("lost"))
        refilled = set()
        if disp not in (wire.LANE, wire.INIT, wire.SHARED):
            self.log(f"[lanes] result r={routed.get('rid')}: {routed.get('cls')} ({disp}), not ingested")
        if disp in (wire.LANE, wire.INIT, wire.SHARED):
            x, y, why = self._packet_rows(decoded)
            if x is None:
                self.journal("no_data", qid=qid, wid=routed.get("rid"), why=why)
                self.log(f"[lanes] result id {routed.get('rid')}: {why}; nothing ingested")
            else:
                lane = line.get("lane") if disp == wire.LANE else None
                tags = self.engine.tags_of(line.get("rid")) if (lane and line.get("rid")) else None
                if tags is None and disp == wire.INIT:
                    tags = {"kind": "init"}
                enow = self.now_engine()
                n0 = int(self.engine.dh.X_all_actual.shape[0])
                p = dict(x=torch.as_tensor(x), y=torch.as_tensor(y), t=enow)
                for ln in self.lanes.values():                     # both lanes see it; ingested once
                    ln.on_result(p, tags, enow)
                rec = self.engine.packets[-1] if self.engine.packets else {}
                self.n_results += 1
                self._save_rows(qid, routed, line, rec, n0, y)
                self.journal("ingested", qid=qid, wid=routed.get("rid"), pid=rec.get("pid"), cls=rec.get("cls"),
                             n=rec.get("n"), lane=rec.get("lane"), act=rec.get("act"), zoom=rec.get("zoom"))
                self.log(f"[lanes] result r={routed.get('rid')} ({routed.get('cls')}) -> {lane or disp} "
                         f"act {line.get('act')} zoom {line.get('zoom')}: {len(y)} rows, best y={float(np.max(y)):.3f}")
                if lane in LANES and tags is not None:
                    self.t_last_result[lane] = time.time()
                    self.n_replenish[lane] = 0
                    self._decide(lane, "regular")                  # the owner decides (T3 run_fake order)
                    refilled.add(lane)
                    other = "back" if lane == "front" else "front"
                    om = self.slots[wire.MAIN_SLOT[other]]
                    if om is not None and om.state == "unread":    # 'after_other': refresh the unread main
                        if self._decide(other, "refresh"):
                            refilled.add(other)
                self._checkpoint()
        # a rail that consumed both of its slots gets new lines now (never leave it with nothing to read)
        replenished = False
        for rail in consumed:
            if rail not in refilled:
                self._decide(rail, "replenish")
                refilled.add(rail)
                replenished = True
        if replenished:
            self._checkpoint()                    # the lane state after these decisions too
        self._log_events()
        if refilled:
            self._write_full("result" if routed.get("cls") in ("matched", "late", "reprint") else "ack")

    # ── DiSCO restart (G2) ──
    def on_link_event(self, eid: int, ev: dict) -> None:
        if not isinstance(ev, dict) or ev.get("type") != "disco_restart":
            self.journal("link_event_ignored", eid=eid, raw=ev)
            return
        self._on_disco_restart(eid, ev)

    def _reissue(self, k: int, now: float) -> Optional[int]:
        """Slot k's unread line again, same content, under a NEW id (persisted before it returns, R4)."""
        s = self.slots[k]
        rec = self.router.ids.by_id.get(s.wid)
        tags = s.line.tags
        if rec is not None and isinstance(rec.line, dict):
            meta = dict(rec.line)
        else:
            meta = wire.make_line("lane", lane=s.lane, rid=tags["rid"], role=s.role, act=int(tags["act"]),
                                  zoom=int(tags["zoom"]))
            meta["ends"] = np.asarray(s.line.ends, float).tolist()
        meta.update(send="restart_reissue", reissue_of=int(s.wid))
        wid = self.router.try_issue(k, meta, now)
        if wid is None:
            self.journal("issue_failed", lane=s.lane, role=s.role, engine_rid=tags["rid"], slot=k,
                         why="disco_restart reissue")
            return None
        self.router.supersede(s.wid, now)
        self.slots[k] = _Slot(wid, s.line, s.lane, s.role, s.start, s.end, t_issue=now)
        self.journal("issue", lane=s.lane, role=s.role, slot=k, wid=wid, engine_rid=tags["rid"], act=tags["act"],
                     zoom=tags["zoom"], kind="restart_reissue", supersedes=s.wid,
                     ends=np.asarray(s.line.ends, float).tolist())
        return wid

    def _on_disco_restart(self, eid: int, ev: dict) -> None:
        """DiSCO restarted at t_ping (its startup self-test pinged us) and wiped its compositions.db. Every
        recipe issued before that is closed (router.on_disco_restart: read -> lost, unread -> unprinted; a
        result that still arrives for one routes by its id as 'late'), and every slot issued before it is
        retired, per lane:
          * main still unread: the lane's lines are still its current plan -> each unread slot is re-issued
            with the same content under a NEW id (a read one is withdrawn);
          * main already read (or empty): the rail has used its current line -> the slots are withdrawn and
            the lane decides its next main + backup (one decision, as on a result; an unread backup is
            superseded exactly as every decision supersedes it, so the rail again has two lines to read).
        Then ONE full map (new ts). The serial side holds the four-slot map until this map is written
        (communication.DiscoRestartLink.map_allowed)."""
        now = time.time()
        try:
            t_ping = float(ev.get("t_ping"))
        except (TypeError, ValueError):
            t_ping = now
        before = {k: (None if s is None else dict(wid=s.wid, state=s.state, t_issue=s.t_issue))
                  for k, s in self.slots.items()}
        self.journal("disco_restart", eid=eid, raw=ev, t_ping=t_ping, slots=before)   # durable BEFORE acting
        pre = {rid: r.status for rid, r in self.router.ids.by_id.items()
               if r.status in wire.OPEN and float(r.t_issue) < t_ping}
        lost = self.router.on_disco_restart(now, issued_before=t_ping)
        unprinted = sorted(rid for rid, st in pre.items() if st == "issued")
        self.t_restart = t_ping if self.t_restart is None else max(self.t_restart, t_ping)
        self.n_restarts += 1

        affected = [k for k in range(4) if self.slots[k] is not None and self.slots[k].t_issue < t_ping]
        reissued, withdrawn, decided, failed = {}, [], [], []
        for lane in LANES:
            ks = [k for k in (wire.MAIN_SLOT[lane], wire.BACKUP_SLOT[lane]) if k in affected]
            if not ks:
                continue
            m = self.slots[wire.MAIN_SLOT[lane]]
            keep_plan = m is not None and m.state == "unread"
            for k in ks:
                s = self.slots[k]
                s.retired = True                          # this object is never written again
                new = self._reissue(k, now) if (keep_plan and s.state == "unread") else None
                if new is None:
                    self.slots[k] = None
                    withdrawn.append(s.wid)
                else:
                    reissued[s.wid] = new
                    self.t_last_write[lane] = time.time()
            if not keep_plan:
                try:
                    if self._decide(lane, "restart", log=False):
                        decided.append(lane)
                except Exception as e:                    # never leave the old map in place
                    failed.append(lane)
                    self.journal("decision_error", lane=lane, kind="restart", error=repr(e),
                                 tb=traceback.format_exc())
        if affected:
            self._write_full("disco_restart", quiet=True)
        if decided or failed:
            self._checkpoint()
        self.journal("disco_restart_done", eid=eid, t_ping=t_ping, lost=lost, unprinted=unprinted,
                     reissued={str(a): b for a, b in reissued.items()}, withdrawn=withdrawn, decided=decided,
                     failed=failed, slots={k: (None if s is None else s.wid) for k, s in self.slots.items()})
        if not affected and not lost and not unprinted and t_ping < self.t_start:
            return                                        # an old restart this session has nothing to do with
        when = datetime.fromtimestamp(t_ping).strftime("%H:%M:%S")
        if affected:
            parts = []
            if reissued:
                parts.append("re-issued " + ", ".join(f"r={a}->{b}" for a, b in reissued.items()))
            if withdrawn:
                parts.append("withdrew r=" + ",".join(str(w) for w in withdrawn))
            if decided:
                parts.append("new lines for " + "+".join(decided))
            if failed:
                parts.append("DECISION FAILED for " + "+".join(failed))
            what = "; ".join(parts) + f"; slots {self._slots_text()}"
        else:
            what = "no slot predates it, nothing re-issued"
        closed = ([f"{len(lost)} read recipe(s) closed as lost {lost}"] if lost else []) + \
                 ([f"{len(unprinted)} unread closed as unprinted"] if unprinted else [])
        self.log(f"[lanes] DiSCO restarted (self-test ping {when}): {what}"
                 + ("; " + ", ".join(closed) if closed else ""))

    _KEY_EVENTS = ("activation", "zoom_in", "needle", "needle_hold", "zone", "dive_end",
                   "x_catchup_hold", "x_catchup_release")

    def _log_events(self) -> None:
        """One console line per key engine event since the last call (zoom-in, needle, new activation...)."""
        try:
            evs = self.engine.events
            i0 = getattr(self, "_ev_seen", 0)
            self._ev_seen = len(evs)
            for ev in evs[i0:]:
                k = ev.get("kind")
                if k not in self._KEY_EVENTS or ev.get("restored"):   # history carried over by a dive-exact resume
                    continue
                extra = {x: ev[x] for x in ("from_zoom", "to_zoom", "reason", "status", "why", "what", "gap")
                         if x in ev and ev[x] is not None}
                self.log(f"[lanes] {ev.get('lane', '?')} act {ev.get('act', '?')}: {k}"
                         + (f" zoom {ev['zoom']}" if "zoom" in ev else "")
                         + ("" if not extra else " " + " ".join(f"{a}={b}" for a, b in extra.items())))
        except Exception:
            pass

    def _save_rows(self, qid, routed, line, rec, n0, y):
        try:
            dh = self.engine.dh
            n1 = int(dh.X_all_actual.shape[0])
            Xa = dh.X_all_actual[n0:n1].detach().cpu().numpy()
            Xe = dh.X_all_expected[n0:n1].detach().cpu().numpy()
            Ys = dh.Y_all[n0:n1].detach().cpu().numpy().reshape(-1)
            for i in range(n1 - n0):
                self.rows_log("row", qid=qid, wid=routed.get("rid"), lane=line.get("lane"), act=line.get("act"),
                              zoom_sent=line.get("zoom"), engine_rid=line.get("rid"), pid=rec.get("pid"),
                              x_actual=Xa[i].tolist(), x_expected=Xe[i].tolist(), y_opt=float(Ys[i]))
        except Exception as e:
            self.journal("rows_error", qid=qid, error=repr(e))
            self.rows_log("raw_rows", qid=qid, wid=routed.get("rid"), y_opt=list(map(float, y)))

    def _checkpoint(self) -> None:
        try:
            dh = self.engine.dh
            state = dict(X_actual=dh.X_all_actual.detach().cpu(), X_expected=dh.X_all_expected.detach().cpu(),
                         Y=dh.Y_all.detach().cpu(), packets=self.engine.packets, zones=_jsonable(self.engine.zones),
                         events=self.engine.events[-500:],
                         decisions={n: ln.decisions for n, ln in self.lanes.items()},
                         lane_state={n: dict(act=int(ln.act), zoom=int(ln.zoom)) for n, ln in self.lanes.items()},
                         router=self.router.to_json(time.time()), t=time.time())
            torch.save(state, self.dir / "engine_ckpt.pt")              # in place (Dropbox: no os.replace)
        except Exception as e:
            self.journal("checkpoint_error", error=repr(e))

    # ── timers ──
    def tick(self) -> None:
        now = time.time()
        lost = self.router.expire(now)
        if lost:
            self.journal("expired", wids=lost)
        changed = False
        for lane in LANES:
            if self.n_decide[lane] == 1 and now - self.t_start >= SECOND_DECIDE_S:
                self._decide(lane, "start2")                       # the second start decision (lag 1)
                changed = True
            elif (self.n_decide[lane] >= 2 and now - max(self.t_last_result[lane], self.t_last_write[lane]) >= REPLENISH_S
                  and self.n_replenish[lane] < MAX_REPLENISH):
                self.n_replenish[lane] += 1
                self.journal("replenish_timer", lane=lane, n=self.n_replenish[lane])
                self._decide(lane, "replenish")
                changed = True
        if changed:
            self._write_full("timer")
            self._checkpoint()

    def poll_once(self) -> int:
        """Objective packets (queue table q) and link events (DiSCO restarts), in arrival order: an event with
        after_qid = A was received after q row A and before q row A + 1. Events are read FIRST, so every q
        row an event points at is already committed when q is read. Returns the number of packets."""
        n = 0
        items = [(int(after), 1, int(eid), ev)
                 for eid, after, ev in communication.read_link_events(self.after_event_id)]
        items += [(int(qid), 0, int(qid), pkt) for qid, pkt in communication.read_objective_queue(self.after_id)]
        items.sort(key=lambda it: it[:3])
        for _, is_event, i, obj in items:
            if is_event:
                if i <= self.after_event_id:
                    continue
                try:
                    self.on_link_event(i, obj)
                except Exception as e:                             # never lose the run on one event
                    self.journal("link_event_error", eid=i, error=repr(e), tb=traceback.format_exc())
                    self.log(f"[lanes] ERROR handling link event {i}: {e!r}")
                self.after_event_id = i
                self.journal("event_cursor", after_event_id=i)
                continue
            qid, pkt = i, obj
            if qid <= self.after_id:
                continue
            try:
                self.on_packet(qid, pkt, time.time())
            except Exception as e:                                 # never lose the run on one packet
                self.journal("packet_error", qid=qid, error=repr(e), tb=traceback.format_exc())
                self.log(f"[lanes] ERROR handling queue id {qid}: {e!r}")
            self.after_id = qid
            self.journal("cursor", after_id=qid)
            n += 1
        return n

    def resume_cursor(self) -> None:
        """Skip queue entries and link events already handled before a restart (journal 'cursor' /
        'event_cursor' records)."""
        for rec in _read_jsonl(self.dir / "journal.jsonl"):
            if rec.get("ev") == "cursor":
                self.after_id = max(self.after_id, int(rec.get("after_id", 0)))
            elif rec.get("ev") == "event_cursor":
                self.after_event_id = max(self.after_event_id, int(rec.get("after_event_id", 0)))

    def run_forever(self, stop: Optional[Callable[[], bool]] = None) -> None:
        if self.resumed:
            self.resume_cursor()
        self.start()
        try:
            while not (stop is not None and stop()):
                try:
                    self.poll_once()
                    self.tick()
                except Exception as e:
                    self.journal("loop_error", error=repr(e), tb=traceback.format_exc())
                    self.log(f"[lanes] ERROR in the driver loop (continuing): {e!r}")
                time.sleep(POLL_S)
        finally:
            self.journal("stop")
            try:
                self.engine.close()
            except Exception:
                pass


def run_lanes_hardware(run_dir, X_init_actual, X_init_expected, Y_init, bounds, hp, *, preset: str = "leader",
                       log: Callable = print, opt_dims: Optional[List[int]] = None,
                       y_to_optimizer: Optional[Callable] = None, stop: Optional[Callable[[], bool]] = None,
                       needles_db=None) -> None:
    """Build the lanes engine on the initial design and drive the hardware until stopped (module docstring).
    ``needles_db``: see LanesHardwareDriver (off by default; the hardware run, run_zombi_main, passes True)."""
    log("=" * 80)
    log(f"STARTING LANES OPTIMIZATION (preset={preset}; four-slot protocol)")
    log("=" * 80)
    drv = LanesHardwareDriver(run_dir, X_init_actual, X_init_expected, Y_init, bounds, hp, preset=preset, log=log,
                              opt_dims=opt_dims, y_to_optimizer=y_to_optimizer, needles_db=needles_db)
    drv.run_forever(stop=stop)


__all__ = ["run_lanes_hardware", "LanesHardwareDriver"]
