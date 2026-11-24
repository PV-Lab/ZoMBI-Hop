"""Lane-aware subclasses of ZoMBIHop / DataHandler / GPSimplex (lanes engine, Phase 1).

These classes replace the sandbox's per-instance monkey-patches (DESIGN.md 1.3.4) with explicit
subclass code. Every body below is copied from the sandbox; the decisions they make are the
sandbox's. Sandbox abbreviations: LZ = benchmarks/zoom_fix/lanes/lane_zombihop.py,
PF = benchmarks/zoom_fix/proposed_fixes.py.

    sandbox patch (installed in LaneStore.__init__)          here
    ------------------------------------------------------   --------------------------------------
    ZoMBIHop(objective=_no_objective, ...)  LZ 289-296       LaneZoMBIHop(engine=..., **kwargs)
    dh.current_activation_mask = store._current_activation_  LaneDataHandler.current_activation_mask
      mask                                  LZ 322, 409-414    -> engine._current_activation_mask()
    PF.install_own_best(dh)                 PF 1004-1038     LaneDataHandler.get_best_unpenalized /
                                                               get_best_in_bounds
    PF.install_fit_reuse(gp, stats, "all")  PF 716-866       LaneGPSimplex.fit (exact fit memo)
    store._install_timers()                 LZ 429-460       the timed overrides below
                                                               (engine.timers.timed(key))

``LaneZoMBIHop.__init__`` runs ``ZoMBIHop.__init__`` unchanged (it builds a plain ``DataHandler``
and a plain ``GPSimplex``, ZH 243 / 357) and THEN calls :func:`attach`, which swaps the class of
those two objects to the lane subclasses. That is the moment the sandbox patched them (right after
``ZoMBIHop(...)`` returned, LZ 297-333), so nothing done during construction (``save_init``, the
init snapshot) is affected, exactly as before. ``src/`` is not modified: the base classes are
never touched, only the two instances change class.

Equivalence notes (why a subclass method == the sandbox instance attribute):
* every patched method is reached through attribute lookup on the instance (``self.m(...)`` /
  ``dh.m(...)``), which finds an instance attribute (sandbox) or the subclass method (here) first;
* GPSimplex's internal ``self.fit(...)`` calls (``fit_from_data_handler``, ``fit_with_locality``)
  hit the memo in both forms;
* the timer sits OUTSIDE the memo (sandbox order: timer(memo(GPSimplex.fit)), LZ 324 then 333);
* the fit memo drops only the caller/origin bookkeeping of PF (``_fit_caller``,
  ``zoom_entry_caller``, ``fit_skip_origin``, ``fit_reused_by``, ``verify``, ``saved``): with
  mode "all" the skip decision never reads the origin (PF 833), and none of it draws RNG.

The engine (``src/lanes/store.py: LaneEngine``) is reached through ``self._lanes_engine``. It
provides ``_current_activation_mask()`` (LZ 409-414 body), ``timers`` (``src.lanes.instrument
.Timers``) and ``fit_stats`` (dict). ``attach`` creates ``timers`` / ``fit_stats`` on the engine
when it has none; the methods look both up on the engine at call time, so an engine that
(re)creates them later is still the one that counts. No import-time side effects.
"""
from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Optional, Tuple

import torch

from src.core.zombihop import ZoMBIHop
from src.utils.datahandler import DataHandler
from src.utils.gp_simplex import GPSimplex

__all__ = ["LaneZoMBIHop", "LaneDataHandler", "LaneGPSimplex", "attach", "FIT_STAT_KEYS"]

# Fit-memo statistics kept from PF install_fit_reuse (DESIGN.md 1.3.4): names and initial values.
FIT_STAT_KEYS = (("fit_calls", 0), ("fit_reused", 0), ("fit_s", 0.0), ("fit_rng_consumed", 0),
                 ("fit_skip_after_rng", 0))


def _no_objective(*a, **k):
    # LZ 289-290 (verbatim message)
    raise RuntimeError("lane ZoMBI-Hop never calls the blocking objective")


def _timed(obj, key: str):
    """``engine.timers.timed(key)`` of the object's engine; a no-op before :func:`attach` (the
    sandbox installed its timers only after ``ZoMBIHop(...)`` returned, LZ 333)."""
    eng = getattr(obj, "_lanes_engine", None)
    t = getattr(eng, "timers", None) if eng is not None else None
    return t.timed(key) if t is not None else nullcontext()


def _fit_stats(engine) -> dict:
    """The engine's fit-memo statistics dict, created / completed on demand (PF 792-800)."""
    st = getattr(engine, "fit_stats", None)
    if st is None:
        st = {}
        engine.fit_stats = st
    for k, v in FIT_STAT_KEYS:
        st.setdefault(k, v)
    return st


def _rng_state():
    # PF 807-811 (verbatim)
    s = [torch.get_rng_state()]
    if torch.cuda.is_available():
        s.append(torch.cuda.get_rng_state())
    return s


# ── ZoMBIHop ─────────────────────────────────────────────────────────────────────────────────

class LaneZoMBIHop(ZoMBIHop):
    """The lanes engine's ZoMBIHop: the blocking objective raises (the lanes call the stock helpers
    only, never ``run()``), and three helpers are timed (LZ 457-459). Built by the engine as
    ``LaneZoMBIHop(engine=self, X_init_actual=..., X_init_expected=..., Y_init=..., bounds=...,
    device=..., dtype=..., checkpoint_dir=..., run_uuid=..., resume=..., verbose=..., **hp)``."""

    _lanes_engine = None          # set by attach(); None during ZoMBIHop.__init__ (timers off)

    def __init__(self, *, engine, **kwargs):
        if "objective" in kwargs:
            raise TypeError("LaneZoMBIHop: the objective is fixed (lanes never call it)")
        super().__init__(objective=_no_objective, **kwargs)
        attach(self, engine)

    # timing pass-throughs (LZ 457-459: "convergence", "declare", "cap_zone")
    def _check_convergence_to_needle(self, *a, **k):
        with _timed(self, "convergence"):
            return super()._check_convergence_to_needle(*a, **k)

    def _declare_needle_from_point(self, *a, **k):
        with _timed(self, "declare"):
            return super()._declare_needle_from_point(*a, **k)

    def _penalize_capped_zone(self, *a, **k):
        with _timed(self, "cap_zone"):
            return super()._penalize_capped_zone(*a, **k)


# ── DataHandler ──────────────────────────────────────────────────────────────────────────────

class LaneDataHandler(DataHandler):
    """DataHandler whose "current activation" is the computing lane's own activation rows, whose
    best-point helpers read only those rows, with three timed methods. Never constructed directly:
    :func:`attach` swaps a built ``DataHandler`` to this class."""

    _lanes_engine = None

    # LZ 322 + 409-414: the instance patch dh.current_activation_mask = store._current_activation_mask
    def current_activation_mask(self) -> torch.Tensor:
        eng = self._lanes_engine
        if eng is None:
            raise RuntimeError("LaneDataHandler has no lanes engine (use src.lanes.handlers.attach)")
        return eng._current_activation_mask()

    # PF 1004-1038 (install_own_best), dh -> self. Own unpenalised rows only; in-box first, then own
    # anywhere, else (None, None, None).
    def _own_unpen(self) -> torch.Tensor:
        return self._penalty_mask & self.current_activation_mask()

    def get_best_unpenalized(self):
        m = self._own_unpen()
        if not bool(m.any()):
            return None, None, None
        idx = torch.where(m)[0]
        j = idx[int(self.Y_all[idx].reshape(-1).argmax())]
        return self.X_all_actual[j], self.Y_all[j], int(j.item())

    def get_best_in_bounds(self, bounds: torch.Tensor):
        m = self._own_unpen()
        if not bool(m.any()):
            return None, None, None
        idx = torch.where(m)[0]
        X = self.X_all_actual[idx]
        in_b = ((X >= bounds[0].unsqueeze(0)) & (X <= bounds[1].unsqueeze(0))).all(dim=1)
        if not bool(in_b.any()):
            return self.get_best_unpenalized()
        idx = idx[in_b]
        j = idx[int(self.Y_all[idx].reshape(-1).argmax())]
        return self.X_all_actual[j], self.Y_all[j], int(j.item())

    # timing pass-throughs (LZ 454-456: "add_points", "snapshot", "relabel")
    def add_all_points(self, *a, **k):
        with _timed(self, "add_points"):
            return super().add_all_points(*a, **k)

    def take_snapshot(self, *a, **k):
        with _timed(self, "snapshot"):
            return super().take_snapshot(*a, **k)

    def _relabel_pared_with_medians(self, *a, **k):
        with _timed(self, "relabel"):
            return super()._relabel_pared_with_medians(*a, **k)


# ── GPSimplex ────────────────────────────────────────────────────────────────────────────────

class LaneGPSimplex(GPSimplex):
    """GPSimplex with the exact fit memo of PF ``install_fit_reuse(mode="all")`` and three timed
    methods. Never constructed directly: :func:`attach` swaps a built ``GPSimplex`` to this class
    and gives it an empty memo (as the sandbox's fresh memo after construction, PF 791).

    A fit is skipped when (a) X and Y, after ``.to(device, dtype)``, are ``torch.equal`` to the
    inputs of the most recent COMPLETED fit, (b) ``gp`` / ``mll`` / ``comp_std`` are still the
    objects that fit produced and (c) ``data_handler.input_noise``, the seed prior (``_seed_keys``
    object, ``_seed_var``, ``_real_var``) are unchanged; a skip re-applies
    ``update_gp_noise(get_output_noise())``. Stats in ``engine.fit_stats``: ``fit_calls``,
    ``fit_reused``, ``fit_s`` (seconds in real fits), ``fit_rng_consumed`` (real fits that moved the
    CPU/CUDA RNG state), ``fit_skip_after_rng`` (skips of such a fit: 0 means the run is exactly the
    run without the memo). See PF 717-786 for the full argument."""

    _lanes_engine = None
    _lanes_memo: Optional[dict] = None

    def _lanes_other_inputs(self) -> Tuple[float, float, float]:
        # PF 813-816 (verbatim, gp_handler -> self, dh -> self.data_handler)
        dh = self.data_handler
        return (float(getattr(dh, "input_noise", 0.0) or 0.0),
                float(getattr(self, "_seed_var", 0.0) or 0.0),
                float(getattr(self, "_real_var", 0.0) or 0.0))

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        # timer outside the memo: sandbox order timer(memo(GPSimplex.fit)) (LZ 324, then 451)
        with _timed(self, "fit"):
            return self._lanes_fit(X, Y)

    def _lanes_fit(self, X: torch.Tensor, Y: torch.Tensor):
        # PF 818-864 with mode "all" and without the caller/origin bookkeeping (DESIGN.md 1.3.4)
        stats = _fit_stats(self._lanes_engine)
        memo = self._lanes_memo
        if memo is None:                                   # attach() always sets it; be safe
            memo = self._lanes_memo = {"X": None}
        stats["fit_calls"] += 1
        Xd = X.to(device=self.device, dtype=self.dtype)
        Yd = Y.to(device=self.device, dtype=self.dtype)
        same = (memo["X"] is not None
                and self.gp is memo["gp"] and self.mll is memo["mll"]
                and self.comp_std is memo["comp_std"]
                and memo["other"] == self._lanes_other_inputs()
                and getattr(self, "_seed_keys", None) is memo["seed_keys"]
                and memo["X"].shape == Xd.shape and memo["Y"].shape == Yd.shape
                and torch.equal(memo["X"], Xd) and torch.equal(memo["Y"], Yd))
        if same:
            stats["fit_reused"] += 1
            if memo["rng"]:
                stats["fit_skip_after_rng"] += 1
            self.data_handler.update_gp_noise(self.get_output_noise())
            return None
        memo.update(X=None, Y=None, gp=None, mll=None, comp_std=None)   # a fit that raises: empty
        rng0 = _rng_state()
        t0 = time.perf_counter()
        out = super().fit(X, Y)
        t_fit = time.perf_counter() - t0
        stats["fit_s"] += t_fit
        rng1 = _rng_state()
        consumed = any(not torch.equal(a, b) for a, b in zip(rng0, rng1))
        if consumed:
            stats["fit_rng_consumed"] += 1
        memo.update(X=Xd.clone(), Y=Yd.clone(), gp=self.gp, mll=self.mll,
                    comp_std=self.comp_std, t=t_fit, rng=consumed,
                    other=self._lanes_other_inputs(), seed_keys=getattr(self, "_seed_keys", None))
        return out

    # timing pass-throughs (LZ 451-452: "candidate", "ellipsoid")
    def get_candidate(self, *a, **k):
        with _timed(self, "candidate"):
            return super().get_candidate(*a, **k)

    def determine_penalty_ellipsoid(self, *a, **k):
        with _timed(self, "ellipsoid"):
            return super().determine_penalty_ellipsoid(*a, **k)


# ── attach ───────────────────────────────────────────────────────────────────────────────────

def attach(z: ZoMBIHop, engine) -> None:
    """Bind a freshly built ZoMBIHop's DataHandler and GPSimplex to a lanes engine.

    Replaces LZ 322 (mask), 323 (own-best), 324 (fit memo) and the dh / gp part of 333 (timers).
    ``z`` is normally a :class:`LaneZoMBIHop` (its ``__init__`` calls this). Refuses anything but a
    plain ``DataHandler`` / ``GPSimplex`` (both plain classes without ``__slots__``, built once each
    by ``ZoMBIHop.__init__``), so it can run once per ZoMBIHop. The engine must provide
    ``_current_activation_mask()``; ``timers`` and ``fit_stats`` are created on it when absent.
    """
    dh, gp = z.data_handler, z.gp_handler
    if type(dh) is not DataHandler or type(gp) is not GPSimplex:
        raise TypeError(f"attach: expected a plain DataHandler and GPSimplex, got "
                        f"{type(dh).__name__} / {type(gp).__name__} (already attached?)")
    if not callable(getattr(engine, "_current_activation_mask", None)):
        raise TypeError("attach: the engine has no _current_activation_mask()")
    if getattr(engine, "timers", None) is None:
        from src.lanes.instrument import Timers      # T1 module; imported only when needed
        engine.timers = Timers()
    _fit_stats(engine)                               # PF 792-800: stats exist from install on
    z._lanes_engine = engine
    dh._lanes_engine = engine
    gp._lanes_engine = engine
    gp._lanes_memo = {"X": None}                     # PF 791: empty memo
    dh.__class__ = LaneDataHandler
    gp.__class__ = LaneGPSimplex
