"""End-to-end smoke of the production lanes engine (DESIGN.md section 5, T3): ``build_lane_engine`` on a 4-D toy
landscape with the axial init, driven by a fake lag-1 objective (port of the sandbox self-test's ``run_fake``,
``benchmarks/zoom_fix/lanes/lane_selftest.py`` ST 156-232) through ``decide`` / ``on_result`` / ``on_pickup`` /
``refresh``, about 24 packets.

Checks: no exception; the independent auditor (``src.lanes.audit``, ST 250-431) and ``conservation()`` are ok;
one DataHandler snapshot per ingested packet, labelled ``act{A}_z{Z}_{F|B}_p{pid}`` (every packet here is
attributed), and one zero-row ``..._d{n}`` snapshot per lane decision (N6); every event kind is a frozen
sandbox kind; Fix 4 is installed while the engine lives and removed by ``close()``.

Small hyperparameters (``raw=64, n_restarts=8``, and ``nat_grad_max_steps=60``: get_candidate's 400 steps
took 54 of 70 s) so it runs in ~25-35 s on the 4090. Delta-format snapshots are never pruned
(datahandler.py 518-521), so every snapshot dir is still there for the count. Writes only under ``tmp_path``.
"""
from __future__ import annotations

import re
from collections import deque

import numpy as np
import pytest
import torch

import src.core.linebo as _linebo
import src.core.zombihop as _zh
import src.utils.domain as _domain
from src.lanes import build_lane_engine
from src.lanes.audit import audit
from src.lanes.catchup import catchup_report
from src.lanes.engine import engine_hparams
from src.lanes.events import STORE_EVENT_KINDS
from src.lanes.geometry import DEV, DT
from src.lanes.labels import LABEL_RE
from src.lanes.rng import is_seeded
from src.utils.axial_init import axial_design

_MODS = (_domain, _zh, _linebo)
D = 4
STEPS = 24
C1 = np.array([0.55, 0.25, 0.15, 0.05])
C2 = np.array([0.10, 0.20, 0.30, 0.40])


# ── REVIEW F9: restore the global RNG streams and the three random_simplex attributes ──
@pytest.fixture(autouse=True)
def _restore_global_state():
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    saved = {m: m.__dict__.get("random_simplex") for m in _MODS}
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        for m, f in saved.items():
            if f is not None:
                setattr(m, "random_simplex", f)


def predict(X) -> np.ndarray:
    """Noise-free toy landscape: two Gaussian bumps on the 4-simplex."""
    X = np.asarray(X, float).reshape(-1, D)
    g = lambda c, s: np.exp(-((X - c) ** 2).sum(axis=1) / (2 * s * s))  # noqa: E731
    return g(C1, 0.15) + 0.6 * g(C2, 0.15)


class Measurer:
    """A printed line: 12 points from L to R with composition jitter, measured with noise."""

    def __init__(self, seed: int, n: int = 12, x_sd: float = 0.01, y_sd: float = 0.03):
        self.rng = np.random.default_rng([seed, 4])
        self.n, self.x_sd, self.y_sd = n, x_sd, y_sd

    def __call__(self, L: torch.Tensor, R: torch.Tensor):
        L, R = L.detach().cpu().numpy(), R.detach().cpu().numpy()
        t = np.linspace(0.0, 1.0, self.n)[:, None]
        X = L + t * (R - L) + self.rng.normal(0.0, self.x_sd, (self.n, D))
        X = np.clip(X, 0.0, None)
        X = X / X.sum(axis=1, keepdims=True)
        yt = predict(X)
        y = yt + self.rng.normal(0.0, self.y_sd, self.n)
        return (torch.as_tensor(X, device=DEV, dtype=DT), torch.as_tensor(y, device=DEV, dtype=DT), yt)


def axial_init(measure):
    """LZ 1432-1445: the axial traces, measured at once (X_expected = the commanded face -> pure line)."""
    design = axial_design(D, lower=np.zeros(D), upper=np.ones(D), drop="auto")
    Xa, Xe, Ys = [], [], []
    for tr in design.traces:
        L = torch.tensor(np.asarray(tr.end, float), device=DEV, dtype=DT)
        R = torch.tensor(np.asarray(tr.start, float), device=DEV, dtype=DT)
        x, y, _ = measure(L, R)
        t = torch.linspace(0, 1, x.shape[0], device=DEV, dtype=DT)[:, None]
        Xa.append(x)
        Xe.append(L + t * (R - L))
        Ys.append(y)
    return torch.cat(Xa), torch.cat(Xe), torch.cat(Ys).reshape(-1, 1)


def run_fake(engine, measure, *, steps: int, pickup: str = "after_other") -> dict:
    """ST 156-232 (null_rate 0, no boost / injection / forced candidate failures)."""
    F, B = engine.lanes["front"], engine.lanes["back"]
    lanes = {"front": F, "back": B}
    rails = {n: deque() for n in lanes}
    slots = {n: None for n in lanes}             # (main, backup) last written, main unread?
    unread = {n: False for n in lanes}
    now = 0.0
    log = dict(refresh_calls=0, refresh_new=0, packets=0)

    def pick(n):
        # a rail holds at most [line being measured, line printed]; it reads its next recipe
        # once the previous one is measured (lag 1)
        if slots[n] is None or len(rails[n]) >= 2:
            return
        main, backup = slots[n]
        if unread[n]:
            rails[n].append(main)
            lanes[n].on_pickup(main.rid, now)
            unread[n] = False
        else:                                    # main already read: the rail takes the backup
            rails[n].append(backup)
            lanes[n].on_pickup(backup.rid, now)
            slots[n] = None

    # start: each lane decides twice (line 1 read at once, line 2 queued and read)
    for _ in range(2):
        for n, ln in lanes.items():
            m, b, _ = ln.decide(now)
            slots[n], unread[n] = (m, b), True
            pick(n)
    order = ["front", "back"]
    for step in range(steps):
        X = order[step % 2]
        Yn = "back" if X == "front" else "front"
        now += 25 * 60.0
        if pickup == "before_other":
            pick(Yn)
        if not rails[X]:
            pick(X)
        line = rails[X].popleft()
        L = torch.as_tensor(line.ends[0], device=DEV, dtype=DT)
        R = torch.as_tensor(line.ends[1], device=DEV, dtype=DT)
        x, y, yt = measure(L, R)
        pkt = dict(x=x, y=y, y_true=yt, t=now)
        for ln in lanes.values():                # both lanes see every packet; the store ingests it once
            ln.on_result(pkt, line.tags, now)
        log["packets"] += 1
        m, b, _ = lanes[X].decide(now)
        slots[X], unread[X] = (m, b), True
        if pickup == "after_other":
            if slots[Yn] is not None and unread[Yn]:
                log["refresh_calls"] += 1
                r = lanes[Yn].refresh(now)
                if r is not None:
                    log["refresh_new"] += 1
                    slots[Yn] = (r[0], r[1])
            pick(Yn)
    return log


def test_engine_smoke(tmp_path):
    torch.manual_seed(0)                         # the hardware driver's job (REVIEW F3); restored by the fixture
    measure = Measurer(seed=0)
    Xa, Xe, Y0 = axial_init(measure)
    hp = engine_hparams(D)
    hp.update(raw=64, n_restarts=8, nat_grad_max_steps=60)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    orig = {m: m.__dict__["random_simplex"] for m in _MODS}

    engine = build_lane_engine(Xa, Xe, Y0, np.stack([np.zeros(D), np.ones(D)]), preset="leader", hparams=hp,
                               seed=0, checkpoint_dir=str(ckpt), predict=predict)
    committed = []
    real_commit = engine.commit

    def spy(kind, label, info=None):             # instance spy: records, then the real commit
        committed.append((kind, label))
        return real_commit(kind, label, info)
    engine.commit = spy
    try:
        assert all(is_seeded(m.__dict__["random_simplex"]) for m in _MODS)      # Fix 4 while the engine lives
        assert engine.opts.is_leader()
        assert [c.__name__ for c in type(engine.lanes["front"]).__mro__][:3] == ["LeaderLane", "CatchupMixin",
                                                                                 "DiveLane"]
        assert (engine.lanes["front"].act, engine.lanes["back"].act) == (0, 1)
        assert engine.build_info["activation_budget"] == 24
        assert engine.build_info["hp_z"]["z2"]["min_iters"] == 3

        log = run_fake(engine, measure, steps=STEPS, pickup="after_other")

        # ── the auditor and conservation ──
        au = audit(engine)
        assert au["ok"], au["fails"]
        cons = engine.conservation()
        assert cons["ok"], cons
        assert len(engine.packets) == STEPS == log["packets"]
        assert {p["cls"] for p in engine.packets} <= {"own_current", "own_outside", "own_earlier", "closed_act",
                                                      "duplicate"}
        assert {e["kind"] for e in engine.events} <= STORE_EVENT_KINDS

        # ── commits: one per ingested packet (labels start act), one per decision ──
        n_dec = sum(len(ln.decisions) for ln in engine.lanes.values())
        pk = [lab for kind, lab in committed if kind == "packet"]
        dc = [lab for kind, lab in committed if kind == "decision"]
        assert len(pk) == STEPS and len(dc) == n_dec
        assert all(lab.startswith("act") and LABEL_RE.match(lab) for lab in pk + dc)
        for p, lab in zip(engine.packets, pk):
            tag = {"front": "F", "back": "B"}[p["lane"]]
            assert lab == f"act{p['act']}_z{p['zoom']}_{tag}_p{p['pid']}"
        for ln in engine.lanes.values():                      # decision labels carry the decision index
            idx = sorted(int(m.group(5)) for m in map(LABEL_RE.match, dc) if m.group(3) == ln.name[0].upper())
            assert idx == list(range(len(ln.decisions)))
        # the same labels exist as snapshot directories (NNNN_<label>), in commit order
        snaps = sorted(p.name for p in ckpt.rglob("snapshots/*") if p.is_dir())
        lane_snaps = [n.split("_", 1)[1] for n in snaps if LABEL_RE.match(n.split("_", 1)[1] if "_" in n else "")]
        assert lane_snaps == [lab for _, lab in committed]
        assert engine.dh._snapshot_count >= len(committed)

        # ── reports run ──
        sm = engine.summary()
        assert sm["packets"] == STEPS and sm["decisions"] == n_dec
        assert sm["fit_calls"] and sm["fit_calls"] >= sm["fit_reused"] >= 0
        assert sm["fit_skip_after_rng"] is not None and sm["fit_rng_consumed"] is not None
        rep = catchup_report(engine)
        assert rep["m"] == 8 and len(rep["dives"]) >= 2
        assert engine.stats["line_gate"]["calls"] >= n_dec
        assert all(re.match(r"^(zbudget|catchup)\.", k) or k == "line_gate" for k in engine.stats)
    finally:
        engine.close()
    # Fix 4 released: the three module attributes are the original objects again
    assert all(m.__dict__["random_simplex"] is orig[m] for m in _MODS)
    engine.close()                                            # idempotent
