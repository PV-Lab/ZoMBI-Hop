"""LaneLineBO (src/lanes/linebo.py): Fix 8 incumbent-mean lines and the line gate.

DESIGN.md section 5 row ``test_lanes_incumbent``. Stub ZoMBIHop / DataHandler / GP (d = 4, CUDA
tensors); no GP fit, no disk writes. The optional sandbox cross-check compares against
``benchmarks/zoom_fix/proposed_fixes.IncumbentMeanLineBO`` and skips when the sandbox is absent.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import src.core.linebo as _linebo
import src.core.zombihop as _zh
import src.utils.domain as _domain
from src.core.linebo import LineBO
from src.lanes import linebo as LL
from src.lanes.linebo import LaneLineBO, no_face_view, valid_lines

D = 4
DT = torch.float64
DEV = torch.device("cuda")
_MODS = (_domain, _zh, _linebo)


# ── F9: restore the global RNG streams and the three random_simplex attributes ──
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


@contextmanager
def _seeded_simplex():
    """Test-local copy of the Fix 4 wrapper (PF 406-420), so a torch.manual_seed fixes every draw
    (without it random_simplex(seed=None) repeats the same samples and the comparison below
    would hold trivially). Restored on exit (and again by the autouse fixture)."""
    orig = _domain.random_simplex
    saved = {m: m.__dict__.get("random_simplex") for m in _MODS}

    def random_simplex(*args, seed=None, **kwargs):
        if seed is None:
            seed = int(torch.randint(0, 2 ** 62, (1,)).item())
        return orig(*args, seed=seed, **kwargs)

    for m in _MODS:
        if hasattr(m, "random_simplex"):
            setattr(m, "random_simplex", random_simplex)
    try:
        yield
    finally:
        for m, f in saved.items():
            if f is not None:
                setattr(m, "random_simplex", f)


# ── stubs ──
C = torch.tensor([0.30, 0.20, 0.35, 0.15], dtype=DT, device=DEV)        # GP mean peak (sum 1)
NEEDLE = torch.tensor([0.25, 0.30, 0.30, 0.15], dtype=DT, device=DEV)   # a repulsive zone centre
BONUS = torch.tensor([3.0, -2.0, 1.0, -1.5], dtype=DT, device=DEV)      # "UCB" bonus direction


def _mean(x: torch.Tensor) -> torch.Tensor:
    return -((x - C) ** 2).sum(dim=-1)


class _GP:
    def predict(self, X):
        X = X.reshape(-1, D)
        return _mean(X).reshape(-1, 1), torch.full((X.shape[0], 1), 0.01, dtype=DT, device=X.device)


class _Base:
    """'UCB' = GP mean + a strong linear bonus (so ranking by UCB would differ from mean)."""

    def __call__(self, X):
        x = X.reshape(-1, D)
        return _mean(x) + 5.0 * (x @ BONUS)


class _Acq:
    """RepulsiveAcquisition-like: base + needle-zone repulsion + simplex-face term (eps > 0)."""

    def __init__(self, eps: float = 0.128, lam: float = 100.0):
        self.base = _Base()
        self.boundary_epsilon = eps
        self.lam = lam

    def __call__(self, X):
        x = X.reshape(-1, D)
        val = self.base(x)
        dist = (x - NEEDLE).norm(dim=1)
        val = val - self.lam * torch.clamp(0.15 - dist, min=0.0) ** 2
        if self.boundary_epsilon > 0:
            val = val - self.lam * torch.exp(-x / self.boundary_epsilon).sum(dim=1)
        return val


class _DH:
    def __init__(self, X, Y, pen, own):
        self.X_all_actual = X
        self.Y_all = Y.reshape(-1, 1)
        self._penalty_mask = pen
        self._own = own

    def current_activation_mask(self):
        return self._own

    def get_penalty_mask(self, X):
        return torch.ones(X.shape[:-1], dtype=torch.bool, device=X.device)

    def min_box_width(self):
        return 0.05


class _Z:
    def __init__(self, dh):
        self.data_handler = dh
        self.gp_handler = _GP()
        self.full_bounds = torch.stack([torch.zeros(D, dtype=DT, device=DEV),
                                        torch.ones(D, dtype=DT, device=DEV)])

    def _is_global_bounds(self, bounds, eps: float = 0.01) -> bool:     # ZH 447-456
        fb = self.full_bounds
        return (torch.abs(bounds[0] - fb[0]).max().item() < eps and
                torch.abs(bounds[1] - fb[1]).max().item() < eps)


BOX = torch.tensor([[0.10, 0.05, 0.15, 0.05], [0.55, 0.45, 0.60, 0.50]], dtype=DT, device=DEV)
FULL = torch.stack([torch.zeros(D, dtype=DT, device=DEV), torch.ones(D, dtype=DT, device=DEV)])


def _rows(n: int = 400, seed: int = 7):
    """Deterministic rows on the simplex (Dirichlet(1) from a private CPU generator)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    E = -torch.log(torch.rand(n, D, generator=g, dtype=DT, device="cpu").clamp(min=1e-300))
    X = (E / E.sum(dim=1, keepdim=True)).to(DEV)
    idx = torch.arange(n, device="cpu")
    own = (idx % 2 == 0).to(DEV)
    pen = (idx % 3 != 0).to(DEV)                       # True = unpenalised
    Y = torch.rand(n, generator=g, dtype=DT, device="cpu").to(DEV)
    return X, Y, pen, own


def _make(dh, *, engine=None, incumbent=True, num_lines=10):
    z = _Z(dh)
    lb = LaneLineBO(None, D, num_points_per_line=100, num_lines=num_lines, device=str(DEV),
                    zombi_ref=[z], engine=engine, incumbent=incumbent)
    return z, lb


def _engine(box):
    return SimpleNamespace(active=SimpleNamespace(box=box), stats={}, timers=None)


def _x_tell():
    return torch.full((D,), 1.0 / D, dtype=DT, device=DEV)


def _point_segment_dist(p, a, b):
    ab = b - a
    t = ((p - a) * ab).sum(dim=1) / (ab * ab).sum(dim=1).clamp(min=1e-300)
    t = t.clamp(0.0, 1.0)
    return (a + t[:, None] * ab - p).norm(dim=1)


# ── tests ──

def test_global_bounds_identical_to_stock_linebo():
    X, Y, pen, own = _rows()
    dh = _DH(X, Y, pen, own)
    _, lb = _make(dh, engine=_engine(FULL))
    ref = LineBO(None, D, num_points_per_line=100, num_lines=10, device=str(DEV))
    acq = _Acq()
    with _seeded_simplex():
        torch.manual_seed(123)
        a = lb.ranked_line_endpoints(_x_tell(), FULL, acq)
        torch.manual_seed(123)
        b = ref.ranked_line_endpoints(_x_tell(), FULL, acq)
        torch.manual_seed(124)
        c = ref.ranked_line_endpoints(_x_tell(), FULL, acq)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(b[0], c[0])                 # the seed matters: comparison not vacuous
    assert lb.n_fallback == 0 and lb.last_incumbent is None
    st = lb._lanes_engine.stats["line_gate"]
    assert st["calls"] == 1 and st["invalid_rows"] == 0 and st["fallback"] == 0
    assert st["rows_returned"] == a[0].shape[0]


def test_incumbent_off_is_stock_linebo_at_every_zoom():
    X, Y, pen, own = _rows()
    _, lb = _make(_DH(X, Y, pen, own), incumbent=False)
    ref = LineBO(None, D, num_points_per_line=100, num_lines=10, device=str(DEV))
    acq = _Acq()
    with _seeded_simplex():
        torch.manual_seed(5)
        a = lb.ranked_line_endpoints(_x_tell(), BOX, acq)
        torch.manual_seed(5)
        b = ref.ranked_line_endpoints(_x_tell(), BOX, acq)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert lb.last_incumbent is None and lb.n_fallback == 0


def test_zoom_box_chords_through_incumbent_ranked_by_mean_plus_zone(monkeypatch):
    X, Y, pen, own = _rows()
    dh = _DH(X, Y, pen, own)
    eng = _engine(BOX)
    _, lb = _make(dh, engine=eng)
    acq = _Acq()

    # expected incumbent: argmax of the stub GP mean over own & unpenalised & in-box rows
    inb = ((X >= BOX[0] - 1e-9) & (X <= BOX[1] + 1e-9)).all(dim=1)
    keep = own & pen & inb
    assert int(keep.sum()) >= 3
    x_star = X[keep][int(_mean(X[keep]).argmax())]

    seen = {}
    orig = LL.batch_line_bounds_segments

    def spy(x0, Dm, b):
        out = orig(x0, Dm, b)
        seen["x0"], seen["out"] = x0.clone(), out
        return out

    monkeypatch.setattr(LL, "batch_line_bounds_segments", spy)
    with _seeded_simplex():
        torch.manual_seed(11)
        xl, xr = lb.ranked_line_endpoints(_x_tell(), BOX, acq)

    assert torch.equal(lb.last_incumbent, x_star)
    assert torch.equal(seen["x0"], x_star)
    assert lb.n_fallback == 0
    assert xl.shape == (10, D) and xr.shape == (10, D)
    # every chord passes through x*
    assert float(_point_segment_dist(x_star.expand(10, D), xl, xr).max()) < 1e-9
    # endpoints in the box and on the simplex
    assert bool(valid_lines(xl, xr, BOX, tol=1e-9).all())
    # order = descending chord mean of (GP mean + zone term); no face term, no UCB bonus
    l, r = seen["out"][0], seen["out"][1]
    good = (r - l).norm(dim=1) > 1e-9
    cl, cr = l[good][:100], r[good][:100]
    assert cl.shape[0] == 100
    pts = lb._line_points(cl, cr)
    mu = _mean(pts)
    nf = no_face_view(acq)
    zone = lb._acquisition_at_points(pts, nf) - lb._acquisition_at_points(pts, nf.base)
    assert float(zone.max()) <= 1e-12 and float(zone.min()) < 0.0      # the zone term is active
    want = torch.argsort((mu + zone).mean(dim=1), descending=True)[:10]
    assert torch.equal(xl, cl[want]) and torch.equal(xr, cr[want])
    # and the test discriminates: ranking with the face term or the UCB bonus would differ
    with_face = torch.argsort(lb._acquisition_at_points(pts, acq).mean(dim=1), descending=True)[:10]
    with_ucb = torch.argsort(lb._acquisition_at_points(pts, nf).mean(dim=1), descending=True)[:10]
    assert not torch.equal(with_face, want) and not torch.equal(with_ucb, want)
    assert acq.boundary_epsilon == 0.128                # the caller's acquisition is not modified
    st = eng.stats["line_gate"]
    assert st == dict(calls=1, invalid_rows=0, fallback=0, degenerate_main=0, rows_returned=10)


def test_no_own_rows_falls_back_to_random_chords(monkeypatch):
    X, Y, pen, own = _rows()
    dh = _DH(X, Y, pen, torch.zeros_like(own))
    _, lb = _make(dh, engine=_engine(BOX))
    calls = []
    orig = LL.random_chords_through_simplex

    def spy(k, b, dev, dt, domain=None):
        out = orig(k, b, dev, dt, domain=domain)
        calls.append((k, b.clone(), out))
        return out

    monkeypatch.setattr(LL, "random_chords_through_simplex", spy)
    with _seeded_simplex():
        torch.manual_seed(3)
        xl, xr = lb.ranked_line_endpoints(_x_tell(), BOX, _Acq())
    assert lb.last_incumbent is None and lb.n_fallback == 1
    assert len(calls) == 1 and calls[0][0] == 100 and torch.equal(calls[0][1], BOX)
    cand_l, cand_r = calls[0][2]
    assert cand_l.shape[0] > 10
    for i in range(xl.shape[0]):              # every returned line is one of the random chords
        hit = ((cand_l == xl[i]).all(dim=1) & (cand_r == xr[i]).all(dim=1)).any()
        assert bool(hit)
    assert bool(valid_lines(xl, xr, BOX, tol=1e-9).all())


def test_fewer_than_ten_chords_through_incumbent_falls_back(monkeypatch):
    X, Y, pen, own = _rows()
    _, lb = _make(_DH(X, Y, pen, own), engine=_engine(BOX))
    orig = LL.batch_line_bounds_segments

    def few(x0, Dm, b):
        l, r, a, c, ok = orig(x0, Dm, b)
        r = r.clone()
        r[9:] = l[9:]                         # only 9 non-degenerate chords left
        return l, r, a, c, ok

    n_rand = []
    orig_rand = LL.random_chords_through_simplex

    def spy(k, b, dev, dt, domain=None):
        n_rand.append(k)
        return orig_rand(k, b, dev, dt, domain=domain)

    monkeypatch.setattr(LL, "batch_line_bounds_segments", few)
    monkeypatch.setattr(LL, "random_chords_through_simplex", spy)
    with _seeded_simplex():
        torch.manual_seed(4)
        lb.ranked_line_endpoints(_x_tell(), BOX, _Acq())
    assert lb.last_incumbent is not None      # there is an incumbent, but too few chords
    assert lb.n_fallback == 1 and n_rand == [100]


def test_gate_drops_invalid_rows_and_falls_back_when_none_valid(monkeypatch):
    X, Y, pen, own = _rows()
    eng = _engine(BOX)
    _, lb = _make(_DH(X, Y, pen, own), engine=eng)
    good_l = torch.tensor([[0.20, 0.30, 0.30, 0.20], [0.30, 0.20, 0.40, 0.10]], dtype=DT, device=DEV)
    good_r = torch.tensor([[0.30, 0.20, 0.30, 0.20], [0.20, 0.30, 0.30, 0.20]], dtype=DT, device=DEV)
    off_simplex = torch.tensor([[0.20, 0.30, 0.30, 0.30]], dtype=DT, device=DEV)        # sum 1.1
    out_of_box = torch.tensor([[0.70, 0.10, 0.10, 0.10]], dtype=DT, device=DEV)          # x0 > 0.55
    replies, args = [], []

    def fake(x_tell, bounds=None, acquisition_function=None):
        args.append((x_tell, bounds, acquisition_function))
        return replies.pop(0)

    monkeypatch.setattr(lb, "_ranked", fake)
    acq = _Acq()
    # (1) one invalid row among valid ones: dropped, no fallback
    replies.append((torch.cat([good_l[:1], off_simplex, good_l[1:]]),
                    torch.cat([good_r[:1], good_r[:1], good_r[1:]])))
    xl, xr = lb.ranked_line_endpoints(_x_tell(), BOX, acq)
    assert torch.equal(xl, good_l) and torch.equal(xr, good_r)
    st = eng.stats["line_gate"]
    assert st == dict(calls=1, invalid_rows=1, fallback=0, degenerate_main=0, rows_returned=2)
    # (2) no valid row: the generator runs again with the same arguments
    replies.append((torch.cat([out_of_box, off_simplex]), torch.cat([good_r[:1], good_r[:1]])))
    replies.append((good_l[:1], good_l[:1]))                    # fallback reply (degenerate main)
    args.clear()
    xl, xr = lb.ranked_line_endpoints(_x_tell(), BOX, acq)
    assert torch.equal(xl, good_l[:1]) and torch.equal(xr, good_l[:1])
    assert len(args) == 2 and args[0][1] is args[1][1] and args[0][2] is args[1][2] is acq
    assert st == dict(calls=2, invalid_rows=3, fallback=1, degenerate_main=1, rows_returned=3)
    assert not replies


def test_gate_skipped_outside_a_lane_context(monkeypatch):
    X, Y, pen, own = _rows()
    eng = SimpleNamespace(active=None, stats={}, timers=None)
    _, lb = _make(_DH(X, Y, pen, own), engine=eng)
    bad = torch.tensor([[0.70, 0.10, 0.10, 0.10]], dtype=DT, device=DEV)
    monkeypatch.setattr(lb, "_ranked", lambda *a: (bad, bad))
    xl, _ = lb.ranked_line_endpoints(_x_tell(), BOX, _Acq())
    assert torch.equal(xl, bad) and "line_gate" not in eng.stats


def test_lines_timer_wraps_the_gate():
    try:
        from src.lanes.instrument import Timers
    except ImportError:                     # T1 module not there yet: a stub with the same contract
        Timers = _StubTimers
    X, Y, pen, own = _rows()
    eng = _engine(BOX)
    eng.timers = Timers()
    _, lb = _make(_DH(X, Y, pen, own), engine=eng)
    with _seeded_simplex():
        lb.ranked_line_endpoints(_x_tell(), BOX, _Acq())
        lb.ranked_line_endpoints(_x_tell(), FULL, _Acq())
    assert eng.timers.comp["lines_n"] == 2 and eng.timers.comp["lines"] > 0.0


def test_valid_lines_and_no_face_view():
    box = BOX
    l = torch.tensor([[0.2, 0.3, 0.3, 0.2], [0.2, 0.3, 0.3, 0.3], [0.7, 0.1, 0.1, 0.1],
                      [float("nan"), 0.3, 0.3, 0.2]], dtype=DT, device=DEV)
    r = torch.tensor([[0.3, 0.2, 0.3, 0.2]] * 4, dtype=DT, device=DEV)
    assert valid_lines(l, r, box).tolist() == [True, False, False, False]
    acq = _Acq()
    v = no_face_view(acq)
    assert v is not acq and v.boundary_epsilon == 0.0 and acq.boundary_epsilon == 0.128
    assert v.base is acq.base
    plain = _Base()
    assert no_face_view(plain) is plain and no_face_view(None) is None


def test_matches_sandbox_incumbent_mean_linebo():
    """Cross-check against the sandbox class the leader ran with (skipped without the sandbox)."""
    PF = pytest.importorskip("benchmarks.zoom_fix.proposed_fixes")
    X, Y, pen, own = _rows()
    for box, seed in ((BOX, 21), (FULL, 22)):
        dh = _DH(X, Y, pen, own)
        z, lb = _make(dh, engine=_engine(box))
        ref = PF.IncumbentMeanLineBO(None, D, num_points_per_line=100, num_lines=10, device=str(DEV),
                                     zombi_ref=[z])
        acq = _Acq()
        with _seeded_simplex():
            torch.manual_seed(seed)
            a = lb.ranked_line_endpoints(_x_tell(), box, acq)
            torch.manual_seed(seed)
            b = ref.ranked_line_endpoints(_x_tell(), box, acq)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        assert lb.n_fallback == ref.n_fallback
        assert (lb.last_incumbent is None) == (getattr(ref, "last_incumbent", None) is None)
    # no own rows: both fall back the same way
    dh = _DH(X, Y, pen, torch.zeros_like(own))
    z, lb = _make(dh, engine=_engine(BOX))
    ref = PF.IncumbentMeanLineBO(None, D, num_points_per_line=100, num_lines=10, device=str(DEV),
                                 zombi_ref=[z])
    with _seeded_simplex():
        torch.manual_seed(23)
        a = lb.ranked_line_endpoints(_x_tell(), BOX, _Acq())
        torch.manual_seed(23)
        b = ref.ranked_line_endpoints(_x_tell(), BOX, _Acq())
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert lb.n_fallback == ref.n_fallback == 1


class _StubTimers:
    """Same contract as src.lanes.instrument.Timers (LZ 429-449 logic), used only while T1's
    module does not exist."""

    def __init__(self):
        from collections import defaultdict
        self.comp = defaultdict(float)
        self._stack = []

    @contextmanager
    def timed(self, key):
        import time
        self._stack.append([0.0])
        t = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t
            child = self._stack.pop()[0]
            if self._stack:
                self._stack[-1][0] += dt
            self.comp[key] += max(dt - child, 0.0)
            self.comp[key + "_n"] += 1
