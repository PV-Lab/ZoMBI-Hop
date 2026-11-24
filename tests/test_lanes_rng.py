"""src/lanes/rng.py: Fix 4 scoped to a lanes engine (DESIGN.md 1.4, N7; REVIEW F2).

* inside ``seeded_simplex()`` two unseeded ``random_simplex`` calls differ and ``torch.manual_seed(s)``
  reproduces both;
* exactly one global draw per unseeded call, ``torch.randint(0, 2**62, (1,))`` on the DEFAULT DEVICE's
  generator (CUDA once src.core.zombihop is imported, REVIEW F2): CUDA and CPU generator states are compared
  with a reference draw;
* an explicit ``seed=`` is passed through without any global draw;
* after the scope the three module attributes are the original objects; unseeded calls still differ,
  since random_simplex itself draws the seed the same way since 2026-11-24;
* nested install / uninstall ref-count; idempotent uninstall; no double wrap over a ``_zoomfix`` wrapper;
* the wrapper is token-identical to the sandbox's PF 412-415 (AST; skipped when the sandbox is absent).

Small n; fast. REVIEW F9 autouse fixture (plus a reset of the SeededSimplex counter).
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from src.core import linebo as _lb
from src.core import zombihop as _zh
from src.lanes import rng as R
from src.lanes.rng import SeededSimplex, seeded_simplex
from src.utils import domain as _dom

REPO = Path(__file__).resolve().parents[1]
PF_PATH = REPO / "benchmarks" / "zoom_fix" / "proposed_fixes.py"
MODS = (_dom, _zh, _lb)


@pytest.fixture(autouse=True)
def _restore_rng_and_simplex():
    """REVIEW F9: restore CPU / CUDA RNG states and the three random_simplex attributes."""
    saved = {m: m.__dict__.get("random_simplex") for m in MODS}
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        SeededSimplex._reset()
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        for m, f in saved.items():
            setattr(m, "random_simplex", f)


def _dev() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _draw(**kw):
    """One sample through the name the sampling code resolves (src.utils.domain.random_simplex)."""
    d = 4
    return _dom.random_simplex(16, torch.zeros(d, dtype=torch.float64), torch.ones(d, dtype=torch.float64),
                               S=1.0, device=_dev(), torch_dtype=torch.float64, **kw)


def _states():
    return torch.get_rng_state().clone(), [s.clone() for s in torch.cuda.get_rng_state_all()]


def _same_states(a, b) -> bool:
    return torch.equal(a[0], b[0]) and len(a[1]) == len(b[1]) and all(torch.equal(x, y) for x, y in zip(a[1], b[1]))


def test_unseeded_calls_differ_and_reproduce():
    with seeded_simplex():
        torch.manual_seed(5)
        a1, a2 = _draw(), _draw()
        torch.manual_seed(5)
        b1, b2 = _draw(), _draw()
    assert not torch.equal(a1, a2)
    assert torch.equal(a1, b1) and torch.equal(a2, b2)
    assert torch.allclose(a1.sum(1), torch.ones(16, dtype=torch.float64, device=a1.device))


def test_exactly_one_default_device_draw_per_call():
    with seeded_simplex():
        torch.manual_seed(11)
        s0 = _states()
        _draw()
        after_call = _states()
        torch.manual_seed(11)
        assert _same_states(_states(), s0)
        torch.randint(0, 2 ** 62, (1,))             # the PF 414 draw: default device, default generator
        after_ref = _states()
    assert _same_states(after_call, after_ref)
    assert not _same_states(after_call, s0)
    if torch.cuda.is_available():
        # REVIEW F2: with zombihop imported the default device is CUDA, so the CPU generator is untouched
        assert torch.empty(0).device.type == "cuda"
        assert torch.equal(after_call[0], s0[0])
        assert not all(torch.equal(x, y) for x, y in zip(after_call[1], s0[1]))


def test_explicit_seed_passes_through_without_draw():
    with seeded_simplex():
        s0 = _states()
        x1 = _draw(seed=123)
        x2 = _draw(seed=123)
        assert _same_states(_states(), s0)
    assert torch.equal(x1, x2)
    assert torch.equal(x1, _draw(seed=123))          # same samples as the unwrapped function with that seed


def test_scope_restores_originals_and_unseeded_behaviour():
    orig = {m: m.random_simplex for m in MODS}
    with seeded_simplex() as tok:
        assert tok.active
        for m in MODS:
            assert m.random_simplex is not orig[m]
            assert getattr(m.random_simplex, "_lanes_seeded", False)
        assert _dom.random_simplex is _zh.random_simplex is _lb.random_simplex
        assert SeededSimplex.installed()
    assert not tok.active
    for m in MODS:
        assert m.random_simplex is orig[m]
    assert not SeededSimplex.installed()
    # random_simplex draws an unseeded call's seed itself since 2026-11-24 (same draw as the wrapper)
    assert not torch.equal(_draw(), _draw())


def test_nested_install_refcount_and_idempotent_uninstall():
    orig = _dom.random_simplex
    n0 = SeededSimplex.active_count()
    t1 = SeededSimplex.install()
    w = _dom.random_simplex
    t2 = SeededSimplex.install()
    assert _dom.random_simplex is w                  # no re-wrap
    assert SeededSimplex.active_count() == n0 + 2
    t1.uninstall()
    t1.uninstall()                                   # idempotent: counts once
    assert SeededSimplex.active_count() == n0 + 1
    assert _dom.random_simplex is w
    t2.uninstall()
    assert SeededSimplex.active_count() == n0
    assert _dom.random_simplex is orig and _zh.random_simplex is orig and _lb.random_simplex is orig


def test_no_double_wrap_over_zoomfix_wrapper():
    orig = _dom.random_simplex

    def fake(*args, seed=None, **kwargs):            # stands for the sandbox PF wrapper
        return orig(*args, seed=seed, **kwargs)
    fake._zoomfix = True
    for m in MODS:
        m.random_simplex = fake
    tok = SeededSimplex.install()
    assert all(m.random_simplex is fake for m in MODS)
    assert SeededSimplex.installed()
    tok.uninstall()
    assert all(m.random_simplex is fake for m in MODS)   # nothing of ours to restore: PF's patch stays


def test_self_heals_when_attributes_were_restored_behind_its_back():
    orig = _dom.random_simplex
    t1 = SeededSimplex.install()
    for m in MODS:                                   # e.g. a test fixture restored them
        m.random_simplex = orig
    t2 = SeededSimplex.install()
    assert getattr(_dom.random_simplex, "_lanes_seeded", False)
    t1.uninstall()
    t2.uninstall()
    assert _dom.random_simplex is orig


def test_uninstall_leaves_a_foreign_replacement_alone():
    orig = _dom.random_simplex
    tok = SeededSimplex.install()

    def other(*a, **k):
        return orig(*a, **k)
    _zh.random_simplex = other
    tok.uninstall()
    assert _dom.random_simplex is orig and _lb.random_simplex is orig
    assert _zh.random_simplex is other


def test_context_manager_uninstalls_on_error():
    orig = _dom.random_simplex
    with pytest.raises(RuntimeError):
        with seeded_simplex():
            raise RuntimeError("boom")
    assert _dom.random_simplex is orig and SeededSimplex.active_count() == 0
    with SeededSimplex.install():
        assert SeededSimplex.installed()
    assert _dom.random_simplex is orig


# ── token-identical wrapper (REVIEW F2) ──────────────────────────────────────────────────────────

def _inner_random_simplex(tree: ast.AST, outer: str) -> str:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == outer:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "random_simplex":
                    return ast.dump(sub)
    raise AssertionError(f"no random_simplex inside {outer}")


@pytest.mark.skipif(not PF_PATH.exists(), reason="sandbox proposed_fixes.py not present")
def test_wrapper_is_token_identical_to_pf():
    pf = _inner_random_simplex(ast.parse(PF_PATH.read_text(encoding="utf-8")), "fix_rng")
    ours = _inner_random_simplex(ast.parse(Path(R.__file__).read_text(encoding="utf-8")), "_wrapper")
    assert ours == pf
