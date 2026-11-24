"""src/lanes/geometry.py: the 8 sandbox geometry unit tests (lane_selftest.py ST 60-137) as pytest tests,
plus small checks of the remaining helpers and an AST identity check against the sandbox source
(benchmarks/zoom_fix/lanes/lane_zombihop.py LZ 48-175; skipped when the sandbox is absent).

Pure tensors; fast. REVIEW F9: an autouse fixture restores the CPU / CUDA RNG states and the three
``random_simplex`` attributes after each test; nothing is written to disk.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from src.lanes import geometry as G

REPO = Path(__file__).resolve().parents[1]
LZ_PATH = REPO / "benchmarks" / "zoom_fix" / "lanes" / "lane_zombihop.py"
DEV, DT = G.DEV, G.DT


@pytest.fixture(autouse=True)
def _restore_rng_and_simplex():
    """REVIEW F9: restore CPU / CUDA RNG states and the three random_simplex attributes."""
    from src.core import linebo as _lb
    from src.core import zombihop as _zh
    from src.utils import domain as _dom
    saved = {m: m.__dict__.get("random_simplex") for m in (_dom, _zh, _lb)}
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        for m, f in saved.items():
            setattr(m, "random_simplex", f)


@pytest.fixture()
def data():
    """ST 63-69 / 81-82: the 6-D parent box, 8 Dirichlet rows, 8 nearly identical rows (seeded).

    ST clamps axis 3 to 0.3 and renormalises, which can push a row back above the parent's 0.3 on that
    axis (ST passed on its unseeded draws by luck); here the 8 rows are taken from draws that lie inside
    the parent, which is what the test means."""
    torch.manual_seed(0)
    parent = torch.tensor([[0.0] * 6, [1, 1, 1, 0.3, 1, 1]], device=DEV, dtype=DT)
    pts = torch.distributions.Dirichlet(torch.ones(6, dtype=DT, device=DEV)).sample((64,)).to(DEV)
    pts = pts[pts[:, 3] <= 0.29][:8]
    assert pts.shape[0] == 8 and bool(G.in_box(pts, parent).all())
    tight = pts[0:1].repeat(8, 1) + 1e-4 * torch.randn(8, 6, device=DEV, dtype=DT)
    tight = (tight.clamp(min=0) / tight.clamp(min=0).sum(1, keepdim=True))
    return parent, pts, tight


# ── ST 60-137 ────────────────────────────────────────────────────────────────────────────────────

def test_spread_rows(data):
    parent, pts, _ = data
    box, info = G.nested_box(pts, pts[0], parent, 0.7, 0.064)
    assert box is not None
    G.check_nested(box, parent, 0.7, 0.064)
    assert bool(G.in_box(pts[0:1], box)[0]), "best row must stay inside the box"
    assert info["n_top"] == 8


def test_rho_cap_binds(data):
    parent, _, _ = data
    wide = torch.tensor([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0.7, 0.3, 0, 0],
                         [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], device=DEV, dtype=DT)
    b2, _ = G.nested_box(wide, wide[0], parent, 0.7, 0.064)
    w = (b2[1] - b2[0]) / (parent[1] - parent[0])
    assert float(w.max()) <= 0.7 + 1e-9, w
    G.check_nested(b2, parent, 0.7, 0.064)
    assert torch.allclose(G.width_ratios(b2, parent), w)


def test_floor_widening(data):
    parent, _, tight = data
    b3, i3 = G.nested_box(tight, tight[0], parent, 0.7, 0.064)
    w3 = b3[1] - b3[0]
    assert float(w3.min()) >= 0.064 - 1e-9 or bool(((w3 - (parent[1] - parent[0])).abs() < 1e-9).any())
    G.check_nested(b3, parent, 0.7, 0.064)
    assert i3["floored"] >= 1


def test_refuse_at_floor(data):
    parent, _, tight = data
    b3, _ = G.nested_box(tight, tight[0], parent, 0.7, 0.064)
    small = torch.stack([b3[0], b3[0] + 0.064])
    b4, i4 = G.nested_box(tight, tight[0], small, 0.7, 0.064)
    assert b4 is None and i4.get("refused") == "floor_blocks_zoom", i4


def test_floor_over_cap(data):
    parent, pts, _ = data
    par5 = parent.clone()
    par5[1, 0] = 0.08
    rows5 = pts.clone()
    rows5[:, 0] = torch.linspace(0.0, 0.08, 8, device=DEV, dtype=DT)
    b5, i5 = G.nested_box(rows5, rows5[0], par5, 0.7, 0.064)
    G.check_nested(b5, par5, 0.7, 0.064)
    assert i5["floor_over_cap"] >= 1


def test_check_nested_rejects(data):
    parent, _, _ = data
    for child in (parent.clone(), torch.stack([parent[0] - 0.01, parent[1]])):
        with pytest.raises(AssertionError):
            G.check_nested(child, parent, 0.7, 0.064)


def test_graded_floor(data):
    parent, _, tight = data
    f1, f2 = G.zoom_floor(0.064, 0.7, 1, 3, "graded"), G.zoom_floor(0.064, 0.7, 2, 3, "graded")
    assert abs(f2 - 0.064) < 1e-12 and abs(f1 - 0.064 / 0.7) < 1e-12
    assert G.zoom_floor(0.064, 0.7, 1, 3, "volume") == 0.064
    assert abs(G.zoom_floor(0.064, 0.7, 0, 3, "graded") - 0.064 / 0.49) < 1e-12
    b6, _ = G.nested_box(tight, tight[0], parent, 0.7, f1)
    G.check_nested(b6, parent, 0.7, f1)
    b7, i7 = G.nested_box(tight, tight[0], b6, 0.7, f2)
    c7 = G.check_nested(b7, b6, 0.7, f2)
    assert c7["over_rho"] == 0 and c7["eq_parent"] == 0, (c7, i7)


def test_inner_box(data):
    parent, pts, _ = data
    box, _ = G.nested_box(pts, pts[0], parent, 0.7, 0.064)
    ib = G.inner_box(box, 0.02, 0.25)
    assert float(ib[0].sum()) <= 1 + 1e-12 and float(ib[1].sum()) >= 1 - 1e-12
    assert bool((ib[0] >= box[0] - 1e-12).all() and (ib[1] <= box[1] + 1e-12).all())


# ── the other helpers ────────────────────────────────────────────────────────────────────────────

def test_in_box_and_overlap():
    box = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 1.0]], device=DEV, dtype=DT)
    X = torch.tensor([[0.2, 0.2, 0.6], [0.6, 0.1, 0.3], [0.5 + 1e-10, 0.0, 0.5]], device=DEV, dtype=DT)
    assert G.in_box(X, box).tolist() == [True, False, True]
    assert G.overlap_frac(box, box) == pytest.approx(1.0)
    half = torch.tensor([[0.25, 0.0, 0.0], [0.75, 0.5, 1.0]], device=DEV, dtype=DT)
    assert G.overlap_frac(box, half) == pytest.approx(0.5)
    far = torch.tensor([[0.6, 0.6, 0.0], [0.9, 0.9, 1.0]], device=DEV, dtype=DT)
    assert G.overlap_frac(box, far) == 0.0


def test_inner_box_keeps_simplex_feasibility():
    # a box whose shrunk version would leave the simplex: margin halved until sum(lo) <= 1 <= sum(hi)
    # (full margin 0.01 per side would give sum(hi) = 0.99 < 1; half the margin keeps 1.005)
    box = torch.tensor([[0.3, 0.3, 0.3], [0.34, 0.34, 0.34]], device=DEV, dtype=DT)
    ib = G.inner_box(box, 0.02, 0.25)
    assert float(ib[0].sum()) <= 1 + 1e-12 and float(ib[1].sum()) >= 1 - 1e-12
    assert bool((ib[0] >= box[0] - 1e-12).all() and (ib[1] <= box[1] + 1e-12).all())
    assert torch.allclose(ib, torch.stack([box[0] + 0.005, box[1] - 0.005]), atol=1e-12)


def test_x_requested_line_on_simplex():
    a = torch.tensor([0.6, 0.2, 0.1, 0.1], device=DEV, dtype=DT)
    b = torch.tensor([0.1, 0.3, 0.3, 0.3], device=DEV, dtype=DT)
    t = torch.linspace(0, 1, 12, device=DEV, dtype=DT)[:, None]
    pts = a + t * (b - a)
    xe = G.x_requested(pts)
    assert xe.shape == pts.shape
    assert torch.allclose(xe.sum(1), torch.ones(12, device=DEV, dtype=DT), atol=1e-9)
    assert bool((xe >= -1e-12).all())
    # every requested point lies on the line through the printed points (they are collinear)
    d = (b - a) / (b - a).norm()
    r = (xe - a) - ((xe - a) @ d)[:, None] * d[None]
    assert float(r.norm(dim=1).max()) < 1e-9
    one = pts[:1]
    out = G.x_requested(one)
    assert torch.equal(out, one) and out is not one


def test_lane_constants():
    from src.lanes import labels
    assert G.LANES == ("front", "back")
    assert G.LANE_CODE == {"front": 0, "back": 1}
    assert G.LANE_TAG == labels.LANE_TAG == {"front": "F", "back": "B"}
    assert G.COUNTED == ("own_current", "own_current_bridge")
    assert G.DT is torch.float64


# ── verbatim move (decision identity) ────────────────────────────────────────────────────────────

def _top_level(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out[node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = ast.dump(node)
    return out


@pytest.mark.skipif(not LZ_PATH.exists(), reason="sandbox lane_zombihop.py not present")
def test_geometry_is_the_sandbox_code():
    sand, prod = _top_level(LZ_PATH), _top_level(Path(G.__file__))
    names = ("DEV", "DT", "LANES", "LANE_CODE", "LANE_TAG", "COUNTED", "in_box", "zoom_floor", "width_ratios",
             "nested_box", "check_nested", "inner_box", "overlap_frac", "x_requested")
    differ = [n for n in names if sand.get(n) != prod.get(n)]
    assert not differ, f"not token-identical to lane_zombihop.py: {differ}"
