"""Golden-master test: a seeded end-to-end simplex campaign must reproduce bit-for-bit.

Guards the search-domain abstraction (``src/utils/domain.py``): routing the
simplex geometry through ``SimplexDomain`` must not change a single number the
optimiser produces. The campaign is small but real — LineBO line selection, GP
fits, natural-gradient candidate search, zooming, needle declaration with the
tangent-space Hessian ellipsoid, and the penalty mask all run un-mocked on CPU.

The golden file was recorded from the pre-abstraction code. Regenerate it ONLY
when a simplex behaviour change is intended:

    .venv/bin/python tests/test_domain_golden.py --regenerate

Exact equality is only meaningful on the same torch build and CPU BLAS the file
was recorded with (torch 2.5.1, this cluster); on a different stack the test
may fail without any code change — regenerate from a known-good commit there.
"""

import sys
from pathlib import Path

import pytest

GOLDEN_PATH = Path(__file__).parent / "fixtures" / "simplex_golden.pt"

pytestmark = pytest.mark.cpu


def _two_bump_simplex(X):
    import torch
    peaks = torch.tensor([[0.70, 0.20, 0.10],
                          [0.10, 0.20, 0.70]], dtype=X.dtype, device=X.device)
    d2 = ((X.unsqueeze(1) - peaks.unsqueeze(0)) ** 2).sum(dim=-1)  # (n, 2)
    return torch.exp(-d2 / (2 * 0.12 ** 2)).max(dim=1).values


def run_golden_campaign(checkpoint_dir):
    """Seeded d=3 simplex campaign. Returns a dict of every tensor worth pinning."""
    import torch
    from src.core.zombihop import ZoMBIHop
    from src.core.linebo import LineBO
    from src.utils.simplex import proj_simplex

    torch.set_default_device("cpu")
    torch.manual_seed(1234)
    dtype = torch.float64
    d = 3
    pts_per_line = 8

    def sim_obj(endpoints):
        left, right = endpoints[0, 0], endpoints[0, 1]
        t = torch.linspace(0.0, 1.0, pts_per_line, dtype=dtype)
        X_req = left.unsqueeze(0) + t.unsqueeze(1) * (right - left).unsqueeze(0)
        X_act = proj_simplex(X_req + 0.01 * torch.randn_like(X_req))
        Y = _two_bump_simplex(X_act) + 0.01 * torch.randn(pts_per_line, dtype=dtype)
        return X_act, Y

    linebo = LineBO(sim_obj, d, num_points_per_line=20, num_lines=10, device="cpu")

    def objective(x_tell, bounds, acq_fn):
        return linebo.sampler(x_tell, bounds, acq_fn)

    X_init = torch.rand(24, d, dtype=dtype)
    X_init = X_init / X_init.sum(dim=1, keepdim=True)
    Y_init = (_two_bump_simplex(X_init) + 0.01 * torch.randn(24, dtype=dtype)).unsqueeze(1)

    z = ZoMBIHop(
        objective=objective,
        X_init_actual=X_init.clone(),
        X_init_expected=X_init.clone(),
        Y_init=Y_init,
        device="cpu",
        dtype=dtype,
        max_zooms=3,
        max_iterations=4,
        n_restarts=4,
        raw=64,
        nat_grad_max_steps=10,
        n_consecutive_converged=1,
        min_zoom_for_needle=1,
        min_iters_per_zoom=1,
        needle_min_repeats=2,
        max_lines_per_activation=10,
        input_noise=0.02,
        resume=False,
        checkpoint_dir=str(checkpoint_dir),
        verbose=False,
    )
    z.run(max_activations=3)

    dh = z.data_handler
    return {
        "X_all_actual": dh.X_all_actual.clone(),
        "X_all_expected": dh.X_all_expected.clone(),
        "Y_all": dh.Y_all.clone(),
        "needles": (dh.needles.clone() if dh.needles is not None
                    else torch.zeros(0, d, dtype=dtype)),
        "needle_M": [m.clone() if m is not None else None for m in dh.needle_M_list],
        "needle_B": dh.needle_B.clone() if dh.needle_B is not None else None,
        "exclusions": dh.exclusions.clone() if dh.exclusions is not None else None,
        "exclusion_M": [m.clone() if m is not None else None for m in dh.exclusion_M_list],
        "bounds": dh.bounds.clone(),
        "penalty_mask": dh.get_penalty_mask().clone(),
    }


def _assert_same(a, b, key):
    import torch
    if a is None or b is None:
        assert a is None and b is None, f"{key}: one side is None"
    elif isinstance(a, list):
        assert len(a) == len(b), f"{key}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{key}[{i}]")
    else:
        assert a.shape == b.shape, f"{key}: shape {tuple(a.shape)} != {tuple(b.shape)}"
        assert torch.equal(a, b), (
            f"{key}: values differ (max abs diff "
            f"{(a.double() - b.double()).abs().max().item():.3e})")


@pytest.fixture()
def _cpu_default_device(torch):
    import src.core.zombihop  # noqa: F401 — import side effect sets cuda default
    yield
    if torch.cuda.is_available():
        torch.set_default_device("cuda")


def test_simplex_campaign_matches_golden(torch, tmp_path, _cpu_default_device):
    if not GOLDEN_PATH.exists():
        pytest.fail(f"golden file missing: {GOLDEN_PATH} (run with --regenerate)")
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    got = run_golden_campaign(tmp_path / "runs")
    # Sanity: the campaign must actually exercise the ellipsoid path, or the
    # comparison proves little about the tangent-basis plumbing.
    assert got["needles"].shape[0] >= 1
    for key in golden:
        _assert_same(got[key], golden[key], key)


if __name__ == "__main__":
    import tempfile
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import torch
    if "--regenerate" not in sys.argv:
        sys.exit("usage: test_domain_golden.py --regenerate")
    with tempfile.TemporaryDirectory() as tmp:
        out = run_golden_campaign(Path(tmp) / "runs")
    GOLDEN_PATH.parent.mkdir(exist_ok=True)
    torch.save(out, GOLDEN_PATH)
    print(f"wrote {GOLDEN_PATH}: {out['X_all_actual'].shape[0]} points, "
          f"{out['needles'].shape[0]} needle(s)")
