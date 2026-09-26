"""ZoMBI-Hop on axis-aligned boxes (``BoxDomain``), end to end and per method.

The simplex path is pinned bit-for-bit by ``test_domain_golden.py``; these tests
check the box path is genuinely box-shaped — nothing sums to 1, points fill the
cube, ellipsoids are full-dimensional — and that the optimiser still finds optima.
"""

import pytest

pytestmark = pytest.mark.cpu

PEAKS_UNIT = [[0.25, 0.70], [0.75, 0.30]]


@pytest.fixture()
def _cpu(torch):
    import src.core.zombihop  # noqa: F401 — import side effect sets cuda default
    torch.set_default_device("cpu")
    yield
    if torch.cuda.is_available():
        torch.set_default_device("cuda")


def _bumps(X, peaks, width):
    import torch
    P = torch.as_tensor(peaks, dtype=X.dtype, device=X.device)
    d2 = ((X.unsqueeze(1) - P.unsqueeze(0)) ** 2).sum(dim=-1)
    return torch.exp(-d2 / (2 * width ** 2)).max(dim=1).values


def _run_box_campaign(checkpoint_dir, f_unit, d=2, seed=7, max_activations=3):
    """Seeded box campaign in the unit cube; ``f_unit`` maps (n, d) unit coords → (n,)."""
    import torch
    from src.core.zombihop import ZoMBIHop
    from src.core.linebo import LineBO
    from src.utils.domain import BoxDomain

    torch.manual_seed(seed)
    dtype = torch.float64
    domain = BoxDomain()
    pts_per_line = 8

    def sim_obj(endpoints):
        left, right = endpoints[0, 0], endpoints[0, 1]
        t = torch.linspace(0.0, 1.0, pts_per_line, dtype=dtype)
        X_req = left.unsqueeze(0) + t.unsqueeze(1) * (right - left).unsqueeze(0)
        X_act = domain.project(X_req + 0.01 * torch.randn_like(X_req))
        return X_act, f_unit(X_act) + 0.01 * torch.randn(pts_per_line, dtype=dtype)

    linebo = LineBO(sim_obj, d, num_points_per_line=20, num_lines=10,
                    device="cpu", domain=domain)

    X_init = torch.rand(24, d, dtype=dtype)
    Y_init = (f_unit(X_init) + 0.01 * torch.randn(24, dtype=dtype)).unsqueeze(1)
    z = ZoMBIHop(
        objective=lambda x, b, acq: linebo.sampler(x, b, acq),
        X_init_actual=X_init.clone(), X_init_expected=X_init.clone(), Y_init=Y_init,
        device="cpu", dtype=dtype, domain=domain,
        max_zooms=3, max_iterations=4, n_restarts=4, raw=64, nat_grad_max_steps=10,
        n_consecutive_converged=1, min_zoom_for_needle=1, min_iters_per_zoom=1,
        needle_min_repeats=2, max_lines_per_activation=10, input_noise=0.02,
        resume=False, checkpoint_dir=str(checkpoint_dir), verbose=False,
    )
    z.run(max_activations=max_activations)
    return z


# ─── Domain methods ────────────────────────────────────────────────────────────

def test_box_domain_methods(torch, _cpu):
    from src.utils.domain import BoxDomain, make_domain, SimplexDomain

    dt = torch.float64
    dom = BoxDomain([-1.0, 0.0, 2.0], [1.0, 5.0, 3.0])
    lo, hi = torch.tensor([-1.0, 0.0, 2.0], dtype=dt), torch.tensor([1.0, 5.0, 3.0], dtype=dt)

    X = torch.tensor([[-3.0, 2.0, 9.0], [0.5, -1.0, 2.5]], dtype=dt)
    assert torch.equal(dom.project(X), torch.tensor([[-1.0, 2.0, 3.0], [0.5, 0.0, 2.5]], dtype=dt))
    assert dom.project(X.unsqueeze(1)).shape == (2, 1, 3)  # (n, q, d) acquisition batches

    a, b = torch.tensor([0.0, 1.0, 2.2], dtype=dt), torch.tensor([0.5, 4.0, 2.4], dtype=dt)
    S = dom.sample(2000, a, b, device="cpu", torch_dtype=dt)
    assert S.shape == (2000, 3) and bool(((S >= a) & (S <= b)).all())
    assert not torch.equal(S, dom.sample(2000, a, b, device="cpu", torch_dtype=dt))
    assert torch.equal(dom.sample(5, a, b, device="cpu", torch_dtype=dt, seed=3),
                       dom.sample(5, a, b, device="cpu", torch_dtype=dt, seed=3))

    D = dom.directions(500, 3, device="cpu", dtype=dt)
    assert torch.allclose(D.norm(dim=1), torch.ones(500, dtype=dt))
    assert D.sum(dim=1).abs().max() > 0.5  # not confined to the zero-sum plane

    assert torch.equal(dom.tangent_basis(3, "cpu", dt), torch.eye(3, dtype=dt))
    assert torch.equal(dom.default_bounds(3, "cpu", dt), torch.stack([lo, hi]))

    x = torch.tensor([[0.9, 4.9, 2.9]], dtype=dt)
    x_new, ok = dom.ascent_step(x, torch.full((1, 3), 1e6, dtype=dt), 0.02, lo, hi)
    assert bool(ok.all()) and torch.equal(x_new, hi.unsqueeze(0))  # clipped to the box
    x_new, _ = dom.ascent_step(torch.zeros(1, 3, dtype=dt), torch.full((1, 3), 1e6, dtype=dt),
                               0.02, -torch.ones(3, dtype=dt), torch.ones(3, dtype=dt))
    assert torch.allclose(x_new, torch.full((1, 3), 0.2, dtype=dt))  # max_step_frac cap

    unit = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=dt)
    half = torch.tensor([[0.0, 0.0], [0.5, 1.0]], dtype=dt)
    assert BoxDomain().region_jaccard(unit, half) == pytest.approx(0.5)
    assert BoxDomain().region_jaccard(unit, unit) == pytest.approx(1.0)

    assert BoxDomain().boundary_repulsion is False
    pen = BoxDomain(boundary_repulsion=True).boundary_penalty(
        torch.tensor([[0.5, 0.5], [0.01, 0.5]], dtype=dt), 0.05)
    assert pen[1] > 10 * pen[0]  # near-face point is repelled far more
    with pytest.raises(AssertionError):
        BoxDomain().check_point(torch.tensor([0.5, 1.2], dtype=dt))

    assert isinstance(make_domain(None), SimplexDomain)
    assert isinstance(make_domain("box"), BoxDomain)
    with pytest.raises(ValueError):
        make_domain("sphere")


def test_unit_box_scaler_roundtrip(torch):
    from src.utils.domain import UnitBoxScaler

    sc = UnitBoxScaler([-5.0, 0.0], [10.0, 15.0])
    X = torch.tensor([[-5.0, 0.0], [10.0, 15.0], [2.5, 7.5]], dtype=torch.float64)
    U = sc.to_unit(X)
    assert torch.allclose(U, torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=torch.float64))
    assert torch.allclose(sc.from_unit(U), X)


def test_box_candidate_search_leaves_the_simplex(torch, _cpu):
    """The acquisition ascent must climb to a box optimum whose coordinates do not
    sum to 1 — the simplex step would renormalise it away."""
    from src.utils.datahandler import DataHandler
    from src.utils.gp_simplex import GPSimplex
    from src.utils.domain import BoxDomain

    torch.manual_seed(0)
    dt = torch.float64
    target = torch.tensor([0.8, 0.85], dtype=dt)
    X = torch.rand(60, 2, dtype=dt)
    Y = (-((X - target) ** 2).sum(dim=1)).unsqueeze(1)

    dh = DataHandler(directory=None, d=2, device="cpu", dtype=dt, input_noise=0.01, verbose=False)
    dh.save_init(X, X.clone(), Y, torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=dt))
    dh.domain = BoxDomain()
    gp = GPSimplex(dh, num_restarts=8, raw_samples=128, ucb_beta=0.0,
                   nat_grad_step=0.02, nat_grad_max_steps=50,
                   device="cpu", dtype=dt, verbose=False, domain=BoxDomain())
    gp.fit(X, Y)
    cand = gp.get_candidate(dh.bounds)
    assert cand is not None
    assert torch.norm(cand - target).item() < 0.1, cand
    assert abs(cand.sum().item() - 1.0) > 0.4


# ─── End to end ────────────────────────────────────────────────────────────────

def test_box_campaign_finds_optima(torch, tmp_path, _cpu):
    z = _run_box_campaign(tmp_path / "runs", lambda X: _bumps(X, PEAKS_UNIT, 0.1))
    dh = z.data_handler
    X = dh.X_all_actual

    assert bool(((X >= 0.0) & (X <= 1.0)).all())
    assert X.sum(dim=1).std().item() > 0.1  # genuinely off the simplex
    assert z.full_bounds.tolist() == [[0.0, 0.0], [1.0, 1.0]]

    assert dh.needles is not None and dh.needles.shape[0] >= 1
    peaks = torch.tensor(PEAKS_UNIT, dtype=X.dtype)
    nearest = torch.cdist(dh.needles, peaks).min(dim=1).values
    assert nearest.min().item() < 0.08, (dh.needles, nearest)

    # Full-dimensional ellipsoids in ambient coordinates.
    assert torch.equal(dh.needle_B, torch.eye(2, dtype=X.dtype))
    assert all(m.shape == (2, 2) for m in dh.needle_M_list if m is not None)


def test_box_campaign_on_native_rectangle_via_scaler(torch, tmp_path, _cpu):
    """Benchmark workflow: the optimiser works in [0,1]^d; the landscape lives on
    [-5,10]×[0,15] and is evaluated through UnitBoxScaler."""
    from src.utils.domain import UnitBoxScaler

    sc = UnitBoxScaler([-5.0, 0.0], [10.0, 15.0])
    peaks_native = sc.from_unit(torch.tensor(PEAKS_UNIT, dtype=torch.float64))
    f_native = lambda X_nat: _bumps(X_nat, peaks_native, 1.5)  # noqa: E731
    z = _run_box_campaign(tmp_path / "runs", lambda U: f_native(sc.from_unit(U)), seed=11)

    needles_native = sc.from_unit(z.data_handler.needles)
    nearest = torch.cdist(needles_native, peaks_native).min(dim=1).values
    assert nearest.min().item() < 1.2, (needles_native, nearest)  # 0.08 of the 15-wide range


def test_resume_under_wrong_domain_is_refused(torch, tmp_path, _cpu):
    from src.core.zombihop import ZoMBIHop

    base = tmp_path / "runs"
    z = _run_box_campaign(base, lambda X: _bumps(X, PEAKS_UNIT, 0.1), max_activations=1)
    assert z.data_handler.needle_B is not None

    empty = torch.zeros(0, 2, dtype=torch.float64)
    kw = dict(objective=None, X_init_actual=empty, X_init_expected=empty,
              Y_init=torch.zeros(0, 1, dtype=torch.float64), device="cpu",
              dtype=torch.float64, run_uuid=z.run_uuid, checkpoint_dir=str(base),
              verbose=False)
    with pytest.raises(ValueError, match="resume the run with the domain"):
        ZoMBIHop(**kw)                      # default = simplex
    z2 = ZoMBIHop(**kw, domain="box")       # the matching domain resumes fine
    assert z2.data_handler.needles.shape[0] == z.data_handler.needles.shape[0]
