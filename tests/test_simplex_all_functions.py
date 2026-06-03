import math


def test_simplex_module_all_functions_smoke(torch):
    from src.utils import simplex as sx

    device = torch.device("cuda")
    dtype = torch.float64

    # subset_sums_and_signs
    caps = torch.tensor([0.2, 0.5, 0.7], device=device, dtype=dtype)
    ss, si = sx.subset_sums_and_signs(caps)
    assert ss.shape[0] == 2 ** caps.numel()
    assert si.shape == ss.shape

    # polytope_volume (basic finite output)
    S = torch.tensor([0.3, 0.9], device=device, dtype=dtype)
    vol = sx.polytope_volume(S, ss, si, power=2, denom=2)
    assert vol.shape == (2,)
    assert torch.all(vol >= 0)

    # newton_in_bracket (returns within [tl, th])
    n = 5
    m = 2
    sr = torch.full((n,), 0.9, device=device, dtype=dtype)
    tl = torch.zeros((n,), device=device, dtype=dtype)
    th = torch.full((n,), 0.5, device=device, dtype=dtype)
    vol_at_tl = sx.polytope_volume(sr - tl, ss, si, power=m, denom=math.factorial(m))
    target_mass = torch.zeros((n,), device=device, dtype=dtype)
    u = torch.rand((n,), device=device, dtype=dtype)
    t = sx.newton_in_bracket(
        sr=sr,
        tl=tl,
        th=th,
        ss=ss,
        si=si,
        m=m,
        denom_int=math.factorial(m),
        denom_vol=math.factorial(m - 1),
        vol_at_tl=vol_at_tl,
        target_mass=target_mass,
        u=u,
        xtol=1e-10,
        max_iter=5,
    )
    assert torch.all(t >= tl - 1e-12)
    assert torch.all(t <= th + 1e-12)

    # full_simplex_ellipsoid + sample_ellipsoid
    ell = sx.full_simplex_ellipsoid(4, device, dtype, ellipsoid_init_radius=0.5)
    S2 = sx.sample_ellipsoid(32, ell, scale=1.0)
    assert S2.shape == (32, 4)
    assert torch.allclose(S2.sum(dim=1), torch.ones(32, device=device, dtype=dtype), atol=1e-5, rtol=0.0)

    # random_simplex: bounded simplex samples sum to S
    a = torch.zeros(4, device=device, dtype=dtype)
    b = torch.ones(4, device=device, dtype=dtype)
    X = sx.random_simplex(128, a, b, S=1.0, device="cuda", torch_dtype=dtype, seed=0)
    assert X.shape == (128, 4)
    assert torch.allclose(X.sum(dim=1), torch.ones(128, device=device, dtype=dtype), atol=1e-6, rtol=0.0)
    assert torch.all(X >= -1e-12)

    # proj_simplex: projects arbitrary points to simplex
    X_raw = torch.randn(10, 4, device=device, dtype=dtype)
    Xp = sx.proj_simplex(X_raw)
    assert Xp.shape == X_raw.shape
    assert torch.allclose(Xp.sum(dim=1), torch.ones(10, device=device, dtype=dtype), atol=1e-8, rtol=0.0)
    assert torch.all(Xp >= -1e-12)

    # random_simplex_direction / alias random_zero_sum_directions
    dirs = sx.random_simplex_direction(64, 5, device="cuda", dtype=dtype, seed=0)
    assert dirs.shape == (64, 5)
    assert torch.allclose(dirs.sum(dim=1), torch.zeros(64, device=device, dtype=dtype), atol=1e-8, rtol=0.0)
    norms = dirs.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6, rtol=0.0)

    dirs2 = sx.random_zero_sum_directions(8, 5, device="cuda")
    assert dirs2.shape == (8, 5)

    # is_on_simplex
    ok = sx.is_on_simplex(Xp)
    assert ok.shape == (10,)
    assert ok.all().item() is True

    # simplex_distance for each metric
    d_e = sx.simplex_distance(Xp[:2], Xp[2:4], metric="euclidean")
    assert d_e.shape == (2, 2)
    d_a = sx.simplex_distance(Xp[:2], Xp[2:4], metric="aitchison")
    assert d_a.shape == (2, 2)
    d_kl = sx.simplex_distance(Xp[:2], Xp[2:4], metric="kl")
    assert d_kl.shape == (2, 2)

    # composition_to_ilr and inverse
    comp = Xp[:5].clamp(min=1e-8)
    ilr = sx.composition_to_ilr(comp)
    assert ilr.shape == (5, comp.shape[1] - 1)
    comp2 = sx.ilr_to_composition(ilr, d=comp.shape[1])
    assert comp2.shape == comp.shape
    assert torch.allclose(comp2.sum(dim=1), torch.ones(5, device=device, dtype=dtype), atol=1e-6, rtol=0.0)

