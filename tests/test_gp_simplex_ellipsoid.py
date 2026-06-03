import math


def test_determine_penalty_ellipsoid_ilr_quadratic(torch):
    """
    Synthetic acquisition that is a quadratic in ILR space:
        alpha(x) = c - 0.5 * ||ilr(x) - ilr(x_star)||^2

    The ILR-space Hessian is -I, so neg_H = I and M = I / (2*Delta).
    With drop_fraction=0.25, c=2.0: Delta = 0.25*2.0 = 0.5, so M = I.
    Semi-axes = 1.0. We pass max_radius=2.0 so no capping occurs.
    B must be None (ILR mode).
    """
    from src.utils.gp_simplex import GPSimplex
    from src.utils.simplex import composition_to_ilr

    device = torch.device("cuda")
    dtype = torch.float64

    class _DH:
        def get_input_noise(self):
            return 1e-3

        def get_needles_and_penalty_radii(self):
            return torch.empty((0, 3), device=device, dtype=dtype), torch.empty((0, 1), device=device, dtype=dtype)

        def get_needle_ellipsoids(self):
            return [], None

        bounds = torch.stack([torch.zeros(3, device=device, dtype=dtype),
                              torch.ones(3, device=device, dtype=dtype)], dim=0)

    dh = _DH()
    gp = GPSimplex(data_handler=dh, device=str(device), dtype=dtype)

    d = 3
    x_star = torch.tensor([0.2, 0.3, 0.5], dtype=dtype, device=device)
    z_star = composition_to_ilr(x_star.unsqueeze(0))  # (1, d-1)
    c = torch.tensor(2.0, dtype=dtype, device=device)

    def acq_fn(Xq):
        X = Xq.reshape(-1, d)
        Z = composition_to_ilr(X)          # (N, d-1)
        delta = Z - z_star                 # (N, d-1)
        val = c - 0.5 * (delta * delta).sum(dim=-1)
        return val.view(*Xq.shape[:-1])

    gp.acq_fn = acq_fn
    # max_radius=2.0 so the unit semi-axes (= 1.0) don't hit the cap.
    M, B2 = gp.determine_penalty_ellipsoid(x_star, drop_fraction=0.25, eigenvalue_floor=1e-12, max_radius=2.0)

    # ILR mode: B must be None
    assert B2 is None
    assert M.shape == (d - 1, d - 1)

    # PSD and symmetric
    assert torch.allclose(M, M.T, atol=1e-10, rtol=0.0)
    eig = torch.linalg.eigvalsh(M)
    assert torch.all(eig >= -1e-10)

    # delta_z = 0 at x_star -> quad = 0 <= 1 (inside)
    z_delta_zero = torch.zeros(d - 1, dtype=dtype, device=device)
    quad0 = (z_delta_zero @ M @ z_delta_zero).item()
    assert quad0 <= 1.0 + 1e-12

    # Delta = 0.5, neg_H = I -> M = I / (2*0.5) = I. semi-axes = 1.0 < max_radius=2.0, no cap.
    I = torch.eye(d - 1, dtype=dtype, device=device)
    assert torch.allclose(M, I, atol=1e-4, rtol=1e-4)


def test_eigenvalue_floor_prevents_singular_M(torch):
    """
    ILR-space acquisition that is nearly flat in one direction.
    The eigenvalue floor should prevent M from having a near-zero eigenvalue.
    """
    from src.utils.gp_simplex import GPSimplex
    from src.utils.simplex import composition_to_ilr

    device = torch.device("cuda")
    dtype = torch.float64

    class _DH:
        def get_input_noise(self):
            return 1e-3

        def get_needles_and_penalty_radii(self):
            return torch.empty((0, 4), device=device, dtype=dtype), torch.empty((0, 1), device=device, dtype=dtype)

        def get_needle_ellipsoids(self):
            return [], None

        bounds = torch.stack([torch.zeros(4, device=device, dtype=dtype),
                              torch.ones(4, device=device, dtype=dtype)], dim=0)

    gp = GPSimplex(data_handler=_DH(), device=str(device), dtype=dtype)

    d = 4
    x_star = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=dtype, device=device)
    z_star = composition_to_ilr(x_star.unsqueeze(0))  # (1, d-1)

    # Nearly flat in ILR direction 1: alpha(x) = 1 - 0.5*(z0^2 + eps*z1^2 + z2^2)
    eps = 1e-16

    def acq_fn(Xq):
        X = Xq.reshape(-1, d)
        Z = composition_to_ilr(X) - z_star   # (N, d-1)
        q = Z[:, 0] ** 2 + eps * Z[:, 1] ** 2 + Z[:, 2] ** 2
        val = 1.0 - 0.5 * q
        return val.view(*Xq.shape[:-1])

    gp.acq_fn = acq_fn
    M, _ = gp.determine_penalty_ellipsoid(x_star, drop_fraction=0.25, eigenvalue_floor=1e-6)

    # Smallest eigenvalue should be floored (not ~0)
    mineig = torch.linalg.eigvalsh(M).min().item()
    assert mineig > 0.0
