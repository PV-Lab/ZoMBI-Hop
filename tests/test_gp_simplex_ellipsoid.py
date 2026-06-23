import math


def test_determine_penalty_ellipsoid_tangent_quadratic(torch):
    """
    Synthetic acquisition quadratic in tangent space:
        alpha(x) = c - 0.5 * ||B^T (x - x_star)||^2

    The tangent-space Hessian is -I, so neg_H = I and M = I / (2*Delta).
    With drop_fraction=0.25, c=2.0: Delta = 0.25*2.0 = 0.5, so M = I.
    """
    from src.utils.gp_simplex import GPSimplex

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
    B = gp._get_tangent_basis(d)
    c = torch.tensor(2.0, dtype=dtype, device=device)

    def acq_fn(Xq):
        X = Xq.reshape(-1, d)
        u = (X - x_star.unsqueeze(0)) @ B
        val = c - 0.5 * (u * u).sum(dim=-1)
        return val.view(*Xq.shape[:-1])

    gp.acq_fn = acq_fn
    M, B2 = gp.determine_penalty_ellipsoid(x_star, drop_fraction=0.25, eigenvalue_floor=1e-12, max_radius=2.0)

    assert B2 is not None
    assert B2.shape == (d, d - 1)
    assert M.shape == (d - 1, d - 1)

    assert torch.allclose(M, M.T, atol=1e-10, rtol=0.0)
    eig = torch.linalg.eigvalsh(M)
    assert torch.all(eig >= -1e-10)

    u_zero = torch.zeros(d - 1, dtype=dtype, device=device)
    quad0 = (u_zero @ M @ u_zero).item()
    assert quad0 <= 1.0 + 1e-12

    I = torch.eye(d - 1, dtype=dtype, device=device)
    assert torch.allclose(M, I, atol=1e-4, rtol=1e-4)


def test_eigenvalue_floor_prevents_singular_M(torch):
    """Nearly flat tangent direction — eigenvalue floor should prevent singular M."""
    from src.utils.gp_simplex import GPSimplex

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
    B = gp._get_tangent_basis(d)
    eps = 1e-16

    def acq_fn(Xq):
        X = Xq.reshape(-1, d)
        u = (X - x_star.unsqueeze(0)) @ B
        q = u[:, 0] ** 2 + eps * u[:, 1] ** 2 + u[:, 2] ** 2
        val = 1.0 - 0.5 * q
        return val.view(*Xq.shape[:-1])

    gp.acq_fn = acq_fn
    M, _ = gp.determine_penalty_ellipsoid(x_star, drop_fraction=0.25, eigenvalue_floor=1e-6)

    mineig = torch.linalg.eigvalsh(M).min().item()
    assert mineig > 0.0
