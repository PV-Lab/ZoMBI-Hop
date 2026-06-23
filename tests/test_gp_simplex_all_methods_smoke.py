def test_gp_simplex_all_methods_smoke(torch, monkeypatch):
    """
    Exercise every method in src/utils/gp_simplex.py at least once.
    Uses tiny synthetic data and monkeypatches botorch fit to keep runtime low.
    """
    import math

    from src.utils.gp_simplex import GPSimplex, RepulsiveAcquisition
    from src.utils.simplex import proj_simplex

    device = torch.device("cuda")
    dtype = torch.float64

    # Monkeypatch fit_gpytorch_mll to avoid slow optimization in unit tests
    import botorch.fit

    monkeypatch.setattr(botorch.fit, "fit_gpytorch_mll", lambda mll, **kwargs: mll)

    class DH:
        def __init__(self):
            _b = torch.zeros(2, 3, device=device, dtype=dtype)
            _b[1] = 1.0
            self.bounds = _b
            self.repulsion_lambda = None
            self.acquisition_type = "ucb"
            self.ucb_beta = 0.1
            self.n_restarts = 4
            self.raw = 32
            self.nat_grad_step = 0.05
            self.nat_grad_max_steps = 5
            self.needles = torch.empty((0, 3), device=device, dtype=dtype)
            self.needle_penalty_radii = torch.empty((0, 1), device=device, dtype=dtype)
            self.needle_M_list = []
            self.needle_B = None

            self.X_all_actual = torch.tensor(
                [[0.2, 0.3, 0.5], [0.25, 0.25, 0.5], [0.1, 0.2, 0.7]],
                device=device,
                dtype=dtype,
            )
            self.Y_all = torch.tensor([[1.0], [0.9], [0.8]], device=device, dtype=dtype)

        def get_gp_data(self):
            return self.X_all_actual, self.Y_all

        def get_needles_and_penalty_radii(self):
            return self.needles, self.needle_penalty_radii

        def get_needle_ellipsoids(self):
            return self.needle_M_list, self.needle_B

        def get_penalty_mask(self, X):
            return torch.ones(X.shape[0], device=X.device, dtype=torch.bool)

        def get_input_noise(self):
            return 1e-3

        def update_gp_noise(self, sigma_y: float) -> None:
            pass

    dh = DH()
    gp = GPSimplex(
        data_handler=dh,
        proj_fn=proj_simplex,
        num_restarts=dh.n_restarts,
        raw_samples=dh.raw,
        repulsion_lambda=None,
        acquisition_type="ucb",
        ucb_beta=0.1,
        nat_grad_step=dh.nat_grad_step,
        nat_grad_max_steps=dh.nat_grad_max_steps,
        device="cuda",
        dtype=dtype,
    )

    # fit, fit_from_data_handler
    X, Y = dh.get_gp_data()
    gp.fit(X, Y)
    gp.fit_from_data_handler()

    # predict
    mu, var = gp.predict(X[:2])
    assert mu.shape[0] == 2
    assert var.shape[0] == 2

    # noise getters
    _ = gp.get_output_noise()

    # PI / LogEI at a point
    x0 = X[0]
    _ = gp.probability_of_improvement(x0, best_f=float(Y.max().item()))
    _ = gp.compute_log_ei_at_point(x0, best_f=float(Y.max().item()))

    # create_acquisition + computed lambda
    acq = gp.create_acquisition(best_f=float(Y.max().item()))
    assert gp.get_last_computed_lambda() is not None

    # RepulsiveAcquisition init/forward (directly)
    base = acq.base  # underlying base acquisition
    rep = RepulsiveAcquisition(
        base=base,
        proj_fn=proj_simplex,
        needles=dh.needles,
        penalty_radii=dh.needle_penalty_radii,
        repulsion_lambda=100.0,
        needle_M_list=[],
        needle_B=None,
    )
    v = rep(X[:2].unsqueeze(1))
    assert v.shape[0] == 2

    # _compute_repulsion_lambda (already hit, but call explicitly)
    _ = gp._compute_repulsion_lambda(base, n_samples=16)
    _ = gp.get_last_computed_lambda()

    # _sample_random (renamed from _sample_ellipsoid)
    S = gp._sample_random(16, dh.bounds)
    assert S.shape == (16, 3)

    # _get_tangent_basis and determine_penalty_ellipsoid
    B = gp._get_tangent_basis(3)
    assert B.shape == (3, 2)
    M, B2 = gp.determine_penalty_ellipsoid(X[0], drop_fraction=0.25, eigenvalue_floor=1e-6)
    assert M.shape == (2, 2)
    assert B2 is not None and B2.shape == (3, 2)

    # get_candidate exercises _optimize_acquisition internally
    cand = gp.get_candidate(bounds=dh.bounds, best_f=float(Y.max().item()), max_attempts=1)
    assert cand is not None
    assert cand.shape == (3,)
    assert torch.allclose(cand.sum(), torch.tensor(1.0, device=device, dtype=dtype), atol=1e-6, rtol=0.0)

