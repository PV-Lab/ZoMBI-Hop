def test_snapshot_called_once_per_objective_and_after_objective(torch):
    """
    Verifies the refactor rule:
    - take_snapshot is called exactly once after each _objective_wrapper returns.
    We run a tiny mocked ZoMBIHop loop with 1 activation / 1 zoom / 1 iteration.
    """
    from src.core.zombihop import ZoMBIHop

    device = torch.device("cuda")
    dtype = torch.float64

    bounds0 = torch.zeros(2, 3, device=device, dtype=dtype)
    bounds0[1] = 1.0

    class FakeDH:
        def __init__(self):
            self.max_zooms = 1
            self.max_iterations = 1
            self.n_consecutive_converged = 10  # never declare needle
            self.raw = 10
            self.nat_grad_step = 0.02
            self.no_improvements = 0
            self.current_zoom_bounds = None
            self.bounds = bounds0.clone()
            self._snapshots = []
            self._obj_returned = 0
            self.X_all_actual = torch.tensor([[0.2, 0.3, 0.5]], dtype=dtype, device=device)
            self.Y_all = torch.tensor([[1.0]], dtype=dtype, device=device)
            self.run_uuid = "TEST"
            self.needles_results = []
            self.needle_vals = torch.empty((0, 1), dtype=dtype, device=device)
            self.needles = torch.empty((0, 3), dtype=dtype, device=device)

        def get_iteration_state(self):
            return 0, 0, 0, 0

        def get_gp_data(self):
            X = self.X_all_actual.clone()
            Y = self.Y_all.clone()
            return X, Y

        def get_best_unpenalized(self):
            return self.X_all_actual[0].clone(), self.Y_all[0].clone(), 0

        def get_penalty_mask(self, X=None):
            if X is None:
                return torch.ones(self.Y_all.shape[0], dtype=torch.bool, device=self.Y_all.device)
            return torch.ones(X.shape[0], dtype=torch.bool, device=X.device)

        def take_snapshot(self, label: str = "", permanent: bool = False, activation=None, zoom=None, iteration=None):
            # Ensure this happens only after objective returned
            assert self._obj_returned == 1, "snapshot called before objective returned"
            self._snapshots.append((label, permanent, activation, zoom, iteration))

        def determine_new_bounds(self):
            return self.bounds.clone()

        def add_needle(self, **kwargs):
            raise AssertionError("Should never add needle in this test")

        def get_all_points(self):
            # Matches DataHandler.get_all_points() return shape expectations
            return self.X_all_actual.clone(), self.X_all_actual.clone(), self.Y_all.clone()

        def get_needle_results(self):
            return []

        def get_needle_locations(self):
            return torch.empty((0, 3), dtype=dtype, device=device)

        def get_all_needle_results(self):
            return list(self.needles_results)

        def get_all_needle_locations(self):
            dd = self.X_all_actual.shape[1]
            if self.needles.ndim == 2 and self.needles.shape[0] > 0:
                return self.needles
            return torch.empty((0, dd), dtype=dtype, device=device)

        def get_all_needle_vals(self):
            if self.needle_vals.ndim == 2 and self.needle_vals.shape[0] > 0:
                return self.needle_vals
            return torch.empty((0, 1), dtype=dtype, device=device)

    class FakeGP:
        def __init__(self):
            self.acq_fn = None
            self.raw_samples = 10
            self.nat_grad_step = 0.02

        def fit(self, X, Y):
            return None

        def create_acquisition(self, best_f=None, penalty_value=None):
            # trivial acquisition
            def acq(Xq):
                return torch.zeros(Xq.shape[:-1], dtype=dtype, device=device)

            self.acq_fn = acq
            return acq

        def get_candidate(self, bounds, best_f=None):
            return torch.tensor([0.2, 0.3, 0.5], dtype=dtype, device=device)

        def determine_penalty_ellipsoid(self, needle, drop_fraction=0.25, eigenvalue_floor=1e-6):
            B = torch.linalg.qr(torch.eye(3, dtype=dtype) - (1.0 / 3.0))[0][:, :2]
            M = torch.eye(2, dtype=dtype)
            return M, B

    z = ZoMBIHop.__new__(ZoMBIHop)
    z.device = device
    z.dtype = dtype
    z.verbose = False
    z._needle_plot_points_ref = None
    z.bounds = bounds0.clone()
    z.ellipsoid_drop_fraction = 0.25
    z.ellipsoid_eigenvalue_floor = 1e-6
    z.needle_shrink_factor = 0.85
    z.needle_stop_noise_multiplier = 3.0
    z.random_sampler = lambda n, a, b, **kwargs: torch.rand((n, 3), dtype=dtype, device=device)
    z.random_direction_sampler = None
    z.proj_fn = lambda X: X
    z.data_handler = FakeDH()
    z.gp_handler = FakeGP()

    def _log(*args, **kwargs):
        return None

    z._log = _log
    z._log_status = lambda *args, **kwargs: None
    z._check_convergence_to_needle = lambda *args, **kwargs: (False, None, None)

    def _objective_wrapper(candidate, bounds, acq_fn):
        z.data_handler._obj_returned = 1
        X = torch.tensor([[0.2, 0.3, 0.5]], dtype=dtype, device=device)
        Y = torch.tensor([[1.0]], dtype=dtype, device=device)
        return X, Y

    z._objective_wrapper = _objective_wrapper

    z.run(max_activations=1, time_limit_hours=None)

    # Expect exactly one *non-permanent* snapshot from the objective receipt,
    # plus the always-taken final permanent snapshot.
    non_perm = [s for s in z.data_handler._snapshots if not s[1]]
    assert len(non_perm) == 1
    label, permanent, act, zoom, it = non_perm[0]
    assert "act0_z0_i0" in label


def test_successive_demotion_stops_when_needles_exhausted(torch):
    """
    Verifies the three-way failure dispatch in _handle_failure_retry:
      - First failure (first_failure_handled=False): gp_handler.recompute_all_ellipsoids called.
      - Second failure (first_failure_handled=True, data_added=False, needle_M_list empty):
        _all_needle_axes_below_min returns True → stop immediately, no shrink.
    """
    from src.core.zombihop import ZoMBIHop

    device = torch.device("cuda")
    dtype = torch.float64

    bounds0 = torch.zeros(2, 3, device=device, dtype=dtype)
    bounds0[1] = 1.0

    recompute_calls = []
    shrink_calls = []

    class FakeDH:
        def __init__(self):
            self.max_zooms = 1
            self.max_iterations = 1
            self.n_consecutive_converged = 2
            self.raw = 10
            self.nat_grad_step = 0.02
            self.no_improvements = 0
            self.current_zoom_bounds = None
            self.bounds = bounds0.clone()
            self.X_all_actual = torch.tensor([[0.2, 0.3, 0.5]], dtype=dtype, device=device)
            self.Y_all = torch.tensor([[1.0]], dtype=dtype, device=device)
            self.run_uuid = "TEST"
            self.needles_results = []
            self.needles = torch.tensor([[0.2, 0.3, 0.5]], dtype=dtype, device=device)
            self.needle_vals = torch.tensor([[1.0]], dtype=dtype, device=device)
            # Empty M list → _all_needle_axes_below_min returns True immediately
            self.needle_M_list = []
            self.input_noise_ilr = 0.03
            self.max_gp_points = 100

        def get_iteration_state(self):
            return 0, 0, 0, 0

        def get_gp_data(self):
            return self.X_all_actual, self.Y_all

        def get_zoom_gp_data(self, bounds):
            return self.X_all_actual, self.Y_all

        def get_best_unpenalized(self):
            return self.X_all_actual[0], self.Y_all[0], 0

        def get_penalty_mask(self, X=None):
            if X is None:
                return torch.ones(self.Y_all.shape[0], dtype=torch.bool, device=self.Y_all.device)
            return torch.ones(X.shape[0], dtype=torch.bool, device=X.device)

        def take_snapshot(self, *args, **kwargs):
            return None

        def determine_new_bounds(self, add_to_history=True):
            return self.bounds.clone()

        def update_all_needle_radii(self, M_list):
            self.needle_M_list = list(M_list)

        def shrink_all_needle_radii(self, factor):
            shrink_calls.append(factor)

        def max_needle_radius(self):
            return 0.0

        def _relabel_pared_with_medians(self):
            pass

        def _update_penalty_mask(self):
            pass

        def get_all_points(self):
            return self.X_all_actual.clone(), self.X_all_actual.clone(), self.Y_all.clone()

        def get_needle_results(self):
            return []

        def get_needle_locations(self):
            return torch.empty((0, 3), dtype=dtype, device=device)

        def get_all_needle_results(self):
            return list(self.needles_results)

        def get_all_needle_locations(self):
            dd = self.X_all_actual.shape[1]
            if self.needles.ndim == 2 and self.needles.shape[0] > 0:
                return self.needles
            return torch.empty((0, dd), dtype=dtype, device=device)

        def get_all_needle_vals(self):
            if self.needle_vals.ndim == 2 and self.needle_vals.shape[0] > 0:
                return self.needle_vals
            return torch.empty((0, 1), dtype=dtype, device=device)

    class FakeGP:
        def __init__(self):
            self.acq_fn = None
            self.raw_samples = 10
            self.nat_grad_step = 0.02

        def fit(self, X, Y):
            return None

        def create_acquisition(self, best_f=None, penalty_value=None):
            def acq(Xq):
                return torch.zeros(Xq.shape[:-1], dtype=dtype, device=device)

            self.acq_fn = acq
            return acq

        def get_candidate(self, bounds, best_f=None):
            return None  # always fail → activation_failed = True

        def recompute_all_ellipsoids(self, needles, X, Y, **kwargs):
            recompute_calls.append(1)
            return []  # empty → needle_M_list stays empty → _all_needle_axes_below_min → True

    z = ZoMBIHop.__new__(ZoMBIHop)
    z.device = device
    z.dtype = dtype
    z.verbose = False
    z._needle_plot_points_ref = None
    z.bounds = bounds0.clone()
    z.d = 3
    z.ellipsoid_drop_fraction = 0.25
    z.ellipsoid_eigenvalue_floor = 1e-6
    z.needle_shrink_factor = 0.85
    z.needle_stop_noise_multiplier = 3.0
    z.max_penalty_radius = 1.0
    z.bounds_shrink_factor = 0.8
    z.min_axis_noise_mult = 2.0
    z.zoom_jaccard_threshold = 0.75
    z.random_sampler = lambda n, a, b, **kwargs: torch.rand((n, 3), dtype=dtype, device=device)
    z.proj_fn = lambda X: X
    z.data_handler = FakeDH()
    z.gp_handler = FakeGP()

    z._log = lambda *args, **kwargs: None
    z._log_status = lambda *args, **kwargs: None
    z._check_convergence_to_needle = lambda *args, **kwargs: (False, None, None)
    z._objective_wrapper = lambda *a, **kw: (
        torch.tensor([[0.2, 0.3, 0.5]], dtype=dtype, device=device),
        torch.tensor([[1.0]], dtype=dtype, device=device),
    )

    z.run(max_activations=float("inf"), time_limit_hours=None)

    # Case 1: first failure → recompute_all_ellipsoids called exactly once
    assert len(recompute_calls) == 1, f"expected 1 recompute call, got {len(recompute_calls)}"
    # Case 3: second failure, no new data, needle_M_list empty → stop before shrink
    assert len(shrink_calls) == 0, f"expected 0 shrink calls, got {len(shrink_calls)}"

