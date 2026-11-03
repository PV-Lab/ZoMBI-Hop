"""LineBO penalty-zone trimming and ZoMBIHop's shrink-and-retry on NoValidSegmentError.

Device-agnostic (CUDA when available, else CPU) so the file also runs with
``pytest --noconftest`` on a machine without a GPU.
"""
import pytest
import torch

from src.core.linebo import LineBO, NoValidSegmentError, free_segments

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float64


def _t(x):
    return torch.tensor(x, device=DEV, dtype=DT)


class _SphereZoneAcq:
    """Acquisition double: value = first coordinate; one spherical penalty zone."""

    def __init__(self, center, radius, min_segment_length=0.0):
        self.center = _t(center)
        self.radius = float(radius)
        self.min_segment_length = min_segment_length

    def __call__(self, Xq):
        return Xq[..., 0]

    def penalty_mask_fn(self, X):
        return (X - self.center).norm(dim=-1) > self.radius


def _full_bounds(d):
    return torch.stack([torch.zeros(d, device=DEV, dtype=DT), torch.ones(d, device=DEV, dtype=DT)])


def test_free_segments_splits_runs_and_scores_them():
    # One line of 6 points: free, free, zone, free, free, free.
    pts = torch.linspace(0, 1, 6, device=DEV, dtype=DT).view(1, 6, 1).repeat(1, 1, 2)
    free = torch.tensor([[True, True, False, True, True, True]], device=DEV)
    acq = _t([[1.0, 3.0, -100.0, 2.0, 2.0, 5.0]])
    xl, xr, score, length = free_segments(pts, free, acq, _t([1.0]))
    assert xl.shape == (2, 2)
    torch.testing.assert_close(score, _t([2.0, 3.0]))
    torch.testing.assert_close(length, _t([0.2, 0.4]))
    torch.testing.assert_close(xl[1], pts[0, 3])
    torch.testing.assert_close(xr[1], pts[0, 5])


def test_free_segments_fully_penalized_line_gives_nothing():
    pts = torch.rand(3, 10, 4, device=DEV, dtype=DT)
    free = torch.zeros(3, 10, dtype=torch.bool, device=DEV)
    xl, _, _, _ = free_segments(pts, free, torch.zeros(3, 10, device=DEV, dtype=DT),
                                torch.ones(3, device=DEV, dtype=DT))
    assert xl.shape[0] == 0


def test_trimmed_lines_stay_out_of_zone_and_are_ranked():
    torch.manual_seed(0)
    d = 3
    acq = _SphereZoneAcq([1 / 3, 1 / 3, 1 / 3], radius=0.2, min_segment_length=0.05)
    lb = LineBO(lambda ep: None, d, num_points_per_line=100, num_lines=10, device=DEV)
    xl, xr = lb.ranked_line_endpoints(_t([0.2, 0.3, 0.5]), _full_bounds(d), acq)
    assert xl.shape[0] == 10
    # Every one of the 24 measured points of every returned segment is outside the
    # zone (the tolerance covers the 1/99 discretisation of a ~1.4-long chord).
    t = torch.linspace(0, 1, 24, device=DEV, dtype=DT).view(1, -1, 1)
    pts = xl.unsqueeze(1) * (1 - t) + xr.unsqueeze(1) * t
    assert ((pts - acq.center).norm(dim=-1) > acq.radius - 0.015).all()
    assert ((xr - xl).norm(dim=1) >= 0.05).all()
    # Ranked best first by mean acquisition (mean of x1 along a straight segment).
    score = 0.5 * (xl[:, 0] + xr[:, 0])
    assert (score[:-1] >= score[1:] - 1e-9).all()


def test_no_acceptable_segment_raises_before_objective():
    called = []
    acq = _SphereZoneAcq([1 / 3, 1 / 3, 1 / 3], radius=10.0)  # covers everything
    lb = LineBO(lambda ep: called.append(ep), 3, num_lines=2, device=DEV)
    with pytest.raises(NoValidSegmentError):
        lb.sampler(_t([0.2, 0.3, 0.5]), _full_bounds(3), acq)
    assert called == []


def test_single_surviving_segment_fills_the_cache_rail():
    acq = _SphereZoneAcq([1 / 3, 1 / 3, 1 / 3], radius=0.0)
    lb = LineBO(lambda ep: None, 3, num_lines=2, device=DEV, num_candidate_lines=1)
    xl, xr = lb.ranked_line_endpoints(_t([0.2, 0.3, 0.5]), _full_bounds(3), acq)
    assert xl.shape[0] == 2
    torch.testing.assert_close(xl[0], xl[1])
    torch.testing.assert_close(xr[0], xr[1])


def test_acquisition_without_mask_keeps_untrimmed_behaviour():
    lb = LineBO(lambda ep: None, 3, num_lines=4, device=DEV)
    xl, xr = lb.ranked_line_endpoints(_t([0.2, 0.3, 0.5]), _full_bounds(3), lambda X: X[..., 0])
    assert xl.shape[0] == 4


def test_objective_wrapper_shrinks_zones_until_a_line_is_found():
    from src.core import zombihop as zh

    d = 3
    attempts = {"n": 0}
    shrinks = []

    class _DH:
        d = 3  # ZoMBIHop.d reads data_handler.d

        def shrink_all_needle_radii(self, factor):
            shrinks.append(factor)

        def get_gp_data(self):
            return torch.rand(5, d, device=DEV, dtype=DT), torch.rand(5, 1, device=DEV, dtype=DT)

        get_zoom_gp_data = lambda self, bounds: self.get_gp_data()

        def write_log(self, msg):
            pass

        def add_all_points(self, X_actual, X_expected, Y):
            return torch.ones(X_actual.shape[0], dtype=torch.bool, device=X_actual.device)

    class _GP:
        def fit(self, X, Y):
            pass

        def create_acquisition(self, best_f=None):
            return "fresh-acq"

    seen_acq = []

    def objective(x, bounds, acq):
        seen_acq.append(acq)
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise NoValidSegmentError("all segments too short")
        X = torch.rand(24, d, device=DEV, dtype=DT)
        return X, X, torch.rand(24, device=DEV, dtype=DT)

    z = zh.ZoMBIHop.__new__(zh.ZoMBIHop)
    z.device, z.dtype = torch.device(DEV), DT
    z.data_handler, z.gp_handler = _DH(), _GP()
    z.objective = objective
    z.proj_fn = lambda X: X
    z.full_bounds = _full_bounds(d)

    X, Y = zh.ZoMBIHop._objective_wrapper(z, _t([0.2, 0.3, 0.5]), z.full_bounds, "stale-acq")
    assert X.shape == (24, d)
    assert shrinks == [zh.SEGMENT_SHRINK_FACTOR] * 2
    assert seen_acq == ["stale-acq", "fresh-acq", "fresh-acq"]


def test_trim_lines_false_hands_linebo_an_untrimmed_acquisition():
    from src.core import zombihop as zh

    d = 3
    acq = _SphereZoneAcq([1 / 3, 1 / 3, 1 / 3], radius=10.0)  # would block every line
    lb = LineBO(lambda ep: None, d, num_lines=2, device=DEV)

    class _DH:
        d = 3

        def add_all_points(self, X_actual, X_expected, Y):
            return torch.ones(X_actual.shape[0], dtype=torch.bool, device=X_actual.device)

    def objective(x, bounds, a):
        xl, xr = lb.ranked_line_endpoints(x, bounds, a)  # raises if still trimming
        X = torch.rand(24, d, device=DEV, dtype=DT)
        return X, X, torch.rand(24, device=DEV, dtype=DT)

    z = zh.ZoMBIHop.__new__(zh.ZoMBIHop)
    z.device, z.dtype = torch.device(DEV), DT
    z.data_handler = _DH()
    z.objective = objective
    z.proj_fn = lambda X: X
    X, _ = zh.ZoMBIHop._objective_wrapper(z, _t([0.2, 0.3, 0.5]), _full_bounds(d), acq,
                                          trim_lines=False)
    assert X.shape == (24, d)
    assert zh.LINE_TRIM_ZOOMS == (0,)
