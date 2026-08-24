"""Protocol invariants: budget, shared init, simplex validity, noise calibration."""

from __future__ import annotations

import numpy as np
import pytest

from zhbench import objectives as O
from zhbench.protocol import (MAX_PRINTABLE_COMPONENTS, ObjectiveRun, Protocol,
                              gen_init_design, line_realization_mode, realize,
                              realize_line, BudgetExhausted)


def _obj(dim=3):
    return O.make_ensemble(dim=dim, n_optima=4, landscape=0, seed=0)


def test_budget_is_never_exceeded_and_truncates_a_straddling_batch():
    obj = _obj()
    p = Protocol(n_samples=100, batch_size=24, noise="none")
    run = ObjectiveRun(fn=obj.fn, dim=obj.dim, protocol=p, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(10):
        try:
            run.evaluate_batch(rng.dirichlet(np.ones(3), size=24))
        except BudgetExhausted:
            break
    assert run.n_samples == 100
    # 4 full batches = 96; the 5th has room for 4 of its 24, so 20 are dropped.
    assert run.n_truncated == 20
    assert run.stacked()["X_actual"].shape[0] == 100


def test_init_design_is_identical_for_the_same_seed():
    obj = _obj()
    p = Protocol(n_samples=240, batch_size=24, noise="hardware")
    out = []
    for _ in range(2):
        run = ObjectiveRun(fn=obj.fn, dim=obj.dim, protocol=p, seed=7)
        out.append(gen_init_design(run, p, 7))
    for a, b in zip(out[0], out[1]):
        assert np.array_equal(a, b)


def test_init_design_differs_across_seeds():
    obj = _obj()
    p = Protocol(n_samples=240, batch_size=24, noise="hardware")
    a = gen_init_design(ObjectiveRun(fn=obj.fn, dim=3, protocol=p, seed=1), p, 1)
    b = gen_init_design(ObjectiveRun(fn=obj.fn, dim=3, protocol=p, seed=2), p, 2)
    assert not np.allclose(a[0], b[0])


@pytest.mark.parametrize("mode", ["none", "hardware", "physics"])
def test_realize_stays_on_the_simplex(mode):
    p = Protocol(noise=mode)
    rng = np.random.default_rng(0)
    X = rng.dirichlet(np.ones(5), size=64)
    A = realize(X, p, rng)
    assert np.allclose(A.sum(axis=1), 1.0, atol=1e-9)
    assert (A >= -1e-12).all()


def test_hardware_mode_actually_realizes_the_hardware_noise_level():
    """The point of solving for the scale rather than assuming it.

    NOISE_LEVEL is the std of (realized - requested) measured on the printer. If we
    inject N(0, 0.128) and project back to the simplex, clipping pulls the realized
    value below 0.128, and the benchmark would quietly be gentler than the lab it
    claims to model. Assert the realized number, not the injected one.
    """
    from zhbench._repo import run_mobo
    target = float(run_mobo().NOISE_LEVEL)
    rng = np.random.default_rng(0)
    p = Protocol(noise="hardware")
    for dim in (3, 4, 6):
        X = rng.dirichlet(np.ones(dim), size=3000)
        got = float(np.std(realize(X, p, rng) - X))
        assert abs(got - target) < 0.1 * target, (dim, got, target)


def test_physics_mode_is_quieter_than_hardware_mode():
    """Kept as a sensitivity, and the reason it is not the primary."""
    rng = np.random.default_rng(0)
    X = rng.dirichlet(np.ones(4), size=2000)
    d_phys = float(np.std(realize(X, Protocol(noise="physics"), rng) - X))
    d_hw = float(np.std(realize(X, Protocol(noise="hardware"), rng) - X))
    assert d_phys < d_hw
    assert d_hw / d_phys > 2.0, (d_phys, d_hw)


def test_line_realization_falls_back_above_ten_components():
    """The printer has ten syringe modules. Above that there is no hardware to
    model, and the run must say so rather than crash or pretend."""
    p = Protocol(noise="hardware")
    assert line_realization_mode(6, p) == "physics+residual"
    assert line_realization_mode(MAX_PRINTABLE_COMPONENTS, p) == "physics+residual"
    assert line_realization_mode(12, p) == "no_printer_model"
    assert line_realization_mode(6, Protocol(noise="physics")) == "physics"

    rng = np.random.default_rng(0)
    left, right = rng.dirichlet(np.ones(12), size=2)
    X = realize_line(left, right, 24, p, rng)
    assert X.shape == (24, 12)
    assert np.allclose(X.sum(axis=1), 1.0, atol=1e-9)


def test_y_true_is_noiseless_and_y_observed_is_not():
    """Metrics must never ride on a lucky measurement, so y_true is recomputed
    noiselessly at the realized composition while the optimizer only ever sees
    y_observed."""
    obj = _obj()
    p = Protocol(n_samples=48, batch_size=24, noise="hardware")
    run = ObjectiveRun(fn=obj.fn, dim=3, protocol=p, seed=0)
    rng = np.random.default_rng(0)
    X = rng.dirichlet(np.ones(3), size=24)
    X_act, y_obs = run.evaluate_batch(X)
    h = run.stacked()
    assert np.allclose(h["y_true"], [obj.fn(x) for x in X_act])
    assert not np.allclose(h["y_observed"], h["y_true"])


def test_none_mode_disables_output_noise_too():
    """The control has to be a genuine control: clean inputs AND clean outputs."""
    obj = _obj()
    p = Protocol(n_samples=48, batch_size=24, noise="none")
    run = ObjectiveRun(fn=obj.fn, dim=3, protocol=p, seed=0)
    X = np.random.default_rng(0).dirichlet(np.ones(3), size=24)
    X_act, _ = run.evaluate_batch(X)
    h = run.stacked()
    assert np.array_equal(h["X_actual"], h["X_requested"])
    assert np.allclose(h["y_observed"], h["y_true"])


def test_importing_the_core_does_not_leak_torch_global_state():
    """The core sets default device=cuda / dtype=float32 at import time when CUDA
    exists. In a benchmark process that would silently run the BoTorch baselines at
    a different precision from the method they are compared against."""
    import torch

    from zhbench import _repo

    before_dtype = torch.get_default_dtype()
    before_device = getattr(torch, "get_default_device", lambda: None)()
    _repo.run_mobo()
    assert torch.get_default_dtype() == before_dtype
    if before_device is not None:
        assert torch.get_default_device() == before_device
