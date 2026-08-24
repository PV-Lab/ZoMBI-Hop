"""ZoMBI-Hop under a sample budget, with no changes to the core.

This mirrors ``optimize/evaluate.py::run_single_eval``, which is already the
reference way to drive the current core against a landscape. The budget is
enforced exactly the way that function enforces its line budget: by raising out of
the objective wrapper, with ``never_terminate=True`` so the optimiser does not
stop on its own. Nothing is patched, no ``max_objective_calls`` hook is
reintroduced, and the benchmark therefore stays in sync with whatever lands on
``brianna`` next.

The one difference from ``run_single_eval``: the budget is counted in SAMPLES, not
lines, because that is the currency every method is charged in here. One LineBO
line is ``NUM_EXPERIMENTS`` (24) samples.
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import protocol as P
from .protocol import BudgetExhausted, ObjectiveRun, Protocol

_HPARAM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hparams")


def load_hparams(name: str | None, dim: int) -> tuple[dict, str]:
    """Resolve a hyperparameter set and say where it came from.

    ``None`` picks the best available for ``dim``: the tuned per-dimension sets
    transcribed from ``warm_start/BEST_HPARAMS.md`` for d=3 and d=4, otherwise
    ``src.default_hparams.DEFAULT_HPARAMS`` (the 6-D MOBO ensemble winner, which is
    the production default).
    """
    if name is None:
        cand = os.path.join(_HPARAM_DIR, f"{dim}d.json")
        name = f"{dim}d" if os.path.exists(cand) else "default"
    if name == "default":
        from src.default_hparams import DEFAULT_HPARAMS, DEFAULT_HPARAMS_PROVENANCE
        return dict(DEFAULT_HPARAMS), f"src.default_hparams ({DEFAULT_HPARAMS_PROVENANCE})"
    path = name if os.path.isabs(name) else os.path.join(_HPARAM_DIR, f"{name}.json")
    with open(path, encoding="utf-8") as fh:
        hp = json.load(fh)
    return hp, path


class ZoMBIHopRunner:
    """Not a ``suggest``/``observe`` optimizer -- it drives its own closed loop."""

    name = "zombihop"
    self_driving = True

    def __init__(self, hparams: str | None = None, device: str | None = None,
                 num_lines: int | None = None, verbose: bool = False, **kwargs):
        self.hparams_name = hparams
        self.device = device
        self.num_lines = num_lines
        self.verbose = verbose
        self.kwargs = kwargs
        self._needles = np.empty((0, 0))
        self.needle_log: list[dict] = []
        self.provenance = ""
        self.resolved_hparams: dict = {}
        self.stop_reason = ""

    def declared_optima(self):
        return self._needles if self._needles.size else None

    def state(self) -> dict:
        return {"name": self.name, "hparams_provenance": self.provenance,
                "n_needles": int(self._needles.shape[0]),
                "stop_reason": self.stop_reason}

    # -- the run ---------------------------------------------------------------
    def run(self, objective, run: ObjectiveRun, protocol: Protocol, seed: int) -> None:
        from ._repo import run_mobo
        rm = run_mobo()
        from optimize.evaluate import _force_zoom_floors
        import torch

        dim = int(objective.dim)
        device = torch.device(self.device) if self.device else rm.DEVICE
        dtype = rm.DTYPE

        rm.torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))

        # 1. Shared initial design -- the same two printed lines every method gets.
        from .protocol import gen_init_design
        X_req0, X_act0, y0 = gen_init_design(run, protocol, seed)

        # 2. The objective the core will call: one call == one LineBO line.
        #
        # This is a local stand-in for ``run_mobo.make_sim_obj`` with the same
        # contract, for two reasons. It routes the line through
        # ``ObjectiveRun.evaluate_batch``, which makes that class the single place
        # noise is applied, samples are counted and the budget is enforced -- for
        # ZoMBI-Hop and for every baseline alike. And it works above ten
        # components, where ``physics_simulate_line`` raises because the printer
        # only has ten syringe modules. LineBO itself is untouched.
        plot_state: dict = {"line_0": None, "line_1": None}
        q = protocol.batch_size
        line_rng = np.random.default_rng(seed + 104_729)

        def sim_obj(endpoints):
            left = endpoints[0, 0].detach().cpu().numpy().astype(float)
            right = endpoints[0, 1].detach().cpu().numpy().astype(float)
            t = np.linspace(0.0, 1.0, q)[:, None]
            X_req = left[None, :] + t * (right - left)[None, :]
            X_phys = P.realize_line(left, right, q, protocol, line_rng)
            X_act, y = run.evaluate_batch(X_req, X_actual=X_phys)
            y_signed = y if objective.maximize else -y
            return (torch.as_tensor(X_act, device=device, dtype=dtype),
                    torch.as_tensor(y_signed, device=device, dtype=dtype))

        inner = rm.make_linebo_wrapper(
            sim_obj, dim, self.num_lines or rm.NUM_LINES, device, dtype, plot_state)

        dh_ref: list = [None]

        def obj_wrapper(x_tell, bounds, acq_fn):
            # evaluate_batch (inside sim_obj) raises BudgetExhausted once the budget
            # is spent, which unwinds out of ZoMBIHop.run exactly the way
            # evaluate._LineBudgetReached does.
            x_req, x_act, y = inner(x_tell, bounds, acq_fn)
            dh = dh_ref[0]
            if dh is not None and dh.needles is not None and dh.needles.shape[0]:
                n = int(dh.needles.shape[0])
                while len(self.needle_log) < n:
                    k = len(self.needle_log)
                    self.needle_log.append({
                        "needle_idx": k,
                        "sample_idx": int(run.n_samples),
                        "batch_idx": int(run.batch_idx),
                        "activation": int(dh.current_activation),
                        "zoom": int(getattr(dh, "current_zoom", -1)),
                    })
            return x_req, x_act, y

        # 3. Hyperparameters, with the force-zooming floors the core requires.
        hp, self.provenance = load_hparams(self.hparams_name, dim)
        # ``input_noise`` is dropped on purpose: the tuned per-dimension JSONs carry
        # the pre-2026-08-12 value (0.064) while ZOMBI_FIXED now carries the
        # hardware-measured 0.128. Taking it from ZOMBI_FIXED keeps the benchmark on
        # the same noise model the team's production runs use. Override via
        # ``optimizer_spec["input_noise"]`` if you want to test the other value.
        _drop = ("run_uuid", "d", "device", "dtype", "input_noise")
        hp = {k: v for k, v in hp.items()
              if k not in _drop and not k.startswith("_")}
        hp.update({k: v for k, v in self.kwargs.items() if not k.startswith("_")})
        if dim > 3 and (hp.get("top_m_points") is None or hp.get("top_m_points", 0) < dim + 1):
            hp["top_m_points"] = max(dim + 1, 4)
        zoom_floor, iter_floor = _force_zoom_floors()
        for key, floor in (("max_zooms", zoom_floor), ("max_iterations", iter_floor)):
            if hp.get(key) is not None and hp[key] < floor:
                hp[key] = floor

        zombi_fixed = dict(rm.ZOMBI_FIXED)
        zombi_fixed["verbose"] = bool(self.verbose)
        for k in ("input_noise", "max_gp_points", "acquisition_type", "verbose"):
            if k in hp:
                zombi_fixed[k] = hp.pop(k)
        self.resolved_hparams = {**zombi_fixed, **hp}

        to_t = lambda a: torch.as_tensor(np.asarray(a, dtype=float), device=device, dtype=dtype)
        optimizer = rm.ZoMBIHop(
            objective=obj_wrapper,
            X_init_actual=to_t(X_act0),
            X_init_expected=to_t(X_req0),
            Y_init=to_t(y0).reshape(-1, 1),
            **zombi_fixed, **hp,
            device=str(device), dtype=dtype,
            run_uuid=None, checkpoint_dir=None,
        )
        dh_ref[0] = optimizer.data_handler

        try:
            optimizer.run(max_activations=float("inf"), never_terminate=True)
            self.stop_reason = "core_returned"
        except BudgetExhausted:
            self.stop_reason = "budget"
        except Exception as exc:  # a crash must not silently score as zero
            self.stop_reason = f"error: {type(exc).__name__}: {exc}"
            raise
        finally:
            dh = optimizer.data_handler
            t = dh.get_all_needle_locations()
            self._needles = (t.detach().cpu().numpy() if t is not None and t.numel()
                             else np.empty((0, dim)))
