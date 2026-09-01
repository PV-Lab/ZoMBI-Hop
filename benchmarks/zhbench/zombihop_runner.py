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


def _jsonable(obj):
    """Coerce a value into something ``json.dump`` will accept, losslessly if it can.

    ``config_resolved.json`` is written with a plain ``json.dump``, and the resolved
    hyperparameter set contains torch dtypes, numpy scalars and occasionally a
    sentinel object. Dropping those keys would defeat the point of recording the set
    at all, so unknown types are stringified rather than skipped.
    """
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, bool, int, float)) or obj is None:
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return repr(obj)


def _resolve_full_hparams(cls, passed: dict) -> dict:
    """``passed`` plus every ``cls.__init__`` default it did not override.

    The drift that motivated this lived entirely in parameters we never passed:
    ``min_zoom_for_needle`` and ``min_iters_per_zoom`` appear in none of our
    hyperparameter JSONs, so their values came from the core's signature and were
    recorded nowhere. Writing the full set down makes a run reproducible against a
    core that has since moved, and makes the diff visible when it does.

    Runtime plumbing (the objective, the initial tensors, device/dtype) is excluded:
    it is recorded elsewhere and is not a hyperparameter.
    """
    import inspect

    skip = {"self", "objective", "X_init_actual", "X_init_expected", "Y_init",
            "device", "dtype", "run_uuid", "checkpoint_dir"}
    out = dict(passed)
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - C-level or exotic __init__
        return out
    for name, p in params.items():
        if name in skip or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if name not in out and p.default is not inspect.Parameter.empty:
            out[name] = p.default
    return out


def load_hparams(name: str | None, dim: int) -> tuple[dict, str]:
    """Resolve a hyperparameter set and say where it came from.

    ``None`` picks the tuned per-dimension set for ``dim`` if one exists
    (transcribed from ``warm_start/BEST_HPARAMS.md`` for d=3 and d=4), otherwise
    ``optimize/hparams/6d_ensemble.json`` -- the 6-D MOBO ensemble winner.

    Note that is the JSON, **not** ``src.default_hparams.DEFAULT_HPARAMS``, even
    though the latter's docstring says it is "copied verbatim" from that file.
    They have drifted: the JSON has ``n_consecutive_converged=2`` (Brianna loosened
    it after the 6-D campaign, commit c4a9358) while ``default_hparams.py`` still
    says 5. The JSON is the live value the campaign actually ran, so it is the
    benchmark default; 5 is available as the ``zombihop_nc5`` sensitivity. Reported
    upstream in ``benchmarks/UPSTREAM_REQUESTS.md``.
    """
    if name is None:
        cand = os.path.join(_HPARAM_DIR, f"{dim}d.json")
        name = f"{dim}d" if os.path.exists(cand) else "6d_ensemble"
    if name == "stale_default":
        from src.default_hparams import DEFAULT_HPARAMS, DEFAULT_HPARAMS_PROVENANCE
        return (dict(DEFAULT_HPARAMS),
                f"src.default_hparams (claims {DEFAULT_HPARAMS_PROVENANCE}; STALE COPY)")
    if name == "6d_ensemble":
        from ._repo import REPO_ROOT
        path = os.path.join(REPO_ROOT, "optimize", "hparams", "6d_ensemble.json")
    elif os.path.isabs(name):
        path = name
    else:
        path = os.path.join(_HPARAM_DIR, f"{name}.json")
    with open(path, encoding="utf-8") as fh:
        hp = json.load(fh)
    # optimize/hparams/*.json nest the values under a "hparams" key.
    if "hparams" in hp and isinstance(hp["hparams"], dict):
        hp = hp["hparams"]
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
        self.hparam_adjustments: list[dict] = []
        self.stop_reason = ""

    def declared_optima(self):
        return self._needles if self._needles.size else None

    def state(self) -> dict:
        return {"name": self.name, "hparams_provenance": self.provenance,
                "n_needles": int(self._needles.shape[0]),
                "stop_reason": self.stop_reason,
                # The full ZoMBIHop configuration this cell actually ran, every
                # __init__ argument resolved -- not just the ones we passed. Without
                # this, a core-side default change re-tunes an arm and leaves no
                # trace in any artifact: the published bundle recorded `n_needles`
                # and a provenance path, and could not answer "what was
                # min_iters_per_zoom on that run?" at all. See DESIGN.md 23.
                "resolved_hparams": _jsonable(self.resolved_hparams),
                # Anything the runner changed out from under the tuned JSON, with
                # the reason. Empty is the expected state; a non-empty list means
                # this arm is not running its tuned configuration.
                "hparam_adjustments": self.hparam_adjustments}

    def _drain_needle_log(self, dh, run: ObjectiveRun) -> None:
        """Record every needle the core has declared but we have not logged yet.

        The core does not call back on declaration, so the only way to timestamp a
        needle is to notice it appeared. Called after every objective evaluation and
        once more in the ``finally``, which is what catches the needles declared by
        the budget-exhausting line.

        ``sample_idx`` is therefore an upper bound -- the budget at the moment we
        noticed, not the moment the core decided -- which is exactly the semantics
        the prefix curves need: a needle is counted at checkpoint N only once the
        samples that justified it have been spent.
        """
        if dh is None or dh.needles is None or not dh.needles.shape[0]:
            return
        n = int(dh.needles.shape[0])
        while len(self.needle_log) < n:
            self.needle_log.append({
                "needle_idx": len(self.needle_log),
                "sample_idx": int(run.n_samples),
                "batch_idx": int(run.batch_idx),
                "activation": int(dh.current_activation),
                "zoom": int(getattr(dh, "current_zoom", -1)),
            })

    # -- the run ---------------------------------------------------------------
    def run(self, objective, run: ObjectiveRun, protocol: Protocol, seed: int) -> None:
        from ._repo import evaluate, run_mobo
        rm = run_mobo()
        _force_zoom_floors = evaluate()._force_zoom_floors
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
            self._drain_needle_log(dh_ref[0], run)
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
            _prev = hp.get("top_m_points")
            hp["top_m_points"] = max(dim + 1, 4)
            self.hparam_adjustments.append(
                {"key": "top_m_points", "from": _prev, "to": hp["top_m_points"],
                 "reason": f"tuned value below dim+1 at d={dim}"})
        # The floors are read from ZoMBIHop.__init__ BY REFLECTION
        # (evaluate._force_zoom_floors), so a core-side default change silently
        # re-tunes an arm with no diff under benchmarks/. That is not hypothetical:
        # min_iters_per_zoom moved 2 -> 3 on origin/brianna, and 4d.json /
        # 6d_ensemble.json both carry max_iterations exactly at the old floor.
        # Keep the raise -- the core genuinely cannot declare a needle below it --
        # but RECORD it, so "this arm is not running its tuned configuration" is
        # visible in config_resolved.json instead of being inferable only from a
        # core commit nobody wrote down. test_core_pins.py is the tripwire.
        zoom_floor, iter_floor = _force_zoom_floors()
        self.hparam_adjustments.append(
            {"key": "_force_zoom_floors", "from": None,
             "to": {"max_zooms": zoom_floor, "max_iterations": iter_floor},
             "reason": "read from ZoMBIHop.__init__ signature by reflection"})
        for key, floor in (("max_zooms", zoom_floor), ("max_iterations", iter_floor)):
            if hp.get(key) is not None and hp[key] < floor:
                self.hparam_adjustments.append(
                    {"key": key, "from": hp[key], "to": floor,
                     "reason": "raised to the core's force-zooming floor; the tuned "
                               "value is NOT what ran"})
                hp[key] = floor

        zombi_fixed = dict(rm.ZOMBI_FIXED)
        zombi_fixed["verbose"] = bool(self.verbose)
        for k in ("input_noise", "max_gp_points", "acquisition_type", "verbose"):
            if k in hp:
                zombi_fixed[k] = hp.pop(k)
        # Record EVERY ZoMBIHop.__init__ argument, not just the ones we pass. The
        # parameters that caused the drift (min_zoom_for_needle, min_iters_per_zoom)
        # appear in no hparam file we own, so they are governed entirely by the
        # core's signature -- and a run that does not write them down cannot be
        # reproduced or even diagnosed after the core moves.
        self.resolved_hparams = _resolve_full_hparams(
            rm.ZoMBIHop, {**zombi_fixed, **hp})

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
            # Drain once more. ``obj_wrapper`` logs a needle only on the NEXT
            # objective call, so a needle declared by the budget-exhausting line --
            # or by anything after the final call -- was never recorded: ``inner()``
            # raises and the append never runs. In the published s1_real bundle that
            # left 6 of 60 cells short by one (569 declared, 563 logged), which is
            # invisible in the final metrics (they read ``dh.needles`` directly) but
            # NOT in the ``@N`` prefix curves, and fig1/fig3/fig4 are all built from
            # those. Two cells disagreed on the headline: real4d/zombihop/s6
            # peak_ratio@2000 0.185 vs 0.222 final, real6d/nc5/s5 0.000 vs 0.015.
            self._drain_needle_log(dh, run)
            t = dh.get_all_needle_locations()
            self._needles = (t.detach().cpu().numpy() if t is not None and t.numel()
                             else np.empty((0, dim)))
