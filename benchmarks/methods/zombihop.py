"""
benchmarks/methods/zombihop.py
==============================
ZoMBI-Hop on the unit box, behind the common :class:`~benchmarks.methods.base.Method`
interface.

This is the optimiser in ``src/`` run with ``domain=BoxDomain()`` — the same
algorithm ``run_mobo`` runs on the simplex, with only the geometry swapped (see
``src/utils/domain.py``; the simplex path is pinned bit-for-bit by
``tests/test_domain_golden.py`` and the box path is exercised by
``tests/test_box_domain.py``). It does not go through ``run_mobo.run_single_trial``,
which is simplex-only throughout (initial lines drawn on the simplex, the
deposition-physics print model, ``proj_simplex`` on every requested line, ternary
plots): the measurement here is the shared :class:`Problem`, so ZoMBI-Hop pays for
its points exactly as every baseline does.

What a line costs
-----------------
ZoMBI-Hop measures a LINE per objective call: LineBO ranks ``linebo_num_lines``
candidate chords through the acquisition maximiser, and the best is measured at
``problem.batch_size`` evenly spaced points (24 by default — the batch every
baseline gets). The initial design is ``n_init_lines`` random chords through the
box (2 x 24 = 48 points, the baselines' ``n_init``). Both count against the budget.

Config
------
hparams             ZoMBI-Hop hyperparameters (a dict, as in the ``optimize/hparams``
                    JSON files). None = ``src.default_hparams.DEFAULT_HPARAMS``. The
                    sweep fills this per dimension from ``benchmarks/sweeps/hparams.py``.
fixed               the infrastructure constants ``run_mobo.ZOMBI_FIXED`` pins
                    (GP cap, UCB, the measured input noise 0.128 as ZoMBI's length
                    scale, quiet logging). A key in both ``hparams`` and ``fixed`` is
                    dropped from ``hparams``, as ``benchmarks/ablations`` does.
n_init_lines        random chords measured before the optimiser starts
linebo_num_lines    candidate chords LineBO ranks per call (run_mobo: 10)
linebo_points_per_line
                    points per candidate chord LineBO scores the acquisition on
                    (run_mobo: 100) — scoring only, not measurement
never_terminate     keep ZoMBI-Hop sampling until the budget is spent instead of
                    stopping on its own heuristics (default True, as in the
                    simplex sweep: a cell that self-terminated after 400 points is
                    not comparable to one that spent 3000)
boundary_repulsion  ``BoxDomain(boundary_repulsion=...)``; off by default because
                    needles may sit on a face of the box
"""

from __future__ import annotations

import math

import numpy as np

from ._paths import ensure_paths
from .base import Method, Problem
from .registry import register

ensure_paths()

#: ``run_mobo.ZOMBI_FIXED``, restated so this module does not import run_mobo.
ZOMBI_FIXED = {"max_gp_points": 3000, "acquisition_type": "ucb",
               "input_noise": 0.128, "verbose": False}


def _box_chord(x0: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Endpoints of the line ``x0 + t*direction`` clipped to the unit box."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (0.0 - x0) / direction
        t1 = (1.0 - x0) / direction
    lo = np.where(direction != 0, np.minimum(t0, t1), -np.inf)
    hi = np.where(direction != 0, np.maximum(t0, t1), np.inf)
    t_min, t_max = float(lo.max()), float(hi.min())
    return (np.clip(x0 + t_min * direction, 0.0, 1.0),
            np.clip(x0 + t_max * direction, 0.0, 1.0))


@register
class ZoMBIHopMethod(Method):
    name = "zombi_hop"
    description = "ZoMBI-Hop (this repo), LineBO lines on BoxDomain()"
    declares_needles = True
    defaults = {
        "hparams": None,
        "fixed": dict(ZOMBI_FIXED),
        "n_init_lines": 2,
        "linebo_num_lines": 10,
        "linebo_points_per_line": 100,
        "never_terminate": True,
        "boundary_repulsion": False,
    }

    def __init__(self, config=None, *, seed=0, device="cpu"):
        super().__init__(config, seed=seed, device=device)
        self.optimizer = None
        self.dim: int | None = None

    # ── hyperparameters ──

    def resolved_hparams(self, dim: int) -> dict:
        from src.default_hparams import DEFAULT_HPARAMS

        hp = dict(self.config["hparams"] if self.config["hparams"] is not None
                  else DEFAULT_HPARAMS)
        for key in set(hp) & set(self.config["fixed"]):
            hp.pop(key)
        # run_single_trial's rule: the top-m ellipsoid fit needs at least d+1 points.
        if dim > 3 and (hp.get("top_m_points") is None or hp["top_m_points"] < dim + 1):
            hp["top_m_points"] = max(dim + 1, 4)
        return hp

    # ── run ──

    def run(self, problem: Problem) -> None:
        import torch

        # Importing src.core.zombihop switches torch's GLOBAL default device to CUDA
        # and default dtype to float32 when a GPU is present. Everything this cell
        # does afterwards (the needle extractor) must not inherit that, so the
        # defaults are restored however the run ends.
        prev_dtype = torch.get_default_dtype()
        prev_device = torch.get_default_device()
        try:
            self._run(problem)
        finally:
            torch.set_default_dtype(prev_dtype)
            torch.set_default_device(prev_device)

    def _run(self, problem: Problem) -> None:
        import torch

        from src.core.linebo import LineBO
        from src.core.zombihop import ZoMBIHop
        from src.utils.domain import BoxDomain

        d = self.dim = problem.dim
        dev = torch.device(self.device)
        dt = torch.float64
        B = problem.batch_size
        domain = BoxDomain(boundary_repulsion=bool(self.config["boundary_repulsion"]))
        t_line = np.linspace(0.0, 1.0, B)

        def measure_line(left: np.ndarray, right: np.ndarray, **tags):
            X_req = left[None, :] + t_line[:, None] * (right - left)[None, :]
            X_act, Y = problem.evaluate(X_req, **tags)
            return X_req[: len(Y)], X_act, Y

        # Initial design: random chords through the box.
        xa, xe, ys = [], [], []
        for _ in range(int(self.config["n_init_lines"])):
            x0 = self.rng.random(d)
            v = self.rng.normal(size=d)
            left, right = _box_chord(x0, v / np.linalg.norm(v))
            X_req, X_act, Y = measure_line(left, right, activation=-1, zoom=-1)
            xe.append(X_req)
            xa.append(X_act)
            ys.append(Y)

        def T(a):
            return torch.as_tensor(np.asarray(a, dtype=float), dtype=dt, device=dev)

        linebo = LineBO(None, d, num_points_per_line=int(self.config["linebo_points_per_line"]),
                        num_lines=int(self.config["linebo_num_lines"]),
                        device=str(dev), domain=domain)

        def objective(x_tell, bounds, acq_fn):
            x_left, x_right = linebo.ranked_line_endpoints(x_tell, bounds, acq_fn)
            dh = self.optimizer.data_handler if self.optimizer is not None else None
            tags = ({"activation": int(dh.current_activation), "zoom": int(dh.current_zoom)}
                    if dh is not None else {})
            X_req, X_act, Y = measure_line(x_left[0].detach().cpu().numpy(),
                                           x_right[0].detach().cpu().numpy(), **tags)
            return T(X_req), T(X_act), T(Y)

        self.optimizer = ZoMBIHop(
            objective=objective,
            X_init_actual=T(np.concatenate(xa)),
            X_init_expected=T(np.concatenate(xe)),
            Y_init=T(np.concatenate(ys)).reshape(-1, 1),
            **self.config["fixed"], **self.resolved_hparams(d),
            device=str(dev), dtype=dt, run_uuid=None, checkpoint_dir=None,
            domain=domain,
        )
        self.optimizer.run(max_activations=math.inf, time_limit_hours=None,
                           never_terminate=bool(self.config["never_terminate"]))

    # ── needles ──

    def declared_needles(self) -> np.ndarray:
        if self.optimizer is None:
            return np.empty((0, self.dim or 0))
        t = self.optimizer.data_handler.get_all_needle_locations()
        if t is None or t.numel() == 0:
            return np.empty((0, self.dim))
        return t.detach().cpu().numpy().reshape(-1, self.dim).astype(float)

    def needle_records(self) -> list[dict]:
        if self.optimizer is None:
            return []
        out = []
        for r in self.optimizer.data_handler.get_all_needle_results():
            mv = r.get("median_value")
            out.append({
                "value": _scalar(r.get("value")),
                "median_value": _scalar(mv),
                "activation": r.get("activation"), "zoom": r.get("zoom"),
                "iteration": r.get("iteration"), "reason": r.get("reason"),
            })
        return out

    def point_columns(self) -> dict[str, np.ndarray]:
        if self.optimizer is None:
            return {}
        mask = self.optimizer.data_handler.get_penalty_mask()
        if mask is None:
            return {}
        return {"penalized": (~mask.detach().cpu().numpy()).astype(int)}

    def summary(self) -> dict:
        if self.optimizer is None:
            return {}
        dh = self.optimizer.data_handler
        return {"n_activations": int(getattr(dh, "current_activation", 0)),
                "hparams": self.resolved_hparams(self.dim)}


def _scalar(v):
    if v is None:
        return None
    try:
        f = float(v.item() if hasattr(v, "item") else v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f
