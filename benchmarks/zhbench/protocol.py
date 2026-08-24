"""Protocol: budget, batching, initial design, and the noise model.

Every method gets the same sample budget, the same batch size, the same initial
design, and the same *magnitude* of realization error.

Noise modes (``Protocol.noise``):

``hardware`` (primary)
    Everyone is perturbed at the level the real printer was measured at:
    ``run_mobo.NOISE_LEVEL`` = 0.128 per component. Lines get the deterministic
    print model plus a random residual; batch points get the random part alone.
    This is the honest primary because ZoMBI-Hop is *told* ``input_noise=0.128``,
    so testing it in a world 3x quieter than that makes it over-conservative for
    no reason, and because ~87% of the measured hardware residual is
    perpendicular to the requested line -- which a ramp-lag/diffusion model cannot
    produce at all.

``physics`` (sensitivity)
    Lines get the deterministic print model only; batch points get a perturbation
    bootstrapped from that model's own residual distribution. Self-consistent, but
    2.5-4x quieter than the printer.

``none`` (control)
    Perfect realization and no output noise. Says how much of the ranking is
    driven by the noise model at all.

In ``hardware`` mode the perturbation scale is *solved* rather than assumed.
Adding ``N(0, 0.128)`` and projecting back to the simplex lands well below 0.128,
because projection clips. ``calibrate_hardware_noise`` bisects on the pre-projection
scale until the realized per-component std of ``X_actual - X_requested`` equals
NOISE_LEVEL, separately for lines (where the print model already contributes) and
for batches. ``tests/test_protocol.py`` asserts the realized value.

Output noise is identical for everyone: multiplicative, std
``run_mobo.OUTPUT_NOISE_FRAC`` (0.045) times |y|.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from .spaces import project_simplex

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_CALIB_PATH = os.path.join(_DATA_DIR, "input_noise_calibration.json")
_HW_PATH = os.path.join(_DATA_DIR, "hardware_noise_calibration.json")

# N=1000 is ~12 h of SDL runtime (Aleks's stated minimum); q=24 is one printed line.
DEFAULT_N_SAMPLES = 1000
DEFAULT_BATCH = 24
DEFAULT_INIT_LINES = 2          # == run_mobo.N_INIT_LINES; 48 points

#: The print model has ten syringe modules and raises above that. A HARDWARE limit:
#: the SDL cannot print an 11-component gradient, so any run above 10 components is
#: a purely computational study. Such runs record
#: ``line_realization: "no_printer_model"`` so it never gets lost in a plot.
MAX_PRINTABLE_COMPONENTS = 10

NOISE_MODES = ("hardware", "physics", "none")


class BudgetExhausted(Exception):
    """Raised by the objective wrapper once N samples have been consumed.

    Mirrors ``optimize.evaluate._LineBudgetReached``: the budget is enforced by
    raising out of the objective, so the ZoMBI-Hop core needs no patching and the
    benchmark stays in sync with whatever brianna commits next.
    """

    def __init__(self, n_samples: int):
        super().__init__(f"sample budget reached: {n_samples}")
        self.n_samples = int(n_samples)


@dataclass
class Protocol:
    """Everything that must be identical across methods for a fair comparison."""

    n_samples: int = DEFAULT_N_SAMPLES
    batch_size: int = DEFAULT_BATCH
    n_init_lines: int = DEFAULT_INIT_LINES
    noise: str = "hardware"
    #: Override the target per-component realization std. None -> run_mobo.NOISE_LEVEL.
    noise_level: float | None = None
    output_noise_frac: float | None = None  # None -> run_mobo.OUTPUT_NOISE_FRAC
    domain: str = "simplex"                 # "simplex" | "cube"
    #: Sample counts at which metrics are also evaluated, by prefix. The endpoint is
    #: always included. 3-D is saturated by uniform sampling well before N=1000, so
    #: the small-N columns are where it discriminates.
    eval_at: tuple[int, ...] = (250, 500, 1000, 2000)

    def __post_init__(self):
        if self.noise not in NOISE_MODES:
            raise ValueError(f"noise must be one of {NOISE_MODES}, got {self.noise!r}")
        self.eval_at = tuple(int(n) for n in self.eval_at if int(n) <= self.n_samples)

    @property
    def n_init_points(self) -> int:
        return self.n_init_lines * self.batch_size

    @property
    def n_decisions(self) -> int:
        """Batches after the initial design. Same for every method."""
        return max(0, (self.n_samples - self.n_init_points) // self.batch_size)

    def resolved_noise_level(self) -> float:
        if self.noise_level is not None:
            return float(self.noise_level)
        from ._repo import run_mobo
        return float(run_mobo().NOISE_LEVEL)

    def resolved_output_noise_frac(self) -> float:
        if self.noise == "none":
            return 0.0
        if self.output_noise_frac is not None:
            return float(self.output_noise_frac)
        from ._repo import run_mobo
        return float(run_mobo().OUTPUT_NOISE_FRAC)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_decisions"] = self.n_decisions
        d["n_init_points"] = self.n_init_points
        d["eval_at"] = list(self.eval_at)
        return d


# --- calibration: the physics model's own residual distribution ---------------

def calibrate_input_noise(dims=(3, 4, 5, 6, 8, 10), n_lines: int = 200,
                          seed: int = 0, path: str = _CALIB_PATH) -> dict:
    """Measure the realization error ``physics_simulate_line`` actually produces.

    Used by ``noise="physics"``. Dimensions above ten are skipped because the
    print model raises there.
    """
    from ._repo import _ensure_path
    _ensure_path()
    import torch
    from optimize.composition_prediction import physics_simulate_line

    rng = np.random.default_rng(seed)
    table: dict[str, dict] = {}
    for d in dims:
        if d > MAX_PRINTABLE_COMPONENTS:
            continue
        mags: list[float] = []
        for _ in range(n_lines):
            a = rng.dirichlet(np.ones(d))
            b = rng.dirichlet(np.ones(d))
            try:
                act = physics_simulate_line(torch.as_tensor(a), torch.as_tensor(b),
                                            num_points=DEFAULT_BATCH)
            except Exception:
                continue
            act = np.asarray(
                act.detach().cpu().numpy() if hasattr(act, "detach") else act,
                dtype=float)
            t = np.linspace(0.0, 1.0, act.shape[0])[:, None]
            req = a[None, :] + t * (b - a)[None, :]
            mags.extend(np.linalg.norm(act - req, axis=1).tolist())
        arr = np.asarray(mags, dtype=float)
        if arr.size == 0:
            continue
        table[str(d)] = {
            "n": int(arr.size),
            "mean_l2": float(arr.mean()),
            "median_l2": float(np.median(arr)),
            "p95_l2": float(np.percentile(arr, 95)),
            "per_component_std": float(arr.mean() / np.sqrt(d)),
            "quantiles": [float(q) for q in np.percentile(arr, np.arange(0, 101, 2))],
        }
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"source": "optimize.composition_prediction.physics_simulate_line",
                   "n_lines_per_dim": n_lines, "seed": seed, "dims": table},
                  fh, indent=2)
    return table


def _load_json(path: str, what: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{what} missing at {path}. Generate it with: "
            "python -m benchmarks.zhbench.protocol --calibrate")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _interp_dim(table: dict, dim: int, key: str) -> np.ndarray | float:
    if str(dim) in table:
        v = table[str(dim)][key]
        return np.asarray(v, dtype=float) if isinstance(v, list) else float(v)
    keys = sorted(int(k) for k in table)
    below = [k for k in keys if k <= dim]
    above = [k for k in keys if k >= dim]
    lo = max(below) if below else keys[0]
    hi = min(above) if above else keys[-1]
    a, b = table[str(lo)][key], table[str(hi)][key]
    if lo == hi:
        return np.asarray(a, dtype=float) if isinstance(a, list) else float(a)
    w = (dim - lo) / (hi - lo)
    if isinstance(a, list):
        return (1 - w) * np.asarray(a, float) + w * np.asarray(b, float)
    return float((1 - w) * a + w * b)


# --- calibration: the pre-projection scale that lands on NOISE_LEVEL ----------

def _base_pairs(dim: int, kind: str, rng: np.random.Generator, n: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """Requested points and their pre-perturbation realization.

    Computed ONCE per (dim, kind) and reused across every bisection step: the
    deterministic print model does not depend on the perturbation scale, and
    re-running it inside the loop costs ~0.4 s per line for nothing.
    """
    if kind == "batch":
        X_req = rng.dirichlet(np.ones(dim), size=n)
        return X_req, X_req.copy()
    from ._repo import _ensure_path
    _ensure_path()
    import torch
    from optimize.composition_prediction import physics_simulate_line
    reqs, acts = [], []
    per_line = DEFAULT_BATCH
    for _ in range(max(1, n // per_line)):
        a, b = rng.dirichlet(np.ones(dim), size=2)
        t = np.linspace(0.0, 1.0, per_line)[:, None]
        reqs.append(a[None, :] + t * (b - a)[None, :])
        act = physics_simulate_line(torch.as_tensor(a), torch.as_tensor(b),
                                    num_points=per_line)
        acts.append(np.asarray(
            act.detach().cpu().numpy() if hasattr(act, "detach") else act, dtype=float))
    return np.vstack(reqs), np.vstack(acts)


def _realized_std(scale: float, X_req: np.ndarray, X_base: np.ndarray,
                  rng: np.random.Generator) -> float:
    """Pooled per-component std of (realized - requested) at a given scale."""
    n, dim = X_base.shape
    X_act = project_simplex(X_base + _perturb(n, dim, scale, rng))
    return float(np.std(X_act - X_req))


def calibrate_hardware_noise(dims=(3, 4, 5, 6, 8, 10, 12), target: float | None = None,
                             seed: int = 0, n: int = 960, path: str = _HW_PATH) -> dict:
    """Solve for the perturbation scale that realizes ``target`` per-component std.

    Bisection, because there is no closed form: projection back to the simplex
    clips, so the realized std is strictly below the injected one by an amount that
    depends on the dimension and on how close the requested point sits to a face.
    Solving it is the difference between "we injected 0.128" and "the samples really
    are 0.128 off", and only the second is what NOISE_LEVEL means.
    """
    if target is None:
        from ._repo import run_mobo
        target = float(run_mobo().NOISE_LEVEL)
    out: dict[str, dict] = {}
    for d in dims:
        kinds = ["batch"] + (["line"] if d <= MAX_PRINTABLE_COMPONENTS else [])
        entry = {}
        for kind in kinds:
            X_req, X_base = _base_pairs(d, kind, np.random.default_rng(seed), n)
            floor = _realized_std(0.0, X_req, X_base, np.random.default_rng(seed))
            lo, hi = 0.0, 1.0
            for _ in range(30):
                mid = 0.5 * (lo + hi)
                if _realized_std(mid, X_req, X_base,
                                 np.random.default_rng(seed)) < target:
                    lo = mid
                else:
                    hi = mid
            scale = 0.5 * (lo + hi)
            got = _realized_std(scale, X_req, X_base, np.random.default_rng(seed + 1))
            entry[kind] = {"scale": float(scale), "realized_std": float(got),
                           "unperturbed_std": float(floor)}
        out[str(d)] = entry
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"target_per_component_std": float(target), "seed": seed,
                   "note": "scale is pre-projection; realized_std is what the "
                           "samples actually come out at; unperturbed_std is the "
                           "print model's own contribution (0 for batches)",
                   "dims": out}, fh, indent=2)
    return out


def _hardware_scale(dim: int, kind: str) -> float:
    tbl = _load_json(_HW_PATH, "hardware noise calibration")["dims"]
    if str(dim) in tbl and kind in tbl[str(dim)]:
        return float(tbl[str(dim)][kind]["scale"])
    sub = {k: v[kind] for k, v in tbl.items() if kind in v}
    if not sub:
        sub = {k: v["batch"] for k, v in tbl.items() if "batch" in v}
    return float(_interp_dim(sub, dim, "scale"))


def _zero_sum_unit(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` random unit directions tangent to the simplex (rows sum to zero)."""
    u = rng.standard_normal((n, dim))
    u -= u.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(u, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    return u / nrm


def _perturb(n: int, dim: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Isotropic tangent-space perturbation with the given per-component scale."""
    return _zero_sum_unit(n, dim, rng) * (scale * np.sqrt(dim) *
                                          np.abs(rng.standard_normal((n, 1))))


def realize(X_requested: np.ndarray, protocol: Protocol,
            rng: np.random.Generator) -> np.ndarray:
    """Turn requested compositions into what the lab would actually make.

    Point-wise, for scattered batches. Lines go through :func:`realize_line`.
    """
    X = np.atleast_2d(np.asarray(X_requested, dtype=float))
    n, dim = X.shape
    if protocol.noise == "none":
        return X.copy()
    if protocol.noise == "hardware":
        pert = _perturb(n, dim, _hardware_scale(dim, "batch"), rng)
    elif protocol.noise == "physics":
        q = np.asarray(_interp_dim(
            _load_json(_CALIB_PATH, "input-noise calibration")["dims"],
            dim, "quantiles"), dtype=float)
        mags = np.interp(rng.random(n), np.linspace(0.0, 1.0, q.size), q)
        pert = _zero_sum_unit(n, dim, rng) * mags[:, None]
    else:
        raise ValueError(f"unknown noise mode: {protocol.noise!r}")
    X_act = X + pert
    if protocol.domain == "cube":
        return np.clip(X_act, 0.0, 1.0)
    return project_simplex(X_act)


def line_realization_mode(dim: int, protocol: Protocol) -> str:
    if protocol.noise == "none":
        return "clean_segment"
    if protocol.domain == "cube":
        return "no_printer_model"
    if dim > MAX_PRINTABLE_COMPONENTS:
        return "no_printer_model"
    return "physics" if protocol.noise == "physics" else "physics+residual"


def realize_line(left: np.ndarray, right: np.ndarray, n_points: int,
                 protocol: Protocol, rng: np.random.Generator | None = None
                 ) -> np.ndarray:
    """Compositions the printer would actually deposit for a requested chord.

    For d <= 10 this starts from the deterministic hardware model (ramp
    lag/overshoot + junction-volume diffusion mixing) the ZoMBI-Hop core uses. In
    ``hardware`` mode a random residual is added on top, because the deterministic
    model reproduces only the along-line part of the error while ~87% of the
    measured hardware residual is perpendicular to the requested line.

    Above ten components no printer exists, so the chord gets the point-wise
    perturbation instead.
    """
    rng = rng or np.random.default_rng(0)
    t = np.linspace(0.0, 1.0, n_points)[:, None]
    clean = left[None, :] + t * (right - left)[None, :]
    dim = left.shape[0]
    mode = line_realization_mode(dim, protocol)
    if mode == "clean_segment":
        return clean
    if mode == "no_printer_model":
        return realize(clean, protocol, rng)

    from ._repo import _ensure_path
    _ensure_path()
    import torch
    from optimize.composition_prediction import physics_simulate_line
    act = physics_simulate_line(torch.as_tensor(left), torch.as_tensor(right),
                                num_points=n_points)
    act = np.asarray(act.detach().cpu().numpy() if hasattr(act, "detach") else act,
                     dtype=float)
    if mode == "physics":
        return act
    act = act + _perturb(n_points, dim, _hardware_scale(dim, "line"), rng)
    return project_simplex(act)


# --- the wrapper every method goes through -----------------------------------

@dataclass
class ObjectiveRun:
    """Counts samples, applies noise, records history, stops the run at N.

    The single entry point is :meth:`evaluate_batch`, and every method goes
    through it -- baselines with a scattered batch, ZoMBI-Hop with a printed line
    (its runner supplies a local ``sim_obj`` that calls this rather than
    ``run_mobo.make_sim_obj``, so the core is untouched but the accounting is
    shared). One counter, one budget exception, one ``OUTPUT_NOISE_FRAC``.
    """

    fn: Callable[[np.ndarray], float]
    dim: int
    protocol: Protocol
    seed: int = 0
    maximize: bool = True
    fn_batch: Callable[[np.ndarray], np.ndarray] | None = None

    n_samples: int = field(default=0, init=False)
    batch_idx: int = field(default=0, init=False)
    X_requested: list = field(default_factory=list, init=False)
    X_actual: list = field(default_factory=list, init=False)
    y_observed: list = field(default_factory=list, init=False)  # what the method saw
    y_true: list = field(default_factory=list, init=False)      # noiseless f(X_actual)
    batch_of: list = field(default_factory=list, init=False)
    n_truncated: int = field(default=0, init=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._out_frac = self.protocol.resolved_output_noise_frac()

    # -- ground truth: never noisy, never counted -----------------------------
    def f_true(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.fn_batch is not None:
            return np.asarray(self.fn_batch(X), dtype=float).ravel()
        return np.asarray([float(self.fn(x)) for x in X], dtype=float)

    def _add_output_noise(self, y: np.ndarray) -> np.ndarray:
        if self._out_frac <= 0:
            return y
        return y + self._rng.standard_normal(y.shape) * (self._out_frac * np.abs(y))

    def _check_budget(self) -> None:
        if self.n_samples >= self.protocol.n_samples:
            raise BudgetExhausted(self.n_samples)

    def evaluate_batch(self, X_requested: np.ndarray,
                       X_actual: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
        """Realize, measure, record, and count a batch.

        ``X_actual`` may be supplied when the caller has already realized the
        request through a different physical path -- the shared initial design and
        the ZoMBI-Hop line path both do this, because those batches really are
        printed lines.
        """
        self._check_budget()
        X_req = np.atleast_2d(np.asarray(X_requested, dtype=float))
        X_act = (realize(X_req, self.protocol, self._rng) if X_actual is None
                 else np.atleast_2d(np.asarray(X_actual, dtype=float)))
        y_clean = self.f_true(X_act)
        y_obs = self._add_output_noise(y_clean)
        self._append(X_req, X_act, y_obs, y_clean)
        return X_act, y_obs

    def _append(self, X_req, X_act, y_obs, y_clean) -> None:
        room = self.protocol.n_samples - self.n_samples
        n = X_act.shape[0]
        if n > room:
            # A line may straddle the budget boundary. Truncate rather than
            # over-spend, and report n_truncated in metrics.json.
            self.n_truncated += n - room
            X_req, X_act = X_req[:room], X_act[:room]
            y_obs, y_clean = y_obs[:room], y_clean[:room]
            n = room
        if n <= 0:
            return
        self.X_requested.append(X_req)
        self.X_actual.append(X_act)
        self.y_observed.append(y_obs)
        self.y_true.append(y_clean)
        self.batch_of.append(np.full(n, self.batch_idx, dtype=int))
        self.n_samples += n
        self.batch_idx += 1

    # -- accessors ------------------------------------------------------------
    def stacked(self) -> dict[str, np.ndarray]:
        if not self.X_actual:
            empty = np.empty((0, self.dim))
            return {"X_requested": empty, "X_actual": empty,
                    "y_observed": np.empty(0), "y_true": np.empty(0),
                    "batch": np.empty(0, dtype=int)}
        return {
            "X_requested": np.vstack(self.X_requested),
            "X_actual": np.vstack(self.X_actual),
            "y_observed": np.concatenate(self.y_observed),
            "y_true": np.concatenate(self.y_true),
            "batch": np.concatenate(self.batch_of),
        }

    def realized_noise_std(self) -> float:
        """Pooled per-component std of (realized - requested), as actually run."""
        h = self.stacked()
        if h["X_actual"].shape[0] == 0:
            return float("nan")
        return float(np.std(h["X_actual"] - h["X_requested"]))


def random_chord(dim: int, rng: np.random.Generator,
                 domain: str = "simplex") -> tuple[np.ndarray, np.ndarray]:
    """A chord through the domain centroid, extended to the boundary."""
    if domain == "cube":
        x0 = np.full(dim, 0.5)
        d = rng.standard_normal(dim)
    else:
        x0 = np.full(dim, 1.0 / dim)
        d = rng.standard_normal(dim)
        d -= d.mean()
    nrm = np.linalg.norm(d)
    d = d / (nrm if nrm > 1e-12 else 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        if domain == "cube":
            t_lo = np.where(d > 0, (0.0 - x0) / d, (1.0 - x0) / d)
            t_hi = np.where(d > 0, (1.0 - x0) / d, (0.0 - x0) / d)
            lo = float(np.max(np.where(np.isfinite(t_lo), t_lo, -np.inf)))
            hi = float(np.min(np.where(np.isfinite(t_hi), t_hi, np.inf)))
        else:
            # x0 + t*d >= 0 componentwise; the sum stays 1 because d sums to zero.
            t = -x0 / d
            lo = float(np.max(np.where(d < 0, t, -np.inf)))
            hi = float(np.min(np.where(d > 0, t, np.inf)))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        lo, hi = -0.1, 0.1
    return x0 + lo * d, x0 + hi * d


def gen_init_design(objective_run: ObjectiveRun, protocol: Protocol,
                    seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The shared initial design: ``n_init_lines`` random chords through the centroid.

    Mirrors the construction in ``optimize.evaluate.gen_init_data``, including
    realizing each chord as a printed line. Routed through :class:`ObjectiveRun`
    so it counts against the budget, and driven by the run seed alone so it is
    byte-identical across methods at a given seed.
    """
    rng = np.random.default_rng(seed)
    dim = objective_run.dim
    X_req_all, X_act_all, y_all = [], [], []
    for _ in range(protocol.n_init_lines):
        left, right = random_chord(dim, rng, domain=protocol.domain)
        t = np.linspace(0.0, 1.0, protocol.batch_size)[:, None]
        X_req = left[None, :] + t * (right - left)[None, :]
        X_phys = realize_line(left, right, protocol.batch_size, protocol, rng)
        X_act, y = objective_run.evaluate_batch(X_req, X_actual=X_phys)
        X_req_all.append(X_req)
        X_act_all.append(X_act)
        y_all.append(y)
    return np.vstack(X_req_all), np.vstack(X_act_all), np.concatenate(y_all)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="noise calibration")
    ap.add_argument("--calibrate", action="store_true",
                    help="regenerate both calibration tables")
    ap.add_argument("--n-lines", type=int, default=200)
    ap.add_argument("--dims", type=int, nargs="+", default=[3, 4, 5, 6, 8, 10, 12])
    args = ap.parse_args()
    if args.calibrate:
        phys = [d for d in args.dims if d <= MAX_PRINTABLE_COMPONENTS]
        print("physics residual distribution:")
        for d, v in calibrate_input_noise(dims=tuple(phys), n_lines=args.n_lines).items():
            print(f"  d={d:>3} mean_l2={v['mean_l2']:.4f} per_comp_std={v['per_component_std']:.4f}")
        print("\nhardware-matched scales (solved so realized std == NOISE_LEVEL):")
        for d, v in calibrate_hardware_noise(dims=tuple(args.dims)).items():
            parts = " ".join(f"{k}: scale={x['scale']:.4f} realized={x['realized_std']:.4f}"
                             for k, x in v.items())
            print(f"  d={d:>3} {parts}")
