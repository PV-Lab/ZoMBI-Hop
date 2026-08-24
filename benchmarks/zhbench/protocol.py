"""Protocol: budget, batching, initial design, and the noise model.

Every method in this benchmark gets the same sample budget, the same batch size,
the same initial design, and the same *magnitude* of realization error. The one
thing that necessarily differs is the SHAPE of that error, and it differs for a
physical reason:

  * ZoMBI-Hop + LineBO asks for a printed LINE. The core turns the requested
    endpoints into the compositions the printer would actually deposit via
    ``optimize.composition_prediction.physics_simulate_line`` -- a deterministic
    model of syringe ramp lag/overshoot plus junction-volume diffusion mixing.
  * A batch baseline asks for q unrelated points. There is no line, so the
    physics model does not apply. Those points get an i.i.d. perturbation whose
    magnitude distribution is CALIBRATED to the physics model at the same
    dimension (``calibrate_input_noise``), then are projected back to the simplex.

Why calibrate rather than reuse ``run_mobo.NOISE_LEVEL`` directly: NOISE_LEVEL
(0.128 per component) was measured on real 6-D hardware, where the mean L2
requested-vs-realized error is 0.271. The physics simulator that ZoMBI-Hop
actually faces in these benchmarks is gentler -- measured mean L2 0.066-0.090 for
d in {3,4,6}. Handing baselines i.i.d. N(0, 0.128) would give them roughly 3x the
handicap ZoMBI-Hop carries, which would silently manufacture the result we are
trying to test.

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

_CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "input_noise_calibration.json")

# Sample budget defaults. N=1000 is ~12 h of SDL runtime (Aleks's stated minimum);
# q=24 is one printed line.
DEFAULT_N_SAMPLES = 1000
DEFAULT_BATCH = 24
DEFAULT_INIT_LINES = 2          # == run_mobo.N_INIT_LINES; 48 points


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
    # "empirical" = bootstrap magnitudes calibrated against physics_simulate_line
    #               (the default, and the only mode fair to both sides)
    # "gaussian"  = i.i.d. N(0, input_noise_std) per component      [ablation]
    # "none"      = perfect realization                             [ablation]
    input_noise: str = "empirical"
    input_noise_std: float | None = None    # only for input_noise == "gaussian"
    output_noise_frac: float | None = None  # None -> run_mobo.OUTPUT_NOISE_FRAC
    domain: str = "simplex"                 # "simplex" | "cube"

    @property
    def n_init_points(self) -> int:
        return self.n_init_lines * self.batch_size

    @property
    def n_decisions(self) -> int:
        """Batches after the initial design. Same for every method."""
        return max(0, (self.n_samples - self.n_init_points) // self.batch_size)

    def resolved_output_noise_frac(self) -> float:
        if self.output_noise_frac is not None:
            return float(self.output_noise_frac)
        from ._repo import run_mobo
        return float(run_mobo().OUTPUT_NOISE_FRAC)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["n_decisions"] = self.n_decisions
        d["n_init_points"] = self.n_init_points
        return d


# --- input-noise calibration -------------------------------------------------

def calibrate_input_noise(dims=(3, 4, 5, 6, 8, 10, 12), n_lines: int = 200,
                          seed: int = 0, path: str = _CALIB_PATH) -> dict:
    """Measure the realization error ``physics_simulate_line`` actually produces.

    For each dimension we draw ``n_lines`` random simplex chords, ask the physics
    model what the printer would deposit, and record the per-point L2 distance
    from the clean straight segment. Those magnitudes are the empirical
    distribution baselines bootstrap from, so both sides of the comparison face
    the same amount of composition error.

    Writes a JSON table and returns it. Run once; re-run only if the print model
    changes.
    """
    from ._repo import _ensure_path
    _ensure_path()
    import torch
    from optimize.composition_prediction import physics_simulate_line

    rng = np.random.default_rng(seed)
    table: dict[str, dict] = {}
    for d in dims:
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
            # A coarse quantile sketch is enough to bootstrap from and keeps the
            # committed table small and readable.
            "quantiles": [float(q) for q in np.percentile(arr, np.arange(0, 101, 2))],
        }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"source": "optimize.composition_prediction.physics_simulate_line",
                   "n_lines_per_dim": n_lines, "seed": seed, "dims": table},
                  fh, indent=2)
    return table


def _load_calibration() -> dict:
    if not os.path.exists(_CALIB_PATH):
        raise FileNotFoundError(
            f"input-noise calibration table missing at {_CALIB_PATH}. "
            "Generate it with: python -m benchmarks.zhbench.protocol --calibrate")
    with open(_CALIB_PATH, encoding="utf-8") as fh:
        return json.load(fh)["dims"]


def _quantiles_for_dim(dim: int) -> np.ndarray:
    """Empirical magnitude quantiles for ``dim``, interpolating across the table."""
    table = _load_calibration()
    if str(dim) in table:
        return np.asarray(table[str(dim)]["quantiles"], dtype=float)
    keys = sorted(int(k) for k in table)
    below = [k for k in keys if k <= dim]
    above = [k for k in keys if k >= dim]
    lo = max(below) if below else keys[0]
    hi = min(above) if above else keys[-1]
    q_lo = np.asarray(table[str(lo)]["quantiles"], dtype=float)
    q_hi = np.asarray(table[str(hi)]["quantiles"], dtype=float)
    if lo == hi:
        return q_lo
    w = (dim - lo) / (hi - lo)
    return (1.0 - w) * q_lo + w * q_hi


def _zero_sum_unit(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` random unit directions tangent to the simplex (rows sum to zero)."""
    u = rng.standard_normal((n, dim))
    u -= u.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(u, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    return u / nrm


def realize(X_requested: np.ndarray, protocol: Protocol,
            rng: np.random.Generator) -> np.ndarray:
    """Turn requested compositions into the ones a printer would actually make.

    Point-wise, for scattered batches. Lines go through the core's physics model
    instead (see ``zombihop_runner``).
    """
    X = np.atleast_2d(np.asarray(X_requested, dtype=float))
    n, dim = X.shape
    mode = protocol.input_noise
    if mode == "none":
        return X.copy()
    if mode == "gaussian":
        std = protocol.input_noise_std
        if std is None:
            from ._repo import run_mobo
            std = float(run_mobo().NOISE_LEVEL)
        pert = rng.standard_normal((n, dim)) * float(std)
    elif mode == "empirical":
        q = _quantiles_for_dim(dim)
        mags = np.interp(rng.random(n), np.linspace(0.0, 1.0, q.size), q)
        pert = _zero_sum_unit(n, dim, rng) * mags[:, None]
    else:
        raise ValueError(f"unknown input_noise mode: {mode!r}")
    X_act = X + pert
    if protocol.domain == "cube":
        return np.clip(X_act, 0.0, 1.0)
    return project_simplex(X_act)


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
        return np.asarray([float(self.fn(x)) for x in X], dtype=float)

    def _add_output_noise(self, y: np.ndarray) -> np.ndarray:
        return y + self._rng.standard_normal(y.shape) * (self._out_frac * np.abs(y))

    def _check_budget(self) -> None:
        if self.n_samples >= self.protocol.n_samples:
            raise BudgetExhausted(self.n_samples)

    def evaluate_batch(self, X_requested: np.ndarray,
                       X_actual: np.ndarray | None = None
                       ) -> tuple[np.ndarray, np.ndarray]:
        """Realize, measure, record, and count a batch. The baseline entry point.

        ``X_actual`` may be supplied when the caller has already realized the
        request through a different physical path -- the shared initial design does
        this, because those batches really are printed LINES and go through
        ``physics_simulate_line`` for every method alike.
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


#: The print model has ten syringe modules and raises above that
#: (``composition_prediction.physics_simulate_line``: "physics model supports up
#: to 10 modules"). This is a HARDWARE limit, not a software one -- the SDL
#: cannot print an 11-component gradient -- so any run above 10 components is a
#: purely computational study with no printer to model. Runs at d > 10 record
#: ``line_realization: "no_printer_model"`` so this never gets lost in a plot.
MAX_PRINTABLE_COMPONENTS = 10


def line_realization_mode(dim: int, protocol: Protocol) -> str:
    if protocol.domain == "cube":
        return "clean_segment"
    if protocol.input_noise == "none":
        return "clean_segment"
    if dim > MAX_PRINTABLE_COMPONENTS:
        return "no_printer_model"
    return "physics"


def realize_line(left: np.ndarray, right: np.ndarray, n_points: int,
                 protocol: Protocol, rng: np.random.Generator | None = None
                 ) -> np.ndarray:
    """Compositions the printer would actually deposit for a requested chord.

    For d <= 10 this is the deterministic hardware model (ramp lag/overshoot +
    junction-volume diffusion mixing) the ZoMBI-Hop core uses for every line.
    Above 10 components no printer exists, so the chord gets the same calibrated
    point-wise perturbation the batch baselines get, extrapolated from d=10.
    """
    t = np.linspace(0.0, 1.0, n_points)[:, None]
    clean = left[None, :] + t * (right - left)[None, :]
    mode = line_realization_mode(left.shape[0], protocol)
    if mode == "clean_segment":
        return clean
    if mode == "no_printer_model":
        return realize(clean, protocol, rng or np.random.default_rng(0))
    from ._repo import _ensure_path
    _ensure_path()
    import torch
    from optimize.composition_prediction import physics_simulate_line
    act = physics_simulate_line(torch.as_tensor(left), torch.as_tensor(right),
                                num_points=n_points)
    return np.asarray(act.detach().cpu().numpy() if hasattr(act, "detach") else act,
                      dtype=float)


def gen_init_design(objective_run: ObjectiveRun, protocol: Protocol,
                    seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The shared initial design: ``n_init_lines`` random chords through the centroid.

    Mirrors the construction in ``optimize.evaluate.gen_init_data``, including
    realizing each chord through ``physics_simulate_line`` -- these batches really
    are printed lines, for every method alike. Routed through
    :class:`ObjectiveRun` so it counts against the budget, and driven by the run
    seed alone so it is byte-identical across methods at a given seed.
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

    ap = argparse.ArgumentParser(description="input-noise calibration")
    ap.add_argument("--calibrate", action="store_true",
                    help="regenerate the input-noise calibration table")
    ap.add_argument("--n-lines", type=int, default=200)
    ap.add_argument("--dims", type=int, nargs="+", default=[3, 4, 5, 6, 8, 10, 12])
    args = ap.parse_args()
    if args.calibrate:
        tbl = calibrate_input_noise(dims=tuple(args.dims), n_lines=args.n_lines)
        for d, v in tbl.items():
            print(f"d={d:>3}  n={v['n']:>6}  mean_l2={v['mean_l2']:.4f}  "
                  f"median={v['median_l2']:.4f}  p95={v['p95_l2']:.4f}")
        print(f"\nwritten to {_CALIB_PATH}")
