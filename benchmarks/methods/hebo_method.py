"""
benchmarks/methods/hebo_method.py
=================================
HEBO — Heteroscedastic Evolutionary Bayesian Optimisation (Cowen-Rivers et al.,
JAIR 2022, arXiv:2012.03826), run through the authors' own package (``hebo`` 0.3.6,
huawei-noah/HEBO) rather than a re-implementation: input warping, output power
transforms, and the MACE multi-objective acquisition ensemble (EI / PI / UCB traded
off on an NSGA-II Pareto front) are all the reference code.

Installation — why it is vendored
---------------------------------
``pip install HEBO`` resolves to pins that would DOWNGRADE this repo's shared
environment (numpy 2.4 -> 1.24, scipy, pandas, matplotlib) — the environment the
MOBO fleet runs from. So HEBO is installed ``--no-deps`` into a private directory,
``benchmarks/methods/_vendor/`` (git-ignored), together with the two small pure
packages it needs that the environment lacks (``pymoo`` 0.6.1.3, ``disjoint-set``,
``alive-progress``). HEBO 0.3.6 runs correctly on numpy 2 in that arrangement.
pymoo's compiled extensions are deleted after install: they need a newer
``libstdc++`` (GLIBCXX_3.4.29) than the cluster's, and pymoo falls back to its pure
Python implementations without them. Nothing in ``.venv`` changes.

    python -m benchmarks.methods install-hebo

HEBO minimises; the adapter hands it ``-Y``. It runs on the CPU (the package
manages its own torch tensors), whatever ``--device`` says.

Config
------
n_init       random (Sobol') design before the model is used; default 48, two
             batches, the same as every other method here
model_name   HEBO surrogate; "gp" (default, the paper's)
es           acquisition optimiser; "nsga2" (default, the paper's)
sampling     "point" (default) or "line" (``benchmarks/sweeps --sampling line``; see
             ``benchmarks/sweeps/LINEWISE.md``). Line mode starts from the
             ``n_init_lines`` chords every line-mode method shares, then each step
             runs HEBO's own ``suggest`` preamble unchanged (power transform, GP
             fit, best_y, kappa — with ``n_suggestions`` = points per line, so a
             line is one HEBO iteration) to build the MACE acquisition, draws
             ``linebo_num_lines`` random chords through the box, averages each MACE
             objective along every chord (LCB directly; EI and PI as EI and PI, not
             as the logs MACE returns), keeps the non-dominated chords, and measures
             one of them chosen at random — as ``suggest`` picks at random from its
             NSGA-II front. The evolutionary optimiser is not used in line mode.
n_init_lines, linebo_num_lines, linebo_points_per_line
             line-mode settings (2, 10, 100: zombi_hop's)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import numpy as np

from . import _lines
from .base import AskTellMethod, Problem
from .registry import register

VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor")
#: Exactly what ``install_hebo`` puts in :data:`VENDOR_DIR`.
VENDOR_PACKAGES = ("hebo==0.3.6", "pymoo==0.6.1.3", "disjoint-set==0.9.0",
                   "alive-progress==3.3.0")


def install_hebo(target: str = VENDOR_DIR) -> None:
    """Install HEBO and its missing pure dependencies into ``target``, no-deps."""
    os.makedirs(target, exist_ok=True)
    uv = shutil.which("uv")
    if uv:
        cmd = [uv, "pip", "install", "--python", sys.executable, "--link-mode=copy",
               "--target", target, "--no-deps", *VENDOR_PACKAGES]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--target", target,
               "--no-deps", *VENDOR_PACKAGES]
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    cy = os.path.join(target, "pymoo", "cython")
    removed = 0
    if os.path.isdir(cy):
        for fn in os.listdir(cy):
            if fn.endswith(".so") or fn.endswith(".pyd"):
                os.remove(os.path.join(cy, fn))
                removed += 1
    print(f"  HEBO installed to {target} (removed {removed} compiled pymoo "
          "extension(s); pymoo falls back to pure Python)")


def _import_hebo():
    if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
        sys.path.insert(0, VENDOR_DIR)
    try:
        # The compiled extensions were removed on purpose (see the module
        # docstring); without this pymoo prints a banner about it on every run.
        from pymoo.config import Config

        Config.warnings["not_compiled"] = False
        from hebo.design_space.design_space import DesignSpace
        from hebo.optimizers.hebo import HEBO
    except ImportError as exc:
        raise ImportError(
            f"HEBO is not importable ({exc}). Install it into the private vendor "
            "directory with:  python -m benchmarks.methods install-hebo") from exc
    return DesignSpace, HEBO


@register
class HEBOMethod(AskTellMethod):
    name = "hebo"
    description = "HEBO 0.3.6 (Cowen-Rivers et al. 2022), reference implementation"
    defaults = {"n_init": 48, "model_name": "gp", "es": "nsga2", **_lines.LINE_DEFAULTS}

    def setup(self, problem: Problem) -> None:
        DesignSpace, HEBO = _import_hebo()
        import pandas as pd  # noqa: F401 — HEBO speaks DataFrames

        self.dim = problem.dim
        self.cols = [f"x{i}" for i in range(self.dim)]
        space = DesignSpace().parse([{"name": c, "type": "num", "lb": 0.0, "ub": 1.0}
                                     for c in self.cols])
        self.opt = HEBO(space, model_name=self.config["model_name"],
                        rand_sample=int(self.config["n_init"]),
                        es=self.config["es"], scramble_seed=self.seed)
        self.n_fallback = 0
        self.line_mode = _lines.check_sampling(self.name, self.config, problem.batch_size)
        if self.line_mode:
            self.B = int(problem.batch_size)
            self._queue = _lines.LineQueue()
            for left, right in _lines.init_chords(self.rng, int(self.config["n_init_lines"]),
                                                  self.dim):
                self._queue.push_line(left, right, self.B)
            self.front_sizes: list[int] = []

    def ask(self, n: int) -> np.ndarray:
        if self.line_mode:
            if not len(self._queue):
                try:
                    left, right = self._mace_line()
                except Exception as exc:  # noqa: BLE001 — a failed step must not end the run
                    self.n_fallback += 1
                    print(f"    [hebo] MACE line failed ({type(exc).__name__}: {exc}); "
                          "measuring a random chord instead", flush=True)
                    (left,), (right,) = _lines.random_chords(
                        self.rng, 1, np.zeros(self.dim), np.ones(self.dim))
                self._queue.push_line(left, right, self.B)
            return self._queue.pop(n)
        try:
            rec = self.opt.suggest(n_suggestions=n)
            X = rec[self.cols].to_numpy(dtype=float)
            if X.shape != (n, self.dim) or not np.isfinite(X).all():
                raise ValueError(f"HEBO suggested {X.shape}, wanted {(n, self.dim)}")
            return np.clip(X, 0.0, 1.0)
        except Exception as exc:  # noqa: BLE001 — a failed step must not end the run
            self.n_fallback += 1
            print(f"    [hebo] suggest failed ({type(exc).__name__}: {exc}); "
                  "measuring a random batch instead", flush=True)
            return self.rng.random((n, self.dim))

    def _mace(self):
        """HEBO.suggest's model + MACE construction (hebo 0.3.6, optimizers/hebo.py
        lines 126-162), verbatim but for ``n_suggestions`` = points per line."""
        import torch
        from hebo.acquisitions.acq import MACE
        from hebo.models.model_factory import get_model
        from sklearn.preprocessing import power_transform

        o = self.opt
        X, Xe = o.space.transform(o.X)
        try:
            if o.y.min() <= 0:
                y = torch.FloatTensor(power_transform(o.y / o.y.std(), method="yeo-johnson"))
            else:
                y = torch.FloatTensor(power_transform(o.y / o.y.std(), method="box-cox"))
                if y.std() < 0.5:
                    y = torch.FloatTensor(power_transform(o.y / o.y.std(), method="yeo-johnson"))
            if y.std() < 0.5:
                raise RuntimeError("Power transformation failed")
            model = get_model(o.model_name, o.space.num_numeric, o.space.num_categorical, 1,
                              **o.model_config)
            model.fit(X, Xe, y)
        except Exception:  # noqa: BLE001 — HEBO's own bare-except fallback
            y = torch.FloatTensor(o.y).clone()
            model = get_model(o.model_name, o.space.num_numeric, o.space.num_categorical, 1,
                              **o.model_config)
            model.fit(X, Xe, y)

        best_x = o.X.iloc[[o.get_best_id()]]
        py_best, _ = model.predict(*o.space.transform(best_x))
        py_best = py_best.detach().numpy().squeeze()
        it = max(1, o.X.shape[0] // self.B)
        upsi, delta = 0.5, 0.01
        kappa = np.sqrt(upsi * 2 * ((2.0 + o.X.shape[1] / 2.0) * np.log(it)
                                    + np.log(3 * np.pi ** 2 / (3 * delta))))
        return MACE(model, best_y=py_best, kappa=kappa)

    def _mace_line(self) -> tuple[np.ndarray, np.ndarray]:
        """A random non-dominated chord under the chord-averaged MACE objectives."""
        import pandas as pd
        import torch

        acq = self._mace()
        k = int(self.config["linebo_num_lines"])
        P = int(self.config["linebo_points_per_line"])
        left, right = _lines.random_chords(self.rng, k, np.zeros(self.dim), np.ones(self.dim))
        pts = _lines.line_points(left, right, P).reshape(-1, self.dim)
        out = acq(*self.opt.space.transform(pd.DataFrame(pts, columns=self.cols)))
        out = out.detach().double().reshape(k, P, 3)          # (lcb, -log EI, -log PI)
        logP = torch.log(torch.tensor(float(P), dtype=out.dtype))
        F = torch.stack([out[..., 0].mean(dim=1),                           # mean LCB
                         -(torch.logsumexp(-out[..., 1], dim=1) - logP),    # -log mean EI
                         -(torch.logsumexp(-out[..., 2], dim=1) - logP)],   # -log mean PI
                        dim=1).numpy()
        F = np.where(np.isfinite(F), F, np.inf)
        front = _lines.non_dominated(F)
        if len(front) == 0:
            front = np.arange(k)
        self.front_sizes.append(int(len(front)))
        j = int(self.rng.choice(front))
        return left[j], right[j]

    def tell(self, X_requested, X_actual, Y) -> None:
        import pandas as pd

        self.opt.observe(pd.DataFrame(X_actual, columns=self.cols),
                         -np.asarray(Y, dtype=float).reshape(-1, 1))

    def summary(self) -> dict:
        out = {"n_fallback_batches": int(getattr(self, "n_fallback", 0)),
               "sampling": self.config["sampling"]}
        if getattr(self, "front_sizes", None):
            out["mean_pareto_chords"] = float(np.mean(self.front_sizes))
        return out
