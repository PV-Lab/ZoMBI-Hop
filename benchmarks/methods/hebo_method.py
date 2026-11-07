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
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import numpy as np

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
    defaults = {"n_init": 48, "model_name": "gp", "es": "nsga2"}

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

    def ask(self, n: int) -> np.ndarray:
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

    def tell(self, X_requested, X_actual, Y) -> None:
        import pandas as pd

        self.opt.observe(pd.DataFrame(X_actual, columns=self.cols),
                         -np.asarray(Y, dtype=float).reshape(-1, 1))

    def summary(self) -> dict:
        return {"n_fallback_batches": int(getattr(self, "n_fallback", 0))}
