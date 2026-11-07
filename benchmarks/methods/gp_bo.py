"""
benchmarks/methods/gp_bo.py
===========================
Standard Bayesian optimisation: one global GP, one acquisition function, batches
chosen by sequential greedy optimisation.

"Standard" is taken literally — BoTorch's stock ``SingleTaskGP`` with its default
priors (dimension-scaled lognormal lengthscale prior, Hvarfner et al. 2024, which is
what makes an untuned GP behave at d = 10) and default ``Standardize`` outcome
transform, and batch log-noisy-EI (Ament et al. 2023), the acquisition BoTorch
recommends for noisy problems. Nothing is tuned to the needle landscapes, so this is
the method a practitioner reaching for "BO" would run.

Batches have ``q = problem.batch_size`` points, optimised one at a time conditioned
on the pending ones (``sequential=True``) — joint optimisation of a 24-point batch in
10 dimensions is a 240-dimensional problem and not what anyone runs in practice.

Config
------
n_init          initial Sobol' design size (default 48: two batches, the same as
                ZoMBI-Hop's two initial lines)
acquisition     "qlognei" (default) | "qlogei" | "qucb"
ucb_beta        exploration weight for "qucb"
num_restarts, raw_samples
                multi-start acquisition optimisation (BoTorch defaults for q-batch)
max_train_points
                cap on GP training points; the best half and a random half are
                kept above it. None (default) = every point; exact GP to 3000
                points is fine on a GPU.
"""

from __future__ import annotations

import numpy as np

from .base import AskTellMethod, Problem, sobol_design
from .registry import register


@register
class GPBO(AskTellMethod):
    name = "gp_bo"
    description = "standard BO: BoTorch SingleTaskGP + batch qLogNEI"
    defaults = {
        "n_init": 48,
        "acquisition": "qlognei",
        "ucb_beta": 2.0,
        "num_restarts": 10,
        "raw_samples": 512,
        "max_train_points": None,
    }

    def setup(self, problem: Problem) -> None:
        import torch

        if self.config["acquisition"] not in ("qlognei", "qlogei", "qucb"):
            raise ValueError(f"gp_bo: unknown acquisition {self.config['acquisition']!r}")
        self.dim = problem.dim
        self.tdev = torch.device(self.device)
        self.bounds_t = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]).to(
            dtype=torch.float64, device=self.tdev)
        self.X = np.empty((0, self.dim))
        self.Y = np.empty(0)
        self._init = sobol_design(int(self.config["n_init"]), self.dim, self.seed)
        self._init_used = 0
        self.n_fallback = 0

    def _train_set(self) -> tuple[np.ndarray, np.ndarray]:
        cap = self.config["max_train_points"]
        if cap is None or len(self.Y) <= int(cap):
            return self.X, self.Y
        half = int(cap) // 2
        top = np.argsort(-self.Y)[:half]
        rest = np.setdiff1d(np.arange(len(self.Y)), top)
        idx = np.concatenate([top, self.rng.choice(rest, int(cap) - half, replace=False)])
        return self.X[idx], self.Y[idx]

    def ask(self, n: int) -> np.ndarray:
        if self._init_used < len(self._init):
            out = self._init[self._init_used:self._init_used + n]
            self._init_used += len(out)
            if len(out) == n:
                return out
            return np.vstack([out, self.rng.random((n - len(out), self.dim))])
        try:
            return self._bo_batch(n)
        except Exception as exc:  # noqa: BLE001 — a failed step must not end the run
            self.n_fallback += 1
            print(f"    [gp_bo] BO step failed ({type(exc).__name__}: {exc}); "
                  "measuring a random batch instead", flush=True)
            return self.rng.random((n, self.dim))

    def _bo_batch(self, n: int) -> np.ndarray:
        import torch
        from botorch.acquisition import qUpperConfidenceBound
        from botorch.acquisition.logei import (
            qLogExpectedImprovement,
            qLogNoisyExpectedImprovement,
        )
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood

        Xn, Yn = self._train_set()
        X = torch.as_tensor(Xn, dtype=torch.float64, device=self.tdev)
        Y = torch.as_tensor(Yn, dtype=torch.float64, device=self.tdev).unsqueeze(-1)
        model = SingleTaskGP(X, Y)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

        acq_name = self.config["acquisition"]
        if acq_name == "qlognei":
            acq = qLogNoisyExpectedImprovement(model, X_baseline=X, prune_baseline=True)
        elif acq_name == "qlogei":
            with torch.no_grad():
                best_f = model.posterior(X).mean.max()
            acq = qLogExpectedImprovement(model, best_f=best_f)
        else:
            acq = qUpperConfidenceBound(model, beta=float(self.config["ucb_beta"]))
        cand, _ = optimize_acqf(acq, bounds=self.bounds_t, q=n,
                                num_restarts=int(self.config["num_restarts"]),
                                raw_samples=int(self.config["raw_samples"]),
                                sequential=True)
        return cand.detach().cpu().numpy()

    def tell(self, X_requested, X_actual, Y) -> None:
        self.X = np.vstack([self.X, X_actual])
        self.Y = np.concatenate([self.Y, Y])

    def summary(self) -> dict:
        return {"n_fallback_batches": int(getattr(self, "n_fallback", 0))}
