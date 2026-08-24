"""BoTorch GP baselines with real joint q-batch acquisitions.

Model: ``SingleTaskGP`` with an ARD Matern kernel on isometric-log-ratio (ILR)
coordinates, so the GP works in the (d-1)-dimensional space the simplex actually
occupies rather than pretending the compositions are free in R^d.

Batching: sequential greedy maximisation of ``qLogEI`` / ``qUCB`` over a finite
pool of valid compositions, conditioning each pick on the ones already in the
batch via ``X_pending``. This is the standard BoTorch recipe for a discrete
candidate set, and it is what makes the batch a real batch. The previous benchmark
took greedy top-k of a single-point analytic acquisition, which returns q nearly
identical points -- that would hand ZoMBI-Hop an unearned win.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..spaces import composition_to_ilr_np
from .base import BaseOptimizer


class GPBatch(BaseOptimizer):
    """``kind`` is "logei" or "ucb"."""

    def __init__(self, kind: str = "ucb", pool_size: int = 2048, ucb_beta: float = 2.0,
                 max_train_points: int = 1024, device: str = "cpu",
                 mc_samples: int = 128, **kwargs):
        super().__init__(**kwargs)
        if kind not in ("logei", "ucb"):
            raise ValueError("kind must be 'logei' or 'ucb'")
        self.kind = kind
        self.name = f"gp_q{kind}"
        self.pool_size = int(pool_size)
        self.ucb_beta = float(ucb_beta)
        self.max_train_points = int(max_train_points)
        self.device = device
        self.mc_samples = int(mc_samples)

    # -- coordinates ----------------------------------------------------------
    def _to_model_space(self, X: np.ndarray) -> np.ndarray:
        if self.domain == "cube":
            return np.asarray(X, dtype=float)
        return composition_to_ilr_np(np.asarray(X, dtype=float))

    def _fit(self, torch):
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms import Normalize, Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood

        X, y = self.X, self.y
        if X.shape[0] > self.max_train_points:
            # Keep the best half and the most recent half: a plain tail window
            # throws away the peaks the model most needs.
            n = self.max_train_points
            best = np.argsort(-y)[: n // 2]
            recent = np.arange(X.shape[0])[-(n - best.size):]
            keep = np.unique(np.concatenate([best, recent]))
            X, y = X[keep], y[keep]

        Z = self._to_model_space(X)
        tX = torch.as_tensor(Z, dtype=torch.double)
        ty = torch.as_tensor(y, dtype=torch.double).reshape(-1, 1)
        model = SingleTaskGP(
            tX, ty,
            input_transform=Normalize(d=tX.shape[-1]),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll)
        return model

    def _acq(self, torch, model, X_pending):
        from botorch.acquisition.logei import qLogExpectedImprovement
        from botorch.acquisition.monte_carlo import qUpperConfidenceBound
        from botorch.sampling.normal import SobolQMCNormalSampler

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_samples]),
                                        seed=self.seed + self._n_suggest)
        if self.kind == "ucb":
            return qUpperConfidenceBound(model, beta=self.ucb_beta, sampler=sampler,
                                         X_pending=X_pending)
        best_f = torch.as_tensor(float(np.max(self.y)), dtype=torch.double)
        return qLogExpectedImprovement(model, best_f=best_f, sampler=sampler,
                                       X_pending=X_pending)

    # -- the loop -------------------------------------------------------------
    def suggest(self, q: int) -> np.ndarray:
        import torch

        q = int(q)
        rng = np.random.default_rng(self.seed + 2_000_003 + self._n_suggest)
        pool = self._sample_domain(self.pool_size, rng=rng)
        model = self._fit(torch)
        Zpool = torch.as_tensor(self._to_model_space(pool), dtype=torch.double)

        chosen: list[int] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Build the acquisition once and update X_pending in place. Rebuilding
            # it per greedy step re-creates the QMC sampler 24 times per decision
            # for no benefit, and a fresh sampler each step would also make the
            # greedy sequence noisier than it needs to be.
            acq = self._acq(torch, model, None)
            cand = Zpool.unsqueeze(1)
            for _ in range(q):
                if chosen:
                    acq.set_X_pending(Zpool[chosen])
                with torch.no_grad():
                    vals = acq(cand).cpu().numpy()
                vals[chosen] = -np.inf
                chosen.append(int(np.argmax(vals)))
        self._n_suggest += 1
        return pool[np.asarray(chosen, dtype=int)]


class GPThompson(GPBatch):
    """Batch Thompson sampling over the same GP and candidate pool.

    qUCB and qLogEI are greedy by construction: they concentrate a batch where the
    posterior mean is already high, so "standard BO recovers fewer optima than
    ZoMBI-Hop" risks being read as "qUCB is greedy" rather than as a statement
    about BO. Thompson sampling draws each batch member from a different posterior
    realisation, so it spreads across modes the posterior thinks are plausible --
    the cheapest genuinely diversity-friendly BO baseline available. TuRBO and
    ROBOT cover the same objection more thoroughly later.

    ``replacement=False`` so a batch cannot be 24 copies of one point.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("kind", "ucb")     # unused; TS has no acquisition
        super().__init__(**kwargs)
        self.name = "gp_ts"

    def suggest(self, q: int) -> np.ndarray:
        import torch
        from botorch.generation import MaxPosteriorSampling

        q = int(q)
        rng = np.random.default_rng(self.seed + 5_000_011 + self._n_suggest)
        pool = self._sample_domain(self.pool_size, rng=rng)
        model = self._fit(torch)
        Zpool = torch.as_tensor(self._to_model_space(pool), dtype=torch.double)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.manual_seed(self.seed + self._n_suggest)
            sampler = MaxPosteriorSampling(model=model, replacement=False)
            chosen = sampler(Zpool, num_samples=q)
        self._n_suggest += 1
        # Map the chosen model-space rows back to their pool compositions.
        idx = torch.cdist(chosen, Zpool).argmin(dim=1).cpu().numpy()
        return pool[idx]
