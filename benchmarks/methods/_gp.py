"""
benchmarks/methods/_gp.py
=========================
The one exact-GP recipe the needle extractor and TuRBO share: constant mean,
``Scale(Matern-5/2, ARD)`` covariance, Gaussian likelihood, fitted by maximum
marginal likelihood on standardised outputs. (Standard BO deliberately uses
BoTorch's stock ``SingleTaskGP`` instead — see ``gp_bo.py``.)

Outputs are standardised by hand rather than with an outcome transform, so a model
fitted on a subset can hand its hyperparameters to a model conditioned on all the
data with a plain ``load_state_dict``: the two are then built from identical
modules and every parameter lines up.
"""

from __future__ import annotations

import numpy as np


def tensor(a, device: str):
    import torch

    return torch.as_tensor(np.asarray(a, dtype=float), dtype=torch.float64,
                           device=torch.device(device))


def _build(X, Y, *, lengthscale_bounds: tuple[float, float] | None,
           noise_lb: float):
    from botorch.models import SingleTaskGP
    from gpytorch.constraints import GreaterThan, Interval
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood

    d = X.shape[-1]
    ls_con = Interval(*lengthscale_bounds) if lengthscale_bounds else None
    covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d,
                                     lengthscale_constraint=ls_con))
    lik = GaussianLikelihood(noise_constraint=GreaterThan(noise_lb))
    return SingleTaskGP(X, Y, covar_module=covar, likelihood=lik,
                        outcome_transform=None)


def fit_matern_gp(X: np.ndarray, Y: np.ndarray, *, device: str = "cpu",
                  lengthscale_bounds: tuple[float, float] | None = (0.005, 4.0),
                  noise_lb: float = 1e-6, fit_idx: np.ndarray | None = None):
    """Fit on ``X[fit_idx]`` (all rows by default), condition on all of ``X``.

    Returns ``(model, y_mean, y_std)``; the model predicts STANDARDISED values,
    ``(y - y_mean) / y_std``. A failed hyperparameter fit (a non-PSD Cholesky on a
    degenerate batch, say) keeps the initial hyperparameters rather than raising:
    callers are optimisers mid-run, and a slightly worse model beats a dead cell.
    """
    import torch
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    Y = np.asarray(Y, dtype=float).reshape(-1)
    y_mean = float(Y.mean())
    y_std = float(Y.std()) or 1.0
    Xt = tensor(X, device)
    Yt = tensor((Y - y_mean) / y_std, device).unsqueeze(-1)

    idx = None if fit_idx is None else torch.as_tensor(np.asarray(fit_idx), device=Xt.device)
    Xf, Yf = (Xt, Yt) if idx is None else (Xt[idx], Yt[idx])
    model = _build(Xf, Yf, lengthscale_bounds=lengthscale_bounds, noise_lb=noise_lb)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        fit_gpytorch_mll(mll)
    except Exception as exc:  # noqa: BLE001 — see docstring
        print(f"    [gp] hyperparameter fit failed ({type(exc).__name__}: {exc}); "
              "keeping initial hyperparameters", flush=True)
    if idx is not None:
        full = _build(Xt, Yt, lengthscale_bounds=lengthscale_bounds, noise_lb=noise_lb)
        full.load_state_dict(model.state_dict())
        model = full
    model.eval()
    return model, y_mean, y_std


def posterior_mean_sd(model, X: np.ndarray | "torch.Tensor", *, device: str = "cpu",
                      chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Latent posterior mean and sd (standardised units), chunked to bound memory."""
    import torch

    Xt = X if hasattr(X, "detach") else tensor(X, device)
    mus, sds = [], []
    with torch.no_grad():
        for i in range(0, Xt.shape[0], chunk):
            post = model.posterior(Xt[i:i + chunk])
            mus.append(post.mean.reshape(-1))
            sds.append(post.variance.clamp_min(0.0).sqrt().reshape(-1))
    if not mus:
        return np.empty(0), np.empty(0)
    return (torch.cat(mus).cpu().numpy(), torch.cat(sds).cpu().numpy())
