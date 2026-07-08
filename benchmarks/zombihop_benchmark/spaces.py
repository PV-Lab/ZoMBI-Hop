from __future__ import annotations

import math

import numpy as np


def _as_2d(X: np.ndarray | list[float] | list[list[float]], name: str = "X") -> tuple[np.ndarray, bool]:
    arr = np.asarray(X, dtype=float)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array, got shape {arr.shape}")
    if arr.shape[1] < 2:
        raise ValueError(f"{name} must have at least 2 simplex components")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr, was_1d


def project_simplex(X: np.ndarray) -> np.ndarray:
    """Project rows of X onto the probability simplex."""
    arr, was_1d = _as_2d(X)
    u = np.sort(arr, axis=1)[:, ::-1]
    cssv = np.cumsum(u, axis=1) - 1.0
    ind = np.arange(1, arr.shape[1] + 1)
    cond = u - cssv / ind > 0
    rho = cond.sum(axis=1) - 1
    theta = cssv[np.arange(arr.shape[0]), rho] / (rho + 1)
    projected = np.maximum(arr - theta[:, None], 0.0)
    projected /= projected.sum(axis=1, keepdims=True)
    return projected[0] if was_1d else projected


def sample_simplex(n: int, n_components: int, seed: int | None = None) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    if n_components < 2:
        raise ValueError("n_components must be at least 2")
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(n_components), size=n)


def validate_simplex(X: np.ndarray, atol: float = 1e-6) -> bool:
    arr, _ = _as_2d(X)
    if (arr < -atol).any():
        raise ValueError("simplex rows must be nonnegative")
    row_sums = arr.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError(f"simplex rows must sum to 1 within {atol}; got {row_sums}")
    return True


def composition_to_ilr_np(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Helmert ILR transform for row-wise compositions."""
    arr, was_1d = _as_2d(X)
    validate_simplex(arr, atol=1e-5)
    arr = np.clip(arr, eps, None)
    arr = arr / arr.sum(axis=1, keepdims=True)
    log_x = np.log(arr)
    n, d = arr.shape
    out = np.empty((n, d - 1), dtype=float)
    for i in range(d - 1):
        coef = math.sqrt((i + 1) / (i + 2))
        out[:, i] = coef * (log_x[:, : i + 1].mean(axis=1) - log_x[:, i + 1])
    return out[0] if was_1d else out


def ilr_to_composition_np(Z: np.ndarray, n_components: int, eps: float = 1e-12) -> np.ndarray:
    """Inverse Helmert ILR transform for row-wise coordinates."""
    arr = np.asarray(Z, dtype=float)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"Z must be a 1D or 2D array, got shape {arr.shape}")
    if n_components < 2:
        raise ValueError("n_components must be at least 2")
    if arr.shape[1] != n_components - 1:
        raise ValueError(
            f"ILR coordinate dimension must be n_components - 1; got {arr.shape[1]} for {n_components}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("Z contains NaN or infinite values")

    log_x = np.zeros((arr.shape[0], n_components), dtype=float)
    for i in range(n_components - 1):
        coef = math.sqrt((i + 1) / (i + 2))
        contribution = arr[:, i] * coef
        log_x[:, : i + 1] += contribution[:, None] / (i + 1)
        log_x[:, i + 1] -= contribution

    log_x -= np.max(log_x, axis=1, keepdims=True)
    x = np.exp(log_x)
    x = np.maximum(x, eps)
    x /= x.sum(axis=1, keepdims=True)
    return x[0] if was_1d else x


def ilr_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X_ilr = composition_to_ilr_np(X)
    Y_ilr = composition_to_ilr_np(Y)
    if X_ilr.ndim == 1:
        X_ilr = X_ilr.reshape(1, -1)
    if Y_ilr.ndim == 1:
        Y_ilr = Y_ilr.reshape(1, -1)
    if X_ilr.shape[1] != Y_ilr.shape[1]:
        raise ValueError("X and Y must have the same number of simplex components")
    diff = X_ilr[:, None, :] - Y_ilr[None, :, :]
    return np.linalg.norm(diff, axis=2)

