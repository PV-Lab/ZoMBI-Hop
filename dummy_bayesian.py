#!/usr/bin/env python3
"""
Visual demo of Bayesian optimization on a random Taylor polynomial over [0, 1].

Shows the true objective (hidden in real BO, revealed here for teaching), Gaussian
process posterior mean, uncertainty bands, observed samples, and UCB acquisition.
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

RNG = np.random.default_rng()


@dataclass
class BOState:
    x_obs: list[float] = field(default_factory=list)
    y_obs: list[float] = field(default_factory=list)
    x_cand: float | None = None
    iteration: int = 0


def random_taylor_coeffs(degree: int, seed: int | None = None) -> np.ndarray:
    """
    Coefficients for a Taylor expansion about x=0.5 so the curve can wiggle on [0, 1]:
        f(x) = sum_k c_k * (x - 0.5)^k / k!
    Mid-to-high order terms get a boosted, alternating envelope for expressiveness.
    """
    gen = np.random.default_rng(seed)
    k = np.arange(degree + 1, dtype=float)
    # Bell envelope peaked in the mid degrees → several bumps, not a flat parabola.
    peak = degree * 0.45
    envelope = np.exp(-0.5 * ((k - peak) / max(2.0, degree * 0.22)) ** 2)
    envelope[0] *= 0.35
    signs = np.where(k % 2 == 0, 1.0, -1.0)
    raw = gen.normal(0, 1.2, size=degree + 1) * envelope * signs
    raw += gen.normal(0, 0.35, size=degree + 1) * (1.0 / (1.0 + k))
    return raw


def taylor(x: np.ndarray, coeffs: np.ndarray, center: float = 0.5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    u = x - center
    out = np.zeros_like(x, dtype=float)
    for k, c in enumerate(coeffs):
        out += float(c) * np.power(u, k) / float(math.factorial(k))
    return out


@dataclass(frozen=True)
class Objective:
    """Taylor polynomial on [0, 1] with fixed affine scaling from the full grid."""

    coeffs: np.ndarray
    y_min: float
    y_span: float

    @classmethod
    def from_coeffs(cls, coeffs: np.ndarray, x_grid: np.ndarray) -> Objective:
        raw = taylor(x_grid, coeffs)
        y_min = float(raw.min())
        y_span = float(raw.max() - raw.min())
        if y_span < 1e-9:
            y_span = 1.0
        return cls(coeffs=coeffs, y_min=y_min, y_span=y_span)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raw = taylor(np.asarray(x, dtype=float), self.coeffs)
        return 4.0 * (raw - self.y_min) / self.y_span - 2.0


def fit_gp(x_obs: np.ndarray, y_obs: np.ndarray) -> GaussianProcessRegressor:
    kernel = ConstantKernel(1.0, (1e-2, 10.0)) * RBF(length_scale=0.08, length_scale_bounds=(0.02, 0.5))
    kernel += WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-6, 1e-1))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=3,
        normalize_y=True,
        random_state=0,
    )
    gp.fit(x_obs.reshape(-1, 1), y_obs)
    return gp


def ucb(gp: GaussianProcessRegressor, x_grid: np.ndarray, kappa: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, std = gp.predict(x_grid.reshape(-1, 1), return_std=True)
    return mu, std, mu + kappa * std


def propose_next(
    gp: GaussianProcessRegressor,
    x_grid: np.ndarray,
    kappa: float,
    x_obs: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu, std, acq = ucb(gp, x_grid, kappa)
    # Avoid re-sampling already observed points (within grid resolution).
    for xo in x_obs:
        acq[np.isclose(x_grid, xo)] = -np.inf
    idx = int(np.argmax(acq))
    return float(x_grid[idx]), mu, std, acq, acq[idx]


def run_bo(
    objective: Objective,
    n_iters: int,
    n_init: int,
    kappa: float,
    x_grid: np.ndarray,
    seed: int,
) -> tuple[list[BOState], float, float]:
    gen = np.random.default_rng(seed)
    true_y = objective(x_grid)
    true_max_idx = int(np.argmax(true_y))
    true_x_max = float(x_grid[true_max_idx])
    true_y_max = float(true_y[true_max_idx])

    state = BOState()
    history: list[BOState] = []

    x_init = gen.uniform(0.05, 0.95, size=n_init)
    for x in sorted(x_init):
        state.x_obs.append(float(x))
        state.y_obs.append(float(objective(np.array([x]))[0]))
    gp = fit_gp(np.array(state.x_obs), np.array(state.y_obs))
    x_next, _, _, _, _ = propose_next(gp, x_grid, kappa, np.array(state.x_obs))
    history.append(
        BOState(
            x_obs=state.x_obs.copy(),
            y_obs=state.y_obs.copy(),
            x_cand=x_next,
            iteration=0,
        )
    )

    for it in range(1, n_iters + 1):
        x_obs = np.array(state.x_obs)
        y_obs = np.array(state.y_obs)
        gp = fit_gp(x_obs, y_obs)
        x_next, _, _, _, _ = propose_next(gp, x_grid, kappa, x_obs)
        state.x_obs.append(x_next)
        state.y_obs.append(float(objective(np.array([x_next]))[0]))
        gp = fit_gp(np.array(state.x_obs), np.array(state.y_obs))
        x_after, _, _, _, _ = propose_next(gp, x_grid, kappa, np.array(state.x_obs))
        history.append(
            BOState(
                x_obs=state.x_obs.copy(),
                y_obs=state.y_obs.copy(),
                x_cand=x_after,
                iteration=it,
            )
        )

    return history, true_x_max, true_y_max


def draw_frame(
    ax_obj,
    ax_acq,
    ax_meta,
    objective: Objective,
    x_grid: np.ndarray,
    snap: BOState,
    kappa: float,
    true_x_max: float,
    true_y_max: float,
    step_label: str,
):
    true_y = objective(x_grid)
    coeffs = objective.coeffs
    x_obs = np.array(snap.x_obs)
    y_obs = np.array(snap.y_obs)

    ax_obj.clear()
    ax_acq.clear()
    ax_meta.clear()
    ax_meta.axis("off")

    ax_obj.plot(x_grid, true_y, color="#2c3e50", lw=2, alpha=0.35, label="True objective (demo only)")
    ax_obj.axvline(true_x_max, color="#2c3e50", ls=":", lw=1.2, alpha=0.5)
    ax_obj.scatter([true_x_max], [true_y_max], s=80, marker="*", c="#2c3e50", zorder=2, label="True maximum")

    if len(x_obs) >= 2:
        gp = fit_gp(x_obs, y_obs)
        mu, std = gp.predict(x_grid.reshape(-1, 1), return_std=True)
        lo = mu - 2 * std
        hi = mu + 2 * std

        ax_obj.fill_between(x_grid, lo, hi, color="#3498db", alpha=0.25, label="GP 95% band (±2σ)")
        ax_obj.plot(x_grid, mu, color="#2980b9", lw=2, label="GP posterior mean")
        ax_obj.plot(x_grid, lo, color="#2980b9", lw=0.8, ls="--", alpha=0.7)
        ax_obj.plot(x_grid, hi, color="#2980b9", lw=0.8, ls="--", alpha=0.7)

        _, _, acq = ucb(gp, x_grid, kappa)
        for xo in x_obs:
            acq[np.isclose(x_grid, xo)] = np.nan
        ax_acq.plot(x_grid, acq, color="#e67e22", lw=2)
        ax_acq.set_ylabel(f"UCB (κ={kappa})")
        ax_acq.set_title("Acquisition — where to sample next")
        ax_acq.grid(True, alpha=0.3)

        if snap.x_cand is not None:
            y_cand_mu = float(gp.predict([[snap.x_cand]])[0])
            ax_obj.axvline(snap.x_cand, color="#e74c3c", ls="--", lw=1.5, alpha=0.9)
            ax_obj.scatter([snap.x_cand], [y_cand_mu], s=120, facecolors="none", edgecolors="#e74c3c", lw=2, zorder=6)
            acq_val = float(mu[np.argmin(np.abs(x_grid - snap.x_cand))] + kappa * std[np.argmin(np.abs(x_grid - snap.x_cand))])
            ax_acq.axvline(snap.x_cand, color="#e74c3c", ls="--", lw=1.5)
            ax_acq.scatter([snap.x_cand], [acq_val], s=120, c="#e74c3c", zorder=5)

    ax_obj.scatter(x_obs, y_obs, s=55, c="#27ae60", edgecolors="white", lw=0.8, zorder=5, label="Observed samples")
    best_idx = int(np.argmax(y_obs))
    ax_obj.scatter([x_obs[best_idx]], [y_obs[best_idx]], s=140, marker="D", c="#8e44ad", zorder=6, label="Best observed so far")

    ax_obj.set_xlim(0, 1)
    ax_obj.set_xlabel("x")
    ax_obj.set_ylabel("f(x)")
    ax_obj.set_title(f"Bayesian optimization — {step_label}")
    ax_obj.legend(loc="upper left", fontsize=8)
    ax_obj.grid(True, alpha=0.3)

    ax_acq.set_xlim(0, 1)
    ax_acq.set_xlabel("x")

    poly_str = " + ".join(
        f"{c:.2f}·(x-0.5)^{k}/{k}!" for k, c in enumerate(coeffs[: min(5, len(coeffs))])
    )
    if len(coeffs) > 5:
        poly_str += " + …"
    lines = [
        "Principle:",
        "  1. Fit a GP to noisy-free observations → mean μ(x) and uncertainty σ(x).",
        "  2. Acquisition = μ + κσ (UCB): exploit high μ, explore high σ.",
        "  3. Evaluate true f at argmax acquisition; repeat.",
        "",
        f"Random Taylor objective: f(x) ≈ {poly_str}",
        f"Samples: {len(x_obs)}   Best seen: {y_obs[best_idx]:.4f}   True max: {true_y_max:.4f}",
    ]
    if snap.x_cand is not None:
        lines.append(f"Next proposed x (red): {snap.x_cand:.4f}")
    ax_meta.text(0, 1, "\n".join(lines), va="top", fontsize=9, family="monospace")


def make_animation(
    history, objective: Objective, x_grid, kappa, true_x_max, true_y_max, out_path: str | None, interval_ms: int
):
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.2, 1.1], hspace=0.35)
    ax_obj = fig.add_subplot(gs[0])
    ax_acq = fig.add_subplot(gs[1])
    ax_meta = fig.add_subplot(gs[2])

    def update(i):
        snap = history[i]
        if snap.iteration == 0:
            label = "initial random samples"
        else:
            label = f"iteration {snap.iteration} / {history[-1].iteration}"
        draw_frame(ax_obj, ax_acq, ax_meta, objective, x_grid, snap, kappa, true_x_max, true_y_max, label)
        fig.suptitle(
            f"Live BO demo — one iteration every {interval_ms / 1000:.1f}s",
            fontsize=11,
            y=0.995,
        )

    update(0)
    anim = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=interval_ms,
        repeat=True,
        blit=False,
    )
    if out_path:
        anim.save(out_path, writer=PillowWriter(fps=max(1, 1000 // interval_ms)))
        print(f"Saved animation to {out_path}")
    else:
        print(f"Opening live window ({interval_ms / 1000:.1f}s per iteration). Close the window to exit.")
        plt.show(block=True)


def make_snapshot_grid(history, objective: Objective, x_grid, kappa, true_x_max, true_y_max, out_path: str | None):
    # Pick evenly spaced frames including start and end.
    indices = np.unique(np.linspace(0, len(history) - 1, num=min(6, len(history)), dtype=int))
    n = len(indices)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.2 * n), gridspec_kw={"width_ratios": [2.2, 1]})
    if n == 1:
        axes = np.array([axes])

    for row, idx in enumerate(indices):
        snap = history[idx]
        ax_obj = axes[row, 0]
        ax_acq = axes[row, 1]
        ax_meta = ax_obj.twinx()
        ax_meta.set_visible(False)

        true_y = objective(x_grid)
        x_obs = np.array(snap.x_obs)
        y_obs = np.array(snap.y_obs)

        ax_obj.plot(x_grid, true_y, color="#2c3e50", lw=1.5, alpha=0.35)
        if len(x_obs) >= 2:
            gp = fit_gp(x_obs, y_obs)
            mu, std = gp.predict(x_grid.reshape(-1, 1), return_std=True)
            lo, hi = mu - 2 * std, mu + 2 * std
            ax_obj.fill_between(x_grid, lo, hi, color="#3498db", alpha=0.25)
            ax_obj.plot(x_grid, mu, color="#2980b9", lw=1.8)
            _, _, acq = ucb(gp, x_grid, kappa)
            for xo in x_obs:
                acq[np.isclose(x_grid, xo)] = np.nan
            ax_acq.plot(x_grid, acq, color="#e67e22", lw=1.8)
            if snap.x_cand is not None:
                ax_obj.axvline(snap.x_cand, color="#e74c3c", ls="--", lw=1.2)
                ax_acq.axvline(snap.x_cand, color="#e74c3c", ls="--", lw=1.2)

        ax_obj.scatter(x_obs, y_obs, s=40, c="#27ae60", zorder=5)
        ax_obj.set_xlim(0, 1)
        ax_obj.set_ylabel("f(x)")
        ax_acq.set_xlim(0, 1)
        ax_acq.set_ylabel("UCB")
        ax_obj.set_title(f"Step {idx} — n={len(x_obs)}")
        ax_acq.grid(True, alpha=0.25)
        ax_obj.grid(True, alpha=0.25)

    fig.suptitle("Bayesian optimization snapshots (GP ±2σ + UCB)", fontsize=13, y=1.01)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved snapshot grid to {out_path}")
    else:
        plt.show()


def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for Taylor coefficients and initial points")
    parser.add_argument("--degree", type=int, default=22, help="Taylor polynomial degree (higher = more wiggles)")
    parser.add_argument("--n-init", type=int, default=3, help="Initial random evaluations")
    parser.add_argument("--n-iters", type=int, default=15, help="BO iterations after initialization")
    parser.add_argument("--kappa", type=float, default=2.0, help="UCB exploration weight")
    parser.add_argument("--grid", type=int, default=400, help="Grid points for plotting and acquisition")
    parser.add_argument("--mode", choices=["animate", "grid"], default="animate")
    parser.add_argument("--save", type=str, default=None, help="Output path (.gif for animate, .png for grid)")
    parser.add_argument(
        "--interval",
        type=int,
        default=5000,
        help="Milliseconds between BO iterations in the live animation (default: 5000 = 5s)",
    )
    args = parser.parse_args()

    coeffs = random_taylor_coeffs(args.degree, seed=args.seed)
    x_grid = np.linspace(0, 1, args.grid)
    objective = Objective.from_coeffs(coeffs, x_grid)
    history, true_x_max, true_y_max = run_bo(
        objective, args.n_iters, args.n_init, args.kappa, x_grid, seed=args.seed
    )

    print("Taylor coefficients (c_k for c_k (x-0.5)^k / k!, then curve normalized):")
    for k, c in enumerate(objective.coeffs):
        print(f"  k={k}: {c:+.4f}")
    print(f"True maximum near x={true_x_max:.4f}, f(x)={true_y_max:.4f}")

    if args.mode == "animate":
        make_animation(history, objective, x_grid, args.kappa, true_x_max, true_y_max, args.save, args.interval)
    else:
        make_snapshot_grid(history, objective, x_grid, args.kappa, true_x_max, true_y_max, args.save)


if __name__ == "__main__":
    main()
