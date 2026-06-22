"""
Shared helpers for the interactive oracle Dash explorer.

Builds tuned 3D oracles, evaluates them on a ternary grid, and renders Plotly
ternary figures (landscape + optional campaign-style sample preview).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from synthetic_data.ackley import Ackley, load_config as load_ackley_config, save_config as save_ackley_config
from synthetic_data.oracles import (
    GaussianMixtureOracle,
    MessyCampaignOracle,
    PlantedBumpField,
    RastriginILROracle,
    ackley_centers_for_layout,
)

HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "oracle_configs"

ORACLE_CHOICES = (
    "realistic_ackley",
    "messy",
    "gaussian",
    "planted_bumps",
    "rastrigin_ilr",
)

DISPLAY_LABELS = {
    "realistic_ackley": "Realistic Ackley (Dirichlet peaks + noise)",
    "messy": "Messy campaign (bumps + ILR ripples)",
    "gaussian": "Gaussian mixture",
    "planted_bumps": "Planted bumps (major + micro)",
    "rastrigin_ilr": "Rastrigin in ILR",
}

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "realistic_ackley": {
        "layout": "2",
        "seed": 42,
        "n_optima": 10,
        "basin_width": 50.0,
        "noise_freq": 8.0,
        "noise_amp": 5.0,
    },
    "messy": {
        "layout": "2",
        "seed": 42,
        "n_micro": 150,
        "n_ripples": 30,
        "major_sigma": 0.055,
    },
    "gaussian": {
        "layout": "2",
        "seed": 42,
        "sigma": 0.07,
    },
    "planted_bumps": {
        "layout": "2",
        "seed": 42,
        "n_micro": 40,
        "major_sigma": 0.09,
        "signed_micro": False,
    },
    "rastrigin_ilr": {
        "layout": "2",
        "seed": 42,
        "amplitude": 10.0,
    },
}

# Which slider groups apply to each oracle (for Dash visibility).
SLIDER_GROUPS: dict[str, tuple[str, ...]] = {
    "realistic_ackley": ("common", "realistic"),
    "messy": ("common", "messy"),
    "gaussian": ("common", "gaussian"),
    "planted_bumps": ("common", "planted"),
    "rastrigin_ilr": ("common", "rastrigin"),
}


def build_grid(grid_n: int) -> np.ndarray:
    return np.array([
        (i / grid_n, j / grid_n, (grid_n - i - j) / grid_n)
        for i in range(grid_n + 1)
        for j in range(grid_n + 1 - i)
    ], dtype=float)


def load_oracle_params(oracle: str) -> dict[str, Any]:
    if oracle == "realistic_ackley":
        ack = load_ackley_config()
        base = dict(DEFAULT_PARAMS["realistic_ackley"])
        base.update({
            "n_optima": int(ack["n_optima"]),
            "basin_width": float(ack["basin_width"]),
            "noise_freq": float(ack["noise_freq"]),
            "noise_amp": float(ack["noise_amp"]),
        })
        return base
    path = CONFIG_DIR / f"{oracle}.json"
    base = dict(DEFAULT_PARAMS.get(oracle, {}))
    if path.is_file():
        with open(path, encoding="utf-8") as f:
            base.update(json.load(f))
    return base


def save_oracle_params(oracle: str, params: dict[str, Any]) -> None:
    if oracle == "realistic_ackley":
        save_ackley_config({
            "n_optima": int(params["n_optima"]),
            "basin_width": float(params["basin_width"]),
            "noise_freq": float(params["noise_freq"]),
            "noise_amp": float(params["noise_amp"]),
        })
        return
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / f"{oracle}.json"
    payload = {k: params[k] for k in DEFAULT_PARAMS.get(oracle, {}) if k in params}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def build_tuned_oracle(oracle: str, params: dict[str, Any], *, dim: int = 3):
    """Return (callable_oracle, list_of_optima, title_suffix)."""
    layout = str(params.get("layout", "2"))
    seed = int(params.get("seed", 42))
    centers = ackley_centers_for_layout(dim, layout)

    if oracle == "realistic_ackley":
        fn = Ackley(
            "realistic",
            dim=dim,
            n_optima=int(params.get("n_optima", 10)),
            basin_width=float(params.get("basin_width", 50.0)),
            noise_freq=float(params.get("noise_freq", 8.0)),
            noise_amp=float(params.get("noise_amp", 5.0)),
            peak_seed=seed,
        )
        optima = [c.copy() for c in fn.centers]
        suffix = f"{len(optima)} peaks, b={params.get('basin_width')}"
        return fn, optima, suffix

    if oracle == "messy":
        obj = MessyCampaignOracle(
            centers,
            n_micro=int(params.get("n_micro", 150)),
            n_ripples=int(params.get("n_ripples", 30)),
            major_sigma=float(params.get("major_sigma", 0.055)),
            seed=seed,
        )
        suffix = f"{len(centers)} major, {params.get('n_micro')} micro, {params.get('n_ripples')} ripples"
        return obj, obj.true_optima, suffix

    if oracle == "gaussian":
        sigma = float(params.get("sigma", 0.07))
        obj = GaussianMixtureOracle(centers, sigma=sigma)
        return obj, obj.true_optima, f"σ={sigma}"

    if oracle == "planted_bumps":
        obj = PlantedBumpField(
            centers,
            n_micro=int(params.get("n_micro", 40)),
            major_sigma=float(params.get("major_sigma", 0.09)),
            signed_micro=bool(params.get("signed_micro", False)),
            seed=seed,
        )
        tag = "signed" if params.get("signed_micro") else "unsigned"
        return obj, obj.true_optima, f"{params.get('n_micro')} micro ({tag})"

    if oracle == "rastrigin_ilr":
        amp = float(params.get("amplitude", 10.0))
        obj = RastriginILROracle(dim, amplitude=amp)
        return obj, obj.true_optima, f"amplitude={amp}"

    raise ValueError(f"Unknown oracle {oracle!r}")


def evaluate_oracle(fn, grid: np.ndarray, *, realistic_ackley: bool = False) -> np.ndarray:
    if realistic_ackley:
        return fn.predict(grid)
    return np.array([float(fn(x)) for x in grid], dtype=float)


def preview_campaign_samples(
    fn,
    optima: list[np.ndarray],
    *,
    n_samples: int = 300,
    seed: int = 42,
    noise_std: float = 0.07,
    realistic_ackley: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Lightweight line+local sampling for the samples panel."""
    rng = np.random.default_rng(seed)
    dim = optima[0].shape[0]
    rows: list[np.ndarray] = []

    n_lines = max(1, int(round(n_samples * 0.6)) // 8)
    for _ in range(n_lines):
        x0 = rng.dirichlet(np.ones(dim))
        x1 = rng.dirichlet(np.ones(dim))
        for t in np.linspace(0.0, 1.0, 8):
            rows.append((1.0 - t) * x0 + t * x1)

    n_local = max(0, n_samples - len(rows))
    per_peak = max(1, n_local // max(1, len(optima)))
    for center in optima:
        for x in rng.dirichlet(30.0 * center, size=per_peak):
            rows.append(x)

    X = np.array(rows[:n_samples], dtype=float)
    if realistic_ackley:
        y = fn.predict(X)
    else:
        y = np.array([float(fn(x)) for x in X], dtype=float)
    y = y + rng.normal(0.0, noise_std, size=y.shape)
    return X, y


def ternary_figure(
    grid: np.ndarray,
    values: np.ndarray,
    optima: list[np.ndarray],
    *,
    title: str,
    width: int = 900,
    height: int = 820,
    marker_size: float = 4.0,
) -> go.Figure:
    ga, gb, gc = grid[:, 2], grid[:, 0], grid[:, 1]
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    traces = [
        go.Scatterternary(
            a=ga, b=gb, c=gc, mode="markers", name="objective", hoverinfo="skip",
            marker=dict(
                color=values, colorscale="Viridis",
                cmin=vmin, cmax=vmax, size=marker_size,
                showscale=True, colorbar=dict(title="Objective", x=1.02),
            ),
        )
    ]
    if optima:
        peaks = np.asarray(optima, dtype=float)
        traces.append(go.Scatterternary(
            a=peaks[:, 2], b=peaks[:, 0], c=peaks[:, 1], mode="markers",
            name="planted peak",
            marker=dict(symbol="star", color="red", size=14,
                        line=dict(color="white", width=1)),
        ))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        ternary=dict(
            sum=1,
            aaxis=dict(title="Comp3"),
            baxis=dict(title="Comp1"),
            caxis=dict(title="Comp2"),
        ),
        legend=dict(x=1.18, y=1.0),
        width=width, height=height,
        margin=dict(t=60),
    )
    return fig


def samples_figure(
    X: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    width: int = 900,
    height: int = 820,
) -> go.Figure:
    if len(X) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Enable sample preview to see campaign-style points",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#666"),
        )
        fig.update_layout(title=title, width=width, height=height, margin=dict(t=60))
        return fig
    fig = go.Figure(data=[
        go.Scatterternary(
            a=X[:, 2], b=X[:, 0], c=X[:, 1], mode="markers", name="samples",
            marker=dict(
                color=y, colorscale="Viridis", size=7, opacity=0.85,
                showscale=True, colorbar=dict(title="Objective", x=1.02),
            ),
        )
    ])
    fig.update_layout(
        title=title,
        ternary=dict(
            sum=1,
            aaxis=dict(title="Comp3"),
            baxis=dict(title="Comp1"),
            caxis=dict(title="Comp2"),
        ),
        width=width, height=height,
        margin=dict(t=60),
    )
    return fig
