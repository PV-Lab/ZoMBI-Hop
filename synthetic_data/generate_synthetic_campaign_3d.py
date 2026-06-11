"""
generate_synthetic_campaign_3d.py
=================================
Build a synthetic 3D composition "campaign" from a planted Multi-Ackley oracle,
train an RF surrogate, and plot oracle vs surrogate on the same ternary layout
used by ``interactive_test_zombi.py``.  Also trains the campaign1a RF and
adds it as a comparison panel.

Recipe (3D pilot before 10D):
  - Oracle: messy (default) — major peaks + signed micro-bumps + ILR ripples
  - Dataset: ~700 rows, campaign sampling (65% line chords, 25% local, 10% uniform)
  - Surrogate: 500-tree Random Forest (+ optional HistGradientBoosting)
  - Reference optima: planted centres (no L-BFGS-B refinement)

Usage
-----
  conda activate zombi-hop-linebo
  python synthetic_data/generate_synthetic_campaign_3d.py

  MPLBACKEND=Agg python synthetic_data/generate_synthetic_campaign_3d.py --no-show

  # Default is a messy oracle + campaign-style line sampling (~700 rows):
  python synthetic_data/generate_synthetic_campaign_3d.py

  python synthetic_data/generate_synthetic_campaign_3d.py --oracle ackley --layout 1
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib

if not sys.stdin.isatty():
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)

# ── Multi-Ackley oracle (same formulas as optimize/run_mobo_10d.py) ───────────

_ACKLEY_A = 20.0
_ACKLEY_B_SKINNY = 1.2
_ACKLEY_C = 2.0 * math.pi
_ACKLEY_SCALE = 30.0


def simplex_vertex(d: int, i: int) -> np.ndarray:
    p = np.zeros(d, dtype=float)
    p[i] = 1.0
    return p


def centroid_composition(d: int) -> np.ndarray:
    return np.ones(d, dtype=float) / d


def edge_midpoint(d: int, i: int, j: int) -> np.ndarray:
    p = np.zeros(d, dtype=float)
    p[i] = 0.5
    p[j] = 0.5
    return p


def ackley_centers_for_layout(d: int, layout: str) -> list[np.ndarray]:
    if d < 2:
        raise ValueError("Multi-Ackley requires d >= 2")
    centers: list[np.ndarray] = [
        centroid_composition(d),
        simplex_vertex(d, 0),
        edge_midpoint(d, 0, 1),
    ]
    if layout in ("2", "3"):
        if d < 3:
            raise ValueError(f"Layout {layout} requires d >= 3")
        centers.append(simplex_vertex(d, 1))
        centers.append(simplex_vertex(d, 2))
    if layout == "3":
        if d < 5:
            raise ValueError("Layout 3 requires d >= 5")
        centers.append(simplex_vertex(d, 3))
        centers.append(simplex_vertex(d, 4))
    if layout not in ("1", "2", "3"):
        raise ValueError(f"Unknown layout {layout!r}; use '1', '2', or '3'.")
    return centers


def _ackley_negated(
    x: np.ndarray,
    center: np.ndarray,
    *,
    b: float = _ACKLEY_B_SKINNY,
) -> float:
    x = np.asarray(x, dtype=float)
    center = np.asarray(center, dtype=float)
    d = x.shape[0]
    delta = x - center
    t1 = -_ACKLEY_A * math.exp(-b * math.sqrt(np.sum(delta ** 2) / d))
    t2 = -math.exp(float(np.sum(np.cos(_ACKLEY_C * delta)) / d))
    return _ACKLEY_SCALE * (t1 + t2 + _ACKLEY_A + math.e)


class MultiAckleyND:
    maximize = True

    def __init__(
        self,
        centers: list[np.ndarray],
        *,
        b: float = _ACKLEY_B_SKINNY,
        layout_name: str = "custom",
    ):
        self.centers = [np.asarray(c, dtype=float).copy() for c in centers]
        self.b = b
        self.layout_name = layout_name

    def __call__(self, x: np.ndarray) -> float:
        return float(sum(_ackley_negated(x, c, b=self.b) for c in self.centers))

    @property
    def true_optima(self) -> list[np.ndarray]:
        return [c.copy() for c in self.centers]


def composition_to_ilr_np(x: np.ndarray) -> np.ndarray:
    """Helmert ILR (numpy), matching ``src.utils.simplex.composition_to_ilr``."""
    x = np.asarray(x, dtype=float)
    eps = 1e-10
    log_x = np.log(x + eps)
    d = x.shape[-1]
    ilr = np.empty(d - 1, dtype=float)
    for i in range(d - 1):
        coef = math.sqrt((i + 1) / (i + 2))
        ilr[i] = coef * (log_x[: i + 1].sum() / (i + 1) - log_x[i + 1])
    return ilr


class GaussianMixtureOracle:
    """Sum of Gaussians on the simplex (smooth, but sharper than Ackley with small σ)."""

    maximize = True

    def __init__(self, peaks: list[np.ndarray], *, sigma: float = 0.07):
        self.peaks = [normalize_rows(np.asarray(p, dtype=float).reshape(1, -1))[0] for p in peaks]
        self.sigma = sigma

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(sum(
            math.exp(-np.sum((x - p) ** 2) / (2.0 * self.sigma ** 2))
            for p in self.peaks
        ))

    @property
    def true_optima(self) -> list[np.ndarray]:
        return [p.copy() for p in self.peaks]


class RastriginILROracle:
    """
    Rastrigin in ILR coordinates — many cosine ripples, very multimodal.
    Maximum at the uniform composition (ILR origin).
    """

    maximize = True

    def __init__(self, d: int = 3, *, amplitude: float = 10.0):
        self.d = d
        self.amplitude = amplitude
        self._centroid = centroid_composition(d)

    def __call__(self, x: np.ndarray) -> float:
        z = composition_to_ilr_np(np.asarray(x, dtype=float))
        n = z.shape[0]
        rastrigin = self.amplitude * n + float(np.sum(z ** 2 - self.amplitude * np.cos(2.0 * math.pi * z)))
        return -rastrigin

    @property
    def true_optima(self) -> list[np.ndarray]:
        return [self._centroid.copy()]


class PlantedBumpField:
    """
    Major Gaussians at planted centres plus many small random micro-bumps.
    Mimics broad optima with fine-scale rug — closer to campaign1a texture.
    """

    maximize = True

    def __init__(
        self,
        major_centers: list[np.ndarray],
        *,
        n_micro: int = 40,
        major_sigma: float = 0.09,
        signed_micro: bool = False,
        seed: int = 42,
    ):
        self.major_centers = [np.asarray(c, dtype=float).copy() for c in major_centers]
        self.major_sigma = major_sigma
        rng = np.random.default_rng(seed)
        d = major_centers[0].shape[0]
        self._micro: list[tuple[np.ndarray, float, float]] = []
        for _ in range(n_micro):
            center = rng.dirichlet(np.ones(d))
            sigma = float(rng.uniform(0.015, 0.05))
            if signed_micro:
                weight = float(rng.uniform(-0.50, 0.50))
            else:
                weight = float(rng.uniform(0.08, 0.35))
            self._micro.append((center, sigma, weight))

    def _bump(self, x: np.ndarray, center: np.ndarray, sigma: float, weight: float) -> float:
        return weight * math.exp(-np.sum((x - center) ** 2) / (2.0 * sigma ** 2))

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        val = sum(
            self._bump(x, c, self.major_sigma, 1.0) for c in self.major_centers
        )
        val += sum(self._bump(x, c, s, w) for c, s, w in self._micro)
        return float(val)

    @property
    def true_optima(self) -> list[np.ndarray]:
        return [c.copy() for c in self.major_centers]


class MessyCampaignOracle:
    """
    Campaign1a-like oracle: major peaks + signed micro-bumps + ILR ripples.
    """

    maximize = True

    def __init__(
        self,
        major_centers: list[np.ndarray],
        *,
        n_micro: int = 150,
        n_ripples: int = 30,
        seed: int = 42,
    ):
        self._bumps = PlantedBumpField(
            major_centers,
            n_micro=n_micro,
            major_sigma=0.055,
            signed_micro=True,
            seed=seed,
        )
        rng = np.random.default_rng(seed + 1)
        d = major_centers[0].shape[0]
        self._ripples: list[tuple[np.ndarray, float, float]] = []
        for _ in range(n_ripples):
            freq = rng.normal(0.0, 1.0, size=d - 1)
            freq /= np.linalg.norm(freq) + 1e-12
            amp = float(rng.uniform(0.06, 0.18))
            phase = float(rng.uniform(0.0, 2.0 * math.pi))
            self._ripples.append((freq, amp, phase))
        self.major_centers = self._bumps.major_centers

    def __call__(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        val = self._bumps(x)
        z = composition_to_ilr_np(x)
        for freq, amp, phase in self._ripples:
            val += amp * math.sin(float(np.dot(freq, z)) + phase)
        return float(val)

    @property
    def true_optima(self) -> list[np.ndarray]:
        return [c.copy() for c in self.major_centers]


ORACLE_CHOICES = (
    "messy",
    "ackley",
    "gaussian",
    "rastrigin_ilr",
    "planted_bumps",
)


def build_oracle(
    name: str,
    d: int,
    layout: str,
    *,
    seed: int,
) -> tuple[object, list[np.ndarray], str]:
    """Return (callable oracle, reference optima, display label)."""
    centers = ackley_centers_for_layout(d, layout)
    if name == "messy":
        obj = MessyCampaignOracle(centers, n_micro=150, n_ripples=30, seed=seed)
        label = f"Messy campaign ({len(centers)} major + 150 signed micro + 30 ILR ripples)"
        return obj, obj.true_optima, label
    if name == "ackley":
        obj = MultiAckleyND(centers, b=ACKLEY_B, layout_name=f"layout-{layout}")
        label = f"Multi-Ackley (layout {layout}, b={ACKLEY_B})"
        return obj, obj.true_optima, label
    if name == "gaussian":
        obj = GaussianMixtureOracle(centers, sigma=0.07)
        label = f"Gaussian mixture ({len(centers)} peaks, σ=0.07)"
        return obj, obj.true_optima, label
    if name == "rastrigin_ilr":
        obj = RastriginILROracle(d)
        label = "Rastrigin in ILR (cosine ripples)"
        return obj, obj.true_optima, label
    if name == "planted_bumps":
        obj = PlantedBumpField(centers, n_micro=40, seed=seed)
        label = f"Planted bumps ({len(centers)} major + 40 micro)"
        return obj, obj.true_optima, label
    raise ValueError(f"Unknown oracle {name!r}; choose from {ORACLE_CHOICES}")


# ── Match interactive_test_zombi.py ──────────────────────────────────────────

DIM = 3
LAYOUT = "2"
ACKLEY_B = _ACKLEY_B_SKINNY
TARGET_DATASET_SIZE = 700   # match campaign1a.csv scale (~700 rows)
LOCAL_FRACTION = 0.25       # uniform-mode: share of rows near planted peaks
LOCAL_DIRICHLET_ALPHA = 30.0
DATASET_NOISE_STD = 0.07    # measurement noise on synthetic y
OUTLIER_FRAC = 0.06         # fraction of rows with extra-large noise
OUTLIER_NOISE_MULT = 5.0
# campaign-mode split (fractions of target, must sum to 1)
CAMPAIGN_LINE_FRACTION = 0.60
CAMPAIGN_LOCAL_FRACTION = 0.30
CAMPAIGN_UNIFORM_FRACTION = 0.10
CAMPAIGN_PTS_PER_LINE = 8
RF_N_ESTIMATORS = 500
TERNARY_GRID_N = 120

COMPOSITION_COLS = [f"Comp{i + 1}" for i in range(DIM)]
OBJECTIVE_COL = "Objective"
CORNER_LABELS = tuple(COMPOSITION_COLS)

CAMPAIGN1A_COLS = ["FAPbI3", "MAPbI3", "MAPbBr3"]
CAMPAIGN1A_CORNER_LABELS = tuple(CAMPAIGN1A_COLS)

_SQRT3_2 = math.sqrt(3) / 2


def comp_to_xy(comp: np.ndarray) -> np.ndarray:
    """(N, 3) simplex compositions → (N, 2) Cartesian ternary coordinates."""
    p = np.asarray(comp, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    s = p.sum(axis=-1, keepdims=True)
    p = p / np.where(s == 0, 1.0, s)
    return np.column_stack([p[:, 1] + 0.5 * p[:, 2], _SQRT3_2 * p[:, 2]])


def draw_ternary_frame(
    ax,
    pad: float = 0.04,
    corner_labels: tuple[str, str, str] = CORNER_LABELS,
) -> None:
    """Draw triangle outline, equal aspect, and corner labels."""
    ax.plot([0, 1, 0.5, 0], [0, 0, _SQRT3_2, 0], "k-", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, _SQRT3_2 + 0.16)
    ax.axis("off")
    ax.text(-pad, -pad, corner_labels[0], ha="right", va="top", fontsize=9)
    ax.text(1 + pad, -pad, corner_labels[1], ha="left", va="top", fontsize=9)
    ax.text(0.5, _SQRT3_2 + pad, corner_labels[2], ha="center", va="bottom", fontsize=9)


def ternary_grid(n: int = TERNARY_GRID_N) -> np.ndarray:
    """Return (N, 3) uniform lattice on the probability simplex."""
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append([i / n, j / n, (n - i - j) / n])
    return np.array(pts, dtype=float)


def normalize_rows(X: np.ndarray) -> np.ndarray:
    s = X.sum(axis=1, keepdims=True)
    return X / np.where(s == 0, 1.0, s)


def dataset_sizes(
    n_peaks: int,
    *,
    target: int = TARGET_DATASET_SIZE,
    local_fraction: float = LOCAL_FRACTION,
) -> tuple[int, int]:
    """Split ``target`` rows into uniform + per-peak local samples."""
    if n_peaks < 1:
        raise ValueError("n_peaks must be >= 1")
    n_local_total = int(round(target * local_fraction))
    n_local_per_peak = max(1, n_local_total // n_peaks)
    n_uniform = target - n_local_per_peak * n_peaks
    if n_uniform < 1:
        raise ValueError(
            f"target={target} too small for {n_peaks} peaks "
            f"at local_fraction={local_fraction}"
        )
    return n_uniform, n_local_per_peak


def _append_sample(
    rows: list[tuple[float, ...]],
    oracle,
    x: np.ndarray,
    rng: np.random.Generator,
    noise_std: float,
    *,
    outlier_frac: float = OUTLIER_FRAC,
) -> None:
    sigma = noise_std
    if outlier_frac > 0 and rng.random() < outlier_frac:
        sigma *= OUTLIER_NOISE_MULT
    y = float(oracle(x)) + float(rng.normal(0.0, sigma))
    rows.append((*np.asarray(x, dtype=float), y))


def generate_dataset(
    oracle,
    centers: list[np.ndarray],
    *,
    n_uniform: int,
    n_local_per_peak: int,
    local_alpha: float,
    noise_std: float,
    seed: int,
    outlier_frac: float = OUTLIER_FRAC,
) -> pd.DataFrame:
    """Uniform Dirichlet bulk + local clusters around planted centres."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[float, ...]] = []

    for x in rng.dirichlet(np.ones(DIM), size=n_uniform):
        _append_sample(rows, oracle, x, rng, noise_std, outlier_frac=outlier_frac)

    for center in centers:
        for x in rng.dirichlet(local_alpha * center, size=n_local_per_peak):
            _append_sample(rows, oracle, x, rng, noise_std, outlier_frac=outlier_frac)

    return pd.DataFrame(rows, columns=COMPOSITION_COLS + [OBJECTIVE_COL])


def generate_campaign_dataset(
    oracle,
    centers: list[np.ndarray],
    *,
    target: int,
    local_alpha: float,
    noise_std: float,
    seed: int,
    pts_per_line: int = CAMPAIGN_PTS_PER_LINE,
    outlier_frac: float = OUTLIER_FRAC,
) -> tuple[pd.DataFrame, str]:
    """
    Campaign-style sampling: mostly points along random simplex chords,
    plus local clusters and a sparse uniform background.
    """
    rng = np.random.default_rng(seed)
    rows: list[tuple[float, ...]] = []

    n_line_pts = int(round(target * CAMPAIGN_LINE_FRACTION))
    n_local_total = int(round(target * CAMPAIGN_LOCAL_FRACTION))
    n_uniform = target - n_line_pts - n_local_total
    if n_uniform < 0:
        n_uniform = 0

    n_lines = max(1, n_line_pts // pts_per_line)
    pts_per_line = max(2, n_line_pts // n_lines)

    for _ in range(n_lines):
        x0 = rng.dirichlet(np.ones(DIM))
        x1 = rng.dirichlet(np.ones(DIM))
        for t in np.linspace(0.0, 1.0, pts_per_line):
            x = (1.0 - t) * x0 + t * x1
            _append_sample(rows, oracle, x, rng, noise_std, outlier_frac=outlier_frac)

    n_local_per_peak = max(1, n_local_total // max(1, len(centers)))
    for center in centers:
        for x in rng.dirichlet(local_alpha * center, size=n_local_per_peak):
            _append_sample(rows, oracle, x, rng, noise_std, outlier_frac=outlier_frac)

    for x in rng.dirichlet(np.ones(DIM), size=n_uniform):
        _append_sample(rows, oracle, x, rng, noise_std, outlier_frac=outlier_frac)

    # Trim or pad to exact target (line discretisation can overshoot slightly).
    if len(rows) > target:
        rows = rows[:target]
    while len(rows) < target:
        x = rng.dirichlet(np.ones(DIM))
        _append_sample(rows, oracle, x, rng, noise_std, outlier_frac=outlier_frac)

    desc = (
        f"{n_lines} lines × {pts_per_line} pts + "
        f"{n_local_per_peak} local × {len(centers)} peaks + "
        f"{n_uniform} uniform"
    )
    return pd.DataFrame(rows, columns=COMPOSITION_COLS + [OBJECTIVE_COL]), desc


def train_rf(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, n_jobs=-1, random_state=42
    )
    rf.fit(X, y)
    return rf


def resolve_campaign1a_path(explicit: str | None = None) -> str:
    if explicit and os.path.isfile(explicit):
        return os.path.normpath(explicit)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        explicit,
        os.path.join(script_dir, "data", "campaign1a.csv"),
        os.path.join(script_dir, "campaign1a.csv"),
        os.path.join(script_dir, "..", "data", "campaign1a.csv"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.normpath(path)
    tried = "\n".join(f"  {os.path.normpath(p)}" for p in candidates if p)
    raise FileNotFoundError(f"campaign1a.csv not found. Tried:\n{tried}")


def load_campaign1a(csv_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Load campaign1a compositions and Objective (same as interactive_test_zombi)."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=CAMPAIGN1A_COLS + [OBJECTIVE_COL])
    X = df[CAMPAIGN1A_COLS].values.astype(float)
    X = normalize_rows(X)
    y = df[OBJECTIVE_COL].values.astype(float)
    return X, y, len(df)


def train_hgb(X: np.ndarray, y: np.ndarray) -> HistGradientBoostingRegressor:
    hgb = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_depth=8, random_state=42
    )
    hgb.fit(X, y)
    return hgb


def plot_ternary_landscapes(
    *,
    grid_pts: np.ndarray,
    panels: list[dict],
    save_path: str | None,
    show: bool,
) -> None:
    """
    Plot one or more ternary landscapes (same style as interactive_test_zombi).

    Each panel dict accepts:
      title, vals, corner_labels (optional), reference_optima (optional list).
    """
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(7.5 * n_panels, 6.8))
    if n_panels == 1:
        axes = [axes]

    gxy = comp_to_xy(grid_pts)

    for ax, panel in zip(axes, panels):
        title = panel["title"]
        vals = panel["vals"]
        corner_labels = panel.get("corner_labels", CORNER_LABELS)
        reference_optima = panel.get("reference_optima")

        draw_ternary_frame(ax, corner_labels=corner_labels)
        ax.set_title(title, fontsize=10)
        sc = ax.scatter(
            gxy[:, 0],
            gxy[:, 1],
            c=vals,
            cmap="viridis",
            s=8,
            alpha=0.80,
            zorder=2,
            rasterized=True,
        )
        if reference_optima:
            ref_xy = comp_to_xy(np.array(reference_optima))
            ax.scatter(
                ref_xy[:, 0],
                ref_xy[:, 1],
                marker="*",
                s=340,
                c="blue",
                zorder=12,
                edgecolors="navy",
                linewidths=1.3,
            )
        fig.colorbar(
            sc,
            ax=ax,
            label="Objective (maximize — higher better)",
            fraction=0.046,
            pad=0.04,
        )

    fig.suptitle(
        "Δ³ surrogate comparison — synthetic Ackley vs campaign1a RF",
        fontsize=11,
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic 3D Ackley campaign + RF plot."
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(__file__), "data", "campaign3d_synthetic_messy.csv"
        ),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--plot",
        default=os.path.join(
            os.path.dirname(__file__), "plots", "campaign3d_synthetic_messy.png"
        ),
        help="Output PNG path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=TARGET_DATASET_SIZE,
        help=f"Total synthetic rows (default {TARGET_DATASET_SIZE}, ~campaign1a).",
    )
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument(
        "--with-hgb",
        action="store_true",
        help="Also train HistGradientBoosting on synthetic data and add a panel.",
    )
    parser.add_argument(
        "--campaign1a",
        default=None,
        help="Path to campaign1a.csv (auto-detected if omitted).",
    )
    parser.add_argument(
        "--skip-campaign1a",
        action="store_true",
        help="Omit the campaign1a RF comparison panel.",
    )
    parser.add_argument(
        "--oracle",
        choices=ORACLE_CHOICES,
        default="messy",
        help=(
            "Ground-truth landscape. "
            "messy=default (signed micro-bumps + ILR ripples); "
            "ackley=smooth; gaussian=moderate; "
            "rastrigin_ilr=cosine ripples; planted_bumps=major + micro bumps."
        ),
    )
    parser.add_argument(
        "--sampling",
        choices=("campaign", "uniform"),
        default="campaign",
        help="campaign=line chords + sparse background (default); uniform=Dirichlet bulk.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=DATASET_NOISE_STD,
        help=f"Gaussian noise on Objective (default {DATASET_NOISE_STD}).",
    )
    parser.add_argument(
        "--outlier-frac",
        type=float,
        default=OUTLIER_FRAC,
        help=f"Fraction of noisy outlier rows (default {OUTLIER_FRAC}).",
    )
    args = parser.parse_args()

    oracle, centers, oracle_label = build_oracle(
        args.oracle, DIM, LAYOUT, seed=args.seed
    )

    print("=" * 70)
    print("Synthetic 3D campaign — oracle + RF surrogate")
    print(f"  Oracle: {oracle_label}")
    print(f"  Reference optima: {len(centers)} planted centre(s)")
    print(f"  Sampling: {args.sampling}")
    print(f"  Target size: {args.n_samples} rows (~campaign1a)")
    print("=" * 70)

    print("\n[1] Generating dataset …")
    sampling_desc = ""
    if args.sampling == "campaign":
        df, sampling_desc = generate_campaign_dataset(
            oracle,
            centers,
            target=args.n_samples,
            local_alpha=LOCAL_DIRICHLET_ALPHA,
            noise_std=args.noise_std,
            seed=args.seed,
            outlier_frac=args.outlier_frac,
        )
    else:
        n_uniform, n_local_per_peak = dataset_sizes(
            len(centers), target=args.n_samples
        )
        sampling_desc = (
            f"{n_uniform} uniform + {n_local_per_peak} local × {len(centers)} peaks"
        )
        df = generate_dataset(
            oracle,
            centers,
            n_uniform=n_uniform,
            n_local_per_peak=n_local_per_peak,
            local_alpha=LOCAL_DIRICHLET_ALPHA,
            noise_std=args.noise_std,
            seed=args.seed,
            outlier_frac=args.outlier_frac,
        )
    print(f"    Sampling mix: {sampling_desc}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"    {len(df)} rows → {args.output}")

    print("\n[2] Training surrogates …")
    X = normalize_rows(df[COMPOSITION_COLS].values.astype(float))
    y = df[OBJECTIVE_COL].values.astype(float)

    rf = train_rf(X, y)
    rf_r2 = rf.score(X, y)
    print(f"    RF train R² = {rf_r2:.4f}")

    hgb = None
    hgb_r2 = None
    if args.with_hgb:
        hgb = train_hgb(X, y)
        hgb_r2 = hgb.score(X, y)
        print(f"    HGB train R² = {hgb_r2:.4f}")

    print("\n[3] Reference optima (planted centres):")
    for i, c in enumerate(centers):
        print(f"    peak {i + 1}: {np.round(c, 4)}  y={oracle(c):.4f}")

    rf_c1a = None
    rf_c1a_r2 = None
    n_c1a = 0
    if not args.skip_campaign1a:
        print("\n[4] Training campaign1a RF …")
        c1a_path = resolve_campaign1a_path(args.campaign1a)
        X_c1a, y_c1a, n_c1a = load_campaign1a(c1a_path)
        rf_c1a = train_rf(X_c1a, y_c1a)
        rf_c1a_r2 = rf_c1a.score(X_c1a, y_c1a)
        print(f"    Loaded {n_c1a} rows from {c1a_path}")
        print(f"    campaign1a RF train R² = {rf_c1a_r2:.4f}")

    print("\n[5] Building ternary grid and plotting …")
    grid_pts = ternary_grid(TERNARY_GRID_N)
    oracle_vals = np.array([oracle(x) for x in grid_pts])
    rf_syn_vals = rf.predict(grid_pts)

    plot_panels: list[dict] = [
        {
            "title": f"Oracle: {oracle_label}",
            "vals": oracle_vals,
            "reference_optima": centers,
        },
        {
            "title": f"Synthetic RF ({RF_N_ESTIMATORS} trees, R²={rf_r2:.4f})",
            "vals": rf_syn_vals,
            "reference_optima": centers,
        },
    ]
    if hgb is not None and hgb_r2 is not None:
        plot_panels.append(
            {
                "title": f"Synthetic HGB (R²={hgb_r2:.4f})",
                "vals": hgb.predict(grid_pts),
                "reference_optima": centers,
            }
        )
    if rf_c1a is not None and rf_c1a_r2 is not None:
        plot_panels.append(
            {
                "title": f"campaign1a RF ({n_c1a} rows, R²={rf_c1a_r2:.4f})",
                "vals": rf_c1a.predict(grid_pts),
                "corner_labels": CAMPAIGN1A_CORNER_LABELS,
            }
        )

    plot_ternary_landscapes(
        grid_pts=grid_pts,
        panels=plot_panels,
        save_path=args.plot,
        show=not args.no_show,
    )
    print("Done.")


if __name__ == "__main__":
    main()
