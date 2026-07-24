#!/usr/bin/env python3
"""Build RF approximations of every generated ELA landscape and an HTML gallery.

For each ``ela/runs/ela_3d_*`` run that has a landscape image and a recoverable
expression, this script:

1. evaluates the evolved landscape at 650 Sobol simplex samples,
2. fits the standard 500-tree ELA RF surrogate,
3. evaluates fidelity on a separate 4,096-point Sobol sample,
4. compares both landscapes with the RF trained on the original 3D campaign,
5. writes a three-panel PNG, per-run JSON metrics, aggregate JSON, and HTML.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20_000))

from ela.evolve_context import load_context_from_run  # noqa: E402
from ela.features import (  # noqa: E402
    DEFAULT_INPUT_NOISE,
    composition_to_ilr,
    sample_simplex_sobol,
    train_rf_surrogate,
)

# Multiplicative y-noise fraction matched to MOBO / interactive sims of 2nd_real_run.
DEFAULT_OUTPUT_NOISE_FRAC = 0.045
from ela.gp_tree import (  # noqa: E402
    UNARY_OPS,
    parse_expression_calibration,
    predict_calibrated,
    predict_raw_clipped,
    tree_from_jsonable,
)
from visualization.needle_overlay import comp_to_xy, ternary_grid  # noqa: E402

CORNER_LABELS = ("FAPbI3", "MAPbI3", "MAPbBr3")
SQRT3_2 = np.sqrt(3.0) / 2.0


class ExpressionParser:
    """Parse the stable, fully-parenthesized output of ``tree_to_string``."""

    _NUMBER = r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"

    def __init__(self, value: str):
        self.value = value
        self.offset = 0

    def remaining(self) -> str:
        while self.offset < len(self.value) and self.value[self.offset].isspace():
            self.offset += 1
        return self.value[self.offset :]

    def parse(self):
        node = self._parse_expression()
        if self.remaining():
            raise ValueError(f"Trailing expression text: {self.remaining()[:80]}")
        return node

    def _parse_expression(self):
        node = self._parse_term()
        while True:
            rest = self.remaining()
            if not (rest.startswith("+") or rest.startswith("-")):
                return node
            operator = rest[0]
            self.offset += 1
            right = self._parse_term()
            node = ("add" if operator == "+" else "sub", node, right)

    def _parse_term(self):
        node = self._parse_factor()
        while True:
            rest = self.remaining()
            if not (rest.startswith("*") or rest.startswith("/")):
                return node
            operator = rest[0]
            self.offset += 1
            right = self._parse_factor()
            node = ("mul" if operator == "*" else "div", node, right)

    def _parse_factor(self):
        rest = self.remaining()
        if rest.startswith("("):
            self.offset += 1
            node = self._parse_expression()
            if not self.remaining().startswith(")"):
                raise ValueError(f"Expected ')' near {self.remaining()[:80]}")
            self.offset += 1
            return node

        match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\(", rest)
        if match:
            name = match.group(1)
            self.offset += match.end()
            if name == "rbf":
                return self._parse_rbf()
            if name in UNARY_OPS:
                argument = self._parse_expression()
                if not self.remaining().startswith(")"):
                    raise ValueError(f"Expected ')' after {name}")
                self.offset += 1
                return (name, argument)

        match = re.match(r"z(\d+)", rest)
        if match:
            self.offset += match.end()
            return ("var", int(match.group(1)))

        match = re.match(self._NUMBER, rest)
        if match:
            self.offset += match.end()
            return ("const", float(match.group(0)))
        raise ValueError(f"Cannot parse expression near {rest[:80]}")

    def _parse_rbf(self):
        match = re.match(
            rf"c=\(\s*({self._NUMBER})\s*,\s*({self._NUMBER})\s*\)\s*,\s*"
            rf"a=\s*({self._NUMBER})\s*,\s*l=\s*({self._NUMBER})\s*\)",
            self.remaining(),
        )
        if not match:
            raise ValueError(f"Cannot parse RBF near {self.remaining()[:100]}")
        self.offset += match.end()
        return (
            "rbf",
            float(match.group(1)),
            float(match.group(2)),
            float(match.group(3)),
            float(match.group(4)),
        )


@dataclass
class LandscapeSource:
    tree: Any
    generation: int
    source: str
    predict: Callable[[np.ndarray], np.ndarray]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _latest_generation(run_dir: Path) -> int:
    history = run_dir / "evolution" / "history.csv"
    if history.is_file():
        with history.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            value = rows[-1].get("generation") or rows[-1].get("gen")
            if value not in (None, ""):
                return int(float(value))
    images = sorted((run_dir / "evolution" / "landscapes").glob("gen_*.png"))
    return int(images[-1].stem.split("_")[-1]) if images else -1


def load_landscape_source(run_dir: Path) -> LandscapeSource:
    """Load final expression, falling back to the latest saved snapshot."""
    context = load_context_from_run(run_dir)
    expression_path = run_dir / "best" / "expression.json"
    metadata: dict[str, Any]
    source: str

    if expression_path.is_file():
        metadata = _read_json(expression_path)
        tree = tree_from_jsonable(metadata["expression"])
        generation = _latest_generation(run_dir)
        source = str(expression_path.relative_to(run_dir))
    else:
        snapshots = sorted((run_dir / "evolution" / "snapshots").glob("gen_*.json"))
        if not snapshots:
            raise FileNotFoundError("no best expression or evolution snapshot")
        snapshot = snapshots[-1]
        metadata = _read_json(snapshot)
        encoded = metadata["expression"]
        tree = (
            ExpressionParser(encoded).parse()
            if isinstance(encoded, str)
            else tree_from_jsonable(encoded)
        )
        generation = int(metadata.get("generation", snapshot.stem.split("_")[-1]))
        source = str(snapshot.relative_to(run_dir))

    linear_calibration = bool(
        metadata.get(
            "linear_calibration_enabled",
            metadata.get("linear_calibration", context.linear_calibration),
        )
    )
    if linear_calibration:
        if "calib_a" in metadata or "calib_b" in metadata:
            calibration = (
                float(metadata.get("calib_a", 1.0)),
                float(metadata.get("calib_b", 0.0)),
            )
        else:
            calibration = parse_expression_calibration(
                metadata, linear_calibration=True
            )

        def predict(z: np.ndarray) -> np.ndarray:
            values, _ = predict_calibrated(tree, z, calib=calibration)
            return values

    else:

        def predict(z: np.ndarray) -> np.ndarray:
            return predict_raw_clipped(tree, z)

    return LandscapeSource(
        tree=tree,
        generation=generation,
        source=source,
        predict=predict,
    )


def draw_ternary_frame(ax: plt.Axes) -> None:
    ax.plot([0, 1, 0.5, 0], [0, 0, SQRT3_2, 0], color="black", lw=1.1)
    ax.set_aspect("equal")
    ax.set_xlim(-0.20, 1.20)
    ax.set_ylim(-0.22, SQRT3_2 + 0.17)
    ax.axis("off")
    ax.text(0, -0.09, CORNER_LABELS[0], ha="center", va="top", fontsize=8)
    ax.text(1, -0.09, CORNER_LABELS[1], ha="center", va="top", fontsize=8)
    ax.text(
        0.5, SQRT3_2 + 0.035, CORNER_LABELS[2],
        ha="center", va="bottom", fontsize=8,
    )


def robust_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(array).ravel() for array in arrays])
    values = values[np.isfinite(values)]
    low, high = np.percentile(values, [1, 99])
    if high <= low:
        padding = max(abs(float(low)) * 1e-3, 1e-6)
        return float(low - padding), float(high + padding)
    return float(low), float(high)


def apply_input_noise(
    x: np.ndarray,
    *,
    input_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perturb compositions like printer sent→measured error, then renormalize."""
    if input_noise <= 0.0:
        return np.asarray(x, dtype=float)
    measured = np.asarray(x, dtype=float) + rng.normal(0.0, input_noise, size=x.shape)
    measured = np.clip(measured, 0.0, None)
    totals = measured.sum(axis=1, keepdims=True)
    bad = totals.ravel() <= 1e-12
    if np.any(bad):
        measured[bad] = np.asarray(x, dtype=float)[bad]
        totals = measured.sum(axis=1, keepdims=True)
    return measured / totals


def apply_output_noise(
    y: np.ndarray,
    *,
    output_noise_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Multiplicative measurement noise: y += N(0, frac · |y|)."""
    if output_noise_frac <= 0.0:
        return np.asarray(y, dtype=float)
    y = np.asarray(y, dtype=float)
    return y + rng.normal(0.0, 1.0, size=y.shape) * (output_noise_frac * np.abs(y))


def sample_noisy_training_set(
    predict: Callable[[np.ndarray], np.ndarray],
    *,
    n_samples: int,
    sample_seed: int,
    input_noise: float,
    output_noise_frac: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sobol intended X → noisy measured X + noisy y (real-run emulation).

    Returns ``(x_measured, y_noisy, x_intended)``. Landscape is evaluated at the
    measured composition (as in hardware), then y gets multiplicative noise.
    """
    rng = np.random.default_rng(sample_seed + 17)
    x_intended = sample_simplex_sobol(3, n_samples, seed=sample_seed)
    x_measured = apply_input_noise(x_intended, input_noise=input_noise, rng=rng)
    y_clean = predict(composition_to_ilr(x_measured))
    y_noisy = apply_output_noise(
        y_clean, output_noise_frac=output_noise_frac, rng=rng,
    )
    return x_measured, y_noisy, x_intended


def run_name(run_dir: Path) -> str:
    for filename in ("pilot_config.resolved.json", "pilot_config.source.json"):
        path = run_dir / filename
        if path.is_file():
            value = _read_json(path).get("name")
            if value:
                return str(value)
    return "?"


def process_run(
    run_dir: Path,
    output_dir: Path,
    *,
    n_samples: int,
    sample_seed: int,
    test_size: int,
    grid_n: int,
    n_estimators: int,
    rf_seed: int,
    input_noise: float,
    output_noise_frac: float,
) -> dict[str, Any]:
    context = load_context_from_run(run_dir)
    source = load_landscape_source(run_dir)

    # Constant seeds intentionally give all runs the same train/test locations.
    # Training uses real-run noise; held-out fidelity is vs the clean landscape.
    x_train, y_train, _x_intended = sample_noisy_training_set(
        source.predict,
        n_samples=n_samples,
        sample_seed=sample_seed,
        input_noise=input_noise,
        output_noise_frac=output_noise_frac,
    )
    sampled_rf = train_rf_surrogate(
        x_train, y_train,
        n_estimators=n_estimators,
        random_state=rf_seed,
    )
    campaign_rf = train_rf_surrogate(
        context.x_campaign,
        context.y_campaign,
        n_estimators=n_estimators,
        random_state=rf_seed,
    )

    x_test = sample_simplex_sobol(3, test_size, seed=sample_seed + 1)
    z_test = composition_to_ilr(x_test)
    y_landscape_test = source.predict(z_test)
    y_sampled_rf_test = sampled_rf.predict(x_test)
    y_campaign_rf_test = campaign_rf.predict(x_test)

    metrics = {
        "rf_fidelity_r2": float(r2_score(y_landscape_test, y_sampled_rf_test)),
        "rf_fidelity_rmse": float(
            np.sqrt(mean_squared_error(y_landscape_test, y_sampled_rf_test))
        ),
        "rf_fidelity_mae": float(
            mean_absolute_error(y_landscape_test, y_sampled_rf_test)
        ),
        "landscape_vs_campaign_rf_r2": float(
            r2_score(y_campaign_rf_test, y_landscape_test)
        ),
        "sampled_rf_vs_campaign_rf_r2": float(
            r2_score(y_campaign_rf_test, y_sampled_rf_test)
        ),
    }

    grid = ternary_grid(grid_n)
    triangulation_xy = comp_to_xy(grid)
    triangulation = mtri.Triangulation(
        triangulation_xy[:, 0], triangulation_xy[:, 1]
    )
    z_grid = composition_to_ilr(grid)
    y_campaign_grid = campaign_rf.predict(grid)
    y_landscape_grid = source.predict(z_grid)
    y_sampled_rf_grid = sampled_rf.predict(grid)

    landscape_low, landscape_high = robust_limits(
        y_landscape_grid, y_sampled_rf_grid
    )
    campaign_low, campaign_high = robust_limits(y_campaign_grid)

    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.4))
    panels = (
        (
            axes[0], y_campaign_grid, campaign_low, campaign_high,
            "Original 3D campaign RF",
            f"n = {len(context.y_campaign)}",
        ),
        (
            axes[1], y_landscape_grid, landscape_low, landscape_high,
            f"Evolved landscape · gen {source.generation}",
            f"vs campaign RF R² = {metrics['landscape_vs_campaign_rf_r2']:.3f}",
        ),
        (
            axes[2], y_sampled_rf_grid, landscape_low, landscape_high,
            f"650-sample RF (+ noise)",
            (
                f"held-out R² = {metrics['rf_fidelity_r2']:.3f} · "
                f"vs campaign RF R² = {metrics['sampled_rf_vs_campaign_rf_r2']:.3f}"
            ),
        ),
    )
    for ax, values, low, high, title, subtitle in panels:
        draw_ternary_frame(ax)
        surface = ax.tripcolor(
            triangulation,
            values,
            cmap="viridis",
            vmin=low,
            vmax=high,
            shading="gouraud",
            rasterized=True,
        )
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        fig.colorbar(
            surface, ax=ax, fraction=0.046, pad=0.04,
            extend="both", label="Landscape value",
        )

    train_xy = comp_to_xy(x_train)
    axes[2].scatter(
        train_xy[:, 0], train_xy[:, 1],
        s=1.8, c="black", alpha=0.13, linewidths=0, rasterized=True,
    )

    job_id = int(run_dir.name.split("_")[-1])
    name = run_name(run_dir)
    fig.suptitle(
        (
            f"{job_id} · {name} · RF from {n_samples} noisy landscape samples "
            f"(σ_x={input_noise:g}, σ_y={output_noise_frac:.1%}·|y|)\n"
            f"held-out RMSE={metrics['rf_fidelity_rmse']:.4g} · "
            f"MAE={metrics['rf_fidelity_mae']:.4g}"
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.91))

    image_path = output_dir / f"{run_dir.name}.png"
    fig.savefig(image_path, dpi=155, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    record = {
        "job_id": job_id,
        "run": run_dir.name,
        "name": name,
        "generation": source.generation,
        "expression_source": source.source,
        "n_campaign": int(len(context.y_campaign)),
        "n_rf_samples": n_samples,
        "sample_seed": sample_seed,
        "test_sample_size": test_size,
        "n_estimators": n_estimators,
        "rf_seed": rf_seed,
        "input_noise": input_noise,
        "output_noise_frac": output_noise_frac,
        "metrics": metrics,
        "image": image_path.name,
    }
    image_path.with_suffix(".json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return record


def write_gallery(
    records: list[dict[str, Any]],
    failures: list[dict[str, str]],
    output_path: Path,
) -> None:
    records.sort(key=lambda row: row["job_id"])
    cards = []
    toc = []
    for row in records:
        metrics = row["metrics"]
        anchor = row["run"]
        toc.append(
            f'<li><a href="#{anchor}"><strong>{row["job_id"]}</strong> · '
            f'{html.escape(row["name"])} · R² {metrics["rf_fidelity_r2"]:.3f}</a></li>'
        )
        cards.append(
            f"""
<article class="card" id="{anchor}">
  <header>
    <h2>{row["job_id"]} · {html.escape(row["name"])}</h2>
    <span>generation {row["generation"]}</span>
  </header>
  <div class="metrics">
    <div><b>{metrics["rf_fidelity_r2"]:.4f}</b><span>RF fidelity R²</span></div>
    <div><b>{metrics["landscape_vs_campaign_rf_r2"]:.4f}</b><span>Landscape vs campaign RF R²</span></div>
    <div><b>{metrics["sampled_rf_vs_campaign_rf_r2"]:.4f}</b><span>Sampled RF vs campaign RF R²</span></div>
    <div><b>{metrics["rf_fidelity_rmse"]:.4g}</b><span>Held-out RMSE</span></div>
  </div>
  <a href="rf_surrogate_gallery/{row["image"]}" target="_blank">
    <img src="rf_surrogate_gallery/{row["image"]}?v=noise" loading="lazy"
         alt="RF surrogate comparison for {row["run"]}">
  </a>
  <footer>
    650 noisy Sobol samples (σ_x={row.get("input_noise", DEFAULT_INPUT_NOISE):g},
    σ_y={100.0 * row.get("output_noise_frac", DEFAULT_OUTPUT_NOISE_FRAC):.1f}%·|y|) ·
    4,096 clean held-out · 500 trees ·
    <code>{html.escape(row["expression_source"])}</code>
  </footer>
</article>"""
        )

    failure_html = ""
    if failures:
        items = "".join(
            f"<li><code>{html.escape(row['run'])}</code>: "
            f"{html.escape(row['error'])}</li>"
            for row in failures
        )
        failure_html = f"<section class='failures'><h2>Skipped runs</h2><ul>{items}</ul></section>"

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ELA landscape RF surrogate gallery</title>
<style>
:root {{
  --bg:#0f1115; --panel:#171a21; --ink:#e8eaed; --muted:#9aa3b2;
  --line:#2a3040; --accent:#78a9ff; --bad:#f87171;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:14px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif; }}
.wrap {{ max-width:1400px; margin:auto; padding:28px 20px 80px; }}
h1 {{ margin:0 0 8px; font-size:26px; }}
.intro {{ color:var(--muted); max-width:85ch; }}
nav {{ margin:22px 0 30px; padding:16px 18px; background:var(--panel);
  border:1px solid var(--line); border-radius:10px; }}
nav h2 {{ margin:0 0 10px; font-size:13px; text-transform:uppercase;
  letter-spacing:.06em; color:var(--muted); }}
nav ul {{ columns:2; list-style:none; padding:0; margin:0; }}
nav li {{ break-inside:avoid; margin:0 0 4px; }}
a {{ color:var(--ink); text-decoration:none; }}
a:hover {{ color:var(--accent); }}
.grid {{ display:grid; grid-template-columns:1fr; gap:20px; }}
.card {{ background:var(--panel); border:1px solid var(--line);
  border-radius:12px; padding:16px; }}
.card header {{ display:flex; align-items:baseline; justify-content:space-between;
  gap:12px; margin-bottom:12px; }}
.card h2 {{ margin:0; font-size:17px; }}
.card header span,.card footer {{ color:var(--muted); font-size:12px; }}
.metrics {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px;
  margin-bottom:12px; }}
.metrics div {{ border-left:2px solid var(--line); padding-left:10px; }}
.metrics b {{ display:block; font:600 17px ui-monospace,SFMono-Regular,monospace; }}
.metrics span {{ display:block; color:var(--muted); font-size:11px; }}
img {{ width:100%; height:auto; display:block; object-fit:contain; }}
.card footer {{ margin-top:8px; }}
code {{ font-family:ui-monospace,SFMono-Regular,monospace; }}
.failures {{ margin-top:28px; color:var(--bad); }}
@media(max-width:750px) {{
  nav ul {{ columns:1; }}
  .metrics {{ grid-template-columns:1fr 1fr; }}
}}
</style>
</head>
<body><main class="wrap">
<h1>ELA landscapes → 650-sample RF surrogates</h1>
<p class="intro">
Each row compares the RF surrogate trained on the original 3D campaign,
the evolved ELA landscape, and a new 500-tree RF trained on 650 Sobol samples
from that landscape with real-run noise: composition σ={DEFAULT_INPUT_NOISE:g}
(sent→measured) and multiplicative y noise {100.0 * DEFAULT_OUTPUT_NOISE_FRAC:.1f}%·|y|.
Held-out fidelity R² uses a separate clean 4,096-point Sobol set (how well the
noisy-trained RF recovers the clean landscape). Campaign-RF R² measures target resemblance.
Generated {datetime.now().isoformat(timespec="minutes")}.
</p>
<nav><h2>Runs by job id ({len(records)})</h2><ul>{''.join(toc)}</ul></nav>
<section class="grid">{''.join(cards)}</section>
{failure_html}
</main></body></html>
"""
    output_path.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "ela" / "runs")
    parser.add_argument("--n-samples", type=int, default=650)
    parser.add_argument("--sample-seed", type=int, default=75)
    parser.add_argument("--test-size", type=int, default=4096)
    parser.add_argument("--grid-n", type=int, default=150)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--rf-seed", type=int, default=42)
    parser.add_argument(
        "--input-noise", type=float, default=DEFAULT_INPUT_NOISE,
        help="Per-component composition noise std before renorm (real-run ≈0.064)",
    )
    parser.add_argument(
        "--output-noise-frac", type=float, default=DEFAULT_OUTPUT_NOISE_FRAC,
        help="Multiplicative y noise fraction (MOBO real-run match ≈0.045)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process first N runs")
    args = parser.parse_args()

    runs_root = args.runs_root.resolve()
    output_dir = runs_root / "rf_surrogate_gallery"
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery_path = runs_root / "ela_rf_surrogate_gallery.html"

    runs = sorted(
        [
            path for path in runs_root.glob("ela_3d_*")
            if path.is_dir()
            and any((path / "evolution" / "landscapes").glob("gen_*.png"))
        ],
        key=lambda path: int(path.name.split("_")[-1]),
    )
    if args.limit is not None:
        runs = runs[: args.limit]

    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, run_dir in enumerate(runs, start=1):
        try:
            record = process_run(
                run_dir,
                output_dir,
                n_samples=args.n_samples,
                sample_seed=args.sample_seed,
                test_size=args.test_size,
                grid_n=args.grid_n,
                n_estimators=args.n_estimators,
                rf_seed=args.rf_seed,
                input_noise=args.input_noise,
                output_noise_frac=args.output_noise_frac,
            )
            records.append(record)
            print(
                f"[{index:02d}/{len(runs)}] {run_dir.name} "
                f"R2={record['metrics']['rf_fidelity_r2']:.4f}",
                flush=True,
            )
        except Exception as exc:
            failures.append({"run": run_dir.name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{index:02d}/{len(runs)}] SKIP {run_dir.name}: {exc}", flush=True)

    aggregate = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "configuration": {
            "n_rf_samples": args.n_samples,
            "sample_seed": args.sample_seed,
            "test_size": args.test_size,
            "grid_n": args.grid_n,
            "n_estimators": args.n_estimators,
            "rf_seed": args.rf_seed,
            "input_noise": args.input_noise,
            "output_noise_frac": args.output_noise_frac,
        },
        "n_runs": len(records),
        "n_failures": len(failures),
        "runs": records,
        "failures": failures,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    write_gallery(records, failures, gallery_path)
    print(f"Gallery: {gallery_path}")
    print(f"Metrics: {output_dir / 'metrics.json'}")
    print(f"Generated {len(records)} runs; skipped {len(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
