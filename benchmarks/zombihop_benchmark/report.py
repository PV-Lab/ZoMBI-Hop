from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from .io import load_yaml, write_json, write_yaml

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


CONVERGENCE_METRICS = ["best_y_so_far", "dist_to_needles", "pct_matched", "dup_fraction"]
FINAL_METRICS = [
    "final_best_y_so_far",
    "final_dist_to_needles",
    "final_pct_matched",
    "final_dup_fraction",
    "final_runtime_s",
    "num_points",
    "num_lines",
    "line_best_y_mean",
    "line_best_y_max",
    "line_length_l2_mean",
    "line_length_ilr_mean",
]
RANK_METRICS = {
    "final_best_y_so_far": "higher",
    "final_dist_to_needles": "lower",
    "final_pct_matched": "higher",
    "final_dup_fraction": "lower",
    "final_runtime_s": "lower",
}
RUN_KEYS = ["report_label", "optimizer", "seed", "run_dir"]


@dataclass
class LoadedAggregate:
    label: str
    aggregate_dir: Path
    suite_config_path: Path | None
    suite_name: str
    objective: str
    objective_kind: str
    objective_family: str
    mode: str
    summary: dict[str, Any]
    run_index: pd.DataFrame
    metrics: pd.DataFrame
    final_metrics: pd.DataFrame
    line_metrics: pd.DataFrame


def run_report(config: dict[str, Any], repo_root: Path) -> Path:
    name = str(config.get("name", "benchmark_report"))
    output_root = _resolve_path(config.get("output_root", "benchmark_runs/reports"), repo_root)
    report_dir = output_root / name / datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=False)

    loaded = [load_aggregate(input_cfg, repo_root) for input_cfg in config.get("inputs", [])]
    if not loaded:
        raise ValueError("report config must contain at least one input aggregate")

    final_by_run = build_final_metrics_by_run(loaded)
    final_by_optimizer = summarize_final_metrics(final_by_run, ["optimizer"])
    final_by_detail = summarize_final_metrics(
        final_by_run,
        ["objective_family", "objective_kind", "objective", "mode", "optimizer"],
    )
    auc_by_run = compute_auc_metrics_by_run(loaded)
    transfer_rank_delta = compute_transfer_rank_delta(final_by_run, config.get("metrics", {}))

    final_by_run.to_csv(report_dir / "final_metrics_by_run.csv", index=False)
    final_by_optimizer.to_csv(report_dir / "final_metrics_by_optimizer.csv", index=False)
    final_by_detail.to_csv(report_dir / "final_metrics_by_optimizer_objective_mode.csv", index=False)
    auc_by_run.to_csv(report_dir / "auc_metrics_by_run.csv", index=False)
    transfer_rank_delta.to_csv(report_dir / "transfer_rank_delta.csv", index=False)

    write_yaml(report_dir / "report_config_resolved.yaml", config)
    plot_paths = generate_plots(report_dir / "plots", loaded, final_by_run, transfer_rank_delta, config)
    manifest = build_manifest(config, loaded, report_dir, plot_paths)
    write_json(report_dir / "report_manifest.json", manifest)
    write_report_markdown(
        report_dir / "report.md",
        loaded,
        final_by_detail,
        transfer_rank_delta,
        manifest,
        plot_paths,
    )
    return report_dir


def load_aggregate(input_cfg: dict[str, Any], repo_root: Path) -> LoadedAggregate:
    label = str(input_cfg["label"])
    aggregate_dir = _resolve_path(input_cfg["aggregate_dir"], repo_root)
    suite_config_path = _resolve_optional_path(input_cfg.get("suite_config"), repo_root)
    if not aggregate_dir.exists():
        hint = ""
        if suite_config_path is not None:
            hint = f" Run: python -m benchmarks.zombihop_benchmark.suite --config {suite_config_path}"
        raise FileNotFoundError(f"Missing aggregate_dir for {label}: {aggregate_dir}.{hint}")

    suite_config = load_yaml(suite_config_path) if suite_config_path and suite_config_path.exists() else {}
    experiment = suite_config.get("experiment", {})
    objective_cfg = suite_config.get("objective", {})
    summary = _read_json(aggregate_dir / "summary.json")

    suite_name = str(experiment.get("name") or summary.get("suite") or label)
    mode = str(input_cfg.get("mode") or experiment.get("mode") or _infer_mode(label))
    objective = str(input_cfg.get("objective") or objective_cfg.get("name") or _infer_objective(label))
    objective_kind = str(input_cfg.get("objective_kind") or objective_cfg.get("kind") or _infer_objective_kind(label))
    objective_family = str(input_cfg.get("objective_family") or _infer_objective_family(label, objective_kind))

    run_index = _annotate(
        _read_csv(aggregate_dir / "run_index.csv"),
        label,
        suite_name,
        objective,
        objective_kind,
        objective_family,
        mode,
        aggregate_dir,
    )
    metrics = _annotate(
        _read_csv(aggregate_dir / "metrics_over_time_long.csv"),
        label,
        suite_name,
        objective,
        objective_kind,
        objective_family,
        mode,
        aggregate_dir,
    )
    final_metrics = _annotate(
        _read_csv(aggregate_dir / "final_metrics.csv"),
        label,
        suite_name,
        objective,
        objective_kind,
        objective_family,
        mode,
        aggregate_dir,
    )
    line_metrics = _annotate(
        _read_csv(aggregate_dir / "line_metrics_long.csv"),
        label,
        suite_name,
        objective,
        objective_kind,
        objective_family,
        mode,
        aggregate_dir,
    )

    return LoadedAggregate(
        label=label,
        aggregate_dir=aggregate_dir,
        suite_config_path=suite_config_path,
        suite_name=suite_name,
        objective=objective,
        objective_kind=objective_kind,
        objective_family=objective_family,
        mode=mode,
        summary=summary,
        run_index=run_index,
        metrics=metrics,
        final_metrics=final_metrics,
        line_metrics=line_metrics,
    )


def build_final_metrics_by_run(loaded: list[LoadedAggregate]) -> pd.DataFrame:
    run_index = _concat([item.run_index for item in loaded])
    final_metrics = _concat([item.final_metrics for item in loaded])
    line_metrics = _concat([item.line_metrics for item in loaded])

    if final_metrics.empty and run_index.empty:
        return pd.DataFrame()

    final = final_metrics.copy()
    rename = {
        "best_y_so_far": "final_best_y_so_far",
        "dist_to_needles": "final_dist_to_needles",
        "pct_matched": "final_pct_matched",
        "dup_fraction": "final_dup_fraction",
        "runtime_s": "final_runtime_s",
    }
    final = final.rename(columns={key: value for key, value in rename.items() if key in final.columns})

    status_cols = [
        "report_label",
        "optimizer",
        "seed",
        "run_dir",
        "status",
        "error",
        "line_budget_requested",
        "line_budget_reached",
    ]
    if not run_index.empty:
        available_status_cols = [col for col in status_cols if col in run_index.columns]
        final = final.merge(
            run_index[available_status_cols].drop_duplicates(RUN_KEYS),
            on=RUN_KEYS,
            how="left",
            suffixes=("", "_run"),
        )

    if "status" not in final.columns:
        final["status"] = "success"
    else:
        final["status"] = final["status"].fillna("success")
    if "error" not in final.columns:
        final["error"] = ""
    else:
        final["error"] = final["error"].fillna("")

    if not run_index.empty:
        existing_keys = set(_key_tuples(final, RUN_KEYS))
        missing_runs = run_index[[key not in existing_keys for key in _key_tuples(run_index, RUN_KEYS)]].copy()
        for col in final.columns:
            if col not in missing_runs.columns:
                missing_runs[col] = np.nan
        final = pd.concat([final, missing_runs[final.columns]], ignore_index=True, sort=False)

    if not line_metrics.empty:
        for col in ["line_best_y", "line_length_l2", "line_length_ilr"]:
            if col in line_metrics.columns:
                line_metrics[col] = pd.to_numeric(line_metrics[col], errors="coerce")
            else:
                line_metrics[col] = np.nan
        line_summary = (
            line_metrics.groupby(RUN_KEYS, dropna=False)
            .agg(
                line_best_y_mean=("line_best_y", "mean"),
                line_best_y_max=("line_best_y", "max"),
                line_length_l2_mean=("line_length_l2", "mean"),
                line_length_ilr_mean=("line_length_ilr", "mean"),
            )
            .reset_index()
        )
        final = final.merge(line_summary, on=RUN_KEYS, how="left", suffixes=("", "_line"))
        for col in ["line_best_y_mean", "line_best_y_max", "line_length_l2_mean", "line_length_ilr_mean"]:
            line_col = f"{col}_line"
            if line_col in final.columns:
                final[col] = pd.to_numeric(final.get(col), errors="coerce")
                final[col] = final[col].combine_first(final[line_col])
                final = final.drop(columns=[line_col])

    for col in FINAL_METRICS:
        if col in final.columns:
            final[col] = pd.to_numeric(final[col], errors="coerce")
    for col in ["seed", "num_points", "num_lines"]:
        if col in final.columns:
            final[col] = pd.to_numeric(final[col], errors="coerce")

    preferred = [
        "report_label",
        "objective_family",
        "objective_kind",
        "objective",
        "mode",
        "optimizer",
        "optimizer_kind",
        "seed",
        "status",
        "error",
        "final_best_y_so_far",
        "final_dist_to_needles",
        "final_pct_matched",
        "final_dup_fraction",
        "final_runtime_s",
        "num_points",
        "num_lines",
        "line_best_y_mean",
        "line_best_y_max",
        "line_length_l2_mean",
        "line_length_ilr_mean",
        "run_dir",
        "suite_name",
        "aggregate_dir",
    ]
    return _order_columns(final, preferred)


def summarize_final_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, "n_runs", "n_success"])

    metric_cols = [col for col in FINAL_METRICS if col in df.columns]
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        success = group[group.get("status", "success").fillna("success") == "success"]
        row["n_runs"] = int(len(group))
        row["n_success"] = int(len(success))
        for metric in metric_cols:
            values = pd.to_numeric(success[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else np.nan)
            row[f"{metric}_count"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def compute_auc_metrics_by_run(loaded: list[LoadedAggregate]) -> pd.DataFrame:
    metrics = _concat([item.metrics for item in loaded])
    if metrics.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for keys, group in metrics.groupby(RUN_KEYS, dropna=False):
        report_label, optimizer, seed, run_dir = keys
        group = group.copy()
        mode = str(group["mode"].iloc[0])
        axis_col = _axis_column(group, mode)
        row = {
            "report_label": report_label,
            "objective_family": group["objective_family"].iloc[0],
            "objective_kind": group["objective_kind"].iloc[0],
            "objective": group["objective"].iloc[0],
            "mode": mode,
            "optimizer": optimizer,
            "optimizer_kind": group["optimizer_kind"].iloc[0] if "optimizer_kind" in group else "",
            "seed": seed,
            "run_dir": run_dir,
            "auc_axis": axis_col,
        }
        for metric in CONVERGENCE_METRICS:
            row[f"auc_{metric}"] = _normalized_auc(group, axis_col, metric)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_transfer_rank_delta(final_by_run: pd.DataFrame, metrics_cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    if final_by_run.empty:
        return pd.DataFrame()

    directions = _rank_directions(metrics_cfg or {})
    rank_metrics = [metric for metric in RANK_METRICS if metric in final_by_run.columns]
    success = final_by_run[final_by_run.get("status", "success").fillna("success") == "success"].copy()
    if success.empty:
        return pd.DataFrame()

    grouped = (
        success.groupby(["objective_family", "mode", "optimizer"], dropna=False)[rank_metrics]
        .mean(numeric_only=True)
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for mode in sorted(grouped["mode"].dropna().unique()):
        mode_df = grouped[grouped["mode"] == mode]
        synthetic = mode_df[mode_df["objective_family"] == "synthetic"]
        real = mode_df[mode_df["objective_family"] == "real_rf"]
        if synthetic.empty or real.empty:
            continue
        for metric in rank_metrics:
            direction = directions.get(metric, RANK_METRICS.get(metric, "higher"))
            ascending = direction == "lower"
            syn = synthetic[["optimizer", metric]].dropna().rename(columns={metric: "synthetic_value"})
            real_df = real[["optimizer", metric]].dropna().rename(columns={metric: "real_value"})
            merged = syn.merge(real_df, on="optimizer", how="inner")
            if merged.empty:
                continue
            merged["synthetic_rank"] = merged["synthetic_value"].rank(ascending=ascending, method="min")
            merged["real_rank"] = merged["real_value"].rank(ascending=ascending, method="min")
            merged["rank_delta"] = merged["real_rank"] - merged["synthetic_rank"]
            for _, row in merged.sort_values(["rank_delta", "optimizer"]).iterrows():
                rows.append(
                    {
                        "mode": mode,
                        "metric": metric,
                        "direction": direction,
                        "optimizer": row["optimizer"],
                        "synthetic_value": row["synthetic_value"],
                        "real_value": row["real_value"],
                        "synthetic_rank": int(row["synthetic_rank"]),
                        "real_rank": int(row["real_rank"]),
                        "rank_delta": int(row["rank_delta"]),
                    }
                )
    return pd.DataFrame(rows)


def generate_plots(
    plot_dir: Path,
    loaded: list[LoadedAggregate],
    final_by_run: pd.DataFrame,
    transfer_rank_delta: pd.DataFrame,
    config: dict[str, Any],
) -> list[Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    plots_cfg = config.get("plots", {})
    dpi = int(plots_cfg.get("dpi", 160))
    paths: list[Path] = []
    metrics = _concat([item.metrics for item in loaded])
    line_metrics = _concat([item.line_metrics for item in loaded])

    if plt is None:
        return _generate_placeholder_plots(plot_dir, metrics, line_metrics, final_by_run, transfer_rank_delta)

    for metric in CONVERGENCE_METRICS:
        if metric in metrics.columns:
            path = plot_dir / f"convergence_{metric}.png"
            _plot_convergence(metrics, metric, path, plots_cfg, dpi)
            paths.append(path)

    if "final_runtime_s" in final_by_run.columns and not final_by_run.empty:
        path = plot_dir / "runtime_summary.png"
        _plot_runtime_summary(final_by_run, path, dpi)
        paths.append(path)

    if not line_metrics.empty:
        if "line_best_y" in line_metrics.columns:
            path = plot_dir / "line_best_y_by_line.png"
            _plot_line_metric(line_metrics, "line_best_y", path, dpi)
            paths.append(path)
        for metric in ["line_length_l2", "line_length_ilr"]:
            if metric in line_metrics.columns:
                path = plot_dir / f"{metric}_distribution.png"
                _plot_line_distribution(line_metrics, metric, path, dpi)
                paths.append(path)

    if not transfer_rank_delta.empty:
        path = plot_dir / "synthetic_real_rank_delta.png"
        _plot_rank_delta(transfer_rank_delta, path, dpi)
        paths.append(path)

    return paths


def _generate_placeholder_plots(
    plot_dir: Path,
    metrics: pd.DataFrame,
    line_metrics: pd.DataFrame,
    final_by_run: pd.DataFrame,
    transfer_rank_delta: pd.DataFrame,
) -> list[Path]:
    paths: list[Path] = []
    for metric in CONVERGENCE_METRICS:
        if metric in metrics.columns:
            path = plot_dir / f"convergence_{metric}.png"
            _write_placeholder_png(path, f"convergence_{metric}")
            paths.append(path)
    if "final_runtime_s" in final_by_run.columns:
        path = plot_dir / "runtime_summary.png"
        _write_placeholder_png(path, "runtime_summary")
        paths.append(path)
    if not line_metrics.empty:
        for name in ["line_best_y_by_line", "line_length_l2_distribution", "line_length_ilr_distribution"]:
            path = plot_dir / f"{name}.png"
            _write_placeholder_png(path, name)
            paths.append(path)
    if not transfer_rank_delta.empty:
        path = plot_dir / "synthetic_real_rank_delta.png"
        _write_placeholder_png(path, "synthetic_real_rank_delta")
        paths.append(path)
    return paths


def _write_placeholder_png(path: Path, title: str) -> None:
    message = [
        title,
        "matplotlib is not installed in this Python environment.",
        "Install matplotlib to regenerate data plots.",
    ]
    try:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (960, 540), "white")
        draw = ImageDraw.Draw(image)
        y = 180
        for line in message:
            draw.text((80, y), line, fill="black")
            y += 34
        image.save(path)
    except Exception:
        path.write_bytes(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
                "de0000000c4944415408d763f8ffff3f0005fe02fea73581e80000000049454e44ae426082"
            )
        )


def build_manifest(
    config: dict[str, Any],
    loaded: list[LoadedAggregate],
    report_dir: Path,
    plot_paths: list[Path],
) -> dict[str, Any]:
    inputs = []
    for item in loaded:
        run_index = item.run_index
        statuses = run_index["status"].fillna("unknown") if "status" in run_index else pd.Series(dtype=object)
        inputs.append(
            {
                "label": item.label,
                "aggregate_dir": str(item.aggregate_dir),
                "suite_config": None if item.suite_config_path is None else str(item.suite_config_path),
                "suite_name": item.suite_name,
                "objective_family": item.objective_family,
                "objective_kind": item.objective_kind,
                "objective": item.objective,
                "mode": item.mode,
                "num_runs": int(len(run_index)),
                "num_success": int((statuses == "success").sum()),
                "num_failed": int((statuses != "success").sum()),
            }
        )
    return {
        "name": config.get("name", "benchmark_report"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_dir": str(report_dir),
        "plot_backend": "matplotlib" if plt is not None else "placeholder_missing_matplotlib",
        "inputs": inputs,
        "outputs": {
            "report_md": str(report_dir / "report.md"),
            "final_metrics_by_run": str(report_dir / "final_metrics_by_run.csv"),
            "final_metrics_by_optimizer": str(report_dir / "final_metrics_by_optimizer.csv"),
            "final_metrics_by_optimizer_objective_mode": str(
                report_dir / "final_metrics_by_optimizer_objective_mode.csv"
            ),
            "auc_metrics_by_run": str(report_dir / "auc_metrics_by_run.csv"),
            "transfer_rank_delta": str(report_dir / "transfer_rank_delta.csv"),
            "plots": [str(path) for path in plot_paths],
        },
    }


def write_report_markdown(
    path: Path,
    loaded: list[LoadedAggregate],
    final_by_detail: pd.DataFrame,
    transfer_rank_delta: pd.DataFrame,
    manifest: dict[str, Any],
    plot_paths: list[Path],
) -> None:
    status_rows = pd.DataFrame(manifest["inputs"])
    status_cols = [
        "label",
        "objective_family",
        "mode",
        "num_runs",
        "num_success",
        "num_failed",
        "aggregate_dir",
    ]
    headline_cols = [
        "objective_family",
        "mode",
        "optimizer",
        "n_success",
        "final_best_y_so_far_mean",
        "final_dist_to_needles_mean",
        "final_pct_matched_mean",
        "final_dup_fraction_mean",
        "final_runtime_s_mean",
        "line_best_y_max_mean",
        "line_length_ilr_mean_mean",
    ]
    rank_cols = [
        "mode",
        "metric",
        "optimizer",
        "synthetic_rank",
        "real_rank",
        "rank_delta",
        "synthetic_value",
        "real_value",
    ]

    lines = [
        "# Milestone 1A Synthetic-to-Real Report",
        "",
        "This report compares synthetic 3D simplex and real 3D RF-surrogate benchmark outputs across point and line mode.",
        "",
        "## Loaded Suites",
        "",
        _markdown_table(status_rows[[col for col in status_cols if col in status_rows.columns]]),
        "",
        "## Final Metrics by Optimizer, Objective, and Mode",
        "",
        _markdown_table(final_by_detail[[col for col in headline_cols if col in final_by_detail.columns]], max_rows=40),
        "",
        "## Synthetic-to-Real Rank Deltas",
        "",
        "Negative rank deltas mean an optimizer ranked better on the real RF surrogate than on synthetic; positive values mean it ranked worse.",
        "",
        _markdown_table(
            transfer_rank_delta[[col for col in rank_cols if col in transfer_rank_delta.columns]],
            max_rows=80,
        ),
        "",
        "## Plots",
        "",
    ]
    for plot_path in plot_paths:
        rel = plot_path.relative_to(path.parent).as_posix()
        title = plot_path.stem.replace("_", " ")
        lines.extend([f"### {title}", "", f"![{title}]({rel})", ""])

    lines.extend(
        [
            "## Caveats",
            "",
            "- The real RF objective is a surrogate benchmark, not hardware validation.",
            "- The default real RF configs use `data/campaign1a.csv`, target `Objective`, components `[FAPbI3, MAPbI3, MAPbBr3]`, and the bundled reference optima JSON when present.",
            "- ZoMBI-Hop line mode uses internal candidate-anchor plus LineBO, while GP/RF/random use benchmark-local line wrappers.",
            "- All line methods share the same line budget, points per line, initial design convention, objective, seed convention, and batched within-line update rule.",
            "- Plot PNGs are generated with matplotlib when it is installed; otherwise placeholder PNGs are written and the CSV tables remain the source of record.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_convergence(metrics: pd.DataFrame, metric: str, path: Path, plots_cfg: dict[str, Any], dpi: int) -> None:
    df = metrics.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    combos = _objective_mode_combos(df)
    fig, axes = _subplots_for_combos(combos)
    include_seed_traces = bool(plots_cfg.get("include_seed_traces", True))
    include_iqr = bool(plots_cfg.get("include_iqr_band", True))
    for ax, (objective_family, mode) in zip(axes, combos):
        subset = df[(df["objective_family"] == objective_family) & (df["mode"] == mode)].copy()
        axis_col = _axis_column(subset, str(mode))
        subset[axis_col] = pd.to_numeric(subset[axis_col], errors="coerce")
        subset = subset.dropna(subset=[axis_col, metric])
        for optimizer, opt_df in subset.groupby("optimizer"):
            curves = []
            for _, run_df in opt_df.groupby(["seed", "run_dir"], dropna=False):
                curve = _curve_by_axis(run_df, axis_col, metric)
                if curve.empty:
                    continue
                curves.append(curve)
                if include_seed_traces:
                    ax.plot(curve[axis_col], curve[metric], alpha=0.22, linewidth=0.8)
            if not curves:
                continue
            curves_df = pd.concat([c.assign(_curve=i) for i, c in enumerate(curves)], ignore_index=True)
            grouped = curves_df.groupby(axis_col)[metric]
            median = grouped.median()
            ax.plot(median.index, median.values, label=str(optimizer), linewidth=2.0)
            if include_iqr:
                q25 = grouped.quantile(0.25)
                q75 = grouped.quantile(0.75)
                ax.fill_between(median.index, q25.values, q75.values, alpha=0.12)
        ax.set_title(f"{objective_family} / {mode}")
        ax.set_xlabel(axis_col)
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    _hide_unused_axes(axes, len(combos))
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_runtime_summary(final_by_run: pd.DataFrame, path: Path, dpi: int) -> None:
    df = final_by_run[final_by_run.get("status", "success").fillna("success") == "success"].copy()
    df["final_runtime_s"] = pd.to_numeric(df["final_runtime_s"], errors="coerce")
    combos = _objective_mode_combos(df)
    fig, axes = _subplots_for_combos(combos)
    for ax, (objective_family, mode) in zip(axes, combos):
        subset = df[(df["objective_family"] == objective_family) & (df["mode"] == mode)].dropna(
            subset=["final_runtime_s"]
        )
        grouped = subset.groupby("optimizer")["final_runtime_s"].mean().sort_values()
        ax.bar(grouped.index.astype(str), grouped.values)
        ax.set_title(f"{objective_family} / {mode}")
        ax.set_ylabel("runtime_s")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    _hide_unused_axes(axes, len(combos))
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_line_metric(line_metrics: pd.DataFrame, metric: str, path: Path, dpi: int) -> None:
    df = line_metrics.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df["line_index"] = pd.to_numeric(df["line_index"], errors="coerce")
    combos = _objective_mode_combos(df)
    fig, axes = _subplots_for_combos(combos)
    for ax, (objective_family, mode) in zip(axes, combos):
        subset = df[(df["objective_family"] == objective_family) & (df["mode"] == mode)].dropna(
            subset=["line_index", metric]
        )
        for optimizer, opt_df in subset.groupby("optimizer"):
            grouped = opt_df.groupby("line_index")[metric].median()
            ax.plot(grouped.index, grouped.values, marker="o", label=str(optimizer))
        ax.set_title(f"{objective_family} / {mode}")
        ax.set_xlabel("line_index")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    _hide_unused_axes(axes, len(combos))
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_line_distribution(line_metrics: pd.DataFrame, metric: str, path: Path, dpi: int) -> None:
    df = line_metrics.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])
    labels = []
    values = []
    for (objective_family, optimizer), group in df.groupby(["objective_family", "optimizer"], dropna=False):
        labels.append(f"{objective_family}\n{optimizer}")
        values.append(group[metric].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.7), 4.5))
    if values:
        _boxplot_with_labels(ax, values, labels)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _boxplot_with_labels(ax, values, labels) -> None:
    try:
        ax.boxplot(values, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(values, labels=labels, showfliers=False)


def _plot_rank_delta(rank_delta: pd.DataFrame, path: Path, dpi: int) -> None:
    df = rank_delta.copy()
    df["row"] = df["mode"].astype(str) + " / " + df["optimizer"].astype(str)
    pivot = df.pivot_table(index="row", columns="metric", values="rank_delta", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(6.5, pivot.shape[1] * 1.5), max(3.5, pivot.shape[0] * 0.35)))
    if not pivot.empty:
        data = pivot.to_numpy(dtype=float)
        max_abs = np.nanmax(np.abs(data)) if not np.isnan(data).all() else 1.0
        max_abs = max(max_abs, 1.0)
        image = ax.imshow(data, cmap="coolwarm", vmin=-max_abs, vmax=max_abs, aspect="auto")
        fig.colorbar(image, ax=ax, label="real rank - synthetic rank")
        ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = data[i, j]
                if not np.isnan(value):
                    ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7)
    ax.set_title("Synthetic-to-real rank delta")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _annotate(
    df: pd.DataFrame,
    label: str,
    suite_name: str,
    objective: str,
    objective_kind: str,
    objective_family: str,
    mode: str,
    aggregate_dir: Path,
) -> pd.DataFrame:
    df = df.copy()
    for col, value in [
        ("report_label", label),
        ("suite_name", suite_name),
        ("objective", objective),
        ("objective_kind", objective_kind),
        ("objective_family", objective_family),
        ("mode", mode),
        ("aggregate_dir", str(aggregate_dir)),
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(value)
        else:
            df[col] = value
    return df


def _resolve_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _resolve_optional_path(path: str | Path | None, repo_root: Path) -> Path | None:
    if path in (None, ""):
        return None
    return _resolve_path(path, repo_root)


def _infer_mode(label: str) -> str:
    return "line" if "line" in label else "point"


def _infer_objective(label: str) -> str:
    return "real_3d_perovskite_rf" if "real" in label or "rf" in label else "synthetic_3d_planted"


def _infer_objective_kind(label: str) -> str:
    return "real_rf_surrogate" if "real" in label or "rf" in label else "synthetic_simplex"


def _infer_objective_family(label: str, objective_kind: str) -> str:
    if "real_rf" in objective_kind or "real" in label or "rf" in label:
        return "real_rf"
    return "synthetic"


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _key_tuples(df: pd.DataFrame, keys: list[str]) -> list[tuple[Any, ...]]:
    if df.empty:
        return []
    return [tuple(row) for row in df[keys].astype(str).itertuples(index=False, name=None)]


def _order_columns(df: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    columns = [col for col in preferred if col in df.columns]
    columns.extend([col for col in df.columns if col not in columns])
    return df[columns]


def _axis_column(df: pd.DataFrame, mode: str) -> str:
    if mode == "line" and "num_lines" in df.columns and pd.to_numeric(df["num_lines"], errors="coerce").notna().any():
        return "num_lines"
    if "num_points" in df.columns and pd.to_numeric(df["num_points"], errors="coerce").notna().any():
        return "num_points"
    if "line_index" in df.columns and pd.to_numeric(df["line_index"], errors="coerce").notna().any():
        return "line_index"
    return "step"


def _curve_by_axis(df: pd.DataFrame, axis_col: str, metric: str) -> pd.DataFrame:
    curve = df[[axis_col, metric]].copy()
    curve[axis_col] = pd.to_numeric(curve[axis_col], errors="coerce")
    curve[metric] = pd.to_numeric(curve[metric], errors="coerce")
    curve = curve.dropna(subset=[axis_col, metric]).sort_values(axis_col)
    if curve.empty:
        return curve
    return curve.groupby(axis_col, as_index=False)[metric].last()


def _normalized_auc(df: pd.DataFrame, axis_col: str, metric: str) -> float:
    if axis_col not in df.columns or metric not in df.columns:
        return math.nan
    curve = _curve_by_axis(df, axis_col, metric)
    if curve.empty:
        return math.nan
    x = curve[axis_col].to_numpy(dtype=float)
    y = curve[metric].to_numpy(dtype=float)
    if len(x) < 2 or float(np.max(x) - np.min(x)) == 0.0:
        return float(y[-1])
    return float(np.trapezoid(y, x) / (np.max(x) - np.min(x)))


def _rank_directions(metrics_cfg: dict[str, Any]) -> dict[str, str]:
    directions = dict(RANK_METRICS)
    for metric in metrics_cfg.get("higher_is_better", []):
        directions[_final_metric_name(metric)] = "higher"
    for metric in metrics_cfg.get("lower_is_better", []):
        directions[_final_metric_name(metric)] = "lower"
    return directions


def _final_metric_name(metric: str) -> str:
    return metric if metric.startswith("final_") else f"final_{metric}"


def _objective_mode_combos(df: pd.DataFrame) -> list[tuple[str, str]]:
    if df.empty:
        return []
    combos = df[["objective_family", "mode"]].drop_duplicates()
    order = {"synthetic": 0, "real_rf": 1, "point": 0, "line": 1}
    return sorted(
        [(str(row.objective_family), str(row.mode)) for row in combos.itertuples()],
        key=lambda item: (order.get(item[0], 99), order.get(item[1], 99), item[0], item[1]),
    )


def _subplots_for_combos(combos: list[tuple[str, str]]):
    n = max(len(combos), 1)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 3.8 * nrows), squeeze=False)
    return fig, list(axes.ravel())


def _hide_unused_axes(axes, used: int) -> None:
    for ax in axes[used:]:
        ax.set_visible(False)


def _markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    out = df.head(max_rows).copy()
    for col in out.columns:
        out[col] = out[col].map(_format_markdown_value)
    header = "| " + " | ".join(map(str, out.columns)) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in out.itertuples(index=False, name=None)]
    if len(df) > max_rows:
        rows.append(f"| ... | {' | '.join([''] * (len(out.columns) - 1))} |")
    return "\n".join([header, sep, *rows])


def _format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate Milestone 1A benchmark reports.")
    parser.add_argument("--config", required=True, help="Path to report YAML config")
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    report_dir = run_report(config, Path.cwd())
    print(report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
