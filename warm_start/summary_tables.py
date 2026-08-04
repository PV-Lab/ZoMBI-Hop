"""
warm_start/summary_tables.py
============================
Per-run FULLGP summary tables for a deploy+eval run.

``deploy_eval.sbatch`` evaluates each method (baseline / static / dynamic / mixture)
on the full-run GP landscape, writing one directory per run
(``eval/<method>/fullgp/trial_0/run_<i>/``) plus an aggregated
``eval/<method>/fullgp/rerun_summary.json``. ``compare_methods.py`` collapses those
into one mean±std comparison; this script instead expands the FULLGP results into
one markdown table PER METHOD, one row per run, so the individual runs and their
plots can be eyeballed at a glance.

Columns: run index, dist to needles, dup fraction, then the plots. Which plots
depends on the dimensionality: 3D runs get the final ``iter_*.png``, the
``convergence.png`` and the ternary ``coverage.png``; 4D runs get ONLY the
``convergence.png`` (their final state is an interactive point_cloud.html and the
coverage plot is a tetrahedron — neither embeds usefully in markdown). The plot
columns embed the images with paths relative to the output file so the markdown
renders inline. ENSEMBLE results are intentionally excluded — this is a
FULLGP-only view.

Pure stdlib (json) — cheap to run after the jobs.

Usage
-----
    python warm_start/summary_tables.py --base optimize/runs/deploy_3d_job123/eval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METHODS = ["baseline", "static", "dynamic", "mixture"]
LANDSCAPE = "fullgp"  # FULLGP only, by design.

# (column header, glob) per dimensionality. 4D keeps convergence only — see module
# docstring.
PLOT_COLUMNS = {
    3: [("plot", "iter_*.png"),
        ("convergence", "convergence.png"),
        ("coverage", "coverage.png")],
    4: [("convergence", "convergence.png")],
}


def _run_dim(run_dir: Path) -> int | None:
    """Dimensionality recorded in a run's run_config.json, if readable."""
    cfg = run_dir / "run_config.json"
    if not cfg.exists():
        return None
    try:
        return json.loads(cfg.read_text()).get("dim")
    except (json.JSONDecodeError, OSError):
        return None


def _per_run_metrics(summary_path: Path) -> dict[int, dict]:
    """Map run index -> {dist_to_needles, dup_fraction} from a rerun_summary.json."""
    if not summary_path.exists():
        return {}
    data = json.loads(summary_path.read_text())
    trials = data.get("trials", [])
    if not trials:
        return {}
    out: dict[int, dict] = {}
    for run in trials[0].get("runs", []):
        out[run["run"]] = run
    return out


def _img_cell(run_dir: Path, pattern: str, out_dir: Path) -> str:
    """Markdown image (path relative to out_dir) for the first file matching pattern."""
    matches = sorted(run_dir.glob(pattern))
    if not matches:
        return "—"
    rel = matches[0].relative_to(out_dir)
    return f"![]({rel})"


def _fmt(val) -> str:
    return f"{val:.4f}" if isinstance(val, (int, float)) else "—"


def build_method_table(method: str, base: Path, out_dir: Path,
                       dim: int | None = None) -> str | None:
    """Markdown section for one method, or None if it has no FULLGP results."""
    fullgp = base / method / LANDSCAPE / "trial_0"
    if not fullgp.is_dir():
        return None
    metrics = _per_run_metrics(base / method / LANDSCAPE / "rerun_summary.json")
    run_dirs = sorted(fullgp.glob("run_*"), key=lambda p: int(p.name.split("_")[1]))
    if not run_dirs:
        return None

    # Dim decides the plot columns; read it off the first run unless overridden.
    resolved = dim or _run_dim(run_dirs[0]) or 3
    plot_cols = PLOT_COLUMNS.get(resolved, PLOT_COLUMNS[3])

    headers = ["run", "dist to needles", "dup fraction"] + [h for h, _ in plot_cols]
    lines = [f"## {method}", "",
             "| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for run_dir in run_dirs:
        idx = int(run_dir.name.split("_")[1])
        m = metrics.get(idx, {})
        cells = [
            str(idx),
            _fmt(m.get("dist_to_needles")),
            _fmt(m.get("dup_fraction")),
        ] + [_img_cell(run_dir, pattern, out_dir) for _, pattern in plot_cols]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-run FULLGP summary tables (one per method).")
    parser.add_argument("--base", required=True,
                        help="Eval dir holding <method>/fullgp/... (e.g. .../eval).")
    parser.add_argument("--out", default=None,
                        help="Output markdown path (default: <base>/summary_fullgp.md).")
    parser.add_argument("--dim", type=int, default=None, choices=[3, 4],
                        help="Override the dimensionality that picks the plot columns "
                             "(default: read from each run's run_config.json).")
    args = parser.parse_args()
    base = Path(args.base)
    out_path = Path(args.out) if args.out else base / "summary_fullgp.md"
    out_dir = out_path.parent

    sections = ["# FULLGP per-run summary", ""]
    found = False
    for method in METHODS:
        table = build_method_table(method, base, out_dir, args.dim)
        if table is None:
            print(f"  [summary] missing: {method}/{LANDSCAPE} (skipped)")
            continue
        found = True
        sections.append(table)
    if not found:
        raise SystemExit(f"no {LANDSCAPE} results found under {base}")
    sections.append("_dist to needles / dup fraction: lower is better._")
    out_path.write_text("\n".join(sections) + "\n")
    print(f"  summary -> {out_path}")


if __name__ == "__main__":
    main()
