"""
optimize/summary_table.py
=========================
Markdown summary of a showdown, with rows grouped **by landscape**.

``pareto.write_top_trials_summary`` and ``collect_rerun_summaries`` both group by
trial: all of one configuration's runs, then all of the next. That is the right
shape when each configuration was scored on its own landscapes, because the rows
within a group are comparable to each other and nothing else.

A showdown inverts that. Every configuration sees the SAME landscapes, so the
comparison that matters is *across* configurations on one landscape — which is
only readable when those rows are adjacent. Hence: the first N rows are every
configuration on landscape 1, the next N on landscape 2, and so on. Reading down
a landscape block answers "which configuration handled this landscape best"; the
old layout scattered that across the table.

Written by ``optimize/showdown.py`` (directly, or via ``--summary-only``).
"""

from __future__ import annotations

import os
import json
import glob
import datetime


# Per-run plots shown as columns (file name -> column heading).
SUMMARY_PLOTS: list[tuple[str, str]] = [
    ("convergence.png",   "convergence"),
    ("conet.png",         "conet"),
    ("conet_uniform.png", "conet uniform"),
]


def _read_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _md_img(src: str | None, alt: str, width: int = 460) -> str:
    """Markdown-table cell for a plot: a thumbnail linking to the full image.

    Table cells cannot hold block elements, so this uses an inline ``<img>`` (which
    every Markdown renderer passes through) rather than ``![]()`` sizing tricks.
    """
    if not src:
        return "—"
    return f'<a href="{src}"><img src="{src}" alt="{alt}" width="{width}"></a>'


def _fmt(v, nd: int = 4) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def find_run_dir(showdown_dir: str, config: str, landscape: int) -> str | None:
    """The per-run artifact directory for one (config, landscape) cell.

    ``evaluate.py`` nests its output as ``<out-dir>/trial_<n>/run_<k>``; a
    ``--hparams-json`` invocation lands on ``trial_0/run_1``, but the trial number
    is not guaranteed, so this globs rather than hard-coding it.
    """
    base = os.path.join(showdown_dir, "runs", f"{config}__ls{landscape}")
    hits = sorted(glob.glob(os.path.join(base, "trial_*", "run_*")))
    for h in hits:
        if os.path.isfile(os.path.join(h, "metrics.json")):
            return h
    return hits[0] if hits else None


def write_landscape_summary(showdown_dir: str, out_path: str | None = None) -> str:
    """Write the landscape-grouped Markdown table for a finished showdown.

    Image paths are written relative to the Markdown file so it renders in place.
    Cells for runs that have not finished (or failed) show as ``—`` rather than
    being dropped, so a partially-complete campaign is still readable and the gaps
    are visible.
    """
    showdown_dir = os.path.abspath(showdown_dir)
    manifest = _read_json(os.path.join(showdown_dir, "showdown_manifest.json"))
    if not manifest:
        raise SystemExit(f"no showdown_manifest.json in {showdown_dir}")

    configs = manifest["configs"]
    landscapes = manifest["landscapes"]
    out_path = out_path or os.path.join(showdown_dir, "showdown_summary.md")
    out_dir = os.path.dirname(out_path)

    L: list[str] = []
    A = L.append
    A(f"# Showdown — {manifest['dim']}D ensemble, "
      f"{len(configs)} configs × {len(landscapes)} landscapes")
    A("")
    A(f"Generated {datetime.datetime.now().isoformat(timespec='seconds')}. "
      f"Planned {manifest['generated']}.")
    A("")
    A(f"Every configuration was run on the **same** {len(landscapes)} ensemble "
      f"landscapes (Sobol indices `{landscapes}`, seed `{manifest['landscape_seed']}`) "
      f"at a {manifest['time_limit_hours']} h per-run budget, so differences between "
      "rows within a landscape block are attributable to the hyperparameters rather "
      "than to landscape luck.")
    A("")
    A("**Rows are grouped by landscape**, not by configuration: read down a block to "
      "compare all configurations on one landscape.")
    A("")

    A("## Configurations")
    A("")
    A("| config | selected for | rank | source run | trial | source dist | source dup |")
    A("|---|---|---|---|---|---|---|")
    for c in configs:
        m = c.get("source_metrics") or {}
        A(f"| `{c['name']}` | {c.get('selected_for') or '—'} | "
          f"{c.get('rank') if c.get('rank') is not None else '—'} | "
          f"`{c.get('source_run') or '—'}` | {c.get('trial') if c.get('trial') is not None else '—'} | "
          f"{_fmt(m.get('dist_to_needles'))} | {_fmt(m.get('dup_fraction'))} |")
    A("")
    A("`selected for` is the objective the configuration topped on the source Pareto "
      "front; `source dist` / `source dup` are its metrics THERE (each on its own "
      "landscapes), which is exactly what this showdown re-measures on common ground.")
    A("")

    A("## Results")
    A("")
    header = (["landscape", "config", "dist to needles", "dup fraction", "runtime (s)"]
              + [h for _, h in SUMMARY_PLOTS])
    A("| " + " | ".join(header) + " |")
    A("|" + "|".join(["---"] * len(header)) + "|")

    n_rows = n_missing = 0
    for ls in landscapes:
        for i, c in enumerate(configs):
            run_dir = find_run_dir(showdown_dir, c["name"], ls)
            met = _read_json(os.path.join(run_dir, "metrics.json")) if run_dir else {}
            if not met:
                n_missing += 1
            # Name the landscape once per block; repeating it on every row is what
            # makes the grouping hard to see.
            cells = [f"**{ls}**" if i == 0 else "",
                     f"`{c['name']}`",
                     _fmt(met.get("dist_to_needles")),
                     _fmt(met.get("dup_fraction")),
                     _fmt(met.get("runtime_s"), 1)]
            for fname, head in SUMMARY_PLOTS:
                path = os.path.join(run_dir, fname) if run_dir else None
                rel = (os.path.relpath(path, out_dir)
                       if path and os.path.isfile(path) else None)
                cells.append(_md_img(rel, f"{c['name']} ls{ls} {head}"))
            A("| " + " | ".join(cells) + " |")
            n_rows += 1

    A("")
    if n_missing:
        A(f"⚠ {n_missing} of {n_rows} cells have no `metrics.json` — those runs have "
          "not finished or failed; their rows are kept so the gaps stay visible.")
        A("")

    A("## Per-landscape winners")
    A("")
    A("| landscape | best dist to needles | best dup fraction |")
    A("|---|---|---|")
    for ls in landscapes:
        rows = []
        for c in configs:
            run_dir = find_run_dir(showdown_dir, c["name"], ls)
            met = _read_json(os.path.join(run_dir, "metrics.json")) if run_dir else {}
            if met.get("dist_to_needles") is not None:
                rows.append((c["name"], float(met["dist_to_needles"]),
                             float(met.get("dup_fraction", float("nan")))))
        if not rows:
            A(f"| {ls} | — | — |")
            continue
        bd = min(rows, key=lambda r: r[1])
        bu = min(rows, key=lambda r: r[2])
        A(f"| {ls} | `{bd[0]}` ({bd[1]:.4f}) | `{bu[0]}` ({bu[2]:.4f}) |")
    A("")

    with open(out_path, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"  summary ({n_rows} rows, {n_missing} incomplete) -> {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("showdown_dir", help="a showdown output directory")
    ap.add_argument("--out", default=None, help="output Markdown path")
    a = ap.parse_args()
    write_landscape_summary(a.showdown_dir, a.out)
