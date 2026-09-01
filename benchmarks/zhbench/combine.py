"""Combine suite runs into one bundle, recording which core produced each arm.

Needed because a re-run is rarely whole. The core merge at ``baa51de`` invalidated
the two ZoMBI-Hop arms and left the four ``random`` / ``gp_*`` arms untouched
(verified at symbol level, ``test_core_pins``), so re-running all 180 cells would
have burned ~40 CPU-hours to reproduce 120 of them exactly. The honest alternative
is a bundle whose arms come from different runs -- which is fine, and *only* fine if
the artifact says so per arm rather than carrying one suite-level SHA that is wrong
for two thirds of the rows.

Precedence is last-wins per ``(objective, optimizer, seed)``. Sources are given
oldest first, so a re-run listed last replaces the rows it supersedes and nothing
else. A cell present in two sources is REPLACED, never averaged or appended --
silently pooling two cores would be the worst possible outcome here.

    python -m benchmarks.zhbench.combine OUT SRC1 SRC2 [...]

Writes ``aggregate.csv``, ``curves.json``, ``matched_curves.json`` and
``provenance.json`` into OUT. The matched |S| is computed on the COMBINED table and
pushed down into each source, because deriving it per source would score each half
at its own |S| and make the halves incomparable -- the exact failure the
matched-|S| basis exists to prevent.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

from . import stats as S


def _cell_key(row: dict) -> tuple:
    return (row["objective"], row["optimizer"], int(float(row["seed"])))


def _config_of(suite_dir: str, cell: str) -> dict:
    p = os.path.join(suite_dir, cell, "config_resolved.json")
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _git_of(suite_dir: str, cell: str) -> dict:
    """The commit a cell was actually run at, from its own artifact."""
    return _config_of(suite_dir, cell).get("git", {})


# Hyperparameters worth carrying into the combined bundle. A combined directory has
# no cell subdirectories, so anything that wants to READ what actually ran -- e.g.
# labelling the nc sensitivity "1 vs 5" at 3-D rather than assuming a uniform 2 --
# would otherwise have to fall back to a generic label or, worse, to whatever the
# config files happen to say today.
_CARRY = ("n_consecutive_converged", "min_zoom_for_needle", "min_iters_per_zoom",
          "max_iterations", "max_zooms", "input_noise")


def _resolved_of(suite_dir: str, cell: str) -> dict:
    st = _config_of(suite_dir, cell).get("optimizer_state") or {}
    rh = st.get("resolved_hparams") or {}
    return {k: rh[k] for k in _CARRY if k in rh}


def combine(out_dir: str, sources: list[str], reference: str = "zombihop") -> dict:
    os.makedirs(out_dir, exist_ok=True)

    merged: dict[tuple, dict] = {}
    merged_curves: dict[str, dict] = {}
    origin: dict[tuple, str] = {}          # cell key -> source dir
    cell_names: dict[tuple, str] = {}      # cell key -> cell dir name

    for src in sources:
        # Read the CSV RAW rather than through stats.load, which coerces every
        # non-text column to float: a carried-over row would come back out with
        # seed "5.0" and n_samples "2000.0", so the combined bundle would not be
        # byte-comparable to the source it was copied from. Nothing here needs the
        # numeric view -- matched_k is computed from the written file below.
        with open(os.path.join(src, "aggregate.csv"), encoding="utf-8") as fh:
            raw = list(csv.DictReader(fh))
        with open(os.path.join(src, "curves.json"), encoding="utf-8") as fh:
            curves = json.load(fh)
        for r in raw:
            if r.get("error"):
                continue
            merged[_cell_key(r)] = r
            origin[_cell_key(r)] = src
        for cell, c in curves.items():
            # curves are keyed by cell NAME, which is stable across runs of the
            # same config, so the same last-wins rule applies.
            key = (c.get("objective"), c.get("optimizer"), S._seed_of(cell))
            if key in merged:
                merged_curves[cell] = c
                cell_names[key] = cell

    rows = list(merged.values())
    if not rows:
        raise SystemExit("no successful cells in any source")

    # Provenance PER ARM, not per bundle. An arm whose cells span two commits is
    # reported as such rather than collapsed to the first one seen.
    prov: dict[str, dict] = {}
    for key, r in merged.items():
        arm = f"{r['objective']} / {r['optimizer']}"
        src = origin[key]
        commit = _git_of(src, cell_names.get(key, "")).get("commit", "")
        e = prov.setdefault(arm, {"source_dirs": set(), "commits": set(), "n": 0,
                                  "resolved": {}})
        e["source_dirs"].add(os.path.basename(os.path.normpath(src)))
        if commit:
            e["commits"].add(commit[:8])
        e["n"] += 1
        if not e["resolved"]:
            e["resolved"] = _resolved_of(src, cell_names.get(key, ""))
    prov = {k: {"source_dirs": sorted(v["source_dirs"]),
                "commits": sorted(v["commits"]), "n_cells": v["n"],
                "resolved_hparams": v["resolved"]}
            for k, v in prov.items()}

    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(os.path.join(out_dir, "aggregate.csv"), "w", encoding="utf-8",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(os.path.join(out_dir, "curves.json"), "w", encoding="utf-8") as fh:
        json.dump(merged_curves, fh)

    # Matched |S| from the COMBINED table, then pushed down per source.
    combined_rows, _ = S.load(out_dir)
    ks = {}
    for obj in S.objectives(combined_rows):
        try:
            ks[obj] = S.matched_k(combined_rows, obj, reference)
        except ValueError:
            continue          # no reference arm for this objective; skip it
    mc: dict[str, dict] = {}
    for src in sources:
        try:
            mc.update(S.matched_curves(src, reference, ks=ks))
        except (FileNotFoundError, KeyError):
            continue
    mc = {cell: v for cell, v in mc.items() if cell in merged_curves}
    with open(os.path.join(out_dir, "matched_curves.json"), "w", encoding="utf-8") as fh:
        json.dump(mc, fh)

    manifest = {"sources": [os.path.abspath(s) for s in sources],
                "precedence": "last wins per (objective, optimizer, seed)",
                "n_cells": len(rows), "matched_k": ks, "per_arm": prov}
    with open(os.path.join(out_dir, "provenance.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return manifest


def provenance_markdown(manifest: dict) -> list[str]:
    """The per-arm provenance table for RESULTS.md."""
    out = ["| arm | cells | core commit(s) | from |", "|---|---|---|---|"]
    for arm in sorted(manifest["per_arm"]):
        e = manifest["per_arm"][arm]
        out.append(f"| `{arm}` | {e['n_cells']} | "
                   f"{', '.join('`%s`' % c for c in e['commits']) or '--'} | "
                   f"{', '.join('`%s`' % d for d in e['source_dirs'])} |")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("sources", nargs="+",
                    help="suite dirs, OLDEST FIRST (last wins per cell)")
    ap.add_argument("--reference", default="zombihop")
    a = ap.parse_args(argv)
    man = combine(a.out, a.sources, a.reference)
    print(f"{man['n_cells']} cells -> {a.out}")
    print(f"matched |S|: {man['matched_k']}")
    print("\n".join(provenance_markdown(man)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
