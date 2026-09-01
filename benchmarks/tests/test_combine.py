"""Acceptance tests for combining suite runs across cores.

The failure this guards against is silent pooling. A bundle whose ZoMBI-Hop arms
come from the merged core and whose baselines come from the August run is fine and
is what we want -- but only if a cell present in two sources is REPLACED rather than
averaged or appended, and only if the artifact records which core produced which
arm. Averaging two cores would be the worst possible outcome, and it would look
exactly like a normal aggregate.
"""

from __future__ import annotations

import csv
import json
import os

import pytest

from zhbench import combine as C

RUN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "runs", "s1_real_20260824_221242")
_has_run = os.path.exists(os.path.join(RUN_DIR, "curves.json"))


def _mksuite(path, rows, commit):
    """A minimal suite directory: aggregate.csv, curves.json, per-cell configs."""
    os.makedirs(path, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with open(os.path.join(path, "aggregate.csv"), "w", encoding="utf-8",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    curves = {}
    for r in rows:
        cell = f"o_{r['objective']}__{r['optimizer']}__s{r['seed']}"
        curves[cell] = {"objective": r["objective"], "optimizer": r["optimizer"],
                        "pr_curve_k": [1, 2, 3],
                        "pr_curve_peak_ratio": [0.1, 0.2, 0.3],
                        "by_n": {}}
        os.makedirs(os.path.join(path, cell), exist_ok=True)
        with open(os.path.join(path, cell, "config_resolved.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"git": {"commit": commit, "branch": "b", "dirty": False}}, fh)
    with open(os.path.join(path, "curves.json"), "w", encoding="utf-8") as fh:
        json.dump(curves, fh)


def _row(obj, opt, seed, peak, n_declared=6):
    return {"objective": obj, "optimizer": opt, "seed": seed,
            "peak_ratio": peak, "n_declared": n_declared, "n_true_optima": 14,
            "n_samples": 2000, "error": ""}


def test_later_source_replaces_a_cell_it_shares(tmp_path):
    old = str(tmp_path / "old")
    new = str(tmp_path / "new")
    _mksuite(old, [_row("real3d", "zombihop", 0, 0.50),
                   _row("real3d", "random", 0, 0.90)], "aaaaaaaa" * 5)
    _mksuite(new, [_row("real3d", "zombihop", 0, 0.10)], "bbbbbbbb" * 5)

    man = C.combine(str(tmp_path / "out"), [old, new])

    with open(os.path.join(tmp_path, "out", "aggregate.csv"), encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2, "a replaced cell must not be appended"
    zh = [r for r in rows if r["optimizer"] == "zombihop"]
    assert len(zh) == 1
    # the NEW value, not the old one and emphatically not the mean (0.30)
    assert float(zh[0]["peak_ratio"]) == pytest.approx(0.10)
    assert man["n_cells"] == 2


def test_provenance_is_per_arm_not_per_bundle(tmp_path):
    old = str(tmp_path / "old")
    new = str(tmp_path / "new")
    _mksuite(old, [_row("real3d", "zombihop", 0, 0.5),
                   _row("real3d", "random", 0, 0.9)], "a" * 40)
    _mksuite(new, [_row("real3d", "zombihop", 0, 0.1)], "b" * 40)

    man = C.combine(str(tmp_path / "out"), [old, new])
    per = man["per_arm"]
    assert per["real3d / zombihop"]["commits"] == ["b" * 8]
    assert per["real3d / random"]["commits"] == ["a" * 8]

    md = "\n".join(C.provenance_markdown(man))
    assert "real3d / zombihop" in md and "b" * 8 in md
    assert "a" * 8 in md


def test_an_arm_spanning_two_commits_is_reported_as_such(tmp_path):
    """Not collapsed to the first commit seen -- that is how a split goes unnoticed."""
    a = str(tmp_path / "a")
    b = str(tmp_path / "b")
    _mksuite(a, [_row("real3d", "random", 0, 0.9)], "a" * 40)
    _mksuite(b, [_row("real3d", "random", 1, 0.9)], "b" * 40)
    man = C.combine(str(tmp_path / "out"), [a, b])
    assert man["per_arm"]["real3d / random"]["commits"] == ["a" * 8, "b" * 8]
    assert man["per_arm"]["real3d / random"]["n_cells"] == 2


def test_errored_cells_are_dropped(tmp_path):
    src = str(tmp_path / "s")
    good = _row("real3d", "random", 0, 0.9)
    bad = _row("real3d", "random", 1, 0.0)
    bad["error"] = "timeout after 7200s"
    _mksuite(src, [good, bad], "a" * 40)
    man = C.combine(str(tmp_path / "out"), [src])
    assert man["n_cells"] == 1


def test_matched_k_comes_from_the_combined_table(tmp_path):
    """If the reference arm is replaced, |S| must follow the NEW declarations.

    Deriving it per source would score the re-run half at one |S| and the carried-
    over half at another, which silently makes the two incomparable -- the exact
    failure the matched-|S| basis exists to prevent.
    """
    old = str(tmp_path / "old")
    new = str(tmp_path / "new")
    _mksuite(old, [_row("real3d", "zombihop", s, 0.5, n_declared=12)
                   for s in range(4)], "a" * 40)
    _mksuite(new, [_row("real3d", "zombihop", s, 0.5, n_declared=4)
                   for s in range(4)], "b" * 40)
    man = C.combine(str(tmp_path / "out"), [old, new])
    assert man["matched_k"]["real3d"] == 4, man["matched_k"]


@pytest.mark.skipif(not _has_run, reason="s1_real run directory not present")
def test_single_source_is_a_faithful_no_op(tmp_path):
    out = str(tmp_path / "out")
    man = C.combine(out, [RUN_DIR])
    assert man["n_cells"] == 180
    assert man["matched_k"] == {"real3d": 6, "real4d": 15, "real6d": 15}

    with open(os.path.join(RUN_DIR, "aggregate.csv"), encoding="utf-8") as fh:
        src = {(r["objective"], r["optimizer"], r["seed"]): r
               for r in csv.DictReader(fh)}
    with open(os.path.join(out, "aggregate.csv"), encoding="utf-8") as fh:
        got = {(r["objective"], r["optimizer"], r["seed"]): r
               for r in csv.DictReader(fh)}
    assert set(src) == set(got)
    for k in src:
        assert float(src[k]["peak_ratio"]) == pytest.approx(float(got[k]["peak_ratio"]))

    # The published bundle really does span two commits, split cleanly by arm:
    # real3d/random was re-run at cd568622 after the budget fix.
    per = man["per_arm"]
    assert per["real3d / random"]["commits"] == ["cd568622"]
    assert per["real3d / gp_ts"]["commits"] == ["d304c411"]
