"""
optimize/clamp_hparams.py
=========================
Project stored trial hyperparameters into the CURRENT ``run_mobo.HPARAM_SPACE``.

Why this exists
---------------
When a range in ``HPARAM_SPACE`` is tightened, every previously recorded trial
that sat outside the new range becomes unreproducible: ``hparams_to_norm``
clamps the normalised coordinate into [0,1], so ``--start-from-best`` on such a
trial silently re-evaluates a DIFFERENT configuration than the file describes,
and a showdown fed the raw file measures the old out-of-range configuration.
Both are confusing in opposite directions.

Writing the clamped configuration to disk makes the projection explicit and
gives one artifact that the showdown and the ``--start-from-best`` seeding both
consume, so the configuration that gets scored is exactly the one that gets
seeded.

The output is a trial.json-shaped blob (``hparams`` + provenance), which is what
both ``evaluate.py --hparams-json`` and ``run_mobo.py --start-from-best``
accept, and it records what moved under ``clamped_from``.

Usage
-----
  python optimize/clamp_hparams.py --out-dir optimize/hparams/clamped_6d \\
      optimize/runs/mobo_ensemble_6d_job19202380/trial_23 [trial_dir ...]

  # Name the outputs explicitly (order matches the trial dirs):
  python optimize/clamp_hparams.py --out-dir DIR --names dist1,dist2 A B
"""

from __future__ import annotations

import os
import sys
import json
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from run_mobo import HPARAM_SPACE, HPARAM_NAMES  # noqa: E402


def clamp_hparams(hp: dict) -> tuple[dict, dict]:
    """Return (clamped hparams, {name: [old, new]} for whatever moved).

    Integer axes are rounded after clamping so the result is a value the space
    can actually produce; a float axis is clamped as-is. Names absent from
    ``HPARAM_SPACE`` are dropped — they are stale axes that the current run
    would ignore anyway, and keeping them invites the illusion they still apply.
    """
    out: dict = {}
    moved: dict = {}
    for name in HPARAM_NAMES:
        if name not in hp:
            continue
        lo, hi, tfm = HPARAM_SPACE[name]
        old = float(hp[name])
        new = min(max(old, float(lo)), float(hi))
        new = int(round(new)) if tfm == "int" else new
        out[name] = new
        if abs(new - old) > 1e-12:
            moved[name] = [old, new]
    return out, moved


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trial_dirs", nargs="+",
                    help="trial_* directories (or trial.json files) to project")
    ap.add_argument("--out-dir", required=True, help="where to write the clamped JSONs")
    ap.add_argument("--names", default=None,
                    help="comma-separated output stems, one per trial dir "
                         "(default: <run>_trial<N>)")
    a = ap.parse_args()

    names = a.names.split(",") if a.names else [None] * len(a.trial_dirs)
    if len(names) != len(a.trial_dirs):
        ap.error(f"--names has {len(names)} entries for {len(a.trial_dirs)} trial dirs")
    os.makedirs(a.out_dir, exist_ok=True)

    for path, name in zip(a.trial_dirs, names):
        jp = path if path.lower().endswith(".json") else os.path.join(path, "trial.json")
        with open(jp) as f:
            blob = json.load(f)
        hp = blob.get("hparams", blob)
        clamped, moved = clamp_hparams(hp)

        run = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(jp))))
        name = name or f"{run}_trial{blob.get('trial', '?')}"
        out = {
            "config_name": name,
            "hparams": clamped,
            # Provenance: the showdown writes this straight into its manifest, and
            # keeping the ORIGINAL metrics under a distinct key stops them being
            # mistaken for measurements OF the clamped configuration — they are
            # the unclamped configuration's scores, on its own landscapes.
            "source_run": run,
            "trial": blob.get("trial"),
            "selected_for": blob.get("selected_for", "clamped"),
            "unclamped_metrics": blob.get("metrics", {}),
            "clamped_from": moved,
            "hparam_space": {k: list(v) for k, v in HPARAM_SPACE.items()},
        }
        dst = os.path.join(a.out_dir, f"{name}.json")
        with open(dst, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  {name}: {len(moved)} hparam(s) clamped -> {dst}")
        for k, (o, n) in moved.items():
            print(f"      {k:<30} {o:>12.5f}  ->  {n:>12.5f}")


if __name__ == "__main__":
    main()
