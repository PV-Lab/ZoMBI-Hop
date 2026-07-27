"""
warm_start/best_hparams_seed.py
===============================
Turn ``warm_start/BEST_HPARAMS.md`` into a ``--start-from-best`` seed directory.

``optimize/run_mobo.py --start-from-best DIR`` re-evaluates the hyperparameters in
``DIR/trial.json`` as an initial trial, so every warm-start tuning run begins by
scoring the *current* best hyperparameters against its own objective. This script
writes exactly that ``trial.json`` from the ``d``-matching JSON block of
BEST_HPARAMS.md, so the sbatch jobs can seed the search without hand-maintaining a
separate seed file.

BEST_HPARAMS.md was produced by earlier tuning runs whose search space did not
include every hyperparameter the current ``HPARAM_SPACE`` tunes (it predates the
penalisation / paring parameters). Any hyperparameter the block omits is filled
with the ZoMBI-Hop constructor default (``src/core/zombihop.py``) — i.e. the value
those best runs actually ran at, since they never tuned it — so the emitted seed
reproduces the real configuration rather than an arbitrary midpoint. A seed whose
keys don't cover the current space is simply skipped by ``load_seed_hparams`` (the
Sobol init still runs), so a future space change fails safe rather than crashing.

Usage
-----
    python warm_start/best_hparams_seed.py --dim 3 --out <run_dir>/best_seed
    # prints the seed dir on stdout; pass it to --start-from-best.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_BEST_HPARAMS_MD = _HERE / "BEST_HPARAMS.md"

# The hyperparameters run_mobo tunes, mirroring optimize.run_mobo.HPARAM_SPACE.
# Kept as a plain list (not imported) so this pre-job helper stays light and does
# not pull in torch/botorch just to write a JSON file.
HPARAM_NAMES = [
    "nat_grad_step", "nat_grad_max_steps", "n_restarts", "raw", "ucb_beta",
    "max_zooms", "max_iterations", "top_m_points", "n_consecutive_converged",
    "input_noise_threshold_mult", "output_noise_threshold_mult",
    "max_penalty_radius", "needle_shrink_factor", "needle_stop_noise_multiplier",
    "paring_spatial_halfnoise", "paring_y_noise_multiplier",
]

# ZoMBI-Hop constructor defaults for the hyperparameters BEST_HPARAMS.md predates
# (src/core/zombihop.py). Used only when a block omits the key.
ZOMBI_DEFAULTS = {
    "max_penalty_radius": 1.0,
    "needle_shrink_factor": 0.85,
    "needle_stop_noise_multiplier": 3.0,
    "paring_spatial_halfnoise": 0.5,
    "paring_y_noise_multiplier": 1.0,
}


def _load_block(dim: int, md_path: Path = _BEST_HPARAMS_MD) -> dict:
    """Return the JSON hyperparameter block whose ``d`` matches ``dim``."""
    if not md_path.exists():
        raise FileNotFoundError(f"BEST_HPARAMS not found: {md_path}")
    text = md_path.read_text()
    for m in re.finditer(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL):
        block = json.loads(m.group(1))
        if int(block.get("d", -1)) == int(dim):
            return block
    raise ValueError(f"No JSON block with d={dim} in {md_path}")


def best_hparams(dim: int, md_path: Path = _BEST_HPARAMS_MD) -> dict:
    """The current best hyperparameters for ``dim`` as a full HPARAM_NAMES dict.

    Values present in the BEST_HPARAMS.md block are used verbatim; every other
    tuned hyperparameter falls back to its ZoMBI-Hop default.
    """
    block = _load_block(dim, md_path)
    hp: dict = {}
    missing_default: list[str] = []
    for name in HPARAM_NAMES:
        if name in block:
            hp[name] = block[name]
        elif name in ZOMBI_DEFAULTS:
            hp[name] = ZOMBI_DEFAULTS[name]
            missing_default.append(name)
        else:  # a tuned key with neither a stored value nor a known default
            raise KeyError(
                f"BEST_HPARAMS d={dim} omits '{name}' and no ZoMBI default is "
                f"registered for it; add one to ZOMBI_DEFAULTS.")
    if missing_default:
        print(f"  [seed] d={dim}: filled {len(missing_default)} hyperparameter(s) "
              f"from ZoMBI defaults: {missing_default}")
    return hp


def write_seed_dir(dim: int, out_dir: Path,
                   md_path: Path = _BEST_HPARAMS_MD) -> Path:
    """Write ``out_dir/trial.json`` seeding the best ``dim`` hyperparameters."""
    hp = best_hparams(dim, md_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    block = _load_block(dim, md_path)
    trial = {
        "trial": f"best_{dim}d",
        "source": str(md_path),
        "run_uuid": block.get("run_uuid"),
        "hparams": hp,
    }
    (out_dir / "trial.json").write_text(json.dumps(trial, indent=2))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Emit a --start-from-best seed dir from BEST_HPARAMS.md.")
    parser.add_argument("--dim", type=int, required=True, help="3 or 4.")
    parser.add_argument("--out", required=True,
                        help="Seed directory to create (holds trial.json).")
    parser.add_argument("--best-hparams", default=str(_BEST_HPARAMS_MD),
                        help=f"Path to BEST_HPARAMS.md (default: {_BEST_HPARAMS_MD}).")
    args = parser.parse_args()
    out = write_seed_dir(args.dim, Path(args.out), Path(args.best_hparams))
    # The sbatch captures stdout to pass to --start-from-best, so print ONLY the
    # path on the last line.
    print(str(out.resolve()))


if __name__ == "__main__":
    main()
