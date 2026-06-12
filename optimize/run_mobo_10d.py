"""
Deprecated shim — use optimize/run_mobo.py instead.

10D Multi-Ackley MOBO now runs through the unified MOBO driver with the same
run layout, logging, and batch infrastructure as campaign RF MOBO::

  python optimize/run_mobo.py --landscape ackley
  python optimize/run_mobo.py --batch --config optimize/mobo_batch_configs/ackley_10d_layout1.json
  sbatch slurm/mobo_10d.sbatch
"""

from __future__ import annotations

import sys
import warnings


def _map_argv(argv: list[str]) -> list[str]:
    """Translate legacy run_mobo_10d.py flags to run_mobo.py equivalents."""
    out = ["run_mobo.py", "--landscape", "ackley"]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--batch":
            out.append("--batch")
            out.extend(["--config", "optimize/mobo_batch_configs/ackley_10d_layout1.json"])
        elif arg == "--save-dir":
            out.extend(["--run-dir", argv[i + 1]])
            i += 1
        elif arg == "--n-mobo-trials":
            n_mobo = int(argv[i + 1])
            n_init = 8
            j = 1
            while j < len(argv):
                if argv[j] == "--n-init-trials":
                    n_init = int(argv[j + 1])
                    break
                j += 1
            out.extend(["--max-trials", str(n_init + n_mobo)])
            i += 1
        elif arg in ("--dim", "--layout", "--ackley-b", "--device", "--no-show",
                     "--n-init-trials", "--max-trials", "--run-dir", "--runs-dir"):
            out.extend([arg, argv[i + 1]])
            i += 1
        elif arg.startswith("--"):
            out.append(arg)
        i += 1
    if "--batch" not in out and "--max-trials" not in out:
        out.extend(["--max-trials", "28"])
    if "--no-show" in argv and "--no-show" not in out:
        out.append("--no-show")
    return out


if __name__ == "__main__":
    warnings.warn(
        "optimize/run_mobo_10d.py is deprecated; use optimize/run_mobo.py "
        "(see module docstring).",
        DeprecationWarning,
        stacklevel=1,
    )
    sys.argv = _map_argv(sys.argv)
    from optimize.run_mobo import main
    main()
