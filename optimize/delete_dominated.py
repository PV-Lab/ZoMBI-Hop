"""
delete_dominated.py
====================
Reclaim disk by deleting the Pareto-DOMINATED trial directories of a MOBO run,
keeping only the globally Pareto-optimal trials.

This is the destructive companion to ``pareto.py``: it determines Pareto
membership with *exactly the same* collection + front logic (it imports
``pareto`` and reuses ``collect_trials`` / ``pareto_mask_min`` and the same
shared-history / collaborator pooling), so the set of trials it keeps is
identical to the stars ``pareto.py`` would plot for the same path. Every trial
that ``pareto.py`` would draw as a dominated dot is a deletion candidate; every
Pareto star is kept.

    python optimize/delete_dominated.py optimize/runs/mobo_ensemble_10d_job17776002

mirrors

    python optimize/pareto.py optimize/runs/mobo_ensemble_10d_job17776002

for classification, then removes each dominated trial's ``trial_<n>/`` directory
(which for an ensemble run holds its ``run_1``..``run_N`` repeats — the bulk of
the disk).

Scope: because the front is a GLOBAL property, passing a single run dir pools its
config-matching sibling runs (and collaborators') just like ``pareto.py``. The
default therefore deletes dominated trials across *every pooled run*, keeping only
the global Pareto set. Use ``--no-shared-history`` / ``--no-collab`` to narrow the
pool exactly as in ``pareto.py`` (with ``--no-shared-history`` a single run dir is
pruned against its own trials only).

Safety
------
* Dry-run by default: nothing is deleted until you pass ``--delete``.
* Only ever removes a directory that (a) exists, (b) is named ``trial_<int>``,
  (c) is owned by the current user, and (d) resolves under an allowed working
  root (this checkout + the configured extra working dirs). Anything failing a
  check (e.g. a collaborator-owned trial) is reported and SKIPPED, never removed.
* Trials that ``pareto.py`` never collected (failed/incomplete trials with no
  valid metrics) are NOT classified as dominated and are left untouched.
* Pareto-optimal trials, and any run-level files outside ``trial_<n>/``
  (mobo_progress.json, results, configs, plots), are never touched.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys

import numpy as np

# Reuse pareto.py's collection + front logic verbatim so "dominated" here always
# means exactly what pareto.py's plot would show as a dominated dot.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import pareto  # noqa: E402  (local module: optimize/pareto.py)

_TRIAL_RE = re.compile(r"^trial_\d+$")


def _allowed_roots() -> list[str]:
    """Real paths under which a trial dir may be deleted.

    The checkout this script lives in, plus any extra working roots passed via
    the ``ZOMBI_EXTRA_WORK_DIRS`` env var (colon-separated). A trial dir whose
    real path is not under one of these is refused — a guard against ever
    removing files reached through a collaborator symlink.
    """
    roots = [os.path.realpath(os.path.dirname(_HERE))]  # the repo root (parent of optimize/)
    extra = os.environ.get("ZOMBI_EXTRA_WORK_DIRS", "")
    roots += [os.path.realpath(p) for p in extra.split(":") if p.strip()]
    return roots


def _under_any(path: str, roots: list[str]) -> bool:
    rp = os.path.realpath(path)
    return any(rp == r or rp.startswith(r + os.sep) for r in roots)


def _collect_records(args) -> tuple[list[dict], np.ndarray]:
    """Reproduce pareto.main()'s collection for *args.runs_dir*.

    Returns (records, pareto_mask) using the identical shared-history /
    collaborator pooling, signature filtering, and 3-objective front.
    """
    runs_dir = os.path.abspath(args.runs_dir)

    single_run = (
        not __import__("glob").glob(os.path.join(runs_dir, "mobo_*", "mobo_progress.json"))
        and os.path.isfile(os.path.join(runs_dir, "mobo_progress.json"))
    )

    only_signature = None
    if single_run and not args.no_shared_history:
        only_signature = pareto._load_run_signature(runs_dir)
        if only_signature is None:
            print(f"  [shared-history] {os.path.basename(runs_dir)} has no run_config.json; "
                  "using this run's trials only.")
        else:
            print(f"  [shared-history] pooling sibling runs matching "
                  f"{os.path.basename(runs_dir)}'s config: {only_signature}")
            runs_dir = os.path.dirname(runs_dir)  # crawl the parent, signature-filtered

    crawl_dirs = [runs_dir]
    if only_signature is not None and not args.no_collab:
        crawl_dirs = pareto._dedup_realpath([runs_dir] + pareto._collaborator_runs_dirs())
        if len(crawl_dirs) > 1:
            print(f"  [collab] pooling {len(crawl_dirs)} runs dirs: " + ", ".join(crawl_dirs))

    records: list[dict] = []
    for d in crawl_dirs:
        records += pareto.collect_trials(
            d, exclude_old=not args.with_old, only_signature=only_signature)
    if not records:
        sys.exit(f"No usable trials found for {args.runs_dir}.")

    # Same 3-objective front as pareto.main() (n_points_penalty stays inert).
    M = np.array([[r["metrics"][pareto.DIST_KEY],
                   r["metrics"][pareto.DUP_KEY],
                   r["time_value"]] for r in records], dtype=float)
    mask = pareto.pareto_mask_min(M)
    return records, mask


def _trial_dir(rec: dict) -> str:
    """Absolute path of a record's trial directory: <source_dir>/<run>/trial_<n>."""
    return os.path.join(rec["source_dir"], rec["source_run"], f"trial_{rec['trial']}")


def _dir_size(path: str) -> int:
    """Total bytes under *path* (follows no symlinks; scoped to this one tree)."""
    total = 0
    for root, _dirs, files in os.walk(path, followlinks=False):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.lstat(fp).st_size
            except OSError:
                pass
    return total


def _human(nbytes: int) -> str:
    x = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.1f} {unit}"
        x /= 1024
    return f"{x:.1f} TB"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Delete the Pareto-DOMINATED trial directories of a MOBO run "
                    "(keeps only the globally Pareto-optimal trials). Dry-run "
                    "unless --delete is given.")
    ap.add_argument("runs_dir",
                    help="A run directory (e.g. optimize/runs/mobo_ensemble_10d_job17776002) "
                         "or a runs parent dir. Pooled exactly as pareto.py pools it.")
    ap.add_argument("--delete", action="store_true",
                    help="Actually remove the dominated trial dirs. Without this, "
                         "only prints what WOULD be deleted (dry run).")
    ap.add_argument("--no-shared-history", action="store_true",
                    help="Do NOT pool config-matching sibling runs; prune a single "
                         "run dir against its own trials only (mirrors pareto.py).")
    ap.add_argument("--no-collab", action="store_true",
                    help="Do NOT pool collaborators' runs directories (mirrors pareto.py).")
    ap.add_argument("--with-old", action="store_true",
                    help="Include mobo_old_jackson trials in the pool (mirrors pareto.py).")
    args = ap.parse_args()

    print("=" * 70)
    print(f"Pareto-dominated trial cleanup  |  {args.runs_dir}")
    print("=" * 70)

    records, mask = _collect_records(args)
    n_total, n_pareto = len(records), int(mask.sum())
    dominated = [records[i] for i in range(len(records)) if not mask[i]]
    print(f"\n  {n_total} trial(s) pooled -> {n_pareto} Pareto-optimal, "
          f"{len(dominated)} dominated (deletion candidates).")

    roots = _allowed_roots()
    me = os.getuid()

    to_delete: list[tuple[dict, str]] = []
    skipped: list[tuple[dict, str, str]] = []   # (rec, path, reason)
    for rec in dominated:
        path = _trial_dir(rec)
        name = os.path.basename(path)
        if not _TRIAL_RE.match(name):
            skipped.append((rec, path, "name is not trial_<int>"))
            continue
        if not os.path.isdir(path):
            skipped.append((rec, path, "directory missing"))
            continue
        try:
            if os.stat(path).st_uid != me:
                skipped.append((rec, path, "owned by another user"))
                continue
        except OSError as exc:
            skipped.append((rec, path, f"stat failed ({exc})"))
            continue
        if not _under_any(path, roots):
            skipped.append((rec, path, "outside allowed working roots"))
            continue
        to_delete.append((rec, path))

    if skipped:
        print(f"\n  Skipping {len(skipped)} candidate(s) (kept, not deleted):")
        for rec, path, reason in skipped:
            print(f"    - {rec['source_run']}/trial_{rec['trial']}: {reason}")

    if not to_delete:
        print("\n  Nothing to delete.")
        return

    # Size is measured on the exact trees we will remove (targeted, not a scan).
    total_bytes = 0
    print(f"\n  {len(to_delete)} dominated trial dir(s) to remove:")
    for rec, path in to_delete:
        sz = _dir_size(path)
        total_bytes += sz
        rel = os.path.relpath(path, os.getcwd())
        print(f"    {rel}   ({_human(sz)})")
    print(f"\n  Total to reclaim: {_human(total_bytes)}")

    if not args.delete:
        print("\n  DRY RUN — nothing deleted. Re-run with --delete to remove the above.")
        return

    print("\n  Deleting ...")
    removed, freed = 0, 0
    for rec, path in to_delete:
        sz = _dir_size(path)
        try:
            shutil.rmtree(path)
            removed += 1
            freed += sz
        except OSError as exc:
            print(f"    ! failed to remove {path}: {exc}")
    print(f"\n  Removed {removed}/{len(to_delete)} trial dir(s); reclaimed ~{_human(freed)}.")


if __name__ == "__main__":
    main()
