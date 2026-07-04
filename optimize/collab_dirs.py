"""Single source of truth for collaborator runs directories (cross-user shared history).

Several ZoMBI-Hop users run the same MOBO configs and pool each other's run
history in three places, which must all agree on WHERE each user's runs live:

  * run_mobo.py   — pools matching trials into the GP live during a run (--share-history)
  * pareto.py     — pools matching trials when collecting the global Pareto front
  * sync_runs.py  — pulls both users' run dirs down from ORCD over ssh

Defining the paths here once stops those three from drifting apart (they had, which
silently broke pooling: sync_runs still pointed at the pre-scratch-move /home paths).

The runs live on scratch (moved off /home for space), reached via each user's
``~/orcd/scratch`` symlink. Cross-user reads work via the shared ``sched_mit_hill``
group grant. This module is intentionally dependency-free (no torch/matplotlib) so
sync_runs.py can import it on machines that lack the heavy MOBO stack.
"""

from __future__ import annotations

# Whether to include collaborators' dirs at all (vs. only the current user's own).
SHARE_COLLABORATOR_HISTORY = False

# One runs dir per collaborating user, in preference order (this account's own dir
# first, since run_mobo/sync operate as that user). The paths are login-node /home
# paths that resolve through ~/orcd/scratch into /orcd/scratch/orcd/<NNN>/<user>.
COLLABORATOR_RUNS_DIRS = [
    "/home/adewinmb/orcd/scratch/ZoMBI-Hop/optimize/runs",
    "/home/eve_lal/orcd/scratch/ZoMBI-Hop/optimize/runs",
]
