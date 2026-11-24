"""
benchmarks/methods/_lines.py
============================
LineBO-style line sampling for the baselines (``sampling="line"``), with the same
geometry ZoMBI-Hop's ``src.core.linebo.LineBO`` uses on ``BoxDomain()``:

* **A chord** is a uniform anchor in a box plus an isotropic random direction,
  extended both ways to the box faces (``random_chords_through_simplex`` with
  ``BoxDomain``: ``BoxDomain.sample`` anchors, ``BoxDomain.directions``
  directions, ``batch_line_bounds_segments`` clipping).
* **One optimiser step** draws ``num_lines`` chords (zombi_hop: 10), scores each on
  ``points_per_line`` evenly spaced points (100) with the method's own acquisition,
  and measures the best chord at ``problem.batch_size`` evenly spaced points (24).
  How a chord's 100 values reduce to one score is per method (gp_bo: mean EI;
  TuRBO: mean of one Thompson sample path; HEBO: Pareto front of the mean MACE
  objectives) — see ``benchmarks/sweeps/LINEWISE.md``.
* **The initial design** is ``n_init_lines`` random chords through the unit box,
  drawn exactly as ``zombi_hop`` line mode draws them (:func:`init_chords`). Every
  method on a cell has the same seed, so every method measures the SAME init lines.

Re-stated in numpy rather than imported: importing ``src.core`` imports
``src.core.zombihop``, which switches torch's global default device and dtype, and
the baselines must not inherit that. It also draws from the method's own seeded
``rng``, so a cell is reproducible.
"""

from __future__ import annotations

import numpy as np

#: zombi_hop line-mode defaults (``linebo_num_lines``, ``linebo_points_per_line``,
#: ``n_init_lines``), shared by every method in line mode.
DEFAULT_NUM_LINES = 10
DEFAULT_POINTS_PER_LINE = 100
DEFAULT_N_INIT_LINES = 2

#: Config keys every baseline gains for line mode.
LINE_DEFAULTS = {
    "sampling": "point",
    "n_init_lines": DEFAULT_N_INIT_LINES,
    "linebo_num_lines": DEFAULT_NUM_LINES,
    "linebo_points_per_line": DEFAULT_POINTS_PER_LINE,
}


def check_sampling(name: str, config: dict, batch_size: int) -> bool:
    """True for line mode; validates the mode and that a line has >= 2 points."""
    mode = config["sampling"]
    if mode not in ("point", "line"):
        raise ValueError(f"{name}: sampling must be 'point' or 'line', got {mode!r}")
    if mode == "line" and int(batch_size) < 2:
        raise ValueError(f"{name}: a line needs batch_size >= 2 (got {batch_size}); "
                         "plan with --batch-size 24 for line mode")
    return mode == "line"


def box_chord(x0: np.ndarray, direction: np.ndarray,
              lo: np.ndarray | None = None, hi: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """Endpoints of the line ``x0 + t*direction`` clipped to ``[lo, hi]`` (unit box
    by default)."""
    d = len(x0)
    lo = np.zeros(d) if lo is None else np.asarray(lo, dtype=float)
    hi = np.ones(d) if hi is None else np.asarray(hi, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (lo - x0) / direction
        t1 = (hi - x0) / direction
    t_lo = np.where(direction != 0, np.minimum(t0, t1), -np.inf)
    t_hi = np.where(direction != 0, np.maximum(t0, t1), np.inf)
    t_min, t_max = float(t_lo.max()), float(t_hi.min())
    return (np.clip(x0 + t_min * direction, lo, hi),
            np.clip(x0 + t_max * direction, lo, hi))


def init_chords(rng: np.random.Generator, n: int, d: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """``n`` random chords through the unit box: zombi_hop line mode's initial design.

    The draw order (anchor, then direction, per chord) is the one zombi_hop has
    always used, so its line-mode runs are unchanged and every baseline seeded the
    same way measures the identical lines.
    """
    out = []
    for _ in range(int(n)):
        x0 = rng.random(d)
        v = rng.normal(size=d)
        out.append(box_chord(x0, v / np.linalg.norm(v)))
    return out


def random_chords(rng: np.random.Generator, k: int, lo: np.ndarray, hi: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """``k`` chords of the box ``[lo, hi]``: uniform anchor, isotropic direction.

    Returns ``(left, right)``, each ``(k, d)``. A degenerate chord (zero length, from
    a zero-width box) is kept; scoring and measuring it is still well defined.
    """
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    d = len(lo)
    left, right = np.empty((k, d)), np.empty((k, d))
    for i in range(int(k)):
        x0 = lo + rng.random(d) * (hi - lo)
        v = rng.normal(size=d)
        while np.linalg.norm(v) < 1e-12:
            v = rng.normal(size=d)
        left[i], right[i] = box_chord(x0, v / np.linalg.norm(v), lo, hi)
    return left, right


def line_points(left: np.ndarray, right: np.ndarray, n: int) -> np.ndarray:
    """``n`` evenly spaced points from ``left`` to ``right`` (both ends included).

    ``left``/``right`` may be ``(d,)`` -> ``(n, d)`` or ``(k, d)`` -> ``(k, n, d)``.
    """
    t = np.linspace(0.0, 1.0, int(n))
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.ndim == 1:
        return left[None, :] + t[:, None] * (right - left)[None, :]
    return left[:, None, :] + t[None, :, None] * (right - left)[:, None, :]


class LineQueue:
    """Measured points owed by a line already chosen, handed out by ``ask(n)``.

    ``AskTellMethod.run`` asks for ``problem.batch_size`` points at a time, so a
    line is normally one ask. The queue keeps a line intact if a caller ever asks
    for fewer (the budget's last call), and serves the init lines first.
    """

    def __init__(self) -> None:
        self._pending: list[np.ndarray] = []

    def push_line(self, left, right, n_points: int) -> None:
        self._pending.extend(line_points(left, right, n_points))

    def __len__(self) -> int:
        return len(self._pending)

    def pop(self, n: int) -> np.ndarray:
        out, self._pending = self._pending[:n], self._pending[n:]
        return np.asarray(out, dtype=float)


def non_dominated(F: np.ndarray) -> np.ndarray:
    """Indices of the rows of ``F`` (``(k, m)``, every column MINIMISED) that no
    other row dominates."""
    F = np.asarray(F, dtype=float)
    keep = []
    for i in range(len(F)):
        dominated = np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1))
        if not dominated:
            keep.append(i)
    return np.asarray(keep, dtype=int)
