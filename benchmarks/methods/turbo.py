"""
benchmarks/methods/turbo.py
===========================
TuRBO — trust-region Bayesian optimisation (Eriksson et al., NeurIPS 2019,
arXiv:1910.01739).

TuRBO-m keeps ``m`` hyper-rectangular trust regions, each with its own local GP
fitted only to the data that region collected. A batch is filled by Thompson
sampling: every region draws ``q`` joint posterior samples over its own candidate
set, and slot ``j`` of the batch goes to whichever region's candidate maximises
sample ``j`` — so regions compete for the batch on their merits. After each batch a
region that improved on its incumbent counts a success, otherwise a failure;
``tau_succ`` consecutive successes double its side length (up to ``length_max``),
``tau_fail`` consecutive failures halve it, and a region shrunk below ``length_min``
is restarted from a fresh initial design, its data discarded. ``m = 1`` (default) is
TuRBO-1.

Faithful to the paper (and to BoTorch's reference tutorial) in every constant:
``length_init = 0.8``, ``length_min = 0.5**7``, ``length_max = 1.6``,
``tau_succ = 3``, ``tau_fail = ceil(max(4/q, d/q))``; the region is centred on its
best observed point and shaped by the GP's ARD lengthscales (normalised to unit
geometric mean); candidates are Sobol' points in the region with each coordinate
perturbed from the centre with probability ``min(1, 20/d)``; ``n_candidates =
min(5000, max(2000, 200 d))``; Matern-5/2 ARD kernel with lengthscales in
``[0.005, 4]``.

One deliberate departure: the reference constrains the GP noise to ``[1e-8, 1e-3]``
because the paper's test functions are noiseless. These landscapes carry 4.5%
multiplicative noise, and pinning the noise near zero would make the local GP
interpolate it. The noise is learned (lower bound ``1e-6``) instead.

TuRBO does not declare a set of optima, so it is scored through the extractor like
the other baselines. Each region's incumbent at restart is still written to
``turbo_trust_regions.csv`` — those are TuRBO's own converged local optima, useful
for seeing how many distinct basins its restarts visited.

Line mode (``sampling="line"``; ``benchmarks/sweeps --sampling line``)
--------------------------------------------------------------------
TuRBO measures LINES, using zombi_hop's LineBO geometry (see
``benchmarks/sweeps/LINEWISE.md``):

* A region's initial design is ``n_init_lines`` random chords through the whole box
  (the first region's are the same chords every line-mode method starts from).
* Each step fits the region's GP as usual, draws ``linebo_num_lines`` random chords
  through the region's box (the same ARD-shaped box ``_candidates`` samples from),
  draws ONE joint Thompson sample over all ``linebo_num_lines x
  linebo_points_per_line`` chord points, and measures the chord with the highest
  mean sample value at ``batch_size`` points. Those are model queries, not
  measurements: only the chosen chord counts against the budget, exactly as
  point-mode TuRBO samples thousands of candidates and measures one. The
  coordinate-perturbation mask of ``_candidates`` has no chord counterpart and is
  not used.
* A line is ONE step for the success/failure counters, so ``tau_fail`` defaults to
  the sequential ``max(4, d)`` (not the q-batch ``ceil(max(4/q, d/q))``, which at
  q = 24 is 1 and would halve the region after every unimproved line).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import _lines
from ._gp import fit_matern_gp
from .base import AskTellMethod, Problem, sobol_design
from .registry import register


@dataclass
class _TrustRegion:
    tr_id: int
    restart: int
    length: float
    idx: list[int] = field(default_factory=list)      # rows of the global arrays
    best_value: float = -math.inf
    best_x: np.ndarray | None = None
    success: int = 0
    failure: int = 0
    pending_init: int = 0      # init points queued/in flight for this region
    n_start: int = 0           # global sample count when this restart began
    lengthscales: np.ndarray | None = None

    @property
    def active(self) -> bool:
        return self.pending_init == 0 and len(self.idx) > 0


@register
class TuRBO(AskTellMethod):
    name = "turbo"
    description = "TuRBO-m trust-region BO (Eriksson et al. 2019); m=1 by default"
    defaults = {
        "n_trust_regions": 1,
        "n_init": 48,
        "length_init": 0.8,
        "length_min": 0.5 ** 7,
        "length_max": 1.6,
        "success_tolerance": 3,
        "failure_tolerance": None,     # None -> ceil(max(4/q, d/q))
        "n_candidates": None,          # None -> min(5000, max(2000, 200 d))
        "max_train_points": None,      # per region; None = all of its data
        "improvement_rtol": 1e-3,
        **_lines.LINE_DEFAULTS,
    }

    # ── setup ──

    def setup(self, problem: Problem) -> None:
        self.dim = problem.dim
        self.q = problem.batch_size
        cfg = self.config
        self.line_mode = _lines.check_sampling(self.name, cfg, problem.batch_size)
        q_step = 1 if self.line_mode else self.q    # a line is one sequential step
        self.tau_fail = int(cfg["failure_tolerance"] or
                            math.ceil(max(4.0 / q_step, self.dim / q_step)))
        self.tau_succ = int(cfg["success_tolerance"])
        self.n_cand = int(cfg["n_candidates"] or min(5000, max(2000, 200 * self.dim)))
        self.X = np.empty((0, self.dim))
        self.Y = np.empty(0)
        self._init_queue: list[tuple[int, np.ndarray]] = []   # (tr_id, point)
        self._line_queue: list[tuple[int, np.ndarray]] = []   # (tr_id, point), line mode
        self._assign: list[tuple[int, bool]] = []             # per asked point
        self._n_sobol_draws = 0
        self.history: list[dict] = []
        self.trs = [self._new_tr(i, 0) for i in range(int(cfg["n_trust_regions"]))]

    def _new_tr(self, tr_id: int, restart: int) -> _TrustRegion:
        tr = _TrustRegion(tr_id=tr_id, restart=restart,
                          length=float(self.config["length_init"]),
                          n_start=len(self.Y))
        if self.line_mode:
            chords = _lines.init_chords(self.rng, int(self.config["n_init_lines"]), self.dim)
            pts = [p for left, right in chords
                   for p in _lines.line_points(left, right, self.q)]
            tr.pending_init = len(pts)
            self._init_queue.extend((tr_id, p) for p in pts)
            return tr
        pts = sobol_design(int(self.config["n_init"]), self.dim,
                           self.seed * 7919 + 104_729 * self._n_sobol_draws + tr_id)
        self._n_sobol_draws += 1
        tr.pending_init = len(pts)
        self._init_queue.extend((tr_id, p) for p in pts)
        return tr

    # ── ask ──

    def ask(self, n: int) -> np.ndarray:
        if self.line_mode:
            return self._ask_line(n)
        pts: list[np.ndarray] = []
        self._assign = []
        while self._init_queue and len(pts) < n:
            tr_id, p = self._init_queue.pop(0)
            pts.append(p)
            self._assign.append((tr_id, True))
        n_ts = n - len(pts)
        if n_ts > 0:
            active = [tr for tr in self.trs if tr.active]
            if active:
                X_ts, owners = self._thompson_batch(active, n_ts)
            else:   # every region is still waiting on its init points
                X_ts, owners = self.rng.random((n_ts, self.dim)), [-1] * n_ts
            pts.extend(X_ts)
            self._assign.extend((o, False) for o in owners)
        return np.asarray(pts, dtype=float).reshape(n, self.dim)

    def _ask_line(self, n: int) -> np.ndarray:
        """Init lines first, then one Thompson-sampled chord per step."""
        if not self._init_queue and not self._line_queue:
            active = [tr for tr in self.trs if tr.active]
            if active:
                left, right, owner = self._thompson_line(active)
            else:   # unreachable with whole init lines; kept for safety
                lo, hi = np.zeros(self.dim), np.ones(self.dim)
                (left,), (right,) = _lines.random_chords(self.rng, 1, lo, hi)
                owner = -1
            self._line_queue.extend((owner, p)
                                    for p in _lines.line_points(left, right, self.q))
        pts: list[np.ndarray] = []
        self._assign = []
        for queue, is_init in ((self._init_queue, True), (self._line_queue, False)):
            while queue and len(pts) < n:
                tr_id, p = queue.pop(0)
                pts.append(p)
                self._assign.append((tr_id, is_init))
        return np.asarray(pts, dtype=float).reshape(len(pts), self.dim)

    def _region_box(self, tr: _TrustRegion) -> tuple[np.ndarray, np.ndarray]:
        """The region's ARD-shaped box, as ``_candidates`` builds it."""
        w = tr.lengthscales / tr.lengthscales.mean()
        w = w / np.prod(w ** (1.0 / self.dim))
        return (np.clip(tr.best_x - w * tr.length / 2.0, 0.0, 1.0),
                np.clip(tr.best_x + w * tr.length / 2.0, 0.0, 1.0))

    def _thompson_line(self, active: list[_TrustRegion]):
        """The chord with the highest mean of one joint posterior sample, over
        ``linebo_num_lines`` chords per region (model queries only)."""
        import torch

        from ._gp import tensor

        k = int(self.config["linebo_num_lines"])
        P = int(self.config["linebo_points_per_line"])
        best = (-np.inf, None, None, -1)
        for tr in active:
            idx = self._train_idx(tr)
            model, _, _ = fit_matern_gp(self.X[idx], self.Y[idx], device=self.device)
            tr.lengthscales = (model.covar_module.base_kernel.lengthscale
                               .detach().cpu().numpy().ravel())
            lo, hi = self._region_box(tr)
            left, right = _lines.random_chords(self.rng, k, lo, hi)
            pts = _lines.line_points(left, right, P).reshape(-1, self.dim)
            with torch.no_grad():
                post = model.posterior(tensor(pts, self.device))
                S = post.rsample(sample_shape=torch.Size([1])).reshape(k, P)
            score = S.mean(dim=1).cpu().numpy()
            j = int(np.argmax(score))
            if score[j] > best[0]:
                best = (float(score[j]), left[j], right[j], tr.tr_id)
            del model, post
        _, left, right, owner = best
        return left, right, owner

    def _train_idx(self, tr: _TrustRegion) -> np.ndarray:
        idx = np.asarray(tr.idx, dtype=int)
        cap = self.config["max_train_points"]
        if cap is None or len(idx) <= int(cap):
            return idx
        # Keep the points nearest the centre: the local model is what matters.
        d = np.linalg.norm(self.X[idx] - tr.best_x, axis=1)
        return idx[np.argsort(d)[: int(cap)]]

    def _candidates(self, tr: _TrustRegion) -> np.ndarray:
        from scipy.stats import qmc

        w = tr.lengthscales / tr.lengthscales.mean()
        w = w / np.prod(w ** (1.0 / self.dim))
        lb = np.clip(tr.best_x - w * tr.length / 2.0, 0.0, 1.0)
        ub = np.clip(tr.best_x + w * tr.length / 2.0, 0.0, 1.0)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sob = qmc.Sobol(d=self.dim, scramble=True,
                            seed=int(self.rng.integers(2 ** 31))).random(self.n_cand)
        pert = lb + (ub - lb) * sob
        prob = min(20.0 / self.dim, 1.0)
        mask = self.rng.random((self.n_cand, self.dim)) <= prob
        empty = ~mask.any(axis=1)
        mask[empty, self.rng.integers(0, self.dim, size=int(empty.sum()))] = True
        cand = np.tile(tr.best_x, (self.n_cand, 1))
        cand[mask] = pert[mask]
        return cand

    def _thompson_batch(self, active: list[_TrustRegion], q: int):
        import torch

        from ._gp import tensor

        samples, cands = [], []
        for tr in active:
            idx = self._train_idx(tr)
            model, _, _ = fit_matern_gp(self.X[idx], self.Y[idx], device=self.device)
            ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
            tr.lengthscales = ls
            C = self._candidates(tr)
            with torch.no_grad():
                post = model.posterior(tensor(C, self.device))
                S = post.rsample(sample_shape=torch.Size([q])).squeeze(-1)  # (q, n_cand)
            samples.append(S.cpu().numpy())
            cands.append(C)
            del model, post
        # Slot j goes to the best candidate under sample j across every region;
        # a chosen candidate is masked so the batch never repeats a point.
        chosen, owners = [], []
        for j in range(q):
            best = (-np.inf, -1, -1)
            for r, S in enumerate(samples):
                k = int(np.argmax(S[j]))
                if S[j, k] > best[0]:
                    best = (float(S[j, k]), r, k)
            _, r, k = best
            chosen.append(cands[r][k])
            owners.append(active[r].tr_id)
            samples[r][:, k] = -np.inf
        return np.asarray(chosen), owners

    # ── tell ──

    def tell(self, X_requested, X_actual, Y) -> None:
        start = len(self.Y)
        self.X = np.vstack([self.X, X_actual])
        self.Y = np.concatenate([self.Y, Y])
        by_tr: dict[int, list[int]] = {}
        for i, (tr_id, is_init) in enumerate(self._assign[: len(Y)]):
            if tr_id < 0:
                continue
            tr = self.trs[tr_id]
            tr.idx.append(start + i)
            if is_init:
                tr.pending_init -= 1
                self._update_best(tr, start + i)
            else:
                by_tr.setdefault(tr_id, []).append(start + i)

        rtol = float(self.config["improvement_rtol"])
        for tr_id, rows in by_tr.items():
            tr = self.trs[tr_id]
            y_new = float(self.Y[rows].max())
            if y_new > tr.best_value + rtol * abs(tr.best_value):
                tr.success, tr.failure = tr.success + 1, 0
            else:
                tr.success, tr.failure = 0, tr.failure + 1
            if tr.success == self.tau_succ:
                tr.length, tr.success = min(2.0 * tr.length,
                                            float(self.config["length_max"])), 0
            elif tr.failure == self.tau_fail:
                tr.length, tr.failure = tr.length / 2.0, 0
            for r in rows:
                self._update_best(tr, r)
            if tr.length < float(self.config["length_min"]):
                self._restart(tr, reason="collapsed")

    def _update_best(self, tr: _TrustRegion, row: int) -> None:
        if self.Y[row] > tr.best_value:
            tr.best_value = float(self.Y[row])
            tr.best_x = self.X[row].copy()

    def _restart(self, tr: _TrustRegion, reason: str) -> None:
        self.history.append(self._record(tr, reason))
        self.trs[tr.tr_id] = self._new_tr(tr.tr_id, tr.restart + 1)

    def _record(self, tr: _TrustRegion, reason: str) -> dict:
        rec = {"tr_id": tr.tr_id, "restart": tr.restart, "reason": reason,
               "n_points_start": tr.n_start, "n_points_end": len(self.Y),
               "n_region_points": len(tr.idx), "final_length": tr.length,
               "best_value": tr.best_value}
        for j in range(self.dim):
            rec[f"x{j}"] = (float(tr.best_x[j]) if tr.best_x is not None else np.nan)
        return rec

    # ── reporting ──

    def summary(self) -> dict:
        return {"sampling": self.config["sampling"],
                "n_restarts": len(getattr(self, "history", [])),
                "tau_fail": getattr(self, "tau_fail", None),
                "n_candidates": getattr(self, "n_cand", None)}

    def write_artifacts(self, trial_dir: str) -> None:
        import os

        import pandas as pd

        if not hasattr(self, "trs"):
            return
        rows = list(self.history) + [self._record(tr, "active_at_end")
                                     for tr in self.trs if tr.idx]
        pd.DataFrame(rows).to_csv(os.path.join(trial_dir, "turbo_trust_regions.csv"),
                                  index=False)
