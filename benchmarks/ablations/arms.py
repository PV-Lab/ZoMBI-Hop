"""
benchmarks/ablations/arms.py
============================
The ablation ARMS — one dataclass per optimiser variant, plus the monkeypatches
that realise the variants hyperparameters alone cannot express.

Why monkeypatches rather than flags in ``src/core/zombihop.py``
--------------------------------------------------------------
An ablation is a claim about the *published* optimiser: "this is what ZoMBI-Hop
does when you take component X away". Threading four ``if ablation_mode == …``
branches through ``ZoMBIHop.run`` / ``LineBO`` / ``GPSimplex`` would put harness
code on the baseline's own execution path, so a bug in the harness could silently
change the numbers the baseline reports. Patching from the outside keeps
``src/core`` byte-identical for the baseline arm: with no arm active, nothing in
this file runs.

Every patch is a context manager that restores the original attribute on exit, so
arms can run back-to-back in one process without leaking into each other.

The four ablations
------------------
A1  ``k_restarts``       — k independent ZoMBI runs (fresh data, fresh GP, no
                           needle memory) sharing the budget, vs one ZoMBI-Hop run.
                           No patch: it is a *harness*-level arm (see restarts.py),
                           because "independent" means separate optimiser objects.
A2  ``no_zoom``          — the trust region never contracts. ``max_zooms=1`` stops
                           the zoom loop advancing and ``min_zoom_for_needle=0``
                           re-permits needle declaration at zoom 0 (the default of 1
                           would otherwise make needles unreachable, and the arm
                           would measure "no needles", not "no zooming"). The patch
                           additionally neuters ``determine_new_bounds``, which the
                           failure-retry path calls directly and which would
                           otherwise reintroduce a contracted box by the back door.
A3  ``random_chords``    — LineBO's acquisition-integrated line ranking replaced by
                           random chords through the GP's best candidate. Isolates
                           "which line" from "which point".
A4  ``isotropic_basins`` — each needle's Hessian ellipsoid is replaced by the
                           VOLUME-MATCHED sphere, so the arm differs from the
                           baseline in basin *shape* only, never in how much of the
                           simplex a needle removes from play.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager, Iterator

from ._paths import ensure_paths

ensure_paths()

import numpy as np  # noqa: E402
import torch  # noqa: E402


# ─── Arm definition ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Arm:
    """One optimiser variant, fully reconstructible from its ``name``.

    A worker process is handed only the arm name (it comes off a task queue), so
    everything needed to reproduce the variant must live in this registry — never
    in the caller's closure.

    Attributes
    ----------
    name : queue-safe identifier; also the run-directory name.
    label : human-readable, used in plot legends and tables.
    hparam_overrides : merged OVER the campaign's base hyperparameters. Keys need
        not appear in ``optimize/hparam_space.HPARAM_SPACE``; anything ``ZoMBIHop``
        accepts as a keyword works, since ``run_mobo.run_single_trial`` splats the
        dict straight into the constructor.
    patches : names in :data:`PATCHES`, entered around the whole trial.
    runner : ``"single"`` (one ZoMBIHop object) or ``"restarts"`` (k independent
        ones; see ``restarts.run_restart_trial``).
    runner_kwargs : extra keyword arguments for the ``"restarts"`` runner. The
        campaign may override these (``--n-restarts``), which is why they are
        recorded in the manifest rather than assumed from this file.
    is_baseline : marks the unablated method. Exactly one arm carries it, and the
        summary plots it first, in a fixed colour, in every ablation's figure.
    """

    name: str
    label: str
    description: str
    hparam_overrides: dict[str, Any] = field(default_factory=dict)
    patches: tuple[str, ...] = ()
    runner: str = "single"
    runner_kwargs: dict[str, Any] = field(default_factory=dict)
    is_baseline: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "hparam_overrides": dict(self.hparam_overrides),
            "patches": list(self.patches),
            "runner": self.runner,
            "runner_kwargs": dict(self.runner_kwargs),
            "is_baseline": self.is_baseline,
        }


@dataclass(frozen=True)
class Ablation:
    """One comparison: a baseline arm and the variant(s) contrasted against it.

    ``arms[0]`` is the reference every paired statistic in the summary is computed
    against, so it is always the baseline.
    """

    key: str
    title: str
    question: str
    arms: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"key": self.key, "title": self.title,
                "question": self.question, "arms": list(self.arms)}


# ─── A2: no zooming ──────────────────────────────────────────────────────────────

@contextlib.contextmanager
def patch_no_zoom_bounds() -> Iterator[None]:
    """Make ``DataHandler.determine_new_bounds`` a no-op returning the global box.

    ``max_zooms=1`` already stops the zoom LOOP advancing, but it is not the only
    caller: ``ZoMBIHop._handle_failure_retry`` case 2 calls
    ``determine_new_bounds(add_to_history=False)`` directly, and the failure path is
    common (it fires whenever acquisition ascent lands in a penalised region).
    Without this, the "no zooming" arm would still spend much of an activation
    inside a contracted box — the very thing the arm is supposed to remove.

    Returning ``_full_bounds_ref`` rather than ``[0,1]^d`` keeps a caller-tightened
    search box (e.g. one axis capped at 0.3) intact; that box is the global region
    as far as the rest of the optimiser is concerned.
    """
    from src.utils.datahandler import DataHandler

    original = DataHandler.determine_new_bounds

    def never_zoom(self, add_to_history: bool = True) -> torch.Tensor:
        ref = getattr(self, "_full_bounds_ref", None)
        if ref is None:
            ref = self.bounds
        return ref.clone()

    DataHandler.determine_new_bounds = never_zoom
    try:
        yield
    finally:
        DataHandler.determine_new_bounds = original


# ─── A3: random chords through the best candidate ────────────────────────────────

@contextlib.contextmanager
def patch_random_chord_lines() -> Iterator[None]:
    """Replace LineBO's acquisition ranking with random chords through ``x_tell``.

    The baseline draws ``num_lines`` chords through random interior points of the
    trust region, integrates the acquisition along each, and returns them sorted by
    that integral — the objective is then measured on the top-ranked line. This arm
    changes both halves: every chord passes through ``x_tell`` (the point the GP
    actually proposed), and the order is a random permutation, so the acquisition
    has no say in orientation.

    Anchoring at ``x_tell`` is what makes this the *interesting* control rather than
    a strictly worse one: the arm still measures through the GP's chosen point, so a
    small gap to the baseline says line ORIENTATION is doing little work and the
    candidate is carrying the search.

    The return signature is byte-compatible with the original — ``(x_left,
    x_right)``, best-first — so ``run_mobo.make_linebo_wrapper``'s ``line_0`` /
    ``line_1`` bookkeeping and ``LineBO.sampler`` both keep working untouched.
    """
    from src.core import linebo as linebo_mod
    from src.utils.simplex import random_simplex, random_simplex_direction

    original = linebo_mod.LineBO.ranked_line_endpoints

    def random_chords_through_candidate(self, x_tell, bounds=None,
                                        acquisition_function=None):
        x_tell = x_tell.to(device=self.device, dtype=self.dtype)
        d = self.d
        if bounds is not None:
            bounds_eff = bounds.to(device=self.device, dtype=self.dtype)
        else:
            bounds_eff = torch.stack([
                torch.zeros(d, device=self.device, dtype=self.dtype),
                torch.ones(d, device=self.device, dtype=self.dtype),
            ], dim=0)

        # x_tell comes from acquisition ascent inside `bounds`, so it is normally
        # interior; a resumed or shrunken box can still leave it a hair outside, and
        # batch_line_bounds_segments needs an inside anchor (outside, every
        # t-interval collapses and NO line comes back). Fall back to a uniform
        # interior draw rather than clamping — clamping onto a face makes half the
        # directions degenerate.
        inside = bool(((x_tell >= bounds_eff[0] - 1e-9) &
                       (x_tell <= bounds_eff[1] + 1e-9)).all())
        if inside:
            anchor = x_tell
        else:
            anchor = random_simplex(1, bounds_eff[0], bounds_eff[1],
                                    device=str(self.device),
                                    torch_dtype=self.dtype)[0]

        # Oversample: degenerate directions (anchor on a face, direction pointing
        # straight out of the box) are dropped by the validity mask.
        k = self.num_lines
        oversample = max(k * 3, k + 20)
        directions = random_simplex_direction(
            oversample, d, device=str(self.device), dtype=self.dtype)
        x_left, x_right, _, _, _ = linebo_mod.batch_line_bounds_segments(
            anchor, directions, bounds_eff)

        if x_left.shape[0] == 0:
            # Degenerate box, or an anchor pinned in a corner: fall back to the
            # baseline's own interior-anchored chords so the run still measures
            # something. Still unranked — this arm never consults the acquisition.
            x_left, x_right = linebo_mod.random_chords_through_simplex(
                k, bounds_eff, self.device, self.dtype)
        if x_left.shape[0] == 0:
            return x_tell.unsqueeze(0), x_tell.unsqueeze(0)

        # Random order: the ablation IS the missing ranking.
        order = torch.randperm(x_left.shape[0], device=x_left.device)[:k]
        return x_left[order], x_right[order]

    linebo_mod.LineBO.ranked_line_endpoints = random_chords_through_candidate
    try:
        yield
    finally:
        linebo_mod.LineBO.ranked_line_endpoints = original


# ─── A4: isotropic (volume-matched) needle basins ────────────────────────────────

def isotropize_precision(M: torch.Tensor, *, max_radius: float) -> torch.Tensor:
    """The sphere with the same volume as the ellipsoid ``u^T M u <= 1``.

    ``M`` is an SPD precision in tangent space; its semi-axes are ``1/sqrt(lambda_i)``
    and the enclosed volume is proportional to ``det(M)^(-1/2)``. Replacing every
    eigenvalue with their GEOMETRIC mean leaves ``det(M)`` — and therefore the
    volume — exactly unchanged, so the isotropic arm removes the same amount of
    simplex from play as the anisotropic one. An arithmetic mean, or the largest or
    smallest axis, would confound shape with size and turn A4 into an accidental
    test of penalty radius.

    ``max_radius`` reapplies the same semi-axis cap ``determine_penalty_ellipsoid``
    enforces, so a volume-matched sphere can never out-reach the ellipsoid it
    replaces.
    """
    evals = torch.linalg.eigvalsh(0.5 * (M + M.transpose(-1, -2)))
    evals = evals.clamp_min(1e-30)
    lam_iso = torch.exp(torch.log(evals).mean())
    lam_iso = torch.clamp(lam_iso, min=1.0 / float(max_radius) ** 2)
    eye = torch.eye(M.shape[-1], device=M.device, dtype=M.dtype)
    return eye * lam_iso


@contextlib.contextmanager
def patch_isotropic_basins() -> Iterator[None]:
    """Force every needle/exclusion basin to a volume-matched sphere.

    Patching ``GPSimplex.determine_penalty_ellipsoid`` covers all three sites that
    ever mint an ``M``: the live declaration in ``_declare_needle_from_point``, the
    capped-activation exclusion zone in ``_penalize_capped_zone``, and the wholesale
    refit in ``recompute_all_ellipsoids`` (which loops over this same method).
    ``shrink_all_needle_radii`` only rescales an existing ``M`` by a scalar, which
    preserves isotropy — so nothing downstream can reintroduce anisotropy later.
    """
    from src.utils.gp_simplex import GPSimplex

    original = GPSimplex.determine_penalty_ellipsoid

    def isotropic_penalty_ellipsoid(self, needle, drop_fraction=0.25,
                                    eigenvalue_floor=1e-6, max_radius=1.0,
                                    acq_fn=None):
        M, B = original(self, needle, drop_fraction=drop_fraction,
                        eigenvalue_floor=eigenvalue_floor,
                        max_radius=max_radius, acq_fn=acq_fn)
        return isotropize_precision(M, max_radius=max_radius), B

    GPSimplex.determine_penalty_ellipsoid = isotropic_penalty_ellipsoid
    try:
        yield
    finally:
        GPSimplex.determine_penalty_ellipsoid = original


# ─── Shared instrumentation (applied to EVERY arm, baseline included) ────────────

@contextlib.contextmanager
def patch_points_csv_zoom_size() -> Iterator[None]:
    """Add a ``zoom_size`` column to every ``points.csv``.

    ``metric_dup_fraction`` shrinks the duplicate radius for points measured inside
    a zoom box, using the per-point zoom size ``run_single_trial`` derives from its
    snapshot records and then throws away. The k-restarts arm has to RECOMPUTE the
    trial-level dup fraction over the pooled points of all its restarts, long after
    those records are gone — so without this column its dup would silently be the
    unscaled variant while every other arm's is scaled, and A1's dup comparison
    would be measuring the metric rather than the optimiser.

    Applied to all arms so the column means the same thing everywhere. It is purely
    additive: every reader of ``points.csv`` in this repo selects columns by name.
    """
    import run_mobo as rm

    original = rm.write_points_csv

    def write_points_csv_with_zoom_size(path, dh, snap_records, *, dim=3):
        original(path, dh, snap_records, dim=dim)
        try:
            import pandas as pd

            df = pd.read_csv(path)
            df["zoom_size"] = rm._zoom_size_per_point(len(df), snap_records)
            df.to_csv(path, index=False)
        except Exception as exc:  # never fail a trial over an extra column
            print(f"    [ablation] zoom_size column failed: {exc}")

    rm.write_points_csv = write_points_csv_with_zoom_size
    try:
        yield
    finally:
        rm.write_points_csv = original


@contextlib.contextmanager
def capture_convergence(sink: list[dict]) -> Iterator[None]:
    """Record the arrays ``run_mobo.plot_convergence`` is called with, per run.

    The k-restarts arm redraws ONE convergence plot over its pooled restarts, which
    needs ``Y_all``, the penalty mask and the needle sample indices — state that
    lives on the ``DataHandler`` and is deliberately released at the end of every
    ``run_single_trial`` (it is the largest object in the process; holding k of them
    is how the SLURM jobs used to get OOM-killed). Tapping the plot call is the one
    place all three are already in hand as plain arrays.

    Calls through, so each restart still writes its own ``convergence.png``.
    """
    import run_mobo as rm

    original = rm.plot_convergence

    def tapped(path, dh, maximize, activations=None):
        try:
            mask = dh.get_penalty_mask()
            needle_idx = getattr(dh, "needle_indices", None)
            sink.append({
                "Y_all": dh.Y_all.detach().cpu().numpy().ravel().copy(),
                "penalty_mask": (mask.detach().cpu().numpy().ravel().copy()
                                 if mask is not None else None),
                "needle_indices": (needle_idx.detach().cpu().numpy().ravel().copy()
                                   if needle_idx is not None and needle_idx.numel() > 0
                                   else None),
                "activations": (np.asarray(activations).ravel().copy()
                                if activations is not None else None),
            })
        except Exception as exc:
            print(f"    [ablation] convergence capture failed: {exc}")
        return original(path, dh, maximize, activations=activations)

    rm.plot_convergence = tapped
    try:
        yield
    finally:
        rm.plot_convergence = original


# ─── Patch registry ──────────────────────────────────────────────────────────────

PATCHES: dict[str, Callable[[], ContextManager[None]]] = {
    "no_zoom_bounds": patch_no_zoom_bounds,
    "random_chord_lines": patch_random_chord_lines,
    "isotropic_basins": patch_isotropic_basins,
}


@contextlib.contextmanager
def arm_context(arm: Arm) -> Iterator[None]:
    """Enter every patch the arm declares and unwind them on exit."""
    with contextlib.ExitStack() as stack:
        for name in arm.patches:
            try:
                factory = PATCHES[name]
            except KeyError:
                raise KeyError(
                    f"arm {arm.name!r} requests unknown patch {name!r}; "
                    f"known patches: {sorted(PATCHES)}") from None
            stack.enter_context(factory())
        yield


# ─── Arm registry ────────────────────────────────────────────────────────────────

BASELINE_ARM = "zombi_hop"

ARMS: dict[str, Arm] = {
    arm.name: arm for arm in (
        Arm(
            name=BASELINE_ARM,
            label="ZoMBI-Hop (full)",
            description=(
                "The published method, unmodified: one optimiser with shared "
                "history, zooming trust regions, acquisition-ranked LineBO lines "
                "and anisotropic Hessian basins. Shared by all four ablations as "
                "the reference arm, so it is run once and compared four ways."
            ),
            is_baseline=True,
        ),
        Arm(
            name="k_restarts",
            label="k independent ZoMBI restarts",
            description=(
                "A1. The wall-clock budget is split across k independent ZoMBI "
                "runs. Each restart gets a fresh optimiser, fresh initial lines and "
                "an empty needle set, so nothing a restart learns — measured points, "
                "GP posterior, or where an optimum already sits — reaches the next "
                "one. Its discovered set is the pooled needles of all restarts, "
                "scored exactly like a single ZoMBI-Hop run's."
            ),
            runner="restarts",
            runner_kwargs={"n_restarts": 4, "max_activations_per_restart": 1,
                           "fill_budget": True},
        ),
        Arm(
            name="no_zoom",
            label="No zooming",
            description=(
                "A2. The trust region stays at the global search box for the whole "
                "run: max_zooms=1 stops the zoom loop advancing and the "
                "determine_new_bounds patch closes the failure-retry back door. "
                "min_zoom_for_needle drops to 0 so needles can still be declared at "
                "zoom 0 — otherwise the arm would measure 'no needles are "
                "reachable', not 'zooming does not help'."
            ),
            hparam_overrides={"max_zooms": 1, "min_zoom_for_needle": 0},
            patches=("no_zoom_bounds",),
        ),
        Arm(
            name="random_chords",
            label="Random chords through best candidate",
            description=(
                "A3. LineBO still measures a full chord through the GP's proposed "
                "candidate, but the chord's orientation is drawn uniformly instead "
                "of being the best of num_lines candidates by integrated "
                "acquisition. Isolates the value of choosing the LINE from the value "
                "of choosing the point."
            ),
            patches=("random_chord_lines",),
        ),
        Arm(
            name="isotropic_basins",
            label="Isotropic (spherical) basins",
            description=(
                "A4. Each needle's anisotropic Hessian ellipsoid is replaced by the "
                "sphere of equal volume, so the arm differs in basin SHAPE only — "
                "the same fraction of the simplex is penalised either way. Tests "
                "whether orienting the basin to the local curvature buys anything "
                "over penalising a ball."
            ),
            patches=("isotropic_basins",),
        ),
    )
}

ABLATIONS: dict[str, Ablation] = {
    ab.key: ab for ab in (
        Ablation(
            key="A1",
            title="Independent restarts vs hopping",
            question="Do k independent ZoMBI restarts match ZoMBI-Hop on the same budget?",
            arms=(BASELINE_ARM, "k_restarts"),
        ),
        Ablation(
            key="A2",
            title="Zooming vs no zooming",
            question="Does the contracting trust region find optima the global search misses?",
            arms=(BASELINE_ARM, "no_zoom"),
        ),
        Ablation(
            key="A3",
            title="LineBO line selection vs random chords",
            question=("Does acquisition-ranked line selection beat a random chord "
                      "through the same candidate?"),
            arms=(BASELINE_ARM, "random_chords"),
        ),
        Ablation(
            key="A4",
            title="Anisotropic vs isotropic basins",
            question=("Does shaping each needle's basin to the local curvature beat "
                      "a volume-matched sphere?"),
            arms=(BASELINE_ARM, "isotropic_basins"),
        ),
    )
}

ABLATION_KEYS = list(ABLATIONS)


def arms_for(ablation_keys: list[str]) -> list[str]:
    """De-duplicated arm names needed to run *ablation_keys*, baseline first.

    The baseline appears in all four ablations; running it four times would burn a
    quarter of the campaign re-measuring the same thing. The queue holds each arm
    once and the summary reads it into every figure that needs it.
    """
    seen: list[str] = []
    for key in ablation_keys:
        for name in ABLATIONS[key].arms:
            if name not in seen:
                seen.append(name)
    return sorted(seen, key=lambda n: (not ARMS[n].is_baseline, n))
