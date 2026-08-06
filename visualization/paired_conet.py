"""
paired_conet.py
===============
Render a run's ``conet.png`` and ``conet_uniform.png`` as ONE map fitted once.

Why
---
Both CoNets used to fit their own UMAP: ``plot_10d.py`` on the run's samples and
``uniform_baseline_conet.py`` on a uniform draw from the same landscape. Two
independent fits means two unrelated coordinate systems, so a cluster at the
top-left of one image has nothing to do with the top-left of the other and the
pair cannot be read side by side — which defeats the point of having a baseline.

Here the UMAP is fitted ONCE, on the uniform baseline. That fit defines the map:
its embedding, its gap/purity layout warp, its component colours and its axis
limits. The run's real samples are then placed into that existing map without
refitting anything, so the real data is drawn in the same shape as the uniform
data and the two PNGs are directly overlayable.

Placing the real samples (out-of-sample extension)
--------------------------------------------------
``build_conet_structure`` fits UMAP with ``metric="precomputed"`` on an N×N
co-occurrence distance matrix, and umap-learn cannot ``.transform()`` new points
under a precomputed metric — there is no parametric mapping to evaluate. So the
real samples are placed by a k-nearest-landmark interpolation, the standard
out-of-sample extension for a landmark embedding:

  1. Compute the same co-occurrence similarity ``S`` between each real sample and
     every uniform sample (``sum_c min(comp_ic, comp_jc)`` over active components).
  2. Convert to the fit's distance, ``d = 1 - (S / S_uu_max) ** CN_BETA``. The
     normaliser is the UNIFORM block's maximum — the scale the embedding was
     fitted under — so distances mean the same thing on both sides.
  3. Place each real sample at the inverse-distance-weighted mean of its ``k``
     nearest uniform samples' raw embedding coordinates.

The uniform draw covers the simplex by construction, so every real sample has
genuinely near landmarks and the interpolation is well posed. What it cannot do
is invent structure the uniform baseline does not resolve: a tight cluster of
real samples that all share the same nearest landmarks collapses toward one
point. That is a faithful statement about the map, not an artifact to correct —
it means the optimiser concentrated inside a region the uniform draw treats as a
single neighbourhood.

One frame per LANDSCAPE
-----------------------
The frame is a pure function of the landscape, never of the run being drawn. That
matters when several runs share a landscape (e.g. a showdown, where four
hyperparameter configurations are scored on the same five landscapes): all of them
must land in one coordinate system or their CoNets cannot be compared, which is
the whole reason for drawing them.

So three things are fixed by the landscape alone:

  * **the uniform draw** — a fixed ``--uniform-n`` count at a fixed ``--seed``.
    (It used to be sized to each run's own ``points.csv`` row count, which meant
    runs on one landscape drew *different* baselines and therefore fitted
    different UMAPs. That is the bug this design removes.)
  * **the axis limits and colour bounds** — taken from the uniform baseline alone,
    not from the uniform+real union, which would vary per run.
  * **the fitted frame itself** — cached under ``--frame-cache`` and keyed on the
    landscape config plus the fit parameters, so every run reuses one transform
    rather than refitting an identical UMAP (a large speedup as well as a
    correctness fix). The uniform PNG is identical for all of them, so it is
    rendered once and copied.

Usage
-----
  python visualization/paired_conet.py --run <RUN_DIR>
  python visualization/paired_conet.py --run <RUN_DIR> --k 12 --uniform-n 20000
  python visualization/paired_conet.py --run <RUN_DIR> --no-frame-cache

``<RUN_DIR>`` must hold an ``ensemble_config.json`` (to rebuild the landscape)
plus ``points.csv`` / ``needles.csv``. Writes ``conet.png`` and
``conet_uniform.png`` into the run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import plot_10d as p10  # noqa: E402
from synthetic_data import Ensemble  # noqa: E402

# How many uniform landmarks each real sample is interpolated from. Small enough
# that a real sample stays local to its own neighbourhood, large enough that the
# placement is not dominated by a single landmark's UMAP jitter.
DEFAULT_K = 8

# Uniform landmark count. FIXED rather than matched to the run's own sample count:
# the baseline has to depend only on the landscape, or two runs of the same
# landscape fit different UMAPs and their CoNets stop being comparable. 15000 sits
# in the middle of the observed run sizes (~5k-18k) and keeps the N×N co-occurrence
# matrix (~1.8 GB) within a normal job's memory.
DEFAULT_UNIFORM_N = 15000

# Bump when a change alters the fitted frame, to invalidate stale cache entries.
FRAME_CACHE_VERSION = 1
DEFAULT_FRAME_CACHE = _REPO / "optimize" / "runs" / ".conet_frames"


# ── landscape + sampling ────────────────────────────────────────────────────────

def _count_csv_rows(path: Path) -> int:
    """Data-row count of a CSV (excludes the header), or 0 if absent."""
    if not path.exists():
        return 0
    with open(path, newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def build_ensemble(run_dir: Path) -> tuple[Ensemble, dict]:
    """Reconstruct the run's ensemble landscape from its ``ensemble_config.json``."""
    cfg_path = Path(run_dir) / "ensemble_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"no ensemble_config.json in {run_dir} — is this an ensemble run?")
    cfg = json.loads(cfg_path.read_text())
    return Ensemble(**cfg), cfg


def uniform_samples(dim: int, n: int, seed: int) -> np.ndarray:
    """N points drawn uniformly from the (dim-1)-simplex (Dirichlet with all-ones)."""
    return np.random.default_rng(seed).dirichlet(np.ones(dim), size=n)


# ── shared composition space ────────────────────────────────────────────────────

def _shared_reduce(Xu: np.ndarray, Xr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row-normalise both sample sets and keep ONE shared set of active columns.

    ``plot_10d._reduce_comp`` drops all-zero columns per dataset. Doing that
    independently could leave the two sets with different column counts (a run
    that never touched a component vs a uniform draw that always does), and then
    their co-occurrence similarities would not be comparable at all. The active
    mask is therefore the UNION over both sets, applied identically to each.
    """
    def _norm(X):
        s = X.sum(1, keepdims=True)
        return X / np.where(s == 0, 1.0, s)

    Cu, Cr = _norm(np.asarray(Xu, float)), _norm(np.asarray(Xr, float))
    active = (Cu.sum(0) > 0) | (Cr.sum(0) > 0)
    if not (2 <= int(active.sum()) < Cu.shape[1]):
        active = np.ones(Cu.shape[1], bool)
    return Cu[:, active], Cr[:, active], active


def _cooccurrence(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cross co-occurrence similarity ``S[i, j] = sum_c min(A_ic, B_jc)``.

    Identical to the within-set ``S`` ``build_conet_structure`` builds (inactive
    fractions zeroed at ``CN_FLOOR`` first), just evaluated between two different
    sample sets so real samples can be compared against uniform landmarks.
    """
    Am = np.where(A > p10.CN_FLOOR, A, 0.0)
    Bm = np.where(B > p10.CN_FLOOR, B, 0.0)
    S = np.zeros((len(A), len(B)))
    for c in range(A.shape[1]):
        S += np.minimum.outer(Am[:, c], Bm[:, c])
    return S


def embed_out_of_sample(comp_new: np.ndarray, comp_fit: np.ndarray,
                        E_fit_raw: np.ndarray, k: int = DEFAULT_K) -> np.ndarray:
    """Place *comp_new* into the map fitted on *comp_fit* by k-nearest-landmark interpolation.

    ``E_fit_raw`` is the fitted set's RAW embedding (pre gap/purity warp) — the
    warps are replayed afterwards by ``build_conet_structure(frame=...)``, so
    interpolating in raw coordinates and then warping applies exactly the same
    transform chain the fitted points went through.
    """
    S_cross = _cooccurrence(comp_new, comp_fit)
    S_fit = _cooccurrence(comp_fit, comp_fit)
    np.fill_diagonal(S_fit, 0.0)
    # The fitted block's maximum is the normaliser the embedding was built under;
    # reusing it (rather than the cross block's own max) keeps the new points'
    # distances on the fit's scale.
    denom = S_fit.max() + 1e-12
    D = 1.0 - (S_cross / denom) ** p10.CN_BETA

    k = int(max(1, min(k, len(comp_fit))))
    nn = np.argpartition(D, k - 1, axis=1)[:, :k]
    d = np.take_along_axis(D, nn, axis=1)
    # Inverse-distance weights: a landmark at distance ~0 dominates (an exact
    # composition match should land exactly on that landmark), and the epsilon
    # keeps that from dividing by zero.
    w = 1.0 / (d + 1e-9)
    w /= w.sum(axis=1, keepdims=True)
    return np.einsum("nk,nkd->nd", w, E_fit_raw[nn])


# ── per-landscape frame cache ───────────────────────────────────────────────────

def frame_key(cfg: dict, *, uniform_n: int, seed: int,
              purity, spread, gap_reach) -> str:
    """Cache key for a fitted frame: the landscape plus every fit parameter.

    The ensemble config alone identifies the landscape (its keys are the
    ``Ensemble(**cfg)`` kwargs, so equal configs are the same surface). The fit
    parameters are folded in because changing any of them changes the map, and a
    stale hit would silently mix two different frames.
    """
    payload = json.dumps({
        "version": FRAME_CACHE_VERSION,
        "cfg": cfg,
        "uniform_n": int(uniform_n),
        "seed": int(seed),
        "purity": purity if purity is not None else p10.CN_PURITY_THR,
        "spread": spread if spread is not None else p10.CN_UMAP_MD,
        "gap_reach": gap_reach if gap_reach is not None else p10.CN_GAP_REACH,
        "beta": p10.CN_BETA, "floor": p10.CN_FLOOR,
        "umap_nn": p10.CN_UMAP_NN, "umap_seed": p10.CN_SEED,
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _save_frame(path: Path, *, comp_u, E_raw, frame, limits, bounds, png: Path) -> None:
    """Persist a fitted frame (arrays in .npz, the rest as an embedded JSON blob).

    Written to a process-unique temporary and renamed into place: several jobs may
    render runs of the SAME landscape concurrently and race to cache it, and a
    reader must never see a half-written file. Rename is atomic within a directory,
    so the loser of a race simply overwrites with byte-identical content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "gap": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in frame["gap"].items()},
        "purity": {k: (v.tolist() if isinstance(v, np.ndarray) else list(v)
                       if isinstance(v, tuple) else v)
                   for k, v in frame["purity"].items()},
        "ccolor": {k: list(v) for k, v in frame["ccolor"].items()},
        "limits": [list(limits[0]), list(limits[1])],
        "bounds": {k: list(v) for k, v in bounds.items()},
    }
    import os
    tmp_npz = path.with_suffix(f".{os.getpid()}.tmp.npz")
    tmp_png = path.with_suffix(f".{os.getpid()}.tmp.png")
    np.savez_compressed(tmp_npz, comp_u=comp_u, E_raw=E_raw,
                        meta=np.array(json.dumps(meta)))
    shutil.copyfile(png, tmp_png)
    # PNG first: _load_frame requires BOTH, so publishing the npz last means a
    # concurrent reader never sees an npz whose PNG has not landed yet.
    os.replace(tmp_png, path.with_suffix(".uniform.png"))
    os.replace(tmp_npz, path)


def _load_frame(path: Path) -> dict | None:
    """Load a cached frame, or None if absent/unreadable (a miss simply refits)."""
    png = path.with_suffix(".uniform.png")
    if not path.is_file() or not png.is_file():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        gap = {k: (np.asarray(v) if isinstance(v, list) else v)
               for k, v in meta["gap"].items()}
        pur = dict(meta["purity"])
        pur["anchors"] = np.asarray(pur["anchors"], dtype=float)
        pur["clip"] = tuple(pur["clip"])
        return {
            "comp_u": z["comp_u"],
            "frame": {"gap": gap, "purity": pur,
                      "ccolor": {k: tuple(v) for k, v in meta["ccolor"].items()},
                      "E_raw": z["E_raw"]},
            "limits": (tuple(meta["limits"][0]), tuple(meta["limits"][1])),
            "bounds": {k: tuple(v) for k, v in meta["bounds"].items()},
            "uniform_png": png,
        }
    except Exception as exc:
        print(f"[paired-conet] cached frame {path.name} unreadable ({exc}); refitting.")
        return None


# ── main render ─────────────────────────────────────────────────────────────────

def build_landscape_frame(ens, cfg, dim, names, *, uniform_n: int, seed: int,
                          purity, spread, gap_reach) -> dict:
    """Fit the map on this landscape's uniform baseline and render its PNG.

    Everything returned depends only on the landscape and the fit parameters — never
    on any run — so every run of this landscape can share it verbatim.
    """
    Xu = uniform_samples(dim, uniform_n, seed)
    Yu = ens.predict(Xu)
    comp_u, active = p10._reduce_comp(Xu)
    resp_u = np.asarray(Yu, float).reshape(-1, 1)
    iters_u = np.arange(len(comp_u), dtype=float)
    # The landscape's TRUE optima, ranked best-first, in the same reduced space.
    centers = np.asarray(ens.centers, dtype=float)
    needles_u = p10._reduce_needles(centers[np.argsort(-ens.predict(centers))], active)

    t0 = time.time()
    Mu, Fu = p10.build_conet(
        comp_u, names, resp_u, iters_u,
        umap_md=spread if spread is not None else p10.CN_UMAP_MD,
        gap_reach=gap_reach if gap_reach is not None else p10.CN_GAP_REACH,
        purity_thr=purity if purity is not None else p10.CN_PURITY_THR,
        needles_comp=needles_u, show_needles=True,
    )
    # Limits and colour bounds from the UNIFORM baseline only. Using the uniform+real
    # union would make both vary per run, which is exactly what has to stop.
    limits = p10._cn_view_limits(Mu["E"])
    bounds = p10.conet_bounds(resp_u)
    print(f"[paired-conet] landscape frame fitted: {len(comp_u)} uniform samples, "
          f"{len(Mu['needles'])} true optima ({time.time() - t0:.0f}s)")
    return {"comp_u": comp_u, "active": active, "M": Mu, "F": Fu,
            "frame": Mu["frame"], "E_raw": Mu["E_raw"],
            "limits": limits, "bounds": bounds, "resp_u": resp_u,
            "seed_tag": cfg.get("seed"), "n_uniform": len(comp_u)}


def render_paired_conets(run_dir, *, k: int = DEFAULT_K,
                         uniform_n: int = DEFAULT_UNIFORM_N,
                         seed: int = 0, show_needles: bool = True,
                         purity=None, spread=None, gap_reach=None,
                         conet_path=None, uniform_path=None,
                         frame_cache: Path | None = DEFAULT_FRAME_CACHE
                         ) -> tuple[Path, Path]:
    """Draw both CoNets of a run in its landscape's shared frame. Returns both PNG paths."""
    run_dir = Path(run_dir)
    conet_path = Path(conet_path) if conet_path else run_dir / "conet.png"
    uniform_path = Path(uniform_path) if uniform_path else run_dir / "conet_uniform.png"

    ens, cfg = build_ensemble(run_dir)
    dim = int(cfg["dim"])

    # Real run samples, loaded through the same path plot_10d.py --run uses.
    comp_r, names_r, resp_r, iters_r, title_r, needles_r = \
        p10.load_run_conet_data(run_dir)
    names = list(names_r)

    # --- 1. the landscape's frame (cached, so all its runs share one) ----------
    cache_path = None
    cached = None
    if frame_cache is not None:
        key = frame_key(cfg, uniform_n=uniform_n, seed=seed,
                        purity=purity, spread=spread, gap_reach=gap_reach)
        cache_path = Path(frame_cache) / f"{key}.npz"
        cached = _load_frame(cache_path)

    if cached is not None:
        comp_u = cached["comp_u"]
        frame_params = cached["frame"]
        E_u_raw = frame_params.pop("E_raw")
        limits, bounds = cached["limits"], cached["bounds"]
        shutil.copyfile(cached["uniform_png"], uniform_path)
        print(f"[paired-conet] reusing cached landscape frame {cache_path.name} "
              f"({len(comp_u)} uniform samples)")
    else:
        built = build_landscape_frame(ens, cfg, dim, names, uniform_n=uniform_n,
                                      seed=seed, purity=purity, spread=spread,
                                      gap_reach=gap_reach)
        comp_u = built["comp_u"]
        frame_params = built["frame"]
        E_u_raw = built["E_raw"]
        limits, bounds = built["limits"], built["bounds"]
        # The uniform PNG is a property of the landscape, so its title names the
        # landscape (config seed), never the run that happened to trigger the fit.
        p10.save_png(
            built["M"], built["F"], bounds,
            f"UNIFORM baseline — landscape seed {built['seed_tag']} · "
            f"{built['n_uniform']} samples · shared UMAP frame",
            str(uniform_path))
        if cache_path is not None:
            try:
                _save_frame(cache_path, comp_u=comp_u, E_raw=E_u_raw,
                            frame=frame_params, limits=limits, bounds=bounds,
                            png=uniform_path)
                print(f"[paired-conet] cached landscape frame -> {cache_path.name}")
            except Exception as exc:
                print(f"[paired-conet] could not cache frame ({exc}); continuing.")

    if comp_r.shape[1] != comp_u.shape[1]:
        raise ValueError(
            f"the run reduced to {comp_r.shape[1]} active components but the "
            f"landscape's uniform baseline has {comp_u.shape[1]}; there is no shared "
            f"composition space to embed into (run: {run_dir})")

    # --- 2. place the real samples into that same map -------------------------
    t0 = time.time()
    E_r_raw = embed_out_of_sample(comp_r, comp_u, E_u_raw, k=k)
    frame = dict(frame_params)
    frame["E_raw"] = E_r_raw
    Mr, _ = p10.build_conet(
        comp_r, names, resp_r, iters_r,
        umap_md=spread if spread is not None else p10.CN_UMAP_MD,
        gap_reach=gap_reach if gap_reach is not None else p10.CN_GAP_REACH,
        purity_thr=purity if purity is not None else p10.CN_PURITY_THR,
        needles_comp=needles_r, show_needles=show_needles,
        frame=frame, limits=limits,
    )
    print(f"[paired-conet] real samples placed into the landscape frame: "
          f"{len(comp_r)} samples, k={k} ({time.time() - t0:.0f}s)")

    # Real samples are interpolated between uniform landmarks, so they normally sit
    # well inside the uniform extent; the purity warp can still push a mixture out.
    # Report rather than silently clip, since a clipped point is a lost observation.
    (x0, x1), (y0, y1) = limits
    E = Mr["E"]
    out = int(np.sum((E[:, 0] < x0) | (E[:, 0] > x1) |
                     (E[:, 1] < y0) | (E[:, 1] > y1)))
    if out:
        print(f"[paired-conet] WARNING: {out}/{len(E)} real samples fall outside the "
              "landscape frame's limits and are clipped.")

    Fr = p10.conet_dominance_fields(Mr, limits=limits)
    out_r = p10.save_png(
        Mr, Fr, bounds,
        f"{title_r} · placed in the landscape's uniform UMAP frame (k={k})",
        str(conet_path))
    return Path(out_r), Path(uniform_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="run directory with ensemble_config.json + points.csv")
    ap.add_argument("--save", default=None, metavar="PNG",
                    help="output path for the real-data CoNet (default: <run>/conet.png)")
    ap.add_argument("--save-uniform", default=None, metavar="PNG",
                    help="output path for the uniform baseline (default: <run>/conet_uniform.png)")
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help="uniform landmarks each real sample is interpolated from "
                         "(default: %(default)s)")
    ap.add_argument("--uniform-n", type=int, default=DEFAULT_UNIFORM_N,
                    help="uniform landmark count (default: %(default)s). FIXED, not "
                         "matched to the run: the baseline must depend only on the "
                         "landscape so runs sharing one are comparable.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the uniform draw (default: %(default)s)")
    ap.add_argument("--frame-cache", default=str(DEFAULT_FRAME_CACHE),
                    help="directory of cached per-landscape frames (default: %(default)s)")
    ap.add_argument("--no-frame-cache", action="store_true",
                    help="always refit the frame instead of reusing a cached one")
    ap.add_argument("--no-needles", action="store_true",
                    help="do not draw the run's discovered-needle stars")
    ap.add_argument("--purity", type=float, default=None,
                    help="CoNet purity layout threshold (default: plot_10d's CN_PURITY_THR)")
    ap.add_argument("--spread", type=float, default=None,
                    help="UMAP min_dist / point spread (default: plot_10d's CN_UMAP_MD)")
    ap.add_argument("--gap-reach", type=float, default=None,
                    help="far-satellite squash factor (default: plot_10d's CN_GAP_REACH)")
    args = ap.parse_args()

    out_r, out_u = render_paired_conets(
        Path(args.run).resolve(), k=args.k, uniform_n=args.uniform_n, seed=args.seed,
        show_needles=not args.no_needles, purity=args.purity, spread=args.spread,
        gap_reach=args.gap_reach, conet_path=args.save, uniform_path=args.save_uniform,
        frame_cache=None if args.no_frame_cache else Path(args.frame_cache),
    )
    print(f"wrote {out_r}")
    print(f"wrote {out_u}")


if __name__ == "__main__":
    main()
