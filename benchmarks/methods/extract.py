"""
benchmarks/methods/extract.py
=============================
Turning a method's samples into a declared set of optima ("needles").

The sweep's headline metric, ``dist_to_needles``, scores a *set* of declared optima
against the landscape's true ones — one-to-one, with every unmatched member of the
larger set charged the full penalty, so declaring too many costs exactly as much as
declaring too few. ZoMBI-Hop declares needles as part of its algorithm. Random
search, BO, TuRBO and HEBO do not: they return samples, and something has to read
optima out of those samples before they can be scored.

That something is an **extractor**: ``extractor(X, Y) -> (needles, records)``. It
sees only what the method measured — no landscape knowledge, no true count, no plain
level — so it is a post-processing step any practitioner could apply to a finished
campaign. The runner applies the SAME extractor to every method's samples, the
declaring ones included (``dist_to_needles_extracted``), so the benchmark always has
one comparison in which the optimisers differ only in where they chose to sample.

Two are provided; add more with :func:`register_extractor`.

``gp_peaks`` (default)
    Fit one exact GP (Matern-5/2 ARD) to all samples; run projected gradient ascent
    on its posterior mean from the best-predicted samples; merge maxima closer than
    ``merge_radius``; keep a maximum only if the model is confident it stands out
    from its surroundings:

    * the surroundings are ``n_probes`` points on an ellipsoidal shell
      ``shell[0]..shell[1]`` of the GP's own ARD lengthscales around the peak —
      where the model says the surface has decorrelated from it, which adapts to
      sharp vs broad basins and to dimension without being told either;
    * only probes where the posterior sd has fallen to ``confident_sd_frac`` of
      the prior sd count, and at least ``min_confident_probes`` must (else the
      peak is rejected as unseen: nothing measured tells you it is a peak);
    * the peak's lower bound ``mu - sd`` must clear the MEDIAN posterior mean of
      the confident probes by ``prominence_z`` noise sds (the GP's fitted noise,
      floored at the instrument's known noise ``noise_frac * median|Y|``).
      1.5 sds on a lower bound is a little stricter than the sweep landscape's own
      resolvability rule (adjacent needles are guaranteed a saddle >= 1 sigma_y).

    Calibrated with ``python -m benchmarks.methods calibrate`` (uniform and
    budget-concentrated sampling, dims 3-10, 48 to 3000 samples, landscapes with
    and without needles; two seeds, job 24016311): zero false positives on every
    needle-free landscape, never declared a decoy cluster of samples on the
    plain, precision 1.00 everywhere except thirty overlapping b=2.2 basins in
    3-d (0.67-0.80), and 9-10/10 on ten b=6 needles from 3000 uniform samples in
    3-d. It is conservative: at 48-96 samples it declares nothing.

    Two alternatives were tried and rejected, and they are why it looks like this.
    Measuring the dip to the *minimum* of the posterior mean at nearby samples is
    an extreme-value statistic — over ~50 samples of a noisy plain, max-minus-min
    routinely clears 2 noise sds, and on a 96-point smoke run it declared 11
    needles on a 3-needle landscape, 9 of them noise bumps on the plain. A fixed
    probe radius (0.25-0.5) was the wrong scale for sharp needles and for 10-d,
    where it found nothing at all.

    Known blind spot, by design: a needle hit by one or two samples and never
    revisited (sharp basins under uniform sampling) is indistinguishable from a
    noise spike, the GP treats it as one, and it is not declared. ZoMBI-Hop's own
    repeatability gate refuses the same case for the same reason.

``nms``
    No model: rank samples by measured ``Y`` and keep a sample if it clears the
    MEDIAN measurement within ``prominence_radius`` by ``prominence_z`` noise sds
    and no kept sample lies within ``merge_radius``. The noise sd is estimated
    from nearest-neighbour differences. Cheap, and a check on how much of a
    method's score is the GP's doing — but its flank level is taken at the
    samples, so it is biased against methods that sample a peak's top densely.

Defaults are tied to the landscape's physical scales, not to its answer:
``merge_radius`` is half the needle-separation floor (``sigma_x / 2 = 0.064``; two
declared needles closer than that cannot be told apart by the apparatus), and
``prominence_z = 3`` is a three-sigma margin.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Protocol

import numpy as np

from ._gp import fit_matern_gp, posterior_mean_sd, tensor

#: ``run_mobo.NOISE_LEVEL``, the needle-separation floor of the sweep landscapes.
SIGMA_X = 0.128


class Extractor(Protocol):
    name: str

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, list[dict]]: ...

    def config(self) -> dict: ...


def _noise_floor(Y: np.ndarray, noise_frac: float | None) -> float:
    """The instrument's known noise sd, ``noise_frac * median |Y|`` (0 if unknown).

    A floor under the fitted noise, not a replacement: with a few dozen samples a
    GP happily explains everything as signal and fits its noise near zero, and
    every wobble then clears "1.5 noise sds". A 48-point run declared 8 needles on
    a 2-needle landscape that way. The measurement noise is a property of the
    apparatus (``run_mobo.OUTPUT_NOISE_FRAC``, measured), not of the landscape, so
    using it leaks nothing about where the needles are.
    """
    if not noise_frac:
        return 0.0
    return float(noise_frac) * float(np.median(np.abs(Y)))


def _greedy_separated(points: np.ndarray, order: np.ndarray, radius: float,
                      limit: int | None = None) -> list[int]:
    """Indices, taken in ``order``, each at least ``radius`` from every one kept."""
    kept: list[int] = []
    for i in order:
        if limit is not None and len(kept) >= limit:
            break
        if not kept or np.min(np.linalg.norm(points[kept] - points[i], axis=1)) >= radius:
            kept.append(int(i))
    return kept


@dataclass
class GPPeakExtractor:
    """Local maxima of a GP posterior mean that the model is confident stand out."""

    merge_radius: float = SIGMA_X / 2
    shell: tuple[float, float] = (1.0, 2.0)
    prominence_z: float = 1.5
    n_probes: int = 256
    min_confident_probes: int = 8
    confident_sd_frac: float = 0.7
    n_starts: int = 128
    ascent_steps: int = 150
    lr: float = 0.01
    fit_max_points: int = 3000
    max_needles: int = 200
    min_points: int = 10
    noise_frac: float | None = None
    device: str = "cpu"
    seed: int = 0

    name = "gp_peaks"

    def config(self) -> dict:
        return {"name": self.name, **asdict(self)}

    def _fit_subset(self, Y: np.ndarray) -> np.ndarray | None:
        """Rows to fit hyperparameters on: all, or the best half plus a random half.

        The best half keeps the peaks' shape (what the lengthscales must describe);
        the random half keeps the plain, so the fitted noise is not just the noise
        on top of the peaks. Conditioning afterwards still uses every sample.
        """
        n = len(Y)
        if n <= self.fit_max_points:
            return None
        rng = np.random.default_rng(self.seed)
        half = self.fit_max_points // 2
        top = np.argsort(-Y)[:half]
        rest = np.setdiff1d(np.arange(n), top)
        return np.sort(np.concatenate(
            [top, rng.choice(rest, size=self.fit_max_points - half, replace=False)]))

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        import torch

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).reshape(-1)
        d = X.shape[1] if X.ndim == 2 else 0
        if len(Y) < max(self.min_points, 2):
            return np.empty((0, d)), []

        model, y_mean, y_std = fit_matern_gp(X, Y, device=self.device,
                                             fit_idx=self._fit_subset(Y))
        noise_sd = max(float(model.likelihood.noise.detach().sqrt().reshape(-1)[0]),
                       _noise_floor(Y, self.noise_frac) / y_std)
        prior_sd = float(model.covar_module.outputscale.detach().sqrt().reshape(-1)[0])
        lengthscales = (model.covar_module.base_kernel.lengthscale.detach()
                        .cpu().numpy().reshape(-1))
        mu_obs, _ = posterior_mean_sd(model, X, device=self.device)

        # Starts: the best-predicted samples, spread out so no two starts sit on
        # the same bump (they would only climb to the same maximum).
        starts = _greedy_separated(X, np.argsort(-mu_obs), self.merge_radius,
                                   limit=self.n_starts)
        P = tensor(X[starts], self.device).clone().requires_grad_(True)
        opt = torch.optim.Adam([P], lr=self.lr)
        for _ in range(self.ascent_steps):
            opt.zero_grad()
            loss = -model.posterior(P).mean.sum()
            loss.backward()
            opt.step()
            with torch.no_grad():
                P.clamp_(0.0, 1.0)
        peaks = P.detach()
        mu_p, sd_p = posterior_mean_sd(model, peaks, device=self.device)
        peaks_np = peaks.cpu().numpy()

        keep = _greedy_separated(peaks_np, np.argsort(-mu_p), self.merge_radius)
        rng = np.random.default_rng(self.seed + 1)
        needles, records = [], []
        for i in keep:
            probes = self._shell(peaks_np[i], lengthscales, rng)
            mu_s, sd_s = posterior_mean_sd(model, probes, device=self.device)
            confident = sd_s <= self.confident_sd_frac * prior_sd
            n_conf = int(confident.sum())
            if n_conf < self.min_confident_probes:
                continue        # the model has not seen what surrounds this point
            baseline = float(np.median(mu_s[confident]))
            prom = float(mu_p[i] - sd_p[i] - baseline)
            if prom < self.prominence_z * noise_sd:
                continue
            needles.append(peaks_np[i])
            records.append({"value": round(float(mu_p[i] * y_std + y_mean), 6),
                            "prominence": round(prom * y_std, 6),
                            "posterior_sd": round(float(sd_p[i] * y_std), 6),
                            "n_confident_probes": n_conf})
            if len(needles) >= self.max_needles:
                break
        del model
        return (np.asarray(needles, dtype=float).reshape(-1, d), records)

    def _shell(self, center: np.ndarray, lengthscales: np.ndarray, rng) -> np.ndarray:
        """Probe points ``shell[0]..shell[1]`` lengthscales out from ``center``.

        The shell is an ellipsoid in the GP's own ARD lengthscales, so it sits
        where the model says the surface has decorrelated from the peak — close in
        for a sharp needle or a dense cluster of them, further out for a broad
        basin — without the extractor having to know the landscape's basin width.
        """
        dim = center.shape[0]
        v = rng.normal(size=(self.n_probes, dim))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        r = rng.uniform(self.shell[0], self.shell[1], size=(self.n_probes, 1))
        return np.clip(center[None, :] + r * v * lengthscales[None, :], 0.0, 1.0)


@dataclass
class NMSExtractor:
    """Non-maximum suppression on the raw measurements; no model."""

    merge_radius: float = SIGMA_X / 2
    prominence_radius: float = 0.5
    prominence_z: float = 3.0
    min_flank_points: int = 3
    max_needles: int = 200
    min_points: int = 10
    noise_frac: float | None = None

    name = "nms"

    def config(self) -> dict:
        return {"name": self.name, **asdict(self)}

    @staticmethod
    def noise_sd(X: np.ndarray, Y: np.ndarray) -> float:
        """Robust noise sd from nearest-neighbour differences (MAD / sqrt 2)."""
        from scipy.spatial import cKDTree

        _, nn = cKDTree(X).query(X, k=2)
        diff = np.abs(Y - Y[nn[:, 1]])
        return float(np.median(diff) / (0.6745 * np.sqrt(2.0)))

    def __call__(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).reshape(-1)
        d = X.shape[1] if X.ndim == 2 else 0
        if len(Y) < max(self.min_points, 2):
            return np.empty((0, d)), []
        sd = max(self.noise_sd(X, Y), _noise_floor(Y, self.noise_frac))
        needles, records = [], []
        for i in np.argsort(-Y):
            if needles and np.min(np.linalg.norm(np.asarray(needles) - X[i], axis=1)) \
                    < self.merge_radius:
                continue
            flank = np.linalg.norm(X - X[i], axis=1) <= self.prominence_radius
            if int(flank.sum()) < self.min_flank_points:
                continue
            prom = float(Y[i] - np.median(Y[flank]))
            if prom < self.prominence_z * sd:
                continue
            needles.append(X[i])
            records.append({"value": round(float(Y[i]), 6),
                            "prominence": round(prom, 6), "n_flank": int(flank.sum())})
            if len(needles) >= self.max_needles:
                break
        return np.asarray(needles, dtype=float).reshape(-1, d), records


EXTRACTORS: dict[str, Callable[..., Extractor]] = {
    "gp_peaks": GPPeakExtractor,
    "nms": NMSExtractor,
}


def register_extractor(name: str, factory: Callable[..., Extractor]) -> None:
    EXTRACTORS[name] = factory


def make_extractor(name: str = "gp_peaks", **kwargs) -> Extractor:
    """An extractor by name. ``device``/``seed``/``noise_frac`` are dropped for
    extractors that do not take them."""
    if name not in EXTRACTORS:
        raise KeyError(f"unknown extractor {name!r}; known: {sorted(EXTRACTORS)}")
    factory = EXTRACTORS[name]
    fields = getattr(factory, "__dataclass_fields__", None)
    if fields is not None:
        # Only the runner-supplied keys are optional; a typo in a user setting
        # still reaches the constructor and fails there.
        for k in ("device", "seed", "noise_frac"):
            if k not in fields:
                kwargs.pop(k, None)
    return factory(**kwargs)
