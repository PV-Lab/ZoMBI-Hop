"""
visualization/lag_compare.py
============================
Compare the **lag-0** and **lag-k** pairings of a run's composition log, in any
dimensionality (the ternary views in ``discrepancy.py`` only work for d=3).

Background
----------
``composition_log.jsonl`` pairs, per objective call, the line the optimiser
*sent* with the compositions the apparatus *measured* in response. On run_39af
that lag-0 pairing is far worse than pairing each measured line with the line
sent **two calls earlier**: the readback appears to be two calls stale. This
script draws that comparison so it can be judged by eye instead of by summary
statistics.

Two figures
-----------
``--summary`` (default)  Four panels over the whole run:
  1. lag sweep - mean point-to-point and centroid distance as a function of the
     lag applied to the pairing; the minimum is the implied staleness;
  2. per-line best lag - histogram of the lag that minimises each line's own
     mean distance (a single spike = a systematic offset, a spread = noise);
  3. distance distributions - ECDF of every point's |measured - sent| under the
     lag-0 and lag-k pairings;
  4. distance vs. position along the line, both pairings, averaged over lines.

``--lines`` (gallery, one row per line, three panels each):
  * left   - measured line vs. the line sent on the *same* call (lag 0);
  * middle - measured line vs. the line sent ``k`` calls earlier (lag k);
    both projected into the sent line's own frame (x along the sent line, y the
    dominant orthogonal offset), with a connector per matched point pair;
  * right  - per-point distance along the line under both pairings.

Lines are picked by ``--pick``: ``worst`` (largest lag-0 error), ``best``,
``random`` or an explicit comma-separated list of call numbers.

Usage
-----
  # repo-root .venv
  .venv/Scripts/python visualization/lag_compare.py --run runs/run_39af
  .venv/Scripts/python visualization/lag_compare.py --run runs/run_39af --lines --n 6
  .venv/Scripts/python visualization/lag_compare.py --run runs/run_39af --lines --pick 3,56,71
  .venv/Scripts/python visualization/lag_compare.py --run runs/run_39af --lag 3

Flags
-----
  --run PATH      Run directory, or bare run name resolved under ``runs/``.
  --comp-log PATH Composition log (or its directory) to read directly.
  --lag K         Lag to compare against lag 0 (default: the sweep minimum).
  --rail NAME     Rail to use (default ``main``; ``cache`` is empty on 39af).
  --summary       Draw the four-panel run summary (default when neither given).
  --lines         Draw the per-line gallery.
  --n N           Number of lines in the gallery (default 5).
  --pick WHAT     ``worst`` (default), ``best``, ``random``, or ``3,56,71``.
  --seed S        Seed for ``--pick random``.
  --out PATH      Output image. Default: ``<run>/lag_summary.png`` /
                  ``<run>/lag_lines.png``.
  --show          Show the figure interactively instead of only saving it.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
RUNS_DIR = _HERE.parent / "runs"
COMP_LOG = "composition_log.jsonl"

# Lags scanned by the sweep. Negative = the measured line is compared against a
# line sent *earlier* (a stale readback); positive would mean the measured data
# ran ahead of the request, which is unphysical but is plotted as a control.
SWEEP = list(range(-8, 4))

C_SENT0 = "#888888"   # lag-0 sent line
C_SENTK = "#1f77b4"   # lag-k sent line
C_MEAS = "#d62728"    # measured line


# -- loading -------------------------------------------------------------------

def resolve_comp_log(target: str | Path) -> Path:
    """Accept a log file, a run directory, or a bare run name under runs/."""
    p = Path(target)
    if p.is_file():
        return p
    for cand in (p, RUNS_DIR / p.name):
        if (cand / COMP_LOG).is_file():
            return cand / COMP_LOG
    raise SystemExit(f"No {COMP_LOG} found for '{target}'")


class Line:
    """One rail of one objective call: the sent line and what came back."""

    __slots__ = ("call", "sent", "meas", "y", "endpoints")

    def __init__(self, call, sent, meas, y, endpoints):
        self.call = call
        self.sent = sent
        self.meas = meas
        self.y = y
        self.endpoints = endpoints

    def __len__(self):
        return len(self.sent)


def load_lines(log_path: Path, rail_name: str = "main") -> tuple[list[Line], list[int]]:
    """Read the log into per-call Line objects (empty rails skipped)."""
    lines: list[Line] = []
    dims: list[int] = []
    for raw in log_path.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        dims = dims or list(rec.get("optimizing_dims", []))
        for rail in rec.get("rails", []):
            if rail.get("name") != rail_name or not rail.get("sent"):
                continue
            sent = np.asarray(rail["sent"], float)
            meas = np.asarray(rail["measured"], float)
            if sent.shape != meas.shape:
                continue
            ep = rail.get("sent_endpoints")
            ep = np.asarray(ep, float) if ep and ep[0] is not None else None
            lines.append(Line(int(rec.get("call", len(lines))), sent, meas,
                              np.asarray(rail.get("y", []), float).ravel(), ep))
    return lines, dims


# -- pairing metrics -----------------------------------------------------------

def pair(lines: list[Line], i: int, lag: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Measured line `i` against the line sent at index ``i + lag``.

    The lag is signed the way the plots label it: ``-2`` means the request that
    went out two calls *earlier* (a stale readback). Pairs are truncated to the
    shorter of the two - dedup makes lines variable-length - and matched
    point-by-point in send order.
    """
    j = i + lag
    if not 0 <= j < len(lines):
        return None
    meas, sent = lines[i].meas, lines[j].sent
    q = min(len(meas), len(sent))
    if q == 0:
        return None
    return meas[:q], sent[:q]


def dists(lines: list[Line], i: int, lag: int) -> np.ndarray:
    p = pair(lines, i, lag)
    if p is None:
        return np.empty(0)
    return np.linalg.norm(p[0] - p[1], axis=1)


def sweep_stats(lines: list[Line]) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Mean point-to-point and centroid distance for each lag in SWEEP."""
    pp, cc = [], []
    for lag in SWEEP:
        d, c = [], []
        for i in range(len(lines)):
            p = pair(lines, i, lag)
            if p is None:
                continue
            m, s = p
            d.append(np.linalg.norm(m - s, axis=1).mean())
            c.append(np.linalg.norm(m.mean(0) - s.mean(0)))
        pp.append(np.mean(d) if d else np.nan)
        cc.append(np.mean(c) if c else np.nan)
    return SWEEP, np.asarray(pp), np.asarray(cc)


def best_lags(lines: list[Line]) -> list[int]:
    """The lag minimising each line's own mean distance."""
    out = []
    for i in range(len(lines)):
        scored = [(float(dists(lines, i, lag).mean()), lag)
                  for lag in SWEEP if pair(lines, i, lag) is not None]
        if scored:
            out.append(min(scored)[1])
    return out


def auto_lag(lines: list[Line]) -> int:
    """The non-zero lag with the lowest mean point-to-point distance."""
    lags, pp, _ = sweep_stats(lines)
    order = np.argsort(np.where(np.isnan(pp), np.inf, pp))
    for k in order:
        if lags[k] != 0:
            return lags[k]
    return 0


# -- summary figure ------------------------------------------------------------

def _ecdf(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.sort(v)
    return v, np.arange(1, len(v) + 1) / len(v)


def figure_summary(lines: list[Line], lag: int, source: str, dims: list[int]) -> plt.Figure:
    k = abs(lag)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    lags, pp, cc = sweep_stats(lines)

    ax = axes[0, 0]
    ax.plot(lags, pp, "o-", color=C_MEAS, label="mean point-to-point")
    ax.plot(lags, cc, "s--", color=C_SENTK, label="centroid distance")
    lo = int(np.nanargmin(pp))
    ax.axvline(lags[lo], color="0.4", lw=1, ls=":")
    ax.annotate(f"min at lag {lags[lo]}\n{pp[lo]:.3f}",
                (lags[lo], pp[lo]), textcoords="offset points", xytext=(12, 26),
                fontsize=9, color="0.2",
                arrowprops=dict(arrowstyle="-", color="0.6", lw=0.8))
    ax.scatter([0], [pp[lags.index(0)]], s=90, facecolor="none",
               edgecolor="k", zorder=5, label="as logged (lag 0)")
    ax.set_xlabel("lag applied to pairing (calls; negative = stale readback)")
    ax.set_ylabel("mean |measured - sent|")
    ax.set_title("Lag sweep")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    bl = best_lags(lines)
    vals, counts = np.unique(bl, return_counts=True)
    ax.bar(vals, counts, color=[C_MEAS if v == -k else "0.65" for v in vals])
    ax.set_xlabel("lag minimising that line's own error")
    ax.set_ylabel("number of lines")
    ax.set_title(f"Per-line best lag  (n = {len(bl)} lines)")
    for v, c in zip(vals, counts):
        ax.text(v, c, f"{c}", ha="center", va="bottom", fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    d0 = np.concatenate([dists(lines, i, 0) for i in range(len(lines))])
    dk = np.concatenate([dists(lines, i, -k) for i in range(len(lines))])
    for d, c, lb in ((d0, C_SENT0, f"lag 0   mean {d0.mean():.3f}"),
                     (dk, C_MEAS, f"lag -{k}  mean {dk.mean():.3f}")):
        x, y = _ecdf(d)
        ax.plot(x, y, color=c, lw=2, label=lb)
    ax.set_xlabel("|measured - sent| per point")
    ax.set_ylabel("fraction of points")
    ax.set_title(f"Point-error distribution ({len(d0)} vs {len(dk)} points)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for lg, c, lb in ((0, C_SENT0, "lag 0"), (-k, C_MEAS, f"lag -{k}")):
        per_idx: dict[int, list[float]] = {}
        for i in range(len(lines)):
            for j, v in enumerate(dists(lines, i, lg)):
                per_idx.setdefault(j, []).append(float(v))
        ks = sorted(per_idx)
        ax.plot(ks, [np.mean(per_idx[j]) for j in ks], "o-", color=c, label=lb, ms=4)
    ax.set_xlabel("point index along the line")
    ax.set_ylabel("mean |measured - sent|")
    ax.set_title("Error along the line")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(f"{source} - lag 0 vs lag {-k} pairing "
                 f"(d = {len(dims)}: dims {dims})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# -- per-line gallery ----------------------------------------------------------

def _line_frame(meas: np.ndarray, sent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project both point sets into the sent line's own 2-D frame.

    A plain PCA of the stacked sets puts its first axis along the *offset*
    between them, which squashes the sent line to a dot whenever the two are far
    apart. Instead: x runs along the sent line's principal direction, y along the
    dominant component of the residual that is orthogonal to it. The sent line is
    then always spread along x and the discrepancy is legible on y.
    """
    origin = sent.mean(0)
    cs = sent - origin
    u1 = np.linalg.svd(cs, full_matrices=False)[2][0] if len(sent) > 1 else None
    if u1 is None or not np.isfinite(u1).all():
        u1 = np.eye(sent.shape[1])[0]
    res = (meas - sent)
    res = res - np.outer(res @ u1, u1)
    if len(res) > 1 and np.linalg.norm(res) > 1e-12:
        u2 = np.linalg.svd(res, full_matrices=False)[2][0]
    else:  # degenerate: any direction orthogonal to u1 will do
        seed = np.eye(sent.shape[1])[1]
        u2 = seed - (seed @ u1) * u1
    u2 = u2 - (u2 @ u1) * u1
    n2 = np.linalg.norm(u2)
    u2 = u2 / n2 if n2 > 1e-12 else np.zeros_like(u1)
    basis = np.column_stack([u1, u2])
    return (meas - origin) @ basis, (sent - origin) @ basis


def _panel(ax, meas, sent, title, sent_color, sent_label):
    """One projected sent-vs-measured comparison with per-pair connectors."""
    m2, s2 = _line_frame(meas, sent)
    for a, b in zip(s2, m2):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="0.75", lw=0.8, zorder=1)
    ax.plot(s2[:, 0], s2[:, 1], "-o", color=sent_color, ms=4, lw=1.4,
            label=sent_label, zorder=2)
    ax.plot(m2[:, 0], m2[:, 1], "-o", color=C_MEAS, ms=5, lw=1.4,
            label="measured", zorder=3)
    ax.scatter(s2[:1, 0], s2[:1, 1], s=110, facecolor="none",
               edgecolor=sent_color, zorder=4)
    ax.scatter(m2[:1, 0], m2[:1, 1], s=110, facecolor="none",
               edgecolor=C_MEAS, zorder=4)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("along sent line", fontsize=8)
    ax.set_ylabel("orthogonal offset", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")


def pick_indices(lines: list[Line], how: str, n: int, seed: int) -> list[int]:
    """Resolve --pick into line indices."""
    if how not in ("worst", "best", "random"):
        wanted = {int(t) for t in how.replace(" ", "").split(",") if t}
        idx = [i for i, ln in enumerate(lines) if ln.call in wanted]
        if not idx:
            raise SystemExit(f"No lines with call number(s) {sorted(wanted)}")
        return idx
    if how == "random":
        return sorted(random.Random(seed).sample(range(len(lines)), min(n, len(lines))))
    score = [(float(dists(lines, i, 0).mean()), i) for i in range(len(lines))]
    score.sort(reverse=(how == "worst"))
    return [i for _, i in score[:n]]


def figure_lines(lines: list[Line], idx: list[int], lag: int, source: str) -> plt.Figure:
    k = abs(lag)
    rows = len(idx)
    fig, axes = plt.subplots(rows, 3, figsize=(14, 3.6 * rows), squeeze=False)
    for r, i in enumerate(idx):
        ln = lines[i]
        p0, pk = pair(lines, i, 0), pair(lines, i, -k)
        ax0, axk, axd = axes[r]

        if p0 is not None:
            m, s = p0
            _panel(ax0, m, s,
                   f"call {ln.call}  |  lag 0  |  mean "
                   f"{np.linalg.norm(m - s, axis=1).mean():.3f}",
                   C_SENT0, f"sent (call {ln.call})")
        else:
            ax0.set_axis_off()

        if pk is not None:
            m, s = pk
            _panel(axk, m, s,
                   f"call {ln.call}  |  lag -{k}  |  mean "
                   f"{np.linalg.norm(m - s, axis=1).mean():.3f}",
                   C_SENTK, f"sent (call {lines[i - k].call})")
        else:
            axk.set_axis_off()
            axk.text(0.5, 0.5, f"no call {k} steps back", ha="center",
                     va="center", fontsize=9, transform=axk.transAxes)

        d0, dk = dists(lines, i, 0), dists(lines, i, -k)
        if d0.size:
            axd.plot(d0, "-o", color=C_SENT0, ms=4, label=f"lag 0  ({d0.mean():.3f})")
        if dk.size:
            axd.plot(dk, "-o", color=C_SENTK, ms=4, label=f"lag -{k}  ({dk.mean():.3f})")
        axd.set_xlabel("point index")
        axd.set_ylabel("|measured - sent|")
        axd.set_ylim(bottom=0)
        axd.set_title("per-point error", fontsize=10)
        axd.legend(fontsize=8)
        axd.grid(alpha=0.3)

    fig.suptitle(f"{source} - measured line vs. its own request (left) and vs. the "
                 f"request {k} calls earlier (middle).\nProjected into the sent "
                 f"line's own frame (x along that line, y = orthogonal offset); grey "
                 f"links join matched points; open circles mark each line's first "
                 f"point.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# -- cli -----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare the lag-0 and lag-k sent/measured pairings of a run.")
    ap.add_argument("--run", help="run directory, or a bare run name under runs/")
    ap.add_argument("--comp-log", help="composition_log.jsonl (or its directory)")
    ap.add_argument("--lag", type=int, default=None,
                    help="lag to compare against 0 (default: sweep minimum)")
    ap.add_argument("--rail", default="main")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--lines", action="store_true")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--pick", default="worst")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if not (args.run or args.comp_log):
        ap.error("give --run or --comp-log")
    if not args.show:
        matplotlib.use("Agg")

    log_path = resolve_comp_log(args.comp_log or args.run)
    run_dir = log_path.parent
    lines, dims = load_lines(log_path, args.rail)
    if len(lines) < 2:
        raise SystemExit(f"{log_path}: only {len(lines)} usable '{args.rail}' lines")

    lag = abs(args.lag) if args.lag is not None else abs(auto_lag(lines))
    source = f"{run_dir.name}/{log_path.name}"
    print(f"{source}: {len(lines)} '{args.rail}' lines, d={len(dims)}, "
          f"comparing lag 0 vs lag -{lag}")

    if args.summary or not args.lines:
        fig = figure_summary(lines, lag, source, dims)
        out = (Path(args.out) if (args.out and not args.lines)
               else run_dir / "lag_summary.png")
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")
    if args.lines:
        idx = pick_indices(lines, args.pick, args.n, args.seed)
        fig = figure_lines(lines, idx, lag, source)
        out = Path(args.out) if args.out else run_dir / "lag_lines.png"
        fig.savefig(out, dpi=150)
        print(f"wrote {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    sys.exit(main())
