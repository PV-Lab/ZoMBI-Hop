"""Line records of the lanes engine.

``LineRec`` (one chosen line with its provenance and life cycle) and ``LaneLine`` (the line handed to a
controller / harness: endpoints + tags) are moved verbatim from the sandbox prototype
``benchmarks/zoom_fix/lanes/lane_zombihop.py`` LZ 187-233 (DESIGN.md section 1.2). Field names, order and
defaults are unchanged: the sandbox harness (``lane_system.py``), the auditor and the logs read them
(DESIGN.md 1.1). In the ported engine ``bridge`` is always False and ``eff_zoom == zoom`` (DESIGN N4,
1.3.2 item 11); the fields stay so the logs and the auditor keep their shape.

tests/test_lanes_support.py compares both class ASTs with the sandbox file when it is present.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


# ── lines (LZ 187-233 verbatim) ──────────────────────────────────────────────────────────────

@dataclass
class LineRec:
    line_id: int
    rid: str
    lane: str
    act: int
    zoom: int                 # zoom the line was chosen at (sent zoom)
    eff_zoom: int             # zoom it counts for (a bridge: k+1 once the lane zoomed in)
    bridge: bool
    rank: int                 # 0 main, 1 backup
    decision: int             # lane decision index
    kind: str                 # 'regular' | 'refresh'
    ends: np.ndarray          # (2, d) requested endpoints
    box: np.ndarray           # (2, d) the zoom box the line is for (bridge: the predicted box)
    line_box: np.ndarray      # (2, d) the inner box the endpoints were drawn in
    candidate: Optional[torch.Tensor]
    t_decide: float
    cand_src: str = "gp"      # gp | fallback_mean (get_candidate failed: GP-mean argmax) | none
    cand_fail: Optional[str] = None   # gp.last_candidate_failure when get_candidate returned None
    status: str = "queued"    # queued | picked | measured | replaced | lost
    t_pick: Optional[float] = None
    t_result: Optional[float] = None
    pkt: Optional[int] = None

    def tags(self) -> dict:
        return dict(lane=self.lane, act=self.act, zoom=self.zoom, line=self.line_id, rid=self.rid,
                    bridge=self.bridge, rank=self.rank, decision=self.decision, kind=self.kind)


class LaneLine:
    """A line handed to the harness: ``ends`` (2, d) [left; right] and its ``tags``."""
    __slots__ = ("ends", "tags")

    def __init__(self, rec: LineRec):
        self.ends = rec.ends.copy()
        self.tags = rec.tags()

    @property
    def rid(self) -> str:
        return self.tags["rid"]

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.ends, dtype=dtype)

    def __repr__(self):
        t = self.tags
        return f"LaneLine({t['rid']} a{t['act']} z{t['zoom']}{' bridge' if t['bridge'] else ''})"


__all__ = ["LineRec", "LaneLine"]
