"""Store event kinds of the lanes engine (frozen to the sandbox strings; DESIGN.md N5).

Every ``engine.events`` record carries ``kind`` = one of these strings. They are FROZEN: on the emulator
path ``lane_system.ZoMBILaneSystem.drain`` / ``_convert`` (LS 154-221) forward every store event to the
harness log, and ``explore/compare_logs.py`` compares that section by (kind, lane, act, zoom, t), so the
port must emit exactly the sandbox kinds with the sandbox payloads.

Rules for the engine code (REVIEW F1):
* The method events of the explore framework were named ``x_<method>_<kind>`` by ``ctx.event``
  (FW 619-624), which also filled ``lane`` / ``act`` / ``zoom`` from ``store.active`` when the call did not
  pass them. The ported lane passes ``lane=self.name, act=self.act, zoom=self.zoom`` explicitly on
  ``EV_ZB_BIND`` / ``EV_ZB_FALLBACK`` (the computing lane is ``store.active`` there) and the catch-up
  events keep their explicit fields (CU 214, 249).
* No new kind may be added on the emulator path: commit bookkeeping, needles.db re-syncs, timers and the
  like are NOT store events (they would reach the compared harness log).

Kinds of the sandbox that the port never emits (switches not ported, DESIGN N4) are listed in
``NOT_PORTED_KINDS`` for tests and readers.
"""
from __future__ import annotations

# ── lane / store events (sandbox lane_zombihop.py) ────────────────────────────────────────────────
EV_ACTIVATION = "activation"                  # LZ 653  a lane starts a new activation (dive)
EV_ZOOM_IN = "zoom_in"                        # LZ 1039 a lane zoomed in (from_zoom -> to_zoom)
EV_ZOOM_REFUSED = "zoom_refused"              # LZ 1017 nested_box refused (floor blocks the zoom)
EV_BOX_OVERLAPS_OTHER = "box_overlaps_other"  # LZ 983  new box overlaps the other lane's active box
EV_NEEDLE = "needle"                          # LZ 1117 needle declared
EV_NEEDLE_HOLD = "needle_hold"                # LZ 1084, 1091 needle try held (no_points / not_repeatable)
EV_OVERPEN_SHRINK = "overpen_shrink"          # LZ 1256 over-penalisation shrink
EV_ZONE = "zone"                              # LZ 1273 exclusion zone at a dive end
EV_DIVE_END = "dive_end"                      # LZ 1279 dive ended without a zone
EV_EXTEND = "extend"                          # LZ 1313 / ZB 329 one more line at a zoom (ext_cap)

# ── method events (explore framework names x_<method>_<kind>, FW 623) ─────────────────────────────
EV_ZB_BIND = "x_zbudget_bind"                 # ZB 238  per-zoom rule differs from the stock rule
EV_ZB_FALLBACK = "x_zbudget_fallback"         # ZB 341, 345 extensions exhausted below the raised min
EV_CU_HOLD = "x_catchup_hold"                 # CU 214  catch-up holds a zoom exit
EV_CU_RELEASE = "x_catchup_release"           # CU 249  catch-up releases a held transition


def method_event(method_id: str, kind: str) -> str:
    """The explore framework's event kind for a method event (FW 623): ``x_<method>_<kind>``."""
    return f"x_{method_id}_{kind}"


# Every kind the ported engine may emit.
STORE_EVENT_KINDS = frozenset({
    EV_ACTIVATION, EV_ZOOM_IN, EV_ZOOM_REFUSED, EV_BOX_OVERLAPS_OTHER, EV_NEEDLE, EV_NEEDLE_HOLD,
    EV_OVERPEN_SHRINK, EV_ZONE, EV_DIVE_END, EV_EXTEND,
    EV_ZB_BIND, EV_ZB_FALLBACK, EV_CU_HOLD, EV_CU_RELEASE,
})

# Sandbox kinds of switches that are not ported (bridge, confirm_revise, gate; LZ 830, 1052-1070,
# 1146-1244, 1422). The port never emits them.
NOT_PORTED_KINDS = frozenset({"needle_final", "needle_refine", "box_confirm", "gate", "bridge"})

# The kinds ``ZoMBILaneSystem.finish`` keeps in its ``events`` summary (LS 130-131), for reference.
LS_FINISH_KEEP = ("zoom_in", "box_confirm", "needle", "needle_refine", "needle_final", "zone", "dive_end",
                  "gate", "extend", "zoom_refused", "box_overlaps_other", "overpen_shrink")

__all__ = ["EV_ACTIVATION", "EV_ZOOM_IN", "EV_ZOOM_REFUSED", "EV_BOX_OVERLAPS_OTHER", "EV_NEEDLE", "EV_NEEDLE_HOLD",
           "EV_OVERPEN_SHRINK", "EV_ZONE", "EV_DIVE_END", "EV_EXTEND", "EV_ZB_BIND", "EV_ZB_FALLBACK",
           "EV_CU_HOLD", "EV_CU_RELEASE", "method_event", "STORE_EVENT_KINDS", "NOT_PORTED_KINDS",
           "LS_FINISH_KEEP"]
