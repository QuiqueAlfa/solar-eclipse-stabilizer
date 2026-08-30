from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VISIBLE_VERSION = 2
# Phase 3: the contrast fallback became a continuously-anchored robust history
# (``raw_center - local_contrast_center``) instead of a one-time fixed offset.
# v4: ``contrast_dynamic_offset`` stores the robust recent offset applied on
# every frame once history exists (plus ``fallback_supported`` / raw-sample
# diagnostics); blank/no-history frames no longer emit a finite center.
# v5: the fallback re-anchors only on accepted *trusted* geometry
# (``reanchor_mask``: limbo evidence before the horizon latch, never propagated
# SRC_NONE frames nor post-horizon false circles).  Without a mask the legacy
# re-anchor rule (any ``geometry_source != SRC_NONE``) is kept for old dicts.
CONTRAST_VERSION = 5
CLIP_VERSION = 3
TEAR_VERSION = 1
# Phase 5: the analysis gained an exposed-photosphere measurement and temporal
# content-loss arbitration. The new per-frame evidence is not reconstructible
# from a v4 cache without decoding the source again.
# v6 removes the former profile trajectory repairs and their ``timed_repaired``
# output. Invalidating v5 prevents that obsolete array from surviving when an
# older no-profile cache is loaded and saved again.
CACHE_VERSION = 6

# Per-frame geometric measurement source: how ``raw_center`` was obtained.
#   SRC_NONE            - no image evidence this frame; only the temporal
#                         prediction is propagated.  A propagated prediction is
#                         never labelled a measurement (``raw_center`` is NaN).
#   SRC_RADIAL          - accepted outer-arc radial measurement (primary source).
#   SRC_RADIAL_DISAGREE - radial measurement accepted, but the binary contour
#                         candidate disagreed beyond the tolerance; the radial
#                         evidence outranks it (auditable disagreement).
#   SRC_CONTOUR         - bootstrap / recovery: the contour candidate was used
#                         after being confirmed by real radial outer-edge
#                         evidence at its proposed center plus temporal/phase
#                         agreement when history exists.
SRC_NONE = 0
SRC_RADIAL = 1
SRC_RADIAL_DISAGREE = 2
SRC_CONTOUR = 3
SRC_NAMES = {
    SRC_NONE: "ninguna",
    SRC_RADIAL: "radial",
    SRC_RADIAL_DISAGREE: "radial_desacuerdo",
    SRC_CONTOUR: "contorno",
}

# Content states: whether the visible solar content is intact within the canvas.
CONTENT_USABLE = 0
CONTENT_CLIPPED = 1
CONTENT_UNCERTAIN = 2
# Centering states: how the exported center was obtained.
CENTER_RELIABLE = 0
CENTER_RECONSTRUCTED = 1
CENTER_UNRESOLVED = 2

MIN_RADIAL_POINTS = 35  # count of radial samples; not scaled with width
# Photometric floor for "the Sun is visible" decisions (~3% of 8-bit range).
# It is a brightness value, not a spatial distance: reviewed but never
# multiplied by the analysis-width scale.
MIN_BRIGHTNESS = 8.0
EDGE_BITS = {"left": 1, "right": 2, "top": 4, "bottom": 8}
EDGE_NAMES = {1: "left", 2: "right", 4: "top", 8: "bottom"}
CONTENT_NAMES = {CONTENT_USABLE: "usable", CONTENT_CLIPPED: "clipped", CONTENT_UNCERTAIN: "uncertain"}
CENTER_NAMES = {
    CENTER_RELIABLE: "reliable",
    CENTER_RECONSTRUCTED: "reconstructed",
    CENTER_UNRESOLVED: "unresolved",
}
# Tracking regimes: how the per-frame solar reference is obtained.
REGIME_LIMBO = 0
REGIME_TRANSIENT = 1
REGIME_HORIZON = 2
REGIME_NAMES = {
    REGIME_LIMBO: "limbo",
    REGIME_TRANSIENT: "respaldo_transitorio",
    REGIME_HORIZON: "horizonte_confirmado",
}
