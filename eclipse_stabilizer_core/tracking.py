from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .constants import (
    CENTER_NAMES,
    CENTER_RECONSTRUCTED,
    CENTER_RELIABLE,
    CENTER_UNRESOLVED,
    CONTENT_CLIPPED,
    CONTENT_NAMES,
    CONTENT_UNCERTAIN,
    CONTENT_USABLE,
    MIN_BRIGHTNESS,
    MIN_RADIAL_POINTS,
    REGIME_HORIZON,
    REGIME_LIMBO,
    REGIME_NAMES,
    REGIME_TRANSIENT,
    SRC_NAMES,
    SRC_NONE,
)
from .geometry import analysis_scale, bitmask_to_edges

# Arc coverage (degrees) below which the outer limb is too short to be a
# reliable circle; the dual signal is the angular gap that must not exceed
# ``360 - MIN_ARC_COVERAGE``.
MIN_ARC_COVERAGE = 150.0
# Fixed-radius scale drift tolerance, identical to ``warn_scale_drift``.
RADIUS_SCALE_TOLERANCE = 0.08


def transient_outliers(raw: np.ndarray, relative: np.ndarray, trusted: np.ndarray, gate: float = 4.0) -> np.ndarray:
    keep = trusted.copy()
    n = len(raw)
    for i in range(1, n - 1):
        if not trusted[i] or not trusted[i - 1] or not trusted[i + 1]:
            continue
        error_before = np.linalg.norm((raw[i] - raw[i - 1]) - relative[i])
        error_after = np.linalg.norm((raw[i + 1] - raw[i]) - relative[i + 1])
        neighbors = np.linalg.norm((raw[i + 1] - raw[i - 1]) - (relative[i] + relative[i + 1]))
        if error_before > gate and error_after > gate and neighbors < gate:
            keep[i] = False
    return keep


def solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    n = len(diag)
    c = upper.copy()
    d = rhs.copy()
    b = diag.copy()
    for i in range(1, n):
        pivot = b[i - 1] if abs(b[i - 1]) > 1e-12 else 1e-12
        factor = lower[i] / pivot
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]
    x = np.empty(n, np.float64)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def weighted_path(raw: np.ndarray, relative: np.ndarray, abs_weight: np.ndarray, rel_weight: np.ndarray) -> np.ndarray:
    n = len(raw)
    result = np.zeros((n, 2), np.float64)
    for axis in (0, 1):
        lower = np.zeros(n, np.float64)
        upper = np.zeros(n, np.float64)
        diag = abs_weight.copy() + 1e-9
        rhs = abs_weight * np.nan_to_num(raw[:, axis], nan=0.0)
        for i in range(1, n):
            weight = rel_weight[i]
            diag[i - 1] += weight
            diag[i] += weight
            lower[i] -= weight
            upper[i - 1] -= weight
            delta = relative[i, axis]
            rhs[i - 1] -= weight * delta
            rhs[i] += weight * delta
        result[:, axis] = solve_tridiagonal(lower, diag, upper, rhs)
    return result


def robust_path_solution(
    anchor_center: np.ndarray,
    relative: np.ndarray,
    response: np.ndarray,
    limb_anchor: np.ndarray,
    limb_quality: np.ndarray,
    contrast_anchor: np.ndarray,
    contrast_score: np.ndarray,
    scale: float,
    auto_repair: bool = True,
) -> dict:
    """Resolve the existing global weighted path with robust signal weights.

    This is deliberately a weighting pass around :func:`weighted_path`, not a
    second solver.  Absolute limb measurements are strongest, contrast remains
    an additional weak measurement, and phase motion closes gaps.  Innovations
    are evaluated both per-frame and cumulatively so many individually small
    errors cannot grow into an excursion.
    """
    n = len(anchor_center)
    positive = response[np.isfinite(response) & (response > 0)]
    normal = float(np.median(positive)) if len(positive) else 1.0
    relative_ok = np.isfinite(relative).all(axis=1)
    rel_strength = np.clip(response / max(normal, 1e-6), 0.0, 1.0)
    # Phase correlation is useful for gap shape and confirmed camera jumps, but
    # the evolving lunar edge can bias it.  Keep it materially weaker than a
    # coherent outer-limb anchor.
    rel_weight = 0.03 + 0.50 * rel_strength
    rel_weight[~relative_ok] = 0.0
    rel_weight[0] = 0.0

    abs_weight = np.zeros(n, np.float64)
    abs_weight[limb_anchor] = 6.0 + 18.0 * np.clip(limb_quality[limb_anchor], 0.0, 1.0)
    contrast_only = contrast_anchor & ~limb_anchor
    abs_weight[contrast_only] = 0.02 + 0.18 * np.clip(
        contrast_score[contrast_only], 0.0, 1.0
    )
    if not np.any(abs_weight > 0):
        raise SystemExit("No hay detecciones solares fiables para resolver la trayectoria.")

    raw_solution = weighted_path(anchor_center, relative, abs_weight, rel_weight)
    finite_anchor = (abs_weight > 0) & np.isfinite(anchor_center).all(axis=1)
    absolute_innovation = np.zeros(n, np.float64)
    absolute_innovation[finite_anchor] = np.linalg.norm(
        anchor_center[finite_anchor] - raw_solution[finite_anchor], axis=1
    )
    local_innovation = np.zeros(n, np.float64)
    pair = finite_anchor[1:] & finite_anchor[:-1] & relative_ok[1:]
    local_innovation[1:][pair] = np.linalg.norm(
        (anchor_center[1:][pair] - anchor_center[:-1][pair]) - relative[1:][pair],
        axis=1,
    )
    # Compare absolute measurements with the relative-motion shape after
    # removing its slow local offset.  This detects a bounded excursion that
    # returns to its neighbors without mistaking the normal long-term
    # difference between phase correlation and solar geometry for drift.
    relative_step = np.where(relative_ok[:, None], relative, 0.0)
    relative_position = np.cumsum(relative_step, axis=0)
    offset = anchor_center - relative_position
    closure_innovation = np.zeros(n, np.float64)
    half_window = 60
    for i in np.flatnonzero(finite_anchor):
        block = slice(max(0, i - half_window), min(n, i + half_window + 1))
        valid = finite_anchor[block]
        if valid.sum() < 5:
            continue
        baseline = np.median(offset[block][valid], axis=0)
        closure_innovation[i] = float(np.linalg.norm(offset[i] - baseline))

    # A gap bounded by reliable limb anchors has an explicit closure condition.
    # Weak contrast measurements may describe local shape, but cannot accumulate
    # a permanent offset between those endpoints.
    limb_indices = np.flatnonzero(limb_anchor & finite_anchor)
    for left, right in zip(limb_indices[:-1], limb_indices[1:]):
        if right - left <= 1 or right - left > 150:
            continue
        alpha = np.linspace(0.0, 1.0, right - left + 1)[:, None]
        expected = (1.0 - alpha) * offset[left] + alpha * offset[right]
        indices = np.arange(left, right + 1)
        valid = finite_anchor[indices]
        closure_innovation[indices[valid]] = np.maximum(
            closure_innovation[indices[valid]],
            np.linalg.norm(offset[indices[valid]] - expected[valid], axis=1),
        )

    cumulative_innovation = closure_innovation
    combined = np.maximum(absolute_innovation, closure_innovation)
    gate = np.full(n, 2.0 * scale, np.float64)
    for i in range(n):
        history = combined[max(0, i - 60) : i]
        history = history[np.isfinite(history) & (history > 0)]
        if len(history) < 5:
            continue
        median = float(np.median(history))
        mad = float(np.median(np.abs(history - median)))
        gate[i] = min(
            3.0 * scale,
            max(2.0 * scale, median + 3.0 * 1.4826 * mad),
        )

    transient_spike = np.zeros(n, bool)
    for i in range(1, n - 1):
        if not (finite_anchor[i - 1] and finite_anchor[i] and finite_anchor[i + 1]):
            continue
        # A coherent outer-limb measurement outranks phase and contrast backup.
        # Contrast-only neighbors must not vote an isolated limb anchor out and
        # then serve as its interpolation endpoints; only neighbouring limb
        # anchors may flag it as a spike.
        if limb_anchor[i] and not (limb_anchor[i - 1] and limb_anchor[i + 1]):
            continue
        neighbor_error = np.linalg.norm(
            (anchor_center[i + 1] - anchor_center[i - 1])
            - (relative_step[i] + relative_step[i + 1])
        )
        # Neighbor closure is scale/gate-relative and conservative: the two
        # neighbours must be coherent with the summed phase *within* a small
        # multiple of the robust gate (so a large ``local_innovation`` spike is
        # not enough on its own to declare corruption).  A real measurement
        # discontinuity has contradictory neighbours whose summed-phase error
        # is large, so it is preserved; only an isolated spike whose neighbours
        # are coherent within the closure gate is repaired.
        transient_spike[i] = (
            local_innovation[i] > gate[i]
            and local_innovation[i + 1] > gate[i]
            and neighbor_error <= 2.0 * gate[i]
        )

    anchor_factor = np.ones(n, np.float64)
    suspect = finite_anchor & transient_spike
    ratio = np.minimum(1.0, gate[suspect] / np.maximum(combined[suspect], 1e-9))
    ratio = np.minimum(ratio, 0.05)
    anchor_factor[suspect] = ratio * ratio
    # Contrast is never promoted to an absolute truth; disagreement weakens it
    # more quickly than a coherent outer-limb measurement.
    anchor_factor[contrast_only] = np.minimum(
        anchor_factor[contrast_only],
        np.where(
            combined[contrast_only] > gate[contrast_only],
            0.5
            * np.square(
                gate[contrast_only] / np.maximum(combined[contrast_only], 1e-9)
            ),
            1.0,
        ),
    )
    excursion_seed = finite_anchor & (closure_innovation > 2.0 * gate)
    accumulated_excursion = np.zeros(n, bool)
    padded_seed = np.r_[False, excursion_seed, False]
    changes = np.flatnonzero(padded_seed[1:] != padded_seed[:-1])
    for start, stop in changes.reshape(-1, 2):
        if stop - start < 3:
            continue
        before_candidates = np.flatnonzero(finite_anchor[max(0, start - 30) : start])
        after_candidates = np.flatnonzero(finite_anchor[stop : min(n, stop + 30)])
        if not len(before_candidates) or not len(after_candidates):
            continue
        before = max(0, start - 30) + int(before_candidates[-1])
        after = stop + int(after_candidates[0])
        closure_gate = 2.0 * max(gate[before], gate[after])
        if np.linalg.norm(offset[before] - offset[after]) > closure_gate:
            continue
        expanded_start = max(0, start - 30)
        expanded_stop = min(n, stop + 30)
        accumulated_excursion[expanded_start:expanded_stop] = finite_anchor[
            expanded_start:expanded_stop
        ]
    if accumulated_excursion.any() and len(limb_indices):
        accumulated_excursion[limb_indices[0]] = False
        accumulated_excursion[limb_indices[-1]] = False
    # A coherent, high-quality outer limb outranks phase correlation. Automatic
    # excursion suppression is reserved for weak limb fits or contrast-only
    # measurements; explicit profiles remain available for known source defects.
    accumulated_excursion &= contrast_only | (
        limb_anchor & (limb_quality < 0.15)
    )
    anchor_factor[accumulated_excursion] = np.minimum(
        anchor_factor[accumulated_excursion], 1e-6
    )
    repaired_anchor = anchor_center.copy()
    for i in np.flatnonzero(transient_spike):
        before = np.flatnonzero(finite_anchor[max(0, i - 5) : i] & ~transient_spike[max(0, i - 5) : i])
        after = np.flatnonzero(finite_anchor[i + 1 : min(n, i + 6)] & ~transient_spike[i + 1 : min(n, i + 6)])
        if not len(before) or not len(after):
            continue
        left = max(0, i - 5) + int(before[-1])
        right = i + 1 + int(after[0])
        alpha = (i - left) / (right - left)
        repaired_anchor[i] = (1.0 - alpha) * anchor_center[left] + alpha * anchor_center[right]
        anchor_factor[i] = 1.0

    relative_residual = np.zeros(n, np.float64)
    relative_residual[1:] = np.linalg.norm(
        np.diff(raw_solution, axis=0) - np.nan_to_num(relative[1:], nan=0.0), axis=1
    )
    rel_factor = np.ones(n, np.float64)
    bad_relative = relative_ok & (relative_residual > 4.0 * scale) & (response < normal)
    rel_factor[bad_relative] = 4.0 * scale / np.maximum(
        relative_residual[bad_relative], 1e-9
    )
    spike_edges = transient_spike.copy()
    spike_edges[1:] |= transient_spike[:-1]
    rel_factor[spike_edges] = np.minimum(rel_factor[spike_edges], 0.01)

    repaired_solution = weighted_path(
        repaired_anchor,
        relative,
        abs_weight * anchor_factor,
        rel_weight * rel_factor,
    )
    center = repaired_solution if auto_repair else raw_solution
    correction = np.linalg.norm(repaired_solution - raw_solution, axis=1)
    phase_length = np.linalg.norm(np.nan_to_num(relative, nan=0.0), axis=1)
    jump_confirmed = (
        relative_ok
        & (response >= 0.12)
        & (phase_length > 4.0 * scale)
        & (local_innovation <= 2.0 * scale)
    )
    jitter_candidate = transient_spike & ~jump_confirmed
    excursion_candidate = excursion_seed & ~jump_confirmed
    repair_scope = accumulated_excursion | transient_spike
    auto_repaired = auto_repair & repair_scope & (correction > 0.05 * scale)
    reasons = np.full(n, "sin reparación automática", dtype="U96")
    reasons[jitter_candidate] = "innovación local sin respaldo del movimiento relativo"
    reasons[excursion_candidate] = "innovación acumulada incompatible con anclas y movimiento relativo"
    reasons[jump_confirmed] = "salto real confirmado por movimiento relativo y medición independiente"
    reasons[auto_repaired & ~(jitter_candidate | excursion_candidate)] = (
        "ponderación robusta de innovación"
    )
    if not auto_repair:
        reasons[:] = "reparación automática desactivada (--no-auto-repair)"
    return {
        "center": center,
        "raw_solved_center": raw_solution,
        "repaired_center": repaired_solution,
        "abs_weight": abs_weight,
        "robust_abs_weight": abs_weight * anchor_factor,
        "rel_weight": rel_weight,
        "robust_rel_weight": rel_weight * rel_factor,
        "absolute_innovation": absolute_innovation,
        "local_innovation": local_innovation,
        "cumulative_innovation": cumulative_innovation,
        "correction_magnitude": correction,
        "auto_repaired": auto_repaired,
        "repair_reason": reasons,
        "jitter_candidate": jitter_candidate,
        "excursion_candidate": excursion_candidate,
        "jump_confirmed": jump_confirmed,
    }


def classify_regime(
    analysis: dict,
    min_quality: float = 0.01,
    scale: float = 1.0,
    sustain: int | None = None,
    recover: int = 5,
    min_strong: int = 2,
    coherence_gate: float | None = None,
) -> dict:
    """Classify every frame as limbo / transient backup / confirmed horizon.

    The decision is made from the per-frame geometry arrays alone (quality,
    residual, coverage, radial samples, radial scale, arc shape, border contact
    and centers) and never uses a percentage of the sequence length nor the
    mere fact that the video ended.

    - A frame is ``limbo`` while a coherent circular outer limb is measured by
      an *accepted* geometry source.  A propagated prediction (``SRC_NONE``,
      raw center non-finite) is never limb evidence.
    - A loss of that geometry is only ``transient backup`` until it is
      sustained, combined across several signals and monotonic: the limbo must
      never come back with several reliable, consecutive detections that are
      coherent with the trajectory.
    - The first frame of the final sustained+combined loss run is
      ``horizon_start``.  Confirmation is retrospective (the whole sequence is
      known) and the latch is irreversible: any circular detection from there
      on is kept only as diagnostic ``false_circle_after_horizon`` and is never
      used as an anchor.

    Confirmation scans whole transient runs (not a fixed initial window): a run
    must follow a stable limbo history, carry a sustained core of combined
    radial loss (including radius-scale and arc-shape degradation), and show
    monotonic physical decline over a longer fps-derived look-ahead window.  A
    real horizon can begin with a soft one-signal ramp (arc partially measurable
    but disagreeing) before a full collapse; a flat, persistently poor cloud
    tail never confirms a horizon.
    """
    n = len(analysis["quality"])
    frame_index = np.arange(n)
    fps = float(analysis["source_fps"][0])
    if sustain is None:
        sustain = max(12, int(round(0.5 * fps)))
    # Longer fps-derived look-ahead: the degradation may ramp softly before the
    # combined collapse appears, so the monotonic trends are judged over about
    # 1.5 s instead of the first ``sustain`` frames only.
    trend_window = max(3 * sustain, int(round(1.5 * fps)))
    raw = analysis["raw_center"]
    relative = analysis["relative"]
    response = analysis["response"]
    quality = analysis["quality"]
    coverage = analysis["coverage"]
    residual = analysis["median_residual"]
    radial_points = analysis["radial_points"]
    touch = analysis["touch"].astype(bool)
    if "clip_edges" in analysis and int(np.asarray(analysis["clip_edges"]).shape[0]) == n:
        clipped = np.asarray(analysis["clip_edges"], np.int8) != 0
    else:
        # Older caches predate the clipping classifier; the legacy single
        # threshold touch signal stands in for border contact.
        clipped = touch.copy()
    if coherence_gate is None:
        coherence_gate = max(10.0 * scale, 0.60 * float(analysis["radius"][0]))

    finite = np.isfinite(raw).all(axis=1)
    if "geometry_source" in analysis and int(np.asarray(analysis["geometry_source"]).shape[0]) == n:
        # A propagated position (SRC_NONE) is never an accepted limb
        # measurement, even if a stale center were finite.
        finite = finite & (np.asarray(analysis["geometry_source"]) != SRC_NONE)
    residual_ok = np.isfinite(residual) & (residual <= 4.5 * scale)
    limbo_evidence = (
        finite
        & ~touch
        & ~clipped
        & (coverage >= MIN_ARC_COVERAGE)
        & residual_ok
        & (radial_points >= MIN_RADIAL_POINTS)
        & (quality >= min_quality)
    )

    # Temporal coherence: a circular center must stay near the position
    # predicted from the last accepted detection plus the accumulated phase
    # motion.  This prevents a group of precise-looking circles that appear
    # elsewhere (clouds, atmospheric arcs) from being read as a recovery.
    # Spurious frames produce garbage phase shifts; those are ignored exactly
    # like the contrast tracker ignores them (response gate + movement cap).
    predicted_center = np.array([np.nan, np.nan])
    previous_geometric = np.array([np.nan, np.nan])
    coherent = np.zeros(n, bool)
    for i in range(n):
        delta = relative[i]
        if (
            np.isfinite(predicted_center).all()
            and np.isfinite(delta).all()
            and float(response[i]) >= 0.12
            and np.linalg.norm(delta) <= 35.0 * scale
        ):
            predicted_center = predicted_center + delta
        if limbo_evidence[i]:
            phase_coherent = False
            if np.isfinite(predicted_center).all():
                phase_coherent = bool(np.linalg.norm(raw[i] - predicted_center) <= coherence_gate)
            local_coherent = (
                np.isfinite(previous_geometric).all()
                and np.linalg.norm(raw[i] - previous_geometric) <= coherence_gate
            )
            if np.isfinite(predicted_center).all() or np.isfinite(previous_geometric).all():
                coherent[i] = phase_coherent or local_coherent
            else:
                coherent[i] = True
            previous_geometric = raw[i].copy()
            if coherent[i]:
                predicted_center = raw[i].copy()

    bad_quality = quality < max(min_quality, 0.12)
    bad_residual = ~residual_ok
    bad_coverage = coverage < MIN_ARC_COVERAGE
    bad_radial = radial_points < MIN_RADIAL_POINTS
    physical_occlusion = touch | clipped | (analysis["maximum"] < MIN_BRIGHTNESS)
    # Radial-aware signals (guarded: legacy dicts without the arrays simply do
    # not add these channels, keeping the historical loss profile unchanged).
    bad_radius_scale = np.zeros(n, bool)
    if "radius_meas" in analysis and int(np.asarray(analysis["radius_meas"]).shape[0]) == n:
        radius_meas = np.asarray(analysis["radius_meas"], np.float64)
        fixed_radius = float(analysis["radius"][0])
        if fixed_radius > 0:
            bad_radius_scale = ~(
                np.isfinite(radius_meas)
                & (np.abs(radius_meas / fixed_radius - 1.0) <= RADIUS_SCALE_TOLERANCE)
            )
    bad_arc_shape = np.zeros(n, bool)
    if (
        "arc_gap_deg" in analysis
        and "arc_measured" in analysis
        and int(np.asarray(analysis["arc_gap_deg"]).shape[0]) == n
    ):
        arc_gap = np.asarray(analysis["arc_gap_deg"], np.float64)
        arc_measured = np.asarray(analysis["arc_measured"], bool)
        bad_arc_shape = ~(arc_measured & (arc_gap <= 360.0 - MIN_ARC_COVERAGE))
    loss_signals = np.column_stack(
        (
            bad_quality,
            bad_residual,
            bad_coverage,
            bad_radial,
            physical_occlusion,
            bad_radius_scale,
            bad_arc_shape,
        )
    )
    signal_count = loss_signals.sum(axis=1)
    strong_loss = (~limbo_evidence) & (signal_count >= max(2, min_strong))

    # Recovery is intentionally stricter than a single good-looking circle.
    # Before any degradation direct coherent measurements remain limbo anchors;
    # after a transient loss, the whole recovering run is reinstated only once
    # several consecutive coherent detections agree.
    recovery_evidence = limbo_evidence & coherent
    regime = np.full(n, REGIME_TRANSIENT, np.int8)
    in_limbo = True
    i = 0
    while i < n:
        if recovery_evidence[i]:
            end = i + 1
            while end < n and recovery_evidence[end]:
                end += 1
            if in_limbo or end - i >= recover:
                regime[i:end] = REGIME_LIMBO
                in_limbo = True
            i = end
            continue
        in_limbo = False
        i += 1

    def trend_fraction(values: np.ndarray, direction: str, slack: float) -> float:
        finite_values = values[np.isfinite(values)]
        if len(finite_values) < 3:
            return 0.0
        delta = np.diff(finite_values)
        if direction == "down":
            return float(np.mean(delta <= slack))
        return float(np.mean(delta >= -slack))

    # Confirm the horizon on the first transient run that is sustained on its
    # own: it must follow a stable limbo history, carry a core of combined
    # radial loss at least ``sustain`` frames long, and show monotonic physical
    # decline over a longer fps-derived look-ahead window.  A flat, persistently
    # poor cloud tail therefore stays transient.  The scan is retrospective but
    # never depends on where the end of the video lies nor on its length.
    horizon_start = n
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if regime[i] == REGIME_TRANSIENT:
            j = i
            while j < n and regime[j] == REGIME_TRANSIENT:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    for start, end in runs:
        if start < recover:
            continue
        if not np.all(regime[start - recover : start] == REGIME_LIMBO):
            continue
        # Longest contiguous strong-loss core inside the run.  The horizon may
        # begin with a soft one-signal ramp, so the core is measured with the
        # radial-aware channels (a ramp already fails coverage + radius-scale +
        # arc-shape together).
        strong_block = np.asarray(strong_loss[start:end])
        core_len = 0
        best_core = 0
        for value in strong_block:
            if value:
                core_len += 1
                best_core = max(best_core, core_len)
            else:
                core_len = 0
        if best_core < sustain:
            continue
        window_stop = min(end, start + trend_window)
        window_len = window_stop - start
        if window_len < 3:
            continue
        third = max(2, window_len // 3)
        head = slice(start, start + third)
        tail = slice(window_stop - third, window_stop)
        radial_drop = float(np.median(radial_points[head]) - np.median(radial_points[tail]))
        coverage_drop = float(np.median(coverage[head]) - np.median(coverage[tail]))
        quality_drop = float(np.median(quality[head]) - np.median(quality[tail]))
        finite_head = residual[head][np.isfinite(residual[head])]
        finite_tail = residual[tail][np.isfinite(residual[tail])]
        residual_rise = (
            float(np.median(finite_tail) - np.median(finite_head))
            if len(finite_head) and len(finite_tail)
            else 0.0
        )
        physical_progress = float(np.mean(physical_occlusion[tail]) - np.mean(physical_occlusion[head]))
        block = slice(start, window_stop)
        radial_trend = radial_drop >= 8 and trend_fraction(radial_points[block], "down", 2.0) >= 0.60
        coverage_trend = coverage_drop >= 25.0 and trend_fraction(coverage[block], "down", 5.0) >= 0.60
        quality_trend = quality_drop >= 0.03 and trend_fraction(
            quality[block], "down", 0.01
        ) >= 0.60
        residual_trend = residual_rise >= 1.0 * scale and trend_fraction(
            residual[block], "up", 0.25 * scale
        ) >= 0.60
        physical_trend = physical_progress >= 0.25
        if sum((radial_trend, coverage_trend, quality_trend, residual_trend, physical_trend)) < 2:
            continue
        horizon_start = int(start)
        break

    if horizon_start < n:
        regime[horizon_start:] = REGIME_HORIZON

    # Backup (contrast) anchors are needed wherever limbo is not active.
    contrast_start = n
    not_limbo = regime != REGIME_LIMBO
    if not_limbo.any():
        contrast_start = int(np.flatnonzero(not_limbo)[0])

    false_circle = limbo_evidence & (frame_index >= horizon_start)
    reasons = np.full(n, "degradación transitoria o evidencia insuficiente", dtype="U96")
    reasons[regime == REGIME_LIMBO] = "limbo exterior fiable y coherente"
    reasons[regime == REGIME_HORIZON] = "horizonte confirmado: pérdida sostenida y monotónica del limbo"
    reasons[false_circle] = "falso círculo posterior al horizonte; rechazado como ancla"
    return {
        "regime": regime,
        "horizon_start": horizon_start,
        "contrast_start": contrast_start,
        "false_circle_after_horizon": false_circle,
        "limbo_evidence": limbo_evidence,
        "coherent": coherent,
        "strong_loss": strong_loss,
        "signal_count": signal_count.astype(np.int8),
        "bad_radius_scale": bad_radius_scale,
        "bad_arc_shape": bad_arc_shape,
        "regime_reason": reasons,
    }


def apply_profile_discards(
    keep: np.ndarray,
    profile: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Apply profile-declared discards expressed in immutable source frames.

    Without a profile no frame is discarded by its index; the default behavior
    never assumes the original video's known defects.
    """
    selected_keep = keep.copy()
    discarded_mask = np.zeros(len(keep), bool)
    messages: list[str] = []
    if profile is not None:
        for start, end in profile["discards"]:
            if start >= len(selected_keep):
                continue
            end = min(end, len(selected_keep) - 1)
            selected_keep[start : end + 1] = False
            discarded_mask[start : end + 1] = True
            if start == end:
                messages.append(f"frame descartado por perfil: src {start}")
            else:
                messages.append(f"rango descartado por perfil: src {start}-{end}")
    return selected_keep, discarded_mask, messages


def isolate_transient_tears(
    candidates: np.ndarray,
    fps: float,
    eligible: np.ndarray | None = None,
    max_duration_seconds: float = 0.067,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only short eligible sensor-tear runs.

    Readout tears appear and disappear within about two 30-fps frames.  The
    limit is expressed as a duration so equivalent high-fps sources receive an
    equivalent number of frames.  Ineligible candidates (real canvas clipping
    or frames at/after the horizon latch) do not join a run.
    """
    raw = np.asarray(candidates, bool)
    allowed = np.ones(len(raw), bool) if eligible is None else np.asarray(eligible, bool)
    active = raw & allowed
    run_length = np.zeros(len(raw), np.int32)
    detected = np.zeros(len(raw), bool)
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps debe ser finito y positivo")
    padded = np.r_[False, active, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    for start, stop in changes.reshape(-1, 2):
        length = stop - start
        run_length[start:stop] = length
        tolerance = np.finfo(np.float64).eps * max(1.0, length / fps)
        if length / fps <= max_duration_seconds + tolerance:
            detected[start:stop] = True
    return detected, run_length


def detect_transient_exposure_drops(
    exposed_fraction: np.ndarray,
    evaluable: np.ndarray,
    eligible: np.ndarray,
    fps: float,
    window_seconds: float = 0.5,
    min_drop_fraction: float = 0.08,
    noise_multiplier: float = 6.0,
    max_duration_seconds: float = 0.067,
) -> dict:
    """Detect a brief loss of exposed photosphere relative to nearby frames."""
    exposed = np.asarray(exposed_fraction, np.float64)
    usable = np.asarray(evaluable, bool) & np.asarray(eligible, bool) & np.isfinite(exposed)
    n = len(exposed)
    window = max(5, int(round(window_seconds * float(fps))))
    candidate = np.zeros(n, bool)
    baseline = np.full(n, np.nan, np.float64)
    drop = np.full(n, np.nan, np.float64)
    noise = np.full(n, np.nan, np.float64)
    for i in np.flatnonzero(usable):
        start = max(0, i - window)
        stop = min(n, i + window + 1)
        past = exposed[start:i][usable[start:i]]
        future = exposed[i + 1 : stop][usable[i + 1 : stop]]
        if len(past) < 3 or len(future) < 3:
            continue
        baseline[i] = min(float(np.median(past)), float(np.median(future)))
        neighbors = np.r_[past, future]
        median = float(np.median(neighbors))
        noise[i] = 1.4826 * float(np.median(np.abs(neighbors - median)))
        drop[i] = baseline[i] - exposed[i]
        candidate[i] = drop[i] >= max(min_drop_fraction, noise_multiplier * noise[i])
    detected, run_length = isolate_transient_tears(
        candidate,
        fps,
        eligible=usable,
        max_duration_seconds=max_duration_seconds,
    )
    return {
        "candidate": candidate,
        "detected": detected,
        "run_length": run_length,
        "baseline": baseline,
        "drop": drop,
        "noise": noise,
    }


def classify_content(
    analysis: dict,
    clipped: np.ndarray,
    clip_edges: np.ndarray,
    clip_score: np.ndarray,
    horizon_start: int,
    tear_detected: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify frame content as usable / clipped / uncertain.

    ``clipped`` combines confirmed border contact and confirmed sensor tears;
    both mean that visible source content is missing.  ``clip_edges`` remains
    the independent physical-border measurement, never a theoretical-circle
    test.  A frame without either kind of damage but with too little content or
    geometric support becomes ``uncertain`` instead of being discarded.
    """
    n = len(clipped)
    bright = analysis["maximum"] >= MIN_BRIGHTNESS
    radial_ok = analysis["radial_points"] >= MIN_RADIAL_POINTS
    tear = (
        np.zeros(n, bool)
        if tear_detected is None
        else np.asarray(tear_detected, bool)
    )
    frame_index = np.arange(n)
    circular = frame_index < horizon_start
    usable = ~clipped & bright & (circular & radial_ok | ~circular)
    states = np.full(n, CONTENT_UNCERTAIN, np.int8)
    states[usable] = CONTENT_USABLE
    states[clipped] = CONTENT_CLIPPED
    reasons: list[str] = []
    for i in range(n):
        if clipped[i]:
            if tear[i]:
                drop = float(analysis.get("tear_drop_fraction", np.full(n, np.nan))[i])
                reasons.append(
                    f"pérdida temporal de fotosfera confirmada ({drop * 100:.1f}%)"
                )
            else:
                edges = bitmask_to_edges(clip_edges[i])
                reasons.append(
                    f"recorte real en {', '.join(edges) or 'borde'} "
                    f"(acuerdo {int(round(clip_score[i] * 100))}%)"
                )
        elif usable[i]:
            reasons.append("contenido solar completo dentro del lienzo")
        else:
            why: list[str] = []
            if analysis["maximum"][i] < MIN_BRIGHTNESS:
                why.append(f"brillo bajo max={int(analysis['maximum'][i])}")
            if circular[i] and not radial_ok[i]:
                why.append(f"evidencia geométrica débil ({int(analysis['radial_points'][i])} radial)")
            reasons.append("; ".join(why) or "sin evidencia suficiente para decidir")
    return states, np.array(reasons, dtype="U90")


def classify_centering(
    anchor: np.ndarray,
    source_fps: float,
    reconstructed_support: np.ndarray | None = None,
    contradictory: np.ndarray | None = None,
    max_extrap_seconds: float = 1.0,
    max_bridge_seconds: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify centering confidence as reliable / reconstructed / unresolved.

    A directly measured anchor is ``reliable``.  Frames between reliable anchors
    are ``reconstructed`` from relative motion and trustworthy neighbors.  A
    frame with no anchor and no reliable neighbor within the extrapolation
    window cannot guarantee centering and is ``unresolved``.
    """
    n = len(anchor)
    support = (
        np.zeros(n, bool)
        if reconstructed_support is None
        else np.asarray(reconstructed_support, bool)
    )
    conflict = np.zeros(n, bool) if contradictory is None else np.asarray(contradictory, bool)
    states = np.full(n, CENTER_UNRESOLVED, np.int8)
    states[anchor] = CENTER_RELIABLE
    states[support & ~conflict & ~anchor] = CENTER_RECONSTRUCTED
    next_anchor = np.full(n, n, np.int64)
    last = n
    for i in range(n - 1, -1, -1):
        if anchor[i]:
            last = i
        next_anchor[i] = last
    prev_anchor = np.full(n, -1, np.int64)
    last = -1
    for i in range(n):
        if anchor[i]:
            last = i
        prev_anchor[i] = last
    max_extrap = max(10, int(round(max_extrap_seconds * source_fps)))
    max_bridge = max(max_extrap, int(round(max_bridge_seconds * source_fps)))
    for i in range(n):
        if states[i] != CENTER_UNRESOLVED or conflict[i]:
            continue
        has_prev = prev_anchor[i] >= 0
        has_next = next_anchor[i] < n
        if has_prev and has_next and (next_anchor[i] - prev_anchor[i]) <= max_bridge:
            states[i] = CENTER_RECONSTRUCTED
        elif has_prev and (i - prev_anchor[i]) <= max_extrap:
            states[i] = CENTER_RECONSTRUCTED
        elif has_next and (next_anchor[i] - i) <= max_extrap:
            states[i] = CENTER_RECONSTRUCTED
    reasons: list[str] = []
    for i in range(n):
        if anchor[i]:
            reasons.append("medición directa usada como ancla")
        elif states[i] == CENTER_RECONSTRUCTED:
            if support[i]:
                reasons.append("reconstruido con respaldo de contraste y movimiento relativo")
                continue
            reasons.append(
                f"reconstruido entre vecinos fiables (prev {prev_anchor[i]}, next {next_anchor[i]})"
            )
        elif conflict[i]:
            reasons.append("señales de movimiento contradictorias; centrado sin resolver")
        else:
            reasons.append("sin ancla fiable cercana; no se garantiza el centrado")
    return states, np.array(reasons, dtype="U90")


def contradicted_reconstruction(
    content_state: np.ndarray,
    centering_state: np.ndarray,
    geometry_trusted: np.ndarray,
    jitter_candidate: np.ndarray,
    excursion_candidate: np.ndarray,
    jump_confirmed: np.ndarray,
) -> np.ndarray:
    """Flag reconstructed centering contradicted by every available signal.

    A reconstructed center is only trustworthy when it is backed by an
    independent geometric measurement or is not itself the product of a
    spurious tracker signal.  It is contradicted when all of the following
    coincide: UNCERTAIN content, RECONSTRUCTED centering, no trusted geometry,
    a jitter/excursion tracker-error flag, and no confirmed real jump.  The
    caller demotes such frames to ``CENTER_UNRESOLVED`` so they are never
    exported, while preserving genuine geometry-confirmed jumps and
    uncertain-but-directly-measured frames.
    """
    return (
        (content_state == CONTENT_UNCERTAIN)
        & (centering_state == CENTER_RECONSTRUCTED)
        & ~geometry_trusted
        & (jitter_candidate | excursion_candidate)
        & ~jump_confirmed
    )


def solve_tracking(
    analysis: dict,
    min_quality: float,
    profile: dict | None = None,
    auto_repair: bool = True,
) -> dict:
    raw = analysis["raw_center"]
    quality = analysis["quality"]
    touch = analysis["touch"].astype(bool)
    relative = analysis["relative"]
    response = analysis["response"]
    n = len(raw)
    width = int(analysis["analysis_width"][0])
    height = int(analysis["analysis_height"][0])
    radius = float(analysis["radius"][0])
    scale = analysis_scale(width)
    regime_info = classify_regime(analysis, scale=scale)
    regime = regime_info["regime"]
    horizon_start = int(regime_info["horizon_start"])
    contrast_start = int(regime_info["contrast_start"])
    false_circle = regime_info["false_circle_after_horizon"]
    limbo_evidence = regime_info["limbo_evidence"]
    regime_reason = regime_info["regime_reason"]
    frame_index = np.arange(n)
    # Circular anchors are only the pre-latch limbo detections.  A circular
    # detection after a confirmed horizon is kept as diagnostic
    # (false_circle_after_horizon) and is never used as an anchor.
    geometry_usable = limbo_evidence & ~false_circle & (frame_index < horizon_start)
    trusted_initial = geometry_usable
    trusted = transient_outliers(raw, relative, trusted_initial, gate=4.0 * scale)

    anchor_center = raw.copy()
    contrast_center = analysis["contrast_center"]
    contrast_finite = np.isfinite(contrast_center).all(axis=1)
    # The fallback is usable as a backup anchor only once it is genuinely
    # supported by a real geometric anchor.  A blank / no-history sequence
    # emits no finite center and must never become reconstructed/exportable
    # centering.  Backward-compatible: older synthetic dicts/caches without
    # ``fallback_supported`` fall back to "finite contrast" (pre-Phase-3-4).
    if "fallback_supported" in analysis and int(np.asarray(analysis["fallback_supported"]).shape[0]) == n:
        fallback_supported = np.asarray(analysis["fallback_supported"], bool)
    else:
        fallback_supported = contrast_finite.copy()
    # Backup anchor: every non-limbo frame (transient backup or confirmed
    # horizon) is followed by the local-contrast model while it exists and is
    # actually supported by prior geometry.
    backup_anchor = (regime != REGIME_LIMBO) & contrast_finite & fallback_supported & ~trusted
    horizon_visible = (regime == REGIME_HORIZON) & contrast_finite
    horizon_present = (regime == REGIME_HORIZON) & (analysis["maximum"] >= MIN_BRIGHTNESS)
    anchor_center[backup_anchor] = contrast_center[backup_anchor]
    anchor_used = trusted | backup_anchor
    path = robust_path_solution(
        anchor_center,
        relative,
        response,
        trusted,
        quality,
        backup_anchor,
        analysis.get("contrast_score", np.zeros(n, np.float64)),
        scale,
        auto_repair=auto_repair,
    )
    center = path["center"]
    trim_end = n - 1
    margin = np.minimum.reduce(
        [
            center[:, 0] - radius,
            center[:, 1] - radius,
            (width - 1.0) - (center[:, 0] + radius),
            (height - 1.0) - (center[:, 1] + radius),
        ]
    )
    # Diagnostic only: the theoretical circle leaving the canvas must never be
    # the criterion for exclusion (the occulted side may lie outside while every
    # visible pixel stays inside).  Real border clipping comes from the luminous
    # component classifier below.
    cut_geometry = margin < -0.75 * scale

    if (
        "clip_edges" in analysis
        and "clip_score" in analysis
        and int(analysis["clip_edges"].shape[0]) == n
    ):
        clip_edges = np.asarray(analysis["clip_edges"], np.int8)
        clip_score = np.asarray(analysis["clip_score"], np.float64)
        border_clipped = clip_edges != 0
    else:
        # Older caches predate the clipping classifier; fall back to the legacy
        # single-threshold touch signal until the classification is refreshed.
        clip_edges = np.zeros(n, np.int8)
        clip_score = np.zeros(n, np.float64)
        border_clipped = touch.copy()
    tear_eligible = (
        ~border_clipped
        & (frame_index < horizon_start)
    )
    tear_evaluable = np.asarray(
        analysis.get("tear_evaluable", np.zeros(n, bool)), bool
    )
    tear_info = detect_transient_exposure_drops(
        analysis.get("tear_exposed_fraction", np.full(n, np.nan)),
        tear_evaluable,
        tear_eligible,
        float(analysis["source_fps"][0]),
    )
    tear_candidate = tear_info["candidate"]
    tear_detected = tear_info["detected"]
    tear_run_length = tear_info["run_length"]
    tear_veto_reason = np.full(n, "sin caída temporal anómala", dtype="U128")
    for i in range(n):
        if border_clipped[i]:
            tear_veto_reason[i] = "no evaluado: classify_clipping confirmó recorte de borde"
        elif i >= horizon_start:
            tear_veto_reason[i] = "no evaluado: frame en régimen de horizonte confirmado"
        elif not tear_evaluable[i]:
            tear_veto_reason[i] = "no evaluado: exposición o geometría insuficiente"
        elif tear_detected[i]:
            tear_veto_reason[i] = "pérdida temporal breve de fotosfera confirmada"
        elif tear_candidate[i]:
            tear_veto_reason[i] = (
                f"vetado: anomalía sostenida de {int(tear_run_length[i])} frames"
            )
    # Both cases lose visible source content and intentionally share the
    # CONTENT_CLIPPED/manual-recovery policy.  The underlying measurements stay
    # separate for audit and horizon classification.
    clipped = border_clipped | tear_detected
    content_analysis = dict(analysis)
    content_analysis["tear_drop_fraction"] = tear_info["drop"]
    content_state, content_reason = classify_content(
        content_analysis,
        clipped,
        clip_edges,
        clip_score,
        horizon_start,
        tear_detected=tear_detected,
    )
    intensity_delta = np.r_[0.0, np.abs(np.diff(analysis["maximum"].astype(float)))]
    positive_delta = intensity_delta[intensity_delta > 0]
    intensity_gate = (
        max(20.0, float(np.median(positive_delta)) + 6.0 * float(np.median(np.abs(positive_delta - np.median(positive_delta)))))
        if len(positive_delta)
        else 20.0
    )
    corruption_signals = np.column_stack(
        (
            analysis["maximum"] < MIN_BRIGHTNESS,
            response < 0.05,
            intensity_delta > intensity_gate,
            path["absolute_innovation"] > 4.0 * scale,
        )
    )
    corruption_candidate = corruption_signals.sum(axis=1) >= 2
    uncertain_corrupt = corruption_candidate & ~clipped
    content_state[uncertain_corrupt] = CONTENT_UNCERTAIN
    content_reason[uncertain_corrupt] = "candidato a corrupción por varias señales; revisión necesaria"
    contradictory = (
        backup_anchor
        & (path["absolute_innovation"] > max(4.0 * scale, 0.5 * radius))
        & (response < 0.05)
    )
    direct_reliable = trusted & (regime == REGIME_LIMBO)
    centering_state, centering_reason = classify_centering(
        direct_reliable,
        float(analysis["source_fps"][0]),
        reconstructed_support=backup_anchor | (trusted & ~direct_reliable),
        contradictory=contradictory,
    )
    # Contradicted reconstruction: a reconstructed center whose only support is
    # the local-contrast fallback on UNCERTAIN content, with no trusted geometric
    # measurement and a tracker-error signal (jitter/excursion) that is NOT a
    # confirmed real jump.  The reconstructed position is contradicted by the
    # absence of any independent measurement and by the spurious phase signal
    # driving it, so it must not be exported as reliable centering.
    contradicted = contradicted_reconstruction(
        content_state,
        centering_state,
        trusted,
        path["jitter_candidate"],
        path["excursion_candidate"],
        path["jump_confirmed"],
    )
    if contradicted.any():
        centering_state = centering_state.copy()
        centering_reason = centering_reason.copy()
        centering_state[contradicted] = CENTER_UNRESOLVED
        centering_reason[contradicted] = (
            "reconstrucción contradictoria: contenido incierto sin geometría "
            "fiable y señal de error de seguimiento (jitter/excursión) no confirmada"
        )
    # Policy: only a confirmed visible clip excludes automatically.  usable and
    # uncertain content are exported whenever the centering is reliable or
    # reconstructed; unresolved centering is never exported silently.
    keep = ~clipped & (centering_state != CENTER_UNRESOLVED)
    keep, timed_discarded, discard_messages = apply_profile_discards(
        keep,
        profile=profile,
    )
    for message in discard_messages:
        print(f"  descarte por perfil: {message}")
    return {
        "center": center,
        "anchor_center": anchor_center,
        "trusted": anchor_used,
        "geometry_trusted": trusted,
        "horizon_tracked": backup_anchor,
        "backup_tracked": backup_anchor,
        "horizon_visible": horizon_visible,
        "horizon_present": horizon_present,
        "rejected": trusted_initial & ~trusted,
        "trim_end": np.array([trim_end], dtype=np.int64),
        "horizon_start": np.array([horizon_start], dtype=np.int64),
        "contrast_start": np.array([contrast_start], dtype=np.int64),
        "regime": regime,
        "regime_reason": regime_reason,
        "false_circle_after_horizon": false_circle,
        "limbo_evidence": limbo_evidence,
        "strong_loss": regime_info["strong_loss"],
        "signal_count": regime_info["signal_count"],
        "bad_radius_scale": regime_info["bad_radius_scale"],
        "bad_arc_shape": regime_info["bad_arc_shape"],
        "margin": margin,
        "cut_geometry": cut_geometry,
        "keep": keep,
        "timed_discarded": timed_discarded,
        "clipped": clipped,
        "border_clipped": border_clipped,
        "clip_edges": clip_edges,
        "clip_score": clip_score,
        "tear_detected": tear_detected,
        "tear_run_length": tear_run_length,
        "tear_veto_reason": tear_veto_reason,
        "tear_candidate": tear_candidate,
        "tear_baseline_fraction": tear_info["baseline"],
        "tear_drop_fraction": tear_info["drop"],
        "tear_temporal_noise": tear_info["noise"],
        "content_state": content_state,
        "content_reason": content_reason,
        "centering_state": centering_state,
        "centering_reason": centering_reason,
        "contradicted_reconstruction": contradicted,
        "raw_solved_center": path["raw_solved_center"],
        "repaired_center": path["repaired_center"],
        "absolute_innovation": path["absolute_innovation"],
        "local_innovation": path["local_innovation"],
        "cumulative_innovation": path["cumulative_innovation"],
        "correction_magnitude": path["correction_magnitude"],
        "auto_repaired": path["auto_repaired"],
        "repair_reason": path["repair_reason"],
        "jitter_candidate": path["jitter_candidate"],
        "excursion_candidate": path["excursion_candidate"],
        "jump_confirmed": path["jump_confirmed"],
        "corruption_candidate": corruption_candidate,
        "corruption_signal_count": corruption_signals.sum(axis=1).astype(np.int8),
        "anchor_weight": path["abs_weight"],
        "robust_anchor_weight": path["robust_abs_weight"],
    }


def write_tracking_csv(path: Path, analysis: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame",
                "raw_cx",
                "raw_cy",
                "solved_cx",
                "solved_cy",
                "pre_repair_cx",
                "pre_repair_cy",
                "correction_magnitude",
                "absolute_innovation",
                "local_innovation",
                "cumulative_innovation",
                "jitter_candidate",
                "excursion_candidate",
                "jump_confirmed",
                "corruption_candidate",
                "corruption_signal_count",
                "auto_repaired",
                "repair_reason",
                "quality",
                "coverage_deg",
                "median_residual",
                "threshold",
                "touch",
                "trusted",
                "rejected",
                "phase_dx",
                "phase_dy",
                "phase_response",
                "radial_points",
                "radial_strength",
                "radius_meas",
                "arc_gap_deg",
                "strong_loss",
                "signal_count",
                "bad_radius_scale",
                "bad_arc_shape",
                "margin",
                "cut_geometry",
                "keep",
                "visible_cx",
                "visible_cy",
                "horizon_tracked",
                "regime",
                "regime_reason",
                "false_circle_after_horizon",
                "contrast_cx",
                "contrast_cy",
                "contrast_score",
                "timed_discarded",
                "clipped",
                "border_clipped",
                "clip_score",
                "tear_candidate",
                "tear_evaluable",
                "tear_detected",
                "tear_run_length",
                "tear_reason",
                "tear_veto_reason",
                "tear_bright_level",
                "tear_visible_threshold",
                "tear_exposed_fraction",
                "tear_baseline_fraction",
                "tear_drop_fraction",
                "tear_temporal_noise",
                "content_state",
                "content_reason",
                "centering_state",
                "centering_reason",
                "contradicted_reconstruction",
                "geometry_source",
                "arc_measured",
                "arc_valid_points",
                "arc_coverage_deg",
                "contrast_dyn_offset_cx",
                "contrast_dyn_offset_cy",
                "contrast_offset_sample_cx",
                "contrast_offset_sample_cy",
                "fallback_reanchored",
                "fallback_supported",
                "fallback_innovation",
                "fallback_mode",
            ]
        )
        for i in range(len(analysis["quality"])):
            content_code = int(analysis["content_state"][i]) if "content_state" in analysis else -1
            centering_code = int(analysis["centering_state"][i]) if "centering_state" in analysis else -1
            regime_code = int(analysis["regime"][i]) if "regime" in analysis else -1
            geometry_source_code = (
                int(analysis["geometry_source"][i]) if "geometry_source" in analysis else -1
            )
            writer.writerow(
                [
                    i,
                    *analysis["raw_center"][i],
                    *analysis["center"][i],
                    *analysis.get("raw_solved_center", analysis["center"])[i],
                    analysis.get("correction_magnitude", np.zeros(len(analysis["quality"])))[i],
                    analysis.get("absolute_innovation", np.zeros(len(analysis["quality"])))[i],
                    analysis.get("local_innovation", np.zeros(len(analysis["quality"])))[i],
                    analysis.get("cumulative_innovation", np.zeros(len(analysis["quality"])))[i],
                    int(analysis.get("jitter_candidate", np.zeros(len(analysis["quality"]), bool))[i]),
                    int(analysis.get("excursion_candidate", np.zeros(len(analysis["quality"]), bool))[i]),
                    int(analysis.get("jump_confirmed", np.zeros(len(analysis["quality"]), bool))[i]),
                    int(analysis.get("corruption_candidate", np.zeros(len(analysis["quality"]), bool))[i]),
                    int(analysis.get("corruption_signal_count", np.zeros(len(analysis["quality"]), np.int8))[i]),
                    int(analysis.get("auto_repaired", np.zeros(len(analysis["quality"]), bool))[i]),
                    analysis.get("repair_reason", np.full(len(analysis["quality"]), ""))[i],
                    analysis["quality"][i],
                    analysis["coverage"][i],
                    analysis["median_residual"][i],
                    analysis["threshold"][i],
                    int(analysis["touch"][i]),
                    int(analysis["trusted"][i]),
                    int(analysis["rejected"][i]),
                    *analysis["relative"][i],
                    analysis["response"][i],
                    int(analysis["radial_points"][i]),
                    analysis["radial_strength"][i],
                    analysis.get("radius_meas", np.full(len(analysis["quality"]), np.nan))[i],
                    analysis.get("arc_gap_deg", np.zeros(len(analysis["quality"])))[i],
                    int(analysis.get("strong_loss", np.zeros(len(analysis["quality"]), bool))[i]),
                    int(analysis.get("signal_count", np.zeros(len(analysis["quality"]), np.int8))[i]),
                    int(analysis.get("bad_radius_scale", np.zeros(len(analysis["quality"]), bool))[i]),
                    int(analysis.get("bad_arc_shape", np.zeros(len(analysis["quality"]), bool))[i]),
                    analysis["margin"][i],
                    int(analysis["cut_geometry"][i]),
                    int(analysis["keep"][i]),
                    *analysis["visible_center"][i],
                    int(analysis["horizon_tracked"][i]),
                    REGIME_NAMES.get(regime_code, "unknown"),
                    analysis["regime_reason"][i] if "regime_reason" in analysis else "",
                    int(analysis["false_circle_after_horizon"][i])
                    if "false_circle_after_horizon" in analysis
                    else 0,
                    *analysis["contrast_center"][i],
                    analysis["contrast_score"][i],
                    int(analysis["timed_discarded"][i]),
                    int(analysis.get("clipped", analysis["touch"])[i]),
                    int(analysis.get("border_clipped", analysis.get("clipped", analysis["touch"]))[i]),
                    analysis.get("clip_score", np.zeros(len(analysis["quality"])))[i],
                    int(analysis.get("tear_candidate", np.zeros(len(analysis["quality"]), bool))[i]),
                    int(analysis.get("tear_evaluable", np.zeros(len(analysis["quality"]), bool))[i]),
                    int(analysis.get("tear_detected", np.zeros(len(analysis["quality"]), bool))[i]),
                    int(analysis.get("tear_run_length", np.zeros(len(analysis["quality"]), np.int32))[i]),
                    analysis.get("tear_reason", np.full(len(analysis["quality"]), ""))[i],
                    analysis.get("tear_veto_reason", np.full(len(analysis["quality"]), ""))[i],
                    analysis.get("tear_bright_level", np.full(len(analysis["quality"]), np.nan))[i],
                    analysis.get("tear_visible_threshold", np.full(len(analysis["quality"]), np.nan))[i],
                    analysis.get("tear_exposed_fraction", np.full(len(analysis["quality"]), np.nan))[i],
                    analysis.get("tear_baseline_fraction", np.full(len(analysis["quality"]), np.nan))[i],
                    analysis.get("tear_drop_fraction", np.full(len(analysis["quality"]), np.nan))[i],
                    analysis.get("tear_temporal_noise", np.full(len(analysis["quality"]), np.nan))[i],
                    CONTENT_NAMES.get(content_code, "unknown"),
                    analysis["content_reason"][i] if "content_reason" in analysis else "",
                    CENTER_NAMES.get(centering_code, "unknown"),
                    analysis["centering_reason"][i] if "centering_reason" in analysis else "",
                    int(analysis.get("contradicted_reconstruction", np.zeros(len(analysis["quality"]), bool))[i]),
                    SRC_NAMES.get(geometry_source_code, "unknown"),
                    int(analysis.get("arc_measured", np.zeros(len(analysis["quality"]), bool))[i]),
                    int(analysis.get("arc_valid_points", np.zeros(len(analysis["quality"]), np.int16))[i]),
                    analysis.get("arc_coverage", np.zeros(len(analysis["quality"])))[i],
                    *analysis.get("contrast_dynamic_offset", np.full((len(analysis["quality"]), 2), np.nan))[i],
                    *analysis.get("contrast_offset_sample", np.full((len(analysis["quality"]), 2), np.nan))[i],
                    int(analysis.get("fallback_reanchored", np.zeros(len(analysis["quality"]), bool))[i]),
                    int(analysis.get("fallback_supported", np.zeros(len(analysis["quality"]), bool))[i]),
                    analysis.get("fallback_innovation", np.zeros(len(analysis["quality"])))[i],
                    int(analysis.get("fallback_mode", np.zeros(len(analysis["quality"]), np.int8))[i]),
                ]
            )
