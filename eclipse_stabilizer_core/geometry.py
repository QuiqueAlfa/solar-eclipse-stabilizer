from __future__ import annotations

import math

import cv2
import numpy as np

from .constants import (
    EDGE_BITS,
    EDGE_NAMES,
    MIN_BRIGHTNESS,
    MIN_RADIAL_POINTS,
    SRC_CONTOUR,
    SRC_NONE,
    SRC_RADIAL,
    SRC_RADIAL_DISAGREE,
)


EXPOSURE_SAMPLE_RADIUS_R = 0.90
EXPOSURE_MEASURE_RADIUS_R = 0.84
EXPOSURE_MIN_LEVEL_FRACTION = 0.125
EXPOSURE_VISIBLE_FRACTION = 0.20


def analysis_scale(width: int) -> float:
    """Spatial distances are calibrated for an analysis width of 270 px."""
    return max(1.0, width) / 270.0


def edges_to_bitmask(edges: list[str]) -> int:
    mask = 0
    for edge in edges:
        mask |= EDGE_BITS[edge]
    return mask


def bitmask_to_edges(mask: int) -> list[str]:
    return [EDGE_NAMES[bit] for bit in EDGE_NAMES if int(mask) & bit]


def largest_contour(
    gray: np.ndarray, threshold: float, scale: float = 1.0
) -> tuple[np.ndarray | None, np.ndarray, bool]:
    kernel = max(2, int(round(2.0 * scale)))
    mask = (gray >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((kernel, kernel), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, mask, False
    h, w = gray.shape
    reasonable = [c for c in contours if 40.0 * scale * scale <= cv2.contourArea(c) <= 0.72 * h * w]
    if not reasonable:
        return None, mask, False
    contour = max(reasonable, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(contour)
    touch = x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1
    return contour, mask, touch


def clipping_thresholds(gray: np.ndarray) -> list[float]:
    maximum = float(gray.max())
    if maximum < MIN_BRIGHTNESS:
        return []
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    values = [10.0, 15.0, 20.0, 0.20 * maximum, 0.30 * maximum, float(otsu)]
    return sorted({round(v, 1) for v in values if 5.0 <= v <= 0.9 * maximum})


def classify_clipping(
    gray: np.ndarray,
    predicted: np.ndarray | None,
    radius: float,
    scale: float = 1.0,
) -> dict:
    """Decide whether the luminous solar component really touches an image border.

    The search is limited to the region around the predicted solar position, so
    bright landscape, poles or isolated noise elsewhere cannot vote for a clip.
    Several thresholds must agree and the border contact must have real support
    (a band of pixels along the edge, not isolated codec/compression pixels).
    The theoretical solar circle is never used: the occulted side may leave the
    canvas while every visible pixel stays inside.
    """
    h, w = gray.shape
    if predicted is None or not np.isfinite(predicted).all():
        return {
            "clipped": False,
            "edges": [],
            "score": 0.0,
            "reason": "sin posición predicha para localizar el contenido solar",
        }
    search = 1.6 * radius
    x0 = max(0, int(math.floor(predicted[0] - search)))
    x1 = min(w, int(math.ceil(predicted[0] + search + 1)))
    y0 = max(0, int(math.floor(predicted[1] - search)))
    y1 = min(h, int(math.ceil(predicted[1] + search + 1)))
    thresholds = clipping_thresholds(gray)
    if not thresholds:
        return {
            "clipped": False,
            "edges": [],
            "score": 0.0,
            "reason": "sin contenido luminoso suficiente para evaluar el recorte",
        }
    kernel = max(2, int(round(2.0 * scale)))
    morph = np.ones((kernel, kernel), np.uint8)
    min_area = max(12.0, 0.01 * radius * radius)
    max_area = 1.35 * math.pi * radius * radius
    band = max(1, int(round(0.01 * max(w, h))))
    min_support = max(3, int(round(0.015 * min(w, h))))
    votes = {"left": 0, "right": 0, "top": 0, "bottom": 0}
    for threshold in thresholds:
        binary = (gray >= threshold).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph)
        yy, xx = np.ogrid[:h, :w]
        local_aperture = (xx - predicted[0]) ** 2 + (yy - predicted[1]) ** 2 <= (1.30 * radius) ** 2
        window = np.zeros((h, w), np.uint8)
        window[y0:y1, x0:x1] = 255
        window[~local_aperture] = 0
        masked = cv2.bitwise_and(binary, window)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(masked, 8)
        best_label = -1
        best_dist = float("inf")
        for label in range(1, num):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area or area > max_area:
                continue
            centroid = centroids[label]
            dist = float(
                np.hypot(centroid[0] - predicted[0], centroid[1] - predicted[1])
            )
            if dist < best_dist:
                best_dist = dist
                best_label = label
        if best_label < 0:
            continue
        ys, xs = np.nonzero(labels == best_label)
        for edge, (coord, limit, size) in (
            ("left", (xs, 0, w)),
            ("right", (xs, w - 1, w)),
            ("top", (ys, 0, h)),
            ("bottom", (ys, h - 1, h)),
        ):
            if edge in ("left", "top"):
                count = int((coord <= band).sum())
                core_count = int((coord <= 1).sum())
            else:
                count = int((coord >= limit - band).sum())
                core_count = int((coord >= limit - 1).sum())
            if count >= min_support and core_count >= max(2, min_support // 4):
                votes[edge] += 1
    agreement_needed = 2
    edges = sorted(edge for edge, count in votes.items() if count >= agreement_needed)
    clipped = len(edges) > 0
    score = max(votes.values()) / len(thresholds) if thresholds else 0.0
    if clipped:
        reason = "recorte real en " + ", ".join(edges)
    else:
        reason = "sin recorte con soporte real cerca de la posición predicha"
    return {
        "clipped": clipped,
        "edges": edges,
        "score": float(score),
        "reason": reason,
    }


def measure_exposed_fraction(
    gray: np.ndarray,
    predicted: np.ndarray | None,
    radius: float,
    scale: float = 1.0,
) -> dict:
    """Measure the exposed photosphere fraction inside the tracked solar disk.

    The measurement is normalized from each frame's own dark/bright levels, so
    exposure changes do not become shape changes.  Temporal comparison is done
    later, once neighboring frames and clipping/horizon state are known.
    """
    _ = scale
    if (
        predicted is None
        or not np.isfinite(predicted).all()
        or not np.isfinite(radius)
        or radius <= 0
    ):
        return {
            "evaluable": False,
            "reason": "sin modelo geométrico solar finito",
            "exposed_fraction": math.nan,
        }
    center = np.asarray(predicted, np.float64)
    h, w = gray.shape
    yy, xx = np.mgrid[:h, :w]
    radial = np.hypot(xx - center[0], yy - center[1])
    sample = radial <= EXPOSURE_SAMPLE_RADIUS_R * radius
    measure = radial <= EXPOSURE_MEASURE_RADIUS_R * radius
    values = gray[sample].astype(np.float64)
    if len(values) < max(30, int(0.25 * math.pi * radius * radius)):
        return {
            "evaluable": False,
            "reason": "disco de muestreo insuficiente dentro del lienzo",
            "exposed_fraction": math.nan,
        }

    bright_level = float(np.percentile(values, 90))
    annulus = (radial >= 1.03 * radius) & (radial <= 1.18 * radius)
    annulus_values = gray[annulus].astype(np.float64)
    interior_low = float(np.percentile(values, 10))
    annulus_low = (
        float(np.percentile(annulus_values, 20)) if len(annulus_values) else interior_low
    )
    dark_level = min(interior_low, annulus_low)
    dynamic_range = bright_level - dark_level
    # Photometric abstention gate: this is a brightness value, not a spatial
    # distance (never scaled).  The low-exposure abstention cannot change.
    exposure_gate = max(
        4.0 * MIN_BRIGHTNESS, EXPOSURE_MIN_LEVEL_FRACTION * 255.0
    )
    if (
        bright_level < exposure_gate
        or dynamic_range < EXPOSURE_VISIBLE_FRACTION * bright_level
    ):
        return {
            "evaluable": False,
            "reason": "exposición o contraste insuficiente para opinar",
            "bright_level": bright_level,
            "dark_level": dark_level,
            "dynamic_range": dynamic_range,
            "exposed_fraction": math.nan,
        }
    visible_threshold = dark_level + EXPOSURE_VISIBLE_FRACTION * dynamic_range
    exposed_fraction = float(np.mean(gray[measure] >= visible_threshold))
    return {
        "evaluable": True,
        "reason": f"fracción de fotosfera expuesta={exposed_fraction:.4f}",
        "exposed_fraction": exposed_fraction,
        "bright_level": bright_level,
        "dark_level": dark_level,
        "dynamic_range": dynamic_range,
        "visible_threshold": visible_threshold,
    }


def stable_visible_centroid(gray: np.ndarray) -> np.ndarray:
    """Centroid of the visible solar shape using one exposure-stable threshold."""
    maximum = float(gray.max())
    if maximum < MIN_BRIGHTNESS:
        return np.array([np.nan, np.nan])
    threshold = max(8.0, min(12.0, 0.35 * maximum))
    contour, _, _ = largest_contour(gray, threshold, scale=gray.shape[1] / 270.0)
    if contour is None:
        return np.array([np.nan, np.nan])
    moments = cv2.moments(contour)
    if abs(moments["m00"]) <= 1e-9:
        return np.array([np.nan, np.nan])
    return np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float64,
    )


def local_contrast_center(
    gray: np.ndarray, predicted: np.ndarray, radius: float, scale: float = 1.0
) -> tuple[np.ndarray | None, float]:
    """Locate the compact bright solar shape over a globally bright horizon.

    A local high-pass image removes the slowly varying landscape glow. Search
    is limited around the temporal prediction so poles, clouds and frame edges
    elsewhere cannot attract the tracker.
    """
    if predicted is None or not np.isfinite(predicted).all():
        return None, 0.0
    h, w = gray.shape
    search_radius = 1.35 * radius
    x0 = max(0, int(math.floor(predicted[0] - search_radius)))
    x1 = min(w, int(math.ceil(predicted[0] + search_radius + 1)))
    y0 = max(0, int(math.floor(predicted[1] - search_radius)))
    y1 = min(h, int(math.ceil(predicted[1] + search_radius + 1)))
    if x1 - x0 < 12 * scale or y1 - y0 < 12 * scale:
        return None, 0.0
    roi = gray[y0:y1, x0:x1].astype(np.float32)
    sigma = max(7.0 * scale, 0.18 * radius)
    background = cv2.GaussianBlur(roi, (0, 0), sigma)
    contrast = np.maximum(roi - background, 0.0)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    distance = np.hypot(xx - predicted[0], yy - predicted[1])
    local = distance <= search_radius
    positive = contrast[local]
    positive = positive[positive > 0]
    if len(positive) < 20:
        return None, 0.0
    threshold = max(1.0, float(np.percentile(positive, 72)), 0.10 * float(positive.max()))
    weights = np.maximum(contrast - threshold, 0.0)
    weights[~local] = 0.0
    # Suppress isolated one-pixel codec/noise peaks while preserving thin arcs.
    support = (weights > 0).astype(np.uint8)
    support = cv2.morphologyEx(support, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    weights *= support
    total = float(weights.sum())
    if total <= 1e-6 or int((weights > 0).sum()) < 12:
        return None, 0.0
    center = np.array(
        [float((weights * xx).sum() / total), float((weights * yy).sum() / total)]
    )
    if np.linalg.norm(center - predicted) > 0.95 * radius:
        return None, 0.0
    score = min(1.0, total / max(1.0, radius * radius * 0.8))
    return center, score


def kasa_circle(points: np.ndarray) -> tuple[np.ndarray | None, float]:
    if len(points) < 6:
        return None, float("nan")
    p = points.astype(np.float64)
    a = np.column_stack((2.0 * p[:, 0], 2.0 * p[:, 1], np.ones(len(p))))
    b = np.sum(p * p, axis=1)
    try:
        sol, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, float("nan")
    if rank < 3:
        return None, float("nan")
    center = sol[:2]
    radius_sq = sol[2] + np.dot(center, center)
    if radius_sq <= 0:
        return None, float("nan")
    return center, float(math.sqrt(radius_sq))


def outer_limb_points(contour: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Keep real contour pixels near its convex hull, not the hull's artificial chord."""
    hull = cv2.convexHull(contour, returnPoints=True)
    if hull is None or len(hull) < 5:
        return np.empty((0, 2), np.float64)
    line = np.zeros(shape, np.uint8)
    cv2.polylines(line, [hull], True, 255, 1, cv2.LINE_8)
    line = cv2.dilate(line, np.ones((3, 3), np.uint8))
    points = contour[:, 0, :]
    selected = points[line[points[:, 1], points[:, 0]] > 0]
    if len(selected) > 800:
        selected = selected[:: int(math.ceil(len(selected) / 800))]
    return selected.astype(np.float64)


def fit_fixed_radius(points: np.ndarray, radius: float, starts: list[np.ndarray]) -> tuple[np.ndarray | None, np.ndarray]:
    best_center: np.ndarray | None = None
    best_residual = np.empty(0)
    best_cost = float("inf")
    for initial in starts:
        if initial is None or not np.isfinite(initial).all():
            continue
        center = np.asarray(initial, np.float64).copy()
        for _ in range(20):
            vec = center[None, :] - points
            dist = np.linalg.norm(vec, axis=1)
            good = dist > 1e-6
            if good.sum() < 5:
                break
            residual = dist[good] - radius
            jac = vec[good] / dist[good, None]
            huber = max(0.7, radius * 0.012)
            weight = np.ones_like(residual)
            out = np.abs(residual) > huber
            weight[out] = huber / np.maximum(np.abs(residual[out]), 1e-9)
            aw = jac * np.sqrt(weight[:, None])
            bw = -residual * np.sqrt(weight)
            try:
                step, _, _, _ = np.linalg.lstsq(aw, bw, rcond=None)
            except np.linalg.LinAlgError:
                break
            center += step
            if np.linalg.norm(step) < 1e-4:
                break
        residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        cost = float(np.median(residual) + 0.35 * np.percentile(residual, 90))
        if np.isfinite(cost) and cost < best_cost:
            best_cost = cost
            best_center = center
            best_residual = residual
    return best_center, best_residual


def threshold_candidates(gray: np.ndarray) -> list[float]:
    maximum = float(gray.max())
    if maximum < MIN_BRIGHTNESS:
        return []
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    values = [8.0, 12.0, 18.0, 25.0, 0.12 * maximum, 0.22 * maximum, float(otsu)]
    return sorted({round(v, 1) for v in values if 5.0 <= v <= 0.9 * maximum})


def _largest_angular_gap(valid: np.ndarray, angles: np.ndarray) -> tuple[float, float]:
    """Largest contiguous invalid (unmeasured) arc and its midpoint orientation.

    Returns ``(gap_deg, gap_angle_deg)``.  ``gap_angle_deg`` is NaN when there is
    no gap (full coverage) or no valid sample at all.
    """
    n = len(valid)
    idx = np.nonzero(valid)[0]
    if len(idx) == 0:
        return 360.0, math.nan
    edges = np.r_[idx, idx[0] + n]
    diffs = np.diff(edges)
    gap_steps = int(diffs.max()) - 1
    gap_deg = gap_steps * 360.0 / n
    if gap_steps <= 0:
        return 0.0, math.nan
    j = int(np.argmax(diffs))
    mid_index = (edges[j] + edges[j + 1]) / 2.0
    return gap_deg, math.degrees(angles[int(mid_index) % n])


def measure_radial_limb(
    gray: np.ndarray, initial: np.ndarray, radius: float, scale: float = 1.0
) -> dict:
    """Measure the outer photospheric limb from threshold-independent radial drops.

    Around a predicted center, ``360`` angular directions are sampled in a narrow
    band around the fixed solar radius.  Each direction votes only when it shows a
    real outward intensity drop of sufficient contrast (a genuine outer-limb edge),
    so directions occulted by the Moon or without enough local contrast do not
    vote.  The valid samples are refined by fitting a fixed-radius circle, keeping
    the measurement independent of the crescent angle and free of the branch
    switching that binary thresholds introduce under changing exposure.

    The returned dict describes the measurement, never a propagated position:
    ``measured`` is True only when the frame itself supplied enough coherent
    samples (``>= MIN_RADIAL_POINTS``) and the fit converged close to the seed.
    A predicted center carried over without image evidence returns
    ``measured=False`` and is therefore not a measurement.

    All spatial values are in analysis-resolution pixels; angles are in degrees.
    """
    center = np.asarray(initial, np.float64).copy()
    angles = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    offsets = np.linspace(-6.0 * scale, 6.0 * scale, 49)
    valid_count = 0
    median_strength = 0.0
    final_residual = np.empty(0)
    last_valid = np.zeros(len(angles), dtype=bool)
    source = gray.astype(np.float32)
    for _ in range(3):
        radii = radius + offsets[:, None]
        map_x = (center[0] + radii * unit[None, :, 0]).astype(np.float32)
        map_y = (center[1] + radii * unit[None, :, 1]).astype(np.float32)
        profile = cv2.remap(
            source,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        # Positive when intensity drops while moving outwards.
        drop = profile[:-4] - profile[4:]
        best_idx = np.argmax(drop, axis=0)
        columns = np.arange(len(angles))
        strength = drop[best_idx, columns]
        edge_offset = offsets[best_idx + 2]
        inner_idx = int(np.argmin(np.abs(offsets + 3.5 * scale)))
        outer_idx = int(np.argmin(np.abs(offsets - 3.5 * scale)))
        inner = profile[inner_idx]
        outer = profile[outer_idx]
        brightness_gate = max(7.0, 0.055 * float(gray.max()))
        contrast_gate = max(3.0, 0.018 * float(gray.max()))
        valid = (
            (inner >= brightness_gate)
            & (strength >= contrast_gate)
            & ((inner - outer) >= contrast_gate)
            & (edge_offset > -5.8 * scale)
            & (edge_offset < 5.8 * scale)
        )
        if valid.sum() < MIN_RADIAL_POINTS:
            break
        points = center[None, :] + (radius + edge_offset[valid])[:, None] * unit[valid]
        refined, residual = fit_fixed_radius(points, radius, [center])
        if refined is None or np.linalg.norm(refined - center) > 5.0 * scale:
            break
        center = refined
        valid_count = int(valid.sum())
        median_strength = float(np.median(strength[valid]))
        final_residual = residual
        last_valid = valid.copy()
    measured = valid_count >= MIN_RADIAL_POINTS
    if len(final_residual):
        median_residual = float(np.median(final_residual))
        p90_residual = float(np.percentile(final_residual, 90))
    else:
        median_residual = float("inf")
        p90_residual = float("inf")
    gap_deg, gap_angle_deg = _largest_angular_gap(last_valid, angles)
    coverage_deg = max(0.0, 360.0 - gap_deg)
    # Conditioning of the fixed-radius fit: for a short arc the direction normal
    # to the arc bisector is poorly constrained.  We report the eigenvalue ratio
    # of the angular scatter of valid samples (1 for full coverage, >>1 when the
    # measured arc is short and the fit is ill-conditioned).
    n_valid = valid_count
    if n_valid >= 3:
        dirs = unit[last_valid]
        scatter = (dirs[:, :, None] * dirs[:, None, :]).sum(axis=0)
        eig = np.linalg.eigvalsh(scatter)
        condition = float(max(eig[1], 1e-9) / max(eig[0], 1e-9))
    else:
        condition = float("inf")
    return {
        "center": center,
        "measured": measured,
        "valid_points": n_valid,
        "coverage_deg": coverage_deg,
        "median_residual": median_residual,
        "p90_residual": p90_residual,
        "median_strength": median_strength,
        "largest_gap_deg": gap_deg,
        "gap_angle_deg": gap_angle_deg,
        "condition": condition,
        "arc_short": bool(n_valid > 0 and coverage_deg < 180.0),
    }


def refine_radial_limb(
    gray: np.ndarray, initial: np.ndarray, radius: float, scale: float = 1.0
) -> tuple[np.ndarray, int, float]:
    """Compatibility wrapper over :func:`measure_radial_limb`.

    Returns ``(center, valid_points, median_strength)`` and preserves the original
    public behavior for callers and validation that rely on the tuple form.
    """
    result = measure_radial_limb(gray, initial, radius, scale)
    return result["center"], result["valid_points"], result["median_strength"]


def radial_quality(
    coverage_deg: float, median_residual: float, count: int, radius: float
) -> float:
    """Quality score for a radial measurement, in the spirit of ``detect_limb``.

    Built only from radial-evidence values (angular coverage, fixed-radius fit
    residual and valid-sample count), so it is threshold independent.  Returns
    0.0 when the residual is not finite (no usable fit).
    """
    if not np.isfinite(median_residual):
        return 0.0
    coverage_score = np.clip(coverage_deg / 150.0, 0.0, 1.0)
    residual_score = math.exp(-median_residual / max(0.8, radius * 0.012))
    count_score = min(1.0, count / 100.0)
    return float(coverage_score * residual_score * (0.7 + 0.3 * count_score))


class ArcGeometryTracker:
    """Temporally-continuous arbitration of the per-frame geometric center.

    Advanced once per frame inside the analysis loop.  It forms the prediction
    from the last accepted geometric center plus a finite, response-gated,
    bounded phase step, measures the outer photospheric arc around that
    prediction as the primary source, and falls back to the binary contour as
    bootstrap / recovery.

    Arbitration (per frame):

    * radial measurement around the temporal prediction (primary) wins;
    * otherwise a contour candidate is accepted only when it is temporally /
      phase-consistent and there is real radial outer-edge evidence at its
      proposed center;
    * otherwise there is *no* measurement: the prediction is propagated but is
      never returned as a measurement (``raw_center`` stays NaN).

    All spatial gates are multiplied by ``scale`` so behavior is invariant to
    the analysis resolution.  A genuine large camera move is preserved: when the
    contour displacement agrees with phase and its center has radial support, it
    re-seeds the tracker instead of being suppressed or clamped away.

    Distances are analysis-resolution pixels; angles are degrees.
    """

    def __init__(
        self,
        radius: float,
        scale: float = 1.0,
        min_valid: int | None = None,
        min_coverage_deg: float = 120.0,
        max_residual_scale: float = 4.5,
        max_innovation_scale: float = 10.0,
        response_gate: float = 0.12,
        motion_cap_scale: float = 35.0,
        contour_phase_tol_scale: float = 10.0,
        contour_local_tol_scale: float = 6.0,
        disagree_tol_scale: float = 4.0,
    ) -> None:
        self.radius = float(radius)
        self.scale = float(scale)
        self.min_valid = int(min_valid) if min_valid is not None else int(MIN_RADIAL_POINTS)
        self.min_coverage_deg = float(min_coverage_deg)
        self.max_residual = float(max_residual_scale) * self.scale
        self.max_innovation = float(max_innovation_scale) * self.scale
        self.response_gate = float(response_gate)
        self.motion_cap = float(motion_cap_scale) * self.scale
        self.contour_phase_tol = float(contour_phase_tol_scale) * self.scale
        self.contour_local_tol = float(contour_local_tol_scale) * self.scale
        self.disagree_tol = float(disagree_tol_scale) * self.scale
        self.last_accepted: np.ndarray | None = None

    def predict(self, relative: np.ndarray, response: float) -> np.ndarray | None:
        """Temporal prediction = last accepted center + validated phase step."""
        if self.last_accepted is None:
            return None
        predicted = self.last_accepted.copy()
        if (
            np.isfinite(relative).all()
            and response >= self.response_gate
            and np.linalg.norm(relative) <= self.motion_cap
        ):
            predicted = predicted + relative
        return predicted

    def _contour_consistent(
        self,
        contour: np.ndarray | None,
        predicted: np.ndarray | None,
        relative: np.ndarray,
        response: float,
    ) -> bool:
        if contour is None or not np.isfinite(contour).all():
            return False
        # Phase available and valid: the contour displacement must agree with it.
        if (
            self.last_accepted is not None
            and np.isfinite(relative).all()
            and response >= self.response_gate
        ):
            displacement = contour - self.last_accepted
            return bool(np.linalg.norm(displacement - relative) <= self.contour_phase_tol)
        if predicted is not None and np.isfinite(predicted).all():
            return bool(np.linalg.norm(contour - predicted) <= self.contour_local_tol)
        if self.last_accepted is not None:
            return bool(np.linalg.norm(contour - self.last_accepted) <= self.contour_local_tol)
        return True

    def step(
        self,
        gray: np.ndarray,
        relative: np.ndarray,
        response: float,
        contour: np.ndarray | None,
        contour_touch: bool = False,
    ) -> dict:
        """Advance one frame and return the arbitration result."""
        predicted = self.predict(relative, response)
        if predicted is None:
            # Bootstrap seed, local only: it is never persisted as accepted
            # state.  ``last_accepted`` is updated only after an accepted
            # radial / confirmed-contour measurement, so an unmeasured
            # frame-center seed cannot become a persistent anchor.
            if contour is not None and np.isfinite(contour).all():
                predicted = contour.astype(np.float64).copy()
            else:
                predicted = np.array([gray.shape[1] / 2.0, gray.shape[0] / 2.0])

        attempted = measure_radial_limb(gray, predicted, self.radius, self.scale)
        radial_ok = (
            attempted["measured"]
            and attempted["valid_points"] >= self.min_valid
            and attempted["coverage_deg"] >= self.min_coverage_deg
            and np.isfinite(attempted["median_residual"])
            and attempted["median_residual"] <= self.max_residual
            and np.linalg.norm(attempted["center"] - predicted) <= self.max_innovation
        )
        contour_finite = contour is not None and np.isfinite(contour).all()
        contour_consistent = self._contour_consistent(
            contour, predicted, relative, response
        )
        confirming = None
        contour_ok = False
        if contour_finite and contour_consistent and not contour_touch:
            # A contour recovery must be confirmed by real radial outer-edge
            # evidence at its proposed center (a self-consistent-but-wrong
            # circle, e.g. a corrupt frame or a landscape arc, cannot re-seed).
            confirming = measure_radial_limb(
                gray, contour.astype(np.float64), self.radius, self.scale
            )
            contour_ok = (
                confirming["measured"]
                and confirming["valid_points"] >= self.min_valid
                and confirming["coverage_deg"] >= self.min_coverage_deg
                and np.isfinite(confirming["median_residual"])
                and confirming["median_residual"] <= self.max_residual
            )

        center: np.ndarray | None = None
        source = SRC_NONE
        # ``arc`` holds the evidence that justified the selected center: the
        # primary radial attempt for a radial source, or the confirming radial
        # measurement at the contour center for a confirmed contour recovery.
        arc = attempted
        if radial_ok:
            center = attempted["center"]
            if contour_finite and np.linalg.norm(contour - center) > self.disagree_tol:
                source = SRC_RADIAL_DISAGREE
            else:
                source = SRC_RADIAL
        elif contour_ok:
            center = contour.astype(np.float64).copy()
            source = SRC_CONTOUR
            arc = confirming

        accepted = center is not None
        innovation = (
            float(np.linalg.norm(center - predicted)) if accepted else float("nan")
        )
        if accepted:
            self.last_accepted = center.copy()
        return {
            "center": center,
            "predicted": predicted,
            "source": source,
            "accepted": accepted,
            "innovation": innovation,
            "arc": arc,
            # Diagnostic only: the radial attempt around the temporal prediction.
            # For a confirmed contour recovery this is the *failed* attempt, kept
            # separately and never reported as the selected arc evidence.
            "attempted_arc": attempted,
            "radial_ok": radial_ok,
            "contour_consistent": contour_consistent,
            "contour_ok": contour_ok,
        }


def detect_limb(
    gray: np.ndarray, radius: float, predicted: np.ndarray | None = None, scale: float = 1.0
) -> dict:
    h, w = gray.shape
    best: dict | None = None
    for threshold in threshold_candidates(gray):
        contour, _, touch = largest_contour(gray, threshold, scale=scale)
        if contour is None:
            continue
        points = outer_limb_points(contour, gray.shape)
        if len(points) < 12:
            continue
        algebraic_center, algebraic_radius = kasa_circle(points)
        starts = [algebraic_center]
        if predicted is not None:
            starts.insert(0, predicted)
        center, residual = fit_fixed_radius(points, radius, starts)
        if center is None or len(residual) == 0:
            continue
        if not (-radius <= center[0] <= w + radius and -radius <= center[1] <= h + radius):
            continue
        angles = np.sort(np.mod(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]), 2 * np.pi))
        gaps = np.diff(np.r_[angles, angles[0] + 2 * np.pi])
        coverage = float(2 * np.pi - gaps.max()) if len(gaps) else 0.0
        median = float(np.median(residual))
        p90 = float(np.percentile(residual, 90))
        inlier = float(np.mean(residual <= max(1.25, radius * 0.018)))
        coverage_score = np.clip(coverage / math.radians(150.0), 0.0, 1.0)
        residual_score = math.exp(-median / max(0.8, radius * 0.012))
        count_score = min(1.0, len(points) / 100.0)
        quality = float(coverage_score * residual_score * (0.45 + 0.55 * inlier) * (0.7 + 0.3 * count_score))
        if touch:
            quality *= 0.55
        candidate = {
            "center": center,
            "quality": quality,
            "threshold": threshold,
            "touch": touch,
            "coverage_deg": math.degrees(coverage),
            "median_residual": median,
            "p90_residual": p90,
            "inlier": inlier,
            "points": len(points),
            "algebraic_radius": algebraic_radius,
        }
        moments = cv2.moments(contour)
        if abs(moments["m00"]) > 1e-9:
            candidate["visible_center"] = np.array(
                [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
                dtype=np.float64,
            )
        else:
            candidate["visible_center"] = np.array([np.nan, np.nan])
        if best is None or candidate["quality"] > best["quality"]:
            best = candidate
    if best is None:
        return {
            "center": np.array([np.nan, np.nan]),
            "quality": 0.0,
            "threshold": 0.0,
            "touch": False,
            "coverage_deg": 0.0,
            "median_residual": float("inf"),
            "p90_residual": float("inf"),
            "inlier": 0.0,
            "points": 0,
            "algebraic_radius": float("nan"),
            "radial_points": 0,
            "radial_strength": 0.0,
            "visible_center": np.array([np.nan, np.nan]),
        }
    refined, radial_points, radial_strength = refine_radial_limb(gray, best["center"], radius, scale)
    if radial_points >= 35 and np.linalg.norm(refined - best["center"]) <= 5.0 * scale:
        best["center"] = refined
        # A broad atmospheric edge lowers binary-fit quality, but many coherent
        # radial samples are still useful evidence.
        radial_score = min(1.0, radial_points / 180.0) * min(1.0, radial_strength / 12.0)
        best["quality"] = max(best["quality"], 0.35 * radial_score)
    best["radial_points"] = radial_points
    best["radial_strength"] = radial_strength
    return best


def calibrate_radius(frames: list[tuple[int, np.ndarray]]) -> float:
    estimates: list[float] = []
    for _, bgr in frames:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        for threshold in (12.0, 18.0, 25.0):
            contour, _, touch = largest_contour(gray, threshold, scale=gray.shape[1] / 270.0)
            if contour is None or touch or len(contour) < 30:
                continue
            points = outer_limb_points(contour, gray.shape)
            _, radius = kasa_circle(points)
            if np.isfinite(radius) and 0.20 * gray.shape[1] < radius < 0.48 * gray.shape[1]:
                estimates.append(radius)
    if len(estimates) < 4:
        raise SystemExit("No se pudo calibrar el radio solar con los fotogramas iniciales.")
    values = np.asarray(estimates)
    median = float(np.median(values))
    good = values[np.abs(values - median) <= max(2.0, 0.04 * median)]
    radius = float(np.median(good)) if len(good) else median
    print(f"Radio solar calibrado: {radius:.3f} px de análisis (n={len(good)}/{len(values)})")
    return radius
