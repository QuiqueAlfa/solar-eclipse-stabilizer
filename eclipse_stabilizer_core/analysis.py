from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from .constants import (
    CACHE_VERSION,
    CLIP_VERSION,
    CONTRAST_VERSION,
    MIN_RADIAL_POINTS,
    SRC_CONTOUR,
    SRC_NONE,
    TEAR_VERSION,
    VISIBLE_VERSION,
)
from .geometry import (
    ArcGeometryTracker,
    analysis_scale,
    calibrate_radius,
    classify_clipping,
    detect_limb,
    edges_to_bitmask,
    local_contrast_center,
    measure_exposed_fraction,
    radial_quality,
    stable_visible_centroid,
)
from .tracking import (
    classify_regime,
    solve_tracking,
    transient_outliers,
    write_tracking_csv,
)
from .video import Progress, VideoInfo, iter_ffmpeg_gray, scaled_shape, sparse_frames


CACHE_REQUIRED_ARRAYS = frozenset(
    {
        "raw_center",
        "quality",
        "coverage",
        "median_residual",
        "threshold",
        "touch",
        "radial_points",
        "radial_strength",
        "visible_center",
        "relative",
        "response",
        "maximum",
        "radius",
        "radius_meas",
        "clip_edges",
        "clip_score",
        "tear_evaluable",
        "tear_bright_level",
        "tear_visible_threshold",
        "tear_exposed_fraction",
        "tear_reason",
        "analysis_width",
        "analysis_height",
        "source_frames",
        "source_fps",
        "contrast_center",
        "contrast_score",
        "arc_center",
        "arc_measured",
        "arc_valid_points",
        "arc_coverage",
        "arc_median_residual",
        "arc_strength",
        "arc_gap_deg",
        "arc_gap_angle",
        "geometry_source",
        "geometry_prediction",
        "geometry_innovation",
        "contrast_dynamic_offset",
        "contrast_offset_sample",
        "fallback_reanchored",
        "fallback_supported",
        "fallback_innovation",
        "fallback_mode",
    }
)


def calibration_frames(video: Path, info: VideoInfo, width: int) -> list[tuple[int, np.ndarray]]:
    end = info.frames - 1
    indices = np.unique(np.linspace(0, end, 16, dtype=int))
    print("Calibración: 16 seeks dispersos por toda la secuencia (sin recorrer todo el vídeo).")
    return sparse_frames(video, indices, width)


def resolve_radius(args: argparse.Namespace, video: Path, info: VideoInfo, width: int) -> float:
    """Use --radius when given (analysis units), otherwise calibrate from
    samples dispersed across the whole sequence."""
    if args.radius is not None:
        print(f"Radio solar fijado por --radius: {args.radius:.3f} px de análisis")
        return float(args.radius)
    calib = calibration_frames(video, info, width)
    return calibrate_radius(calib)


def phase_image(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32)
    return small - cv2.GaussianBlur(small, (0, 0), 2.0)


def cache_path(out_dir: Path) -> Path:
    return out_dir / "analysis.npz"


def profile_hash(profile: dict | None) -> str:
    """Stable canonical hash of a loaded profile dict; empty marker when absent."""
    if profile is None:
        return ""
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_cache_identity(
    video: Path,
    info: VideoInfo,
    analysis_width: int,
    analysis_height: int,
    radius_requested: str,
    profile: dict | None,
    profile_path: str | None,
    min_quality: float,
) -> dict:
    """Serializable identity of the expensive analysis configuration.

    Every value here changes the detected trajectory (source file, probe
    properties, analysis size, requested radius and profile) or pins the
    algorithm versions.  ``--no-auto-repair`` intentionally does not appear:
    it only changes the cheap final solve and must re-resolve without
    invalidating the cached detections.
    """
    video_path = Path(video).expanduser().resolve()
    stat = video_path.stat()
    resolved_profile = Path(profile_path).expanduser().resolve() if profile_path else ""
    return {
        "version": int(CACHE_VERSION),
        "source_path": str(video_path),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
        "width": int(info.width),
        "height": int(info.height),
        "fps": float(info.fps),
        "frames": int(info.frames),
        "frame_count_exact": bool(info.frame_count_exact),
        "analysis_width": int(analysis_width),
        "analysis_height": int(analysis_height),
        "radius_requested": str(radius_requested),
        "profile_path": str(resolved_profile),
        "profile_hash": profile_hash(profile),
        "min_quality": float(min_quality),
        "visible_version": int(VISIBLE_VERSION),
        "contrast_version": int(CONTRAST_VERSION),
        "clip_version": int(CLIP_VERSION),
        "tear_version": int(TEAR_VERSION),
    }


def identity_arrays(identity: dict) -> dict[str, np.ndarray]:
    """Pack an identity into numpy scalar/Unicode arrays (allow_pickle=False)."""
    arrays: dict[str, np.ndarray] = {}
    for key, value in identity.items():
        if isinstance(value, str):
            arrays[f"id_{key}"] = np.array([value], dtype=np.str_)
        else:
            arrays[f"id_{key}"] = np.array([value])
    return arrays


def identity_from_arrays(arrays: dict[str, np.ndarray]) -> dict:
    """Reconstruct the identity fields stored under the ``id_`` prefix."""
    identity: dict = {}
    for key, value in arrays.items():
        if not key.startswith("id_"):
            continue
        name = key[3:]
        flat = np.asarray(value).reshape(-1)
        identity[name] = flat[0].item() if len(flat) else None
    return identity


def cache_mismatch_reasons(loaded: dict, expected: dict) -> list[str]:
    """Compare every stored identity field against the expected one.

    Returns a human-readable list of reasons; an empty list means the cached
    analysis is fully compatible and can be reused as-is.
    """
    loaded_id = identity_from_arrays(loaded)
    if not loaded_id:
        return ["caché legacy sin metadatos de identidad (campos id_*)"]
    reasons: list[str] = []
    for field, wanted in expected.items():
        if field not in loaded_id:
            reasons.append(f"falta el campo de identidad '{field}'")
            continue
        if loaded_id[field] != wanted:
            reasons.append(f"'{field}' difiere: caché={loaded_id[field]!r}, esperado={wanted!r}")
    return reasons


def cache_structure_reasons(keys: set[str]) -> list[str]:
    missing = sorted(CACHE_REQUIRED_ARRAYS - keys)
    if not missing:
        return []
    shown = ", ".join(missing[:6])
    suffix = "..." if len(missing) > 6 else ""
    return [f"caché incompleta; faltan arrays: {shown}{suffix}"]


def _classify_cache_state(
    arrays: dict[str, np.ndarray], keys: set[str], expected: dict
) -> tuple[str, list[str]]:
    """Classify loaded cache identity arrays + structure against ``expected``.

    ``arrays`` carries the stored arrays (identity fields are read under the
    ``id_`` prefix) and ``keys`` the full stored array name set.  Returns a
    compact ``(state, reasons)`` pair where state is ``valid`` (exact match),
    ``refreshable`` (complete structure and only ``contrast_version`` differs)
    or ``incompatible`` (hard identity or structure conflict).
    """
    structure = cache_structure_reasons(keys)
    reasons = cache_mismatch_reasons(arrays, expected) + structure
    if not reasons:
        return "valid", []
    if not structure:
        loaded_id = identity_from_arrays(arrays)
        if loaded_id and all(
            field == "contrast_version" or (field in loaded_id and loaded_id[field] == wanted)
            for field, wanted in expected.items()
        ):
            return "refreshable", reasons
    return "incompatible", reasons


def cache_status(path: Path, expected: dict) -> tuple[str, list[str]]:
    """Cheap initial cache status for the pre-command banner.

    Only the small ``id_*`` arrays are decoded, so this never duplicates the
    expensive analysis just to report the state.
    """
    if not path.exists():
        return "missing", []
    try:
        with np.load(path, allow_pickle=False) as loaded:
            keys = set(loaded.files)
            arrays = {key: loaded[key].copy() for key in loaded.files if key.startswith("id_")}
    except Exception as exc:
        return "incompatible", [f"caché ilegible: {exc}"]
    return _classify_cache_state(arrays, keys, expected)


def save_analysis_cache(path: Path, analysis: dict) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **analysis)
    os.replace(temporary, path)


def _robust_median(history: deque) -> np.ndarray | None:
    """Robust median of a deque of (2,) offset vectors; None when empty."""
    arr = np.asarray(list(history), np.float64)
    if len(arr) == 0:
        return None
    return np.median(arr, axis=0)


def _robust_spread(history: deque) -> float:
    """Robust MAD-based spread of a deque of offset vectors; 0 when tiny."""
    arr = np.asarray(list(history), np.float64)
    if len(arr) < 3:
        return 0.0
    med = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - med), axis=0)
    return float(np.median(mad)) * 1.4826


class AnchoredContrastFallback:
    """Continuously-anchored local-contrast fallback track (Phase 3/4).

    Replaces the old one-time fixed offset / long causal chain.  A frame with
    explicit *trusted* geometry evidence (``geometry_source != SRC_NONE`` with a
    finite ``raw_center`` *and* ``reanchor=True``) re-anchors the fallback to
    the geometric center and, when the local contrast is measurable, feeds a
    robust rolling history of ``raw_center - local_contrast_center``.  Frames
    without geometry, or whose geometry is vetoed (transient/untrusted loss or a
    post-horizon false circle), propagate the last fallback center with finite /
    high-response / bounded relative motion, detect local contrast near the
    prediction, correct toward ``detected + recent_offset`` and robustly cap
    innovation so a single bad detection cannot yank the path.

    All temporal windows derive from ``source_fps`` and all spatial gates from
    ``scale``/``radius``; no source frame, time or profile is hardcoded.  The
    re-anchor criterion is explicit *accepted trusted* geometry evidence
    (``reanchor_mask``); without a mask the legacy rule (any
    ``geometry_source != SRC_NONE``) is kept so old dicts keep their behavior.
    """

    def __init__(
        self,
        radius: float,
        scale: float = 1.0,
        fps: float = 30.0,
        offset_window_seconds: float = 2.0,
        min_offset_samples: int = 6,
        alpha_base: float = 0.55,
        alpha_conf: float = 0.35,
        innovation_cap_scale: float = 20.0,
        innovation_min_scale: float = 2.5,
        innovation_spread_mult: float = 4.0,
        response_gate: float = 0.12,
        motion_cap_scale: float = 35.0,
        detect_score_gate: float = 0.02,
    ) -> None:
        self.radius = float(radius)
        self.scale = float(scale)
        self.window = max(min_offset_samples, int(round(offset_window_seconds * fps)))
        self.min_offset_samples = int(min_offset_samples)
        self.alpha_base = float(alpha_base)
        self.alpha_conf = float(alpha_conf)
        self.innovation_cap = float(innovation_cap_scale) * scale
        self.innovation_min = float(innovation_min_scale) * scale
        self.innovation_spread_mult = float(innovation_spread_mult)
        self.response_gate = float(response_gate)
        self.motion_cap = float(motion_cap_scale) * scale
        self.detect_score_gate = float(detect_score_gate)
        self.offset_history: deque = deque(maxlen=self.window)
        self.recent_offset: np.ndarray | None = None
        self.previous: np.ndarray | None = None
        self.innovation_limit = max(self.innovation_min, 4.0 * scale)

    def step(
        self,
        gray: np.ndarray,
        geometry_center: np.ndarray | None,
        relative: np.ndarray,
        response: float,
        reanchor: bool = True,
    ) -> dict:
        """Advance one frame; returns per-frame track, score and diagnostics.

        ``dynamic_offset`` is the *robust recent offset actually applied* on
        this frame (``None`` until history exists).  ``offset_sample`` is the
        optional raw per-frame ``geometry_center - contrast_center`` sample on
        a re-anchor frame (``None`` otherwise) and is kept only as audit.
        ``supported`` is True once the fallback has a real geometric anchor
        (``self.previous`` finite); before that no finite center is invented.

        ``reanchor=False`` vetoes a re-anchor even when a geometry center is
        present (transient/untrusted loss, post-horizon false circle): the frame
        then follows the propagate branch and never anchors to that geometry.
        """
        measured = geometry_center is not None and np.isfinite(geometry_center).all()
        if measured and reanchor:
            detected, confidence = local_contrast_center(
                gray, geometry_center, self.radius, self.scale
            )
            offset_sample: np.ndarray | None = None
            if detected is not None and confidence > self.detect_score_gate:
                offset_sample = geometry_center - detected
                self.offset_history.append(offset_sample)
                self.recent_offset = _robust_median(self.offset_history)
                if len(self.offset_history) >= self.min_offset_samples:
                    self.innovation_limit = max(
                        self.innovation_min,
                        min(
                            self.innovation_cap,
                            self.innovation_spread_mult * _robust_spread(self.offset_history),
                        ),
                    )
            current = geometry_center.astype(np.float64).copy()
            self.previous = current.copy()
            return {
                "center": current,
                "score": confidence if detected is not None else 0.0,
                "reanchored": True,
                "dynamic_offset": (
                    self.recent_offset.copy() if self.recent_offset is not None else None
                ),
                "offset_sample": offset_sample,
                "innovation": 0.0,
                "mode": 0,
                "supported": True,
            }
        # No geometry: propagate with validated relative motion and correct
        # toward the local contrast plus the recent dynamic offset.
        supported = self.previous is not None
        delta = np.asarray(relative, np.float64)
        if (
            not np.isfinite(delta).all()
            or float(np.linalg.norm(delta)) > self.motion_cap
            or float(response) < self.response_gate
        ):
            delta = np.zeros(2)
        if supported:
            predicted = self.previous + delta
        else:
            # No real geometric anchor yet: never invent a finite center.
            predicted = np.array([gray.shape[1] / 2.0, gray.shape[0] / 2.0])
        detected, confidence = local_contrast_center(
            gray, predicted, self.radius, self.scale
        )
        offset = self.recent_offset if self.recent_offset is not None else np.zeros(2)
        innovation_len = 0.0
        if detected is not None and confidence > self.detect_score_gate:
            adjusted = detected + offset
            innovation = adjusted - predicted
            length = float(np.linalg.norm(innovation))
            if length > self.innovation_limit:
                innovation = innovation * (self.innovation_limit / max(length, 1e-9))
                innovation_len = self.innovation_limit
            else:
                innovation_len = length
            alpha = self.alpha_base + self.alpha_conf * confidence
            current = predicted + alpha * innovation
        else:
            current = predicted
        if supported:
            current[0] = float(np.clip(current[0], -0.25 * self.radius, gray.shape[1] + 0.25 * self.radius))
            current[1] = float(np.clip(current[1], -0.25 * self.radius, gray.shape[0] + 0.25 * self.radius))
            self.previous = current.copy()
        else:
            current = np.full(2, np.nan, np.float64)
        return {
            "center": current,
            "score": confidence if (detected is not None and supported) else 0.0,
            "reanchored": False,
            "dynamic_offset": (
                self.recent_offset.copy() if self.recent_offset is not None else None
            ),
            "offset_sample": None,
            "innovation": innovation_len,
            "mode": 1,
            "supported": supported,
        }


def _geometry_trusted_mask(analysis: dict, scale: float) -> np.ndarray:
    """Accepted-trusted geometry mask used to re-anchor the contrast fallback.

    Mirrors ``solve_tracking``'s anchor semantics: circular anchors are only the
    pre-latch limbo detections, cleaned by the transient-outlier gate.  Frames
    whose geometry is transient/untrusted (a soft ramp) or a post-horizon false
    circle are excluded, so the fallback never anchors to them.
    """
    regime_info = classify_regime(analysis, scale=scale)
    frame_index = np.arange(len(analysis["quality"]))
    geometry_usable = (
        regime_info["limbo_evidence"]
        & ~regime_info["false_circle_after_horizon"]
        & (frame_index < int(regime_info["horizon_start"]))
    )
    return transient_outliers(
        analysis["raw_center"],
        analysis["relative"],
        geometry_usable,
        gate=4.0 * scale,
    )


def refresh_contrast_track_frames(
    analysis: dict, gray_frames, reanchor_mask: np.ndarray | None = None
) -> dict:
    """Compute the anchored contrast fallback over an iterable of gray frames.

    ``gray_frames`` yields one analysis-resolution grayscale frame per source
    frame (``idx`` from 0 to ``source_frames - 1``).  Every explicit *trusted*
    geometric measurement re-anchors the fallback; other frames (or vetoed
    geometry when ``reanchor_mask[idx]`` is False) propagate.  ``reanchor_mask``
    is the accepted-trusted geometry mask (limbo evidence before the horizon
    latch); when ``None`` the legacy rule (any ``geometry_source != SRC_NONE``)
    is kept.  The per-frame diagnostics (dynamic offset, re-anchor flags,
    innovation and mode) are returned for audit.  The legacy ``contrast_offset``
    vector is kept as the final/recent robust offset for compatibility.
    """
    width = int(analysis["analysis_width"][0])
    height = int(analysis["analysis_height"][0])
    frames = int(analysis["source_frames"][0])
    radius = float(analysis["radius"][0])
    scale = analysis_scale(width)
    fps = float(analysis["source_fps"][0])
    raw = analysis["raw_center"]
    relative = analysis["relative"]
    response = analysis["response"]
    geometry_source = analysis["geometry_source"]

    fallback = AnchoredContrastFallback(radius, scale=scale, fps=fps)
    track = np.full((frames, 2), np.nan, np.float64)
    score = np.zeros(frames, np.float64)
    dynamic_offset = np.full((frames, 2), np.nan, np.float64)
    offset_sample = np.full((frames, 2), np.nan, np.float64)
    reanchored = np.zeros(frames, bool)
    supported = np.zeros(frames, bool)
    innovation = np.zeros(frames, np.float64)
    mode = np.zeros(frames, np.int8)
    if reanchor_mask is not None:
        reanchor_mask = np.asarray(reanchor_mask, bool)
    decoded = 0
    for idx, gray in enumerate(gray_frames):
        measured = geometry_source[idx] != SRC_NONE and np.isfinite(raw[idx]).all()
        geometry_center = raw[idx] if measured else None
        allow = reanchor_mask[idx] if reanchor_mask is not None else True
        result = fallback.step(
            gray, geometry_center, relative[idx], float(response[idx]), reanchor=measured and allow
        )
        track[idx] = result["center"]
        score[idx] = result["score"]
        if result["dynamic_offset"] is not None:
            dynamic_offset[idx] = result["dynamic_offset"]
        if result["offset_sample"] is not None:
            offset_sample[idx] = result["offset_sample"]
        reanchored[idx] = result["reanchored"]
        supported[idx] = result["supported"]
        innovation[idx] = result["innovation"]
        mode[idx] = result["mode"]
        decoded = idx + 1
    if decoded != frames:
        raise SystemExit(f"Actualización del contraste incompleta: {decoded}/{frames}")
    recent_offset = fallback.recent_offset if fallback.recent_offset is not None else np.zeros(2)
    return {
        "contrast_center": track,
        "contrast_score": score,
        "contrast_dynamic_offset": dynamic_offset,
        "contrast_offset_sample": offset_sample,
        "fallback_reanchored": reanchored,
        "fallback_supported": supported,
        "fallback_innovation": innovation,
        "fallback_mode": mode,
        "contrast_offset": recent_offset,
    }


def refresh_contrast_track(video: Path, analysis: dict, reanchor_mask: np.ndarray | None = None) -> None:
    width = int(analysis["analysis_width"][0])
    height = int(analysis["analysis_height"][0])
    frames = int(analysis["source_frames"][0])
    progress = Progress("rastreador contraste", frames)

    def frames_with_progress():
        for idx, gray in iter_ffmpeg_gray(video, width, height, frames, exact_total=True):
            progress.update(idx + 1, force=idx + 1 == frames)
            yield gray

    analysis.update(refresh_contrast_track_frames(analysis, frames_with_progress(), reanchor_mask))
    analysis["contrast_version"] = np.array([CONTRAST_VERSION], np.int16)


def warn_scale_drift(
    radius_meas: np.ndarray,
    radial_points: np.ndarray,
    fixed_radius: float,
    fps: float,
    tolerance: float = 0.08,
) -> None:
    """Warn when per-frame radius measurements suggest an incompatible scale change.

    The video is assumed to keep a constant zoom, so the solar radius is fixed.
    A sustained deviation of the measured outer-limb radius from the fixed value
    means the assumption may be broken and the centered export would drift.
    """
    n = len(radius_meas)
    valid = np.isfinite(radius_meas) & (radial_points >= MIN_RADIAL_POINTS)
    if valid.sum() < 20 or fixed_radius <= 0:
        return
    window = max(1, int(round(0.5 * fps)))
    frame_index = np.arange(n)
    groups: list[tuple[int, int, float]] = []
    for start in range(0, n, window):
        end = min(n, start + window)
        idx = frame_index[start:end][valid[start:end]]
        if len(idx) < 5:
            continue
        groups.append((start, end - 1, float(np.median(radius_meas[idx]))))
    outliers = [(a, b, med) for a, b, med in groups if abs(med / fixed_radius - 1.0) > tolerance]
    if not outliers:
        return
    print("Aviso de escala:")
    for a, b, med in outliers:
        print(
            f"  frames {a}-{b}: radio medido {med:.1f}px frente al fijo {fixed_radius:.1f}px "
            f"({(med / fixed_radius - 1.0) * 100:+.1f}%). "
            "Posible cambio de zoom incompatible con el radio fijo."
        )


def analyze_video(args: argparse.Namespace, video: Path, info: VideoInfo, out_dir: Path, force: bool = False) -> dict:
    path = cache_path(out_dir)
    analysis_w, analysis_h = scaled_shape(info, args.analysis_width)
    radius_requested = "auto" if args.radius is None else f"{float(args.radius):.6g}"
    profile = getattr(args, "profile", None)
    profile_path = getattr(args, "profile_path", None)
    expected = build_cache_identity(
        video, info, analysis_w, analysis_h, radius_requested, profile, profile_path, args.min_quality
    )

    arrays: dict[str, np.ndarray] | None = None
    if path.exists() and not (force or args.force):
        try:
            with np.load(path, allow_pickle=False) as loaded:
                arrays = {key: loaded[key].copy() for key in loaded.files}
        except Exception as exc:
            print(f"Caché ilegible ({path}): {exc}. Regenerando análisis completo.")
            arrays = None
        if arrays is not None:
            # A ``contrast_version`` mismatch is recoverable: the geometric
            # detections are compatible, so the anchored fallback is recomputed
            # over them on load instead of discarding the whole analysis.  Any
            # other identity or structure mismatch forces full re-analysis.
            state, reasons = _classify_cache_state(arrays, set(arrays), expected)
            if state == "incompatible":
                print("Caché rechazada: " + "; ".join(reasons) + ". Regenerando análisis completo.")
                arrays = None
    elif path.exists():
        print("Caché ignorada por --force; regenerando análisis completo.")

    if arrays is not None:
        analysis = arrays
        print(f"Cargando análisis cacheado: {path}")
        loaded_contrast = int(identity_from_arrays(analysis).get("contrast_version", -1))
        if loaded_contrast != int(expected["contrast_version"]):
            print("Caché geométrica compatible; recalculando el respaldo de contraste anclado.")
            reanchor = _geometry_trusted_mask(
                analysis, analysis_scale(int(analysis["analysis_width"][0]))
            )
            refresh_contrast_track(video, analysis, reanchor_mask=reanchor)
        if "radius_meas" in analysis:
            warn_scale_drift(
                analysis["radius_meas"],
                analysis["radial_points"],
                float(analysis["radius"][0]),
                float(analysis["source_fps"][0]),
            )
        # Re-solving is cheap and lets quality/repair improvements reuse the
        # expensive frame detections and phase correlations.
        analysis.update(
            solve_tracking(
                analysis,
                args.min_quality,
                profile,
                auto_repair=not getattr(args, "no_auto_repair", False),
            )
        )
        analysis.update(identity_arrays(expected))
        save_analysis_cache(path, analysis)
        write_tracking_csv(out_dir / "tracking.csv", analysis)
        return analysis

    radius = resolve_radius(args, video, info, analysis_w)
    scale = analysis_scale(analysis_w)
    n = info.frames
    raw_center = np.full((n, 2), np.nan, np.float64)
    quality = np.zeros(n, np.float64)
    coverage = np.zeros(n, np.float64)
    median_residual = np.full(n, np.inf, np.float64)
    threshold = np.zeros(n, np.float64)
    touch = np.zeros(n, bool)
    radial_points = np.zeros(n, np.int16)
    radial_strength = np.zeros(n, np.float64)
    visible_center = np.full((n, 2), np.nan, np.float64)
    relative = np.zeros((n, 2), np.float64)
    response = np.zeros(n, np.float64)
    maximum = np.zeros(n, np.uint8)
    radius_meas = np.full(n, np.nan, np.float64)
    clip_edges = np.zeros(n, np.int8)
    clip_score = np.zeros(n, np.float64)
    tear_evaluable = np.zeros(n, bool)
    tear_bright_level = np.full(n, np.nan, np.float32)
    tear_visible_threshold = np.full(n, np.nan, np.float32)
    tear_exposed_fraction = np.full(n, np.nan, np.float32)
    tear_reason = np.full(n, "sin evaluar", dtype="U128")
    arc_center = np.full((n, 2), np.nan, np.float32)
    arc_measured = np.zeros(n, bool)
    arc_valid_points = np.zeros(n, np.int16)
    arc_coverage = np.zeros(n, np.float32)
    arc_median_residual = np.full(n, np.inf, np.float32)
    arc_strength = np.zeros(n, np.float32)
    arc_gap_deg = np.zeros(n, np.float32)
    arc_gap_angle = np.zeros(n, np.float32)
    geometry_source = np.zeros(n, np.int8)
    geometry_prediction = np.full((n, 2), np.nan, np.float32)
    geometry_innovation = np.full(n, np.nan, np.float32)

    print(f"Análisis completo a {analysis_w}x{analysis_h}; el original no se decodifica a tamaño completo.")
    progress = Progress("análisis", n)
    previous_phase = None
    hann = cv2.createHanningWindow((analysis_w // 2, analysis_h // 2), cv2.CV_32F)
    tracker = ArcGeometryTracker(radius, scale=scale)
    decoded = 0
    for idx, gray in iter_ffmpeg_gray(video, analysis_w, analysis_h, n, exact_total=True):
        maximum[idx] = gray.max()
        current_phase = phase_image(gray)
        if previous_phase is not None:
            try:
                shift, resp = cv2.phaseCorrelate(previous_phase, current_phase, hann)
                relative[idx] = [2.0 * shift[0], 2.0 * shift[1]]
                response[idx] = float(resp)
            except cv2.error:
                pass
        predicted = tracker.predict(relative[idx], response[idx])
        # The binary contour is the bootstrap / recovery candidate.  It is
        # seeded with the temporal prediction so the fixed-radius fit starts
        # near the expected solar location without hiding a real jump.
        detection = detect_limb(gray, radius, predicted, scale)
        contour = detection["center"] if np.isfinite(detection["center"]).all() else None
        result = tracker.step(gray, relative[idx], response[idx], contour, bool(detection["touch"]))
        if result["accepted"] and result["center"] is not None:
            raw_center[idx] = result["center"]
            if result["source"] == SRC_CONTOUR:
                quality[idx] = detection["quality"]
                coverage[idx] = detection["coverage_deg"]
                median_residual[idx] = detection["median_residual"]
                radial_points[idx] = detection["radial_points"]
                radial_strength[idx] = detection["radial_strength"]
            else:
                arc = result["arc"]
                quality[idx] = radial_quality(
                    arc["coverage_deg"], arc["median_residual"], arc["valid_points"], radius
                )
                coverage[idx] = arc["coverage_deg"]
                median_residual[idx] = arc["median_residual"]
                radial_points[idx] = arc["valid_points"]
                radial_strength[idx] = arc["median_strength"]
        threshold[idx] = detection["threshold"]
        # Physical border contact for the *selected* source.  For a selected
        # radial measurement, binary-contour touch is irrelevant (an unrelated
        # touching contour must not invalidate a good radial center): clipping
        # is decided by the dedicated classify_clipping result.  Binary touch
        # is retained only when the contour source is actually selected.
        touch[idx] = detection["touch"] if result["source"] == SRC_CONTOUR else False
        radius_meas[idx] = detection["algebraic_radius"]
        visible_center[idx] = stable_visible_centroid(gray)
        arc = result["arc"]
        arc_center[idx] = arc["center"]
        arc_measured[idx] = arc["measured"]
        arc_valid_points[idx] = arc["valid_points"]
        arc_coverage[idx] = arc["coverage_deg"]
        arc_median_residual[idx] = arc["median_residual"]
        arc_strength[idx] = arc["median_strength"]
        arc_gap_deg[idx] = arc["largest_gap_deg"]
        arc_gap_angle[idx] = arc["gap_angle_deg"]
        geometry_source[idx] = result["source"]
        geometry_prediction[idx] = result["predicted"]
        geometry_innovation[idx] = result["innovation"]
        clip_pred = raw_center[idx]
        if not np.isfinite(clip_pred).all():
            clip_pred = visible_center[idx]
        if not np.isfinite(clip_pred).all():
            clip_pred = result["predicted"]
        clip = classify_clipping(gray, clip_pred, radius, scale)
        clip_edges[idx] = edges_to_bitmask(clip["edges"])
        clip_score[idx] = clip["score"]
        exposure = measure_exposed_fraction(gray, clip_pred, radius, scale)
        tear_evaluable[idx] = exposure["evaluable"]
        tear_reason[idx] = exposure["reason"]
        tear_bright_level[idx] = exposure.get("bright_level", np.nan)
        tear_visible_threshold[idx] = exposure.get("visible_threshold", np.nan)
        tear_exposed_fraction[idx] = exposure["exposed_fraction"]
        previous_phase = current_phase
        decoded = idx + 1
        progress.update(decoded, force=decoded == n)
    if decoded != n:
        raise SystemExit(f"FFmpeg entregó {decoded}/{n} frames; se cancela para no generar una trayectoria incompleta.")

    warn_scale_drift(radius_meas, radial_points, radius, info.fps)

    analysis = {
        "raw_center": raw_center,
        "quality": quality,
        "coverage": coverage,
        "median_residual": median_residual,
        "threshold": threshold,
        "touch": touch,
        "radial_points": radial_points,
        "radial_strength": radial_strength,
        "visible_center": visible_center,
        "visible_version": np.array([VISIBLE_VERSION], np.int16),
        "relative": relative,
        "response": response,
        "maximum": maximum,
        "radius": np.array([radius]),
        "radius_meas": radius_meas,
        "clip_edges": clip_edges,
        "clip_score": clip_score,
        "clip_version": np.array([CLIP_VERSION], np.int16),
        "tear_evaluable": tear_evaluable,
        "tear_bright_level": tear_bright_level,
        "tear_visible_threshold": tear_visible_threshold,
        "tear_exposed_fraction": tear_exposed_fraction,
        "tear_reason": tear_reason,
        "tear_version": np.array([TEAR_VERSION], np.int16),
        "arc_center": arc_center,
        "arc_measured": arc_measured,
        "arc_valid_points": arc_valid_points,
        "arc_coverage": arc_coverage,
        "arc_median_residual": arc_median_residual,
        "arc_strength": arc_strength,
        "arc_gap_deg": arc_gap_deg,
        "arc_gap_angle": arc_gap_angle,
        "geometry_source": geometry_source,
        "geometry_prediction": geometry_prediction,
        "geometry_innovation": geometry_innovation,
        "analysis_width": np.array([analysis_w]),
        "analysis_height": np.array([analysis_h]),
        "source_frames": np.array([n]),
        "source_fps": np.array([info.fps]),
    }
    analysis.update(identity_arrays(expected))
    # The fallback re-anchors only on accepted trusted geometry (never on the
    # soft-loss ramp, post-horizon false circles or propagated predictions).
    refresh_contrast_track(
        video, analysis, reanchor_mask=_geometry_trusted_mask(analysis, scale)
    )
    solved = solve_tracking(
        analysis,
        args.min_quality,
        profile,
        auto_repair=not getattr(args, "no_auto_repair", False),
    )
    analysis.update(solved)
    save_analysis_cache(path, analysis)
    write_tracking_csv(out_dir / "tracking.csv", analysis)
    print(f"Caché guardada: {path}")
    return analysis
