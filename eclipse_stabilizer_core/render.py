from __future__ import annotations

import csv
import math
import os
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from .constants import (
    CENTER_NAMES,
    CENTER_RECONSTRUCTED,
    CENTER_RELIABLE,
    CENTER_UNRESOLVED,
    CONTENT_NAMES,
    CONTENT_UNCERTAIN,
    CONTENT_USABLE,
    MIN_RADIAL_POINTS,
    REGIME_HORIZON,
    REGIME_LIMBO,
    REGIME_NAMES,
    REGIME_TRANSIENT,
)
from .geometry import analysis_scale, bitmask_to_edges, refine_radial_limb
from .video import Progress, VideoInfo, iter_ffmpeg_bgr, resolve_ffmpeg, scaled_shape, sparse_frames


def draw_detection(bgr: np.ndarray, detection: dict, radius: float, frame_idx: int) -> np.ndarray:
    out = bgr.copy()
    center = detection["center"]
    color = (60, 255, 60) if detection["quality"] >= 0.18 and not detection["touch"] else (0, 180, 255)
    circular_model = detection["median_residual"] <= 4.2 and detection["radial_points"] >= MIN_RADIAL_POINTS
    if np.isfinite(center).all():
        c = tuple(np.rint(center).astype(int))
        if circular_model:
            cv2.circle(out, c, int(round(radius)), color, 2, cv2.LINE_AA)
        cv2.drawMarker(out, c, (255, 255, 255), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
    lines = [
        f"frame {frame_idx}",
        f"q={detection['quality']:.3f} arc={detection['coverage_deg']:.0f} deg",
        f"err={detection['median_residual']:.2f}px th={detection['threshold']:.0f} touch={int(detection['touch'])}",
        "modelo=limbo circular" if circular_model else "modelo=no circular/insuficiente",
    ]
    for row, text in enumerate(lines):
        y = 23 + row * 21
        cv2.putText(out, text, (7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (7, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_contact_sheet(images: list[np.ndarray], columns: int = 4) -> np.ndarray:
    if not images:
        raise ValueError("No hay imágenes para la hoja de contacto")
    h, w = images[0].shape[:2]
    rows = int(math.ceil(len(images) / columns))
    sheet = np.zeros((rows * h, columns * w, 3), np.uint8)
    for i, image in enumerate(images):
        y, x = divmod(i, columns)
        sheet[y * h : (y + 1) * h, x * w : (x + 1) * w] = image
    return sheet


def print_analysis_summary(
    analysis: dict,
    info: VideoInfo,
    selection: dict | None = None,
) -> None:
    trusted = analysis["trusted"].astype(bool)
    raw = analysis["raw_center"]
    anchor = analysis.get("anchor_center", raw)
    center = analysis["center"]
    residual = np.linalg.norm(anchor[trusted] - center[trusted], axis=1)
    trim_end = int(analysis["trim_end"][0])
    keep = (
        selection["mask"].astype(bool)
        if selection is not None
        else analysis["keep"].astype(bool)
    )
    usable = trusted.copy()
    usable[trim_end + 1 :] = False
    consecutive = usable[1:] & usable[:-1]
    stabilized = anchor - center
    transition = np.linalg.norm(np.diff(stabilized, axis=0)[consecutive], axis=1)
    source_scale = info.width / int(analysis["analysis_width"][0])
    print("\nResumen de seguimiento")
    print(f"  detecciones fiables : {trusted.sum()}/{len(trusted)}")
    print(f"  fiables en tramo exportado: {usable.sum()}/{trim_end + 1}")
    print(f"  falsos saltos rechazados: {analysis['rejected'].sum()}")
    if "auto_repaired" in analysis:
        print(
            f"  candidatos: {int(analysis['jitter_candidate'].sum())} jitter, "
            f"{int(analysis['excursion_candidate'].sum())} excursión, "
            f"{int(analysis['corruption_candidate'].sum())} corrupción; "
            f"{int(analysis['auto_repaired'].sum())} frames ajustados"
        )
    print(f"  último frame exportable: {trim_end} ({trim_end / info.fps:.2f} s)")
    horizon_start = int(analysis["horizon_start"][0])
    print(f"  transición a modelo de horizonte: frame {horizon_start} ({horizon_start / info.fps:.2f} s)")
    if "regime" in analysis:
        regime = analysis["regime"].astype(int)
        false_circle = analysis.get("false_circle_after_horizon")
        print(
            f"  regímenes: {int((regime == REGIME_LIMBO).sum())} limbo, "
            f"{int((regime == REGIME_TRANSIENT).sum())} respaldo transitorio, "
            f"{int((regime == REGIME_HORIZON).sum())} horizonte confirmado"
        )
        if false_circle is not None:
            print(f"  falsos círculos tras el horizonte (no ancla): {int(false_circle.sum())}")
    if "clipped" in analysis:
        clipped = analysis["clipped"].astype(bool)
        content = analysis.get("content_state")
        centering = analysis.get("centering_state")
        print(
            f"  contenido: {int((content == CONTENT_USABLE).sum())} usable, "
            f"{int(clipped.sum())} recortado, "
            f"{int((content == CONTENT_UNCERTAIN).sum())} incierto"
        )
        if centering is not None:
            print(
                f"  centrado: {int((centering == CENTER_RELIABLE).sum())} fiable, "
                f"{int((centering == CENTER_RECONSTRUCTED).sum())} reconstruido, "
                f"{int((centering == CENTER_UNRESOLVED).sum())} sin resolver"
            )
    print(
        f"  frames escritos/descartados: {keep.sum()}/{len(keep) - keep.sum()} "
        f"(círculo teórico cortado, solo diagnóstico: {analysis['cut_geometry'].sum()})"
    )
    if "timed_discarded" in analysis and int(analysis["timed_discarded"].sum()) > 0:
        print(f"  descartados por perfil: {int(analysis['timed_discarded'].sum())}")
    if selection is not None:
        mask = selection["mask"].astype(bool)
        origin = selection["origin"]
        manual_kept = int(np.flatnonzero((origin == "manual") & mask).size)
        manual_dropped = int(np.flatnonzero((origin == "manual") & ~mask).size)
        if manual_kept or manual_dropped:
            print(
                f"  decisiones manuales: {manual_kept} recuperados, "
                f"{manual_dropped} descartados"
            )
    discarded = ~keep[: trim_end + 1]
    ranges: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(np.r_[discarded, False]):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            ranges.append((start, idx - 1))
            start = None
    if ranges:
        shown = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges[:12])
        suffix = " ..." if len(ranges) > 12 else ""
        print(f"  rangos descartados ({len(ranges)}): {shown}{suffix}")
    if len(residual):
        print(
            "  ajuste absoluto (px análisis): "
            f"mediana={np.median(residual):.3f}, p95={np.percentile(residual, 95):.3f}, max={residual.max():.3f}"
        )
    if len(transition):
        print(
            "  salto residual consecutivo: "
            f"p95={np.percentile(transition, 95):.3f}, max={transition.max():.3f} px análisis "
            f"(max aprox. {transition.max() * source_scale:.2f} px fuente)"
        )


@contextmanager
def _atomic_output(destination: Path) -> Iterator[Path]:
    """Yield a unique sibling temporary that preserves the final extension.

    FFmpeg infers the container from the extension, so the temporary keeps
    ``destination.suffix`` and adds a uuid to guarantee uniqueness.  The parent
    directory is created on demand and ``os.replace`` publishes the file only
    when the ``with`` block exits without an exception.  Any ``BaseException``
    inside the block, or any failure of ``os.replace`` itself, unlinks the
    temporary and re-raises: a previous destination is never touched unless the
    block completed and the replacement succeeded.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(f"{destination.stem}.{uuid.uuid4().hex}{destination.suffix}")
    try:
        yield temp
    except BaseException:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise
    try:
        os.replace(temp, destination)
    except BaseException:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise


def _finish_encoder(encoder: subprocess.Popen) -> bytes:
    """Close stdin, drain stderr and reap the process, tolerating a broken pipe.

    ``stdin.close`` on an encoder that already died raises ``BrokenPipeError``;
    that is ignored.  Any other ``OSError`` from the close is remembered and
    only re-raised after the process is reaped and stderr drained, so a pipe
    failure never skips cleanup.  stderr is drained before ``wait`` to avoid
    deadlocking on a full stderr buffer.  Idempotent: after the first call the
    process is reaped and stderr drained exactly once.
    """
    if encoder.returncode is not None:
        if encoder.stderr is not None:
            try:
                return encoder.stderr.read()
            finally:
                encoder.stderr.close()
                encoder.stderr = None
        return b""
    close_error: OSError | None = None
    if encoder.stdin is not None:
        try:
            encoder.stdin.close()
        except BrokenPipeError:
            pass
        except OSError as exc:
            close_error = exc
    stderr = b""
    try:
        if encoder.stderr is not None:
            try:
                stderr = encoder.stderr.read()
            finally:
                encoder.stderr.close()
                encoder.stderr = None
    finally:
        encoder.wait()
    if close_error is not None:
        raise close_error
    return stderr


def export_video(
    video: Path,
    info: VideoInfo,
    analysis: dict,
    destination: Path,
    width: int,
    height: int,
    speed: float,
    crf: int,
    preset: str,
    debug_overlay: bool,
    threads: int | None = None,
    selection: dict | None = None,
) -> None:
    analysis_w = int(analysis["analysis_width"][0])
    analysis_h = int(analysis["analysis_height"][0])
    trim_end = int(analysis["trim_end"][0])
    source_total = trim_end + 1
    if selection is not None:
        keep = selection["mask"].astype(bool)[:source_total]
    else:
        keep = analysis["keep"].astype(bool)[:source_total]
    expected_written = int(keep.sum())
    output_fps = info.fps * speed
    with _atomic_output(destination) as temp:
        command = [
            resolve_ffmpeg(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{output_fps:.8f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
        ]
        if threads is not None and threads > 0:
            command.extend(["-threads", str(threads)])
        command.extend(
            [
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(temp),
            ]
        )
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert encoder.stdin is not None
        progress = Progress("exportación", source_total)
        target = np.array([width / 2.0, height / 2.0])
        scale = np.array([width / analysis_w, height / analysis_h])
        written = 0
        decoded = 0
        try:
            for idx, frame in iter_ffmpeg_bgr(video, width, height, source_total):
                decoded = idx + 1
                if not keep[idx]:
                    progress.update(idx + 1, force=idx + 1 == source_total)
                    continue
                source_center = analysis["center"][idx] * scale
                shift = target - source_center
                matrix = np.float32([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]])
                stable = cv2.warpAffine(
                    frame,
                    matrix,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                if debug_overlay:
                    c = (int(round(width / 2.0)), int(round(height / 2.0)))
                    cv2.drawMarker(stable, c, (80, 255, 80), cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
                encoder.stdin.write(stable.tobytes())
                written += 1
                progress.update(idx + 1, force=idx + 1 == source_total)
        except BrokenPipeError as exc:
            stderr = _finish_encoder(encoder)
            raise SystemExit(
                stderr.decode(errors="replace").strip()
                or "El codificador FFmpeg cerró el pipe antes de terminar la exportación."
            ) from exc
        except RuntimeError as exc:
            stderr = _finish_encoder(encoder)
            raise SystemExit(str(exc)) from None
        finally:
            stderr = _finish_encoder(encoder)
        if encoder.returncode != 0:
            raise SystemExit(stderr.decode(errors="replace"))
        if decoded != source_total or written != expected_written:
            raise SystemExit(
                f"Exportación incompleta: decodificados {decoded}/{source_total}, "
                f"escritos {written}/{expected_written}"
            )
    print(
        f"Vídeo escrito: {destination} ({width}x{height}, {output_fps:.3f} fps, "
        f"{expected_written} frames; descartados {source_total - expected_written})"
    )


def validate_preview(preview: Path, analysis: dict, selection: dict | None = None) -> dict:
    cap = cv2.VideoCapture(str(preview))
    if not cap.isOpened():
        return {"validated": 0}
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    analysis_w = int(analysis["analysis_width"][0])
    radius = float(analysis["radius"][0]) * width / analysis_w
    target = np.array([width / 2.0, height / 2.0])
    if selection is not None:
        source_indices = np.flatnonzero(selection["mask"].astype(bool))
    else:
        source_indices = np.flatnonzero(analysis["keep"].astype(bool))
    horizon_start = int(analysis["horizon_start"][0])
    residuals: list[tuple[int, int, float]] = []
    detected_centers: list[np.ndarray] = []
    detected_outputs: list[int] = []
    progress = Progress("validación completa", frames)
    try:
        idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            source_idx = int(source_indices[idx]) if idx < len(source_indices) else -1
            center = np.array([np.nan, np.nan])
            expected = target
            valid_detection = False
            if 0 <= source_idx < horizon_start:
                center, radial_points, _ = refine_radial_limb(gray, target, radius, scale=analysis_scale(width))
                valid_detection = radial_points >= MIN_RADIAL_POINTS and np.isfinite(center).all()
            # The horizon regime is intentionally not judged with a circle or
            # centroid after translation: refraction, terrestrial occlusion and
            # glow make either metric change with the content itself. It remains
            # in the preview for visual review and is tracked temporally.
            if valid_detection:
                residuals.append((idx, source_idx, float(np.linalg.norm(center - expected))))
                # Normalize both regimes to the same origin for jitter metrics.
                detected_centers.append(center - expected + target)
                detected_outputs.append(idx)
            idx += 1
            progress.update(idx, force=idx == frames)
    finally:
        cap.release()
    values = np.asarray([item[2] for item in residuals])
    result = {"validated": int(len(values))}
    if len(values):
        centers_array = np.asarray(detected_centers)
        output_array = np.asarray(detected_outputs)
        adjacent = np.diff(output_array) == 1
        jitter = np.linalg.norm(np.diff(centers_array, axis=0)[adjacent], axis=1)
        result.update(
            median=float(np.median(values)),
            p95=float(np.percentile(values, 95)),
            maximum=float(values.max()),
            worst=[
                {"output_frame": frame, "source_frame": source, "residual_px": residual}
                for frame, source, residual in sorted(residuals, key=lambda item: item[2], reverse=True)[:10]
            ],
        )
        if len(jitter):
            result.update(
                jitter_p95=float(np.percentile(jitter, 95)),
                jitter_maximum=float(jitter.max()),
                jitter_over_1px=int((jitter > 1.0).sum()),
            )
        print(
            f"Validación sobre preview ({len(values)}/{frames} frames, px de preview): "
            f"mediana={result['median']:.3f}, p95={result['p95']:.3f}, max={result['maximum']:.3f}"
        )
        if len(jitter):
            print(
                f"Vibración residual entre frames: p95={result['jitter_p95']:.3f}, "
                f"max={result['jitter_maximum']:.3f}, >1px={result['jitter_over_1px']}"
            )
    return result


def _estimate_jpeg_bytes(video: Path, info: VideoInfo, width: int, samples: int = 8) -> float:
    """Encode a few sampled frames in memory to approximate the JPEG size."""
    end = max(0, info.frames - 1)
    indices = np.unique(np.linspace(0, end, min(samples, info.frames), dtype=int))
    frames = sparse_frames(video, indices, width)
    sizes: list[int] = []
    for _, bgr in frames:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            sizes.append(len(buf))
    return float(np.mean(sizes)) if sizes else 0.0


def _draw_debug_overlay(
    frame: np.ndarray,
    analysis: dict,
    idx: int,
    scale: tuple[float, float],
    target: tuple[int, int],
    fps: float,
    selection: dict | None,
) -> np.ndarray:
    """Small, legible overlay describing one source frame for review.mp4."""
    out = frame.copy()
    h, w = frame.shape[:2]
    sx, sy = scale

    def point(key: str):
        arr = analysis.get(key)
        if arr is None:
            return None
        c = arr[idx]
        if np.isfinite(c).all():
            return (int(round(c[0] * sx)), int(round(c[1] * sy)))
        return None

    raw = point("raw_center")
    solved = point("raw_solved_center") if "raw_solved_center" in analysis else None
    final = point("center")
    visible = point("visible_center")

    content_code = int(np.asarray(analysis.get("content_state", np.zeros(1, np.int8))).reshape(-1)[idx])
    centering_code = int(np.asarray(analysis.get("centering_state", np.zeros(1, np.int8))).reshape(-1)[idx])
    regime_code = int(np.asarray(analysis.get("regime", np.zeros(1, np.int8))).reshape(-1)[idx])
    content_name = CONTENT_NAMES.get(content_code, "unknown")
    centering_name = CENTER_NAMES.get(centering_code, "unknown")
    regime_name = REGIME_NAMES.get(regime_code, "unknown")

    def flag(key: str) -> bool:
        arr = analysis.get(key)
        if arr is None:
            return False
        return bool(np.asarray(arr).reshape(-1)[idx])

    false_circle = flag("false_circle_after_horizon")
    jitter = flag("jitter_candidate")
    excursion = flag("excursion_candidate")
    corruption = flag("corruption_candidate")
    tear = flag("tear_detected")
    auto_repaired = flag("auto_repaired")
    clip_edges = int(np.asarray(analysis.get("clip_edges", np.zeros(1, np.int8))).reshape(-1)[idx])
    correction = float(np.asarray(analysis.get("correction_magnitude", np.zeros(1))).reshape(-1)[idx])

    final_pt = point("center")
    # The export translation is target - final_center, so the estimated
    # post-transform residual is exactly zero by construction. Independent
    # preview validation is reported separately and never feeds the trajectory.
    residual = 0.0 if final_pt is not None else float("nan")

    if raw is not None:
        cv2.drawMarker(out, raw, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 1, cv2.LINE_AA)
    if solved is not None:
        cv2.drawMarker(out, solved, (255, 170, 0), cv2.MARKER_DIAMOND, 10, 1, cv2.LINE_AA)
    if final is not None:
        cv2.drawMarker(out, final, (0, 255, 0), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
        cv2.circle(out, final, 8, (0, 255, 0), 1, cv2.LINE_AA)
    if raw is not None and final is not None:
        cv2.arrowedLine(out, raw, final, (0, 200, 255), 1, cv2.LINE_AA, tipLength=0.2)
    cv2.drawMarker(out, target, (80, 80, 255), cv2.MARKER_SQUARE, 14, 1, cv2.LINE_AA)

    flags = []
    if jitter:
        flags.append("jitter")
    if excursion:
        flags.append("excursion")
    if corruption:
        flags.append("corrupcion")
    if tear:
        flags.append("tear-sensor")
    if auto_repaired:
        flags.append("auto-reparado")
    flags_text = ",".join(flags) or "sin avisos"

    border = bitmask_to_edges(clip_edges) if clip_edges else []
    reason = str(np.asarray(analysis.get("content_reason", np.full(1, "", "U90"))).reshape(-1)[idx])
    if border:
        reason = f"{reason} (borde: {','.join(border)})"
    decision_origin = str(selection["origin"][idx]) if selection is not None else "auto"
    decision = {
        "manual": "manual",
        "profile": "perfil",
        "auto": "automatico",
    }.get(decision_origin, decision_origin)
    decision_reason = (
        str(selection["reason"][idx]) if selection is not None else ""
    )
    centering_reason = str(
        np.asarray(analysis.get("centering_reason", np.full(1, "", "U90"))).reshape(-1)[idx]
    )
    local_innovation = float(
        np.asarray(analysis.get("local_innovation", np.zeros(1))).reshape(-1)[idx]
    )
    accumulated = float(
        np.asarray(analysis.get("cumulative_innovation", np.zeros(1))).reshape(-1)[idx]
    )

    lines = [
        f"src {idx}  t={idx / fps:.2f}s",
        f"content={content_name}  {reason[:60]}",
        f"centering={centering_name} {centering_reason[:32]}  regime={regime_name}"
        + ("  FALSO-CIRCULO" if false_circle else ""),
        f"raw={raw} solved={solved} final={final} visible={visible}",
        f"corr={correction:.2f}px innov={local_innovation:.2f}px deriva={accumulated:.2f}px",
        f"objetivo=({target[0]},{target[1]}) residuo-post={residual:.2f}px",
        f"avisos={flags_text}",
        f"decision={decision} {decision_reason[:55]}",
    ]
    bar = np.zeros((min(h, len(lines) * 20 + 14), w, 3), np.uint8)
    bar[:] = (20, 20, 20)
    out[0 : bar.shape[0], :] = cv2.addWeighted(out[0 : bar.shape[0], :], 0.45, bar, 0.55, 0)
    for row, text in enumerate(lines):
        y = 18 + row * 19
        cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_debug_csv(path: Path, analysis: dict, selection: dict | None) -> None:
    """Write a row per source frame (usables included) with every state, reason,
    center, correction, regime, weight/signal, auto keep, effective keep and the
    manual decision."""
    n = int(len(analysis["keep"]))
    if selection is None:
        selection = {"mask": analysis["keep"].astype(bool), "origin": np.full(n, "auto", "U16"), "reason": np.full(n, "", "U80")}
    headers = [
        "frame",
        "content_state",
        "content_reason",
        "centering_state",
        "centering_reason",
        "regime",
        "regime_reason",
        "false_circle_after_horizon",
        "raw_cx",
        "raw_cy",
        "raw_solved_cx",
        "raw_solved_cy",
        "repaired_cx",
        "repaired_cy",
        "final_cx",
        "final_cy",
        "correction_magnitude",
        "absolute_innovation",
        "local_innovation",
        "cumulative_innovation",
        "anchor_weight",
        "robust_anchor_weight",
        "rel_weight",
        "robust_rel_weight",
        "quality",
        "coverage_deg",
        "median_residual",
        "radial_points",
        "radial_strength",
        "phase_dx",
        "phase_dy",
        "phase_response",
        "clipped",
        "border_clipped",
        "clip_score",
        "tear_candidate",
        "tear_evaluable",
        "tear_detected",
        "tear_run_length",
        "tear_reason",
        "tear_veto_reason",
        "tear_exposed_fraction",
        "tear_baseline_fraction",
        "tear_drop_fraction",
        "tear_temporal_noise",
        "jitter_candidate",
        "excursion_candidate",
        "jump_confirmed",
        "corruption_candidate",
        "corruption_signal_count",
        "auto_repaired",
        "repair_reason",
        "keep_auto",
        "keep_effective",
        "decision",
        "decision_reason",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for idx in range(n):
            content_code = int(analysis["content_state"][idx])
            centering_code = int(analysis["centering_state"][idx])
            regime_code = int(analysis["regime"][idx])
            clip_edges = int(analysis["clip_edges"][idx]) if "clip_edges" in analysis else 0
            decision = str(selection["origin"][idx])
            writer.writerow(
                [
                    idx,
                    CONTENT_NAMES.get(content_code, "unknown"),
                    analysis["content_reason"][idx] if "content_reason" in analysis else "",
                    CENTER_NAMES.get(centering_code, "unknown"),
                    analysis["centering_reason"][idx] if "centering_reason" in analysis else "",
                    REGIME_NAMES.get(regime_code, "unknown"),
                    analysis["regime_reason"][idx] if "regime_reason" in analysis else "",
                    int(analysis["false_circle_after_horizon"][idx])
                    if "false_circle_after_horizon" in analysis
                    else 0,
                    *analysis["raw_center"][idx],
                    *(analysis["raw_solved_center"][idx] if "raw_solved_center" in analysis else (float("nan"), float("nan"))),
                    *(analysis["repaired_center"][idx] if "repaired_center" in analysis else (float("nan"), float("nan"))),
                    *analysis["center"][idx],
                    analysis["correction_magnitude"][idx] if "correction_magnitude" in analysis else 0.0,
                    analysis["absolute_innovation"][idx] if "absolute_innovation" in analysis else 0.0,
                    analysis["local_innovation"][idx] if "local_innovation" in analysis else 0.0,
                    analysis["cumulative_innovation"][idx] if "cumulative_innovation" in analysis else 0.0,
                    analysis["anchor_weight"][idx] if "anchor_weight" in analysis else 0.0,
                    analysis["robust_anchor_weight"][idx] if "robust_anchor_weight" in analysis else 0.0,
                    analysis["rel_weight"][idx] if "rel_weight" in analysis else 0.0,
                    analysis["robust_rel_weight"][idx] if "robust_rel_weight" in analysis else 0.0,
                    analysis["quality"][idx],
                    analysis["coverage"][idx],
                    analysis["median_residual"][idx],
                    int(analysis["radial_points"][idx]),
                    analysis["radial_strength"][idx],
                    *analysis["relative"][idx],
                    analysis["response"][idx],
                    int(analysis["clipped"][idx]) if "clipped" in analysis else 0,
                    int(analysis["border_clipped"][idx]) if "border_clipped" in analysis else 0,
                    analysis["clip_score"][idx] if "clip_score" in analysis else 0.0,
                    int(analysis["tear_candidate"][idx]) if "tear_candidate" in analysis else 0,
                    int(analysis["tear_evaluable"][idx]) if "tear_evaluable" in analysis else 0,
                    int(analysis["tear_detected"][idx]) if "tear_detected" in analysis else 0,
                    int(analysis["tear_run_length"][idx]) if "tear_run_length" in analysis else 0,
                    analysis["tear_reason"][idx] if "tear_reason" in analysis else "",
                    analysis["tear_veto_reason"][idx] if "tear_veto_reason" in analysis else "",
                    analysis["tear_exposed_fraction"][idx] if "tear_exposed_fraction" in analysis else np.nan,
                    analysis["tear_baseline_fraction"][idx] if "tear_baseline_fraction" in analysis else np.nan,
                    analysis["tear_drop_fraction"][idx] if "tear_drop_fraction" in analysis else np.nan,
                    analysis["tear_temporal_noise"][idx] if "tear_temporal_noise" in analysis else np.nan,
                    int(analysis["jitter_candidate"][idx]) if "jitter_candidate" in analysis else 0,
                    int(analysis["excursion_candidate"][idx]) if "excursion_candidate" in analysis else 0,
                    int(analysis["jump_confirmed"][idx]) if "jump_confirmed" in analysis else 0,
                    int(analysis["corruption_candidate"][idx]) if "corruption_candidate" in analysis else 0,
                    int(analysis["corruption_signal_count"][idx]) if "corruption_signal_count" in analysis else 0,
                    int(analysis["auto_repaired"][idx]) if "auto_repaired" in analysis else 0,
                    analysis["repair_reason"][idx] if "repair_reason" in analysis else "",
                    int(analysis["keep"][idx]),
                    int(selection["mask"][idx]),
                    decision,
                    selection["reason"][idx],
                ]
            )


def write_debug(
    video: Path,
    info: VideoInfo,
    analysis: dict,
    out_dir: Path,
    debug_width: int = 320,
    max_images: int = 10000,
    selection: dict | None = None,
) -> None:
    """Create review.mp4, frames.csv, per-state JPEGs and a contact sheet.

    The source is decoded once at ``debug_width`` (proportion preserved).  The
    review and CSV always cover the complete sequence (keep is not applied);
    only the per-frame JPEG export honours ``max_images``.  ``review.mp4`` is
    published atomically (temp sibling + ``os.replace``) and only after FFmpeg
    reports success with exactly ``n`` decoded and written frames; the CSV,
    JPEGs and contact sheet remain outside that atomicity.
    """
    debug_dir = out_dir / "debug"
    frames_dir = debug_dir / "frames"
    for sub in ("usable", "clipped", "uncertain"):
        (frames_dir / sub).mkdir(parents=True, exist_ok=True)
    for old_image in frames_dir.rglob("*.jpg"):
        old_image.unlink()
    old_sheet = debug_dir / "contact_sheet.jpg"
    if old_sheet.exists():
        old_sheet.unlink()
    debug_w, debug_h = scaled_shape(info, debug_width)
    n = int(len(analysis["keep"]))
    content = np.asarray(analysis.get("content_state", np.zeros(n, np.int8)))
    clipped = np.asarray(analysis.get("clipped", np.zeros(n, bool))).astype(bool)
    problem = (
        np.asarray(analysis.get("jitter_candidate", np.zeros(n, bool)), bool)
        | np.asarray(analysis.get("excursion_candidate", np.zeros(n, bool)), bool)
        | np.asarray(analysis.get("corruption_candidate", np.zeros(n, bool)), bool)
        | np.asarray(analysis.get("auto_repaired", np.zeros(n, bool)), bool)
    )
    clipped_idx = np.flatnonzero(clipped)
    uncertain_idx = np.flatnonzero(content == CONTENT_UNCERTAIN)
    problem_idx = np.flatnonzero(problem)
    neighbor_set: set[int] = set()
    for i in np.r_[clipped_idx, uncertain_idx, problem_idx]:
        for d in (-2, -1, 1, 2):
            j = int(i) + d
            if 0 <= j < n:
                neighbor_set.add(j)

    if n <= max_images:
        selected = set(range(n))
    else:
        selected: set[int] = set()
        priority: list[int] = []
        seen: set[int] = set()
        for group in (clipped_idx, uncertain_idx, problem_idx, sorted(neighbor_set)):
            for i in group:
                i = int(i)
                if i not in seen:
                    seen.add(i)
                    priority.append(i)
        selected = set(priority[:max_images])
        remaining = max_images - len(selected)
        if remaining > 0:
            usable_pool = [
                i for i in range(n) if int(content[i]) == CONTENT_USABLE and i not in selected
            ]
            if usable_pool:
                k = min(remaining, len(usable_pool))
                picks = np.unique(np.linspace(0, len(usable_pool) - 1, k, dtype=int))
                for p in picks:
                    selected.add(int(usable_pool[p]))
    selected_idx = sorted(selected)
    omitted = n - len(selected_idx)

    avg_bytes = _estimate_jpeg_bytes(video, info, debug_w)
    if avg_bytes > 0:
        print(
            f"[debug] tamaño JPEG estimado ~{avg_bytes / 1024:.0f} KiB/frame; "
            f"total aprox {avg_bytes * len(selected_idx) / (1024 * 1024):.1f} MiB "
            f"para {len(selected_idx)} imagen(es)"
        )

    write_debug_csv(debug_dir / "frames.csv", analysis, selection)

    target = (int(round(debug_w / 2.0)), int(round(debug_h / 2.0)))
    scale = (debug_w / int(analysis["analysis_width"][0]), debug_h / int(analysis["analysis_height"][0]))

    with _atomic_output(debug_dir / "review.mp4") as review_temp:
        command = [
            resolve_ffmpeg(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{debug_w}x{debug_h}",
            "-r",
            f"{info.fps:.8f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(review_temp),
        ]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert encoder.stdin is not None
        progress = Progress("debug", n, interval=0.5)
        exported = 0
        decoded = 0
        written = 0
        sheet_clipped: list[np.ndarray] = []
        sheet_clipped_omitted = 0
        sheet_limit = 200
        sheet_uncertain: list[np.ndarray] = []
        sheet_uncertain_set: set[int] = set()
        if len(uncertain_idx):
            k = min(50, len(uncertain_idx))
            sheet_uncertain_set = set(
                int(uncertain_idx[p]) for p in np.unique(np.linspace(0, len(uncertain_idx) - 1, k, dtype=int))
            )
        try:
            for idx, frame in iter_ffmpeg_bgr(video, debug_w, debug_h, n, exact_total=True):
                decoded = idx + 1
                overlay = _draw_debug_overlay(frame, analysis, idx, scale, target, info.fps, selection)
                encoder.stdin.write(overlay.tobytes())
                written += 1
                if idx in selected_idx:
                    state = CONTENT_NAMES.get(int(content[idx]), "uncertain")
                    cv2.imwrite(
                        str(frames_dir / state / f"frame_{idx:05d}_{state}.jpg"),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 75],
                    )
                    exported += 1
                if idx in clipped_idx:
                    if len(sheet_clipped) < sheet_limit:
                        sheet_clipped.append(frame)
                    else:
                        sheet_clipped_omitted += 1
                if idx in sheet_uncertain_set:
                    sheet_uncertain.append(frame)
                progress.update(idx + 1, force=idx + 1 == n)
        except BrokenPipeError as exc:
            stderr = _finish_encoder(encoder)
            raise SystemExit(
                stderr.decode(errors="replace").strip()
                or "El codificador FFmpeg cerró el pipe al generar review.mp4."
            ) from exc
        except RuntimeError as exc:
            stderr = _finish_encoder(encoder)
            raise SystemExit(str(exc)) from None
        finally:
            stderr = _finish_encoder(encoder)
        if encoder.returncode != 0:
            raise SystemExit(stderr.decode(errors="replace"))
        if decoded != n or written != n:
            raise SystemExit(
                f"Vídeo de revisión incompleto: decodificados {decoded}/{n}, "
                f"escritos {written}/{n}"
            )

    if sheet_clipped or sheet_uncertain:
        sheet = make_contact_sheet(sheet_clipped + sheet_uncertain, columns=4)
        cv2.imwrite(
            str(debug_dir / "contact_sheet.jpg"),
            sheet,
            [cv2.IMWRITE_JPEG_QUALITY, 75],
        )
        print(
            f"[debug] contact_sheet.jpg: {len(sheet_clipped)} clipped, "
            f"{len(sheet_uncertain)} uncertain"
        )
        if sheet_clipped_omitted:
            print(
                f"[debug] {sheet_clipped_omitted} clipped no incluidos en la hoja de contacto "
                f"(límite {sheet_limit})"
            )
    else:
        print("[debug] sin frames clipped ni uncertain; no se genera contact_sheet.jpg")

    if omitted:
        print(
            f"[debug] se exportaron {exported}/{n} JPEG; {omitted} omitidos por "
            f"--debug-max-images {max_images}. Sube el límite para revisar cada frame "
            f"como imagen, p.ej. --debug-max-images {n}."
        )
    else:
        print(f"[debug] JPEG exportados: {exported}/{n} (límite {max_images})")
