#!/usr/bin/env python3
"""Fast, geometry-first stabilizer for partial solar eclipses.

The anchor is the outer solar limb, measured with radial scans around a temporal
prediction.  A binary contour only bootstraps or recovers the tracker, and phase
correlation supplies relative displacement rather than an absolute solar center.

Recommended workflow:
  python eclipse_stabilizer.py preview VIDEO
  # Watch and approve preview.mp4
  python eclipse_stabilizer.py export VIDEO

Use inspect or analyze --debug only when the preview needs investigation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from eclipse_stabilizer_core import (
    CACHE_REQUIRED_ARRAYS,
    AnchoredContrastFallback,
    Progress,
    VideoInfo,
    analyze_video,
    build_cache_identity,
    cache_path,
    cache_status,
    default_output_dir,
    detect_limb,
    draw_detection,
    ensure_capabilities,
    export_video,
    ffmpeg_version,
    find_video,
    identity_arrays,
    load_profile,
    make_contact_sheet,
    parse_probe_json,
    parse_vfrdet_output,
    print_analysis_summary,
    print_manual_selection,
    print_video_info,
    probe_opencv,
    probe_video,
    profile_hash,
    reset_executable_cache,
    resolve_ffmpeg,
    resolve_ffprobe,
    resolve_output_dir,
    resolve_radius,
    resolve_selection,
    scaled_shape,
    sparse_frames,
    validate_preview,
    verify_exact_cfr,
    write_debug,
)
from eclipse_stabilizer_core import (
    EDGE_BITS,
    EDGE_NAMES,
    MIN_BRIGHTNESS,
    MIN_RADIAL_POINTS,
    ROOT,
    VISIBLE_VERSION,
    CONTRAST_VERSION,
    CLIP_VERSION,
    TEAR_VERSION,
    CACHE_VERSION,
    CONTENT_USABLE,
    CONTENT_CLIPPED,
    CONTENT_UNCERTAIN,
    CENTER_RELIABLE,
    CENTER_RECONSTRUCTED,
    CENTER_UNRESOLVED,
    CONTENT_NAMES,
    CENTER_NAMES,
    REGIME_LIMBO,
    REGIME_TRANSIENT,
    REGIME_HORIZON,
    REGIME_NAMES,
)
from eclipse_stabilizer_core import (
    run_capture,
    read_frame_at,
    iter_ffmpeg_gray,
    iter_ffmpeg_bgr,
    analysis_scale,
    edges_to_bitmask,
    bitmask_to_edges,
    largest_contour,
    clipping_thresholds,
    classify_clipping,
    stable_visible_centroid,
    local_contrast_center,
    measure_exposed_fraction,
    kasa_circle,
    outer_limb_points,
    fit_fixed_radius,
    threshold_candidates,
    measure_radial_limb,
    refine_radial_limb,
    calibrate_radius,
    radial_quality,
    ArcGeometryTracker,
)
from eclipse_stabilizer_core import (
    SRC_NONE,
    SRC_RADIAL,
    SRC_RADIAL_DISAGREE,
    SRC_CONTOUR,
    SRC_NAMES,
)
from eclipse_stabilizer_core import (
    parse_frame_spec,
    effective_selection,
)
from eclipse_stabilizer_core import (
    transient_outliers,
    solve_tridiagonal,
    weighted_path,
    robust_path_solution,
    classify_regime,
    detect_transient_exposure_drops,
    isolate_transient_tears,
    apply_profile_discards,
    classify_content,
    classify_centering,
    contradicted_reconstruction,
    solve_tracking,
    write_tracking_csv,
)
from eclipse_stabilizer_core import (
    calibration_frames,
    phase_image,
    save_analysis_cache,
    refresh_contrast_track,
    refresh_contrast_track_frames,
    warn_scale_drift,
)
from eclipse_stabilizer_core import (
    write_debug_csv,
)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def load_or_analyze(args: argparse.Namespace, video: Path, info: VideoInfo, out_dir: Path) -> dict:
    return analyze_video(args, video, info, out_dir, force=getattr(args, "force", False))


def command_inspect(args: argparse.Namespace, video: Path, info: VideoInfo, out_dir: Path) -> None:
    inspect_dir = out_dir / "inspect"
    inspect_dir.mkdir(parents=True, exist_ok=True)
    radius = resolve_radius(args, video, info, args.analysis_width)
    scale = analysis_scale(args.analysis_width)
    start = max(0, args.start_frame)
    end = info.frames - 1 if args.end_frame < 0 else min(info.frames - 1, args.end_frame)
    if end < start:
        raise SystemExit("--end-frame debe ser mayor o igual que --start-frame")
    indices = np.unique(np.linspace(start, end, args.samples, dtype=int))
    print(f"Inspección: {len(indices)} fotogramas aislados repartidos por toda la secuencia.")
    frames = sparse_frames(video, indices, args.analysis_width)
    overlays: list[np.ndarray] = []
    progress = Progress("detector disperso", len(frames), interval=0.25)
    previous = None
    rows: list[list[object]] = []
    for pos, (idx, bgr) in enumerate(frames, 1):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        detection = detect_limb(gray, radius, previous, scale)
        if detection["quality"] >= args.min_quality:
            previous = detection["center"]
        overlays.append(draw_detection(bgr, detection, radius, idx))
        rows.append(
            [
                idx,
                *detection["center"],
                detection["quality"],
                detection["coverage_deg"],
                detection["median_residual"],
                detection["threshold"],
                int(detection["touch"]),
            ]
        )
        progress.update(pos, force=pos == len(frames))
    sheet_path = inspect_dir / "contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), make_contact_sheet(overlays), [cv2.IMWRITE_JPEG_QUALITY, 88])
    with (inspect_dir / "detections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "cx", "cy", "quality", "coverage_deg", "median_residual", "threshold", "touch"])
        writer.writerows(rows)
    print(f"Hoja de contacto: {sheet_path}")


def command_analyze(args: argparse.Namespace, video: Path, info: VideoInfo, out_dir: Path) -> None:
    analysis = analyze_video(args, video, info, out_dir, force=args.force)
    selection = resolve_selection(args, analysis)
    print_analysis_summary(analysis, info, selection=selection)
    if getattr(args, "debug", False):
        print_manual_selection(selection)
        write_debug(
            video,
            info,
            analysis,
            out_dir,
            debug_width=args.debug_width,
            max_images=args.debug_max_images,
            selection=selection,
        )


def command_preview(args: argparse.Namespace, video: Path, info: VideoInfo, out_dir: Path) -> None:
    analysis = load_or_analyze(args, video, info, out_dir)
    selection = resolve_selection(args, analysis)
    print_manual_selection(selection)
    width, height = scaled_shape(info, args.preview_width)
    destination = out_dir / "preview.mp4"
    export_video(
        video,
        info,
        analysis,
        destination,
        width,
        height,
        args.speed,
        crf=30,
        preset="ultrafast",
        debug_overlay=args.debug_overlay,
        selection=selection,
    )
    validation = validate_preview(destination, analysis, selection=selection)
    (out_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")


def command_export(args: argparse.Namespace, video: Path, info: VideoInfo, out_dir: Path) -> None:
    analysis = load_or_analyze(args, video, info, out_dir)
    selection = resolve_selection(args, analysis)
    print_manual_selection(selection)
    destination = out_dir / args.name
    export_video(
        video,
        info,
        analysis,
        destination,
        info.width,
        info.height,
        speed=1.0,
        crf=args.crf,
        preset=args.preset,
        debug_overlay=False,
        threads=args.threads,
        selection=selection,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", help="perfil JSON opcional con descartes específicos del vídeo")
    parser.add_argument("--out", default=None, help="directorio de resultados; por defecto se deriva del nombre del vídeo")
    parser.add_argument("--analysis-width", type=int, default=270, help="ancho de análisis de baja resolución")
    parser.add_argument("--radius", type=float, default=None, help="radio solar fijo en píxeles de análisis; por defecto se calibra automáticamente")
    parser.add_argument("--min-quality", type=float, default=0.18, help="confianza mínima del ajuste absoluto")
    parser.add_argument("--force", action="store_true", help="ignorar y regenerar la caché de análisis")
    parser.add_argument(
        "--no-auto-repair",
        action="store_true",
        help="conservar la solución bruta sin la segunda ponderación robusta automática",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="inspeccionar fotogramas aislados sin recorrer la secuencia")
    inspect.add_argument("video", metavar="VIDEO", help="vídeo fuente")
    inspect.add_argument("--samples", type=int, default=24)
    inspect.add_argument("--start-frame", type=int, default=0)
    inspect.add_argument("--end-frame", type=int, default=-1)

    analyze = sub.add_parser("analyze", help="seguir todo el vídeo a baja resolución y guardar caché")
    analyze.add_argument("video", metavar="VIDEO", help="vídeo fuente")
    analyze.add_argument(
        "--debug",
        action="store_true",
        help="generar review.mp4, frames.csv, imágenes por estado y contact_sheet.jpg",
    )
    analyze.add_argument("--debug-width", type=int, default=320, help="ancho máximo de cada imagen de depuración")
    analyze.add_argument("--debug-max-images", type=int, default=10000, help="límite de JPEG por ejecución de depuración")
    analyze.add_argument("--drop-frames", help="frames/rangos a descartar, p.ej. 20-24,700")
    analyze.add_argument("--keep-frames", help="frames/rangos a recuperar, p.ej. 21-25,701")

    preview = sub.add_parser("preview", help="crear preview 270x480 rápida usando la caché")
    preview.add_argument("video", metavar="VIDEO", help="vídeo fuente")
    preview.add_argument("--preview-width", type=int, default=270)
    preview.add_argument("--speed", type=float, default=2.0, help="2 = todos los frames reproducidos al doble de velocidad")
    preview.add_argument("--debug-overlay", action="store_true", help="dibujar una cruz en el centro de destino")
    preview.add_argument("--drop-frames", help="frames/rangos a descartar, p.ej. 20-24,700")
    preview.add_argument("--keep-frames", help="frames/rangos a recuperar, p.ej. 21-25,701")

    export = sub.add_parser("export", help="exportar a resolución original tras aprobar el preview")
    export.add_argument("video", metavar="VIDEO", help="vídeo fuente")
    export.add_argument("--name", default="stabilized.mp4")
    export.add_argument("--crf", type=int, default=18)
    export.add_argument("--preset", default="fast")
    export.add_argument("--threads", type=int, default=2, help="hilos del codificador; 2 reduce la carga del equipo")
    export.add_argument("--drop-frames", help="frames/rangos a descartar, p.ej. 20-24,700")
    export.add_argument("--keep-frames", help="frames/rangos a recuperar, p.ej. 21-25,701")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.analysis_width < 160:
        parser.error("--analysis-width debe ser >=160 para conservar suficiente limbo solar")
    if args.radius is not None and args.radius <= 0:
        parser.error("--radius debe ser un radio positivo en píxeles de análisis")
    if args.command == "analyze" and args.debug_width < 64:
        parser.error("--debug-width debe ser >=64")
    if args.command == "analyze" and args.debug_max_images < 0:
        parser.error("--debug-max-images debe ser >=0")
    profile_path = args.profile
    args.profile = load_profile(profile_path)
    args.profile_path = profile_path
    video = find_video(args.video)
    info = probe_video(video)
    out_dir = resolve_output_dir(args.out, video)
    need_encoder = args.command in ("preview", "export") or (args.command == "analyze" and args.debug)
    ffmpeg_exe = ensure_capabilities(need_encoder=need_encoder)
    print(f"FFmpeg      : {ffmpeg_exe}")
    print(f"Versión FFmpeg: {ffmpeg_version(ffmpeg_exe)}")
    ffprobe = resolve_ffprobe()
    print(f"FFprobe     : {ffprobe}" if ffprobe else "FFprobe     : OpenCV fallback")
    if args.command != "inspect" and not info.frame_count_exact:
        info = verify_exact_cfr(video, info)
    print_video_info(video, info)
    print(f"Salida      : {out_dir}")
    analysis_w, analysis_h = scaled_shape(info, args.analysis_width)
    radius_requested = "auto" if args.radius is None else f"{float(args.radius):.6g}"
    expected = build_cache_identity(
        video, info, analysis_w, analysis_h, radius_requested, args.profile, profile_path, args.min_quality
    )
    if args.radius is not None:
        print(f"Radio       : manual {args.radius:.3f} px de análisis")
    else:
        print("Radio       : calibración automática")
    if args.profile is not None:
        print(f"Perfil      : {profile_path}")
    else:
        print("Perfil      : ninguno")
    if args.force:
        print("Caché       : forzada (--force)")
    else:
        status, reasons = cache_status(cache_path(out_dir), expected)
        if status == "valid":
            print("Caché       : válida")
        elif status == "missing":
            print("Caché       : ausente")
        elif status == "refreshable":
            shown = "; ".join(reasons[:2])
            print(f"Caché       : refrescable (solo contraste) — {shown}")
        else:
            shown = "; ".join(reasons[:2])
            print(f"Caché       : incompatible — {shown}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "inspect":
        command_inspect(args, video, info, out_dir)
    elif args.command == "analyze":
        command_analyze(args, video, info, out_dir)
    elif args.command == "preview":
        command_preview(args, video, info, out_dir)
    elif args.command == "export":
        command_export(args, video, info, out_dir)


if __name__ == "__main__":
    main()
