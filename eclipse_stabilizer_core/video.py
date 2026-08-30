from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .constants import ROOT


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frames: int
    fps: float
    duration: float
    probe_source: str = "ffprobe"
    frame_count_exact: bool = True


class Progress:
    def __init__(self, label: str, total: int, interval: float = 1.0):
        self.label = label
        self.total = max(1, int(total))
        self.interval = interval
        self.started = time.perf_counter()
        self.last = 0.0

    def update(self, done: int, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and done < self.total and now - self.last < self.interval:
            return
        elapsed = now - self.started
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - done) / rate if rate > 0 else 0.0
        pct = 100.0 * done / self.total
        print(
            f"[{self.label}] {done}/{self.total} ({pct:5.1f}%) | "
            f"{rate:6.1f} fps | {elapsed:6.1f}s | ETA {eta:6.1f}s",
            flush=True,
        )
        self.last = now


def run_capture(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT).decode(errors="replace")
    except FileNotFoundError as exc:
        raise SystemExit(f"No se encontró {command[0]}. Instala FFmpeg y añádelo al PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.output.decode(errors="replace").strip()) from exc


def _capture_bytes(command: list[str], timeout: int = 30) -> bytes:
    """Run a quick check capturing stdout and stderr separately.

    Only ``stdout`` is returned, so ffmpeg diagnostics written to stderr never
    pollute a raw-pipeline length check.  A missing executable, a timeout or a
    non-zero return code become actionable ``SystemExit`` errors.
    """
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"No se encontró {command[0]}.") from exc
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            f"La comprobación de {command[0]} agotó el tiempo ({timeout} s) sin terminar; "
            "revisa el ejecutable FFmpeg."
        ) from exc
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        detail = stderr or "sin salida de error"
        raise SystemExit(f"{command[0]} falló (código {result.returncode}): {detail}")
    return result.stdout


def _executable_responds(exe: str) -> bool:
    try:
        result = subprocess.run(
            [exe, "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


# Per-process resolution and capability cache.  Resolution is deliberately a
# function, not an import-time constant: FFMPEG/FFPROBE are no longer evaluated
# when the module is imported, so a broken configuration fails at run time with
# an actionable message instead of silently pinning a stale path.
_EXECUTABLE_CACHE: dict[str, str | None] = {}
_CAPABILITY_CACHE: dict[str, set[str]] = {}


def reset_executable_cache() -> None:
    """Forget resolved executables and verified capabilities (used by tests)."""
    _EXECUTABLE_CACHE.clear()
    _CAPABILITY_CACHE.clear()


def _explicit_executable(env_var: str) -> str | None:
    explicit = os.environ.get(env_var)
    if not explicit:
        return None
    if not _executable_responds(explicit):
        raise SystemExit(
            f"La variable {env_var} apunta a un ejecutable que no responde: {explicit!r}. "
            f"Corrige esa variable o retírala; no se usará otro ejecutable en su lugar."
        )
    return explicit


def _resolve_ffmpeg() -> str:
    explicit = _explicit_executable("FFMPEG")
    if explicit is not None:
        return explicit
    on_path = shutil.which("ffmpeg")
    if on_path and _executable_responds(on_path):
        return on_path
    try:
        import imageio_ffmpeg
    except Exception:
        imageio_ffmpeg = None  # type: ignore[assignment]
    if imageio_ffmpeg is not None:
        try:
            bundled = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            bundled = None
        if bundled and _executable_responds(bundled):
            return bundled
    raise SystemExit(
        "No se encontró un FFmpeg válido. Define la variable FFMPEG, instala ffmpeg "
        "en el PATH o instala imageio-ffmpeg (pip install imageio-ffmpeg)."
    )


def resolve_ffmpeg() -> str:
    """Resolve the FFmpeg executable to use, cached per process.

    Order: explicit ``FFMPEG`` (invalid -> hard error without fallback), a
    responding ``ffmpeg`` on PATH, then the binary bundled with imageio-ffmpeg.
    """
    if "ffmpeg" in _EXECUTABLE_CACHE:
        return _EXECUTABLE_CACHE["ffmpeg"]  # type: ignore[return-value]
    exe = _resolve_ffmpeg()
    _EXECUTABLE_CACHE["ffmpeg"] = exe
    return exe


def _resolve_ffprobe() -> str | None:
    explicit = _explicit_executable("FFPROBE")
    if explicit is not None:
        return explicit
    on_path = shutil.which("ffprobe")
    if on_path and _executable_responds(on_path):
        return on_path
    return None


def resolve_ffprobe() -> str | None:
    """Resolve the ffprobe executable, cached per process; ``None`` enables the
    OpenCV metadata fallback."""
    if "ffprobe" in _EXECUTABLE_CACHE:
        return _EXECUTABLE_CACHE["ffprobe"]
    exe = _resolve_ffprobe()
    _EXECUTABLE_CACHE["ffprobe"] = exe
    return exe


def ffmpeg_version(exe: str | None = None) -> str:
    """First line of ``ffmpeg -version`` for the resolved executable."""
    exe = exe or resolve_ffmpeg()
    raw = run_capture([exe, "-version"])
    first = raw.splitlines()[0].strip() if raw.strip() else exe
    return first


_RAW_TEST_INPUT = "testsrc=size=32x32:rate=10:duration=0.2"


def _check_raw_pipeline(exe: str, vf: str, pix_fmt: str, expected: int) -> None:
    command = [
        exe,
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        _RAW_TEST_INPUT,
        "-vf",
        vf,
        "-frames:v",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "pipe:1",
    ]
    data = _capture_bytes(command)
    if len(data) != expected:
        raise SystemExit(
            f"FFmpeg no entrega el formato raw esperado ({vf} -> {pix_fmt}): obtuvo "
            f"{len(data)} bytes, esperaba {expected}. Revisa el ejecutable FFmpeg."
        )


def _check_libx264(exe: str) -> None:
    command = [
        exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        _RAW_TEST_INPUT,
        "-frames:v",
        "1",
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
        "-f",
        "null",
        "-",
    ]
    _capture_bytes(command)


def ensure_capabilities(need_encoder: bool = False) -> str:
    """Verify (cached per executable) the resolved FFmpeg's real capabilities.

    Runs the exact raw pipelines the program uses (``gray`` with
    ``scale=area`` and ``bgr24`` with ``scale=lanczos``) once per executable;
    a later encoder-only request reuses that result and only checks
    ``libx264`` with ``yuv420p``.  Returns the executable.
    """
    exe = resolve_ffmpeg()
    verified = _CAPABILITY_CACHE.setdefault(exe, set())
    if "decode" not in verified:
        _check_raw_pipeline(exe, "scale=16:16:flags=area,format=gray", "gray", 16 * 16)
        _check_raw_pipeline(exe, "scale=16:16:flags=lanczos", "bgr24", 16 * 16 * 3)
        verified.add("decode")
    if need_encoder and "encoder" not in verified:
        _check_libx264(exe)
        verified.add("encoder")
    return exe


_VFRDET_RE = re.compile(r"VFR:([0-9.eE+NnIi-]+) \(([0-9]+)/([0-9]+)\)")


def parse_vfrdet_output(raw: str) -> dict | None:
    """Parse the last valid vfrdet measurement from an FFmpeg pass.

    ``VFR:x (vfr/cfr)`` (newer builds append `` min:/max:/avg:`` timestamps;
    older ones print ``CFR:y``) reports how many of the compared transitions
    deviated from a constant cadence.  Returns ``{"vfr": int, "transitions":
    int, "frames": int}`` where ``frames = vfr + cfr + 1`` and ``vfr`` counts
    the irregular transitions; ``vfr == 0`` means CFR.  ``None`` when no valid
    final measurement exists (missing, empty or inconclusive).
    """
    best: dict | None = None
    for match in _VFRDET_RE.finditer(raw):
        vfr_text = match.group(1)
        try:
            vfr_value = float(vfr_text)
        except ValueError:
            continue
        if not math.isfinite(vfr_value):
            continue
        vfr = int(match.group(2))
        cfr = int(match.group(3))
        transitions = vfr + cfr
        if transitions <= 0:
            continue
        best = {"vfr": vfr, "transitions": transitions, "frames": transitions + 1}
    return best


def verify_exact_cfr(video: Path, info: VideoInfo) -> VideoInfo:
    """Upgrade provisional metadata with one full FFmpeg ``vfrdet`` pass.

    Used when ``frame_count_exact`` is False, which happens either when ffprobe
    estimated the count from ``duration * fps`` or when the metadata came from
    the OpenCV fallback.  Requires CFR cadence and a verified frame count equal
    to the estimate; a build without the ``vfrdet`` filter, VFR, a count
    mismatch or inconclusive output aborts with an actionable error before any
    cache or output destination is created.  On success the returned
    ``VideoInfo`` is exact.
    """
    exe = resolve_ffmpeg()
    try:
        raw = run_capture(
            [
                exe,
                "-hide_banner",
                "-loglevel",
                "info",
                "-i",
                str(video),
                "-vf",
                "vfrdet",
                "-an",
                "-f",
                "null",
                "-",
            ]
        )
    except SystemExit as exc:
        lowered = str(exc).lower()
        if "vfrdet" in lowered and (
            "no such filter" in lowered
            or "not available" in lowered
            or "unknown filter" in lowered
        ):
            raise SystemExit(
                "Este FFmpeg no incluye el filtro 'vfrdet' (versión antigua o build sin "
                "vfrdet). No se puede verificar la cadencia CFR: instala un FFmpeg reciente "
                "o define la variable FFMPEG apuntando a uno que incluya vfrdet."
            ) from exc
        raise
    parsed = parse_vfrdet_output(raw)
    if parsed is None:
        raise SystemExit(
            "La verificación CFR del vídeo no pudo concluir (sin medida vfrdet válida); "
            "se cancela antes de generar caché o salida."
        )
    if parsed["vfr"] != 0:
        raise SystemExit(
            f"El vídeo tiene cadencia variable (vfrdet: {parsed['vfr']} desviaciones en "
            f"{parsed['frames']} frames); se rechaza porque el análisis exige CFR."
        )
    if parsed["frames"] != info.frames:
        raise SystemExit(
            f"El conteo verificado ({parsed['frames']} frames) difiere del estimado "
            f"({info.frames}); se cancela antes de generar caché o salida."
        )
    return dataclasses.replace(info, frames=parsed["frames"], frame_count_exact=True)


def find_video(explicit: str | None) -> Path:
    """Validate an explicit video path; silent auto-detection is removed."""
    if not explicit:
        raise SystemExit("Falta el vídeo. Uso: python eclipse_stabilizer.py <comando> VIDEO")
    path = Path(explicit).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"No existe el vídeo: {path}")
    return path


def default_output_dir(video: Path) -> Path:
    """Deterministic per-video output folder under ROOT, never eclipse_output.

    The folder is derived from a sanitized version of the source stem so two
    different videos never share a default directory.  It is not created here;
    callers create it only after the video and output have been resolved.
    """
    resolved = Path(video).expanduser().resolve()
    stem = resolved.stem
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "video"
    path_key = os.path.normcase(str(resolved))
    path_hash = hashlib.sha256(path_key.encode("utf-8")).hexdigest()[:8]
    return ROOT / f"{slug}_{path_hash}_output"


def resolve_output_dir(explicit: str | None, video: Path) -> Path:
    """Explicit --out always wins; otherwise derive one from the video stem."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    return default_output_dir(video)


def _positive_int(value: object, label: str) -> int | None:
    """Positive integer from ffprobe metadata; None for absent or N/A values."""
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if text in ("N/A", ""):
        return None
    try:
        parsed = int(text)
    except ValueError:
        raise SystemExit(f"ffprobe devolvió un {label} no válido: {value!r}.") from None
    if parsed <= 0:
        raise SystemExit(f"ffprobe devolvió un {label} no positivo: {value!r}.")
    return parsed


def _positive_duration(value: object) -> float | None:
    """Positive finite duration from ffprobe metadata; None if unusable."""
    if value is None:
        return None
    try:
        text = str(value).strip()
        if text in ("N/A", ""):
            return None
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _positive_fraction(value: object) -> float | None:
    """Positive finite value of an ffprobe \"num/den\" fraction; None if unusable."""
    if value is None:
        return None
    try:
        text = str(value).strip()
        if text in ("N/A", ""):
            return None
        num, den = text.split("/", 1)
        parsed = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def parse_probe_json(raw: str) -> tuple[VideoInfo, bool]:
    """Parse and validate ffprobe JSON metadata without external processes.

    Returns ``(info, frames_estimated)``.  ``frames_estimated`` is True only
    when ``nb_frames`` was absent or ``N/A`` and the count was estimated as
    ``duration * fps``; an estimate is never presented as an exact count.
    Raises ``SystemExit`` with an actionable Spanish message for malformed
    JSON, a missing video stream, invalid dimensions or unresolvable
    FPS/duration, without leaking a traceback.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise SystemExit("ffprobe devolvió JSON no válido; no se pudieron leer los metadatos del vídeo.") from None
    if not isinstance(data, dict):
        raise SystemExit("ffprobe devolvió metadatos no válidos; no se pudieron leer los metadatos del vídeo.")
    streams = data.get("streams")
    if not isinstance(streams, list) or not streams:
        raise SystemExit("ffprobe no encontró ningún stream de vídeo; comprueba que el archivo sea un vídeo.")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise SystemExit("ffprobe devolvió metadatos no válidos; no se pudieron leer los metadatos del vídeo.")
    width = _positive_int(stream.get("width"), "ancho")
    if width is None:
        raise SystemExit("ffprobe no devolvió un ancho válido; no se puede procesar el vídeo.")
    height = _positive_int(stream.get("height"), "alto")
    if height is None:
        raise SystemExit("ffprobe no devolvió un alto válido; no se puede procesar el vídeo.")
    fmt = data.get("format") if isinstance(data.get("format"), dict) else {}
    duration = _positive_duration(fmt.get("duration"))
    if duration is None:
        duration = _positive_duration(stream.get("duration"))
    nb_frames = _positive_int(stream.get("nb_frames"), "nb_frames")
    fps = _positive_fraction(stream.get("avg_frame_rate"))
    if fps is None:
        fps = _positive_fraction(stream.get("r_frame_rate"))
    if fps is None and nb_frames is not None and duration is not None:
        fps = nb_frames / duration
    if fps is None:
        raise SystemExit("ffprobe no permite derivar los FPS del vídeo; no se asumirá una cadencia arbitraria.")
    if duration is None:
        if nb_frames is None:
            raise SystemExit("ffprobe no permite derivar la duración del vídeo.")
        duration = nb_frames / fps
    if nb_frames is None:
        nb_frames = int(round(duration * fps))
        estimated = True
    else:
        estimated = False
    return (
        VideoInfo(
            width,
            height,
            nb_frames,
            fps,
            duration,
            probe_source="ffprobe",
            frame_count_exact=not estimated,
        ),
        estimated,
    )


def probe_video(video: Path) -> VideoInfo:
    """Probe metadata via ffprobe when available, otherwise OpenCV provisional."""
    ffprobe = resolve_ffprobe()
    if ffprobe is None:
        return probe_opencv(video)
    raw = run_capture(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,avg_frame_rate,r_frame_rate,duration:format=duration",
            "-of",
            "json",
            str(video),
        ]
    )
    info, _ = parse_probe_json(raw)
    return info


def probe_opencv(video: Path) -> VideoInfo:
    """Provisional OpenCV metadata fallback used when ffprobe is unavailable.

    ``CAP_PROP_FRAME_COUNT`` is an estimate, so the result is explicitly
    provisional: positive coherent values are reported for ``inspect``, but
    ``frame_count_exact`` stays False until a full CFR verification upgrades
    it.  Raises ``SystemExit`` when OpenCV cannot produce positive values.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(
            f"OpenCV no puede abrir {video} y no hay ffprobe disponible; "
            "no se pudieron obtener metadatos del vídeo."
        )
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if (
        width <= 0
        or height <= 0
        or not math.isfinite(fps)
        or fps <= 0
        or not math.isfinite(frames)
        or frames <= 0
    ):
        raise SystemExit(
            "El fallback OpenCV no pudo producir metadatos positivos y coherentes; "
            "instala ffprobe o revisa el vídeo."
        )
    duration = frames / fps
    return VideoInfo(
        width,
        height,
        frames,
        fps,
        duration,
        probe_source="opencv",
        frame_count_exact=False,
    )


def scaled_shape(info: VideoInfo, width: int) -> tuple[int, int]:
    height = int(round(info.height * width / info.width))
    height += height % 2
    return int(width), int(height)


def print_video_info(video: Path, info: VideoInfo) -> None:
    print(f"Vídeo      : {video}")
    print(f"Resolución : {info.width}x{info.height}")
    sequence = f"{info.frames} frames, {info.fps:.3f} fps, {info.duration:.2f} s"
    if not info.frame_count_exact:
        sequence += " (metadatos provisionales; cadencia/conteo sin verificar)"
    print(f"Secuencia  : {sequence}")


def read_frame_at(cap: cv2.VideoCapture, frame_idx: int, width: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, bgr = cap.read()
    if not ok or bgr is None:
        return None
    h, w = bgr.shape[:2]
    out_h = int(round(h * width / w))
    return cv2.resize(bgr, (width, out_h), interpolation=cv2.INTER_AREA)


def sparse_frames(video: Path, indices: np.ndarray, width: int) -> list[tuple[int, np.ndarray]]:
    """Read isolated frames using seeks; this does not decode the full sequence."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"OpenCV no puede abrir {video}")
    progress = Progress("seek aislado", len(indices), interval=0.25)
    result: list[tuple[int, np.ndarray]] = []
    try:
        for pos, idx in enumerate(indices, 1):
            frame = read_frame_at(cap, int(idx), width)
            if frame is not None:
                result.append((int(idx), frame))
            progress.update(pos, force=pos == len(indices))
    finally:
        cap.release()
    return result


def iter_ffmpeg_gray(video: Path, width: int, height: int, frames: int, *, exact_total: bool = False):
    """Yield analysis-resolution grayscale frames as (idx, gray).

    ``exact_total=True`` means ``frames`` is the verified total that must end
    at EOF: it requests ``frames + 1``, delivers only ``frames``, and aborts
    when a sentinel frame proves the source is longer (underestimated) or when
    FFmpeg ends early (overestimated), so a silent truncation is impossible.
    ``exact_total=False`` decodes an intentionally bounded prefix and never
    reads a sentinel.
    """
    requested = frames + 1 if exact_total else frames
    command = [
        resolve_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"scale={width}:{height}:flags=area,format=gray",
        "-frames:v",
        str(requested),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=4 * width * height)
    assert proc.stdout is not None
    size = width * height
    delivered = 0
    try:
        for idx in range(requested):
            data = proc.stdout.read(size)
            if len(data) != size:
                break
            if exact_total and idx >= frames:
                raise SystemExit(
                    f"El conteo real del vídeo supera el estimado ({frames} frames): se "
                    "detectó un frame centinela. Se cancela para no truncar la secuencia."
                )
            yield idx, np.frombuffer(data, np.uint8).reshape(height, width)
            delivered += 1
    finally:
        proc.stdout.close()
        if proc.stderr is not None:
            stderr = proc.stderr.read()
            proc.stderr.close()
        else:
            stderr = b""
        proc.wait()
        if proc.returncode not in (0, None) and sys.exc_info()[0] is None:
            raise RuntimeError(stderr.decode(errors="replace"))
    if exact_total and delivered != frames:
        raise SystemExit(
            f"FFmpeg entregó {delivered}/{frames} frames; el conteo estimado era "
            "demasiado alto. Se cancela para no generar una secuencia incompleta."
        )


def iter_ffmpeg_bgr(video: Path, width: int, height: int, frames: int, *, exact_total: bool = False):
    """Yield full-resolution BGR frames as (idx, bgr).

    ``exact_total`` follows the same contract as ``iter_ffmpeg_gray``: with
    True the count must end at EOF (sentinel/short abort), with False it is an
    intentionally bounded prefix such as ``trim_end + 1``.
    """
    requested = frames + 1 if exact_total else frames
    command = [
        resolve_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        f"scale={width}:{height}:flags=lanczos",
        "-frames:v",
        str(requested),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=3 * width * height * 2)
    assert proc.stdout is not None
    size = width * height * 3
    delivered = 0
    try:
        for idx in range(requested):
            data = proc.stdout.read(size)
            if len(data) != size:
                break
            if exact_total and idx >= frames:
                raise SystemExit(
                    f"El conteo real del vídeo supera el estimado ({frames} frames): se "
                    "detectó un frame centinela. Se cancela para no truncar la secuencia."
                )
            yield idx, np.frombuffer(data, np.uint8).reshape(height, width, 3)
            delivered += 1
    finally:
        proc.stdout.close()
        if proc.stderr is not None:
            stderr = proc.stderr.read()
            proc.stderr.close()
        else:
            stderr = b""
        proc.wait()
        if proc.returncode not in (0, None) and sys.exc_info()[0] is None:
            raise RuntimeError(stderr.decode(errors="replace"))
    if exact_total and delivered != frames:
        raise SystemExit(
            f"FFmpeg entregó {delivered}/{frames} frames; el conteo estimado era "
            "demasiado alto. Se cancela para no generar una secuencia incompleta."
        )
