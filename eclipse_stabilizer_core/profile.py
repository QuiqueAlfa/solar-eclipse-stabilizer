from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .constants import CENTER_UNRESOLVED, CONTENT_CLIPPED


def load_profile(path: str | None) -> dict | None:
    """Load and validate the optional per-video JSON profile.

    The profile may only declare human metadata (``version``, ``name``,
    ``description``) and frame ``discards``.  Without a profile no frame is
    dropped by its index.
    """
    if not path:
        return None
    profile_path = Path(path).expanduser().resolve()
    if not profile_path.is_file():
        raise SystemExit(f"No existe el perfil: {profile_path}")
    try:
        with profile_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Perfil inválido ({profile_path}): {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Perfil inválido ({profile_path}): debe ser un objeto JSON.")
    unknown = set(data) - {
        "version",
        "name",
        "description",
        "discards",
    }
    if unknown:
        raise SystemExit(
            f"Perfil inválido ({profile_path}): campos no admitidos: {', '.join(sorted(unknown))}."
        )
    version = data.get("version")
    if version != 1:
        raise SystemExit(f"Perfil inválido ({profile_path}): 'version' debe ser 1.")

    discards: list[tuple[int, int]] = []
    for entry in data.get("discards", []):
        if isinstance(entry, dict) and isinstance(entry.get("frame"), int):
            start = end = entry["frame"]
        elif (
            isinstance(entry, dict)
            and isinstance(entry.get("start"), int)
            and isinstance(entry.get("end"), int)
        ):
            start, end = entry["start"], entry["end"]
        else:
            raise SystemExit(
                f"Perfil inválido ({profile_path}): cada 'discards' debe ser "
                "{{'frame': N}} o {{'start': N, 'end': M}}."
            )
        if start < 0 or end < start:
            raise SystemExit(
                f"Perfil inválido ({profile_path}): rango de descarte inválido {start}-{end}."
            )
        discards.append((start, end))

    return {
        "version": version,
        "discards": discards,
    }


def parse_frame_spec(text: str | None, frame_count: int) -> np.ndarray:
    """Parse a compact frame spec such as ``20-24,700, 900``.

    Accepts individual frames and inclusive ranges separated by commas, with
    optional spaces.  Duplicates are normalized away and the result is sorted.
    Negatives, reverse ranges, malformed tokens and values beyond the video are
    rejected with a clear error instead of being silently ignored.
    """
    if frame_count <= 0:
        raise SystemExit("El vídeo no tiene frames que seleccionar.")
    if text is None or not str(text).strip():
        return np.array([], dtype=np.int64)
    result: list[int] = []
    for raw_part in str(text).split(","):
        part = raw_part.strip()
        if not part:
            raise SystemExit("Selección inválida: hay un elemento vacío entre comas.")
        if part.startswith("-"):
            raise SystemExit(f"Selección inválida: '{part}' no puede ser un frame negativo.")
        if "-" in part:
            tokens = part.split("-")
            if len(tokens) != 2 or not tokens[0].strip() or not tokens[1].strip():
                raise SystemExit(f"Selección inválida: '{part}' no es un rango frame-frame válido.")
            start_text, end_text = tokens[0].strip(), tokens[1].strip()
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError:
                raise SystemExit(f"Selección inválida: '{part}' no contiene números válidos.") from None
            if start < 0 or end < 0:
                raise SystemExit(f"Selección inválida: '{part}' no puede contener frames negativos.")
            if end < start:
                raise SystemExit(f"Selección inválida: rango inverso '{part}' (el inicio supera el fin).")
            if end >= frame_count:
                raise SystemExit(
                    f"Selección fuera del vídeo: '{part}' supera el último frame {frame_count - 1}."
                )
            result.extend(range(start, end + 1))
        else:
            try:
                value = int(part)
            except ValueError:
                raise SystemExit(f"Selección inválida: '{part}' no es un frame ni un rango válido.") from None
            if value < 0:
                raise SystemExit(f"Selección inválida: '{part}' no puede ser un frame negativo.")
            if value >= frame_count:
                raise SystemExit(
                    f"Selección fuera del vídeo: el frame '{value}' supera el último {frame_count - 1}."
                )
            result.append(value)
    return np.unique(np.asarray(result, dtype=np.int64))


def effective_selection(
    analysis: dict,
    drop: str | None = None,
    keep: str | None = None,
) -> dict:
    """Build the effective per-frame export mask without mutating ``analysis``.

    The automatic ``keep`` from :func:`solve_tracking` is the base.  Manual
    ``--drop-frames`` always exclude.  Manual ``--keep-frames`` only recover a
    frame whose centering is not ``CENTER_UNRESOLVED`` (it is meant to rescue
    false positives of the clipping classifier); forcing an unresolved frame is
    an error.  A frame in both options is an error rather than a silent choice.
    Returns the mask plus a per-frame origin (``auto``/``manual``) and the
    reason text of every manual decision.
    """
    n = int(len(analysis["keep"]))
    auto = np.asarray(analysis["keep"]).astype(bool)
    content = np.asarray(analysis.get("content_state", np.zeros(n, np.int8)))
    centering = np.asarray(analysis.get("centering_state", np.zeros(n, np.int8)))

    drop_idx = parse_frame_spec(drop, n)
    keep_idx = parse_frame_spec(keep, n)
    overlap = np.intersect1d(drop_idx, keep_idx)
    if len(overlap):
        shown = ", ".join(str(int(i)) for i in overlap)
        raise SystemExit(f"Conflicto entre --drop-frames y --keep-frames en: {shown}.")

    mask = auto.copy()
    origin = np.full(n, "auto", dtype="U16")
    reason = np.full(n, "", dtype="U80")
    profile_discarded = np.asarray(
        analysis.get("timed_discarded", np.zeros(n, bool)), bool
    )
    origin[profile_discarded] = "profile"
    reason[profile_discarded] = "descartado por perfil explícito"

    for i in keep_idx:
        i = int(i)
        if int(centering[i]) == CENTER_UNRESOLVED:
            raise SystemExit(
                f"No se puede recuperar el frame {i} con --keep-frames: su centrado está sin "
                "resolver (CENTER_UNRESOLVED); la única acción manual admitida es descartarlo."
            )
        if int(content[i]) != CONTENT_CLIPPED:
            raise SystemExit(
                f"No se puede recuperar el frame {i} con --keep-frames: solo se admiten "
                "falsos positivos clasificados como CONTENT_CLIPPED."
            )
        if profile_discarded[i]:
            raise SystemExit(
                f"No se puede recuperar el frame {i} con --keep-frames: está descartado "
                "por el perfil explícito."
            )
        mask[i] = True
        origin[i] = "manual"
        reason[i] = "recuperado manualmente con --keep-frames"
    for i in drop_idx:
        i = int(i)
        mask[i] = False
        origin[i] = "manual"
        reason[i] = "descartado manualmente con --drop-frames"

    return {"mask": mask, "origin": origin, "reason": reason}


def resolve_selection(args: argparse.Namespace, analysis: dict) -> dict:
    """Single shared resolution used by preview, export and analyze/debug."""
    return effective_selection(
        analysis,
        getattr(args, "drop_frames", None),
        getattr(args, "keep_frames", None),
    )


def print_manual_selection(selection: dict) -> None:
    """Print the manual drops and keeps that changed the automatic mask."""
    mask = selection["mask"].astype(bool)
    origin = selection["origin"]
    dropped = np.flatnonzero((origin == "manual") & ~mask)
    kept = np.flatnonzero((origin == "manual") & mask)
    if not len(dropped) and not len(kept):
        return
    print("  decisiones manuales:")
    if len(dropped):
        print(f"    descartados por --drop-frames: {len(dropped)} frame(s)")
    if len(kept):
        print(f"    recuperados por --keep-frames: {len(kept)} frame(s)")
