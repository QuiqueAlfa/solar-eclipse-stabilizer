# FFmpeg and ffprobe

This document describes executable discovery, capability checks and strict media
verification. See the [README](../README.md#supported-platforms) for the platforms and
FFmpeg versions that have actually been validated.

## Executable resolution

Executables are resolved at run time, never pinned during module import.

FFmpeg resolution order:

1. Explicit `FFMPEG` environment variable.
2. `ffmpeg` on `PATH`.
3. The binary packaged by `imageio-ffmpeg`.

ffprobe resolution order:

1. Explicit `FFPROBE` environment variable.
2. `ffprobe` on `PATH`.
3. OpenCV metadata fallback.

An explicit `FFMPEG` or `FFPROBE` value is authoritative. If that path is invalid, the
command fails with an actionable error and does **not** fall back to another binary.

Every command reports the selected tools and versions at startup:

```text
FFmpeg      : C:\path\to\ffmpeg.exe
Versión FFmpeg: ffmpeg version 8.1.1-full_build-www.gyan.dev ...
FFprobe     : C:\path\to\ffprobe.exe
```

When ffprobe is unavailable, the third line is:

```text
FFprobe     : OpenCV fallback
```

## Selecting a specific build

PowerShell:

```powershell
$env:FFMPEG  = "C:\path\to\ffmpeg.exe"
$env:FFPROBE = "C:\path\to\ffprobe.exe"
python eclipse_stabilizer.py inspect VIDEO
```

Linux/macOS shell:

```bash
export FFMPEG=/usr/local/bin/ffmpeg
export FFPROBE=/usr/local/bin/ffprobe
python eclipse_stabilizer.py inspect VIDEO
```

The POSIX example documents configuration only; it is not a claim that the project has
been validated on Linux or macOS.

## Optional ffprobe and strict fallback verification

ffprobe is preferred because it provides exact stream metadata cheaply. It is not a
hard dependency.

Without ffprobe, OpenCV supplies provisional width, height, FPS, duration and frame
count. `inspect` labels that metadata as **provisional**. Before `analyze`, `preview` or
`export` may save or replace any analysis cache or video, the tool upgrades the
metadata through one complete-stream verification:

1. FFmpeg decodes the stream through the `vfrdet` filter.
2. The result must conclusively report CFR.
3. A complete raw decode must deliver the exact expected frame count, including an
   overrun sentinel check.

The command stops before publishing output when:

- the stream is variable-frame-rate;
- `vfrdet` is inconclusive;
- the decoded frame count differs from the expected count;
- FFmpeg ends early or emits an extra frame;
- the selected FFmpeg build lacks the `vfrdet` filter.

Errors explain the failed requirement instead of silently trusting provisional
metadata. The exact-count result participates in cache identity, so an unverified and
a verified probe cannot share a cache accidentally.

## Capability checks

The selected FFmpeg binary is tested with real decode/filter pipelines rather than
accepted only because `ffmpeg -version` succeeds. The checks verify that it can:

- decode media to raw grayscale frames for analysis;
- decode media to raw BGR frames for rendering/debug operations;
- provide the filters required by strict CFR verification.

`libx264` is required only for commands that actually encode video: `preview`,
`export`, and `analyze --debug`. Plain `inspect` and `analyze` require decoding but do
not fail merely because an encoder is unavailable.

The packaged `imageio-ffmpeg` binary is a final FFmpeg fallback, not an ffprobe
fallback. If no ffprobe exists, metadata follows the OpenCV-plus-full-verification path
described above.

## Output media contract

Preview and export video use H.264 through `libx264`, `yuv420p`, no audio (`-an`) and
`+faststart`. Final export preserves source width, height and FPS at 1×; preview may be
smaller and accelerated.

Encoding is atomic: the program writes a unique temporary sibling, validates the
result where applicable, and replaces the destination only after success. See
[Technical guide: Output directory and cache](TECHNICAL.md#output-directory-and-cache)
for artifacts and publication guarantees.
