# solar-eclipse-stabilizer

A geometry-first video stabilizer for **partial solar eclipses**. It measures the
outer photospheric limb and keeps the solar center fixed at the exact center of each
exported frame while the source camera moves, pans or drifts.

The normal workflow is intentionally short: **preview → visual review → export**.
Detection, tracking and media-validation details are documented separately.

## Demo

Web-optimized side-by-side preview of a real partial-eclipse recording:

https://github.com/user-attachments/assets/644b9df7-2fc5-454f-bf4d-25e6ce6e1bff

The full-resolution comparison is available from release `v0.1.0`:

https://github.com/QuiqueAlfa/solar-eclipse-stabilizer/releases/tag/v0.1.0

- **BEFORE** (left): all **5657 source frames**, played at **3×**.
- **AFTER** (right): **5580 unique exported frames**, retimed with timestamps only
  by `3 × 5580 / 5657 ≈ 2.959166×` so both panels share a visual duration of
  **~62.844s**.
- AFTER does **not** reconstruct, interpolate, insert or duplicate frames. The two
  panels are **not source frame-to-frame synchronized**.
- The **77 documented exclusions** are 70 clipped frames (69 border-clipped and one
  temporary photosphere loss), 4 uncertain/unresolved frames and 3 usable frames
  explicitly discarded after review.

The accounting is complete: `5657 = 5580 + 77`; no problematic frame is hidden.

## Support tiers

- **Partial eclipses — supported.** This is the primary and validated use case.
- **Annular eclipses — experimental.** The outer limb remains available in principle,
  but the tool has not yet been validated on a real annular recording.
- **Total eclipses — not supported during totality.** The bright photospheric limb
  disappears and the corona is not a substitute. Partial phases may be processed
  separately, but no continuous-totality promise is made.

## Requirements

- Python **3.10+**.
- FFmpeg for decoding, encoding and strict CFR verification.
- ffprobe is preferred but optional; a strict fallback verification is used when it
  is unavailable.
- Python dependencies from [`requirements.txt`](requirements.txt): NumPy,
  headless OpenCV and `imageio-ffmpeg`.

See [FFmpeg and ffprobe](docs/FFMPEG.md) for executable resolution, environment
variables and fallback behavior.

## Installation

From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Linux/macOS instructions are provided for portability, but those platforms have not
yet been validated for this project.

The commands below assume that the virtual environment is active.

## Quick start

The CLI always requires an explicit `VIDEO` path; it never auto-detects a source.
`VIDEO` means the path to the input video file, including its filename and extension.
Quote the path when it contains spaces. For example, on Windows:

```powershell
python eclipse_stabilizer.py preview "C:\Users\Alice\Videos\partial-eclipse.mp4"
```

1. Create a low-resolution preview:

   ```text
   python eclipse_stabilizer.py preview VIDEO
   ```

   If needed, this command automatically analyzes the complete video and creates a
   reusable cache before writing `<output>/preview.mp4`.

2. **Watch the complete preview.** Confirm that the Sun stays fixed, camera movement
   is compensated without jumps, and any excluded tail or clipped frames are expected.

3. Only after approving the preview, export at full source resolution:

   ```text
   python eclipse_stabilizer.py export VIDEO
   ```

   `export` reuses the compatible analysis cache. It can analyze without an existing
   cache, but exporting without first reviewing a preview is not recommended.

The default output folder is derived safely from the source name and path. On Windows,
the optional helper provides the same flow:

```powershell
.\run.ps1 preview VIDEO
# Watch and approve preview.mp4
.\run.ps1 export VIDEO
```

If the preview needs investigation or manual frame decisions, see the
[technical guide](docs/TECHNICAL.md).

## Tests

```text
python -m unittest -v
```

The synthetic suite contains no real-video frames or private data. Its full coverage
and media-layer validation are described in the [technical guide](docs/TECHNICAL.md#tests).

## Supported platforms

Only **Windows x86-64** has been validated: with system FFmpeg **8.1.1** and the
`imageio-ffmpeg` packaged binary **7.1**. POSIX setup examples and available wheels do
not constitute a Linux/macOS support claim; behavior must be re-verified with the
resolved FFmpeg build.

The project is not distributed through PyPI. Run it from a clone or download of this
repository with the dependencies in `requirements.txt`.

## Limitations

- Translation stabilization only; no rotation and no square export.
- Final export preserves source width, height and FPS at 1×.
- Preview is low resolution and accelerated by default (`--speed 2`).
- Audio is removed.
- Output is re-encoded as H.264 (`libx264`, `yuv420p`, `+faststart`).
- Annular eclipses remain experimental; totality is unsupported.

## Technical documentation

- [Detection, tracking, diagnostics, profiles, cache and artifacts](docs/TECHNICAL.md)
- [FFmpeg, ffprobe and strict media verification](docs/FFMPEG.md)

## License

Code and documentation are licensed under the [MIT License](LICENSE). Audiovisual
material —including the original recording, stabilized output and comparison demo— is
a separate work and is **not** covered by MIT; see [MEDIA_RIGHTS.md](MEDIA_RIGHTS.md).
Dependency and binary notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Contact

Please use **GitHub Issues** on this repository for questions, bug reports and
feedback, and mention **`@QuiqueAlfa`** where a direct reply is needed. This project
does not publish an email address.
