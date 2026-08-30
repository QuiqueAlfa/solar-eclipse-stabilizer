# Third-Party Notices

This project itself remains licensed under the MIT License (see `LICENSE`). The
notices below cover the third-party packages and binaries the project may
resolve, use or redistribute. **These notes are informational and are not legal
advice.** For authoritative terms, read each dependency's own license text and
the notices of the concrete build in use.

## NumPy — BSD-3-Clause

- https://github.com/numpy/numpy/blob/main/LICENSE.txt

NumPy is distributed under the BSD-3-Clause license.

## OpenCV (>= 4.5) — Apache-2.0

- https://opencv.org/license/
- Upstream `LICENSE`: https://github.com/opencv/opencv/blob/4.x/LICENSE

OpenCV 4.5 and later are licensed under the Apache License 2.0. OpenCV does not
guarantee identical codec availability on every platform: the set of decoders
and encoders depends on the concrete build.

## imageio-ffmpeg — BSD-2-Clause (Python wrapper)

- https://github.com/imageio/imageio-ffmpeg/blob/main/LICENSE

The `imageio-ffmpeg` Python wrapper is licensed under BSD-2-Clause. Its PyPI
wheels may include separate platform-specific FFmpeg executables. The wrapper's
BSD license does **not** relicense that bundled FFmpeg binary: the binary keeps
whatever license its upstream build is distributed under.

## FFmpeg — LGPL-2.1-or-later by default; GPL-2.0-or-later for some builds

- https://ffmpeg.org/legal.html
- https://github.com/FFmpeg/FFmpeg/blob/master/LICENSE.md

FFmpeg is licensed LGPL-2.1-or-later by default. Builds configured with
`--enable-gpl`, or that include GPL components such as `libx264`, are
GPL-2.0-or-later.

## x264 — GNU GPL

- https://www.videolan.org/developers/x264.html

x264 is distributed under the GNU GPL.

## What this means for this project

This project verifies at run time that the resolved FFmpeg build actually
provides `libx264` before any encode. Because the exact executable is resolved
at run time (explicit `FFMPEG`, `PATH`, or the `imageio-ffmpeg` bundled
binary), the **concrete resolved build's own license and configuration control**
the obligations that apply to that binary. Users and distributors must inspect
`ffmpeg -version` of the resolved executable and the upstream notices of that
build; the descriptions above reflect typical distributions but do not
determine any particular build's licensing.