# Technical guide

This document contains the implementation, diagnostic and reproducibility details
moved out of the project [README](../README.md). The recommended user path remains
`preview → visual review → export`.

## Detection and tracking model

The stabilizer uses the **outer photospheric limb** as its absolute geometric anchor.
For every frame, threshold-independent radial scans are cast around a temporally
predicted center. Each angular direction votes only when it finds a real outward
intensity drop in a narrow band around the fixed solar radius; valid samples refine a
fixed-radius circle fit. The lunar indentation never votes, so crescent rotation does
not move the anchor and exposure changes do not select a different detection branch.

Three supporting sources operate in parallel and remain auditable per frame:

- **Binary contour** is only a bootstrap or recovery candidate. It initializes the
  measurement and can re-seed after a large confirmed movement, but only when real
  radial outer-edge evidence confirms its proposed center.
- **Phase correlation** supplies relative displacement between consecutive frames. A
  propagated position is never labelled as a measurement: without radial evidence,
  the center is reconstructed or left unresolved.
- **Anchored contrast fallback** continuously re-anchors to the latest reliable
  geometric center and follows the recent crescent offset with bounded innovation. It
  bridges short measurement gaps and the final loss without an abrupt model switch.

The robust path solver suppresses unconfirmed transient spikes and accumulated weak
excursions while preserving geometry-confirmed camera jumps. `--no-auto-repair` keeps
the raw solution for diagnosis; it does not change cache identity.

### Content, centering and provenance

Content and centering are classified independently:

- Content: `usable`, `clipped` or `uncertain`.
- Centering: `reliable`, `reconstructed` or `unresolved`.

Only a confirmed visible clip is excluded automatically. Usable or uncertain content
can be exported when its center is reliable or reconstructed; unresolved centering is
never exported silently.

Per-frame `geometry_source` records whether the center came from a radial measurement,
a radial measurement that disagreed with a contour candidate, a confirmed contour
recovery, or no direct measurement. A phase-propagated prediction alone never becomes
geometric evidence.

### Horizon regime

Near the horizon, confirmation requires sustained combined degradation: radial
coverage, sample count, radius scale and arc shape must deteriorate together with
physical occlusion. Once the irreversible horizon regime is latched, a later isolated
circular shape cannot become a new anchor. The exportable tail can only use the
continuously anchored fallback.

## Diagnostic workflow

Start with the [quick-start preview](../README.md#quick-start). If it is not acceptable,
use the diagnostic commands below.

```text
# Sample isolated frames without decoding the complete sequence
python eclipse_stabilizer.py inspect VIDEO

# Track the complete video and generate detailed review artifacts
python eclipse_stabilizer.py analyze VIDEO --debug
```

`inspect` writes sparse limb fits to `inspect/contact_sheet.jpg` and
`inspect/detections.csv`. `analyze --debug` adds `review.mp4`, `frames.csv`, per-state
JPEG files and a contact sheet under `<output>/debug/`.

After identifying a problem, regenerate the preview and watch it completely before
exporting.

### Manual frame selection

Frame numbers and inclusive ranges are comma-separated:

```text
--drop-frames 20-24,700,900
--keep-frames 220-224,901
```

- `--drop-frames` excludes reviewed source frames.
- `--keep-frames` can recover false positives classified as `CONTENT_CLIPPED`, such as
  a border clip or sensor tear. It cannot force a frame with unresolved centering.
- A frame present in both options is an error.
- Preview and export must receive the same selection; both resolve exactly the same
  effective mask. Manual decisions are recorded in `frames.csv` and the export
  summary.

```text
python eclipse_stabilizer.py preview VIDEO --drop-frames 20-24,700 --keep-frames 220
python eclipse_stabilizer.py export VIDEO --drop-frames 20-24,700 --keep-frames 220
```

## CLI conventions and options

Global options belong **before** the subcommand. Subcommand options belong after
`VIDEO`, as required by `argparse`:

```text
python eclipse_stabilizer.py --force preview VIDEO --speed 2
python eclipse_stabilizer.py --profile profiles/example.json analyze VIDEO --debug
```

### Global options

- `--profile FILE`: optional per-video discard profile.
- `--out DIR`: explicit output directory.
- `--analysis-width N`: low-resolution analysis width; default 270, minimum 160.
- `--radius R`: fixed solar radius in analysis pixels; otherwise calibrated
  automatically from sparse samples.
- `--min-quality Q`: minimum absolute-fit confidence; default 0.18.
- `--force`: ignore and regenerate the analysis cache.
- `--no-auto-repair`: keep the raw tracking solution without the second robust
  weighting pass.

### Subcommands

- `inspect VIDEO`: `--samples` (24), `--start-frame` (0), `--end-frame` (-1 means the
  final frame).
- `analyze VIDEO`: `--debug`, `--debug-width` (320; minimum 64),
  `--debug-max-images` (10000; non-negative), `--drop-frames`, `--keep-frames`.
- `preview VIDEO`: `--preview-width` (270), `--speed` (2.0), `--debug-overlay`,
  `--drop-frames`, `--keep-frames`.
- `export VIDEO`: `--name` (`stabilized.mp4`), `--crf` (18), `--preset` (`fast`),
  `--threads` (2), `--drop-frames`, `--keep-frames`.

Every subcommand requires an explicit `VIDEO`; the tool never selects one implicitly.

## Per-video profiles

A profile is an optional, manually written JSON record of reviewed source-frame
discards. It is not generated by `inspect`, `analyze`, `preview` or `export`.

Supported top-level fields are `version`, `name`, `description` and `discards`.
`name`, `description` and per-entry `reason` values are human metadata; only
`discards` changes the output. Indices are zero-based and specific to one immutable
source recording. Never reuse a profile with another video.

[`profiles/example.json`](../profiles/example.json) demonstrates individual frames and
inclusive ranges. Its values are illustrative and must be replaced after reviewing
the target video.

Pass the same profile to every relevant command:

```text
python eclipse_stabilizer.py --profile profiles/my-video.json analyze VIDEO --debug
python eclipse_stabilizer.py --profile profiles/my-video.json preview VIDEO
# Watch and approve preview.mp4
python eclipse_stabilizer.py --profile profiles/my-video.json export VIDEO
```

Changing the profile path or its `discards`, or omitting the profile, changes cache
identity so results made with different decisions cannot be mixed silently.

## Output directory and cache

By default, results go to a per-video directory derived from the source stem and a
short hash of its resolved path: `<slug>_<8-hex>_output/`. Two videos never share a
default directory. `--out DIR` overrides it.

`analysis.npz` stores expensive detections and the solved per-frame trajectory. The
cache schema is versioned (currently `CACHE_VERSION = 6`) and its identity includes:

- resolved source path, size and modification time;
- source width, height, FPS, frame count and whether that count is exact;
- analysis width and height;
- requested or automatically calibrated radius mode;
- profile path and operational profile hash;
- minimum quality;
- cache, visible-track, contrast, clipping and temporal-loss algorithm versions.

Any incompatible identity or missing required array causes a full re-analysis.
`--force` always regenerates the cache. A contrast-version-only mismatch is
refreshable: geometric detections can be reused while the anchored contrast track is
recomputed.

## Artifacts

- `<output>/inspect/contact_sheet.jpg`: sparse visual limb-fit review.
- `<output>/inspect/detections.csv`: sparse numerical fits.
- `<output>/analysis.npz`: reusable detections, trajectory, classifications, cache
  identity and provenance.
- `<output>/tracking.csv`: per-frame geometric source, measurement/reconstruction
  state, content state, horizon regime and tracking diagnostics.
- `<output>/preview.mp4`: low-resolution preview, accelerated by default.
- `<output>/validation.json`: full preview validation results.
- `<output>/debug/review.mp4`: complete diagnostic review video.
- `<output>/debug/frames.csv`: detailed frame decisions and reasons.
- `<output>/debug/frames/{usable,clipped,uncertain}/`: optional per-state JPEGs.
- `<output>/debug/contact_sheet.jpg`: prioritized clipped/uncertain frames.
- `<output>/stabilized.mp4`: full-resolution final export; `--name` changes its name.

The analysis cache includes radial-arc measurements (`arc_center`, `arc_measured`,
valid-point count, coverage, residual, strength and gap), geometric prediction and
innovation, contrast fallback diagnostics, exposed-photosphere measurements and
temporal content-loss diagnostics. These fields make each center and exclusion
auditable rather than hiding them behind one confidence value.

Preview, export and debug review videos are published atomically. Encoding occurs in a
unique temporary sibling and `os.replace` publishes it only after successful encoding
and, where applicable, validation. A failed run never replaces an existing destination
with a partial file and cleans up its temporary output.

## Windows helper

[`run.ps1`](../run.ps1) is optional. It creates `.venv` when missing, installs
[`requirements.txt`](../requirements.txt), then forwards an explicit command and
`VIDEO` to `eclipse_stabilizer.py`. Without both values it prints usage and exits
non-zero; it never auto-detects a video.

```powershell
.\run.ps1 preview VIDEO
# Watch and approve preview.mp4
.\run.ps1 export VIDEO
```

## Tests

Run the suite from the repository root:

```text
python -m unittest -v
```

All fixtures are synthetic; no real-video frames or private data are included. The
suite covers:

- radial-limb invariance under lunar-bite rotation, known translations and exposure
  changes;
- short arcs without usable evidence, bootstrap, phase prediction and confirmed large
  camera jumps;
- rejection of contours without radial confirmation and blank/corrupt frames that
  must never become anchors;
- robust spike/excursion suppression, preservation of real movement and the
  `--no-auto-repair` path;
- horizon confirmation, transient recovery, post-horizon false-circle rejection and
  framewise tail policy;
- the continuously anchored contrast fallback: re-anchoring, evolving offset,
  bounded innovation and no-history behavior;
- content/centering classification, contradicted reconstruction, clipping, normalized
  exposed-area measurement, smooth eclipse evolution, brief content loss and
  low-exposure abstention;
- manual drop/keep policy, per-video profile discards, debug limits and shared
  preview/export selection;
- cache identity, schema versions, required arrays, invalidation and contrast-only
  refresh;
- FFmpeg/ffprobe resolution priority, real capability checks and the requirement for
  `libx264` only when encoding is requested;
- `vfrdet` parsing, acceptance of CFR/VFR input, correction of provisional frame-count
  mismatches, and rejection of inconclusive output or builds missing the required
  filter;
- provisional OpenCV metadata and its upgrade to exact verified metadata;
- atomic publication and cleanup for preview, export and debug review, including
  broken pipes, encoder/decoder failures and `os.replace` failure.

Media resolution and verification behavior are detailed in [FFMPEG.md](FFMPEG.md).
