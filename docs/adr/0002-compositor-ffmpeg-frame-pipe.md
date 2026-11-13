# ADR 0002: Compositor is NumPy/Pillow frames piped into ffmpeg

**Status:** Accepted, 2026-11-13

## Context

`docs/REVIEW.md` (finding C1) found that Remotion, the compositor named in the original PRD, is
source-available rather than free software, and that companies above a small headcount need a paid
licence. It proposed ffmpeg with NumPy/OpenCV frame rendering, or Manim CE (MIT), as the
replacement. The product needs Arabic text with correct letter joining, a hand-drawn reveal look,
and byte-for-byte repeatable output.

## Decision

Frames are drawn in Python and piped to ffmpeg; no scene-graph or browser renderer is used.

- Each scene is rendered by `compose.render_segment` (`src/narratoflow/render/compose.py`) into
  an FFV1 segment. `FrameSink` (`src/narratoflow/render/encoder.py`) writes HxWx3 uint8 RGB frames
  to an ffmpeg process over `pipe:0` (`-f rawvideo -pix_fmt rgb24`, `-c:v ffv1`).
- Scenes render in parallel: `_run_jobs` (`src/narratoflow/render/build.py`) uses a `spawn`
  `ProcessPoolExecutor`, sized by the `workers` argument of `render` (one process per core, at most
  one per scene, when `None`; inline when 1). The CLI exposes this as `--workers`.
- One final pass per part (`encode_part` in `encoder.py`): concat the segments, burn the libass
  captions (`ass=filename=captions.ass:fontsdir=fonts:shaping=complex`), mux the part's audio, and
  encode with the constants `X264_ARGS` and `AAC_ARGS`. The pass counts the concatenated packets
  and must equal the expected frame count, and writes to a temp name that is renamed only on
  success. `verify_mp4` then checks size, frame count and duration before a part is accepted.
- Output is 9:16 only. A channel that lists the landscape output gets `LANDSCAPE_WARNING`
  ("16:9 is not rendered in v1") in the part's report (`render/build.py`).
- Manim CE was not used.

## Consequences

- Determinism: the worker count does not change the frames; this is pinned by
  `test_workers_do_not_change_frames` in `tests/render/test_build.py`.
- A part whose render inputs are unchanged is skipped. `RenderReport.inputs_key`
  (`src/narratoflow/render/report.py`) is `cache_key("render", RENDER_INPUTS_VERSION, payload)`
  from `inputs_key` in `render/build.py`; a part is kept only if its report matches, its MP4 and
  `.ass` exist, and `verify_mp4` passes (`_kept`). `RENDER_INPUTS_VERSION` must be bumped by any
  change to what the frames or the encode look like (compositor, camera, layouts, caption styling,
  x264 flags). Without a bump a renderer change serves stale videos until `render --force`.
  `RenderResult.skipped` lists the kept parts; the skip is tested in `tests/render/test_build.py`
  (`test_rerun_unchanged_skips_part`).
- libass with HarfBuzz shaping is a hard dependency of the system ffmpeg. `render/ffmpeg.py` fails
  with an install hint (`brew install ffmpeg-full` on macOS) when the `ass` filter is missing or
  when Arabic letters are not joined.
- Rendering is CPU-bound Python plus x264; there is no GPU path. The CPU benchmark
  (`docs/benchmarks/2026-11-13-render-cpu.md`) rendered one 180 s part (5400 frames, 1080x1920,
  30 fps) in 84.5 s with the default workers and 181.8 s with one worker, on one 16-core arm
  machine with ffmpeg-full 9.0.2 (single runs, no variance known), under the 600 s NFR.

## Alternatives considered

- **Remotion:** rejected for its licence (REVIEW C1).
- **Manim CE:** MIT-licensed, but it is a mathematical-animation framework with its own Cairo/LaTeX
  stack; the per-frame NumPy pipeline already gives full control of the reveal and of Arabic
  shaping through libass. Not used, and no code for it exists.
- **Writing PNG frames to disk, then encoding:** needs far more disk and I/O than a pipe.
