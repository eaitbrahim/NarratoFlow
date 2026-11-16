# CPU render benchmark: one 3-minute part (#17)

## Method

One Arabic part of 180 s: 18 scenes of 10 s, each drawing a bundled icon (the catalog
cycled in order). The timeline is written by hand (2 words per second, each scene draws
during its first 60%) and the audio is a sine FLAC, so there is no TTS and no network.
It renders at the full 1080x1920 size, 30 fps, 5400 frames. Every run passes
`force=True` and uses a fresh output directory, so a skipped part can never be timed.
Wall time is `time.perf_counter` around `build_render`: planning, frame rendering, encode
and verification. The benchmark is a script, not a CI test.

## Commands

```
NARRATOFLOW_FFMPEG=/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg uv run python scripts/bench_render.py
NARRATOFLOW_FFMPEG=/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg uv run python scripts/bench_render.py --workers 1
```

The first uses the default workers (one process per core, at most one per scene); the
second renders inline. The runs were made one after the other, never concurrently.

## NFR

A 3-minute video renders in under 600 s with the default workers.

## Results

| date | cpu / cores | ffmpeg | workers | duration s | frames | wall s | NFR (< 600 s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-11-12 | arm / 16 | 9.0.2 | default | 180 | 5400 | 84.5 | pass |
| 2026-11-12 | arm / 16 | 9.0.2 | 1 | 180 | 5400 | 181.8 | pass |
| 2026-11-15 | arm / 16 | 9.0.2 | default | 180 | 5400 | 27.3 | pass |
| 2026-11-15 | arm / 16 | 9.0.2 | 1 (timing wrapper) | 180 | 5400 | 52.4 | pass |

The 2026-11-15 default row is the unmodified `scripts/bench_render.py`. The one-worker row of
2026-11-15 comes from the timing wrapper described below (wall time minus the two probe
draws), not from the plain script.

Two default runs on 2026-11-15 agree within 0.3 s (27.3 s and 27.5 s). The 2026-11-12 runs were
about 3x slower on the same machine class, for a reason that is not known; their rows are kept
as recorded.

## Phase split (#164)

Measured on 2026-11-15 (Apple M4 Max, 16 cores, ffmpeg-full 9.0.2) for the 180 s part, 18
scenes, 5400 frames at 1080x1920. Method: the unmodified `bench_render.run` was timed by
wrapping module attributes (`build._run_jobs`, `build.encode_part`, `verify_mp4`), with one
middle scene drawn alone as a probe whose time is subtracted from the wall figure.

| phase | default workers (two runs) | 1 worker |
| --- | --- | --- |
| whole part render | 27.3 s, 27.5 s | 52.4 s |
| drawing all 18 scenes (`_run_jobs`) | 13.5 s, 14.3 s | 39.5 s |
| final pass (`encode_part`) | 9.8 s, 9.2 s | 8.9 s |
| `verify_mp4` | 3.9 s | 3.9 s |
| drawing one scene alone | 2.2 s inline, 3.6 s through the pool | 2.2 s |

A per-scene segment cache cannot skip the final pass or the verification. Its best case (one
edited scene whose slot does not move) saves about 11-12 s of a 27 s render, so it was measured
and not built (#164). Reopen if a measurement on the owner's real hardware shows the drawing
phase above about 30 s at default workers, or if parts become much longer than 180 s. The
drawing phase also does not get faster past about 4 workers (#192).
