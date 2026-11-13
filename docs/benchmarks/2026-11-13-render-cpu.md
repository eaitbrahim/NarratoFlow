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

Single runs on one developer machine, ffmpeg-full 9.0.2; no repeat, so no variance is known.
