# ADR 0006: The 16:9 export is a re-composition of the checked 9:16 video

**Status:** Accepted, 2026-11-15

## Context

D31 made the 16:9 file a copy of the 9:16 video on white bars. #163 asks for a layout composed for
the wide frame, with only the producer behind `publish/landscape.py` changing. `docs/REVIEW.md` 2.4
asks for one pipeline with two layouts.

## Decision

The stage, top bar and captions zones of the 9:16 frame are carried whole and unscaled into a
1920x1080 frame: the sketch on the left, the text on the right. Every zone keeps its 9:16 pixel
size, so no text is re-typeset and every existing libass zone test still proves the text fits.

| Zone | Source in 1080x1920 (x, y, w, h) | Destination in 1920x1080 |
| --- | --- | --- |
| Stage | (60, 390, 828, 850) | (48, 115, 828, 850) |
| Top bar | (60, 170, 960, 200) | (912, 115, 960, 200) |
| Captions | (60, 1260, 828, 180) | (978, 450, 828, 180) |

```
0        48                     876 912  978                    1806  1872   1920
+---------+-----------------------+---+---+------------------------+----+------+  0
|         +-----------------------+   +----------------------------------+     | 115
|         |   STAGE               |   | TOP BAR  960x200                 |     |
|         |   828x850             |   +----------------------------------+     | 315
|         |   sketch + hand,      |       +------------------------+           | 450
|         |   text card,          |       | CAPTIONS  828x180      |           |
|         |   verse + reference   |       +------------------------+           | 630
|         +-----------------------+                                            | 965
|  LANDSCAPE_BOTTOM (0, 972, 1920, 108): kept clear for the player controls    |
+------------------------------------------------------------------------------+ 1080
```

- An 8 px bleed (`max(1, round(8 * scale))`) travels with each zone.
- `LANDSCAPE_BOTTOM` (0, 972, 1920, 108) stays clear for the player controls, with a 48 px edge
  margin.
- The output size is 1920x1080 times the source canvas scale (320x180 at 1/6).
- One ffmpeg pass in `publish/landscape.py` (`panorama_export`): `X264_ARGS`, audio copied, and
  `verify_mp4` before the file is renamed into place. The station calls it only for a part that
  this export call's publish check has just exported.

## Verse integrity

There is no second subtitle document and no second `shown_sha256`: the verse pixels are the
certified 9:16 burn. The ink guard masks the three bled source zones on every frame (`drawbox`,
then `signalstats` YMIN) and fails when any luma below 195 (tv range; paper is 235) remains. It
also fails when it cannot run, or when its output is missing, short or cannot be parsed. On failure
it writes nothing and removes an older file. There is no fallback: a guard failure is an error.

Measured cost, 180 s benchmark part at full scale (5400 frames, Apple M4 Max, ffmpeg-full 9.0.2):
the guard took 37.8 s (7.0 ms per frame) and the whole export 55.9 s.

## Consequences

- 9:16 output, `inputs_key` and `RENDER_INPUTS_VERSION` are unchanged.
- `LANDSCAPE_WARNING` keeps its text ("16:9 is not rendered in v1") because it is part of
  `inputs_key`, and it stays true: render does not draw 16:9.
- The 16:9 file is a second-generation encode (51 dB PSNR measured on the carried zones).
- It is a derivative, listed in the station's result and not recorded in the publish report, and
  it is rebuilt on every click.
- The layout follows any future change to `STAGE`, `TOP_BAR` or `CAPTIONS`, and the geometry test
  fails if they stop fitting.
- No migration: an old 16:9 file from the previous producer is overwritten by the next Export 16:9
  and deleted by the next Export 9:16.

## Limits

- Text is not re-typeset for the wide frame; caption lines keep the 9:16 length.
- A verse scene leaves the right column almost empty.
- The guard does not see ink paler than luma 195 (`ChannelStyle.ink` defaults to `#111111`).

## Alternatives considered

- A second render with a landscape `Canvas`, its own ASS document and verse certificate. With this
  layout it would draw the same picture; it is the path to take if re-typesetting is wanted.
- Rendering both ratios in `render/build.py`: it touches the frame-exact core for no gain here.
- Keeping the previous centred-on-bars copy as a fallback. Rejected: a guard failure must be an
  error, not a quieter file.
