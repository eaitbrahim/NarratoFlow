# Word onset ground truth for edge-tts and CTC timings

## Purpose

Step 1 of #133: establish what the waveform says about word onsets, so the ~115 ms edge-versus-CTC offset from `docs/spikes/2026-11-12-arabic-word-timing.md` can be attributed to one side. It also defines the measurement behind #25's "median word boundary error" as two numbers:

- CTC minus energy at the boundaries energy can see: the first word's start, post-pause word starts and the last word's end;
- CTC minus hand labels for mid-sentence words, when `labels.json` exists.

Sign rule everywhere: estimate minus waveform (or minus label). Positive means the estimate is later.

The tools live in `src/narratoflow/audio/onsets.py` (energy regions) and `src/narratoflow/audio/mp3info.py` (MP3 header reader). Nothing in production imports them yet.

## Decoder priming

Measured on ffmpeg 9.0.2 with libmp3lame at 24 kHz mono, 48 kbit/s, written to a seekable file (2026-11-12, the characterization test in `tests/audio/test_mp3info.py`):

| Encoder setting | What the file declares | Burst authored at 200 ms decodes at |
|---|---|---|
| `-write_xing 1` (Info tag) | encoder delay 576, padding 960, encoder `Lavc63.1.` | 200.0 ms (ffmpeg trims the priming) |
| `-write_xing 0` (no tag) | nothing | 245.0 ms, i.e. +45.0 ms |

The plan's earlier figures were 200.02 ms and 246.06 ms, which is (576 + 529) / 24000 s, about 46.04 ms late. The tests assert plus or minus 3 ms around these values, so the two measurements agree. ffmpeg writes no Info tag to a pipe.

What the edge MP3's header must show to settle it: if `--from` prints tag `Info` or `Xing` with a LAME or Lavc extension and delay 576, `decode()` already trims the priming and the decoder is not the source of the offset. If it prints no tag (None), `decode()` places the audio about 46 ms late, and that part of the offset is the decoder.

## Capture

One networked run. It sends only the 11 fixed sentences in `scripts/spike_word_timing.py` to the edge-tts service.

```
uv sync --extra tts-edge --extra align
uv run python scripts/spike_word_timing.py --save "$DIR"
uv sync
```

The last command restores the environment. `$DIR` receives, per sentence `NN` (01 to 11): `NN.mp3` (raw edge bytes), `NN.edge.json` (`[[text, start_ms, end_ms], ...]`), `NN.words.json` (caption words) and `NN.ctc.json` (CTC spans, only when the align extra is installed), plus `manifest.json`. Do not commit the directory.

## Offline analysis

```
uv run python scripts/spike_word_timing.py --from "$DIR"
```

No extra is needed. CTC columns use `NN.ctc.json` when present, otherwise CTC runs live if the align extra is installed, otherwise the output says `CTC: not available`. Output, in order: the MP3 header table, a note on what the header means for `decode()`, the edge-versus-waveform table with pooled medians, and the labelled-word summary when `DIR/labels.json` exists.

## Manual labelling

1. Open each `NN.mp3` in Audacity, after Tracks > Resample to 48 kHz, so the clock matches `decode()`.
2. Add a label track.
3. Label 2 or 3 mid-sentence word onsets per sentence at the first visible energy rise of the word's first consonant, checked by ear with the selection-play.
4. Export the labels, then convert them by hand into `DIR/labels.json`, with word indices from `NN.words.json`:

```json
{"version": 1, "labels": [{"sentence": 1, "word": 3, "onset_ms": 812}]}
```

`word` is a 0-based index into `NN.words.json`; `onset_ms` is on the decoded-audio clock.

## What energy can and cannot decide

Energy onsets are reliable at the sentence start, after pauses and at the final word's end. They are not reliable inside connected speech, where words run together without a dip in energy, so mid-sentence words need the labels. Resolution is one 5 ms window. The rule: RMS over 5 ms windows, voiced above -45 dBFS (the level production already uses for its first onset), voiced runs under 20 ms dropped, gaps under 150 ms merged here so only real pauses count. Quiet fricative onsets are detected late, which biases results toward "edge earlier"; the labels cross-check this.

## Results

not yet run: needs one networked `--save`

## Next steps

#133 steps 2 to 4, unchanged: choose between shifting native edge starts by the measured offset (keeping native first) and CTC first, decide the last-word end gap the same way, and confirm on more voices, sentences and digits before hard-coding anything. This spike does not close #133.
