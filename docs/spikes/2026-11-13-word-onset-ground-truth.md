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

The plan's earlier figures were 200.02 ms and 246.06 ms; the 246.06 was a slip, since (576 + 529) / 24000 s is 46.04 ms and puts the burst at 246.04 ms (the script now computes it). The tests assert plus or minus 3 ms around these values, so the two measurements agree. ffmpeg writes no Info tag to a pipe.

What the edge MP3's header must show to settle it: if `--from` prints tag `Info` or `Xing` with a LAME or Lavc extension and delay 576, `decode()` already trims the priming and the decoder is not the source of the offset. If it prints no tag (None), `decode()` places the audio about 46 ms late, and that part of the offset is the decoder.

## Capture

One networked run. It sends only the 11 fixed sentences in `scripts/spike_word_timing.py` to the edge-tts service.

```
uv sync --extra tts-edge --extra align
uv run python scripts/spike_word_timing.py --save "$DIR"
uv sync
```

The last command restores the environment. `--save` refuses (exit 2) a `$DIR` that already holds files, so an old `NN.ctc.json` can never sit next to new audio. `$DIR` receives, per sentence `NN` (01 to 11): `NN.mp3` (raw edge bytes), `NN.edge.json` (`[[text, start_ms, end_ms], ...]`), `NN.words.json` (caption words) and `NN.ctc.json` (CTC spans, only when the align extra is installed), plus `manifest.json` (`--from` rejects a manifest whose `version` is missing or not 1). Do not commit the directory.

## Offline analysis

```
uv run python scripts/spike_word_timing.py --from "$DIR"
```

No extra is needed. CTC columns use `NN.ctc.json` when present, otherwise CTC runs live if the align extra is installed, otherwise the output says `CTC: not available`. Output, in order: the MP3 header table, a note on what the header means for `decode()`, the edge-versus-waveform table with pooled medians, and the labelled-word summary when `DIR/labels.json` exists.

Post-pause matching: each pause onset (a voiced region after a gap of at least 150 ms) is matched to one caption word, once, for every estimator together. Among words 1 and up, the word whose start has the smallest summed distance to the onset over all estimators is taken, provided each estimator has it within 300 ms; otherwise the onset is dropped for all of them. An estimator running about 100 ms early therefore cannot pull its own column onto the next word, as the earlier per-estimator nearest match did for sentence 11's edge (w2 and w6 against CTC's w1 and w5).

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

Captured 2026-11-13 02:56 UTC with edge-tts 7.3.1, voice `ar-SA-HamedNeural`, the 11 fixed sentences, CTC from the `align` extra saved as `NN.ctc.json`. Analysed offline with `--from` on ffmpeg 9.0.2. No `labels.json` yet, so the mid-sentence (labelled) number of D6 is still open. The capture directory is not committed. The `--from` output, verbatim (byte for byte, in a code block so its own `##` headings stay as printed). This is the second run: it uses the post-pause matching rule described under Offline analysis and differs from the first only in the 246.04 ms note and in sentence 11's edge post-pause cell (now w1:-120, w4:-110, w5:-108, w8:-110; before w2:+92, w4:-110, w6:+67, w8:-110).

```text
## MP3 header (decoder priming)

| # | tag | encoder | encoder_delay | encoder_padding |
|---|---|---|---|---|
| 1 | None | None | None | None |
| 2 | None | None | None | None |
| 3 | None | None | None | None |
| 4 | None | None | None | None |
| 5 | None | None | None | None |
| 6 | None | None | None | None |
| 7 | None | None | None | None |
| 8 | None | None | None | None |
| 9 | None | None | None | None |
| 10 | None | None | None | None |
| 11 | None | None | None | None |

With an Info tag declaring the delay, ffmpeg trims the priming: a burst authored at 200 ms decodes at 200.02 ms. With no tag the decoded audio is (576 + 529) / 24000 s = 46.04 ms late (burst at 246.04 ms). Measured on ffmpeg 9.0.2, libmp3lame, 24 kHz mono.

## edge vs waveform

All deltas in ms, estimate minus waveform (positive = the estimate is later).

| # | edge first - onset | ctc first - onset | edge last end - offset | ctc last end - offset | edge post-pause starts - onset | ctc post-pause starts - onset |
|---|---|---|---|---|---|---|
| 1 | -100 | +20 | +62 | +945 | - | - |
| 2 | -105 | +15 | +50 | +920 | w4:-128 | w4:+20 |
| 3 | -100 | +40 | -5 | +855 | w4:-110 | w4:+40 |
| 4 | -100 | +40 | +157 | +1030 | - | - |
| 5 | -105 | -25 | +60 | +930 | - | - |
| 6 | -100 | -20 | +37 | +895 | - | - |
| 7 | -105 | -5 | +37 | +885 | w2:-108 | w2:+95 |
| 8 | -110 | +70 | +27 | +895 | - | - |
| 9 | -100 | +40 | +7 | +875 | - | - |
| 10 | -125 | -5 | +67 | +930 | - | - |
| 11 | -100 | +20 | +22 | +895 | w1:-120, w4:-110, w5:-108, w8:-110 | w1:-25, w4:+20, w5:+85, w8:+35 |
| median | -100 | +20 | +37 | +895 | -110 | +35 |
```

Reading (sign rule: estimate minus waveform, positive = the estimate is later):

- **Decoder priming.** No edge MP3 carries a Xing/Info or VBRI tag (read by hand from the saved files, not shown in the committed output above: 24 kHz mono for all 11 by `read_mp3_info`, and about 48 kbit/s from file size over duration, e.g. sentence 1 is 21744 bytes for 3.624 s), so `decode()` cannot trim the priming. If edge's encoder primes like libmp3lame (576 samples), the decoded waveform sits about 46 ms after the edge clock (D7's no-tag case). CTC aligns `decode(audio)`, the same decoded clock the energy onsets use, so priming cancels for the CTC columns and only the edge columns move. Edge spans sit on the authored clock and the energy onsets on the decoded clock, so the measured edge delta is E - (S + 46) and edge's true error would be about 46 ms less negative than shown (first word about -54 ms, post-pause words about -64 ms), while CTC's stay as measured. The encoder delay itself cannot be read from these files.
- **First word (n = 11).** Edge starts a median 100 ms before the energy onset (range -125 to -100). CTC starts a median 20 ms after it (range -25 to +70). The ~115 ms edge-minus-CTC offset of the earlier spike is therefore mostly on the edge side: edge is early against the waveform, and even after removing up to ~46 ms of decoder priming it stays early (about -54 ms) against CTC's +20 ms. CTC's median is within about one 20 ms frame of energy (+20 ms), but single sentences are not: its range is -25 to +70 ms.
- **Post-pause words (edge n = 7, CTC n = 7).** Edge median -110 ms, CTC median +35 ms: the same picture as the first word.
- **Last word's end (n = 11).** Edge ends a median 37 ms after the energy offset. CTC ends a median 895 ms after it, because CTC stretches the last word over the trailing silence to the end of the clip (sentence 1: CTC's last span ends at 3620 ms in a 3624 ms file). The ~860 ms last-word gap of the earlier spike is a CTC artefact, not an edge one.
- **#25, first number (D6: CTC minus energy at the boundaries energy can see).** First-word start median +20 ms, post-pause starts median +35 ms, last-word end median +895 ms (a CTC artefact).
- **#25, second number (CTC minus labels for mid-sentence words).** Not yet measured: needs a hand-made `labels.json` (see Manual labelling), then a re-run of `--from`.

## Next steps

#133 steps 2 to 4, unchanged: choose between shifting native edge starts by the measured offset (keeping native first) and CTC first, decide the last-word end gap the same way, and confirm on more voices, sentences and digits before hard-coding anything. This spike does not close #133.
