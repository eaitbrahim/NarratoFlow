# Arabic word-timing accuracy: edge-tts boundaries vs CTC (spike, #13)

## Setup

- Date: 2026-11-12. Script: `scripts/spike_word_timing.py`.
- Versions: edge-tts 7.3.1, ctc-forced-aligner 1.0.2, onnxruntime 1.31.1 (public CTC ONNX model, downloaded once).
- Voice: `ar-SA-HamedNeural`, default rate and pitch.
- Input: 11 fixed, fully diacritized MSA sentences (92 words). Clitics covered: وَ فَ بِ لِ كَ and الـ; attached pronouns (كِتَابُهُ, قَلَمُهَا, بَيْتُنَا); punctuation ، ؛ ؟ and «»; a spelled number (خَمْسَةَ عَشَرَ); one sentence of 26 words.
- Method: each sentence is voiced once; native timings are `timings_from_spans(words, speech.spans)`; the reference is `CtcAligner().align(...)` on the same decoded audio mapped with `timings_from_spans`, as `audio/narrate.py` does. Drift is the absolute per-word difference.

CTC is a reference, not ground truth. It has its own error (frame stride, model bias), so the numbers below measure disagreement between two estimates, not the error of either one. The drift is absolute, so it does not say which side is early or late.

## Results

Second run (script now also reports signed deltas), same setup. Native minus CTC: positive means the edge word starts later than CTC.

| # | words | spans | native mapped | ctc mapped | abs start median/p95/max ms |
|---|---|---|---|---|---|
| 1 | 5 | 5 | yes | yes | 125 / 190 / 205 |
| 2 | 8 | 8 | yes | yes | 115 / 172 / 185 |
| 3 | 7 | 7 | yes | yes | 128 / 147 / 150 |
| 4 | 6 | 6 | yes | yes | 125 / 192 / 210 |
| 5 | 6 | 6 | yes | yes | 100 / 129 / 130 |
| 6 | 6 | 6 | yes | yes | 92 / 117 / 120 |
| 7 | 6 | 6 | yes | yes | 95 / 184 / 203 |
| 8 | 7 | 7 | yes | yes | 113 / 221 / 238 |
| 9 | 6 | 6 | yes | yes | 110 / 292 / 343 |
| 10 | 26 | 26 | yes | yes | 118 / 207 / 223 |
| 11 | 9 | 9 | yes | yes | 120 / 174 / 193 |

Absolute totals over 92 words: start median 115 ms, p95 206 ms, max 343 ms; end median 61 ms, max 883 ms.

Signed start delta (native minus CTC, positive = edge later), residual after
subtracting the sentence's own median and the single pooled median.

| # | mean | median | std | negatives | own-median resid med/p95 | pooled resid med/p95 |
|---|---|---|---|---|---|---|
| 1 | -137 | -125 | 35 | 5 | 5 / 67 | 10 / 75 |
| 2 | -119 | -115 | 33 | 8 | 23 / 59 | 23 / 59 |
| 3 | -118 | -128 | 23 | 7 | 22 / 41 | 22 / 34 |
| 4 | -124 | -125 | 47 | 6 | 28 / 79 | 28 / 84 |
| 5 | -100 | -100 | 24 | 6 | 22 / 34 | 22 / 46 |
| 6 | -96 | -92 | 14 | 6 | 10 / 26 | 24 / 34 |
| 7 | -112 | -95 | 44 | 6 | 18 / 88 | 31 / 76 |
| 8 | -130 | -113 | 56 | 7 | 25 / 108 | 27 / 106 |
| 9 | -147 | -110 | 89 | 6 | 20 / 182 | 24 / 178 |
| 10 | -120 | -118 | 44 | 26 | 32 / 89 | 31 / 92 |
| 11 | -127 | -120 | 29 | 9 | 22 / 54 | 17 / 59 |

Pooled signed start: mean -121, median -115, std 45, negatives 92 of 92.
Residual after per-sentence medians: median 19, p95 88.
Residual after the pooled median: median 24, p95 91.
Sign of per-sentence median consistent: yes.

Largest signed end delta per sentence (native minus CTC):

| # | word index | last word | signed end delta ms |
|---|---|---|---|
| 1 | 4 | yes | -883 |
| 2 | 7 | yes | -870 |
| 3 | 6 | yes | -860 |
| 4 | 5 | yes | -873 |
| 5 | 5 | yes | -870 |
| 6 | 5 | yes | -858 |
| 7 | 5 | yes | -848 |
| 8 | 6 | yes | -868 |
| 9 | 5 | yes | -868 |
| 10 | 25 | yes | -863 |
| 11 | 8 | yes | -873 |

Pooled signed end median: 25 ms.

Native mapping failures: none

CTC mapping failures: none


Reading:
- Absolute drift (binding rule as written): median 115 ms (limit 80), p95 206 ms (limit 200). The rule on absolute drift fails.
- The drift is a constant offset, not disagreement: all 92 of 92 word starts are earlier in edge than in CTC, the sign is the same in every sentence (per-sentence medians -92 to -128 ms), pooled median -115 ms (mean -121, std 45).
- After subtracting the single pooled median (a fixed global offset), start drift is median 24 ms, p95 91 ms, which meets the rule (80 / 200). Per-sentence medians do slightly better (median 19, p95 88). Worst residual is sentence 9 (p95 178 ms).
- End gap: in every sentence the largest end disagreement is the last word, and it is negative: edge ends that word 848 to 883 ms before CTC does. So CTC runs long on the last word (the pooled end median over all words is only +25 ms, so this is specific to the final word). Likely cause, not verified here: CTC stretches the final word over trailing audio, while the edge boundary ends at the spoken end. Which end is right needs a listen or waveform check.
- CTC is a reference, not ground truth; this shows the two methods differ by a stable amount, not which one is closer to the audio.

## Mapping edge cases

- Clitics (وَ فَ بِ لِ كَ, الـ): edge emits one boundary per whitespace word, clitics stay attached, so counts match and mapping succeeds.
- Attached pronouns (كِتَابُهُ and similar): one boundary per word, mapping succeeds.
- Punctuation (، ؛ ؟ « » and the full stop): the normalization drops punctuation, so differences in how edge reports it never broke the mapping.
- Spelled number (خَمْسَةَ عَشَرَ): two words, two spans, succeeds. Digits were not tested in this spike.
- Long sentence (26 words): 26 spans, succeeds, start drift similar to short ones (median 120 ms, p95 207 ms).

## Recommendation

Rule as written (absolute drift): native mapping 11 of 11, median start drift 115 ms (limit 80), p95 206 ms (limit 200). The absolute rule is not met.

Because the drift is signed and constant, the decision uses the residual after the pooled offset: the sign is consistent across all sentences and the residual after the pooled signed median is median 24 ms, p95 91 ms, which is within 80 / 200.

Recommendation: keep native edge boundaries first, apply a fixed offset of +115 ms to edge word starts (shifting them later, to match CTC), and handle the last-word end (edge ends it about 860 ms before CTC; decide by ear or waveform which is right, and clamp or keep the native end accordingly). CTC stays the fallback.

Follow-up issue (controller files it):
- Title: Shift edge-tts Arabic word starts by a fixed offset and fix the last-word end
- Body: Spike #13 (docs/spikes/2026-11-12-arabic-word-timing.md) measured edge-tts ar-SA-HamedNeural word starts against CTC on 92 words: edge is earlier by a constant 115 ms (all 92 negative, same sign in all 11 sentences, std 45 ms); after subtracting it the start drift is median 24 ms, p95 91 ms. Apply a fixed +115 ms shift to native edge word starts in the Arabic path (keep native first, CTC as fallback). Also, the last word's end differs by about 860 ms (edge ends earlier than CTC) in every sentence: verify by listening or waveform which end is right and handle it. CTC is a reference, not ground truth, so confirm the offset on a few more voices and sentences, and on digits, before hard-coding it.
