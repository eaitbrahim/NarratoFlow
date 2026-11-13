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

Decision rule (binding, as written in the plan): absolute start drift must be within median 80 ms and p95 200 ms. Measured: median 115 ms, p95 206 ms. The binding rule is NOT met. By the rule, the answer is "CTC first for Arabic".

The signed data adds a caveat the rule does not capture: the drift is a constant offset of about 115 ms (all 92 word starts earlier in edge than in CTC), and after subtracting it the residual is median 24 ms, p95 91 ms. The origin of the offset is unknown. Candidates: an edge boundary offset, a CTC frame or blank-token bias, or MP3 decoder priming delay in `decode(speech.audio)`. The same open question applies to the last-word end gap (edge ends it about 860 ms before CTC). Because CTC is a reference and not ground truth, the data cannot say which side is right.

Recommendation: do not change production yet. The first step of the follow-up is to establish onset ground truth on a few words (waveform inspection or by ear) and to measure the decoder priming delay. Then choose between:
- (a) shift native edge starts by the measured offset, in the direction the ground truth shows, keeping native edge first; or
- (b) CTC first.

The same ground-truth check decides how to handle the last-word end gap.

Procedure and tools for that ground-truth check: [docs/spikes/2026-11-13-word-onset-ground-truth.md](2026-11-13-word-onset-ground-truth.md).

Follow-up issue (tracked as #133):
- Title: Arabic word timings: establish onset ground truth, then fix the ~115 ms edge/CTC offset and last-word end
- Body: Spike #13 (docs/spikes/2026-11-12-arabic-word-timing.md) measured edge-tts ar-SA-HamedNeural word starts against CTC on 92 words. The binding absolute-drift rule is not met (median 115 ms vs 80, p95 206 ms vs 200), so by the rule CTC goes first for Arabic. The signed data shows a constant offset of about 115 ms (all 92 negative, same sign in all 11 sentences, std 45 ms) with residual median 24 ms, p95 91 ms; the origin is unknown (edge boundary offset, CTC frame/blank bias, or MP3 decoder priming in `decode(speech.audio)`). The last word's end also differs by about 860 ms. Step 1: establish onset ground truth on a few words (waveform or by ear) and measure the decoder priming delay. Step 2: choose (a) shift native edge starts by the measured offset in the right direction, keeping native first, or (b) CTC first; decide the last-word end the same way. Confirm on more voices, sentences and digits before hard-coding anything.
