# Arabic word-timing accuracy: edge-tts boundaries vs CTC (spike, #13)

## Setup

- Date: 2026-11-12. Script: `scripts/spike_word_timing.py`.
- Versions: edge-tts 7.3.1, ctc-forced-aligner 1.0.2, onnxruntime 1.31.1 (public CTC ONNX model, downloaded once).
- Voice: `ar-SA-HamedNeural`, default rate and pitch.
- Input: 11 fixed, fully diacritized MSA sentences (92 words). Clitics covered: وَ فَ بِ لِ كَ and الـ; attached pronouns (كِتَابُهُ, قَلَمُهَا, بَيْتُنَا); punctuation ، ؛ ؟ and «»; a spelled number (خَمْسَةَ عَشَرَ); one sentence of 26 words.
- Method: each sentence is voiced once; native timings are `timings_from_spans(words, speech.spans)`; the reference is `CtcAligner().align(...)` on the same decoded audio mapped with `timings_from_spans`, as `audio/narrate.py` does. Drift is the absolute per-word difference.

CTC is a reference, not ground truth. It has its own error (frame stride, model bias), so the numbers below measure disagreement between two estimates, not the error of either one. The drift is absolute, so it does not say which side is early or late.

## Results

| # | words | spans | native mapped | start median/p95/max ms | end median/max ms |
|---|---|---|---|---|---|
| 1 | 5 | 5 | yes | 125 / 190 / 205 | 62 / 883 |
| 2 | 8 | 8 | yes | 115 / 172 / 185 | 61 / 870 |
| 3 | 7 | 7 | yes | 128 / 147 / 150 | 27 / 860 |
| 4 | 6 | 6 | yes | 125 / 192 / 210 | 26 / 873 |
| 5 | 6 | 6 | yes | 100 / 129 / 130 | 62 / 870 |
| 6 | 6 | 6 | yes | 92 / 117 / 120 | 86 / 858 |
| 7 | 6 | 6 | yes | 95 / 184 / 203 | 71 / 848 |
| 8 | 7 | 7 | yes | 113 / 221 / 238 | 95 / 868 |
| 9 | 6 | 6 | yes | 110 / 292 / 343 | 138 / 868 |
| 10 | 26 | 26 | yes | 120 / 207 / 223 | 25 / 863 |
| 11 | 9 | 9 | yes | 120 / 174 / 193 | 152 / 873 |

Totals over 92 words: start median 116 ms, p95 206 ms, max 343 ms; end median 61 ms, max 883 ms.

Native mapping failures: none. All 11 sentences produced exactly one span per word and `timings_from_spans` returned timings for every one.

Observations:
- Start drift is steady at roughly 90 to 130 ms in every sentence, which looks like a systematic offset between the two methods rather than random noise. Its sign was not recorded.
- The end drift has a median of 61 ms but a maximum of 848 to 883 ms in every sentence. That is one word per sentence, most likely the last word, where the two methods disagree about where trailing audio ends. This was not isolated per word in this run.

## Mapping edge cases

- Clitics (وَ فَ بِ لِ كَ, الـ): edge emits one boundary per whitespace word, clitics stay attached, so counts match and mapping succeeds.
- Attached pronouns (كِتَابُهُ and similar): one boundary per word, mapping succeeds.
- Punctuation (، ؛ ؟ « » and the full stop): the normalization drops punctuation, so differences in how edge reports it never broke the mapping.
- Spelled number (خَمْسَةَ عَشَرَ): two words, two spans, succeeds. Digits were not tested in this spike.
- Long sentence (26 words): 26 spans, succeeds, start drift similar to short ones (median 120 ms, p95 207 ms).

## Recommendation

Rule: keep native edge boundaries first with CTC as fallback only if native mapping succeeds on every sentence, the median start drift against CTC is at most 80 ms, and the p95 is at most 200 ms.

Measured: mapping succeeded on 11 of 11, but median start drift is 116 ms (limit 80) and p95 is 206 ms (limit 200). The rule is not met.

Recommendation: use CTC first for Arabic, with native edge boundaries as the fallback. File a follow-up issue for that change (controller files it).

Caveat: CTC is a reference, not ground truth. The rule is binding as written, so the recommendation follows it, but the near-constant offset suggests the follow-up should first check by ear or waveform which method is closer to the audio before the order is switched.
