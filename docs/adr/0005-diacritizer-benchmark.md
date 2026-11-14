# ADR 0005: Diacritizer benchmark

**Status:** Accepted (provisional), 2026-11-13

## Context

`docs/REVIEW.md` C9 and section 2.1 ask for the default diacritizer to be chosen on evidence, not
on habit. `ChannelConfig.diacritizer` defaults to `"llm"`; Mishkal is the free, offline option.
Issue #8 asks for a gold set of at least 200 vocalized sentences, DER/WER per engine, and the
results in an ADR.

## Gold set

- **Source:** `dataset/test.txt` of github.com/AliOsm/arabic-text-diacritization (Fadel et al. 2019),
  commit `e083579c03e489936200b69d792fd426bb576743`, sha256
  `4e851ff836f0a178abb15d9f4a8bcf92748b77cc0d67030d9fae2fbe698baa12` (2500 lines).
- **Licence:** the repository is MIT. The text derives from the Tashkeela corpus (Zerrouki and
  Balla 2017), listed as GPLv2 on SourceForge and in the Hugging Face loader (the TFDS catalog says
  "no known license"); its texts come from the Shamela library. This project is Apache-2.0 and that
  layering is not clearly redistributable, so **the repo commits no Arabic text**. It commits
  `scripts/fetch_gold_tashkeel.py` (pinned URL and sha256, writes to the git-ignored
  `.cache/bench/tashkeel/`) and `docs/benchmarks/tashkeel-gold.toml` (source, commit, file sha256,
  licence notes, selection rule, and each chosen line's number and sha256).
- **Subset:** the first 300 lines of the file, in order, with 5 to 60 whitespace-separated tokens
  (1-based line numbers; the 300th is line 427). 8,121 words; 98 lines contain digits or Latin
  characters. A "sentence" here is one line, Fadel's distributed segment, which may hold more than
  one sentence. 300 is at least the 200 required.

## Metrics

Definitions follow Fadel et al.'s `diacritization_stat.py` (`src/narratoflow/text/tashkeel/bench.py`):

- Only Arabic letters (U+0621 to U+063A, U+0641 to U+064A) are scored. Digits, Latin, punctuation,
  whitespace and tatweel are in neither numerator nor denominator.
- A letter's class is the sorted set of harakat (U+064B to U+0652) that follow it, so shadda plus a
  vowel in either order is one class. Superscript alef, tatweel, the Qur'anic marks and the other
  marks in U+0653 to U+065F are stripped from both sides, as `strip_diacritics` does.
- A gold letter with no harakat has class "" and counts, as in Fadel's main setting.
- **DER** is wrong letters over scored letters, times 100. **WER**: a word is a whitespace token
  with at least one Arabic letter and is wrong when any of its scored letters is wrong.
- Each metric has two variants: with every letter, and **without case ending**, which drops each
  word's last letter (a one-letter word then scores nothing and is correct).
- The engine's input is `strip_diacritics(gold_line)`.
- An output whose base letters differ from the gold's, or that raises `SkeletonMismatchError` or
  `ProviderError`, is **rejected**: all its scored letters and words count as wrong. Production
  would reject it too (`vocalize_scene` falls back to the undiacritized source), so it scores as
  undiacritized or worse. This makes DER stricter than the paper's.

## Results

Run on a macOS arm64 laptop, CPU only, 2026-11-13. Rows are copied from
`docs/benchmarks/2026-11-13-tashkeel-results.json` (numbers only); the runtime cell of the two not-run rows also names the follow-up issue.

| engine | lines | rejected | DER | DER (no case ending) | WER | WER (no case ending) | runtime |
| --- | --- | --- | --- | --- | --- | --- | --- |
| identity | 300 | 0 | 82.77 | 83.60 | 99.49 | 98.75 | 0.0 s |
| mishkal | 300 | 1 | 25.45 | 18.83 | 65.03 | 38.79 | 6.2 s |
| llm | not run | not run | not run | not run | not run | not run | - (#155) |
| catt | not run | not run | not run | not run | not run | not run | - (#156) |

- **identity** returns its input: the floor, since most gold letters carry a mark.
- **mishkal** is `MishkalDiacritizer` (mishkal 0.4.1) with the skeleton guard.
- **llm**: not run (paid). No paid model call is made in this work (ruling R3). Follow-up: #155.
- **catt**: not run. `catt-tashkeel` 1.0.2 (Apache-2.0) requires `onnxruntime-gpu>=1.22.0`, which has
  no macOS arm64 wheel, so it cannot be installed on the benchmark machine. Its ONNX weights are a
  72 to 86 MB download, and its training data may overlap Tashkeela, so its score on this gold set
  may be contaminated. Follow-up: #156.

Commands used:

```bash
uv run python scripts/fetch_gold_tashkeel.py
uv run python scripts/fetch_gold_tashkeel.py --make-manifest docs/benchmarks/tashkeel-gold.toml
uv run --with mishkal==0.4.1 python scripts/bench_tashkeel.py --gold .cache/bench/tashkeel \
  --manifest docs/benchmarks/tashkeel-gold.toml --engines identity,mishkal,llm,catt \
  --out docs/benchmarks/2026-11-13-tashkeel-results.json
```

## Decision

The default stays `llm`. No row can show it worse, because it was not run. Mishkal is the measured
free and offline option, at the DER and WER above. Revisit when #155 (the LLM row) lands, and when
#156 (CATT) does.

## Consequences

- The skeleton guard stays mandatory for every engine: even Mishkal had one rejected line.
- Anyone reproducing the numbers needs the network once (about 1.7 MB) to fetch the gold file.
- The numbers are not directly comparable with the paper's: the rejected-line rule and the
  300-line subset differ.
