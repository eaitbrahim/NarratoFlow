# ADR 0005: Diacritizer benchmark

**Status:** Accepted (provisional), 2026-11-13; llm and catt rows added 2026-11-14

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

Rows are copied from `docs/benchmarks/2026-11-14-tashkeel-results.json` (numbers only; it replaces
the not-run rows of `2026-11-13-tashkeel-results.json`, which stays as the first run). Hosts, all
2026-11-14, on one Apple M4 Max with 64 GB:

- identity, mishkal, llm: macOS arm64, CPU for the first two; llm on local Ollama (Metal GPU).
- catt: a linux/arm64 Docker container on the same Mac, CPU only (onnxruntime-gpu falls back to CPU).

| engine | lines | rejected | DER | DER (no case ending) | WER | WER (no case ending) | runtime |
| --- | --- | --- | --- | --- | --- | --- | --- |
| identity | 300 | 0 | 82.77 | 83.60 | 99.49 | 98.75 | 0.0 s |
| mishkal | 300 | 1 | 25.45 | 18.83 | 65.03 | 38.79 | 6.5 s |
| llm (ollama GLM-4.7-Flash Q4_K_M) | 300 | 197 | 74.76 | 74.49 | 78.73 | 76.66 | 8475.5 s |
| catt | 300 | 98 | 35.45 | 35.28 | 36.49 | 35.73 | 23.6 s |

- **identity** returns its input: the floor, since most gold letters carry a mark.
- **mishkal** is `MishkalDiacritizer` (mishkal 0.4.1) with the skeleton guard.
- **llm** is `LlmDiacritizer` on `OllamaClient`, provider ollama, model
  `hf.co/unsloth/GLM-4.7-Flash-GGUF:Q4_K_M`, temperature 0, thinking off, the production
  `TASHKEEL_SYSTEM` prompt and a JSON-schema `format`. Cost: $0, no paid API (ruling R3, R4).
  197 of 300 lines are rejected: 31 hit the 120 s client timeout (the server log shows HTTP 500
  after 2 min) and the rest failed the skeleton check or returned invalid output; each counts as all wrong. The 103 accepted
  lines were scored, but with so many rejections the row mostly says this local quantized model
  cannot return the skeleton intact on long lines. It is not evidence about hosted models.
- **catt** is `catt-tashkeel` 1.0.2 (Apache-2.0), `CATTEncoderOnly`, batch size 1, ONNX weights
  (72.1 MB download; the onnxruntime-gpu wheel is about 300 MB). 98 lines are rejected, the same
  count as the gold lines that contain digits or Latin characters: CATT drops characters outside
  the Arabic alphabet from its output (a probe of one sentence with digits and Latin text came
  back with both gone), so the skeleton guard rejects them. Its DER and WER are therefore a
  production-like floor, not its quality on pure Arabic text.
- **Possible contamination:** CATT's training data may overlap Tashkeela, so its score on this
  gold set may be inflated.

Commands used:

```bash
uv run python scripts/fetch_gold_tashkeel.py
uv run python scripts/fetch_gold_tashkeel.py --make-manifest docs/benchmarks/tashkeel-gold.toml
# identity, mishkal, llm (Mac; Ollama serves the model at http://localhost:11434):
uv run --with mishkal==0.4.1 python scripts/bench_tashkeel.py --engines identity,mishkal,llm \
  --llm-url http://localhost:11434 \
  --llm-model hf.co/unsloth/GLM-4.7-Flash-GGUF:Q4_K_M --out <rows-mac.json>
# catt (linux/arm64 container on the same Mac; UV_PROJECT_ENVIRONMENT keeps the Mac .venv untouched):
docker run --rm --platform linux/arm64 -v "$PWD":/w -w /w -e UV_PROJECT_ENVIRONMENT=/tmp/venv \
  ghcr.io/astral-sh/uv:python3.12-bookworm-slim \
  uv run --with catt-tashkeel==1.0.2 python scripts/bench_tashkeel.py --engines catt \
  --out /w/.cache/rows-catt.json
```

The two outputs were merged into `docs/benchmarks/2026-11-14-tashkeel-results.json`.
catt is not a project dependency: the universal `uv.lock` cannot resolve onnxruntime-gpu on macOS
arm64, so an extra would break `uv lock`. Reproducing the catt row needs Docker or Linux.

## Decision

Revisited 2026-11-14 (#155, #156). The code default `ChannelConfig.diacritizer = "llm"` is **not
changed here**. Recommendation, on this evidence: stop defaulting to `llm` and default to
`mishkal`, with `llm` as an opt-in. On this gold set Mishkal has the lowest DER (25.45), is free,
offline, and needs 6.5 s per 300 lines, and the local LLM was 197 of 300 rejected at 8,475 s.
CATT's WER (36.49) is lower than Mishkal's (65.03) and its DER (35.45) is higher, but 98 of its 300
lines are rejected only because it drops digits and Latin text, so it needs an adapter that keeps
non-Arabic characters before it could be compared fairly or shipped, and it needs onnxruntime.

Caveat: the llm row used a local 4-bit model. A hosted model may score very differently, so the
claim "Mishkal beats the LLM" holds for this local model only. If the default decision depends on
the hosted `llm` row (it does, since `llm` is the current default and its production provider is
hosted), a paid-provider row is needed. It is human-owned (no paid runs here), and is not filed by
this PR.

## Consequences

- The skeleton guard stays mandatory for every engine: even Mishkal had one rejected line.
- Anyone reproducing the numbers needs the network once (about 1.7 MB) to fetch the gold file.
- The numbers are not directly comparable with the paper's: the rejected-line rule and the
  300-line subset differ.
