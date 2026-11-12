# Plan 9 - Arabic Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close #125, #126, #11 and #13:
- the voice reads signs and symbols around numbers in the right order;
- the publish content review sees the text that is actually voiced;
- a free command flags Arabic sentences that look like dialect rather than MSA;
- a measured spike settles whether edge-tts word boundaries are good enough for Arabic captions.

**Architecture:**
- **Amounts.** A pure function in a new `text/amounts.py` takes the edge characters that `spoken._split_edges` already separates (prefix, core, suffix). It consumes a sign, a currency symbol or a percent sign next to a number core, and returns the spoken phrase in the right order. `spoken_text` calls it after the lexicon and before plain number spelling. One author token still becomes one group, so captions are unchanged.
- **Review text.** `publish/content.review_text` takes the series language and the channel lexicon. Its `[spoken]` line shows `spoken_text(...)` of the voiced text, the same function narration uses.
- **Dialect.** A new `text/dialect.py` holds:
  - a pure `check_dialect(text, model, threshold)` over a small `DialectModel` protocol;
  - a lazily imported CAMeL Tools adapter;
  - the `narratoflow check-dialect FILE` command, which prints flagged sentences and writes a JSON report beside the file.

  It flags sentences and never rejects them.
- **Spike.** A script voices fixed diacritized Arabic sentences with the real edge adapter. It maps the boundaries onto words with the production `timings_from_spans`, aligns the same audio with the production `CtcAligner`, and writes a measured report and recommendation to `docs/spikes/`.

**Tech Stack:** Python 3.12, pydantic, typer, pytest. No new dependencies: `camel-tools` 1.6.0 is already locked in the `arabic` extra, and `edge-tts` and `ctc-forced-aligner` are in the `tts-edge` and `align` extras.

**Spec:**
- `docs/PRD.md` §3, the strict non-dialect policy;
- `docs/REVIEW.md` row C9, "flag, not silently reject";
- the issue texts of #125, #126, #11 and #13;
- Plan 8 (`docs/superpowers/plans/2026-11-12-plan-8-speech-gaps.md`) for the spoken-form contract.

## Global Constraints

- `.py` and `.toml` sources are pure ASCII. Arabic literals in source use `\u` escapes. The Arabic in this plan is for reading only.
- mypy strict and Ruff clean, with no `# type: ignore` added.
- Tests never use the network and never sleep for real. A test that needs an optional extra or downloaded model data is skipped when they are absent.
- `NarratoFlowError` is caught only in `cli.main`.
- Secrets never appear in logs or error messages. Library errors are reported by exception type only, as in `align/ctc.py`.
- The spoken-form invariant from Plan 8 holds: `sum(groups) == len(text.split())`, with exactly one group per author token. Text that has no sign, currency or percent next to a number stays byte-identical, so cache keys stay stable.
- Stage files by name in commits; never `git add -A`.
- Gate: `uv lock --check && uv run ruff check --no-cache && uv run ruff format --check && uv run mypy && uv run coverage run -m pytest -o addopts="" -q && uv run coverage report`.

## Owner decisions (defaults used)

- **Signs.** `-` (U+002D) or `−` (U+2212) directly before the number core is read as "minus", or "سالب" in Arabic. `+` is read as "plus", or "زائد". A sign that is not touching the digits (`- 5`) is left alone.
- **Percent.** `%` or `٪` (U+066A) directly after the core is read as "percent", or "بالمائة". It is the same for every number.
- **Currency.** `$`, `€` and `£` count, either directly before the core (`$5`) or directly after it (`5$`), and the unit is said after the number.
  - English singular and plural: dollar/dollars, euro/euros, pound/pounds. The singular is used only for the integer 1.
  - An English amount with exactly two decimal places is read as main unit and minor unit: `$1.50` is "one dollar and fifty cents", `$0.99` is "ninety-nine cents", and `$3.00` is "three dollars". `€` uses cent/cents, and `£` uses penny/pence. Any other decimal is read as "<number> dollars" (`$1.5` is "one point five dollars").
  - Arabic uses the counted-noun rule, with forms singular, dual, plural and pausal accusative:
    - **$:** دولار، دولاران، دولارات، دولارا
    - **€:** يورو، يوروان، يوروهات، يورو
    - **£:** جنيه، جنيهان، جنيهات، جنيها

    Which form is used:
    - 1 is "<singular> واحد";
    - 2 is the dual alone;
    - an integer whose last two digits are 3 to 10 takes the plural after the number;
    - last two digits 11 to 99 take the accusative after the number;
    - anything else (0, round hundreds, decimals, and last two digits 01 or 02 above 100) takes the singular after the number.

    Arabic has no minor-unit reading.
- **Amounts never read as years.** `$1453`, `-1500` and `1999%` use the cardinal, not the English year reading.
- **One symbol per side.** A sign and a currency may both appear before the core, in either order (`-$5`, `$-5`). A token with both a currency and a percent sign, or two currencies, is left unchanged.
- **Review staleness.** A stored content review whose text now differs is stale. That happens when a part's spoken form differs from what the old `[spoken]` line showed: numbers, lexicon entries or signs. `narratoflow publish` then shows its usual stale WARN until `narratoflow review` is run again. A FAIL stays a FAIL, as today.
- **Dialect check.**
  - It is a separate free command, `check-dialect`, and `plan` does not run it.
  - A sentence is flagged when the model's top label is not `MSA` and that label's probability is at least `--threshold`, with a default of 0.5.
  - Every Arabic sentence is stored with its top label and confidence.
  - The command exits 0 even when sentences are flagged.
- **CAMeL data.** The model package is `dialectid-model26` (274 MB), installed with `camel_data -i dialectid-model26`. NarratoFlow never downloads it.

## Review Focus

1. **Plain text is untouched.** Text with no sign, currency or percent next to a number must produce exactly the Plan 8 spoken text, so no scene is re-voiced or re-billed. Test: `test_plain_text_unchanged_by_amounts`, Task 1.
2. **Hyphens that are not minus signs.**
   - A range `3-5`, a word `well-known`, a trailing `1990-` and a standalone `-` must all stay as Plan 8 voiced them.
   - `(-5)` must keep its parenthesis.

   Test: `test_hyphen_not_a_sign`, Task 1.
3. **The review shows what is voiced.** When `tts_text` is None, the `[spoken]` line must come from `source_text`, which is what narration voices. A part with no numbers and no lexicon hits must hash exactly as before Plan 9. Tests: `test_spoken_line_uses_source_when_no_tts_text` and `test_review_hash_unchanged_without_spoken_changes`, Task 2.
4. **A dialect check must not need the model for non-Arabic lines.** Lines with no Arabic letters, such as English, numbers or headings, are never sent to the model or flagged. An empty file reports 0 sentences and does not crash. Tests: `test_non_arabic_sentences_not_checked` and `test_empty_text_reports_nothing`, Task 3.
5. **The extra is missing.** Without `camel_tools`, or without its model data, `check-dialect` exits through `cli.main` with one line that names both install steps and no traceback. Test: `test_check_dialect_without_extra_names_install_steps`, Task 3.

---

### Task 1: Read signs, currency and percent around numbers (#125)

**Files:**
- Create: `src/narratoflow/text/amounts.py`
- Modify: `src/narratoflow/text/numbers.py` (add `years` keyword)
- Modify: `src/narratoflow/text/spoken.py` (call `spell_amount` in `rewrite`)
- Modify: `README.md` (number rules section: the new readings, plus the note that only scenes containing such tokens are voiced again once)
- Test: `tests/text/test_amounts.py`, `tests/text/test_spoken.py`, `tests/text/test_numbers.py`

**Interfaces:**
- Consumes: `numbers.spell_number(token, language) -> str | None`; `spoken._split_edges(token) -> (prefix, core, suffix)`.
- Produces:
  - `spell_number(token: str, language: Language, *, years: bool = True) -> str | None`. With `years=False`, an English 1100 to 1999 integer is read as a cardinal.
  - `amounts.spell_amount(prefix: str, core: str, suffix: str, language: Language) -> tuple[str, str, str] | None`. It returns `(remaining_prefix, phrase, remaining_suffix)`, or None when no sign, currency or percent touches the core, or when the core is not a number `spell_number` handles.

- [ ] **Step 1: Write the failing tests** in `tests/text/test_amounts.py`, with Arabic as `\u` escapes. Parametrize `spell_amount(prefix, core, suffix, lang)`:
  - English:
    - `("$", "5", "", EN) -> ("", "five dollars", "")`
    - `("$", "1", ",", EN) -> ("", "one dollar", ",")`
    - `("$", "1.50", "", EN) -> ("", "one dollar and fifty cents", "")`
    - `("$", "0.99", "", EN) -> ("", "ninety-nine cents", "")`
    - `("$", "3.00", "", EN) -> ("", "three dollars", "")`
    - `("$", "1.5", "", EN) -> ("", "one point five dollars", "")`
    - `("£", "2.01", "", EN) -> ("", "two pounds and one penny", "")`
    - `("€", "1453", "", EN) -> ("", "one thousand four hundred and fifty-three euros", "")`
    - `("", "50", "%.", EN) -> ("", "fifty percent", ".")`
    - `("(-", "5", ")", EN) -> ("(", "minus five", ")")`
    - `("-$", "5", "", EN)` and `("$-", "5", "", EN)` both `-> ("", "minus five dollars", "")`
    - `("+", "3", "", EN) -> ("", "plus three", "")`
    - `("", "5", "$", EN) -> ("", "five dollars", "")`
  - Arabic, by name:
    - `$1` gives "دولار واحد"
    - `$2` gives "دولاران"
    - `$5` gives "خمسة دولارات"
    - `$1005` gives "ألف وخمسة دولارات"
    - `$20` gives "عشرون دولارا"
    - `$100` gives "مائة دولار"
    - `$2.5` gives "اثنان فاصلة خمسة دولار"
    - `€5` gives "خمسة يوروهات"
    - `٥٠٪` gives "خمسون بالمائة"
    - `-٥` gives "سالب خمسة"
  - None cases:
    - `("$", "5", "%", EN)`
    - `("$€", "5", "", EN)`
    - `("", "5", "", EN)`
    - `("$", "abc", "", EN)`
    - `("- ", "5", "", EN)`. The prefix ends in a space, which `_split_edges` never produces, but it pins "touching only".

  In `tests/text/test_numbers.py`: `spell_number("1453", EN, years=False) == "one thousand four hundred and fifty-three"`, and the default call is unchanged.

  In `tests/text/test_spoken.py`:
  - `test_amount_is_one_group`: `spoken_text("Pay $5 now", EN, Lexicon.EMPTY)` gives `text == "Pay five dollars now"` and `groups == (1, 2, 1)`.
  - `test_plain_text_unchanged_by_amounts`: the Plan 8 sample texts already in the file give identical `SpokenText`, and `"a  1453\tb"` keeps its separators.
  - `test_hyphen_not_a_sign`: `3-5`, `well-known`, `1990-` and a lone `-` give the same output as `spell_number`-only Plan 8 behaviour, and `(-5)` gives `"(minus five)"`.

- [ ] **Step 2: Run** `uv run pytest tests/text -q -o addopts=""`. Expected: the new tests fail with ImportError or wrong output.

- [ ] **Step 3: Implement `spell_amount`** in `text/amounts.py`, using the owner decisions above as the tables.
  - Peel at most one sign and one currency off the end of `prefix`, in either order. Then peel one currency or one percent off the start of `suffix`.
  - Spell the core with `spell_number(core, language, years=False)`. For the English two-decimal currency reading, spell the integer part and the cents separately.
  - The phrase order is sign, number, unit (or percent).
  - Add the `years` keyword to `spell_number`.
  - In `spoken_text.rewrite`, after a lexicon miss, try `spell_amount(prefix, core, suffix, language)` before `spell_number(core, language)`. Attach its remaining prefix and suffix with `_attach`.

- [ ] **Step 4: Run the gate.** Expected: all green, coverage not lower than main's 97%.

- [ ] **Step 5: Update `README.md`** in the Pronunciation and number section. Add the sign, percent and currency readings with one English and one Arabic example each. Add one sentence saying that scenes containing such tokens are voiced, and on a paid voice billed, once more after upgrading.

- [ ] **Step 6: Commit**

```bash
git add src/narratoflow/text/amounts.py src/narratoflow/text/numbers.py src/narratoflow/text/spoken.py tests/text/test_amounts.py tests/text/test_numbers.py tests/text/test_spoken.py README.md
git commit -m "feat(text): read signs, currency and percent around numbers (#125)"
```

---

### Task 2: Publish review shows the spoken form (#126)

**Files:**
- Modify: `src/narratoflow/publish/content.py` (`review_text`, `review_sha256`, `review_prompt`, `_stale`, `content_checks`)
- Modify: `src/narratoflow/publish/build.py:258` and `src/narratoflow/publish/review.py` (pass `language=series.language, lexicon=channel_lexicon(channel)`)
- Modify: `README.md` (publish review section: the `[spoken]` line is the spoken form; the one-time stale WARN after upgrading)
- Test: `tests/publish/test_content.py`, plus the existing review and build tests that call these functions

**Interfaces:**
- Consumes: `spoken.spoken_text(text, language, lexicon) -> SpokenText`; `spoken.channel_lexicon(channel) -> Lexicon`; `Series.language`.
- Produces: `review_text`, `review_sha256`, `review_prompt` and `content_checks` each gain the keyword-only parameters `language: Language, lexicon: Lexicon`, with no defaults, so no caller can forget them.

- [ ] **Step 1: Write the failing tests** in `tests/publish/test_content.py`:
  - `test_spoken_line_shows_spelled_numbers`: a scene with `source_text="In 1453 the city fell"` and `tts_text=None` (English) gives a review text containing `"[s1 spoken] In fourteen fifty-three the city fell"`.
  - `test_spoken_line_applies_lexicon`: a lexicon `{"WHO": "the World Health Organization"}` and narration `"WHO said"` give a `[spoken]` line with the phrase.
  - `test_spoken_line_uses_source_when_no_tts_text`: with `tts_text=None` the spoken line is derived from `source_text`. With a `tts_text` that differs, it is derived from `tts_text`, so the raw `tts_text` does not appear as its own line.
  - `test_review_hash_unchanged_without_spoken_changes`: a part whose `tts_text` equals its spoken form gives a `review_sha256` equal to a literal hex digest computed on main before this change. Record that digest in the test, with a comment naming the commit it was taken from.
  - `test_number_edit_makes_review_stale`: changing `1453` to `1454` in `source_text` makes `content_checks` return the stale WARN.

- [ ] **Step 2: Run** `uv run pytest tests/publish -q -o addopts=""`. Expected: the new tests fail, with TypeError on the unknown keyword or wrong text.

- [ ] **Step 3: Implement.**
  - In `review_text`, compute `spoken = spoken_text(narration.tts_text or narration.source_text, language, lexicon).text`. Emit `[<id> spoken] <spoken>` when `spoken != narration.source_text`.
  - Thread the two keywords through `review_sha256`, `review_prompt`, `_stale` and `content_checks`, and through both callers. The callers build the lexicon once per run with `channel_lexicon(channel)`.
  - Update the module docstring line that describes the spoken line.

- [ ] **Step 4: Run the gate.** Expected: green.

- [ ] **Step 5: Update `README.md`** in the publish and review section. The `[spoken]` line is what the voice says. After upgrading, parts with numbers or pronunciation entries show the stale-review WARN once, until `narratoflow review` is run again.

- [ ] **Step 6: Commit**

```bash
git add src/narratoflow/publish/content.py src/narratoflow/publish/build.py src/narratoflow/publish/review.py tests/publish README.md
git commit -m "feat(publish): content review sees the spoken form (#126)"
```

Stage only the test files you changed under `tests/publish`, by name.

---

### Task 3: Dialect flagging with CAMeL Tools (#11)

**Files:**
- Delete: `src/narratoflow/text/language.py` (an empty stub that nothing imports)
- Create: `src/narratoflow/text/dialect.py`
- Modify: `src/narratoflow/errors.py` (add `DialectUnavailableError(NarratoFlowError)`)
- Modify: `src/narratoflow/cli.py` (add the `check-dialect` command, next to `check-input`)
- Modify: `README.md` (a "Dialect check" section: install steps, command, threshold, report file, "flags, never rejects")
- Test: `tests/text/test_dialect.py`, `tests/text/test_dialect_camel.py` (real model, skipped when absent), `tests/test_cli.py`

**Interfaces:**
- Produces:
  - `class DialectModel(Protocol): def predict(self, sentences: Sequence[str]) -> list[tuple[str, float]]`. This is one `(top_label, top_probability)` per sentence, in order.
  - `class SentenceDialect(BaseModel)`: frozen, with `line: int` (1-based line of the sentence's start), `text: str`, `label: str`, `confidence: float` (in [0, 1]) and `flagged: bool`.
  - `class DialectReport(BaseModel)`: frozen, with `threshold: float`, `model: str` and `sentences: tuple[SentenceDialect, ...]`, plus a `flagged` property that returns the flagged sentences.
  - `split_sentences(text: str) -> list[tuple[int, str]]`. It splits on newlines and after `.`, `!`, `?`, `؟` (U+061F) and `۔` (U+06D4). It strips each piece and drops empty pieces.
  - `check_dialect(text: str, model: DialectModel, *, threshold: float, model_name: str) -> DialectReport`. It normalizes with `quality.normalize_input`, sends only sentences that contain at least one Arabic letter (U+0621 to U+064A) to `model.predict` in a single call, and skips the call when there are none. It raises `ValueError` unless `0 < threshold <= 1`.
  - `load_camel_model() -> DialectModel`:
    - imports `camel_tools.dialectid` lazily and calls `DialectIdentifier.pretrained()`;
    - the adapter's `predict` calls `model.predict(list(sentences), output="label")` and maps each `DIDPred` to `(pred.top, pred.scores[pred.top])`;
    - a failure to import or load becomes `DialectUnavailableError` naming the exception type and `"install with: uv sync --extra arabic && uv run camel_data -i dialectid-model26"`;
    - a failure in `predict` becomes `ProviderError` naming the exception type only.

    The API was verified against the camel-tools 1.6.0 wheel pinned in uv.lock (sha256 `17e51cf9...`):
    - `camel_tools.dialectid.DialectIdentifier` is `DIDModel26`;
    - `DIDModel26.pretrained()` loads `did_pretrained_{major}{minor}.dill` from the catalogue path `dialectid/model26` and raises `PretrainedModelError` when it is absent;
    - `predict(sentences, output='label') -> list[DIDPred(top: str, scores: dict[str, float])]`;
    - the MSA label is `"MSA"`;
    - importing `camel_tools.data` loads the catalogue at import time.

    Repeat these notes in the module docstring, as `ctc.py` does.
  - CLI: `narratoflow check-dialect FILE [--threshold 0.5] [--report PATH]`. The report defaults to `FILE` with `.dialect.json` appended.
    - It prints one line per flagged sentence: `line N: LABEL 0.87 «excerpt»`, with the excerpt at most 40 characters.
    - It then prints a summary `K of M Arabic sentences flagged (threshold 0.5); report: PATH`.
    - It writes the report with `model_dump_json(indent=2)` in UTF-8.
    - It exits 0.

- [ ] **Step 1: Write the failing tests** in `tests/text/test_dialect.py`, using a `FakeModel` that returns scripted labels and records its calls:
  - `test_split_sentences_lines_and_marks`: `"أ ب. ج؟\n\nd"` gives `[(1, "أ ب."), (1, "ج؟"), (3, "d")]`.
  - `test_flags_non_msa_above_threshold`: the labels `("CAI", 0.8), ("MSA", 0.9), ("BEI", 0.3)` at threshold 0.5 flag only the first sentence. All three are stored with their labels and confidences.
  - `test_threshold_boundary_inclusive`: `("CAI", 0.5)` at threshold 0.5 is flagged.
  - `test_non_arabic_sentences_not_checked`: `"Hello.\n1453\nكتب"` makes the model receive exactly `["كتب"]`.
  - `test_empty_text_reports_nothing`: `""` and `"Hello"` make no model call and give `sentences == ()`.
  - `test_bad_threshold_rejected`: thresholds of 0, -1 and 1.5 raise ValueError.
  - `test_report_round_trips`: `DialectReport.model_validate_json(r.model_dump_json()) == r`.
  - `test_camel_adapter_maps_didpred`: monkeypatch a fake `camel_tools.dialectid` module into `sys.modules` whose `DialectIdentifier.pretrained()` returns an object with `predict` returning `DIDPred`-like tuples. The adapter returns `[("MSA", 0.9)]`.
  - `test_camel_import_error_is_unavailable`: when the import raises, the result is `DialectUnavailableError`, and its message contains `camel_data -i dialectid-model26` and does not contain the exception's message text.
  - `test_camel_predict_error_is_provider_error`, with the same rule about the message.

  In `tests/test_cli.py`:
  - `test_check_dialect_prints_flagged_and_writes_report`: monkeypatch `load_camel_model` to the fake, run the command through `CliRunner`, and assert exit 0, the flagged line, the summary and the report file contents.
  - `test_check_dialect_without_extra_names_install_steps`: patch the loader to raise `DialectUnavailableError` and assert exit 1, a one-line message with the install command, and no `Traceback`.

  In `tests/text/test_dialect_camel.py`, add one test that is skipped unless `importlib.util.find_spec("camel_tools")` is found and `load_camel_model()` succeeds. It checks that an MSA sentence (`"ذهب الطالب إلى المدرسة"`) is `MSA`, and that an Egyptian one (`"إزيك عامل إيه النهارده"`) is not.

- [ ] **Step 2: Run** `uv run pytest tests/text/test_dialect.py tests/test_cli.py -q -o addopts=""`. Expected: ImportError, or no such command.

- [ ] **Step 3: Implement** `text/dialect.py`, the error class and the CLI command, as the Interfaces block says. Delete `text/language.py`.

- [ ] **Step 4: Run the gate.** Expected: green, with the real-model test skipped.

- [ ] **Step 5: Update `README.md`**: add the Dialect check section, and a row in the "Where settings live" table for `--threshold` as a per-run CLI option.

- [ ] **Step 6: Commit**

```bash
git add src/narratoflow/text/dialect.py src/narratoflow/errors.py src/narratoflow/cli.py tests/text/test_dialect.py tests/text/test_dialect_camel.py tests/test_cli.py README.md
git rm src/narratoflow/text/language.py
git commit -m "feat(text): flag dialect sentences with CAMeL Tools DID (#11)"
```

---

### Task 4: Spike - Arabic word-timing accuracy (#13)

**Files:**
- Create: `scripts/spike_word_timing.py`
- Create: `docs/spikes/2026-11-12-arabic-word-timing.md` (the measured report)
- Test: `tests/align/test_spike_word_timing.py` (pure functions only; load the script the way `tests/visuals/test_fetch_icons.py` does with `_script`)

**Interfaces:**
- Consumes:
  - `tts.edge.EdgeProvider().synthesize(SpeechRequest) -> Speech`, with its word spans;
  - `audio.words.timings_from_spans(words, spans) -> list[WordTiming] | None`;
  - `align.ctc.CtcAligner().align(...)`. Read its signature in `align/ctc.py`.
- Produces, inside the script:
  - `SENTENCES: tuple[str, ...]`, with 10 to 12 fully diacritized Arabic sentences. Between them they cover:
    - the proclitics وَ, فَ, بِ, لِ, كَ and الـ;
    - attached pronouns (كِتَابُهُ);
    - the punctuation ، ؛ ؟ and «»;
    - a spelled number;
    - a sentence of more than 25 words.
  - `DriftStats` is a frozen dataclass with `words: int`, `median_start_ms: float`, `p95_start_ms: float`, `max_start_ms: int`, `median_end_ms: float` and `max_end_ms: int`.
  - `drift_stats(native: Sequence[WordTiming], reference: Sequence[WordTiming]) -> DriftStats` takes absolute differences per word. It raises ValueError when the lengths or the word texts differ, or when it is given no words.
  - `main() -> int`. For each sentence it:
    - voices the sentence with edge-tts (`ar-SA-HamedNeural`);
    - records the span count against the word count and whether `timings_from_spans` returned None (with the first differing normalized text when it did);
    - aligns the audio with CTC;
    - prints a Markdown table to stdout.

- [ ] **Step 1: Write the failing tests** in `tests/align/test_spike_word_timing.py`:
  - `test_drift_stats_values`: two 3-word timings with start offsets of 0, 40 and 100 ms give `median_start_ms == 40`, `max_start_ms == 100`, and the p95 by linear interpolation is `94.0`.
  - `test_drift_stats_rejects_mismatch`, for lengths and for texts, plus an empty input.
  - `test_sentences_are_diacritized_arabic`: every sentence contains a harakah (U+064B to U+0652), one sentence has more than 25 words, and the set contains `«`, `؟` and `،`.

- [ ] **Step 2: Run** `uv run pytest tests/align/test_spike_word_timing.py -q -o addopts=""`. Expected: FAIL, because the script is missing.

- [ ] **Step 3: Implement the script.**
  - It imports the network and model extras inside `main()` only, so the tests import it without them.
  - When an extra is missing it exits 2 and names `uv sync --extra tts-edge --extra align`.

- [ ] **Step 4: Run the gate.** Expected: green.

- [ ] **Step 5: Run the spike for real.** This sends only the fixed sample sentences to the free edge-tts service and downloads the public CTC model once.

```bash
uv sync --extra tts-edge --extra align
uv run python scripts/spike_word_timing.py > "$TMPDIR/spike-word-timing.md"
uv sync   # restore the default environment
```

  If the edge service or the model download is unreachable, write the report with "blocked: <reason>" in place of the measurements, and keep the recommendation section as "no data". Do not invent numbers.

- [ ] **Step 6: Write `docs/spikes/2026-11-12-arabic-word-timing.md`** with these sections:
  - **Setup:** versions, voice, date.
  - **Results:** the table, the totals, and native mapping failures with their cause.
  - **Mapping edge cases:** clitics, punctuation, numbers and long sentences, each with the observed behaviour.
  - **Recommendation**, decided by this rule:
    - If native mapping succeeds on every sentence, the median start drift against CTC is at most 80 ms, and the p95 is at most 200 ms, keep native edge boundaries first with CTC as the fallback. That is the current design.
    - Otherwise, recommend CTC first for Arabic, and say which follow-up issue to file. The controller files it.

  State plainly that CTC is a reference, not ground truth.

- [ ] **Step 7: Commit**

```bash
git add scripts/spike_word_timing.py tests/align/test_spike_word_timing.py docs/spikes/2026-11-12-arabic-word-timing.md
git commit -m "spike(align): measure edge-tts Arabic word timings against CTC (#13)"
```
