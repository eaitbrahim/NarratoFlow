# Plan 8 - Speech Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the speech gaps left in backlog issues #10, #12 and #23. Numbers are spoken as words, each channel has a pronunciation table, a voice request that fails for a temporary reason is retried, and the marker sound is lowered while the narrator speaks.

**Architecture:**
- **Spoken form.** A pure function turns a scene's voiced text (`tts_text or source_text`) into its *spoken form*. It works one caption token at a time: a lexicon hit is replaced by the author's phrase, a number is spelled out, and anything else is kept.
  - It also records how many spoken words each caption token became (`groups`).
  - Narration voices, caches and prices the spoken text. Word timings come from the spoken words and are merged back onto the caption tokens through `groups`, so captions still show `1453` while the voice says the words.
- **Retries.** A wrapper around each network voice retries only errors that the adapter marks as transient.
- **Ducking.** A pure function lowers the SFX bus under the voice bus before the two are summed.

**Tech Stack:** Python 3.12, numpy, pydantic, pytest. No new dependencies. Both number spellers are written in-repo, which avoids num2words (LGPL) and keeps the output exact.

**Spec:** `docs/superpowers/specs/2026-11-10-sketch-library-pipeline-design.md`, plus the issue texts of #10, #12 and #23. The lexicon and numeral requirement comes from `docs/REVIEW.md` line 34.

## Global Constraints

- `.py` and `.toml` sources are pure ASCII. Arabic literals in source use `\u` escapes.
- mypy strict and Ruff clean, with no `# type: ignore` added.
- Tests never use the network and never sleep for real. Inject the sleep function.
- `NarratoFlowError` is caught only in `cli.main`.
- Secrets never appear in logs or error messages.
- Channel models keep `ConfigDict(extra="forbid", hide_input_in_errors=True)`.
- Stage files by name in commits; never `git add -A`.
- Gate: `uv lock --check && uv run ruff check --no-cache && uv run ruff format --check && uv run mypy && uv run coverage run -m pytest -o addopts="" -q && uv run coverage report`.

## Owner decisions (defaults used)

- **Arabic number style.** Arabic numbers are spelled in the plain pausal form, with no case endings and no tashkeel, in masculine form, joined with "و" ("and"). For example, 1453 becomes "ألف وأربعمائة وثلاثة وخمسون". The voices vocalize them themselves.
- **English years.** A bare 4-digit English integer from 1100 to 1999 is read as a year: 1453 becomes "fourteen fifty-three", and 1900 becomes "nineteen hundred". Any other integer is read as a cardinal: 2005 becomes "two thousand and five", with a British-style "and".
- **What counts as a number.** Integers from 0 to 999,999,999,999 count, with an optional thousands separator ("," or U+066C) and an optional decimal part ("." or U+066B), written in any Unicode decimal digits. English ordinals with a suffix also count (`1st`, `22nd`, `3rd`, `19th`). Anything longer or of another shape stays as digits, for the voice to read.
- **Lexicon order.** The lexicon wins over number spelling, so an author can force a reading of `1453` or `COVID-19`.

## Review Focus

1. **Token-count invariant.** A scene whose caption has N tokens must still produce exactly N caption timings when one token becomes several spoken words. Test: `test_numeral_scene_keeps_caption_digits_with_merged_timing`, Task 2.
2. **Cache and price honesty.** Changing a lexicon entry must re-voice only the affected scenes, and the estimate must count the spoken characters, not the digits. Tests: `test_lexicon_change_revoices_only_that_scene` and `test_voice_chars_counts_spoken_form`, Task 2.
3. **Retries never double-bill.** A `BilledProviderError`, or an ElevenLabs read timeout that may have been billed, is never retried. Tests: `test_billed_error_not_retried` (Task 3) and `test_read_timeout_not_transient` (Task 3).
4. **Retries end.** When every attempt fails transiently, the last error is raised after exactly `RETRY_ATTEMPTS` calls, and the voice chain then falls back as before. Test: `test_gives_up_after_attempts`, Task 3.
5. **Ducking never boosts or clips.** The SFX gain stays in [-DUCK_DB, 0] dB, and SFX in silence is unchanged. Tests: `test_gain_bounded`, `test_sfx_in_silence_unchanged`, Task 4.

---

### Task 1: Spoken form - numbers and lexicon

**Files:**
- Create: `src/narratoflow/text/numbers.py`
- Create: `src/narratoflow/text/spoken.py`
- Modify: `src/narratoflow/channels/config.py` (add the `pronunciation` field to `ChannelConfig`)
- Test: `tests/text/test_numbers.py`, `tests/text/test_spoken.py`, `tests/test_channels.py`

**Interfaces:**
- Produces:
  - `numbers.spell_number(token: str, language: Language) -> str | None`: the number core of a token, spelled out, or None when the token is not a number this plan handles. The input has no surrounding punctuation.
  - `spoken.Lexicon`: a frozen mapping from `norm_for_timing(key)` to its replacement phrase. It has `Lexicon.from_entries(entries: Mapping[str, str]) -> Lexicon`, and `Lexicon.lookup(token: str) -> str | None`, which compares the token's `norm_for_timing` with the key. `Lexicon.EMPTY` is the empty lexicon.
  - `spoken.SpokenText`: a frozen dataclass with `text: str` and `groups: tuple[int, ...]`. `__post_init__` raises `ValueError` unless every group is at least 1 and `sum(groups) == len(text.split())`.
  - `spoken.spoken_text(text: str, language: Language, lexicon: Lexicon) -> SpokenText`.
  - `ChannelConfig.pronunciation: dict[str, str]`, with default `{}`.

- [ ] **Step 1: Write the failing speller tests** in `tests/text/test_numbers.py`. Write Arabic strings as `\u` escapes. Parametrize:
  - English cardinals:
    - `("0", "zero")`
    - `("7", "seven")`
    - `("13", "thirteen")`
    - `("21", "twenty-one")`
    - `("100", "one hundred")`
    - `("101", "one hundred and one")`
    - `("2005", "two thousand and five")`
    - `("1,000,000", "one million")`
    - `("999999999999", "nine hundred and ninety-nine billion nine hundred and ninety-nine million nine hundred and ninety-nine thousand nine hundred and ninety-nine")`
  - English years: `("1453", "fourteen fifty-three")`, `("1900", "nineteen hundred")`, `("1905", "nineteen oh five")`.
  - English decimals: `("3.14", "three point one four")`.
  - English ordinals: `("1st", "first")`, `("22nd", "twenty-second")`, `("19th", "nineteenth")`, `("100th", "one hundredth")`.
  - Arabic, with the expected value written as a `\u` escape in the test:
    - `"0"` -> "صفر"
    - `"1"` -> "واحد"
    - `"2"` -> "اثنان"
    - `"11"` -> "أحد عشر"
    - `"12"` -> "اثنا عشر"
    - `"21"` -> "واحد وعشرون"
    - `"100"` -> "مائة"
    - `"200"` -> "مائتان"
    - `"300"` -> "ثلاثمائة"
    - `"1000"` -> "ألف"
    - `"2000"` -> "ألفان"
    - `"3000"` -> "ثلاثة آلاف"
    - `"11000"` -> "أحد عشر ألفا"
    - `"1453"` -> the owner-decision string above
    - `"1000000"` -> "مليون"
  - Arabic-Indic and Persian digits: `"١٤٥٣"` and `"۱۴۵۳"` must equal `spell_number("1453", ARABIC)`.
  - Arabic decimal: `"٣٫٥"` -> "ثلاثة فاصلة خمسة".
  - Rejections, each returning None:
    - `"1234567890123"` (out of range)
    - `"12a"`
    - `"1,23"` (a bad grouping)
    - `"abc"`
    - `""`
    - `"1.2.3"`

  Arabic plural rules to pin with tests:
  - 3 to 10 thousand use "آلاف", and 11 or more use the singular "ألفا".
  - Millions follow the same pattern: "مليون", "مليونان", "ملايين", "مليونا".
  - Billions: "مليار", "ملياران", "مليارات" (3 to 10), "مليارا" (11 or more).

- [ ] **Step 2: Run** `uv run pytest tests/text/test_numbers.py -q -o addopts=""`. Expected: it fails with an ImportError.

- [ ] **Step 3: Implement `spell_number`** in `src/narratoflow/text/numbers.py`:
  - Parse with a regex over `\d` (Unicode). Turn digits into ints with `unicodedata.decimal`.
  - When a token has thousands separators, every group after the first must have exactly 3 digits.
  - English: the hundreds part uses "and" before tens and units, and a hyphen joins tens and units. Build an ordinal from the cardinal's last word (one->first, two->second, three->third, five->fifth, eight->eighth, nine->ninth, twelve->twelfth, -y->-ieth, otherwise +th).
  - Arabic: build the scale groups from the highest down, and join non-empty groups with " و". Inside a group under 100, the units come before the tens, also joined with " و". A scale word whose count is 1 or 2 stands alone (its dual form).

- [ ] **Step 4: Run the speller tests.** Expected: PASS.

- [ ] **Step 5: Write the failing spoken-form tests** in `tests/text/test_spoken.py`:
  - `test_plain_text_unchanged`: `spoken_text("hello world.", ENGLISH, Lexicon.EMPTY)` gives `SpokenText("hello world.", (1, 1))`.
  - `test_number_token_expands_with_punctuation_kept`: `"In 1453, Constantinople fell."` gives text `"In fourteen fifty-three, Constantinople fell."` and groups `(1, 2, 1, 1)`. Leading and trailing punctuation (Unicode categories P* and S*) stays attached to the first and last spoken word.
  - `test_lexicon_beats_number`: with `Lexicon.from_entries({"1453": "the year of the conquest"})`, the token `1453,` becomes `"the year of the conquest,"`, with group 5.
  - `test_lexicon_matches_ignoring_diacritics`: an Arabic key without tashkeel matches a vocalized token, and the author's replacement is used verbatim.
  - `test_mixed_token_left_alone`: `"COVID-19"` stays unchanged, with group 1, when it has no lexicon entry.
  - `test_groups_invariant`: `SpokenText("a b", (1,))` raises `ValueError`, and so does `SpokenText("a", (0, 1))`.
  - `test_lexicon_rejects_bad_entries`: `from_entries` raises `ValueError` in three cases: a key containing whitespace, an empty key or replacement after stripping, and two keys with the same `norm_for_timing`. The message names the key but not the replacement.

- [ ] **Step 6: Implement `src/narratoflow/text/spoken.py`.** Use `tokens` and `norm_for_timing` from `narratoflow.audio.words`. If that import creates a cycle, move both functions into `narratoflow.text.spoken` and re-export them from `audio.words`; the existing tests must stay green.

- [ ] **Step 7: Add the channel field.** Write the failing test `test_pronunciation_table_loads_and_validates` in `tests/test_channels.py`:
  - a channel TOML with `[pronunciation]` containing `"NASA" = "nasa"` loads;
  - a key containing a space fails `load_channel` with `ChannelConfigError`;
  - the error shows neither the replacement text nor any other value.

  Then add the field `pronunciation: dict[str, str] = Field(default_factory=dict)`, with a field validator that calls `Lexicon.from_entries` so that bad tables fail at load. Also add a property-free helper `channel_lexicon(channel: ChannelConfig) -> Lexicon` in `spoken.py`.

- [ ] **Step 8: Run the gate, then commit:** `feat(text): spoken form with number spelling and pronunciation lexicon`.

---

### Task 2: Narration speaks the spoken form

**Files:**
- Modify: `src/narratoflow/audio/narrate.py` (`_spoken`, `_check_tokens`, `_key` callers, `uncached_chars`, `_timings`, `_caption_words`, `narrate_part`)
- Modify: `src/narratoflow/audio/build.py:321-333` (pass the lexicon)
- Modify: `src/narratoflow/costs/tracker.py:201` (`voice_chars`)
- Modify: `src/narratoflow/planning/pipeline.py:188` (the `voice_chars` call)
- Modify: `README.md` (a "Pronunciation" subsection next to the channel settings)
- Test: `tests/audio/test_narrate.py`, `tests/audio/test_build.py`, `tests/costs/`

**Interfaces:**
- Consumes: `spoken_text`, `SpokenText`, `Lexicon` and `channel_lexicon` from Task 1.
- Produces:
  - `narrate_part(..., lexicon: Lexicon = Lexicon.EMPTY)`
  - `uncached_chars(part, choice, cache, *, language: Language, lexicon: Lexicon) -> int`
  - `voice_chars(part: Part, language: Language, lexicon: Lexicon) -> int`

- [ ] **Step 1: Write the failing tests** in `tests/audio/test_narrate.py`, using the existing fake provider and cache fixtures:
  - `test_numeral_scene_keeps_caption_digits_with_merged_timing`:
    - The scene's source text is `"In 1453 it fell"`.
    - The fake provider records the request text and returns one span per spoken word.
    - Assert that the request text is `"In fourteen fifty-three it fell"`.
    - Assert that the caption words are `["In", "1453", "it", "fell"]`.
    - Assert that `1453` starts at the "fourteen" span's start and ends at the "fifty-three" span's end.
  - `test_lexicon_change_revoices_only_that_scene`: voice two scenes, then change one lexicon entry that affects only scene 2. The provider is called once more, only for scene 2.
  - `test_estimated_timings_use_spoken_words`: the provider returns no spans and there is no aligner. The `1453` caption token covers the duration of two spoken words. Assert that its duration is greater than "In"'s.
  - In `tests/costs`, `test_voice_chars_counts_spoken_form`: `voice_chars` of a part with `"1453"` equals `len("fourteen fifty-three")`.

- [ ] **Step 2: Run them.** Expected: they fail, either on the request text or with a TypeError for the new keyword.

- [ ] **Step 3: Implement.**
  - `_spoken(scene, language, lexicon) -> SpokenText` applies `spoken_text` to `tts_text or source_text`.
  - `_check_tokens` compares `len(spoken.groups)` with the number of caption tokens. Keep the same error message.
  - The cache key, `uncached_chars`, the request text and the aligner text all use `spoken.text`.
  - `_timings` times the spoken words.
  - `_caption_words` walks `groups`: caption token i takes the start of its group's first word and the end of its group's last word.
  - Bump `ADAPTER_VERSION` to `"2"` only if the stored meta format changes. It should not, because keys change naturally when the text changes. Say which you chose in the report.
  - `build.py` passes `channel_lexicon(channel)` and `self.series.language` to both `uncached_chars` and `narrate_part`.
  - `planning/pipeline.py` passes `channel.language` and `channel_lexicon(channel)` to `voice_chars`.

- [ ] **Step 4: Update the README.** Add a "Pronunciation" subsection with:
  - a `[pronunciation]` TOML example (`"NASA" = "nasa"`, plus an Arabic name);
  - a note that keys match ignoring tashkeel, case and punctuation;
  - a note that numbers are spoken as words while captions keep the digits;
  - the English year rule;
  - a note that changing an entry re-voices only the scenes that use it.

- [ ] **Step 5: Run the gate, then commit:** `feat(audio): voice the spoken form and merge its timings onto caption words`.

---

### Task 3: Retry transient voice failures

**Files:**
- Modify: `src/narratoflow/errors.py` (add `TransientProviderError(ProviderError)`)
- Modify: `src/narratoflow/tts/edge.py`, `src/narratoflow/tts/elevenlabs.py` (raise the transient error where it applies)
- Create: `src/narratoflow/tts/retry.py`
- Modify: `src/narratoflow/tts/factory.py` (wrap edge and ElevenLabs; leave piper unwrapped)
- Modify: `README.md` (retry behaviour, plus the edge-tts risk: an unofficial, undocumented Microsoft endpoint that may break or be rate-limited without notice, with the fallback chain as the mitigation)
- Test: `tests/tts/test_retry.py`, `tests/tts/test_edge.py`, `tests/tts/test_elevenlabs.py`, `tests/tts/test_factory.py`

**Interfaces:**
- Produces:
  - `RETRY_ATTEMPTS = 3` and `RETRY_DELAYS_S = (1.0, 4.0)`
  - `class RetryingSpeechProvider` with `__init__(self, inner: SpeechProvider, *, sleep: Callable[[float], None] = time.sleep)`. Its `name` and `model` mirror `inner`, and its `synthesize(req) -> Speech` retries.
- Consumes: nothing from Tasks 1-2.

- [ ] **Step 1: Write the failing tests** in `tests/tts/test_retry.py`, using a scripted fake inner provider and a recording `sleep`:
  - `test_succeeds_after_transient`: the inner provider fails with `TransientProviderError` once, then succeeds. Expect 2 calls, sleeps `[1.0]`, and the inner provider's `Speech` back.
  - `test_gives_up_after_attempts`: the inner provider always fails transiently. Expect 3 calls and sleeps `[1.0, 4.0]`, and the last error is re-raised.
  - `test_plain_provider_error_not_retried`: one call, no sleep.
  - `test_billed_error_not_retried`: a `BilledProviderError` gives one call.
  - `test_retry_logged_without_text`: each retry logs one WARNING with the provider name, the attempt number and the delay. Assert that the request text is absent from the log.
  - `test_name_and_model_mirror_inner`.

- [ ] **Step 2: Write the failing adapter tests.**
  - edge, with the fake `edge_tts` module:
    - a stream that raises `OSError` gives `TransientProviderError`;
    - a deadline timeout gives `TransientProviderError`;
    - a malformed chunk gives a plain `ProviderError` that is not transient.
  - `test_edge_arabic_timings_monotonic`: a fake Arabic stream with word boundaries gives spans whose starts never decrease, and whose last end is at or before the audio's length.
  - ElevenLabs, with `httpx.MockTransport`:
    - `httpx.ConnectError` gives a transient error;
    - HTTP 429, 500, 502, 503 and 504 give transient errors;
    - HTTP 400 and 401 are not transient (401 stays a `ProviderAuthError`);
    - `test_read_timeout_not_transient`: `httpx.ReadTimeout` gives a plain `ProviderError`, because the request may already have been billed.
  - Factory: the chain's edge and ElevenLabs voices are `RetryingSpeechProvider`, and the piper voice is a `PiperProvider` (so `_model_file` still sees it).

- [ ] **Step 3: Run them.** Expected: they fail.

- [ ] **Step 4: Implement.**
  - edge: the existing library-exception branch and the timeout branch raise `TransientProviderError`, with the same messages. `_parse` errors stay `ProviderError`.
  - ElevenLabs: `httpx.ConnectError`, `httpx.ConnectTimeout` and `httpx.PoolTimeout` are transient. Other `TimeoutException`s and other `HTTPError`s stay plain. A status of 429 or 500 or above raises `TransientProviderError(self._http_error(response))`.
  - `RetryingSpeechProvider` catches only `TransientProviderError`, and the class docstring says so.
  - The factory wraps each provider at construction.

- [ ] **Step 5: Update the README,** then run the gate and commit: `feat(tts): retry transient voice failures with backoff`.

---

### Task 4: Duck the marker SFX under narration

**Files:**
- Create: `src/narratoflow/audio/duck.py`
- Modify: `src/narratoflow/audio/mixer.py:325` (`mixed = voice_bus + duck(sfx_bus, voice_bus)`), and add one sentence to the module docstring
- Modify: `README.md` (one line in the audio section)
- Test: `tests/audio/test_duck.py`, `tests/audio/test_mixer.py`

**Interfaces:**
- Produces:
  - `DUCK_DB = 8.0`
  - `DUCK_THRESHOLD_DBFS = -40.0`
  - `DUCK_FRAME_MS = 10`
  - `DUCK_ATTACK_MS = 30`
  - `DUCK_RELEASE_MS = 250`
  - `duck(sfx: Samples, voice: Samples) -> Samples`: same length and dtype as `sfx`, and raises `ValueError` on a length mismatch.
  - `duck_gain_db(voice: Samples) -> np.ndarray`: the per-sample gain in dB, exposed for tests.

- [ ] **Step 1: Write the failing tests** in `tests/audio/test_duck.py`, using synthetic 48 kHz numpy signals (use `aio.SAMPLE_RATE`):
  - `test_sfx_under_steady_voice_reduced`: a 1 s tone at -20 dBFS as the voice and constant noise as the SFX. After the first 100 ms, the RMS ratio of ducked to dry SFX is -8 dB, within 0.5 dB.
  - `test_sfx_in_silence_unchanged`: the voice is all zeros, and the output equals the input exactly.
  - `test_gain_bounded`: random voice gives `duck_gain_db` within [-8.0, 0.0].
  - `test_release_is_gradual`: the voice stops at 500 ms. At 600 ms the gain is strictly between -8 and 0. By 500 + 250 + 20 ms, it is back at 0 within 0.1 dB.
  - `test_attack_is_fast`: the voice starts at 200 ms, and the gain reaches -8 dB within 0.1 dB by 200 + 30 + 20 ms.
  - `test_length_mismatch_raises`.
  - In `tests/audio/test_mixer.py`, `test_mix_ducks_sfx_under_speech`: an SFX hit that lands during speech is quieter in the mix than the same hit in a gap. Compare the SFX contribution by subtracting a mix with no SFX pack.

- [ ] **Step 2: Run them.** Expected: they fail with an ImportError.

- [ ] **Step 3: Implement.**
  - Take the frame RMS of the voice in `DUCK_FRAME_MS` frames. The frame target is `-DUCK_DB` when its dBFS is above `DUCK_THRESHOLD_DBFS`, and 0 otherwise.
  - Slew-limit across frames: the gain falls by at most `DUCK_DB * DUCK_FRAME_MS / DUCK_ATTACK_MS` per frame, and rises by at most `DUCK_DB * DUCK_FRAME_MS / DUCK_RELEASE_MS` per frame.
  - Interpolate per sample with `np.interp` at frame centres.
  - Multiply by `10 ** (gain / 20)`.
  - Return the input unchanged (the same values) when no frame is active.

- [ ] **Step 4: Wire it into the mixer and update the README.** Run the gate, including a run with `NARRATOFLOW_FFMPEG=/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg`, and confirm the existing master-loudness tests still pass. Commit: `feat(audio): duck marker SFX under narration`.
