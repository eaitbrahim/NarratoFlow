# Plan 2: Script Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a pasted script into an approved, costed **series storyboard**. The steps are:
- declare the script's source and rights, and block in-copyright third-party work;
- detect Qur'anic verses against a verified corpus;
- split the text verbatim into scenes, choosing a drawable idea for each one with an LLM;
- vocalize the narration with the skeleton guard;
- pack the scenes into 60–90 s parts;
- estimate the cost and show the approval list, enforcing the spending limit.

Nothing in this plan generates images or audio.

**Architecture:**
- **Storyboard IR v2** (`models.py`): `Series → Part → Scene`, persisted as `series.json`.
- **Provider adapters** sit behind small Protocols (`LLMClient`, `Embedder`, `Diacritizer`). Paid is the default: Anthropic for the LLM, Voyage for embeddings. The free options are Ollama for both, and Mishkal for diacritics.
- **Every paid call goes through a `SpendTracker`.** It checks the per-part limit before the call and records the cost after it.
- **Planning is two LLM passes**, both with structured output:
  1. verbatim segmentation, enforced by a concatenation guard;
  2. idea assignment, given candidate keys from the channel library.
- **The CLI** adds `plan` and `approve`.

**Tech Stack:** Python 3.12, `anthropic` (1.x) SDK, httpx (Ollama and Voyage REST), pydantic 2, SQLAlchemy/Alembic (migration 0002), Typer, pytest.

**Spec:** `docs/superpowers/specs/2026-11-10-sketch-library-pipeline-design.md`. Sections implemented here: §2.2 (matching & scale), §2.2a–§2.2e (people, unseen, series, verbatim narration, verses: detection and on-screen text only), §3 steps 1–3, §4.1 (LLM and embedding rows), §4.2, §5.1, §5.2, the §6.3 script-source row, and §8 (v1 scope).

---

## Global Constraints

- **Narration is the author's exact words [spec §2.2d].** Segmentation may only split, and `"".join(scene.source_text for each scene)` must equal the normalized input with whitespace collapsed. Hooks, "Part N of M", recaps and teasers are **on-screen text only**.
- **Diacritization adds harakat only.** Every diacritizer output passes `assert_skeleton_preserved` (`text/tashkeel/verify.py`). On failure, keep the undiacritized text and set `scene.flags` to include `"tashkeel_rejected"`; the failure is never silent.
- **Protected text is never diacritized or rewritten.** Qur'anic verses are replaced by the verified corpus text, Uthmani for display. Verse scenes have `kind = SceneKind.VERSE` and no sketch.
- **People and the unseen [spec §2.2a–c]:**
  - no human figure is ever an idea;
  - prophets appear only as `name-card|…` or a non-figurative symbol;
  - God never appears as a visual, only as text;
  - angels and the unseen appear only as abstract symbols;
  - brands become generic objects.

  These rules are given to the LLM **and** enforced after it by `validate_idea` (Task 9).
- **Idea keys go through `parse_idea_key`** (Plan 1). Planner output that doesn't parse is rejected and retried once, then the scene fails loudly.
- **In-copyright third-party scripts are blocked before any spend [owner].**
- **Spending:**
  - every paid call goes through `SpendTracker.charge(...)`;
  - the per-part limit is `ChannelConfig.spending_limit_usd`;
  - an estimate above the limit forces the approval list even when `approval_list = false`;
  - planning cost is estimated and checked against the limit *before* the first LLM call.
- **Paid providers are the default; a missing key fails fast** (`check_provider_keys`, Plan 1). Free providers need no key: `ollama`, `edge`, `piper`, `local-diffusion`, plus `mishkal` (added here).
- **Secrets** are `SecretStr` from `Settings`. Add `embed_api_key` (env `NARRATOFLOW_EMBED_API_KEY`) and `ProviderKind.EMBED`. Keys are never logged or printed.
- **Anthropic usage [claude-api skill]:**
  - default model in examples and docs: `claude-opus-5-5`, but the model always comes from the channel config;
  - use `client.messages.parse(..., output_format=<PydanticModel>)` for structured output;
  - check `stop_reason == "refusal"` before reading content, then raise `ProviderRefusalError(category)`;
  - catch typed SDK errors most-specific-first, never a bare `APIStatusError` alone;
  - effort is configurable (`LlmConfig.effort`, default `"medium"`); no `temperature` and no prefill.
- **Tests never touch the network.** Providers are exercised through `httpx.MockTransport` or fakes implementing the Protocols.
- Same tooling gate as Plan 1: `ruff check --no-cache` → `ruff format --check` → `mypy` (strict, src+tests) → `coverage run -m pytest` → `coverage report`.
- Source files keep special characters as `\u` escapes, and Arabic regex ranges are ASCII-escaped. Visible Arabic letters in test data are fine.
- **Carry-overs from Plan 1:**
  - Migration 0002 adds CHECK constraints on `sketch.origin` and `sketch.status`.
  - Callers of `add_sketch` and `copy_to` must check `.status`. There are none in this plan.

## Review Focus

1. **A pasted script where the LLM "helpfully" fixes a typo or drops a repeated word during segmentation.** Expected: the concatenation guard rejects it, retries once, then raises `SegmentationError` naming the first differing offset. → Task 9 test.
2. **A verse quoted without ﴿﴾, with different hamza/alef spelling or partial diacritics** (the Yahya script quotes Maryam 19:12–13 in "…"). Expected: it is detected as surah:ayah after normalization, and partial matches below the certainty threshold are listed as *uncertain* in the approval list, not silently accepted. → Task 4 test.
3. **The LLM returns a person idea (`man|generic`, `prophet-yahya|generic` as a figure) or a brand logo.** Expected: `validate_idea` rejects it and the scene gets one retry with the violation stated, then fails loudly. It is never stored. → Task 9 test.
4. **A spending limit lower than the planning cost itself.** Expected: `BudgetExceededError` before any LLM call, with the estimate and the limit in the message. → Task 6 / Task 11 test.
5. **A script whose last part ends mid-thought** (the trailing `:` in reference script 1). Expected: the final part gets `flags=["unfinished_ending"]`, shown in the approval list. → Task 10 test.

---

## File Structure

```
src/narratoflow/
  models.py                    # MODIFY: IR v2 (Series, Part, Scene v2, ScriptSource, VerseRef, enums), v1→v2 upgrade
  errors.py                    # MODIFY: + ScriptBlockedError, ProviderError, ProviderRefusalError, BudgetExceededError, SegmentationError, PlanningError, VerseCorpusError
  settings.py                  # MODIFY: + embed_api_key
  channels/config.py           # MODIFY: LlmConfig.effort, + EmbedConfig, ChannelConfig.embed, diacritizer field
  channels/keys.py             # MODIFY: ProviderKind.EMBED, "mishkal" free
  script/__init__.py           # NEW
  script/source.py             # NEW: ScriptSource loading (front-matter) + rights gate
  quran/__init__.py            # NEW
  quran/normalize.py           # NEW: match_normalize()
  quran/corpus.py              # NEW: QuranCorpus (Tanzil txt-2 format), default_corpus()
  quran/detect.py              # NEW: detect_verses()
  llm/base.py                  # NEW: LLMClient protocol, StructuredResult, Usage
  llm/anthropic_client.py      # NEW
  llm/ollama.py                # REPLACE stub: OllamaClient
  llm/factory.py               # NEW: make_llm(channel, settings, tracker)
  embed/__init__.py            # NEW: Embedder protocol, VoyageEmbedder, OllamaEmbedder, make_embedder
  costs/__init__.py            # NEW
  costs/prices.py              # NEW: PriceTable (TOML), default table path
  costs/tracker.py             # NEW: SpendTracker, estimate_* helpers
  text/tashkeel/engines.py     # NEW: LlmDiacritizer, MishkalDiacritizer, vocalize_scene()
  library/migrations/versions/0002_embeddings_checks.py  # NEW
  library/orm.py               # MODIFY: IdeaEmbedding table + CHECKs
  library/candidates.py        # NEW: candidate_keys()
  planning/__init__.py         # NEW
  planning/prompts.py          # NEW: frozen system prompts (rules)
  planning/segment.py          # NEW: segment_script()
  planning/ideas.py            # NEW: assign_ideas(), validate_idea()
  planning/series.py           # NEW: split_into_parts()
  planning/approval.py         # NEW: render_approval_list(), approval gate
  planning/pipeline.py         # NEW: plan_series() orchestration
  cli.py                       # MODIFY: plan, approve commands
assets/quran/README.md, LICENSE-tanzil.txt   # NEW (corpus files fetched by script, not committed)
scripts/fetch_quran.py         # NEW: downloads Tanzil simple-clean + uthmani txt-2 with sha256 check
configs/prices.toml            # NEW: editable price table
tests/fixtures/quran/          # NEW: small Tanzil-format subset (Luqman 31:33, Maryam 19:1-15)
```

`text/scenes.py`, `text/protected.py` and `text/normalize.py` are unused stubs. Delete them in Task 1, because their roles move to `planning/` and `quran/`.

---

### Task 1: Storyboard IR v2 and errors

**Files:** Modify `src/narratoflow/models.py` and `src/narratoflow/errors.py`. Delete the stubs listed above. Tests: `tests/test_models.py` (extend).

**Interfaces — Produces:**
- `SCHEMA_VERSION = 2`.
- `class SceneKind(StrEnum): NARRATION="narration"; VERSE="verse"`.
- `class RightsBasis(StrEnum): OWN="own"; PUBLIC_DOMAIN="public_domain"; PERMISSION="permission"; THIRD_PARTY_IN_COPYRIGHT="third_party_in_copyright"`.
- `class ScriptSource(BaseModel)`:
  - `author: str`, `work: str`, `edition: str = ""`;
  - `rights: RightsBasis`, `permission_note: str = ""`;
  - `ai_transcribed: bool = False`.
- `class VerseRef(BaseModel)`:
  - `surah: int = Field(ge=1, le=114)`, `ayah_start: int = Field(ge=1)`, `ayah_end: int = Field(ge=1)` (validator: `ayah_end >= ayah_start`);
  - `certain: bool`, `uthmani_text: str`.
- `Scene` v2 keeps all existing fields and adds:
  - `kind: SceneKind = SceneKind.NARRATION`, `verse: VerseRef | None = None`;
  - `idea_key: str | None = None`, `idea_description: str = ""`;
  - `match: Literal["reused", "bundled", "new"] | None = None`;
  - `est_duration_s: float = 0`, `flags: list[str] = []`.
- `class Part(BaseModel)`:
  - `index: int = Field(ge=1)`, `count: int = Field(ge=1)`;
  - `hook_text: str`, `recap_text: str = ""`, `teaser_text: str = ""`;
  - `scenes: list[Scene]`, `flags: list[str] = []`;
  - `est_cost_usd: Decimal = Decimal(0)`, property `est_duration_s`.
- `class Series(BaseModel)`:
  - `schema_version: int = SCHEMA_VERSION`, `title: str`, `channel_slug: str`, `language: Language`;
  - `source: ScriptSource`, `parts: list[Part]`;
  - `approved: bool = False`, `est_planning_cost_usd: Decimal = Decimal(0)`, `ai_disclosure: bool = True`;
  - `save(path)` and `load(path)` as in `Project`. `load` raises `PlanningError` on a schema_version other than 2.
- `Project` (v1) stays for `validate`, unchanged.
- `errors.py` adds, each subclassing `NarratoFlowError`:
  - `ScriptBlockedError`;
  - `ProviderError`, with subclass `ProviderRefusalError(category: str | None)`;
  - `BudgetExceededError(estimate_usd: Decimal, limit_usd: Decimal)`, whose message includes both values;
  - `SegmentationError`, `PlanningError`, `VerseCorpusError`.

- [ ] **Step 1: Write the failing tests**

```python
def test_series_roundtrip(tmp_path: Path) -> None:
    s = Series(title="t", channel_slug="history", language=Language.ARABIC,
               source=ScriptSource(author="a", work="w", rights=RightsBasis.OWN),
               parts=[Part(index=1, count=1, hook_text="h",
                           scenes=[Scene(id="p1s1", narration=Narration(source_text="x"), est_duration_s=4.5)])])
    s.save(tmp_path / "series.json")
    assert Series.load(tmp_path / "series.json") == s
    assert s.parts[0].est_duration_s == 4.5

def test_verse_ref_range_validated() -> None:
    with pytest.raises(ValidationError):
        VerseRef(surah=19, ayah_start=13, ayah_end=12, certain=True, uthmani_text="x")

def test_series_load_rejects_other_schema(tmp_path: Path) -> None: ...  # schema_version=1 JSON -> PlanningError

def test_budget_error_message_has_values() -> None:
    e = BudgetExceededError(Decimal("1.20"), Decimal("1.00"))
    assert "1.20" in str(e) and "1.00" in str(e)
```

- [ ] **Step 2:** Run `uv run pytest tests/test_models.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement per the Interfaces block and delete the three stubs.
- [ ] **Step 4:** Run the gate. Expected: green.
- [ ] **Step 5:** Commit: `feat: storyboard IR v2 (series, parts, verse refs) and planning errors`.

---

### Task 2: Script source declaration and rights gate

**Files:** Create `src/narratoflow/script/__init__.py` and `script/source.py`. Test: `tests/script/test_source.py`.

**Interfaces — Consumes:** `ScriptSource`, `RightsBasis`, `ScriptBlockedError`, `require_clean` (Plan 1).

**Interfaces — Produces:**
- `load_script(path: Path, language: Language) -> tuple[ScriptSource, str]`:
  - The file has TOML front matter between `+++` lines: `author`, `work`, `edition`, `rights`, `permission_note`, `ai_transcribed`. The body follows.
  - The body goes through `require_clean`, so corrupt text raises `CorruptInputError`.
  - Missing front matter raises `ScriptBlockedError("declare the script source …")` with an example block in the message.
- `check_rights(source: ScriptSource) -> None`:
  - `THIRD_PARTY_IN_COPYRIGHT` raises `ScriptBlockedError` (owner: blocked, no permission path in v1);
  - `PERMISSION` with an empty `permission_note` raises `ScriptBlockedError`.
- `load_script` calls `check_rights` **before** returning, so no caller can skip it.

- [ ] **Step 1: Write the failing tests**

```python
def test_own_script_loads(tmp_path: Path) -> None: ...            # front matter rights="own" + clean_ar body -> (source, body)
def test_in_copyright_blocked(tmp_path: Path) -> None:
    p = write_script(tmp_path, rights="third_party_in_copyright", body=read("clean_en.txt"))
    with pytest.raises(ScriptBlockedError, match="copyright"):
        load_script(p, Language.ENGLISH)
def test_permission_requires_note(tmp_path: Path) -> None: ...    # rights="permission", no note -> ScriptBlockedError
def test_missing_front_matter(tmp_path: Path) -> None: ...        # -> ScriptBlockedError whose message contains "+++"
def test_corrupt_body_rejected(tmp_path: Path) -> None: ...       # corrupt_ar body -> CorruptInputError
def test_ai_transcribed_flag_kept(tmp_path: Path) -> None: ...
```

- [ ] **Steps 2–5:** Red, implement, run the gate, then commit `feat: script source declaration with rights gate`.

---

### Task 3: Qur'an corpus and match normalization

**Files:**
- Create `quran/__init__.py`, `quran/normalize.py`, `quran/corpus.py`, `scripts/fetch_quran.py`, `assets/quran/README.md` and `assets/quran/LICENSE-tanzil.txt`.
- Create `tests/fixtures/quran/quran-simple-clean.txt` and `quran-uthmani.txt`: a subset of Luqman 31:33 and Maryam 19:1–15 in Tanzil `txt-2` format (`sura|aya|text`), copied verbatim from Tanzil.
- Test: `tests/quran/test_corpus.py`.
- Add `assets/quran/*.txt` to `.gitignore`. The corpus is fetched, not committed. Pin the sha256 of each file in `fetch_quran.py`, so a changed upstream file fails loudly.

**Interfaces — Produces:**
- `match_normalize(text: str) -> str`. It runs NFC, then:
  - strips harakat, Qur'anic annotation marks (U+06D6–U+06ED), tatweel and superscript alef;
  - maps alef forms (U+0622, U+0623, U+0625, U+0671) to U+0627, U+0649 to U+064A, U+0629 to U+0647, and U+0624/U+0626 to U+0621;
  - removes punctuation and ﴿﴾;
  - collapses whitespace.

  It is pure, and all ranges are written as `\u` escapes.
- `@dataclass(frozen=True) class Ayah: surah: int; ayah: int; simple: str; uthmani: str; norm_words: tuple[str, ...]`.
- `class QuranCorpus`:
  - `from_files(simple: Path, uthmani: Path) -> QuranCorpus`. Mismatched surah/ayah keys raise `VerseCorpusError`.
  - `ayah(surah, ayah) -> Ayah`.
  - `__iter__`, which iterates over ayat.
  - `text_range(surah, start, end) -> str`, returning the joined Uthmani text.
- `default_corpus(settings: Settings) -> QuranCorpus`. It reads from `settings.data_dir / "quran"`, or `assets/quran` in a dev checkout. If the files are missing, it raises `VerseCorpusError("run `uv run python scripts/fetch_quran.py` …")`.
- `scripts/fetch_quran.py`:
  - downloads the `simple-clean` and `uthmani` `txt-2` files from tanzil.net, keeping Tanzil's licence header lines intact;
  - verifies the pinned sha256 and writes to the given dir.
  - It has no tests that hit the network. It may only be smoke-run manually, and that is noted in its docstring.
- `LICENSE-tanzil.txt` reproduces Tanzil's terms: verbatim copy, attribution, link to tanzil.net.

- [ ] **Step 1: Write the failing tests**

```python
def test_normalize_spelling_variants() -> None:
    assert match_normalize("إِنَّ") == match_normalize("ان")
    assert match_normalize("يَحْيَىٰ") == match_normalize("يحيي")
    assert match_normalize("﴿الحمد لله﴾") == "الحمد لله"

def test_corpus_loads_fixture() -> None:
    c = QuranCorpus.from_files(FIXTURES / "quran/quran-simple-clean.txt", FIXTURES / "quran/quran-uthmani.txt")
    assert c.ayah(31, 33).surah == 31
    assert "يوم" in c.ayah(31, 33).simple

def test_corpus_key_mismatch(tmp_path: Path) -> None: ...  # uthmani file missing one ayah -> VerseCorpusError
def test_default_corpus_missing_explains_fetch(tmp_path, monkeypatch) -> None: ...  # -> VerseCorpusError mentioning fetch_quran.py
```

- [ ] **Steps 2–5:** Red, implement, run the gate, then commit `feat: verified Qur'an corpus loader and match normalization`.

---

### Task 4: Verse detection

**Files:** Create `quran/detect.py`. Test: `tests/quran/test_detect.py`.

**Interfaces — Consumes:** `QuranCorpus`, `match_normalize`, `VerseRef`.

**Interfaces — Produces:**
- `@dataclass(frozen=True) class VerseHit: start: int; end: int; ref: VerseRef`. Here `start` and `end` are character offsets into the **original** text.
- `detect_verses(text: str, corpus: QuranCorpus, *, min_words: int = 4, certain_words: int = 6) -> list[VerseHit]`.

**Algorithm (decided here):**
1. Tokenize the original text into words, keeping each word's character span. Normalize each word with `match_normalize`.
2. Build an index from each normalized 4-gram to its `(surah, ayah, word_pos)` in the corpus.
3. Scan the text for 4-gram hits and extend each hit greedily forward while words keep matching consecutively, allowing a crossing into the next ayah of the same surah.
4. Keep the maximal non-overlapping runs of at least `min_words` words, preferring the longest.
5. A run is `certain = True` if it covers a whole ayah (or ayah range) exactly, or is at least `certain_words` long. Otherwise it is uncertain.
6. Fill `ref.uthmani_text` from the corpus for the covered ayah range. For a partial ayah, use the whole ayah: the verified text is shown in full, never a fragment.

Results are sorted by `start`. Arabic-free text returns `[]` quickly.

- [ ] **Step 1: Write the failing tests** (using the fixture corpus)

```python
def test_bracketed_full_ayah_certain(corpus) -> None:
    text = "قال تعالى ﴿يَوْمًا لَا يَجْزِي وَالِدٌ عَنْ وَلَدِهِ وَلَا مَوْلُودٌ هُوَ جَازٍ عَنْ وَالِدِهِ شَيْئًا﴾ ..."
    [hit] = detect_verses(text, corpus)
    assert (hit.ref.surah, hit.ref.ayah_start, hit.ref.ayah_end, hit.ref.certain) == (31, 33, 33, True)
    assert text[hit.start:hit.end].startswith("يَوْمًا") or "يوم" in text[hit.start:hit.end]

def test_unbracketed_quote_with_variant_spelling_detected(corpus) -> None:   # Review Focus 2
    text = 'قال الله تعالى "وآتيناه الحكم صبيا وحنانا من لدنا وزكاة وكان تقيا" فتأمل'
    [hit] = detect_verses(text, corpus)
    assert (hit.ref.surah, hit.ref.ayah_start, hit.ref.ayah_end) == (19, 12, 13)

def test_short_overlap_is_uncertain(corpus) -> None: ...   # 4-5 matching words -> certain is False
def test_three_words_ignored(corpus) -> None: ...          # below min_words -> []
def test_english_text_no_hits(corpus) -> None: ...
```

- [ ] **Steps 2–5:** Red, implement, run the gate, then commit `feat: corpus-based Qur'anic verse detection`.

---

### Task 5: LLM clients (Anthropic paid default, Ollama free)

**Files:**
- Create `llm/base.py` and `llm/anthropic_client.py`, and replace `llm/ollama.py`. (The factory lives in Task 6, because it needs the spend tracker.)
- Modify `channels/config.py`: `LlmConfig` gets `effort: Literal["low","medium","high","xhigh","max"] = "medium"`, `max_tokens: int = 16000` and `refusal_fallback: bool = True`.
- Modify `pyproject.toml` (`uv add "anthropic>=1"`).
- Tests: `tests/llm/test_anthropic_client.py`, `tests/llm/test_ollama.py` and `tests/llm/fakes.py`.

**Interfaces — Produces:**
- `class Usage(BaseModel): input_tokens: int = 0; output_tokens: int = 0`.
- `@dataclass(frozen=True) class StructuredResult[T: BaseModel]: value: T; usage: Usage; model: str`.
- `class LLMClient(Protocol)`:
  - `name: str`;
  - `def complete_structured[T: BaseModel](self, *, system: str, user: str, output: type[T]) -> StructuredResult[T]: ...`.
- `AnthropicClient(api_key: SecretStr, model: str, effort: str, max_tokens: int, refusal_fallback: bool, client: anthropic.Anthropic | None = None)`:
  - uses `client.messages.parse(model=…, max_tokens=…, system=…, messages=[{"role":"user","content":user}], output_format=output, output_config={"effort": effort})`;
  - when `refusal_fallback` is set and the model supports it, sends the beta `server-side-fallback-2026-07-01` with `fallbacks="default"` through the beta messages surface. **Verify the SDK binding** (`client.beta.messages.parse` and the `betas`/`fallbacks` kwargs) against the installed `anthropic` package. If it isn't available, send without fallback and log once at INFO;
  - checks `stop_reason == "refusal"` and raises `ProviderRefusalError(category)` from `stop_details`;
  - raises `ProviderError` if `parsed_output` is None;
  - maps typed SDK errors (`AuthenticationError`, `NotFoundError`, `RateLimitError`, `APIStatusError`, `APIConnectionError`) to `ProviderError` with a message that never includes the key, chaining with `raise … from e`.
- `OllamaClient(base_url: str, model: str, http: httpx.Client | None = None)` posts to `/api/chat` with `format=output.model_json_schema()` and `stream=false`, then validates `message.content` with `output.model_validate_json`. Invalid JSON or a validation failure raises `ProviderError`. Usage comes from `prompt_eval_count` / `eval_count`.
- `tests/llm/fakes.py`: `FakeLLM(responses: list[BaseModel | Exception])`, which records calls and returns queued values with fixed usage.

- [ ] **Step 1: Write the failing tests** (no network)
  - Anthropic: build a real `anthropic.Anthropic(api_key="test", http_client=…)` on an httpx2 mock transport returning canned Messages API JSON, or monkeypatch `client.messages.parse` / `client.beta.messages.parse`. Cover: the success path (parsed value + usage), refusal → `ProviderRefusalError` with category, a 401 → `ProviderError` whose message doesn't contain the key, and the effort being passed through in `output_config`.
  - Ollama: `httpx.MockTransport` for a valid JSON reply, invalid JSON → `ProviderError`, and a schema sent in `format`.
- [ ] **Steps 2–5:** Red, implement, run the gate, then commit `feat: structured-output LLM clients (Anthropic default, Ollama free)`.

---

### Task 6: Price table, spend tracker, estimates

**Files:** Create `costs/__init__.py`, `costs/prices.py`, `costs/tracker.py`, `llm/factory.py` and `configs/prices.toml`. Modify `settings.py` (`ollama_url: str = "http://localhost:11434"`). Tests: `tests/costs/test_prices.py`, `tests/costs/test_tracker.py` and `tests/llm/test_factory.py`.

**Interfaces — Consumes:** `Usage`, `LLMClient`, `AnthropicClient`, `OllamaClient` (Task 5).

**Interfaces — Produces:**
- `configs/prices.toml` has creator-editable USD prices, with a dated header comment:

```toml
# Prices change — edit these. Checked 2026-11-10.
[llm."anthropic/claude-opus-5-5"]
input_per_mtok = "4.00"
output_per_mtok = "20.00"
[llm."anthropic/claude-sonnet-5-5"]
input_per_mtok = "2.00"
output_per_mtok = "10.00"
[llm."anthropic/claude-haiku-5-5"]
input_per_mtok = "0.10"
output_per_mtok = "0.50"
[embed."voyage/voyage-multilingual-2"]
per_mtok = "0.12"
[image."openai/gpt-image-1"]
per_image = "0.04"
[vision."anthropic/claude-opus-5-5"]
per_image_check = "0.01"
[voice."elevenlabs/eleven_multilingual_v2"]
per_kchar = "0.30"
```

  Free providers (`ollama/*`, `edge/*`, `piper/*`, `mishkal/*`, `local-diffusion/*`) cost 0 implicitly.
- `class PriceTable`:
  - `load(path: Path) -> PriceTable`, where invalid TOML or values raise `ChannelConfigError`;
  - `llm(provider, model) -> tuple[Decimal, Decimal]`, `embed(...) -> Decimal`, `image(...)`, `vision(...)`, `voice(...)`.

  An unknown **paid** provider/model raises `PlanningError("no price for X; add it to prices.toml")`. A free provider returns 0.
- `default_prices(settings) -> PriceTable`. It reads `settings.data_dir/"prices.toml"` if present, else the packaged `configs/prices.toml`. The file is located robustly, and the plan accepts moving it under `src/narratoflow/data/` if packaging requires it.
- `class SpendTracker(limit_usd: Decimal | None)`:
  - `spent: Decimal`;
  - `ensure_room(estimate: Decimal, what: str) -> None` raises `BudgetExceededError` when `spent + estimate > limit`;
  - `charge_llm(provider, model, usage: Usage, prices: PriceTable) -> Decimal`;
  - `charge(amount: Decimal, what: str) -> None`;
  - `entries: list[SpendEntry]`, where `SpendEntry` is `what, amount_usd, at` (UTC).
- Estimators, all deterministic and documented in docstrings:
  - `estimate_tokens(text: str) -> int`: `ceil(len(text) / 3)` for Arabic and `ceil(len(text) / 4)` otherwise, using the Arabic-letter ratio above 0.3;
  - `estimate_planning_cost(text, channel, prices) -> Decimal`: two passes, each `input = system prompt tokens + text tokens`, `output = 1.2 × text tokens`, using the LLM price, plus the embedding cost when the catalog is above 300 keys;
  - `estimate_part_cost(part, channel, prices, *, voice_chars: int) -> Decimal`: new sketches × (image + vision) plus voice characters × voice price.

- `make_llm(channel: ChannelConfig, settings: Settings, tracker: SpendTracker) -> LLMClient`:
  - provider `"anthropic"` builds an `AnthropicClient` (missing key raises `MissingApiKeyError`);
  - `"ollama"` builds an `OllamaClient` with base_url from `Settings.ollama_url` (default `http://localhost:11434`);
  - any other provider raises `ChannelConfigError`;
  - the returned client is wrapped by `TrackedLLM(inner, tracker, provider, model)`, which calls `tracker.charge_llm(...)` with the usage after each call .
  - `make_llm` takes a `prices: PriceTable` argument too, passed to `TrackedLLM`.

- [ ] **Step 1: Write the failing tests**

```python
def test_free_provider_costs_zero(prices) -> None:
    assert prices.llm("ollama", "qwen2.5") == (Decimal(0), Decimal(0))

def test_unknown_paid_model_explains(prices) -> None:
    with pytest.raises(PlanningError, match="prices.toml"):
        prices.llm("anthropic", "made-up")

def test_tracker_blocks_before_spend() -> None:                        # Review Focus 4
    t = SpendTracker(Decimal("0.05"))
    with pytest.raises(BudgetExceededError):
        t.ensure_room(Decimal("0.06"), "planning")
    assert t.spent == 0

def test_charge_llm_uses_prices(prices) -> None: ...   # 1M in + 1M out on opus-5-5 -> 24.00
def test_no_limit_never_blocks() -> None: ...
def test_estimate_tokens_arabic_vs_english() -> None: ...
def test_make_llm_unknown_provider() -> None: ...       # -> ChannelConfigError
def test_make_llm_anthropic_without_key() -> None: ...  # -> MissingApiKeyError
def test_tracked_llm_charges_usage() -> None: ...       # FakeLLM usage -> tracker.spent > 0
```

- [ ] **Steps 2–5:** Red, implement, run the gate, then commit `feat: editable price table, spend tracker and cost estimates`.

---

### Task 7: Diacritizers with the skeleton guard

**Files:** Create `text/tashkeel/engines.py`. Modify `channels/config.py` to add `diacritizer: Literal["llm", "mishkal"] = "llm"`, and modify `channels/keys.py` to add `"mishkal"` to the free providers. Test: `tests/text/test_tashkeel_engines.py`.

**Interfaces — Consumes:** `LLMClient`, `assert_skeleton_preserved`, `SkeletonMismatchError`, `Scene`, `SceneKind`.

**Interfaces — Produces:**
- `class Diacritizer(Protocol): name: str; def diacritize(self, text: str) -> str: ...` (the existing `interfaces.Diacritizer` shape).
- `LlmDiacritizer(llm: LLMClient)` sends a `Vocalized(text: str)` output model. The frozen system prompt in `planning/prompts.py` is `TASHKEEL_SYSTEM`: "Add full Classical Arabic diacritics (harakat, tanwin, shadda, sukun) including case endings. Do not change, add, remove or reorder any letter, word or punctuation. Return only the text."
- `MishkalDiacritizer()` is a lazy import of `mishkal.tashkeel` (extra `arabic`). If it isn't installed, it raises `ChannelConfigError("install the 'arabic' extra …")`.
- `vocalize_scene(scene: Scene, d: Diacritizer) -> Scene`:
  - For `VERSE` scenes, or when `scene.narration.protected` is set, it returns the scene unchanged.
  - Otherwise it calls `d.diacritize(source_text)`, then `assert_skeleton_preserved`.
  - On success it sets `narration.tts_text` and sets `caption_text = source_text`.
  - On a mismatch or a `ProviderError` it sets `tts_text = source_text` and appends `"tashkeel_rejected"` to `flags`. The mismatch is logged at WARNING with the scene id only, never the text.
  - It returns a new Scene (`model_copy`).

- [ ] **Step 1: Write the failing tests** (FakeLLM)

```python
def test_llm_diacritics_accepted() -> None:
    d = LlmDiacritizer(FakeLLM([Vocalized(text="اَلْعِلْمُ نُورٌ")]))
    s = vocalize_scene(scene("العلم نور"), d)
    assert s.narration.tts_text == "اَلْعِلْمُ نُورٌ" and s.narration.caption_text == "العلم نور" and not s.flags

def test_word_change_rejected_and_flagged() -> None:
    d = LlmDiacritizer(FakeLLM([Vocalized(text="اَلْعِلْمُ ضِيَاءٌ")]))
    s = vocalize_scene(scene("العلم نور"), d)
    assert s.narration.tts_text == "العلم نور" and "tashkeel_rejected" in s.flags

def test_verse_scene_untouched() -> None: ...
def test_provider_error_flagged_not_raised() -> None: ...
def test_mishkal_missing_extra_explains(monkeypatch) -> None: ...
```

- [ ] **Steps 2–5:** Red, implement, run the gate, then commit `feat: LLM and Mishkal diacritizers behind the skeleton guard`.

---

### Task 8: Embeddings, migration 0002, candidate keys

**Files:**
- Create `embed/__init__.py`, `library/candidates.py` and `library/migrations/versions/0002_embeddings_checks.py`.
- Modify `library/orm.py`, `settings.py` (`embed_api_key: SecretStr | None = None`), `channels/config.py` (`class EmbedConfig(BaseModel): provider: str = "voyage"; model: str = "voyage-multilingual-2"`; `ChannelConfig.embed: EmbedConfig = EmbedConfig()`) and `channels/keys.py` (`ProviderKind.EMBED` → `NARRATOFLOW_EMBED_API_KEY`; Voyage is paid).
- Tests: `tests/embed/test_embedders.py`, `tests/library/test_candidates.py` and extend `tests/library/test_db.py`.

**Interfaces — Produces:**
- `class Embedder(Protocol): name: str; model: str; def embed(self, texts: Sequence[str]) -> list[list[float]]: ...`.
- `VoyageEmbedder(api_key: SecretStr, model: str, http: httpx.Client | None)` makes a REST `POST https://api.voyageai.com/v1/embeddings` with `{"input": texts, "model": model}` and bearer auth. It batches at 128 and raises `ProviderError` on non-2xx without echoing the key.
- `OllamaEmbedder(base_url, model, http)` makes a `POST /api/embed` with `{"model", "input"}`.
- `make_embedder(channel, settings, tracker) -> Embedder` wraps the embedder with cost tracking, using `prices.embed` × the token estimate.
- ORM table `idea_embedding`:
  - columns `id`, `style_version_id` FK (indexed), `idea_key`, `description`, `embed_model`, `vector` (`LargeBinary`, float32 little-endian), `created_at` (`UTCDateTime`);
  - `UNIQUE(style_version_id, idea_key, embed_model, description)`.
- Migration `0002`:
  - creates `idea_embedding`;
  - adds the CHECK constraints `sketch.status IN ('passed','unchecked','failed')` and `sketch.origin IN ('generated','bundled','copied')`. Use `batch_alter_table`, because SQLite needs a table rebuild, and keep the ORM in sync (`create_constraint=True` with matching names) so `test_migrations_match_models` stays empty;
  - has a working `downgrade`.
- `candidate_keys(library: SketchLibrary, engine: Engine, style_version_id: int, scenes: Sequence[str], embedder: Embedder | None, *, full_limit: int = 300, top_k: int = 5) -> list[list[tuple[str, str]]]`:
  - With 300 catalog entries or fewer, **every** scene gets the full catalog, with no embedding call.
  - Above 300, it embeds catalog descriptions, cached in `idea_embedding` and computed only for missing rows. It then embeds the scene texts, and returns the cosine top-k per scene. Cosine is computed in Python and numpy is allowed.
  - Above 300 with no embedder, it raises `PlanningError`.

- [ ] **Step 1: Write the failing tests**
  - Voyage and Ollama with MockTransport: request shape, batching at 128, 401 → `ProviderError` without the key in the message.
  - `test_upgrade_from_0001_adds_checks`: stamp a DB at 0001 with a sketch row, upgrade, then insert `status='bogus'` → `IntegrityError`; also test downgrade.
  - `test_migrations_match_models` must still pass.
  - Candidates with a small library (≤300) return the full catalog for every scene with zero embed calls (FakeEmbedder counts calls). With `full_limit=2` and 3 keys, a deterministic fake embedder returns top-k by cosine, and catalog embeddings are cached (second run embeds 0 catalog items).
- [ ] **Steps 2–5:** Red, implement, run the gate, then commit `feat: embeddings, idea-embedding cache (migration 0002 + CHECKs), candidate keys`.

---

### Task 9: Segmentation and idea assignment

**Files:** Create `planning/__init__.py`, `planning/prompts.py`, `planning/segment.py` and `planning/ideas.py`. Tests: `tests/planning/test_segment.py` and `tests/planning/test_ideas.py`.

**Interfaces — Consumes:** `LLMClient`, `FakeLLM`, `VerseHit`, `parse_idea_key`, `IdeaKey`, `Scene`, `Narration`, `SceneKind`.

**Interfaces — Produces:**
- `planning/prompts.py` holds frozen module constants, with no timestamps or IDs so prompt caching works:
  - `SEGMENT_SYSTEM`: split into scenes of 1–3 sentences at sentence or clause boundaries, about 5–10 s each; never change text; return spans as exact substrings in order; propose a short on-screen `title` per paragraph group in the script's language.
  - `IDEAS_SYSTEM`:
    - the drawing rules from spec §2.2a–§2.2c verbatim: no human figures; prophets only as a name card or a non-figurative symbol; God never as a visual; angels and the unseen only as abstract symbols; brands as generic objects;
    - the key format `object|context` in ASCII kebab-case, with `generic` when culture/period doesn't matter;
    - reuse a candidate key whenever it's the same object and same culture/period;
    - a ≤15-word English description of the drawing.
- `class SegmentOut(BaseModel): scenes: list[SegmentSpan]` with `SegmentSpan(text: str, title: str)`.
- `segment_script(text: str, verses: Sequence[VerseHit], llm: LLMClient) -> list[Scene]`:
  - **Verse spans are cut out first** and become `VERSE` scenes, with `narration.source_text` = the original span, `narration.protected = True` and `verse = hit.ref`.
  - Only the non-verse stretches are sent to the LLM, one call per stretch.
  - **Verbatim guard:** `_collapse_ws("".join(spans)) == _collapse_ws(stretch)`. On a mismatch it retries once with an appended note naming the first differing offset. A second mismatch raises `SegmentationError` naming the offset.
  - Scene ids are `s001`, `s002`, … in text order.
- `class IdeaOut(BaseModel): idea_key: str; description: str; depicts_person: bool; depicts_logo: bool`.
- `validate_idea(out: IdeaOut) -> list[str]` returns violations:
  - `depicts_person` or `depicts_logo` is true;
  - the key's object is in the `FORBIDDEN_OBJECTS` set (`man, woman, person, people, child, boy, girl, prophet, figure, face, silhouette, angel, god, allah, logo`);
  - the key fails `parse_idea_key`.

  `name-card|*` keys are always allowed.
- `assign_ideas(scenes: list[Scene], candidates: list[list[tuple[str, str]]], llm: LLMClient) -> list[Scene]`:
  - One call per NARRATION scene, with the scene text plus its candidate list rendered as `key — description` lines.
  - On violations it retries once with the violations listed. If it still fails, it raises `PlanningError(scene id + violations)`.
  - On success it sets `idea_key` = `str(parse_idea_key(...))` and `idea_description`.
  - VERSE scenes are skipped. They have no idea and no sketch.

- [ ] **Step 1: Write the failing tests** (FakeLLM)

```python
def test_verbatim_segmentation_and_verse_cut() -> None: ...          # verse span becomes VERSE scene; others NARRATION; join == input
def test_rewrite_rejected_after_one_retry() -> None:                  # Review Focus 1
    text = "العلم نور. والجهل ظلام."
    llm = FakeLLM([SegmentOut(scenes=[SegmentSpan(text="العلم نور.", title="t"), SegmentSpan(text=" والجهل ظلم.", title="t")])] * 2)
    with pytest.raises(SegmentationError, match="offset"):
        segment_script(text, [], llm)
    assert len(llm.calls) == 2

def test_person_idea_rejected_then_fixed() -> None: ...               # Review Focus 3: man|generic -> retry -> eye|generic accepted
def test_person_idea_rejected_twice_raises() -> None: ...
def test_name_card_allowed() -> None:
    assert validate_idea(IdeaOut(idea_key="name-card|yahya", description="name", depicts_person=False, depicts_logo=False)) == []
def test_logo_rejected() -> None: ...
def test_candidates_rendered_in_prompt() -> None: ...                 # FakeLLM call text contains "road|two-horizons — a road"
```

- [ ] **Steps 2–5:** Red, implement, run the gate, then commit `feat: verbatim scene segmentation and rule-checked idea assignment`.

---

### Task 10: Series splitting and on-screen text

**Files:** Create `planning/series.py`. Test: `tests/planning/test_series.py`.

**Interfaces — Produces:**
- `estimate_scene_seconds(scene: Scene, language: Language) -> float`:
  - Narration is words ÷ (120 wpm for Arabic, 150 for English) × 60, plus a 0.4 s pause.
  - Verse scenes use the Arabic word count ÷ 80 wpm × 60 (recitation is slower), plus 0.8 s of buffers (spec §2.2e smoothing).
- `split_into_parts(scenes: list[Scene], titles: list[str], *, min_s: int, max_s: int) -> list[Part]`:
  - It packs scenes greedily, in order, into parts. A scene is never split.
  - It closes a part when adding the next scene would exceed `max_s`. If closing would leave the part under `min_s`, it still closes it, unless the part is empty, in which case the oversized scene is kept alone and flagged `"over_max"`.
  - It merges a final part shorter than `min_s / 2` into the previous one if that stays ≤ `max_s × 1.15`.
  - **On-screen text:**
    - `hook_text` is the first scene's segment title;
    - `recap_text` (part 2 onwards) is the previous part's hook;
    - `teaser_text` (all but the last part) is the next part's hook.
  - It sets `index`, `count` and `est_duration_s` per scene.
  - **Unfinished ending:** if the last narration scene of the **last** part ends with `:`, `،` or `,`, or has no terminal punctuation (`.`, `!`, `?`, `؟`, `…`, `»`, `"`), that part gets `flags=["unfinished_ending"]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_parts_respect_bounds() -> None: ...            # 20 scenes of 8 s, 60-90 -> parts each <= 90 s, >= 60 except possibly last
def test_scene_never_split_oversized_flagged() -> None: ...
def test_tiny_tail_merged() -> None: ...
def test_hook_recap_teaser_chain() -> None: ...         # 3 parts: p2.recap == p1.hook, p2.teaser == p3.hook, p3.teaser == ""
def test_unfinished_ending_flagged() -> None:            # Review Focus 5
    parts = split_into_parts([narr("... في استطراد عجيب:")], ["t"], min_s=60, max_s=90)
    assert "unfinished_ending" in parts[-1].flags
def test_verse_duration_slower_than_narration() -> None: ...
```

- [ ] **Steps 2–5:** Red, implement, run the gate, then commit `feat: series splitting with on-screen hook, recap and teaser`.

---

### Task 11: Planning pipeline, approval list, CLI

**Files:** Create `planning/approval.py` and `planning/pipeline.py`. Modify `cli.py`. Tests: `tests/planning/test_pipeline.py` and `tests/test_cli.py` (extend).

**Interfaces — Consumes:** everything above, plus Plan 1's `load_channel`, `check_provider_keys`, `default_library`, `style_fingerprint`, `match_idea`, `BundledIcons`, `parse_idea_key`.

**Interfaces — Produces:**
- `plan_series(script_path: Path, channel: ChannelConfig, settings: Settings, *, llm: LLMClient | None = None, embedder: Embedder | None = None, corpus: QuranCorpus | None = None, prices: PriceTable | None = None) -> Series`. The injectable dependencies are for tests. It runs, in order:
  1. `load_script` (rights gate and clean check).
  2. `check_provider_keys`.
  3. Build the tracker with `limit = channel.spending_limit_usd`.
  4. `estimate_planning_cost`, then `tracker.ensure_room` → `BudgetExceededError` **before any call**.
  5. `detect_verses` (Arabic only).
  6. `segment_script`.
  7. `candidate_keys`.
  8. `assign_ideas`.
  9. Set `scene.match` with `match_idea` (library → bundled-generic → new).
  10. `vocalize_scene` (Arabic only).
  11. `split_into_parts`.
  12. Per-part `estimate_part_cost` with voice chars = `len(tts_text)`.
  13. Set `series.est_planning_cost_usd = tracker.spent` and return the series (not approved).
- `needs_approval(series: Series, channel: ChannelConfig) -> bool` is true when `channel.approval_list`, **or** any part's estimate is above `channel.spending_limit_usd`, **or** any uncertain verse, `tashkeel_rejected`, `unfinished_ending` or `over_max` flag is present.
- `render_approval_list(series: Series) -> rich.table.Table` has one row per scene with these columns: part, scene id, narration (truncated to 60 characters), idea key, match (reused / bundled / **new (paid)**), verse ref (with an `uncertain` marker) and flags. It is followed by per-part estimate rows and a series total. Pure rendering, no I/O.
- CLI:
  - `narratoflow plan SCRIPT --channel CHANNEL_TOML [--out series.json]`:
    - runs `plan_series` and writes `series.json`;
    - prints the approval list if `needs_approval` is true;
    - otherwise auto-approves (`approved = True`) and prints a one-line summary.
  - `narratoflow approve SERIES_JSON` loads the series, prints the list, asks with `typer.confirm`, sets `approved = True` and saves. It has `--yes` for non-interactive use.
  - Errors surface through `main()` as before: `ScriptBlockedError`, `BudgetExceededError`, `SegmentationError`, `PlanningError`, `ProviderError` and `VerseCorpusError` all exit 1 with no traceback.

- [ ] **Step 1: Write the failing tests** (all fakes, fixture corpus, tmp data dir)

```python
def test_plan_reference_script_1_end_to_end(tmp_path, channel_file, monkeypatch) -> None:
    # clean_ar.txt with rights="own"; FakeLLM scripted for segments + ideas + vocalization
    series = plan_series(script, channel, settings, llm=fake, embedder=None, corpus=fixture_corpus, prices=prices)
    verse_scenes = [s for p in series.parts for s in p.scenes if s.kind is SceneKind.VERSE]
    assert verse_scenes and verse_scenes[0].verse.surah == 31
    assert "".join(...) == ...           # verbatim across all scenes
    assert all(s.idea_key for p in series.parts for s in p.scenes if s.kind is SceneKind.NARRATION)

def test_budget_below_planning_cost_blocks_before_llm(...) -> None:   # Review Focus 4: FakeLLM.calls == []
def test_in_copyright_blocked_before_llm(...) -> None: ...
def test_needs_approval_forced_by_limit(...) -> None: ...            # approval_list=False but part estimate > limit -> True
def test_cli_plan_writes_series_and_prints_list(...) -> None: ...    # patch plan_series to return a fixed Series
def test_cli_approve_yes(...) -> None: ...
def test_cli_budget_error_exit_1(...) -> None: ...
```

- [ ] **Steps 2–5:** Red, implement, run the gate, then update the README Quick start (`plan`, `approve`, `fetch_quran.py`), and commit `feat: plan and approve commands with spending-limit gate`.

---

## Done when

- The gate is green in CI.
- `narratoflow plan` on reference script 1 (rights `own`) with fakes produces a series where:
  - verse 31:33 is detected and shown in Uthmani text;
  - the narration is verbatim;
  - every narration scene has a valid idea key;
  - the approval list shows reuse vs. paid.
- An in-copyright script and a too-low spending limit both stop before any LLM call.
- Migration 0002 upgrades a Plan-1 library in place, and the CHECK constraints reject bad status and origin values.
