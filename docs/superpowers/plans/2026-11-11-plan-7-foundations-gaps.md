# Plan 7: Foundations Gaps (icons, orchestration, config) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three largest gaps left in the old backlog:
- **#14:** ship a licence-tracked pack of 210 bundled line icons, and let the planner and the creator find them;
- **#4:** settle pipeline orchestration by deleting the stub and making the staged CLI safe to re-run from the top;
- **#3:** give every setting one home and turn typos in config into errors.

**Architecture:**
- **Icons enter the existing path as PNG line art.**
  - `scripts/fetch_icons.py` downloads one pinned Tabler Icons npm tarball and checks its sha256. It reads each curated icon's path geometry from the tarball's `tabler-nodes-outline.json` and rasterizes it to a 512x512 grayscale PNG with fixed constants. It refuses any icon that fails `check_density` or yields no strokes, then writes `assets/icons/<object>.png` and `assets/icons/index.toml` (with a per-file sha256 and an English description).
  - The PNGs are committed, as the fonts are in Plan 5. Runtime code never reads SVG and gains no dependency. From there each icon goes through the same `binarize` → `check_density` → `extract_strokes` path as every sketch (spec §4.3). It is copied into the channel library on first use and passes the vision/outlier check (already built in `sketching/generate.py::_add_bundled`).
- **Search fits the real lookup.** The planner chooses keys only from the candidate catalog (`library/candidates.py::candidate_keys`): the whole catalog up to 300 entries, and the top 5 by multilingual embedding above that (spec §2.2). Bundled icons are merged into that catalog, so the LLM can pick `lightbulb|generic` for an Arabic scene, and matching stays an exact key lookup (spec §4.2). The creator gets a plain keyword search, `narratoflow library icons WORDS`, for choosing a key for `sketch resolve --swap`. No `visuals/library.py` is created.
- **No orchestrator.** Each stage of the staged CLI already persists its progress, and a re-run spends nothing on work that is done. The one exception is `plan`, which overwrites `series.json` and pays again. It now refuses to overwrite unless given `--replace`. The `pipeline/orchestrator.py` stub is deleted.
- **Config has one home per kind of setting.**
  - Secrets, machine paths and the log level are in `Settings` (environment or `.env`).
  - Channel choices are in the channel TOML (`ChannelConfig`, with its defaults in code).
  - Per-run choices are CLI options.
  - `configs/default.toml` is never loaded and contradicts the code, so it is deleted.
  - An unknown key in a channel TOML is an error, and so is an unknown `NARRATOFLOW_*` variable.

**Tech Stack:** Python 3.12, pydantic 2, pydantic-settings, opencv-python-headless, scikit-image (all already deps). One new **dev-group** dependency: `svgelements` (MIT, pure Python), which only `scripts/fetch_icons.py` and its test use. The icons come from Tabler Icons 3.49.0 (MIT).

**Spec:** `docs/superpowers/specs/2026-11-10-sketch-library-pipeline-design.md`. This plan implements:
- §3 step 2 (the bundled icon library as the second lookup; generic keys only, for culture-agnostic ideas; copied into the channel library on first use; the same marker re-render and outlier check);
- §2.2 (the two-tier candidate lookup now covers bundled keys) and §4.2 (exact matching is unchanged);
- §4.3 (every icon becomes strokes through the same line-art path);
- §6.2 (every icon records its source, licence and sha256; missing provenance is a FAIL, which Plan 6's `provenance_checks` already enforces);
- §3 closing line ("each stage is cached by its inputs") as the reason for closing #4;
- §7.2 (secrets in env-only `Settings`; channel settings in Pydantic-validated TOML) as the frame for #3.

---

## Decisions for the owner

The spec leaves each of these to the owner. The plan proceeds with the recommended default.

1. **The icon pack.** Default: **Tabler Icons 3.49.0** (MIT), from the npm tarball `https://registry.npmjs.org/@tabler/icons/-/icons-3.49.0.tgz` (published 2026-10-05; sha256 `aa3042907eb0bdb079fd70733e2d75f785683d4726279128a0ca2343099f7074`, checked 2026-11-11). It has 5,184 outline icons, all 24-unit stroke drawings in one style, and the tarball ships `LICENSE` (MIT, "Copyright (c) 2020-2026 Paweł Kuna"). The alternative is Lucide (ISC), which has fewer icons. Tabler's GitHub releases attach no assets, so the npm tarball is the pinned artifact.
2. **The 210-icon selection** (the list in Task 5). It leaves out:
   - every person, face and head (`user`, `mood-*`, `skull`, `ghost`, `mask`);
   - animals drawn with a head (the vision check FAILs "any face or head, human or animal");
   - religious buildings and symbols, currencies, country flags, zodiac signs and brands.

   Some kept icons could read as culture-specific: `moon` (Tabler's moon is a crescent), `star`, `crown`, `sword`, `bow`, `axe` and `candle`. Default: **ship all 210**. Striking any of them is one line in the curation list.
3. **The planner sees the bundled icons inside the 300-key catalog.** This has two costs:
   - every narration scene's idea call carries about 210 more candidate lines (roughly 2,500 input tokens; the implementer reports the exact `estimate_tokens(render_candidates(...))` figure in the PR);
   - the embedding tier starts once the channel library has more than 90 keys, not 300.

   The alternative is to offer only each scene's top 5 icons by embedding, which needs an embedder from the first video. Default: **merge the icons into the catalog and keep the spec's 300 rule.**
4. **`plan --replace` discards the old series.** That includes its approval, attached sketch ids and recorded per-part spend, so each part's spending limit (§5.2) starts again from zero. The alternative is to carry the old per-part spend over when part indexes match. Default: **discard it, and say so in the confirmation message.**

---

## Global Constraints

- Python 3.12 and uv. The gate, identical locally and in CI: `uv lock --check && uv run ruff check --no-cache && uv run ruff format --check && uv run mypy && uv run coverage run -m pytest -o addopts="" -q && uv run coverage report`. Ruff, mypy strict on `src` and `tests`, pytest with branch coverage.
- Sources are pure ASCII. Write non-ASCII with `\u` escapes (for example the em dash `—` in `render_candidates`).
- **Tests never use the network.** `scripts/fetch_icons.py` is run once by hand (it needs network), and its output is committed. Its test builds a fake tarball in `tmp_path`.
- `NarratoFlowError` is caught only in `cli.main`. Secrets are never logged and never appear in an error message.
- TDD: each task's new test fails first, for the reason given in its step 2. A *characterization* test that pins behaviour which already exists is marked so, and is expected to pass at once.
- Tests check what a user sees: CLI output and exit codes, files written, prompts sent to the fake LLM, match results. They do not check private helpers.
- **Never spend without the existing gates.** Nothing in this plan adds a paid call. `plan` must still refuse before any LLM or embedding call (rights, keys, budget), and the new overwrite refusal comes before all of them.
- **Icon pack constants** (Task 5):
  - URL `https://registry.npmjs.org/@tabler/icons/-/icons-3.49.0.tgz`, sha256 `aa3042907eb0bdb079fd70733e2d75f785683d4726279128a0ca2343099f7074`;
  - geometry member `package/tabler-nodes-outline.json` (`{name: [["path", {"d": ...}], ...]}`), licence member `package/LICENSE`;
  - per-icon `source` = `<URL>#package/icons/outline/<tabler-name>.svg` (that member must exist in the tarball);
  - licence id `MIT`;
  - raster: `SIZE_PX = 512`, `SCALE_PX = 20` per SVG unit, `OFFSET_PX = 16`, `STROKE_PX = 8`, black ink on white, 8-bit grayscale PNG.
- **Icon keys** are `<object>|generic`. The object is the Tabler name unless the curation list renames it. Every key and description passes `planning.ideas.validate_idea`, and every description is at most 15 words of English.
- **Where each setting lives** (Task 1 writes this table into README):

  | Kind | Lives in | Examples |
  |---|---|---|
  | Secrets, machine paths, log level | environment or `.env` (`Settings`, `NARRATOFLOW_*`) | `*_API_KEY`, `DATA_DIR`, `FFMPEG`, `LOG_LEVEL`, `OLLAMA_URL` |
  | Channel choices | channel TOML (`ChannelConfig`; defaults in code) | style, voice, llm, embed, vision, `spending_limit_usd`, `approval_list`, part lengths, `riwaya`, `diacritizer` |
  | Per-run choices | CLI options | `--part`, `--out`, `--workers`, `--yes`, `--replace` |
  | Prices / platform rules | `data/prices.toml` / `data/platform_rules.toml`, each overridable in `NARRATOFLOW_DATA_DIR` | |
  | Output contract | code constants pinned by Plans 4-6 | 1080x1920, 30 fps, -14 LUFS |

## Review Focus

1. **A typo in the channel TOML silently drops a gate.** Case: `spending_limt_usd = 2` (typo). Today it is ignored, so there is no spending limit. Expected: `ChannelConfigError` naming `spending_limt_usd`, before anything runs. → Task 1 `test_typo_in_spending_limit_is_an_error`.
2. **A misspelled secret variable.** Case: `NARRATOFLOW_LLM_APIKEY=sk-SECRET` in the environment or in `.env`. Expected: `SettingsError` naming the variable, with `sk-SECRET` nowhere in the message or its cause. Keys without the `NARRATOFLOW_` prefix in a shared `.env` stay ignored. → Task 2 tests.
3. **Re-running the whole pipeline from the top after a crash.** Case: `series.json` exists (approved, sketches attached, spend recorded) and the user runs `narratoflow plan` again. Expected: exit 1 with "already exists; nothing was spent" and the file unchanged. No LLM call is made. → Task 3 `test_plan_refuses_to_overwrite_a_series`.
4. **The channel already owns an icon's key.** Case: the library has a paid `eye|generic` with description "an eye drawn in ink", and the pack has `eye|generic`. Expected: the planner prompt lists `eye|generic` once, with the channel's description, and matching gives `reused`, never `bundled`. → Task 6 `test_channel_entry_wins_in_the_catalog`.
5. **Icons push the catalog over 300.** Case: an empty library and a pack of 301 icons. Expected: the planning estimate includes embeddings (`require_prices(..., embed=True)`), and the injected embedder is called. A missing embedding price refuses before any spend. → Task 6 `test_bundled_icons_count_toward_the_embed_threshold`.

---

## File Structure

```
configs/default.toml                 # DELETE (never loaded; contradicts ChannelConfig)
src/narratoflow/
  channels/config.py                 # MODIFY: extra="forbid" on every channel model
  settings.py                        # MODIFY: extra="forbid"; unknown NARRATOFLOW_* env names rejected in get_settings
  pipeline/orchestrator.py           # DELETE (stub; #4 closed as superseded)
  pipeline/__init__.py               # MODIFY: docstring "Content-addressed stage cache."
  cli.py                             # MODIFY: plan --replace + overwrite refusal; `library icons`
  visuals/bundled.py                 # MODIFY: description + sha256 fields, catalog(), search(), __len__
  library/candidates.py              # MODIFY: merge_catalog; candidate_keys(extra=...)
  planning/pipeline.py               # MODIFY: merged catalog for estimate, prices and candidates
  planning/prompts.py                # MODIFY: IDEAS_SYSTEM names the bundled icons
scripts/fetch_icons.py               # NEW: pinned tarball -> PNGs + index (curation list inside)
assets/icons/*.png                   # NEW (generated, committed): 210 icons
assets/icons/index.toml              # REGENERATED by scripts/fetch_icons.py
assets/icons/LICENSE-tabler.txt      # NEW: package/LICENSE from the tarball
assets/icons/README.md               # MODIFY: generated; do not edit by hand
assets/LICENSES.md                   # MODIFY: Tabler row
pyproject.toml / uv.lock             # MODIFY: dev group svgelements; mypy override svgelements.*
README.md                            # MODIFY: settings table, re-running section, library icons, layout
.env.example                         # MODIFY: unknown NARRATOFLOW_* names are errors
tests/test_channels.py, tests/test_settings.py, tests/test_cli.py
tests/visuals/test_bundled.py, tests/library/test_bundled_icons.py, tests/library/test_candidates.py
tests/planning/conftest.py           # NEW: autouse empty pack, as tests/sketching/conftest.py does
tests/planning/test_pipeline.py, tests/audio/test_build.py
```

---

### Task 1: One home per setting; channel TOML rejects unknown keys

**Files:**
- Modify `src/narratoflow/channels/config.py`.
- Delete `configs/default.toml` (and the now-empty `configs/` directory).
- Modify `README.md`:
  - add a "Where settings live" section with the Global Constraints table;
  - in Layout, replace the `configs/` line with nothing.
- Tests: `tests/test_channels.py`.

**Interfaces — Produces:**
- `StyleConfig`, `VoiceConfig`, `LlmConfig`, `EmbedConfig`, `VisionConfig` and `ChannelConfig` each get `model_config = ConfigDict(extra="forbid")`.
- `load_channel` is unchanged. Pydantic's message already names the location (`voice.fallbak`) and says "Extra inputs are not permitted", inside `ChannelConfigError("Invalid channel config <path>: ...")`.

Why delete rather than load `configs/default.toml`: every key in it is one of three things.
- **A channel field with a different value from the code:**
  - `outputs = ["16:9"]` against v1's 9:16 only;
  - `tts.engine`/`fallback = "piper"` against `VoiceConfig.fallback = ["edge"]`;
  - `tashkeel.engine = "mishkal"` against `diacritizer = "llm"`;
  - `llm` against the channel's own `[llm]`.
- **A CLI option:** `render.workers = 4` against `--workers`, whose default is one per core.
- **A pinned constant** (`fps = 30`, `loudness_lufs = -14`), or **dead** (`visuals.default_source`, `seed`).

Loading it would add a third, conflicting place.

- [ ] **Step 1: Write the failing tests** in `tests/test_channels.py`. Each appends text to the `channel_file` fixture copy:
  - `test_typo_in_spending_limit_is_an_error`: `spending_limt_usd = 2` at top level gives `ChannelConfigError` whose message contains `spending_limt_usd` and `Extra inputs are not permitted`;
  - `test_unknown_nested_key_is_an_error`, parametrized over `("style", "line_weigth_px = 3")`, `("voice", "fallbak = [\"edge\"]")`, `("llm", "efort = \"low\"")`, `("embed", "modle = \"x\"")`, `("vision", "provder = \"none\"")`. Each puts the key in its table and expects a message containing `"<table>.<key>"`;
  - `test_fixture_channel_still_loads` (`tests/fixtures/channels/history.toml` loads unchanged).
- [ ] **Step 2: Run** `uv run pytest tests/test_channels.py -v`. Expected: the two typo tests FAIL (no error raised); the fixture test passes.
- [ ] **Step 3: Implement** the six `model_config` lines, delete `configs/default.toml`, and edit README.
- [ ] **Step 4: Run the full gate.** Expected: green. If any test or fixture built a channel with a now-forbidden key, fix the fixture, not the rule.
- [ ] **Step 5: Commit** `feat: channel config rejects unknown keys; drop unused configs/default.toml`.

### Task 2: Unknown `NARRATOFLOW_*` settings are errors

**Files:** Modify `src/narratoflow/settings.py` and `.env.example` (add the header comment "Unknown NARRATOFLOW_* names are an error; other names are ignored."). Tests: `tests/test_settings.py`.

**Interfaces — Produces:**
- `Settings.model_config` gets `extra="forbid"` (instead of `"ignore"`). With `env_prefix="NARRATOFLOW_"`, pydantic-settings rejects unknown *prefixed* keys in `.env` and ignores keys without the prefix. The tests below pin both behaviours. `hide_input_in_errors=True` stays.
- `get_settings()`, before constructing `Settings()`, scans `os.environ`. Any name that starts with `NARRATOFLOW_` (case-insensitive) and is not `"NARRATOFLOW_" + field.upper()` for a field in `Settings.model_fields` raises `SettingsError("unknown NarratoFlow setting(s) in the environment: <NAMES, sorted, comma-separated>; known: <all known names, sorted>")`. Only names appear in the message, never values.
- The existing wrapper still turns a `ValidationError` (including the `.env` extra-forbidden one) into `SettingsError("invalid NarratoFlow settings (environment or .env):\n...")`.
- If the installed pydantic-settings turns out to reject unprefixed `.env` keys under `forbid` (`test_foreign_keys_in_dotenv_are_ignored` fails), keep `extra="ignore"`. Instead, check the names in `.env` the same way as the environment: read them with `dotenv.dotenv_values(".env")` and add `python-dotenv` as a direct dependency. The behaviour the tests pin must not change.

- [ ] **Step 1: Write the failing tests**:
  - replace `test_unknown_key_in_dotenv_is_ignored` with `test_unknown_narratoflow_key_in_dotenv_is_an_error`: `.env` has `NARRATOFLOW_IMAGE_APIKEY=sk-SECRET123`. Expected: `SettingsError`, with `narratoflow_image_apikey` in `str(exc).lower()`, and `sk-SECRET123` absent from `str(exc)` and `str(exc.__cause__)`;
  - `test_unknown_narratoflow_env_var_is_an_error`: `monkeypatch.setenv("NARRATOFLOW_LLM_APIKEY", "sk-SECRET456")`. Expected: `SettingsError` whose message contains `NARRATOFLOW_LLM_APIKEY` and `NARRATOFLOW_LLM_API_KEY` (in the known list), and not `sk-SECRET456`;
  - `test_foreign_keys_in_dotenv_are_ignored`: `.env` has `OPENAI_API_KEY=x` and `NARRATOFLOW_LOG_LEVEL=DEBUG`. Expected: loads, with `log_level == "DEBUG"`;
  - `test_environment_beats_dotenv`: `.env` has `NARRATOFLOW_LOG_LEVEL=DEBUG` and the environment has `ERROR`. Expected: `"ERROR"` (characterization: passes already).
- [ ] **Step 2: Run** `uv run pytest tests/test_settings.py -v`. Expected: the two "is an error" tests FAIL (settings load); the other two pass.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full gate.** Expected: green. The autouse `_isolate_env` fixture already clears `NARRATOFLOW_*`, so no other test should leak a variable.
- [ ] **Step 5: Commit** `feat: reject unknown NARRATOFLOW_* settings without echoing values`.

### Task 3: Close #4: delete the orchestrator stub; `plan` never overwrites a series

The premise of #4 was a driver that runs stages over a `Project`. The staged CLI already meets each acceptance criterion:
- **Per-input caching.** `audio/narrate.py` caches each scene's TTS in `StageCache` under `cache_key("tts", ADAPTER_VERSION, provider/model/voice/text)`. Sketches are reused through the library. Re-runs keep attached sketches (`tests/sketching/test_generate.py::test_rerun_keeps_an_attached_sketch_and_its_match`).
- **Resumable after interruption.** `sketch generate`, `sketch resolve`, `audio build`, `review` and `render` save their progress on error, and a re-run spends nothing on finished work (`tests/audio/test_build.py::test_spent_written_back_and_cache_rerun_spends_zero`).
- **Fakes.** Tests run with fake engines throughout.

A single `run` command could not cross the approval list (§5.1) or the quarantine decisions (§6.3: regenerate, swap or text card, chosen by the creator) without asking. So it would stop exactly where the staged commands stop, and add a second way to do the same thing. One real gap remains: `plan` is the only stage whose re-run pays again and throws progress away. A user who re-runs the pipeline from the top after a crash loses the approval, the attached sketches and the recorded per-part spend (which the spending limit counts across stages). So `plan` now refuses to overwrite.

**Files:**
- Delete `src/narratoflow/pipeline/orchestrator.py`.
- Modify `src/narratoflow/pipeline/__init__.py` (docstring `"""Content-addressed stage cache."""`) and `src/narratoflow/cli.py` (`plan`).
- Modify `README.md`:
  - in Layout: `pipeline/ content-addressed stage cache`;
  - a new "Re-running after an error" section with one line per command: `plan` refuses an existing series (`--replace` to re-plan); `approve` is idempotent; `sketch generate` keeps attached sketches and pays only for missing ones; `audio build` reuses each cached scene; `render` re-renders the selected parts deterministically at no cost (pass `--part` for the unfinished ones it lists); `review` asks with an estimate each time; `publish` is free.
- Tests: `tests/test_cli.py`, `tests/audio/test_build.py`.

**Interfaces — Produces:**
- `plan(..., replace: Annotated[bool, typer.Option("--replace", help="Re-plan over an existing series file (discards its approval, sketches and recorded spend).")] = False)`.
- Right after the existing "output directory does not exist" check, and before `plan_series`: when `target.exists() and not replace`, raise `PlanningError(f"{target} already exists; nothing was spent. Continue with: narratoflow approve {target} (or the next stage), or re-plan with --replace, which discards its approval, attached sketches and recorded spend")`.

- [ ] **Step 1: Write the failing tests**:
  - `tests/test_cli.py::test_plan_refuses_to_overwrite_a_series`:
    - write any valid `Series` to `tmp_path / "series.json"` and remember its bytes;
    - monkeypatch `cli.plan_series` to raise `AssertionError` if called;
    - run `run_main(monkeypatch, "plan", <script>, "--channel", <channel_file>)`;
    - assert exit code 1, `"already exists; nothing was spent"` and `"--replace"` in stderr, and the file bytes unchanged;
  - `tests/test_cli.py::test_plan_replace_overwrites`: same setup, with `cli.plan_series` faked as in the existing plan tests (line ~275) and `--replace`. Expected: exit 0, and the file now holds the fake's series;
  - `tests/audio/test_build.py::test_editing_one_scene_revoices_only_that_scene` (characterization):
    - a two-scene part (`narr("s1", "alpha beta")`, `narr("s2", "gamma delta")`) is built once;
    - then s2's `narration.tts_text` and `caption_text` are changed to `"gamma epsilon"` (s1 unchanged) and the part is rebuilt with a fresh `FakeVoice`;
    - assert `[r.text for r in again.requests] == ["gamma epsilon"]` and `rerun.spent_usd == cost("gamma epsilon")`.
- [ ] **Step 2: Run** `uv run pytest tests/test_cli.py -k "plan_refuses or plan_replace" tests/audio/test_build.py -k editing -v`. Expected: `test_plan_refuses_to_overwrite_a_series` FAILS (the fake is called); `test_plan_replace_overwrites` FAILS (no `--replace` option); the audio test passes.
- [ ] **Step 3: Implement**, delete the stub, and update README.
- [ ] **Step 4: Run the full gate.** Expected: green. Existing plan tests that write over an existing `series.json` must pass `--replace`; nothing imports `pipeline.orchestrator`.
- [ ] **Step 5: Commit** `feat: plan refuses to overwrite a series; remove orchestrator stub (#4 superseded)`.

### Task 4: Icon index records a description and a sha256; keyword search

**Files:** Modify `src/narratoflow/visuals/bundled.py`. Tests: `tests/visuals/test_bundled.py` (its `_entry` helper gains the two new fields), `tests/library/test_matching.py` and `tests/sketching/test_generate.py`, `tests/planning/test_pipeline.py` (wherever they build a `BundledIcon` or an index, add the fields).

**Interfaces — Produces:**
- `BundledIcon(BaseModel)`: `key: str; file: Path; license: str; source: str; description: str; sha256: str`.
- The index entry fields are exactly `{"file", "description", "license", "source", "sha256"}`, all required and non-blank.
  - `description` has at most `MAX_DESCRIPTION_WORDS` (15, from `planning/ideas.py`) words. Import the constant; do not copy it.
  - The file's sha256 must equal `sha256`, otherwise `ChannelConfigError("bundled icon index entry '<key>': sha256 mismatch for <file>")`.
  - The `.png` rule and `confined_file` are unchanged.
- `BundledIcons.__len__(self) -> int`.
- `BundledIcons.catalog(self) -> list[tuple[str, str]]`: `(key, description)` sorted by key, in the same shape as `SketchLibrary.idea_catalog`.
- `BundledIcons.search(self, words: Sequence[str]) -> list[BundledIcon]`: icons, sorted by key, for which every word (lowercased, stripped; blank words dropped) is a substring of `f"{key} {description}".lower()`. No words returns every icon.

- [ ] **Step 1: Write the failing tests** in `tests/visuals/test_bundled.py`:
  - `test_entry_needs_description_and_sha256`: the parametrized violations gain a missing/blank `description`, a missing `sha256`, and a 16-word description;
  - `test_sha256_mismatch_is_rejected`: the file is edited after the index was written, and the message names the key and "sha256 mismatch";
  - `test_catalog_is_sorted_key_description_pairs`;
  - `test_search_matches_every_word_in_key_or_description`: icons `lightbulb|generic` "a lightbulb, an idea" and `bolt|generic` "a lightning bolt, energy". `search(["idea"])` gives the lightbulb; `search(["LIGHT"])` gives both; `search(["light", "energy"])` gives the bolt; `search([])` gives both; `search(["zzz"])` gives `[]`.
- [ ] **Step 2: Run** `uv run pytest tests/visuals/test_bundled.py -v`. Expected: FAIL (unknown fields `description`/`sha256`; no `catalog`/`search`).
- [ ] **Step 3: Implement.** Update the comment header of the (still empty) `assets/icons/index.toml` to list the five fields.
- [ ] **Step 4: Run the full gate.** Expected: green.
- [ ] **Step 5: Commit** `feat: bundled icon index records description and sha256; keyword search`.

### Task 5: The Tabler pack: fetch script, 210 committed icons, licence

**Files:**
- Create `scripts/fetch_icons.py`, `assets/icons/LICENSE-tabler.txt` and `assets/icons/*.png` (generated).
- Regenerate `assets/icons/index.toml`.
- Modify `assets/icons/README.md` ("Generated by `uv run python scripts/fetch_icons.py`; do not edit by hand; curation list in the script") and `assets/LICENSES.md`:
  - a row `| Tabler Icons 3.49.0 (210 outline icons, rasterized) | assets/icons/ | https://registry.npmjs.org/@tabler/icons/-/icons-3.49.0.tgz | MIT (LICENSE-tabler.txt) | Copyright (c) 2020-2026 Paweł Kuna |`;
  - the re-fetch line in the footer.
- Modify `pyproject.toml`: run `uv add --dev "svgelements>=1.9"`, which also updates `uv.lock`, and add `svgelements.*` to the mypy `ignore_missing_imports` list.
- Tests: `tests/library/test_bundled_icons.py` (shipped pack, wheel) and `tests/visuals/test_fetch_icons.py` (the script, loaded the way `tests/render/test_assets.py::_script` loads `fetch_fonts`).

**Interfaces — Produces:**
- `scripts/fetch_icons.py` mirrors `scripts/fetch_fonts.py`:
  - `URL`, `SHA256` (Global Constraints);
  - `@dataclass(frozen=True) class IconSpec: tabler: str; obj: str; description: str`;
  - `ICONS: tuple[IconSpec, ...]`;
  - `rasterize(paths: Sequence[str]) -> bytes`: a PNG from SVG path `d` strings, using the Global Constraints raster constants. Each `svgelements.Path(d)` is split into subpaths. Each segment is sampled at `max(8, ceil(length_px / 4))` points, and the points are drawn with `cv2.polylines(..., isClosed=False, color=0, thickness=STROKE_PX, lineType=cv2.LINE_AA)` on a white `uint8` canvas. A closed subpath also draws back to its start;
  - `build_pack(data: bytes, out_dir: Path) -> str`, which:
    - verifies the tarball sha256;
    - opens it with `tarfile` and reads only the named members via `extractfile` (never `extractall`);
    - for each `IconSpec`, requires the name in the nodes JSON with every node `["path", {"d": ...}]` and no other attribute, and requires the `package/icons/outline/<tabler>.svg` member to exist;
    - rasterizes it;
    - requires `check_density(binarize(png), line_weight_px=3.0).ok` and `len(extract_strokes(binarize(png)).strokes) > 0`.

    Every failure is collected, and one `ValueError` lists all of them; nothing is written in that case. Otherwise it writes `<obj>.png` per icon and `LICENSE-tabler.txt`, and returns the index text: the comment header from Task 4, then entries in `ICONS` order with `file`, `description`, `license = "MIT"`, `source` and `sha256`;
  - `main()` downloads (`urllib.request.urlopen(URL, timeout=120)`, `# noqa: S310`), calls `build_pack` on `assets/icons/` and writes `index.toml`.
- The curation list. Each token is `tabler-name` or `tabler-name=key-object`, and the key is `<object>|generic`:

  ```
  book books notebook notes file-text=document files=documents news=newspaper script=scroll map
  map-pin compass globe world=earth hourglass clock calendar crown sword shield ship sailboat
  anchor flag key lock lock-open=open-lock door door-exit=exit-door bulb=lightbulb bolt flame
  droplet tree trees plant leaf seedling sun sunrise sunset moon star stars cloud
  cloud-rain=rain-cloud cloud-storm=storm-cloud mountain ripple=waves road route=path
  building-bridge=bridge scale=scales eye hand-stop=open-hand hand-finger=pointing-hand
  hand-grab=grabbing-hand heart heart-broken=broken-heart brain puzzle settings=gear tool tools
  hammer pencil ballpen=pen feather mail=envelope mail-opened=open-envelope message=speech-bubble
  bubble=thought-bubble bell camera phone device-mobile=mobile-phone device-tv=television truck
  car plane train bike steering-wheel box package container hanger coin coins cash=banknote
  credit-card=bank-card chart-bar=bar-chart chart-line=line-chart chart-pie=pie-chart
  trending-up=rising-trend trending-down=falling-trend target trophy medal award gift balloon
  home building tent fence ladder stairs window armchair lamp candle infinity atom dna microscope
  flask telescope prism rocket planet magnet battery plug wifi network affiliate=connected-nodes
  sitemap=hierarchy share link lifebuoy umbrella snowflake wind tornado volcano paw apple bread
  coffee bottle glass egg cup bowl campfire arrow-right=arrow arrow-left arrow-up arrow-down
  arrows-split=split-arrows arrows-join=joining-arrows refresh=cycle-arrows recycle
  question-mark exclamation-mark check=check-mark x=x-mark plus minus equal circle square
  triangle hexagon diamond spiral wave-sine=sine-wave eyeglass=eyeglasses binoculars
  search=magnifier ruler writing=pen-writing signature brush palette music microphone
  speakerphone=megaphone headphones player-play=play-button video movie=film-clapper
  photo=picture frame=picture-frame alarm stopwatch building-lighthouse=lighthouse wall
  barrier-block=roadblock wheel chess cards target-arrow=bullseye bow axe pick=pickaxe shovel
  bucket basket shopping-cart scale-outline=weighing-scale weight wheat cactus flower virus pill
  stethoscope first-aid-kit heartbeat
  ```

  All 210 names were checked against 3.49.0 on 2026-11-11: each exists, uses only plain `path` nodes, and has a key that passes `validate_idea`. The implementer writes each `description`. It is plain English naming the drawn object and, where it helps the planner, what it commonly stands for, for example `bulb=lightbulb` → "a lightbulb, an idea or insight", `scale=scales` → "a balance scale, justice or weighing deeds", `road` → "a road leading into the distance, a journey". It names no person, figure, brand or culture. If `build_pack` refuses an icon, remove it from the list and say so in the PR; the pack must keep at least 150.

- [ ] **Step 1: Write the failing tests**:
  - `tests/visuals/test_fetch_icons.py::test_build_pack_pins_sha_and_refuses_bad_icons`. A fake `.tgz` in `tmp_path` holds a nodes JSON, `package/LICENSE` and `package/icons/outline/*.svg` members, and the script's `SHA256`, `URL` and `ICONS` are monkeypatched to match it. Then:
    - a wrong sha raises and writes nothing;
    - an icon missing from the nodes JSON, one with a `fill` attribute, and a filled square drawn as many parallel lines (which fails density) are all listed in one `ValueError`;
    - a good icon (a circle `M3 12a9 9 0 1 0 18 0a9 9 0 1 0 -18 0`) writes a 512x512 grayscale PNG and an index entry whose sha256 matches the file;
  - `tests/library/test_bundled_icons.py`:
    - `test_shipped_pack_has_at_least_150_mit_icons`: `len >= 150`; every `license == "MIT"`; every `source` starts with `URL + "#package/icons/outline/"`; `assets/icons/LICENSE-tabler.txt` exists; `assets/LICENSES.md` contains `Tabler Icons 3.49.0`;
    - `test_every_shipped_icon_obeys_the_drawing_rules`: every key is generic, `validate_idea(IdeaOut(idea_key=key, description=desc, depicts_person=False, depicts_logo=False)) == []`, and no key object token is in the culture denylist `{"church", "mosque", "synagogue", "temple", "cathedral", "pagoda", "cross", "crescent", "menorah", "torii", "dollar", "euro", "pound", "yen", "rupee", "lira", "dirham", "riyal", "bitcoin", "brand", "zodiac", "currency"}` (tokens are the object split on `-`), and no `source` member name starts with `brand-`, `currency-` or `zodiac-`;
    - `test_every_shipped_icon_becomes_strokes`: for every icon, `check_density(binarize(png), line_weight_px=3.0).ok` and the strokes are non-empty;
    - `test_wheel_contains_the_icon_index`, extended: the wheel also has `narratoflow/assets/icons/lightbulb.png` and `narratoflow/assets/icons/LICENSE-tabler.txt`.
- [ ] **Step 2: Run** `uv run pytest tests/visuals/test_fetch_icons.py tests/library/test_bundled_icons.py -v`. Expected: FAIL (no script; the shipped index is empty).
- [ ] **Step 3: Implement** the script, then run `uv run python scripts/fetch_icons.py` once (network) and commit its output. Update `assets/LICENSES.md` and the README of `assets/icons/`.
- [ ] **Step 4: Run the full gate.** Expected: green. Note the run time of `test_every_shipped_icon_becomes_strokes` in the PR. If it exceeds 30 s, mark it `slow`, which keeps it in CI.
- [ ] **Step 5: Commit** `feat: bundle 210 Tabler (MIT) line icons via a pinned, sha256-checked fetch`.

### Task 6: The planner is offered the bundled icons

**Files:**
- Modify `src/narratoflow/library/candidates.py`, `src/narratoflow/planning/pipeline.py` and `src/narratoflow/planning/prompts.py`.
- Create `tests/planning/conftest.py`: an autouse fixture that monkeypatches `pipeline_module.default_bundled_icons` to `BundledIcons({})`, so the existing estimate and prompt pins keep meaning "empty pack". New tests that need icons override it.
- Tests: `tests/library/test_candidates.py`, `tests/planning/test_pipeline.py`.

**Interfaces — Consumes:** `BundledIcons.catalog()` (Task 4).

**Interfaces — Produces:**
- `merge_catalog(channel: Sequence[Entry], bundled: Sequence[Entry]) -> list[Entry]`. It returns one entry per key, sorted by key. A channel entry wins over a bundled one with the same key. If the channel's description is blank, the bundled description is used.
- `candidate_keys(library, style_version_id, scenes, embedder, *, extra: Sequence[Entry] = (), full_limit=300, top_k=5)`. The catalog is `merge_catalog(library.idea_catalog(style_version_id), extra)`; everything else is unchanged. Embeddings of bundled entries are cached in the same `IdeaEmbedding` table under the style version (that table has no foreign key to a sketch).
- `plan_series`:
  - `bundled = default_bundled_icons()` moves above the catalog line (it still loads before any spend);
  - `catalog = merge_catalog(library.idea_catalog(version), bundled.catalog())` feeds `estimate_planning`, `require_prices(..., embed=len(catalog) > CATALOG_EMBED_THRESHOLD)` and the embedder decision;
  - `_candidates` gains `extra: Sequence[Entry]` and passes it to `candidate_keys`.
- `IDEAS_SYSTEM`: replace the sentence "Candidate keys from the channel library are listed with each scene." with "Candidate keys from the channel library and from the bundled icons are listed with each scene. Bundled icon keys all end in |generic: use one only when the culture or period does not matter."

- [ ] **Step 1: Write the failing tests**:
  - `tests/library/test_candidates.py`:
    - `test_merge_catalog_channel_wins_and_fills_blank_description`;
    - `test_extra_entries_are_offered_and_counted`: a library with 2 entries, `extra` with 1, `full_limit=2`. Without an embedder: `PlanningError` naming 3 ideas. With `FakeEmbedder`: the extra entry can be ranked first;
  - `tests/planning/test_pipeline.py`. Override the autouse fixture with a two-icon pack written to `tmp_path` (`lightbulb|generic`, `eye|generic`, with real sha256s):
    - `test_bundled_icons_are_offered_to_the_planner`: the fake LLM's idea prompt contains `"lightbulb|generic — "` followed by the icon's description; the fake answers `lightbulb|generic`; the scene's `match == "bundled"`;
    - `test_channel_entry_wins_in_the_catalog` (Review Focus 4);
    - `test_bundled_icons_count_toward_the_embed_threshold` (Review Focus 5): a 301-icon pack (tiny PNGs, real sha256s) and an empty library. The injected `FakeEmbedder` is called. A `PriceTable` without the embed price raises the existing missing-price error before any LLM call;
    - `test_ideas_system_names_bundled_icons`: `"bundled icons" in IDEAS_SYSTEM` and `"|generic" in IDEAS_SYSTEM`.
- [ ] **Step 2: Run** `uv run pytest tests/library/test_candidates.py tests/planning/test_pipeline.py -v`. Expected: the new tests FAIL (no `merge_catalog`, no `extra`, icons absent from the prompt).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full gate.** Expected: green. Report the catalog token figure for Owner Decision 3 in the PR: `estimate_tokens(render_candidates(default_bundled_icons().catalog()))`.
- [ ] **Step 5: Commit** `feat: offer bundled icons to the planner through the candidate catalog`.

### Task 7: `narratoflow library icons` for the creator

**Files:** Modify `src/narratoflow/cli.py` and `README.md` (one Quick start line: `uv run narratoflow library icons light   # bundled icon keys to use with sketch resolve --swap`). Tests: `tests/test_cli.py`.

**Interfaces — Consumes:** `default_bundled_icons()`, `BundledIcons.search` (Task 4); `render_candidates` (`planning/ideas.py`).

**Interfaces — Produces:**
- `@library_app.command("icons") def library_icons(words: Annotated[list[str] | None, typer.Argument(help="Words that must all appear in the key or description.")] = None) -> None`:
  - prints `render_candidates([(i.key, i.description) for i in matches])` with `markup=False, highlight=False`;
  - with no match, prints `no bundled icon matches '<words joined by a space>'; propose a new key with sketch resolve --swap, or use --text-card` and exits 0;
  - needs no channel, no settings keys and no network.

- [ ] **Step 1: Write the failing tests** (monkeypatch `cli.default_bundled_icons` to the two-icon pack from Task 6):
  - `test_library_icons_lists_matches`: `library icons idea` exits 0 and its output contains `lightbulb|generic` but not `eye|generic`;
  - `test_library_icons_without_words_lists_all`;
  - `test_library_icons_no_match_hint`: the output contains `no bundled icon matches 'zzz'` and `--text-card`.
- [ ] **Step 2: Run** `uv run pytest tests/test_cli.py -k library_icons -v`. Expected: FAIL (no such command).
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the full gate.** Expected: green.
- [ ] **Step 5: Commit** `feat: narratoflow library icons keyword search`.

---

## Done when

- The gate is green in CI, with no network.
- `assets/icons/index.toml` lists at least 150 MIT icons, each with a sha256-verified PNG, a description and a source pointing into the pinned tarball. `assets/LICENSES.md` and `LICENSE-tabler.txt` record the pack.
- With an empty channel library, an idea prompt lists the bundled icons, and a planner answer of `lightbulb|generic` is matched `bundled`. On `sketch generate` it is copied into the channel library through the density and vision checks, as before.
- `narratoflow library icons WORDS` finds icon keys.
- `configs/default.toml` and `pipeline/orchestrator.py` are gone. README says where every setting lives and what re-running each command does.
- A typo in a channel TOML key, or an unknown `NARRATOFLOW_*` name, stops the run with a message naming it, never its value. `plan` never overwrites a series without `--replace`.
- Issue notes for the owner, who closes the issues:
  - **#14:** done. `visuals/library.py` was not created: the search is the planner's candidate catalog plus `library icons`, and matching stays exact per spec §4.2. The text card is the existing `sketch resolve --text-card`.
  - **#4:** superseded by the staged CLI, for the reasons in Task 3.
  - **#3:** done as narrowed by spec §7.2. There is no overlay: env beats `.env`, and the channel TOML has its defaults in code.
