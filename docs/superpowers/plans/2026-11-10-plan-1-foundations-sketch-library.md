# Plan 1: Foundations & Sketch Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the tooling, add typed settings and secrets, channel configs with automatic style versioning, a persistent per-channel sketch library (SQLite + Alembic) with exact idea-key matching, and the input quality gate. Expose all of it through the CLI.

**Architecture:**
- **Channel configs** are TOML files validated by Pydantic. The style fields are hashed into a *fingerprint*; the library stores one `StyleVersion` row per fingerprint, so editing a channel's style automatically starts a new version.
- **The library** is SQLAlchemy 2.0 on SQLite (WAL mode). Its schema is managed by Alembic migrations that run on open. Sketch files are stored content-addressed beside the database.
- Repositories return Pydantic DTOs, never ORM objects.

**Tech Stack:** Python 3.12, uv, pydantic 2 + pydantic-settings, SQLAlchemy 2.0, Alembic, Typer, Ruff, mypy (strict), pytest + coverage.

**Spec:** `docs/superpowers/specs/2026-11-10-sketch-library-pipeline-design.md`. Sections implemented here: §2.1, §2.2, §2.3, §3 steps 1a–2, §4.2, §4.5 (selection only), §4.6, §7.1–§7.6.

---

## Roadmap (this plan is 1 of 6)

| Plan | Delivers | Depends on |
|---|---|---|
| **1. Foundations & Sketch Library** (this) | Settings and secrets, channels and style versions, library DB, idea matching, input gate, CLI | none |
| 2. Script Planning | Verse detection (Tanzil) and protected spans, diacritization with the skeleton guard, LLM adapter (structured JSON), scene and idea planner, series splitting, script-source declaration, approval list, cost table, spending limit | 1 |
| 3. Sketch Generation | Image provider adapter(s), vision checks (likeness, logo, outlier), quarantine flow, centerline strokes, marker re-render | 1, 2 |
| 4. Audio | Voice adapter(s) with fallback, word timings and alignment, licensed Hafs reciter audio, mixing (no SFX or music under verses, loudnorm) | 2 |
| 5. Render | 9:16 layout and safe zones, stroke reveal, free variations, Arabic and English captions, on-screen hooks and Part N, FFmpeg encoder | 3, 4 |
| 6. Publish | Publishing-check framework and rules file, provenance check, production report and series summary, export blocking | 5 |

---

## Global Constraints

- Python pinned to **3.12**: `requires-python = ">=3.12,<3.13"` and `.python-version` = `3.12`.
- Dev tools only in `[dependency-groups] dev`. New runtime deps for this plan: `pydantic-settings`, `sqlalchemy>=2.0`, `alembic`.
- Ruff rule set: `E, F, I, B, UP, SIM, RUF, S, G, ASYNC`, with `"tests/**" = ["S101"]`. `docs` is excluded.
- mypy `strict = true` on `src` **and** `tests`.
- Gate order, the same locally and in CI: `ruff check --no-cache` → `ruff format --check` → `mypy` → `coverage run -m pytest` + `coverage report`.
- API keys come only from the environment, as `SecretStr`, and have **no defaults**. A missing key for a selected paid provider fails **before any work starts** (§7.2).
- Logging uses `logging.getLogger(__name__)` with lazy `%s` arguments and is configured once in the CLI entrypoint. Package code never calls `print`.
- No exception is swallowed. All project errors subclass `NarratoFlowError` (Task 2).
- Timestamps are timezone-aware UTC (`datetime.now(UTC)`).
- Idea keys have the form `object|context`, lowercase ASCII kebab-case. The generic context is the literal `generic` (§2.2).
- A sketch with status `failed` is **never** returned for reuse (§2.3, §6.3).
- Bundled icons match **only** keys whose context is `generic` (§3 step 2).
- Free providers that need no key: `ollama`, `local-diffusion`, `edge`, `piper`. Any other provider name is paid and requires a key.
- Tests never touch the network, the real data dir, or a local `.env`.

## Review Focus

1. **The LLM later emits an Arabic or spaced idea key** (`قلم|generic`, `Old Scroll|Roman`). Expected: `parse_idea_key` normalizes case and spaces but rejects non-ASCII with a clear `InvalidIdeaKeyError`. → Task 6 tests.
2. **Two runs open the library at once.** Expected: no "database is locked" crash on a normal write. The library uses WAL and a busy timeout. → Task 4 test asserts `journal_mode == "wal"`.
3. **The same image is stored twice** (re-run, or copying a sketch). Expected: dedupe to one blob, with no `IntegrityError` surfacing. → Task 5 tests.
4. **Pasted text with a BOM, CRLF line endings, or RTL/LTR marks** (common when copying from Word or Claude). Expected: the quality gate normalizes these and does not flag them. → Task 7 test.
5. **The channel's reference sketch is missing or moved.** Expected: a `ChannelConfigError` that names the missing path, not a raw `FileNotFoundError` traceback. → Task 3 test.

---

## File Structure

```
src/narratoflow/
  errors.py                 # NEW: error hierarchy
  settings.py               # NEW: Settings, get_settings (replaces config.py stub)
  logging_setup.py          # NEW: setup_logging(level)
  channels/__init__.py      # NEW
  channels/config.py        # NEW: ChannelConfig + nested models, load_channel, style_fingerprint
  channels/keys.py          # NEW: ProviderKind, required_key_env_vars, check_provider_keys
  library/__init__.py       # NEW
  library/db.py             # NEW: open_library(path) -> Engine, runs migrations, WAL
  library/orm.py            # NEW: SQLAlchemy models
  library/schemas.py        # NEW: Pydantic DTOs (SketchRecord, Provenance, enums)
  library/blobs.py          # NEW: BlobStore
  library/repository.py     # NEW: SketchLibrary
  library/keys.py           # NEW: IdeaKey, parse_idea_key
  library/matching.py       # NEW: MatchKind, MatchResult, match_idea, choose_variant
  library/migrations/       # NEW: Alembic env.py, script.py.mako, versions/0001_initial.py
  visuals/bundled.py        # NEW: BundledIcons (replaces visuals/library.py stub)
  text/quality.py           # NEW: check_input
  cli.py                    # MODIFY: channel / library / check-input commands
  config.py                 # DELETE (stub)
  visuals/library.py        # DELETE (stub)
assets/icons/index.toml     # NEW: bundled icon index (empty table to start)
.env.example                # NEW
tests/conftest.py           # NEW: env isolation, tmp data dir, settings cache clear
tests/__init__.py           # DELETE
tests/fixtures/...          # NEW: channel TOML + reference PNG + input texts
```

---

### Task 1: Tooling hardening (§7.1)

**Files:**
- Modify: `pyproject.toml`, `.python-version`, `.github/workflows/ci.yml`, `.gitignore`
- Create: `.env.example`
- Delete: `tests/__init__.py`, `src/narratoflow/config.py`, `src/narratoflow/visuals/library.py`

**Interfaces:**
- Produces: the gate commands that every later task runs.

- [ ] **Step 1: Update `pyproject.toml`**
  - `requires-python = ">=3.12,<3.13"`.
  - Ruff: `target-version = "py312"`, `select` adds `"S", "G", "ASYNC"`, and add `[tool.ruff.lint.per-file-ignores] "tests/**" = ["S101"]`.
  - mypy: `python_version = "3.12"`, `files = ["src", "tests"]`, remove `packages`. Add `[[tool.mypy.overrides]] module = ["cv2.*", "pyarabic.*", "skimage.*"] ignore_missing_imports = true`.
  - Add `[tool.coverage.run] branch = true, source = ["src/narratoflow"]` and `[tool.coverage.report] show_missing = true`.
  - Add `coverage>=7.6` to the dev group.
- [ ] **Step 2: Runtime deps.** Run `uv add pydantic-settings "sqlalchemy>=2.0" alembic`. Expected: `uv.lock` is updated.
- [ ] **Step 3: Pin files.** Set `.python-version` to `3.12`. Append `.env` and `.env.*` (with an exception `!.env.example`) to `.gitignore`.
- [ ] **Step 4: Create `.env.example`**

```
# Required only when the channel selects a paid provider of that kind.
NARRATOFLOW_IMAGE_API_KEY=
NARRATOFLOW_VOICE_API_KEY=
NARRATOFLOW_LLM_API_KEY=
# Optional
NARRATOFLOW_DATA_DIR=~/.narratoflow
NARRATOFLOW_LOG_LEVEL=INFO
```

- [ ] **Step 5: Update CI.**
  - `astral-sh/setup-uv@v6` with `with: version: "0.11.31"`.
  - `uv sync --locked --group dev`.
  - Steps in this order: `uv run ruff check --no-cache .`, `uv run ruff format --check .`, `uv run mypy`, `uv run coverage run -m pytest`, `uv run coverage report`.
- [ ] **Step 6: Delete the three stub files listed above.** Fix any test whose type annotations mypy now flags under `files = ["src","tests"]`.
- [ ] **Step 7: Run the gate locally.** Run:
  ```bash
  uv run ruff check --no-cache .
  uv run ruff format --check .
  uv run mypy
  uv run coverage run -m pytest
  uv run coverage report
  ```
  Expected: 0 failures, 100% pass.
- [ ] **Step 8: Commit.** `git commit -m "build: harden tooling per python-best-practices (py3.12, ruff S/G/ASYNC, strict mypy on tests, coverage, locked CI)"`

---

### Task 2: Errors, settings, logging

**Files:**
- Create: `src/narratoflow/errors.py`, `src/narratoflow/settings.py`, `src/narratoflow/logging_setup.py`, `tests/conftest.py`
- Test: `tests/test_settings.py`

**Interfaces:**
- Produces:
  - `errors.py`: `class NarratoFlowError(Exception)`. Subclasses: `MissingApiKeyError`, `ChannelConfigError`, `InvalidIdeaKeyError`, `LibraryError`, `CorruptInputError`. Later plans add `BudgetExceededError`, `PublishCheckFailedError` and `ProviderError` here.
  - `settings.py`: `class Settings(BaseSettings)` with `env_prefix="NARRATOFLOW_"`, `env_file=".env"`, and fields:
    - `data_dir: Path = Path("~/.narratoflow")`, expanded by a validator;
    - `log_level: Literal["DEBUG","INFO","WARNING","ERROR"] = "INFO"`;
    - `image_api_key: SecretStr | None = None`, `voice_api_key: SecretStr | None = None`, `llm_api_key: SecretStr | None = None`.

    The keys are optional at the type level. Requiring them per channel is Task 3's `check_provider_keys`. Also `get_settings() -> Settings` with `@lru_cache`.
  - `logging_setup.py`: `setup_logging(level: str) -> None`, using `basicConfig` with the format `"%(asctime)s %(levelname)s %(name)s %(message)s"`.

- [ ] **Step 1: Write `tests/conftest.py`.** It provides:
  - an autouse fixture that deletes every `NARRATOFLOW_*` env var through `monkeypatch`, then `chdir`s to `tmp_path` so no real `.env` is read;
  - `get_settings.cache_clear()` before and after each test;
  - a `data_dir` fixture that sets `NARRATOFLOW_DATA_DIR` to `tmp_path / "data"`.
- [ ] **Step 2: Write the failing tests**

```python
def test_defaults_without_env() -> None:
    s = get_settings()
    assert s.log_level == "INFO"
    assert s.image_api_key is None
    assert s.data_dir == Path("~/.narratoflow").expanduser()

def test_key_is_secret_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NARRATOFLOW_IMAGE_API_KEY", "sk-test-123")
    s = get_settings()
    assert s.image_api_key is not None
    assert s.image_api_key.get_secret_value() == "sk-test-123"
    assert "sk-test-123" not in repr(s)

def test_invalid_log_level_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NARRATOFLOW_LOG_LEVEL", "LOUD")
    with pytest.raises(ValidationError):
        get_settings()

def test_all_project_errors_share_base() -> None:
    for cls in (MissingApiKeyError, ChannelConfigError, InvalidIdeaKeyError, LibraryError, CorruptInputError):
        assert issubclass(cls, NarratoFlowError)
```

- [ ] **Step 3: Run.** `uv run pytest tests/test_settings.py -v`. Expected: FAIL with ImportError.
- [ ] **Step 4: Implement** `errors.py`, `settings.py` and `logging_setup.py` per the Interfaces block.
- [ ] **Step 5: Run.** Same command. Expected: PASS. Then run the full gate. Expected: green.
- [ ] **Step 6: Commit.** `git commit -m "feat: add error hierarchy, typed settings with SecretStr keys, logging setup"`

---

### Task 3: Channel config, style fingerprint, provider-key check

**Files:**
- Create: `src/narratoflow/channels/__init__.py`, `channels/config.py`, `channels/keys.py`, `tests/fixtures/channels/history.toml`, `tests/fixtures/channels/ref.png` (any small PNG, written by the test fixture through Pillow; don't commit a binary if one can be generated)
- Test: `tests/test_channels.py`

**Interfaces:**
- Consumes: `Settings`, `MissingApiKeyError`, `ChannelConfigError` (Task 2), and `Language`, `AspectRatio` (`narratoflow.models`).
- Produces:
  - `StyleConfig(BaseModel)`:
    - `reference_sketch: Path`;
    - `instructions: str` (stripped, `min_length=1`);
    - `image_provider: str`, `image_model: str`;
    - `line_weight_px: float = Field(3.0, gt=0, le=20)`;
    - `ink: str = Field("#111111", pattern=r"^#[0-9a-fA-F]{6}$")`.
  - `VoiceConfig(BaseModel)`: `provider: str`, `model: str = ""`, `voice_id: str`.
  - `LlmConfig(BaseModel)`: `provider: str`, `model: str`.
  - `ChannelConfig(BaseModel)`:
    - `name: str`, `slug: str` (pattern `^[a-z0-9]+(-[a-z0-9]+)*$`);
    - `language: Language`;
    - `style: StyleConfig`, `voice: VoiceConfig`, `llm: LlmConfig`;
    - `outputs: list[AspectRatio] = [AspectRatio.PORTRAIT]`;
    - `spending_limit_usd: Decimal | None = Field(None, gt=0)`;
    - `approval_list: bool = True`;
    - `background_music: bool = False`;
    - `part_min_s: int = 60`, `part_max_s: int = 90`, with a validator enforcing `part_min_s < part_max_s`;
    - `riwaya: str = "hafs"`.
  - `load_channel(path: Path) -> ChannelConfig`. It reads with `tomllib`, resolves `reference_sketch` to an absolute path relative to the TOML file's directory (`(path.parent / ref_path).resolve()`), and wraps `OSError`, `TOMLDecodeError` and `ValidationError` in `ChannelConfigError` with the path in the message (`raise … from e`).
  - `style_fingerprint(style: StyleConfig) -> str`. It returns the sha256 hex, first 16 characters, of the JSON (`sort_keys`) of `{instructions, image_provider, image_model, line_weight_px, ink, reference_sha256}`, where `reference_sha256` is the hash of the reference file's bytes.
  - `channels/keys.py`:
    - `class ProviderKind(StrEnum): IMAGE="image"; VOICE="voice"; LLM="llm"`;
    - `FREE_PROVIDERS: frozenset[str] = frozenset({"ollama","local-diffusion","edge","piper"})`;
    - `required_key_env_vars(channel: ChannelConfig) -> dict[ProviderKind, str]`, mapping each paid kind to its env var name (e.g. `NARRATOFLOW_IMAGE_API_KEY`);
    - `check_provider_keys(channel: ChannelConfig, settings: Settings) -> None`, which raises `MissingApiKeyError` listing **all** missing env var names in one message.

- [ ] **Step 1: Write the fixture TOML** with `slug = "history"`, `language = "ar"`, a paid image provider `"openai"`, voice provider `"edge"`, and LLM provider `"ollama"`. Then write the failing tests:

```python
def test_load_resolves_reference_relative(channel_file: Path) -> None:
    ch = load_channel(channel_file)
    assert ch.style.reference_sketch == (channel_file.parent / "ref.png").resolve()
    assert ch.outputs == [AspectRatio.PORTRAIT]
    assert ch.background_music is False and ch.riwaya == "hafs"

def test_fingerprint_changes_with_model_and_reference(channel_file: Path) -> None:
    ch = load_channel(channel_file)
    fp = style_fingerprint(ch.style)
    assert fp == style_fingerprint(ch.style)                     # stable
    other = ch.style.model_copy(update={"image_model": "other-model"})
    assert style_fingerprint(other) != fp
    ch.style.reference_sketch.write_bytes(b"different")
    assert style_fingerprint(ch.style) != fp

def test_missing_reference_names_path(channel_file: Path) -> None:      # Review Focus 5
    ch = load_channel(channel_file)
    ch.style.reference_sketch.unlink()
    with pytest.raises(ChannelConfigError, match="ref.png"):
        style_fingerprint(ch.style)

def test_invalid_toml_wrapped(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"; bad.write_text("slug = [")
    with pytest.raises(ChannelConfigError, match="bad.toml"):
        load_channel(bad)

def test_part_bounds_validated(channel_file: Path) -> None: ...  # part_min_s=90, part_max_s=60 -> ChannelConfigError

def test_only_paid_kinds_need_keys(channel_file: Path) -> None:
    ch = load_channel(channel_file)
    assert required_key_env_vars(ch) == {ProviderKind.IMAGE: "NARRATOFLOW_IMAGE_API_KEY"}

def test_missing_key_fails_before_work(channel_file: Path) -> None:
    with pytest.raises(MissingApiKeyError, match="NARRATOFLOW_IMAGE_API_KEY"):
        check_provider_keys(load_channel(channel_file), get_settings())

def test_present_key_passes(channel_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NARRATOFLOW_IMAGE_API_KEY", "x")
    check_provider_keys(load_channel(channel_file), get_settings())
```

  In `test_missing_reference_names_path`, `style_fingerprint` must raise `ChannelConfigError` (not `FileNotFoundError`) when the reference file can't be read.
- [ ] **Step 2: Run.** `uv run pytest tests/test_channels.py -v`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement** per the Interfaces block.
- [ ] **Step 4: Run.** Expected: PASS. Then run the full gate. Expected: green.
- [ ] **Step 5: Commit.** `git commit -m "feat: channel config with style fingerprint and fail-fast provider key check"`

---

### Task 4: Library database, ORM and first migration

**Files:**
- Create: `src/narratoflow/library/__init__.py`, `library/orm.py`, `library/schemas.py`, `library/db.py`, `library/migrations/env.py`, `library/migrations/script.py.mako`, `library/migrations/versions/0001_initial.py`
- Test: `tests/library/test_db.py`

**Interfaces:**
- Consumes: `LibraryError` (Task 2).
- Produces:
  - `schemas.py`:
    - `class SketchOrigin(StrEnum): GENERATED="generated"; BUNDLED="bundled"; COPIED="copied"`;
    - `class SketchStatus(StrEnum): PASSED="passed"; UNCHECKED="unchecked"; FAILED="failed"`;
    - `class Provenance(BaseModel)`: `source: str`, `license: str`, `provider: str | None = None`, `model: str | None = None`, `created_at: datetime`, `usage_terms: str | None = None`, `copied_from_sketch_id: int | None = None`;
    - `class SketchRecord(BaseModel)`: `id: int`, `style_version_id: int`, `idea_key: str`, `idea_description: str`, `blob_hash: str`, `suffix: str`, `origin: SketchOrigin`, `status: SketchStatus`, `status_reason: str | None`, `mirror_safe: bool`, `provenance: Provenance`, `created_at: datetime`.
  - `orm.py`: a SQLAlchemy 2.0 `DeclarativeBase` with typed `Mapped[...]`, and these tables:
    - `channel(id, slug UNIQUE NOT NULL, created_at)`;
    - `style_version(id, channel_id FK→channel ON DELETE CASCADE INDEX, fingerprint NOT NULL, created_at, UNIQUE(channel_id, fingerprint))`;
    - `sketch(id, style_version_id FK→style_version INDEX, idea_key NOT NULL INDEX, idea_description NOT NULL DEFAULT '', blob_hash NOT NULL, suffix NOT NULL, origin, status NOT NULL, status_reason NULL, mirror_safe NOT NULL, provenance JSON NOT NULL, created_at, UNIQUE(style_version_id, idea_key, blob_hash))`.

    Enums are stored as their string values. Timestamps use `DateTime(timezone=True)` with `default=lambda: datetime.now(UTC)`.
  - `db.py`: `open_library(db_path: Path) -> Engine`. It:
    - creates the parent directory;
    - builds the engine `sqlite:///{db_path}` with `connect_args={"timeout": 30}`;
    - on every connect, executes `PRAGMA journal_mode=WAL` and `PRAGMA foreign_keys=ON` through an event listener;
    - runs Alembic `upgrade head` programmatically, using `script_location` = the package's `migrations` dir, located with `importlib.resources`.
  - `0001_initial.py`: written by hand to match `orm.py`. An autogenerated draft is allowed only if it's reviewed.

- [ ] **Step 1: Write the failing tests**

```python
def test_open_creates_schema_and_wal(tmp_path: Path) -> None:          # Review Focus 2
    engine = open_library(tmp_path / "lib" / "library.db")
    with engine.connect() as c:
        assert c.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"
        assert c.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
    assert {"channel", "style_version", "sketch"} <= set(inspect(engine).get_table_names())

def test_open_is_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "library.db"
    open_library(p); open_library(p)   # second open must not fail or re-create

def test_migrations_match_models(tmp_path: Path) -> None:
    engine = open_library(tmp_path / "library.db")
    with engine.connect() as c:
        diff = compare_metadata(MigrationContext.configure(c), Base.metadata)
    assert diff == []
```

- [ ] **Step 2: Run.** `uv run pytest tests/library/test_db.py -v`. Expected: FAIL with ImportError.
- [ ] **Step 3: Implement** `schemas.py`, `orm.py`, `db.py` and the Alembic files. `env.py` must import `narratoflow.library.orm` and set `target_metadata = Base.metadata`. Use `render_as_batch=True`, since SQLite needs batch mode for later ALTERs.
- [ ] **Step 4: Run.** Expected: PASS. Then run the full gate. Expected: green.
- [ ] **Step 5: Commit.** `git commit -m "feat: sketch library schema with Alembic migrations on SQLite (WAL)"`

---

### Task 5: Blob store and library repository

**Files:**
- Create: `src/narratoflow/library/blobs.py`, `library/repository.py`
- Test: `tests/library/test_repository.py`

**Interfaces:**
- Consumes: `open_library`, the ORM models, `SketchRecord`, `Provenance`, `SketchOrigin`, `SketchStatus` (Task 4), `LibraryError`, and `Settings` (Task 2).
- Produces:
  - `BlobStore(root: Path)`:
    - `put(data: bytes, suffix: str) -> str` returns the sha256 hex and writes `root/ab/abcdef….suffix` only if it's absent (dedupe);
    - `path(blob_hash: str, suffix: str) -> Path`, which raises `LibraryError` if the file is missing.
  - `SketchLibrary(engine: Engine, blobs: BlobStore)`. Each method uses its own `with Session(engine) as s:` unit of work:
    - `ensure_style_version(channel_slug: str, fingerprint: str) -> int`. Creates the channel and/or version if missing and returns the version id. Idempotent.
    - `add_sketch(style_version_id: int, idea_key: str, data: bytes, suffix: str, origin: SketchOrigin, status: SketchStatus, mirror_safe: bool, provenance: Provenance, status_reason: str | None = None, *, description: str = "") -> SketchRecord`. `description` is the short English idea description used for retrieval (design §2.2). Plans 2 and 3 must always pass it; it's empty only for legacy or test rows. If the same (version, key, hash) already exists, it returns the existing record and does not raise.
    - `find_reusable(style_version_id: int, idea_key: str) -> list[SketchRecord]`. Returns status `PASSED` or `UNCHECKED` only, ordered by id.
    - `idea_keys(style_version_id: int) -> list[str]`. Distinct keys of reusable sketches, sorted.
    - `idea_catalog(style_version_id: int) -> list[tuple[str, str]]`. `(key, description)` pairs for the same keys, using the most recent non-empty description per key. This catalog goes to the planner in Plan 2, in full up to 300 keys and via embeddings above that.
    - `quarantine(sketch_id: int, reason: str) -> None`. Sets status `FAILED` and the reason. Raises `LibraryError` for an unknown id.
    - `copy_to(sketch_id: int, target_style_version_id: int) -> SketchRecord`. Origin `COPIED`, the same blob, provenance copied with `copied_from_sketch_id` set, and status carried over. Copying a `FAILED` sketch raises `LibraryError`.
  - `default_library(settings: Settings) -> SketchLibrary`. Opens `settings.data_dir / "library.db"` with blobs in `settings.data_dir / "blobs"`.

- [ ] **Step 1: Write the failing tests** (with a `library` fixture on `tmp_path` and a `prov()` helper that returns a valid `Provenance`)

```python
def test_style_version_is_idempotent_and_per_channel(library: SketchLibrary) -> None:
    a = library.ensure_style_version("history", "fp1")
    assert library.ensure_style_version("history", "fp1") == a
    assert library.ensure_style_version("history", "fp2") != a
    assert library.ensure_style_version("tech", "fp1") != a

def test_reuse_excludes_failed(library: SketchLibrary) -> None:
    v = library.ensure_style_version("history", "fp1")
    ok = library.add_sketch(v, "eye|generic", b"A", ".png", SketchOrigin.GENERATED, SketchStatus.PASSED, True, prov())
    bad = library.add_sketch(v, "eye|generic", b"B", ".png", SketchOrigin.GENERATED, SketchStatus.PASSED, True, prov())
    library.quarantine(bad.id, "face detected")
    assert [r.id for r in library.find_reusable(v, "eye|generic")] == [ok.id]

def test_other_style_version_not_reused(library: SketchLibrary) -> None:
    v1 = library.ensure_style_version("history", "fp1"); v2 = library.ensure_style_version("history", "fp2")
    library.add_sketch(v1, "eye|generic", b"A", ".png", SketchOrigin.GENERATED, SketchStatus.PASSED, True, prov())
    assert library.find_reusable(v2, "eye|generic") == []

def test_duplicate_add_dedupes(library: SketchLibrary, tmp_path: Path) -> None:      # Review Focus 3
    v = library.ensure_style_version("history", "fp1")
    a = library.add_sketch(v, "eye|generic", b"A", ".png", SketchOrigin.GENERATED, SketchStatus.PASSED, True, prov())
    b = library.add_sketch(v, "eye|generic", b"A", ".png", SketchOrigin.GENERATED, SketchStatus.PASSED, True, prov())
    assert a.id == b.id
    assert len(list((tmp_path / "blobs").rglob("*.png"))) == 1

def test_copy_keeps_provenance_and_blob(library: SketchLibrary) -> None:
    v1 = library.ensure_style_version("history", "fp1"); v2 = library.ensure_style_version("tech", "fpX")
    src = library.add_sketch(v1, "eye|generic", b"A", ".png", SketchOrigin.GENERATED, SketchStatus.PASSED, True, prov())
    dst = library.copy_to(src.id, v2)
    assert dst.origin is SketchOrigin.COPIED and dst.blob_hash == src.blob_hash
    assert dst.provenance.copied_from_sketch_id == src.id

def test_copy_failed_rejected(library: SketchLibrary) -> None: ...   # quarantine then copy_to -> LibraryError
def test_quarantine_unknown_id(library: SketchLibrary) -> None: ...  # -> LibraryError
def test_idea_keys_lists_only_reusable(library: SketchLibrary) -> None: ...  # failed-only key absent; sorted
def test_idea_catalog_returns_latest_description(library: SketchLibrary) -> None: ...  # two sketches same key, descriptions "a" then "b" -> [(key, "b")]
```

- [ ] **Step 2: Run.** `uv run pytest tests/library/test_repository.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** per the Interfaces block. Convert ORM rows to `SketchRecord` with `model_validate` (`from_attributes=True`), and validate `provenance` from its JSON.
- [ ] **Step 4: Run.** Expected: PASS. Then run the full gate. Expected: green.
- [ ] **Step 5: Commit.** `git commit -m "feat: content-addressed blob store and sketch library repository"`

---

### Task 6: Idea keys, bundled icons, matching, variant choice

**Files:**
- Create: `src/narratoflow/library/keys.py`, `library/matching.py`, `src/narratoflow/visuals/bundled.py`, `assets/icons/index.toml`
- Test: `tests/library/test_matching.py`

**Interfaces:**
- Consumes: `SketchLibrary`, `SketchRecord` (Task 5), and `InvalidIdeaKeyError`.
- Produces:
  - `keys.py`:
    - `GENERIC = "generic"`;
    - `@dataclass(frozen=True) class IdeaKey: object: str; context: str`, with `__str__` returning `"object|context"` and a property `is_generic`;
    - `parse_idea_key(raw: str) -> IdeaKey`. It strips, lowercases, and turns runs of spaces or underscores into `-`. It then requires exactly one `|` and both parts matching `^[a-z0-9]+(-[a-z0-9]+)*$`. Otherwise it raises `InvalidIdeaKeyError` quoting `raw`.
  - `visuals/bundled.py`:
    - `BundledIcon(BaseModel)`: `key: str`, `file: Path`, `license: str`, `source: str`;
    - `BundledIcons(icons: Mapping[str, BundledIcon])`, with `classmethod from_index(index_path: Path) -> BundledIcons`. The TOML has `[icons."eye|generic"] file="…" license="MIT" source="https://…"`, and file paths are resolved relative to the index. Every key must parse as an `IdeaKey` **with a `generic` context**; otherwise it raises `InvalidIdeaKeyError`.
    - `get(key: IdeaKey) -> BundledIcon | None`.
  - `matching.py`:
    - `class MatchKind(StrEnum): REUSED="reused"; BUNDLED="bundled"; NEW="new"`;
    - `@dataclass(frozen=True) class MatchResult: kind: MatchKind; key: IdeaKey; candidates: tuple[SketchRecord, ...] = (); icon: BundledIcon | None = None`;
    - `match_idea(library: SketchLibrary, style_version_id: int, key: IdeaKey, bundled: BundledIcons) -> MatchResult`. The order is: the library's reusable sketches, then the bundled icon **only if `key.is_generic`**, then NEW (§3 step 2, Global Constraints).
    - `choose_variant(candidates: Sequence[SketchRecord], video_id: str, scene_id: str) -> SketchRecord`. It's deterministic: the index is `int(sha256(f"{video_id}:{scene_id}"), 16) % len(candidates)`. Empty candidates raise `ValueError`. (§4.5: the same video re-renders identically, while different videos vary.)
  - `assets/icons/index.toml`: starts with a comment header and an empty `[icons]` table. The icon packs come in Plan 3.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("raw,expected", [
    ("eye|generic", "eye|generic"),
    ("  Old Scroll|Roman ", "old-scroll|roman"),
    ("papyrus|ancient_egypt", "papyrus|ancient-egypt"),
])
def test_parse_normalizes(raw: str, expected: str) -> None:
    assert str(parse_idea_key(raw)) == expected

@pytest.mark.parametrize("raw", ["قلم|generic", "eye", "a|b|c", "|generic", "eye|"])   # Review Focus 1
def test_parse_rejects(raw: str) -> None:
    with pytest.raises(InvalidIdeaKeyError):
        parse_idea_key(raw)

def test_library_beats_bundled(library, bundled_eye) -> None: ...   # stored eye|generic -> REUSED, candidates non-empty
def test_bundled_only_for_generic(library, tmp_path) -> None:
    # bundled index containing "scroll|generic"; key scroll|roman with empty library -> NEW (never BUNDLED)
    ...
def test_bundled_for_generic_when_library_empty(library, bundled_eye) -> None: ...  # -> BUNDLED with icon
def test_bundled_index_rejects_non_generic(tmp_path: Path) -> None: ...  # [icons."scroll|roman"] -> InvalidIdeaKeyError
def test_failed_sketch_does_not_match(library, bundled_empty) -> None: ...  # only quarantined sketch -> NEW

def test_choose_variant_deterministic_and_varies() -> None:
    cands = [make_record(i) for i in range(5)]
    assert choose_variant(cands, "vid1", "s3") == choose_variant(cands, "vid1", "s3")
    picks = {choose_variant(cands, f"vid{n}", "s3").id for n in range(50)}
    assert len(picks) > 1
```

- [ ] **Step 2: Run.** `uv run pytest tests/library/test_matching.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** per the Interfaces block.
- [ ] **Step 4: Run.** Expected: PASS. Then run the full gate. Expected: green.
- [ ] **Step 5: Commit.** `git commit -m "feat: exact idea-key matching with generic-only bundled icons and deterministic variant choice"`

---

### Task 7: Input quality gate (§3 step 1a)

**Files:**
- Create: `src/narratoflow/text/quality.py`, `tests/fixtures/input/clean_ar.txt`, `tests/fixtures/input/clean_en.txt`, `tests/fixtures/input/corrupt_ar.txt`
- Test: `tests/text/test_quality.py`

**Fixture contents:**
- `clean_ar.txt`: the first two paragraphs of the owner's reference script 1, retyped cleanly, starting «﴿يوم لا يجزى والد عن ولده…﴾ ولكن هذه الحقيقة البسيطة الواضحة…».
- `clean_en.txt`: the first two paragraphs of reference script 3 («The coordination gap - and why AI matters …»).
- `corrupt_ar.txt`: the first 30 lines of reference script 2, verbatim with the corruption, including the `ييييدعى`, `يومن رررا`, `31` and `37` lines.

**Interfaces:**
- Consumes: `Language`, `CorruptInputError`.
- Produces:
  - `class IssueKind(StrEnum)`, with these values:
    - `STRETCHED_LETTERS`: any Arabic letter repeated 3 or more times in a row (after harakat are stripped), or a Latin letter 4 or more times. Tatweel is **not** an issue; `normalize_input` removes it;
    - `PAGE_NUMBER`: a line that is only digits, Arabic-Indic digits or punctuation;
    - `FRAGMENTED_LINES`: more than 20% of non-empty lines are 3 characters or fewer, reported once;
    - `REPLACEMENT_CHAR`: U+FFFD or a private-use codepoint.
  - `@dataclass(frozen=True) class QualityIssue: kind: IssueKind; line: int; excerpt: str`, where `line` is 1-based and the excerpt is at most 40 characters.
  - `@dataclass(frozen=True) class QualityReport: issues: tuple[QualityIssue, ...]`, with a property `ok: bool`.
  - `normalize_input(text: str) -> str`. It strips the BOM, converts CRLF to LF, applies NFC, and removes U+200E, U+200F, U+061C and **tatweel U+0640**. Tatweel is visual justification only and changes no word; the skeleton guard already ignores it. It **does not** touch letters or harakat.
  - `check_input(text: str, language: Language) -> QualityReport`, which runs on `normalize_input(text)`. For Arabic letter stretching, **only harakat** (U+064B–U+065F, U+0670) are stripped in memory before matching, so harakat-interleaved repetitions (e.g. `يَيَيَ`) are caught. **Do not reuse `strip_diacritics` from `verify.py`**: it also removes tatweel and Qur'anic marks. Line numbers refer to the original text.
  - `require_clean(text: str, language: Language) -> str`. It returns the normalized text, or raises `CorruptInputError` whose message lists up to 10 issues as `line N: KIND «excerpt»`.

- [ ] **Step 1: Write the failing tests**

```python
def test_clean_arabic_passes() -> None:
    assert check_input(read("clean_ar.txt"), Language.ARABIC).ok

def test_clean_english_passes() -> None:
    assert check_input(read("clean_en.txt"), Language.ENGLISH).ok

def test_corrupt_arabic_flagged_with_locations() -> None:
    report = check_input(read("corrupt_ar.txt"), Language.ARABIC)
    kinds = {i.kind for i in report.issues}
    assert IssueKind.STRETCHED_LETTERS in kinds and IssueKind.PAGE_NUMBER in kinds
    assert all(i.line >= 1 for i in report.issues)

def test_tatweel_is_normalized_not_flagged() -> None:
    assert normalize_input("يــــــــوحنا") == "يوحنا"
    assert check_input("الحمــــد لله", Language.ARABIC).ok

def test_vowel_interleaved_stretching_flagged() -> None:
    assert not check_input("يَيَيَ المعمدان", Language.ARABIC).ok

def test_bom_crlf_and_bidi_marks_not_flagged() -> None:          # Review Focus 4
    text = "\ufeff" + read("clean_ar.txt").replace("\n", "\r\n").replace(" ", " \u200f", 5)
    assert check_input(text, Language.ARABIC).ok

def test_normalize_keeps_diacritics() -> None:
    assert normalize_input("اَلْعِلْمُ\r\n") == "اَلْعِلْمُ\n"

def test_legit_doubled_letters_not_stretched() -> None:
    # shadda and natural double letters (الله، اللّيل) must never trip the 3+ rule
    assert check_input("إنّ الله مع الصابرين، واللّيل إذا يسر", Language.ARABIC).ok

def test_require_clean_raises_with_line_numbers() -> None:
    with pytest.raises(CorruptInputError, match=r"line \d+: STRETCHED_LETTERS"):
        require_clean(read("corrupt_ar.txt"), Language.ARABIC)
```

- [ ] **Step 2: Run.** `uv run pytest tests/text/test_quality.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** per the Interfaces block. Use the regex `([\u0621-\u063A\u0641-\u064A])\1{2,}` (Arabic letters excluding tatweel; write it with `\u` escapes in code) on the harakat-stripped line for Arabic stretching and `([A-Za-z])\1{3,}` for Latin.
- [ ] **Step 4: Run.** Expected: PASS. Then run the full gate. Expected: green.
- [ ] **Step 5: Commit.** `git commit -m "feat: input quality gate rejecting PDF copy-paste corruption before any spend"`

---

### Task 8: CLI wiring

**Files:**
- Modify: `src/narratoflow/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `get_settings`, `setup_logging`, `load_channel`, `style_fingerprint`, `check_provider_keys`, `default_library`, `require_clean`, `parse_idea_key`, and all errors.
- Produces these commands:
  - an app callback that calls `setup_logging(get_settings().log_level)` once;
  - `narratoflow channel check CHANNEL_TOML`: loads the channel, checks keys, prints the name, the style fingerprint, and the paid kinds;
  - `narratoflow library keys CHANNEL_TOML`: ensures the style version and prints `idea_keys` one per line;
  - `narratoflow library copy SKETCH_ID TARGET_CHANNEL_TOML`: prints the new sketch id;
  - `narratoflow check-input FILE --language ar|en`: prints `OK` or the issue list.

  Every `NarratoFlowError` is caught **only** in a single `main()` wrapper (the `[project.scripts]` entry, changed to `narratoflow.cli:main`). The wrapper prints the message in red through `rich` and exits with code 1. Any other exception propagates.

- [ ] **Step 1: Write the failing tests** (using `typer.testing.CliRunner` and the `data_dir` fixture)

```python
def test_check_input_ok(tmp_path: Path) -> None: ...            # clean_ar.txt -> exit 0, "OK"
def test_check_input_corrupt_exit_1() -> None: ...              # corrupt_ar.txt -> exit 1, "STRETCHED_LETTERS" in output
def test_channel_check_missing_key_exit_1(channel_file) -> None: ...  # exit 1, "NARRATOFLOW_IMAGE_API_KEY" in output, no traceback
def test_channel_check_ok(channel_file, monkeypatch) -> None: ...     # key set -> exit 0, fingerprint printed
def test_library_keys_empty(channel_file, monkeypatch) -> None: ...   # exit 0, empty output; library.db created under data_dir
```

  For the error paths, run through `main` (wrap the runner invocation with `standalone_mode=False`, or call `main` via `subprocess` and `uv run`).
- [ ] **Step 2: Run.** `uv run pytest tests/test_cli.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the commands and the `main()` wrapper. Remove the old `init` / `validate` / `strip` / `render` commands only if no test references them. Keep `validate`, since `test_models` uses `Project`.
- [ ] **Step 4: Run the full gate:**
  ```bash
  uv run ruff check --no-cache .
  uv run ruff format --check .
  uv run mypy
  uv run coverage run -m pytest
  uv run coverage report
  ```
  Expected: 0 failures, 100% pass across all tests.
- [ ] **Step 5: Update the README** "Quick start" with `channel check`, `library keys` and `check-input`, plus `.env.example` usage.
- [ ] **Step 6: Commit.** `git commit -m "feat: CLI for channel check, library keys/copy, and input check"`

---

## Done when

- The gate is green in CI on `main` (ruff, mypy strict, pytest branch coverage).
- `uv run narratoflow check-input` rejects reference script 2 and accepts scripts 1 and 3.
- `uv run narratoflow channel check` fails fast, naming the missing env var, for a paid provider without a key.
- The library survives reopen, never reuses a quarantined sketch, never serves a bundled icon for a non-generic key, and starts a new style version whenever the image model or reference sketch changes.
