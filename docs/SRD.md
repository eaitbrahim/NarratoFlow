# System Requirements Document (SRD)
## Project: Open-Source Whiteboard & Sketch AI Video Engine ("NarratoFlow")

> **Status:** v1.1, 2026-11-13; supersedes v1.0 (see git history). It says how the system is built, as shipped: every claim cites the code or an ADR. What and why is in [`PRD.md`](PRD.md); the review that drove the corrections is [`REVIEW.md`](REVIEW.md). Statements about the review station (#165, issue #30) describe a PR that lands before this one is merged.
>
> A line marked **Recommendation, owner to confirm** is a decision the code does not make.

## Changes from v1.0

| # | v1.0 claim | Correction in v1.1 | Section |
| :--- | :--- | :--- | :--- |
| C1 | Remotion or MoviePy composites | NumPy/Pillow frames piped into ffmpeg; Remotion is source-available and was dropped; MoviePy is not used | 4, 7 |
| C2 | edge-tts as the only voice, zero lock-in | A voice chain with retry, a Piper fallback, a cache and per-part terms ids | 3, 4 |
| C3 | XTTS-v2 / F5-TTS local engines | Non-commercial weights; not implemented; listed in the licence matrix | 8 |
| C4 | Farasa diacritizer | Research and non-commercial; not used | 8 |
| C5 | Whisper word alignment, sub-millisecond | Provider spans, else CTC forced alignment, else an estimate | 4 |
| C6 | A reshaper and a bidi package before `.ass` | Logical-order Unicode into libass; neither is used | 4, 7 |
| C7 | Potrace outlines animated by dash offset | Centerline strokes ordered for the reveal (`visuals/strokes.py`) | 4 |
| C8 | SDXL Turbo + LineArt ControlNet | `lineart-diffusion`: the documented FLUX.1-schnell (`style.image_model` is required, no code default), an optional LoRA, pinned seed; "open-weights" | 4, 8 |
| C9 | Mishkal validates classical grammar | Dialect is flagged by CAMeL DID; the diacritizer is benchmarked (ADR 0005) | 4 |
| C10 | `moviepy>=1.0.3`, unused and missing packages | Extras in `pyproject.toml`, `uv.lock`, `uv sync --locked` in CI | 7 |

---

## 1. Architecture

NarratoFlow is a **staged command-line pipeline over one file**, `series.json` (ADR 0004). A stage reads the file, does its work and, if it changes the plan, writes the file back atomically; stages that make artifacts (audio, timelines, renders, reports) write them beside it.

```
 script.txt + channel.toml
        |
   check-input / check-dialect          (free, flag-only checks)
        v
     plan  ------------------->  series.json  (Series -> Part -> Scene)
        v                              ^   ^
     approve  (estimate, lock)         |   |  edits
        v                              |   |
 sketch generate / resolve / upload ---+   |
        v                                  |
   audio build  (vocalize, voice, time)    |        narratoflow ui
        v        audio/part-NN.flac        |        (review station)
     render      + timeline                +--------  edits series.json,
        v        part-NN.mp4 + .ass                  re-runs one part
     review      (content check, optional)
        v
     publish  ->  export/ + publish report
```

- Commands (`src/narratoflow/cli.py`): `validate`, `check-input`, `check-dialect`, `channel check`, `library keys|icons|copy`, `plan`, `approve`, `sketch generate|upload|resolve`, `audio build`, `render`, `review`, `publish`, `ui`.
- `Series` is the only editable IR. `plan`, `approve`, `sketch ...`, `audio build` and `review` write it; `render` and `publish` only read it. `Series.load` and `Series.save` are the only readers and writers (atomic: a temp file, then `os.replace`).
- The sketch library is a SQLite database (SQLAlchemy, Alembic migrations in `src/narratoflow/library/migrations`) of sketches, their provenance, style versions and idea embeddings.
- Every paid call goes through a tracker: `SpendTracker` (`costs/tracker.py`) checks the remaining budget before a call and charges after success, against `ChannelConfig.spending_limit_usd`.
- Heavy engines are optional extras (section 7); the base install and the tests use no GPU and no network.

---

## 2. IR schema

The IR is `src/narratoflow/models.py`. `SCHEMA_VERSION = 2`; `Series.load` refuses any other or missing version and there are no migrations.

### Series, Part, Scene

| Model | Fields (type, default) |
| :--- | :--- |
| `Series` | `schema_version` (2), `title`, `channel_slug`, `language` (`ar`/`en`), `source` (`ScriptSource`), `parts`, `approved` (False), `est_planning_cost_usd`, `ai_disclosure` (True) |
| `ScriptSource` | `author`, `work`, `edition`, `rights` (`own`, `public_domain`, `permission`, `third_party_in_copyright`), `permission_note`, `ai_transcribed`; extra fields forbidden |
| `Part` | `index`, `count`, `hook_text`, `recap_text`, `teaser_text`, `end_card_text`, `scenes`, `flags`, `est_cost_usd`, `spent_usd`, `spend` (per charge line), `content_review` (`ContentReview`, None until `review` runs) |
| `Scene` | `id`, `narration`, `title`, `visual`, `layouts` (`AspectRatio` to `Box`), `min_duration_ms` (3000), `draw_fraction` (0.65), `kind` (`narration`/`verse`), `verse`, `idea_key`, `idea_description`, `match` (`reused`/`bundled`/`new`), `sketch_id`, `sketch_attempt`, `est_duration_s`, `flags`, `card_text` |
| `Narration` | `source_text` (as authored), `tts_text` (vocalized, for the voice), `caption_text` (lighter, for captions), `protected` (verified text, never auto-diacritized), `audio_path`, `words` (`WordTiming`: `text`, `start_ms`, `end_ms`) |
| `Visual` | `source` (`icon`, `diffusion`, `image_to_sketch`, `text_card`), `prompt`, `asset_path`, `strokes_path`, `seed` |
| `VerseRef` | `surah` (1 to 114), `ayah_start`, `ayah_end`, `certain`, `uthmani_text` |
| `Box` | normalized `x`, `y`, `w`, `h` in [0, 1] |
| `Voice` / `Project` | the v1 `storyboard.json` shape; `Project` (schema 1) remains only for `validate` |

Rules that matter downstream: the author's words are never changed (`source_text`); a verse scene carries verified Uthmani text and is voiced by a recording, not a synthesizer; a scene with `visual.source = image_to_sketch` is the creator's upload and is never regenerated.

### Artifact versions

Each stage artifact is versioned on its own (ADR 0004): `PartTimeline` `TIMELINE_SCHEMA_VERSION = 1` (`audio/timeline.py`), `RenderReport` `REPORT_SCHEMA_VERSION = 1` (`render/report.py`), publish reports `PUBLISH_SCHEMA_VERSION = 1` (`publish/report.py`), `StrokeSet` `version = 1` (`visuals/strokes.py`; format in [`strokes-schema.md`](strokes-schema.md)). Every loader refuses another version. Adding an optional field with a default does not bump a version; removing, renaming or re-typing one does.

### Library records

`SketchRecord` (`library/schemas.py`) has an origin (`generated`, `bundled`, `copied`, `uploaded`), a status (`passed`, `unchecked`, `failed`) and `Provenance` (provider, model, licences, attribution, style version). An `uploaded` record is never offered for reuse and is not relabelled when copied to another channel.

---

## 3. Interfaces

Implementations are selected from the channel file; tests use fakes (`tests/fakes`).

| Protocol | Contract | Implementations (file) |
| :--- | :--- | :--- |
| `Diacritizer` | `name`, `diacritize(text) -> str` (`interfaces.py`) | `LlmDiacritizer`, `MishkalDiacritizer` (`text/tashkeel/engines.py`), behind `vocalize_scene` and the skeleton guard (`text/tashkeel/verify.py`) |
| `SpeechProvider` | `synthesize(SpeechRequest) -> Speech` with audio, optional word spans, provider, model, billed chars (`tts/__init__.py`) | `EdgeProvider` (`tts/edge.py`), `PiperProvider` (`tts/piper.py`), `ElevenLabsProvider` (`tts/elevenlabs.py`); wrapped by `RetryingSpeechProvider` (`tts/retry.py`) and `TrackedSpeechProvider` (`tts/factory.py`) |
| `Aligner` | `align(samples, sample_rate, text, language) -> list[TimedSpan]` (`align/__init__.py`) | `CtcAligner` (`align/ctc.py`, extra `align`) |
| `ImageProvider` | `name`, `model`, `generate(ImageRequest) -> GeneratedImage` (`images/__init__.py`) | `OpenAIImageProvider` (`images/openai_images.py`), `LocalDiffusionProvider` (image to image, `images/local_diffusion.py`), `LineArtDiffusionProvider` (text to image, `visuals/diffusion.py`); built by `images/factory.py` |
| `TextToImagePipeline` | the injected diffusers pipeline of `lineart-diffusion` (`visuals/diffusion.py`) | the lazy diffusers factory; a fake in tests |
| `LLMClient` | `complete_structured(system, user, output, images) -> StructuredResult` (`llm/base.py`) | `AnthropicClient` (`llm/anthropic_client.py`), `OllamaClient` (`llm/ollama.py`); wrapped by `TrackedLLM` |
| `Embedder` | embeds idea descriptions for library matching (`embed/__init__.py`) | `VoyageEmbedder`, `OllamaEmbedder`, `TrackedEmbedder` |
| `VisualProvider` | `produce(scene, out_dir) -> Visual` (`interfaces.py`) | a protocol only; no implementation is wired. Visual sources are chosen in `sketching/` and `visuals/` |
| Compositor | not a protocol: `render_segment(SceneJob)` in `render/compose.py` and `encode_part` in `render/encoder.py` | one implementation (ADR 0002) |

The provider-selecting channel keys are `voice.provider` and `voice.fallback`, `style.image_provider` (`openai`, `local-diffusion`, `lineart-diffusion`), `llm.provider` (`anthropic`, `ollama`), `diacritizer` (`llm`, `mishkal`) and `embed.provider`.

---

## 4. Modules (corrected)

### 4.1 Classical Arabic NLP and diacritization (C4, C9)
- `ChannelConfig.diacritizer` is `llm` (default) or `mishkal`. `vocalize_scene` runs the engine on a narration scene and applies the **skeleton guard** (`assert_skeleton_preserved`): the consonants of the output must equal the input's. On a mismatch or a provider error the scene keeps its undiacritized text and gets the flag `tashkeel_rejected` (`REJECTED_FLAG`). Protected and verse text is never diacritized.
- **Benchmark (ADR 0005).** `scripts/bench_tashkeel.py` scores engines on a pinned gold set (300 lines of Fadel et al. 2019, fetched by `scripts/fetch_gold_tashkeel.py`; the repo commits no Arabic text, only `docs/benchmarks/tashkeel-gold.toml`). Metrics: DER and WER, with and without case endings, with a rejected output counted wrong (`text/tashkeel/bench.py`). Results: identity DER 82.77, WER 99.49; **Mishkal DER 25.45, WER 65.03** (no case ending: 18.83 and 38.79); LLM and CATT not run (#155, #156). The default stays `llm`.
- **Dialect.** `check-dialect` marks sentences that read as dialect using CAMeL Tools DID and never rejects (`text/dialect.py`; the 274 MB model data is installed by the user, not downloaded by NarratoFlow).
- **Input quality.** `check-input` rejects corrupt text such as pasted PDF output (`text/quality.py`) before any spend.
- **Spoken text.** A lexicon and number handling (`text/spoken.py`, `text/numbers.py`, `text/amounts.py`) turn digits, signs and currency into spoken words before the voice.

### 4.2 Voice chain, timing and audio (C2, C3, C5; ADR 0003)
- `build_voice_chain` returns the primary voice then each `voice.fallback` entry (`edge` or `piper`), validated before any spend. `narrate_part` voices a whole part with one voice and, if it fails, re-voices the whole part with the next, recording `fallback_reasons`. Retry inside a voice is three attempts, 1 s and 4 s apart, on `TransientProviderError` only.
- **Word times** (`_timings` in `audio/narrate.py`): provider spans when they map onto the spoken words (`TimingSource.PROVIDER`), else the CTC aligner (`ALIGNED`), else a length-weighted estimate (`ESTIMATED`, with a warning). Alignment runs on the undiacritized text. The size of the edge-versus-CTC offset on Arabic is not settled (#133; `docs/spikes/`), so no precision is claimed.
- **Verse scenes** use a recording of an allow-listed reciter (`quran/reciters.py`, `audio/recitation.py`); there is no TTS fallback for a verse (`ReciterError`). No reciter is bundled (`assets/LICENSES.md`).
- Marker sound effects (`audio/sfx.py`, synthesized in-repo), ducking (`audio/duck.py`) and loudness mastering to -14 LUFS, true peak -1 dBTP (`audio/mixer.py`, `audio/io.py`).
- The voice **cache** key covers provider, model, voice id, spoken text and, for a local model, the model file's path, size and mtime (`pipeline/cache.py`, `_key` in `narrate.py`). `estimate_voice_cost` (`audio/build.py`) prices only the uncached characters (added with the review station).

### 4.3 Visuals (C7, C8)
- **Source choice per scene:** a bundled Tabler icon, a reused library sketch, a generated sketch, an upload, or a text card. Candidates come from idea keys (`library/keys.py`, `library/candidates.py`); `generate_missing` (`sketching/generate.py`) attaches, generates or quarantines.
- **Uploads (#28, PR #158).** `sketch upload` (`cli.py`) calls `attach_upload` (`sketching/upload.py`): the image is decoded, size-checked, turned into line art by `xdog_lineart` (`visuals/lineart.py`, scikit-image), density-checked, stored as an `uploaded` record with the creator's licence (validated against `[licences.assets]`; `own-work` for the creator's own) and source, and attached to the scene with `visual.source = image_to_sketch`. A refused image stores nothing. It is free (no vision call, status `unchecked`). `generate_missing` never generates for such a scene.
- **Diffusion (#29, PR #162).** `LineArtDiffusionProvider` (`visuals/diffusion.py`) draws from text with a pinned model, an optional LoRA and a seed; the look comes from the pins, not a ControlNet. The pipeline is injected and built lazily, so no run imports torch or diffusers unless the channel selects `lineart-diffusion` and a scene needs a new sketch. Pins are the channel's `[style.diffusion]` table (`DiffusionStyle` in `channels/config.py`): model licence and source (mandatory), model revision, LoRA with its own licence and source, `lora_scale`, `seed`, `steps` (4), `guidance_scale` (0.0), `size` (1024), `device` (`auto`, `cpu`, `cuda`, `mps`). The documented model (README) is `black-forest-labs/FLUX.1-schnell` (Apache-2.0, a gated repository); `style.image_model` is required and has no code default. Each drawing mixes the channel seed with the video, scene and attempt. Licences are copied into provenance and a composite is written `<model> AND <lora>`; publish checks them against `[licences.generated]` (`Apache-2.0`, `MIT` pass; `OpenRAIL++-M` warns; anything else, such as FLUX.1-dev, fails). No LoRA ships (#161).
- **VRAM (not measured; figures from model cards, see #160):**

  | Model | VRAM | Status |
  | :--- | :--- | :--- |
  | FLUX.1-schnell, fully on GPU | about 24 GB or more | not measured (#160) |
  | FLUX.1-schnell with CPU offload | lower peak, slower | not measured (#160) |
  | SDXL fp16 (OpenRAIL++-M) | about 8 GB typical, 6 GB tight | not measured (#160) |
  | CPU only | none | works, impractically slow |

- **Strokes.** `extract_strokes` (`visuals/strokes.py`) skeletonizes a sketch and orders centerline strokes (outer first, then by proximity); the reveal draws them with a hand sprite (`render/reveal.py`, `visuals/marker.py`). A quarantined, text-card or verse scene has no sketch.
- Every sketch passes a free density check, and a vision check when a vision-capable model is configured (`sketching/vision.py`).

### 4.4 Compositor (C1; ADR 0002)
- `render_segment` (`render/compose.py`) draws each scene's frames with NumPy and Pillow and writes them to an ffmpeg process over `pipe:0` as FFV1. Scenes render in a `spawn` process pool (`--workers`, default one per core). One final pass per part (`encode_part`) concatenates, burns the libass captions, muxes the audio and encodes with fixed x264 and AAC arguments; `verify_mp4` checks size, frame count and duration.
- Geometry is pinned: 1080x1920, 30 fps (`render/layout.py`), with safe zones; a camera track gives pan and zoom (`render/camera.py`).
- **Output is 9:16.** A channel listing `16:9` gets `LANDSCAPE_WARNING` in the part's report. The 16:9 file is a pillarboxed copy made by the review station (4.6); a native layout is #163.
- A part whose render inputs are unchanged is skipped (`RenderReport.inputs_key`; `RENDER_INPUTS_VERSION` is bumped by any change to what frames or the encode look like).
- libass with HarfBuzz is a hard dependency of the system ffmpeg; `render/ffmpeg.py` fails with an install hint when the `ass` filter is missing or Arabic letters do not join.

### 4.5 Captions (C6)
- `captions/ass.py` writes the `.ass` file with **logical-order Unicode** and lets libass shape it (`shaping=complex`); no reshaping or reordering package is used. Fonts are Amiri (Arabic) and Noto Sans (Latin), passed with `fontsdir`. Karaoke highlighting follows the word times with `\k` tags, never `\kf`, which libass sweeps left to right even over right-to-left text; verse captions are in `captions/verse.py`.

### 4.6 Review station (#30, PR #165)
- `narratoflow ui SERIES_JSON --channel CHANNEL.toml` (`cli.py`; options `--audio`, `--export`, `--host`, `--port`) serves a local Gradio page (extra `ui`).
- **Pure functions, thin layer.** `ui/station.py` holds the logic: `StationPaths`, `list_scenes`, `scene_view`, `edit_text`, `edit_timing`, `upload_visual`, `regenerate_cost`, `regenerate_visual`, `use_text_card`, `rebuild_scene`, `export`. `ui/app.py` only wires them to Gradio (`build_app` takes the `gradio` module as an argument, so it is tested with a fake and the package imports without the extra) and turns a `NarratoFlowError` into `gr.Error`.
- **No state of its own.** Every handler that writes takes the `series.json` sha256 the page was shown; a stale page is refused with "reload", and each returns the new sha256. The paths are derived exactly as the CLI does (`pipeline/paths.py`: `default_audio_dir`, `default_render_dir`, `default_export_dir`, `default_cache_root`).
- **Re-runs only that scene's work.** `rebuild_scene` runs `generate_missing(only={scene})`, `build_audio(parts=[n])` (the TTS cache serves the other scenes, so no provider call for them) and `build_render(parts=[n])`. Other parts are not passed to the renderer, so their files are untouched. Rendering still redraws every scene of the edited part. A per-scene segment cache inside a part was considered and **withdrawn**: it would change the frame-exact render core (ADR 0002) and the encoder; it is tracked in #164.
- Verse text is not editable; the author's words (`source_text`) are read-only; paid actions show an estimate and need a confirm.
- **Exports.** Export 9:16 renders (unchanged parts are kept), checks and exports every part as `publish` does. Export 16:9 writes `part-NN.16x9.mp4` with `pillarbox_export` (`publish/landscape.py`): the 9:16 video centred on white bars at 1920x1080, re-encoded with the project's x264 arguments, audio copied, verified before it is renamed into place. It is a derivative file, listed in the station's result and not recorded in the publish report.

### 4.7 Planning and publishing
- **Planning** (`planning/`): `plan` ingests the script, segments it with the LLM, locating the pieces in the author's text so a rewrite is caught (`segment.py`), assigns idea keys (`ideas.py`), matches the library (`planning/pipeline.py`) and estimates cost; `approve` locks the plan (`planning/approval.py`).
- **Publish** (`publish/`): the checks are `rules_file`, `render_current`, `asset_provenance`, `script_source`, `sketch_likeness`, `sketch_style`, `verse_text`, `ai_disclosure`, `content_rules` and `technical` (`publish/checks.py`), against `src/narratoflow/data/platform_rules.toml` (version and `checked` date, platform limits, licence tables, content policies). A FAIL blocks a part; a WARN is recorded. The AI-disclosure text and licence provenance come from `publish/disclosure.py` and `publish/provenance.py`.

---

## 5. Non-functional requirements

| # | Requirement | Evidence |
| :--- | :--- | :--- |
| NFR-1 | **Render time.** A 180 s part renders in under 600 s on CPU with the default workers. Measured 84.5 s (16-core arm, ffmpeg-full 9.0.2); 181.8 s with one worker. Single runs, no variance known. | ADR 0002, `docs/benchmarks/2026-11-13-render-cpu.md`, `scripts/bench_render.py` |
| NFR-2 | **Determinism.** The same inputs give the same frames; the worker count does not change them. Diffusion seeds are the channel pin mixed with video, scene and attempt; this is proved offline with an injected pipeline. Whether a real model is bit-identical on a GPU is not measured (#160). | `test_workers_do_not_change_frames` (`tests/render/test_build.py`), ADR 0002 |
| NFR-3 | **Caching.** (a) The voice cache serves any scene whose provider, model, voice, text and model file are unchanged. (b) A part with unchanged render inputs is skipped. (c) The sketch library reuses a sketch by idea key and style version, never an uploaded one. (d) A per-scene render cache inside a part was **withdrawn** (it would change the frame-exact render core) and is tracked in #164. | `pipeline/cache.py`, `render/report.py`, `library/repository.py` |
| NFR-4 | **Cost control.** A paid call is refused when its estimate exceeds the part's remaining budget; progress already made is kept and a re-run reuses it. | `SpendTracker`, `BudgetExceededError`, `test_budget_for_exactly_two_stops_the_third_and_rerun_reuses` |
| NFR-5 | **Minimum hardware.** The default path is CPU-only (ADR 0002). **Recommendation, owner to confirm:** a 4-core CPU, 8 GB RAM, ffmpeg with libass; NFR-1 re-measured there. The `lineart-diffusion` path needs a GPU whose VRAM is not measured (#160). | PRD 8, answer 4 |
| NFR-6 | **Disk.** Not budgeted by the code. **Recommendation, owner to confirm:** budget per part for its FLAC, lossless intermediate segments and MP4; measure on the benchmark part before setting a figure. | none yet |
| NFR-7 | **No network in tests**, no GPU in tests, no paid call in tests. | `tests/`, CI (`.github/workflows/ci.yml`) |
| NFR-8 | **Quality gate.** CI runs `uv sync --locked`, Ruff, the format check, mypy (strict) and the test suite with a coverage report; CI does not enforce a coverage threshold (no `fail_under`); the delivery gate in the plan holds the coverage total at 97% or more. | `.github/workflows/ci.yml`, the plan's delivery gate |
| NFR-9 | **Privacy.** Publish check reasons never carry the user's directories (`bare_paths` in `publish/checks.py`); a voice retry is logged without the text. | `tests/publish/test_checks.py`, `test_retry_logged_without_text` (`tests/tts/test_retry.py`) |

---

## 6. Failure modes

Each row is a trigger, what the system does, the error raised or flag set, and a test that pins it.

| Trigger | Behaviour | Error or flag | Test |
| :--- | :--- | :--- | :--- |
| edge outage or a failing voice | retry a transient error up to three attempts; then re-voice the whole part with the next voice in `voice.fallback` (Piper); the reason is recorded | `TransientProviderError`, then `fallback_from` and `fallback_reasons` in the timeline | `test_primary_failure_revoices_whole_part_and_records_fallback`, `tests/tts/test_retry.py` |
| voice extra not installed | counts as a failed voice and the chain moves on | `ExtraMissingError` naming the extra | `test_missing_extra_counts_as_failed_voice` |
| a reciter recording cannot be resolved | no TTS fallback for a verse | `ReciterError` | `test_recitation_provider_error_never_falls_back_to_tts` |
| invalid or rewritten LLM output | one retry with a note naming the offset or violation; then the stage fails | `SegmentationError` (segmenting), `PlanningError` (idea rejected) | `test_rewrite_rejected_after_one_retry`, `test_rewrite_fixed_on_retry_names_offset` |
| no icon or library match for a scene | generate a sketch; if it fails the check it is quarantined and the creator resolves it (regenerate, text card, swap, upload) | flag `sketch_quarantined` | `tests/sketching/test_resolve.py` |
| a paid call would exceed the budget | stop before the call, keep progress, name the room left | `BudgetExceededError` | `test_budget_refusal_names_what_spent_and_remaining` |
| a diacritizer changes the words | keep the source text | flag `tashkeel_rejected` | `test_word_change_rejected_and_flagged` |
| a sketch or audio timeline no longer matches the library | refuse before any frame; the message says to rebuild the audio | `StaleTimelineError` | `test_stale_sketch_id_refused_before_any_frame` |
| an optional extra is missing | refuse with an install hint | `ExtraMissingError` (`AlignerUnavailableError` for the aligner) | `test_missing_extra_raises_aligner_unavailable`, `test_piper_missing_extra` |
| a series, timeline, report or stroke file has another schema version | refuse; re-plan or rebuild | `PlanningError` (project, series), `AudioError` (timeline, `audio/timeline.py`), `RenderError` (render report, `render/report.py`), `PublishError` (publish report and series summary, `publish/report.py`), `StrokeDataError` (stroke file, field `version`, `visuals/strokes.py`) | `test_project_load_rejects_other_or_missing_schema`, `test_load_rejects_non_int_schema_version` (`tests/test_models.py`); `test_load_unknown_version_raises_audio_error` (`tests/audio/test_timeline.py`); `test_report_roundtrip_and_strict` (`tests/render/test_report.py`); `test_report_roundtrip_and_strict`, `test_summary_marks_unchecked_and_earlier_parts` (`tests/publish/test_report.py`); `test_from_json_rejects_bad_data` (`tests/visuals/test_strokes.py`) |
| an in-copyright third-party script | block before any spend | `ScriptBlockedError` | `tests/planning/test_pipeline.py` |
| ffmpeg lacks libass or does not join Arabic | refuse with an install hint | `RenderError` | `test_check_arabic_shaping_rejects_unjoined` (`tests/render/test_assets.py`) |
| a part fails a publish check | block that part | `PublishCheckFailedError` | `tests/publish` |
| the review station page is stale | refuse the edit and ask for a reload | `PlanningError` | `test_edit_refuses_stale_view` (#165) |

---

## 7. Dependency policy (C10)

- **Extras** in `pyproject.toml`: `arabic` (`mishkal`, `camel-tools`), `tts-edge` (`edge-tts`), `tts-local` (`piper-tts`), `align` (`ctc-forced-aligner`, `unidecode`), `diffusion` (`torch`, `diffusers`, `transformers`), `ui` (`gradio`), and `all`. The base install needs none of them and imports none of them.
- **Locked.** `uv.lock` is committed and CI runs `uv sync --locked`; `uv lock --check` is part of the delivery gate, not CI.
- **No unused dependencies.** `controlnet-aux` was removed with the diffusion provider.
- **Not used:** MoviePy (the compositor is NumPy/Pillow and ffmpeg), faster-whisper (alignment is provider spans, CTC or an estimate), Streamlit (the station is Gradio), cairosvg (icons are rasterized at fetch time), `arabic-reshaper` and `python-bidi` (libass shapes), Remotion (source-available).
- **System tools:** ffmpeg and ffprobe built with libass and HarfBuzz (`NARRATOFLOW_FFMPEG` overrides the path), found by `render/ffmpeg.py`.
- **Extras are a licence boundary.** Several optional packages are GPL (section 8). They are not imported by the base install, but **distributing a build that bundles a GPL extra brings GPL obligations** for that build.

---

## 8. Licence matrix

Every runtime dependency, extra, model and bundled asset. "Commercial-safe" is **yes**, **warn** (usable, with a condition the creator must check) or **no**. "Checked" is the date the source was read; every PyPI and Hugging Face row was re-read on 2026-11-13 (`curl https://pypi.org/pypi/<pkg>/json`, `curl https://huggingface.co/api/models/<id>`, `gh api repos/<owner>/<repo>`). Versions are the ones in `uv.lock` where a package is locked.

### Core dependencies

| Component | Version | Licence | Commercial-safe | Source checked | Date |
| :--- | :--- | :--- | :--- | :--- | :--- |
| NarratoFlow | 0.1.0 | Apache-2.0 | yes | `pyproject.toml` | 2026-11-13 |
| pydantic | 2.14.3 | MIT | yes | https://pypi.org/pypi/pydantic/json | 2026-11-13 |
| pydantic-settings | 2.16.2 | MIT | yes | https://pypi.org/pypi/pydantic-settings/json | 2026-11-13 |
| typer | 0.29.0 | MIT | yes | https://pypi.org/pypi/typer/json | 2026-11-13 |
| rich | 15.0.0 | MIT | yes | https://pypi.org/pypi/rich/json | 2026-11-13 |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | yes | https://pypi.org/pypi/numpy/json | 2026-11-13 |
| opencv-python-headless | 5.0.0.93 | Apache-2.0 | yes | https://pypi.org/pypi/opencv-python-headless/json | 2026-11-13 |
| pillow | 12.3.0 | MIT-CMU (HPND) | yes | https://pypi.org/pypi/pillow/json | 2026-11-13 |
| scikit-image | 0.26.0 | BSD | yes | https://pypi.org/pypi/scikit-image/json | 2026-11-13 |
| **pyarabic** | 0.6.15 | **GPL** (PyPI license field; GitHub `linuxscout/pyarabic`: GPL-3.0) | **warn: GPL in the core install of an Apache-2.0 project. Not in the review's list; Recommendation, owner to confirm: decide whether to replace it, isolate it behind an extra, or accept GPL for distributed builds** | https://pypi.org/pypi/pyarabic/json, https://api.github.com/repos/linuxscout/pyarabic | 2026-11-13 |
| httpx | 0.28.1 | BSD-3-Clause | yes | https://pypi.org/pypi/httpx/json | 2026-11-13 |
| python-dotenv | 1.2.4 | BSD-3-Clause | yes | https://pypi.org/pypi/python-dotenv/json | 2026-11-13 |
| sqlalchemy | 2.1.8 | MIT | yes | https://pypi.org/pypi/sqlalchemy/json | 2026-11-13 |
| alembic | 1.20.1 | MIT | yes | https://pypi.org/pypi/alembic/json | 2026-11-13 |
| anthropic (SDK) | 2.8.0 | Apache-2.0 | yes (the service has its own terms) | https://pypi.org/pypi/anthropic/json | 2026-11-13 |
| fonttools | 4.68.0 | MIT | yes | https://pypi.org/pypi/fonttools/json | 2026-11-13 |

### Extras

| Component | Version | Licence | Commercial-safe | Source checked | Date |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `arabic`: mishkal | 0.4.1 | GPL (PyPI license field; GitHub `linuxscout/mishkal`: GPL-3.0) | warn: GPL; an optional extra, so a build that bundles it carries GPL obligations | https://pypi.org/pypi/mishkal/json | 2026-11-13 |
| `arabic`: camel-tools | 1.6.0 | MIT | yes (its dialect model data is downloaded by the user; its licence is not checked here) | https://pypi.org/pypi/camel-tools/json | 2026-11-13 |
| `tts-edge`: edge-tts | 7.3.1 | LGPLv3 client, plus Microsoft's service terms | warn: the client is LGPL; the Microsoft read-aloud service grants no explicit commercial licence (`edge-tts-terms` warns at publish) | https://pypi.org/pypi/edge-tts/7.3.1/json | 2026-11-13 |
| `tts-local`: piper-tts | 1.8.0 | GPL-3.0-or-later | warn: GPL; an optional extra | https://pypi.org/pypi/piper-tts/json | 2026-11-13 |
| Piper voice models | per model | each model has its own licence, declared in the channel file (`voice.piper_model_license`) | warn until declared (`piper-model-license`) | channel file | - |
| `align`: ctc-forced-aligner | 1.0.2 | the sdist's LICENSE is the "Deskpai Open Source License (DOSL) Version 1.0", not OSI-approved | documented non-commercial only; publish warns (`ctc-forced-aligner-terms`) (below) | https://pypi.org/pypi/ctc-forced-aligner/1.0.2/json (sdist LICENSE) | 2026-11-13 |
| `align`: its default model `deskpai/ctc_forced_aligner` (ONNX), fetched from `MODEL_URL` in the package | 77d85f9 | **CC-BY-NC-4.0** | documented non-commercial only; publish warns (`ctc-forced-aligner-terms`) when any scene's word times are `ALIGNED` (#166) | https://huggingface.co/api/models/deskpai/ctc_forced_aligner | 2026-11-13 |
| `align`: unidecode | 1.4.0 | GPL (GPLv2 or later) | warn: GPL; an optional extra | https://pypi.org/pypi/unidecode/json | 2026-11-13 |
| `diffusion`: torch | 2.15.0 | BSD-style; PyPI lists Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib | yes | https://pypi.org/pypi/torch/json | 2026-11-13 |
| `diffusion`: diffusers | 0.42.1 | Apache-2.0 | yes | https://pypi.org/pypi/diffusers/json | 2026-11-13 |
| `diffusion`: transformers | 5.22.2 | Apache-2.0 | yes | https://pypi.org/pypi/transformers/json | 2026-11-13 |
| `ui`: gradio | 6.31.0 | Apache-2.0 | yes | https://pypi.org/pypi/gradio/json | 2026-11-13 |

### Models and weights

| Component | Licence | Commercial-safe | Source checked | Date |
| :--- | :--- | :--- | :--- | :--- |
| FLUX.1-schnell (documented `lineart-diffusion` model, README) | Apache-2.0 (gated repository: accept the terms on Hugging Face) | yes (`Apache-2.0` passes at publish) | https://huggingface.co/api/models/black-forest-labs/FLUX.1-schnell | 2026-11-13 |
| FLUX.1-dev | FLUX.1 [dev] Non-Commercial License | no (fails at publish) | https://huggingface.co/api/models/black-forest-labs/FLUX.1-dev | 2026-11-13 |
| Stable Diffusion XL base 1.0 | OpenRAIL++-M (open-weights, use-based restrictions) | warn (`OpenRAIL++-M` warns at publish) | https://huggingface.co/api/models/stabilityai/stable-diffusion-xl-base-1.0 | 2026-11-13 |
| Line-art LoRA for FLUX.1-schnell | none ships; a commercial-safe one was not found (#161) | n/a | README | 2026-11-13 |
| CATT diacritizer (`catt-tashkeel` 1.0.2) | Apache-2.0; not a dependency, not run (#156) | yes | https://pypi.org/pypi/catt-tashkeel/json | 2026-11-13 |
| Coqui XTTS-v2 weights (`coqui/XTTS-v2`) | Coqui Public Model License, non-commercial. Not implemented (ADR 0003). | no | https://huggingface.co/api/models/coqui/XTTS-v2 | 2026-11-13 |
| F5-TTS weights (`SWivid/F5-TTS`) | CC-BY-NC-4.0, non-commercial. Not implemented (ADR 0003). | no | https://huggingface.co/api/models/SWivid/F5-TTS | 2026-11-13 |
| Farasa (Java diacritizer) | research and non-commercial. Not used (REVIEW C4). | no | REVIEW C4 | 2026-11-10 |
| Remotion | source-available, not free software; a paid company licence above a small headcount. Rejected (ADR 0002). | no | REVIEW C1 | 2026-11-10 |

### Services

| Component | Terms | Commercial-safe | Note |
| :--- | :--- | :--- | :--- |
| Microsoft read-aloud (edge-tts endpoint) | no explicit commercial licence; an undocumented endpoint that may change or be limited | warn | `edge-tts-terms` warns at publish |
| ElevenLabs | commercial use needs a paid plan | warn | `elevenlabs-terms` warns at publish |
| Anthropic, OpenAI, Voyage AI | each provider's API terms; used only when the channel selects them | per provider | `openai-terms` passes at publish |

### Bundled assets and data

| Component | Licence | Commercial-safe | Source |
| :--- | :--- | :--- | :--- |
| Amiri 1.003 | OFL-1.1 | yes | `assets/LICENSES.md` |
| Noto Sans 2.015 | OFL-1.1 | yes | `assets/LICENSES.md` |
| Tabler Icons 3.49.0 (205 icons) | MIT | yes | `assets/LICENSES.md` |
| Hand sprite and marker SFX | Apache-2.0 (the project's own work) | yes | `assets/LICENSES.md` |
| Tanzil Qur'an text (fetched, not committed) | CC-BY-3.0: verbatim only, attribute the Tanzil Project, link to tanzil.net | yes, with attribution | `assets/quran/LICENSE-tanzil.txt` |
| Reciter recordings | none shipped; common sources are non-commercial or unlicensed, so verse scenes fail publish until a licensed reciter is added | no until one is added | `assets/LICENSES.md` |
| Diacritizer gold set (fetched, not committed) | repo MIT; its text derives from Tashkeela, GPLv2, so no Arabic text is committed | n/a: benchmark data, not shipped | ADR 0005 |
| scikit-image sample images (tests only, not committed) | CC0-1.0 or public domain | yes | `tests/visuals/lineart_samples.py` |
| Uploaded images | the creator declares the licence at upload (`own-work` or an allowed licence); recorded in provenance | per upload | `sketching/upload.py` |

### Findings

- **The `align` extra is documented non-commercial only** (DOSL package licence; CC-BY-NC-4.0 default model; GPL `unidecode`); publish warns (rule id `ctc-forced-aligner-terms`) when any scene's word times came from the CTC aligner (#166). No aligner or model was swapped.
- **`pyarabic` is GPL and sits in the core dependencies** of an Apache-2.0 project. This was not in the review. Recommendation, owner to confirm: decide whether to replace it, move it behind the `arabic` extra, or accept GPL obligations for distributed builds.
- Three extras are GPL (`arabic`, `tts-local`, `align`); the matrix states the boundary in section 7.
