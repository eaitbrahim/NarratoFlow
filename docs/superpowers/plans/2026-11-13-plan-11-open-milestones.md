# Plan 11 - Open Milestones (#28, #29, #8, #30, #1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship five PRs, one per issue:
- **A** (#28): XDoG line art for uploaded images, a `sketch upload` command, and a sample-set test proving the output feeds `strokes.py` with no manual cleanup.
- **B** (#29, offline half): a `lineart-diffusion` image provider (FLUX.1-schnell plus an optional LoRA) whose pipeline is injected, per-channel model/LoRA/seed pins with licences, and a VRAM note. No model is downloaded and nothing runs on a GPU.
- **C** (#8): a pinned, checksummed gold set of 300 vocalized lines, a DER/WER scorer, a benchmark script, and ADR 0005 with the Mishkal results.
- **D** (#30): a Gradio review station built on pure, tested functions that read and write `series.json`, only the edited scene's stages and part re-run, and 9:16 and 16:9 export buttons.
- **E** (#1): PRD and SRD v1.1, which fold in REVIEW C1 to C10, answer REVIEW §5, and add a licence matrix.

**Architecture:**
- **PR A.** `visuals/lineart.py` gains `xdog_lineart` and `ink_to_png`. A new `sketching/upload.py` turns an uploaded raster into a library sketch, with origin `UPLOADED` (Alembic 0004), and attaches it to a scene whose `visual.source` is `IMAGE_TO_SKETCH`. `generate_missing` never generates for such a scene.
- **PR B.** `visuals/diffusion.py` holds `DiffusionSpec`, a `TextToImagePipeline` protocol, a lazy diffusers factory and `LineArtDiffusionProvider`, which implements `images.ImageProvider`. `StyleConfig.diffusion` holds the pins. `images/factory.py` registers the `lineart-diffusion` provider.
- **PR C.** `text/tashkeel/bench.py` (pure scoring, under coverage), `scripts/fetch_gold_tashkeel.py`, `scripts/bench_tashkeel.py`, the manifest `docs/benchmarks/tashkeel-gold.toml`, `docs/adr/0005-diacritizer-benchmark.md`.
- **PR D.**
  - `ui/station.py` holds pure functions keyed on file paths, with no in-memory state.
  - `ui/app.py` is a thin Gradio layer that takes the `gradio` module as a parameter.
  - `publish/landscape.py` pillarboxes each exported 9:16 MP4 to 1920x1080.
  - `narratoflow ui` launches the station.
- **PR E.** Docs only: `docs/PRD.md`, `docs/SRD.md`, and a status line in `docs/REVIEW.md`.

**Tech Stack:** Python 3.12, numpy, OpenCV, scikit-image, pydantic, typer, SQLAlchemy/Alembic, pytest, ffmpeg. Optional extras are already declared: `arabic` (mishkal), `diffusion`, `ui` (gradio). PR B removes `controlnet-aux` from `diffusion` (D8). No other dependency changes.

**Spec:**
- issue texts #28, #29, #8, #30, #1 (`gh issue view N`);
- `docs/REVIEW.md` (C1 to C10, §2, §3, §4, §5);
- `docs/PRD.md`, `docs/SRD.md`;
- ADRs `docs/adr/0002` to `0004`, `docs/strokes-schema.md`.

**Out of scope:**
- #133 (R2);
- every paid model call and every GPU run (R3);
- a native 16:9 layout (D31);
- choosing a LoRA by visual quality (D13);
- a per-scene segment cache inside a part (D37, withdrawn D30).

## Cold-read resolutions (D37-D46) — these SUPERSEDE any earlier text they contradict

- **D37 (B4, B3; subtracts D30 and Task D2):** the per-scene segment cache is WITHDRAWN. `render/build.py` and `render/encoder.py` are not changed by this plan. "Editing one scene re-renders only that scene" means: `generate_missing(only={S})` touches only S; `build_audio(parts=[n])` re-synthesizes only S (the TTS cache serves the rest, no provider call for them); `build_render(parts=[n])` re-renders only part n. Other parts are not passed to `build_render` at all, so tests assert their MP4 and report files are byte-identical and have unchanged mtimes, NOT that they appear in `RenderResult.skipped`. `RebuildReport` drops `redrawn` and `skipped_parts`; it carries `rendered_parts: list[int]` (== [n]). Why: the cache would change the frame-exact render core (ADR 0002) and needs encoder changes; per-part rendering already meets the criterion at part granularity (a 180 s part renders in 84.5 s). Follow-up issue filed by PR D: "Render: per-scene segment cache inside a part". Cost if wrong: slower station rebuilds for long parts.
- **D38 (B1):** composite licence ids are joined with `" AND "` (SPDX expression style), e.g. `"Apache-2.0 AND OpenRAIL++-M"`, and split on `" AND "` only. A bare id containing `+` (`OpenRAIL++-M`) is looked up whole. Test B3's cases: `"Apache-2.0"` pass, `"OpenRAIL++-M"` warn, `"Apache-2.0 AND OpenRAIL++-M"` warn, `"Apache-2.0 AND FLUX-1-dev-non-commercial"` fail.
- **D39 (B2):** the station derives paths exactly as the CLI does, through one shared helper: `StationPaths(series, channel, audio_dir, export_dir)` plus `render_dir = audio_dir` (`cli.py:570,706`) and `cache_root = series.parent / ".cache"` (`cli.py:483`). PR D moves those two defaults into named functions (e.g. `default_render_dir(audio_dir)`, `default_cache_root(series_json)`) used by both the CLI and the station; one decision, one place.
- **D40 (B5):** `resolve.regenerate` on an `IMAGE_TO_SKETCH` scene resets `visual.source` to `ICON` (the D6 escape hatch: a styled re-draw replaces the upload). `use_text_card` and `swap_idea` gain the same keyword `require_quarantine: bool = True` as `regenerate`; the station passes False. An `IMAGE_TO_SKETCH` scene whose record is missing gets `QUARANTINED_FLAG` and is listed in `result.quarantined`, like any other scene without a sketch. **Revised (ruling review):** `swap_idea` also resets `visual.source` to `ICON` when it is `IMAGE_TO_SKETCH` (it clears `sketch_id` at `resolve.py:156-163`, so the scene would otherwise be quarantined forever), and `check_regenerable` gains the same `require_quarantine: bool = True` keyword so the station's estimate accepts a non-quarantined scene.
- **D41 (B6):** `UPLOADED` records are never reusable: `find_reusable` and `idea_keys` exclude origin `UPLOADED` (edit `library/repository.py`; add it to Task A3's files). The any-style-version rule in `_attached_record` applies only when the attached record's origin is `UPLOADED` and the scene's `visual.source` is `IMAGE_TO_SKETCH`; the idea-key check is waived only in that case. `copy_to` keeps origin `UPLOADED` for uploaded records instead of relabelling them `COPIED`. **Revised (ruling review):** `idea_catalog` (`repository.py:180-195`, which feeds the planner's candidate keys via `candidates.py` and `planning/pipeline.py:156`) excludes `UPLOADED` too, so all three `_REUSABLE` queries agree.
- **D42 (N1, N2):** `_provenance` reads the licences from `channel.style.diffusion` when the provider is `lineart-diffusion`. `DiffusionSpec.model` is `style.image_model`. `DiffusionStyle` gains `device: Literal["auto","cpu","cuda","mps"] = "auto"`; no CPU offload. The field count in D10 is whatever D10 lists plus `device`.
- **D43 (N3, N4, N5):** size checks: first decode (PIL `UnidentifiedImageError` and any decode error -> `SketchRejectedError("undecodable image")`), then long side > `MAX_UPLOAD_SIDE_PX` -> "too large", then short side < `MIN_UPLOAD_SIDE_PX` -> "too small" (so 9000x10 is "too large"). `--license` is validated against `[licences.assets]` at upload time AND at publish. Upload provenance has `provider=None`. `attach_upload` requires `require_series_channel` like `sketch generate`, does not require `series.approved`, refuses `TEXT_CARD` scenes, and uses `channel.style.line_weight_px` for `check_density`. `GenerationResult` gains `uploaded: list[str]` for `_print_generation`.
- **D44 (N7, N8):** `test_committed_manifest_is_well_formed` is written in C2, not C1. Marks U+0653-U+065F count as diacritics exactly as `strip_diacritics` defines them. In WER without case endings, a one-letter word compares as an empty mark sequence (always correct). PRD §8 answer 3 covers "Qur'an or hadith": hadith text is ordinary narration (no protected handling) — recommendation, owner to confirm.
- **D45 (N9, N10, N11, N12, N13):** every station handler that writes `series.json` (edit_text, edit_timing, upload_visual, regenerate_visual, use_text_card, rebuild_scene) takes `expected_sha256` and returns the new sha256 with its result. The layout has 8 buttons; "9" is a typo. Export 16:9 does not require `LANDSCAPE` in `channel.outputs` and is not recorded in the publish report (it is a derivative file, listed in the station's result only). The voice estimate uses a public `audio.build.estimate_voice_cost(series, channel, settings, parts) -> Decimal` extracted from the code around `uncached_chars`. PR B bumps both the rules `version` and `checked` and updates every pinned test.
- **D46 (N14, N15):** commit trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; PR bodies end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. PR E may use read-only network lookups (`curl`, `gh api`) to re-check licences.

## Decisions

### Shared

- **D1: "storyboard.json" in #30 means `series.json`.** ADR 0004 made `Series` (schema v2) the IR; `Project`/`storyboard.json` only serves `validate`. The station edits `series.json` through `Series.load`/`Series.save` (atomic). Cost if wrong: none; there is no other editable IR.
- **D35: Follow-up issues.** Each is filed by the PR named, with `gh issue create`, labelled like its parent and in the parent's milestone. Its number is written into the doc the PR ships:
  1. (C) "Diacritizer benchmark: LLM-engine row (paid)": run `bench_tashkeel.py --engines llm` once with a budget, then update ADR 0005.
  2. (C) "Diacritizer benchmark: CATT row on Linux x86_64".
  3. (B) "Diffusion: real-model determinism and VRAM measurement on a GPU".
  4. (B) "Diffusion: pick and verify a line-art LoRA for FLUX.1-schnell".
  5. (D) "Native 16:9 layout (panorama) to replace the pillarbox export".
  6. (E) "Alignment extra uses a CC-BY-NC-4.0 model: find a commercial-safe aligner".
- **D36: PR order.** See `## PR order`. A and C first, in parallel. B and D after A merges, in parallel. E last.

### PR A (#28)

- **D2: #28 public API, in `src/narratoflow/visuals/lineart.py`.**
  - `xdog_lineart(raster: bytes, *, params: XdogParams = XDOG_DEFAULT) -> np.ndarray`. It returns the same type as `binarize`: a 2-D `bool` ink mask (True = ink), with specks dropped by `_drop_specks`. It raises `SketchRejectedError("empty")` when no ink is left.
  - `ink_to_png(ink: np.ndarray) -> bytes` writes an 8-bit grayscale PNG ("L"): ink 0, paper 255 (#FFFFFF).
  - `binarize(ink_to_png(m))` equals `m` for every mask `m` that `xdog_lineart` returns (checked on the sample set).
  - Why: the existing gates (`check_density`, `extract_strokes`, the library) already consume exactly this mask and PNG, so the uploaded path reuses them unchanged. Cost if wrong: a wrapper function.
- **D3: XDoG parameters, measured in this planning session on the D4 set with `check_density` and `extract_strokes`.**
  - **Decode:** `_to_gray` (alpha over white, 16-bit scaled).
  - **Resize:** the long side to `XDOG_LONG_SIDE_PX = 512` (`cv2.INTER_AREA` when shrinking, `cv2.INTER_CUBIC` when growing).
  - **Contrast:** stretch to the 1st to 99th percentile (`XDOG_PERCENTILES = (1.0, 99.0)`), clipped to [0, 1]; a flat image (hi - lo < 1 grey level) gives no ink.
  - **Filter:** `D = G(sigma) - tau * G(k * sigma)`, with `T = 1` where `D >= eps`, else `1 + tanh(phi * (D - eps))`. Ink is where `T < 0.5`.
  - **Values:** `XdogParams(sigma=1.5, k=1.6, tau=1.0, eps=-0.03, phi=200.0)`.
  - **Size limits:** `MIN_UPLOAD_SIDE_PX = 32`. `MAX_UPLOAD_SIDE_PX = 8192` is read from the header with PIL before decoding (decompression-bomb guard). Either violation raises `SketchRejectedError("too small"/"too large")`.
  - **Measured at these values (strokes / ink ratio / mean stroke width):** camera 127 / 0.040 / 3.3, horse 9 / 0.050 / 4.4, coffee 91 / 0.025 / 3.0, chelsea 244 / 0.038 / 3.4, astronaut 278 / 0.068 / 3.1, rocket 123 / 0.018 / 2.7, text 84 / 0.106 / 3.6. All passed `check_density`.
  - **Alternative rejected:** the classic `(1+p)G1 - pG2` tone form at eps 0.6. It filled dark regions: ink ratio 0.43 to 0.94, and all 8 images failed density.
  - Cost if wrong: retuning constants; the D5 test pins the outcome, not the constants.
- **D4: The sample set is 7 images bundled inside the installed scikit-image 0.26 wheel, which is already a runtime dependency.**
  - The images and their licences, as given in the docstrings of `skimage/data/_fetchers.py`:
    - `camera`: CC0 (Lav Varshney);
    - `horse`: CC0 (openclipart, Andreas Preuss);
    - `coffee`: CC0 (Rachel Michetti);
    - `chelsea`: CC0 (Stefan van der Walt);
    - `astronaut`: public domain (NASA);
    - `rocket`: public domain (SpaceX release);
    - `text`: public domain (Wikipedia Corner.png).
  - Images left out:
    - `clock`: public domain, but low-contrast motion blur gave only 4 strokes;
    - `coins`: "no known copyright restrictions" is not an explicit grant;
    - `logo`, `moon`, `page`: no licence stated.
  - No image binaries are committed. Tests call `skimage.data.<name>()` and encode to PNG in memory. `tests/visuals/lineart_samples.py` holds the list with licence and source per image.
  - Cost if wrong: a future scikit-image drops an image, and the test then fails loudly on a missing attribute.
- **D5: "Feeds strokes.py without manual cleanup" is this test** (`test_xdog_sample_set_feeds_strokes`, parametrized over D4). For each image, `png = ink_to_png(xdog_lineart(raw))`:
  1. The PNG is mode "L" with pixel values a subset of {0, 255}; pixel (0, 0) is 255 for camera.
  2. `binarize(png)` equals the mask.
  3. `check_density(mask).ok` is True.
  4. `extract_strokes(mask)` gives `1 <= len(strokes) <= PACING_STROKES` (400).
  5. `StrokeSet.from_json(s.to_json()) == s`.
  6. Running it twice gives byte-identical PNGs.

  No hand step exists between steps 1 and 4. Cost if wrong: a busy user photo can still fail density; that is refused with the reason (D6), not silently cleaned.
- **D6: How an upload plugs into the scene flow.**
  - **Command:** `narratoflow sketch upload SERIES_JSON SCENE_ID IMAGE --channel C --license ID --source TEXT [--attribution TEXT]`. It runs `sketching.upload.attach_upload`. It is free: no vision call, no spend.
  - **Storage:**
    - The line-art PNG (not the original upload) is stored with `library.add_sketch`, under the scene's idea key and the channel's current style version;
    - origin `SketchOrigin.UPLOADED`, status `UNCHECKED` (reason `"uploaded: vision not run"`), `mirror_safe=False`;
    - its strokes are stored with `rtl` from the channel language.
  - **Scene update:** `visual.source = IMAGE_TO_SKETCH`, `sketch_id`, `match="new"`, and the quarantine flag is cleared.
  - **Refusals, nothing stored:**
    - a density failure or decode failure raises `SketchRejectedError` with the reason;
    - a verse scene, a scene with no idea key, or an unknown scene id raises `PlanningError`.
  - **`generate_missing`:**
    - An `IMAGE_TO_SKETCH` scene is never generated.
    - With a usable attached record it is kept. `_attached_record` accepts an `UPLOADED` record from any style version, since a creator's image is not styled.
    - Without one, it is listed in `result.quarantined`. The CLI prints the hint `narratoflow sketch upload ...`.
  - **No version bumps:** `Series` has no schema change, because the field and enum value already exist. `RENDER_INPUTS_VERSION` is unchanged: a new blob already changes `inputs_key`.
  - Cost if wrong: a creator who wants a styled re-draw of their image must use regenerate; that is a follow-up, not a defect.
- **D7: Upload licences.**
  - `--license` is required. It must pass the publish check of `[licences.assets]`, because an `UPLOADED` record is not `is_generated`, so it is checked as an "icon".
  - `platform_rules.toml` gains `"own-work" = { verdict = "pass", note = "the creator declares they made the image" }`, and `version` becomes `"2026-11-13.1"` with `checked = 2026-11-13`. The pins in `tests/publish/test_rules.py` and `tests/publish/test_provenance.py` are updated.
  - `Provenance` gains `attribution: str = ""`. It is additive JSON with no migration, and is carried into `AssetRecord.attribution` by `_sketch_records`, so a CC-BY-4.0 upload passes only with an attribution.
  - Cost if wrong: one rules line.
- **D6a: Alembic `0004_uploaded_origin`.** It rebuilds `ck_sketch_origin` to `origin IN ('generated','bundled','copied','uploaded')` with `batch_alter_table`, the same pattern as 0002. Downgrade raises `RuntimeError("cannot downgrade: N uploaded sketches exist")` when any row has origin `uploaded`, and otherwise restores the 3-value CHECK.
- **D8: `controlnet_aux` is not used.** Its line-art annotators need torch and a weight download (`lllyasviel/Annotators`), and XDoG meets D5 on CPU with no weights. PR B removes `controlnet-aux` from the `diffusion` extra (REVIEW C10: no unused dependencies) and runs `uv lock`. Cost if wrong: re-adding one line when an annotator path is wanted.

### PR B (#29)

- **D9: The provider shape.**
  - `visuals/diffusion.py` defines:
    - `class TextToImagePipeline(Protocol)`: `__call__(self, *, prompt: str, negative_prompt: str | None, steps: int, guidance_scale: float, width: int, height: int, seed: int) -> PIL.Image.Image`;
    - `PipelineFactory = Callable[[DiffusionSpec], TextToImagePipeline]`;
    - `default_pipeline_factory`, which imports `torch` and `diffusers` lazily;
    - `LineArtDiffusionProvider(spec: DiffusionSpec, factory: PipelineFactory = default_pipeline_factory)`, with `name = "lineart-diffusion"` and `model = spec.model`.
  - The pipeline is built on the first `generate` call, not in `__init__`.
  - `generate` ignores `req.reference_png` (REVIEW C8: no ControlNet on the text path). Style comes from LoRA plus seed.
  - `LocalDiffusionProvider` (img2img, `local-diffusion`) is unchanged.
  - Tests inject a fake factory and never import torch.
  - Cost if wrong: two diffusion providers to document; the README says which to use.
- **D10: Where the pins live.** `StyleConfig.diffusion: DiffusionStyle | None = None`. The channel is NarratoFlow's per-project config, and it already holds `image_provider`/`image_model`. Fields (`extra="forbid"`):
  - `model_license: str`, `model_source: str`, `model_revision: str | None = None`;
  - `lora: str | None = None`, `lora_weight_name: str | None = None`, `lora_revision: str | None = None`;
  - `lora_license: str | None = None`, `lora_source: str | None = None`;
  - `lora_scale: float = Field(1.0, gt=0, le=2)`;
  - `seed: int = Field(0, ge=0, lt=2**32)`;
  - `steps: int = Field(4, ge=1, le=100)`;
  - `guidance_scale: float = Field(0.0, ge=0, le=20)`;
  - `size: Literal[1024] = 1024`.

  Validators:
  - `lora` set needs `lora_license` and `lora_source`;
  - `ChannelConfig` refuses `image_provider = "lineart-diffusion"` without `style.diffusion`.

  Schema rules:
  - Channel TOML is not schema-versioned. `Series` is untouched, so `SCHEMA_VERSION` stays 2.
  - `RENDER_INPUTS_VERSION` stays "2", because the renderer is unchanged and a new image's blob hash already changes the part's `inputs_key`.

  Cost if wrong: moving six fields to `series.json` later, with a schema bump.
- **D11: `style_fingerprint` adds `"diffusion": style.diffusion.model_dump(mode="json")` only when `style.diffusion is not None`.** So every existing channel keeps its fingerprint and its library style version. Test: `test_fingerprint_unchanged_without_diffusion` pins the current value for the `channel_file` fixture, computed before the change and written as a literal. A changed seed, LoRA or scale gives a new style version, which is intended: style consistency.
- **D12: Default model `black-forest-labs/FLUX.1-schnell`, chosen on licence grounds.**
  - **Licence:** Apache-2.0, per the HF API cardData. The pinned revision is `741f7c3ce8b383c54771c7003378a50191e9efe9`. HF marks the repo gated (`auto`), so a user must accept the terms on HF before a download works. The README says so.
  - **SDXL base 1.0, not chosen:** CreativeML OpenRAIL++-M is open-weights with use-based restrictions and not OSI (REVIEW C8). It is still allowed via config, with a `warn` verdict.
  - **FLUX.1-dev:** non-commercial, so it FAILs at publish. Its licence id is not in `[licences.generated]`.
  - The README example and SRD use schnell with `steps = 4` and `guidance_scale = 0.0`, its distilled settings.
  - Cost if wrong: one config default; the code is model-agnostic.
- **D13: No default LoRA.**
  - On 2026-11-13 the HF API listed one FLUX.1-schnell adapter matching "sketch" (`minpeter/soldier-sketch-schnell`, Apache-2.0, a subject LoRA, not line art). No line-art LoRA was found for schnell. Most FLUX LoRAs are trained on FLUX.1-dev, whose licence is non-commercial.
  - The code fully supports a LoRA (`load_lora_weights(lora, weight_name=..., revision=...)`, then `fuse_lora(lora_scale=...)`), with mandatory licence and source.
  - Picking one needs a GPU look at outputs (R3), which is follow-up D35.4.
  - The issue's "+ line-art LoRA" is met as "supported and licence-recorded"; the PR body says so.
  - Cost if wrong: prompt-only line art is weaker until D35.4.
- **D14: Determinism.**
  - The effective seed is `int.from_bytes(sha256(f"{spec.seed}:{req.seed or 0}".encode()).digest()[:4], "big")`.
    - `req.seed` is already `marker_seed(video_id, scene.id, attempt)` from `generate.py`.
    - So the same channel seed, video, scene and attempt give the same seed, and a regenerate changes it.
  - `default_pipeline_factory` passes `generator=torch.Generator("cpu").manual_seed(seed)`: a CPU generator, so seeds reproduce across devices (diffusers guidance).
  - It passes `negative_prompt` only when the pipeline's `__call__` signature has that parameter (FLUX has none in older diffusers).
  - The output PNG is re-encoded with Pillow without metadata (`pnginfo=None`), so equal pixels give equal bytes.
  - The offline proof uses a fake pipeline whose image is a pure function of the seed and the prompt. Real-model bit-equality is follow-up D35.3.
  - Cost if wrong: a GPU kernel's non-determinism shows up only in D35.3.
- **D15: Licences in provenance.**
  - `_provenance` in `sketching/generate.py` gets `license` = `model_license`, or `f"{model_license}+{lora_license}"` when a LoRA is set.
  - `usage_terms` = "Model <model>@<revision> (<model_license>, <model_source>); LoRA <lora> (<lora_license>, <lora_source>)".
  - `provider = "lineart-diffusion"`, `model = spec.model`.
  - `publish/provenance.py` (`_record_findings`): for a generated record, the licence string is split on `"+"` and each part is looked up in `[licences.generated]`. The worst verdict wins: a missing id FAILs naming it; otherwise warn beats pass.
  - `platform_rules.toml` `[licences.generated]` gains:
    - `"Apache-2.0" = pass`;
    - `"MIT" = pass`;
    - `"OpenRAIL++-M" = warn` ("open-weights with use-based restrictions; check them");
    - `version = "2026-11-13.2"`.
  - Cost if wrong: one more licence id to add.
- **D16: When the extra is absent.**
  - `default_pipeline_factory` raises `ExtraMissingError("install the 'diffusion' extra: uv sync --extra diffusion")` on `ImportError` of `torch` or `diffusers`.
  - Because `build_image_provider` is lazy (`generate_missing` builds it on the first new sketch) and the pipeline is built on first `generate`, a run that reuses, bundles, uploads or uses text cards never imports torch.
  - `"lineart-diffusion"` joins `FREE_PROVIDERS` in `channels/keys.py`: no key, price 0.
  - Test `test_default_path_never_imports_diffusers` runs `generate_missing` on a bundled-icon series and asserts `"diffusers" not in sys.modules` and `"torch" not in sys.modules`.
  - Cost if wrong: none.
- **D17: The VRAM requirement is documented, not measured (R3).**
  - The README section "Text-to-sketch (diffusion)" and SRD §4 give these figures, sourced from the model cards and the diffusers memory docs. Each is labelled "not measured; see #<D35.3>":
    - FLUX.1-schnell is about 12B parameters, and its bf16 weights plus T5-XXL are over 30 GB on disk;
    - fully on GPU it needs about 24 GB or more of VRAM;
    - `enable_model_cpu_offload()` lowers the peak;
    - SDXL fp16 is about 7 GB, with 8 GB VRAM typical (6 GB tight, REVIEW C8).
  - **"CPU path still works without it":** with no GPU and no `diffusion` extra, every default stage runs unchanged: bundled icons, library reuse, uploads (PR A), text cards, OpenAI images. Only a channel that selects `lineart-diffusion` needs the extra. `device="auto"` falls back to CPU, which is documented as functional but impractically slow.

### PR C (#8)

- **D18: Gold-set source and licence.**
  - **Source:** `dataset/test.txt` from github.com/AliOsm/arabic-text-diacritization (Fadel et al. 2019) at commit `e083579c03e489936200b69d792fd426bb576743`. URL: `https://raw.githubusercontent.com/AliOsm/arabic-text-diacritization/e083579c03e489936200b69d792fd426bb576743/dataset/test.txt`. sha256 `4e851ff836f0a178abb15d9f4a8bcf92748b77cc0d67030d9fae2fbe698baa12`, 2500 lines, 1,747,544 bytes (all verified in this session).
  - **Licence:**
    - The repository is MIT (GitHub API, LICENSE file).
    - The text is derived from the Tashkeela corpus (Zerrouki and Balla 2017). SourceForge and the HF loader list Tashkeela as GPLv2, while the TFDS catalog lists "no known license". Its texts come from the Shamela library.
    - This project is Apache-2.0. That layering is not clearly redistributable inside the repo.
  - **So the repo commits no Arabic text:** only `scripts/fetch_gold_tashkeel.py` and the manifest `docs/benchmarks/tashkeel-gold.toml` (source URL, commit, file sha256, licence notes, selection rule, and line numbers with per-line sha256).
  - Cost if wrong: a contributor needs network once (about 1.7 MB).
- **D19: The gold subset.**
  - **Rule:** the first 300 lines of `test.txt`, in file order, with 5 to 60 whitespace-separated tokens. Line numbers are 1-based.
  - **Measured:** 1,747 lines qualify. The 300th is line 427. The subset is 8,121 words, and 98 lines contain digits or Latin characters.
  - **Unit:** a line (Fadel's distributed segment) is the "sentence" unit. The ADR says lines may hold more than one sentence. 300 is at least 200.
  - Cost if wrong: re-select with a manifest regeneration (`--make-manifest`).
- **D20: DER/WER definitions (`text/tashkeel/bench.py`), following Fadel et al.'s `diacritization_stat.py` conventions.**
  - **Letters:** U+0621 to U+063A and U+0641 to U+064A. Tatweel and everything else (digits, Latin, punctuation, whitespace) are excluded from numerator and denominator.
  - **Class:** each letter's diacritic class is the sorted string of the harakat U+064B to U+0652 that follow it, so shadda plus fatha in either order is one class. U+0670, tatweel and the Qur'anic marks U+06D6 to U+06ED are stripped from both sides before scoring.
  - **Gold letters with no harakat count**, with class "", as in Fadel's main setting.
  - **DER:** wrong letters over scored letters, times 100.
  - **WER:** a word is a whitespace token with at least one Arabic letter; it is wrong when any scored letter in it is wrong.
  - **Two variants each:** `with_case_ending` scores every letter; `without_case_ending` drops each word's last Arabic letter from scoring.
  - **Engine input** is `strip_diacritics(gold_line)`.
  - Cost if wrong: numbers not comparable with the paper; the ADR states the definitions.
- **D21: An output whose base letters differ from the gold's is "rejected".** It is detected by the same neutral-insensitive base sequence `_transfer_marks` uses, or by `SkeletonMismatchError`/`ProviderError` from the engine. All its scored letters and words count as wrong, and it is counted in `rejected_lines`. Why: production would reject that output (`vocalize_scene` falls back to the undiacritized source), so it scores as undiacritized or worse. Cost if wrong: a stricter DER than the paper's.
- **D22: Engines.**
  - `identity`: returns the input; a free floor.
  - `mishkal`: `MishkalDiacritizer`, run offline on CPU with `uv run --with mishkal==0.4.1`. It took 5.9 s on the 300 lines in this session.
  - `llm`: listed as "not run (paid)" (R3), with follow-up D35.1.
  - `catt`: listed as "not run". Reasons:
    - `catt-tashkeel` 1.0.2 (Apache-2.0, abjadai/catt) requires `onnxruntime-gpu>=1.22.0`, which has wheels only for manylinux x86_64/aarch64 and win_amd64; `uv pip install --dry-run catt-tashkeel` was unsatisfiable on this macOS arm64 machine;
    - its ONNX weights are a 72 to 86 MB release download;
    - its training data may overlap Tashkeela, so the score may be contaminated.

    Follow-up D35.2.
  - The script prints a row for every engine, with the not-run reason in place of numbers.
- **D23: Code location.** Scoring and gold loading live in `src/narratoflow/text/tashkeel/bench.py`, which is under the coverage gate. `scripts/fetch_gold_tashkeel.py` and `scripts/bench_tashkeel.py` are thin argparse wrappers, the same as `bench_render.py`. The downloaded file goes to `.cache/bench/tashkeel/test.txt`, which is git-ignored. Results go to `docs/benchmarks/2026-11-13-tashkeel-results.json`, which holds only numbers.
- **D24: ADR 0005 "Diacritizer benchmark".** Status "Accepted (provisional), 2026-11-13". Sections:
  - Context: REVIEW C9 and §2.1; `ChannelConfig.diacritizer` defaults to `"llm"`.
  - Gold set: D18 and D19.
  - Metrics: D20 and D21.
  - Results: a table with columns engine, lines, rejected, DER, DER (no case ending), WER, WER (no case ending), and runtime. It holds the identity and mishkal rows, plus llm and catt as "not run" with reasons and issue numbers.
  - Decision: the default stays `llm`. No row can show `llm` worse, because it was not run. Mishkal is the measured free and offline option, at its measured DER. Revisit when D35.1 lands.
  - Consequences: the skeleton guard stays mandatory for every engine.

  ADR 0001's list gains 0005. Cost if wrong: the default is revisited in D35.1, as planned.

### PR D (#30)

- **D28: Station scope, as pure functions in `ui/station.py`.** Each one takes `StationPaths(series: Path, channel: Path, audio_dir: Path, export_dir: Path)`, and reloads from disk on every call.
  - **Read:** `list_scenes`, and `scene_view(paths, scene_id) -> SceneView`. A `SceneView` holds `source_text`, `tts_text`, `caption_text`, `min_duration_ms`, `draw_fraction`, the sketch PNG bytes and the audio clip path (cut from the part FLAC by the timeline slot into `audio_dir/.ui/<sha>.wav`, a derived cache, not state).
  - **Edit:**
    - `edit_text(paths, scene_id, *, tts_text, caption_text, expected_sha256)`. Validation uses `audio.build.check_scene_text`, which is `_check_texts` made public per scene. So tashkeel must keep the skeleton and captions must keep the word count. `source_text` is read-only in the station: the author's words.
    - `edit_timing(paths, scene_id, *, min_duration_ms: int (1000..60000), draw_fraction: float (0.1..1.0), expected_sha256)`.
  - **Visual:**
    - `upload_visual(...)` calls PR A's `attach_upload`.
    - `regenerate_visual(..., confirm: bool)` calls `resolve.regenerate(..., require_quarantine=False)`, with the cost from `resolve.regenerate_estimate`, which moves from `cli._regenerate_estimate`.
    - `use_text_card(...)`.
  - **Pipeline:**
    - `rebuild_scene(paths, scene_id, *, confirm: bool) -> RebuildReport` runs `generate_missing(only={scene_id})`, then `build_audio(parts=[n])`, then `build_render(parts=[n])`.
    - `export(paths, ratio: AspectRatio) -> list[Path]` runs `build_render` on all parts (unchanged parts are kept), then `build_publish`, and for `LANDSCAPE` then `pillarbox_export`.
  - **The UI holds no state:** every handler reads `series.json` and writes it back through `Series.save`. The only value a page keeps is the file's sha256 at display time (D33).
- **D29: What "editing one scene re-renders only that scene" means.** For an edit to scene S in part n:
  1. `generate_missing` touches S only.
  2. `build_audio` rebuilds part n only; the TTS cache serves every other scene (no provider call).
  3. `build_render(parts=[n])` re-encodes part n only (D37: other parts are not passed and their files stay byte-identical).
  4. (WITHDRAWN by D37) Inside part n, only scenes whose segment inputs changed are redrawn. That is S alone when S's slot is unchanged (a sketch, text-card or draw-fraction edit). A change to S's duration also redraws the later scenes of part n, because their frame slots moved.

  The final libass/x264 pass of part n always runs once, because captions are burned over the whole part (ADR 0002). Cost if wrong: an owner who wanted per-scene MP4 patching; that would need a re-encode-free splice, which x264 does not offer.
- **D30 — WITHDRAWN by D37. Do not build.** (Original text kept for the record.) The segment cache (`render/build.py`).
  - **Key:** `segment_key = cache_key("segment", RENDER_INPUTS_VERSION, payload)`. The payload is every `SceneJob` field except `segment` and `ffmpeg`: `frames` as `[start, stop]`, the strokes as their `to_json()` sha256, the variation `model_dump`, the hand id and sha256, the canvas scale, `tools.version` and `tools.fingerprint`.
  - **Files:** `out_dir/.segments/part-NN/<key>.mkv`, plus `<key>.frames` holding the frame count.
  - **Reuse:** a segment is reused when both files exist and `.frames` equals `len(job.frames)`.
  - **Miss:** a missing segment renders to `<key>.mkv.part`. It is renamed after `_checked` passes, and then `.frames` is written.
  - **Calls:** `_run_jobs` is called only with the misses, and not at all when there are none (`ProcessPoolExecutor(max_workers=0)` would raise).
  - **Pruning:** after a successful part, files whose key is not in the current job set are deleted.
  - **Failure:** when the encode or verify fails, `out_dir/.segments/part-NN/` is deleted, so the retry renders fresh.
  - **Force:** `force=True` ignores and rewrites the cache.
  - **Result:** `RenderResult.redrawn: dict[int, list[str]]` lists the scene ids actually drawn per part.
  - **Versions:** frames and encode are unchanged, so `RENDER_INPUTS_VERSION` stays "2". The test `test_cached_segments_give_identical_frames` proves the output is unchanged.
  - Cost if wrong: disk use (FFV1 is lossless); pruning bounds it to one copy of the current part.
- **D31: The 16:9 export is a pillarbox derivative.**
  - The compositor renders 9:16 only (ADR 0002, `LANDSCAPE_WARNING`), and a native panorama layout (REVIEW §2.4) is a compositor project.
  - So "Export 16:9" first runs the normal publish check and export, then `publish/landscape.py: pillarbox_export(tools, src: Path, out: Path) -> Path`:
    - filter `scale=-2:1080:flags=lanczos,pad=1920:1080:(ow-iw)/2:0:color=white`;
    - `X264_ARGS`, audio copied;
    - output `export_dir/part-NN.16x9.mp4`, written to a temp name and renamed after `verify_mp4(size=(1920, 1080), frames=<the 9:16 report's frames>, duration_ms=<same>)`.
  - Only parts the publish check exported get a 16:9 file. A FAILed part gets none.
  - The `LANDSCAPE_WARNING` text stays in render reports, and the station shows "16:9 is a pillarboxed copy of the 9:16 video".
  - Follow-up D35.5.
  - Cost if wrong: the owner wanted the panorama layout; the button and file name stay, and only the producer changes.
- **D32: gradio in CI.**
  - CI runs `uv sync --locked --group dev` with no extras (`.github/workflows/ci.yml`), and gradio is not importable in the local venv either (checked). `gradio` 6.31.0 is in `uv.lock` under the `ui` extra.
  - So `ui/app.py: build_app(paths: StationPaths, settings: Settings, gr: Any) -> Any` takes the gradio module as an argument. `narratoflow ui` imports gradio lazily, raising `ExtraMissingError("install the 'ui' extra: uv sync --extra ui")`.
  - Tests:
    - CI tests pass `FakeGradio` (`tests/fakes/gradio.py`), which records components and `.click(fn, inputs, outputs)` wiring and invokes the handlers;
    - one `pytest.importorskip("gradio")` test builds the real Blocks without `launch`.
  - No test needs a browser.
  - Cost if wrong: drift between the fake and gradio 6; the importorskip test catches it locally.
- **D33: A stale-write guard.** Every edit handler takes `expected_sha256`, the sha256 of `series.json` when the view was rendered. If the file changed since, it raises `PlanningError("series.json changed on disk since this view was loaded; reload")` and writes nothing. Why: two tabs or the CLI would otherwise overwrite each other, because nothing is held in memory. Cost if wrong: an extra reload click.
- **D34: Paid actions in the station.** `rebuild_scene` and `regenerate_visual` compute the estimate before spending: the image plus vision price via `regenerate_estimate`, and the voice price for `uncached_chars`. When it is above 0 and `confirm` is False, they raise `PlanningError("estimated cost $X; confirm to continue")` before any call. The Gradio layer shows the estimate and a confirm checkbox. Cost if wrong: one extra click on a free path that is never hit (estimate 0).

### PR E (#1)

- **D25: How the docs are changed.**
  - `docs/PRD.md` and `docs/SRD.md` are rewritten in place as v1.1, with the status line "v1.1, 2026-11-13; supersedes v1.0 (see git history)" and a "Changes from v1.0" table mapping each C1 to C10 to its section.
  - `docs/REVIEW.md` gains one status line after the verdict: "Corrections C1–C10 folded into PRD/SRD v1.1 (#1); §5 answered in PRD §8."
  - The PRD keeps product content only; code-level material moves to the SRD (REVIEW §4).
- **D26: REVIEW §5 answers.** They go in PRD §8, each with its evidence:
  1. **Commercial use.** The code decides that output is meant for monetized platforms: `platform_rules.toml` warns on `edge-tts-terms` ("no explicit commercial licence"), `elevenlabs-terms` ("commercial use needs a paid ElevenLabs plan") and `local-model-license`; the project is Apache-2.0 (`pyproject.toml`). XTTS-v2, F5-TTS and Farasa have no code (ADR 0003 alternatives), and FLUX.1-dev FAILs (D12). The answer: **commercial-safe defaults are a requirement. Recommendation, owner to confirm.**
  2. **Offline.** The code decides: **no, offline is not a hard v1 requirement**. The default fallback voice is edge, which needs the network (ADR 0003); the default LLM provider is `anthropic`; the verse audio fetch needs the network (`audio/build.py` docstring). An offline-capable path exists, listed with its gaps: Piper, Mishkal, Ollama, local diffusion and the `.cache` TTS cache.
  3. **Qur'an content.** The code decides: **yes**. Verse scenes (`SceneKind.VERSE`, `VerseRef`), `Narration.protected`, the Tanzil corpus (`quran/corpus.py`, `scripts/fetch_quran.py`), the reciter allow-list (`quran/reciters.py`) and the verse render checks (`render/build.py` `_check_verse_pages`) are built.
  4. **Target hardware.** The code decides **CPU-only for the default path**: ADR 0002 (no GPU path; 180 s part in 84.5 s with default workers on a 16-core arm machine, against a 600 s NFR). A GPU is needed only for `lineart-diffusion`/`local-diffusion` (D17). **The minimum core count and RAM are not decided by code. Recommendation, owner to confirm:** a 4-core CPU with 8 GB RAM and ffmpeg built with libass, with the NFR re-measured there.
- **D27: Licence-matrix findings.** These were verified in this session from PyPI, GitHub and the HF APIs, and the matrix must carry them:
  - `mishkal` 0.4.1: GPL (PyPI license field);
  - `piper-tts` 1.8.0: GPL-3.0-or-later;
  - `edge-tts` 7.3.1: LGPLv3 client plus Microsoft service terms;
  - `ctc-forced-aligner` 1.0.2: PyPI sdist LICENSE "Deskpai Open Source License (DOSL) 1.0", not OSI. Its default `MODEL_URL` (`deskpai/ctc_forced_aligner` ONNX) is CC-BY-NC-4.0 on HF, so the `align` extra is **not commercial-safe**. This gets follow-up D35.6 and a WARN line in the matrix; no code change in this plan;
  - camel-tools: MIT;
  - gradio, diffusers, transformers and opencv-python-headless: Apache-2.0;
  - torch: Apache-2.0 (with the LLVM exception);
  - scikit-image: BSD;
  - FLUX.1-schnell: Apache-2.0; FLUX.1-dev: non-commercial; SDXL: OpenRAIL++-M;
  - CATT: Apache-2.0; the Fadel benchmark repo: MIT; Tashkeela: GPLv2 (listed);
  - XTTS-v2 weights: CPML (non-commercial); F5-TTS weights: CC-BY-NC; Farasa: research, non-commercial; Remotion: source-available (REVIEW C1, C3, C4).

  The extras are optional, so the matrix states that distributing a build that bundles a GPL extra brings GPL obligations. Each row cites the URL it was checked at and the date. Cost if wrong: a row the PR's verification step corrects.

## Global Constraints

- **R2:** #133 is out of scope. Nothing in this plan edits `audio/narrate.py` timing logic, `align/ctc.py` or `audio/words.py`.
- **R3:** no paid model calls and no GPU runs. #8's LLM row is "not run (paid)", with a follow-up issue (D35.1). #29's real-model determinism check is a follow-up issue (D35.3). #8 and #29 close on offline evidence: fakes, a CPU-only Mishkal run and injected pipelines.
- **R4:** #1's §5 answers come from what code and ADRs already decide. Anything the code does not decide is written as "Recommendation, owner to confirm" (D26).
- **Network:** downloads are allowed in the build when licences are verified and recorded: the gold set (D18), `uv run --with mishkal==0.4.1`, and `uv lock` in PR B. Heavy model weights (diffusion, multi-GB) are never downloaded; tests never touch the network.
- `.py` and `.toml` sources are pure ASCII; Arabic literals use `\u` escapes.
- mypy strict and Ruff clean, with no `# type: ignore` added except the existing `skimage`/`cv2` stub pattern (`[no-untyped-call]`, as in `lineart.py:314`).
- Tests never use the network and never sleep for real. An ffmpeg-dependent test uses the existing skip-locally, fail-in-CI fixtures (`ffmpeg_bin`, `render_tools`). An extra-dependent test uses `pytest.importorskip`, and its module is covered by fake-module tests (the pattern of `test_mishkal_outer_whitespace_restored`).
- `NarratoFlowError` is caught only in `cli.main`.
- `Series.load`/`Series.save` are the only readers and writers of `series.json`. No new field means no `SCHEMA_VERSION` bump. `RENDER_INPUTS_VERSION` stays "2" in every PR (D6, D10, D37).
- Commits stage files by name; never `git add -A`. One PR per issue, branch names as given per PR. PR bodies end with `Closes #N` and the attribution line from the session.
- Every code-writing agent follows the ub-mode skill: failing test first, then the smallest edit.
- **Gate (every PR):** `uv lock --check && uv run ruff check --no-cache . && uv run ruff format --check . && uv run mypy && uv run coverage run -m pytest -o addopts="" -q && uv run coverage report`. The coverage TOTAL stays at or above 97%.
- No command runs longer than about 5 minutes in the foreground; the Mishkal benchmark took 6 s.

## PR order

| Wave | PR | Issue | Branch | Depends on | Parallel with |
|---|---|---|---|---|---|
| 1 | A | #28 | `plan-11-a-image-to-sketch` | none | C |
| 1 | C | #8 | `plan-11-c-tashkeel-bench` | none | A |
| 2 | B | #29 | `plan-11-b-diffusion-adapter` | A merged (both edit `sketching/generate.py`, `publish/provenance.py`, `platform_rules.toml` version, `tests/publish/test_rules.py`) | D |
| 2 | D | #30 | `plan-11-d-review-station` | A merged (`upload_visual` calls `attach_upload`) | B |
| 3 | E | #1 | `plan-11-e-prd-srd-v1-1` | A, B, C, D merged (it documents ADR 0005, the upload, diffusion and station behaviour, and the follow-up numbers) | none |

- Each PR builds in its own worktree.
- **Wave 2 conflicts:** B and D both add a README section. The second to merge rebases and keeps both sections; there are no other shared files.
- **Version pins:** B takes `version = "2026-11-13.2"`. If D merges first nothing conflicts, because D does not touch the rules.

## Review Focus

1. **Upload never stores a rejected image** (A). A density failure, a decode failure, a too-small or too-large image, a verse scene or an unknown scene leaves the library and `series.json` unchanged. Test: `test_upload_refusals_store_nothing`.
2. **Uploaded scenes are never regenerated** (A). With a stale or missing record, the scene is listed as quarantined with the upload hint, and `images` is never called. Test: `test_image_to_sketch_scene_never_generated`.
3. **Existing style fingerprints are unchanged** (B). Test: `test_fingerprint_unchanged_without_diffusion`.
4. **The default path never imports torch or diffusers** (B). Test: `test_default_path_never_imports_diffusers`.
5. **Segment-cache correctness** (D). A cached render is frame-identical to a forced render. A moved slot redraws. A failed encode clears the cache. An all-hit part never calls `_run_jobs`. Tests in Task D2.
6. **No silent overwrite from the station** (D). A stale sha256 refuses the edit. Test: `test_edit_refuses_stale_view`.
7. **Paid actions need confirmation** (D). Test: `test_paid_rebuild_needs_confirm`.

---

## PR A - Image-to-sketch: XDoG line art for uploads (#28)

Branch `plan-11-a-image-to-sketch`. The PR body ends with `Closes #28`, and states D8 (controlnet_aux not used, removed in PR B).

### Task A1: `xdog_lineart` and `ink_to_png`

**Files:**
- Modify: `src/narratoflow/visuals/lineart.py`
- Create: `tests/visuals/lineart_samples.py`, a list of `Sample(name, licence, source)` for D4
- Modify: `tests/visuals/test_lineart.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class XdogParams: sigma: float; k: float; tau: float; eps: float; phi: float`;
  - `XDOG_DEFAULT = XdogParams(1.5, 1.6, 1.0, -0.03, 200.0)`;
  - `XDOG_LONG_SIDE_PX = 512`, `XDOG_PERCENTILES = (1.0, 99.0)`, `MIN_UPLOAD_SIDE_PX = 32`, `MAX_UPLOAD_SIDE_PX = 8192`;
  - `xdog_lineart(raster: bytes, *, params: XdogParams = XDOG_DEFAULT) -> np.ndarray`;
  - `ink_to_png(ink: np.ndarray) -> bytes`.

- [ ] **Step 1: Write the failing tests.**
  - `test_xdog_sample_set_feeds_strokes[name]`: the 7 names of D4, with the 6 assertions of D5.
  - `test_xdog_flat_images_are_empty`: all-white, all-black and mid-grey 256x256 PNGs raise `SketchRejectedError` with message "empty".
  - `test_xdog_size_limits`: a 16x16 PNG raises "too small". A PNG whose header says 9000x10 raises "too large", with no full decode: build it with PIL at 9000x10, which is small in bytes.
  - `test_xdog_undecodable`: `b"not an image"` raises `SketchRejectedError("undecodable image")`.
  - `test_xdog_rgba_composited_on_white`: `fixtures.images.line_drawing_scroll_rgba()` gives a non-empty mask, and `check_density(...).ok` is True.
  - `test_xdog_upscales_small_images`: a 100x80 line drawing gives a mask whose long side is 512.
  - `test_ink_to_png_rejects_non_2d`: a 3-D array raises `ValueError`.
  - `test_xdog_sample_licences_recorded`: every entry in `lineart_samples.SAMPLES` has a non-empty licence in {"CC0-1.0", "public-domain"} and a source.
- [ ] **Step 2: Run** `uv run pytest tests/visuals/test_lineart.py -q -o addopts=""`. Expected: ImportError.
- [ ] **Step 3: Implement** per D2 and D3. Read the size with `PIL.Image.open(...).size` before `_to_gray`. `cv2.GaussianBlur(f, (0, 0), sigma)` runs on float64. The module docstring gains the XDoG formula and the measured D3 table as a comment block, in the style of the existing calibration table.
- [ ] **Step 4: Run the gate.** `lineart.py` stays at 100% branch coverage.
- [ ] **Step 5: Commit** `feat(visuals): XDoG line art for uploaded images (#28)`.

### Task A2: `UPLOADED` origin, migration 0004, `Provenance.attribution`

**Files:**
- Modify: `src/narratoflow/library/schemas.py`
- Create: `src/narratoflow/library/migrations/versions/0004_uploaded_origin.py`
- Modify: `src/narratoflow/publish/provenance.py` (`_sketch_records` carries attribution)
- Test: `tests/library/test_db.py`, `tests/library/test_repository.py`, `tests/publish/test_provenance.py`

- [ ] **Step 1: Write the failing tests.**
  - `test_uploaded_origin_round_trips`: `add_sketch(..., SketchOrigin.UPLOADED, ...)`, then `get_sketch`, keeps the origin.
  - `test_migration_0004_upgrade_downgrade`: on a db at 0003, upgrading to 0004 accepts an `uploaded` row. Downgrading with that row raises `RuntimeError` naming the count. Without it, the downgrade restores the 3-value CHECK (an insert of `uploaded` then fails with `IntegrityError`).
  - `test_provenance_attribution_default_empty`: an old JSON provenance without `attribution` loads with "".
  - `test_upload_cc_by_needs_attribution`: an `UPLOADED` record with licence `CC-BY-4.0` and no attribution gives a FAIL with "needs an attribution". With one it passes.
- [ ] **Step 2: Run.** Expected: failures.
- [ ] **Step 3: Implement** per D6a and D7. `is_generated` is unchanged, so an UPLOADED record is kind "icon".
- [ ] **Step 4: Gate. Step 5: Commit** `feat(library): uploaded sketch origin with migration 0004 (#28)`.

### Task A3: `attach_upload`, the generate rules and `sketch upload`

**Files:**
- Create: `src/narratoflow/sketching/upload.py`
- Modify: `src/narratoflow/sketching/generate.py` (`_Run.scene`, `_attached_record`)
- Modify: `src/narratoflow/cli.py`, `src/narratoflow/data/platform_rules.toml`, `README.md` (an "Uploading your own image" section)
- Test: `tests/sketching/test_upload.py`, `tests/sketching/test_generate.py`, `tests/test_cli.py`, `tests/publish/test_rules.py`, `tests/publish/test_provenance.py` (version pins)

**Interfaces:**
- Produces: `attach_upload(series: Series, scene_id: str, channel: ChannelConfig, image: bytes, *, licence: str, source: str, attribution: str = "", library: SketchLibrary) -> tuple[Series, SketchRecord]`. The input series is not mutated.

- [ ] **Step 1: Write the failing tests.**
  - `test_upload_attaches_lineart`: the camera sample on an approved narration scene.
    - The record is UPLOADED, UNCHECKED, with reason "uploaded: vision not run", and `mirror_safe` False.
    - The stored bytes are `ink_to_png(xdog_lineart(raw))`, and the strokes are stored.
    - The scene has `visual.source == IMAGE_TO_SKETCH`, `sketch_id == record.id` and no quarantine flag.
    - `provenance.license` and `provenance.source` are the given values.
  - `test_upload_refusals_store_nothing`, parametrized:
    - a density failure (`fixtures.images` crosshatched box, passed through `ink_to_png`; patch `xdog_lineart` to return its mask);
    - undecodable bytes;
    - a verse scene;
    - no idea key;
    - an unknown id.

    Each raises the documented error and leaves `library.idea_keys(version)` unchanged.
  - `test_image_to_sketch_scene_never_generated`: an `IMAGE_TO_SKETCH` scene with `sketch_id=None` is in `result.quarantined`, and `FakeImages.requests == []`.
  - `test_uploaded_record_survives_style_change`: after a style change gives a new style version, `generate_missing` keeps the uploaded sketch (`result.kept == 1`).
  - `test_cli_sketch_upload`:
    - `CliRunner` with `--license own-work --source "my photo"` saves the series and prints `<scene>: uploaded sketch <id> (unchecked)`;
    - a missing `--license` exits 2;
    - a density failure exits 1 with the reason.
  - `test_own_work_licence_passes`: the rules load `own-work` as pass, at version `"2026-11-13.1"`.
- [ ] **Step 2: Run.** Expected: failures.
- [ ] **Step 3: Implement** per D6 and D7. `_print_generation` prints the upload hint for quarantined `IMAGE_TO_SKETCH` scenes: `narratoflow sketch upload SERIES SCENE IMAGE --channel C --license ID --source TEXT`.
- [ ] **Step 4: Gate. Step 5: Commit** `feat(sketching): sketch upload attaches XDoG line art to a scene (#28)`.

**Done means (PR A):**
- [ ] `uv run pytest tests/visuals/test_lineart.py -k sample_set -q -o addopts=""` passes 7 cases.
- [ ] `uv run pytest tests/sketching/test_upload.py tests/library -q -o addopts=""` is green.
- [ ] `uv run narratoflow sketch upload --help` lists `--license`, `--source` and `--attribution`.
- [ ] `grep -n '"own-work"' src/narratoflow/data/platform_rules.toml` matches, and `grep -n 'version = "2026-11-13.1"'` matches.
- [ ] `git diff main --stat -- src/narratoflow/render/build.py src/narratoflow/models.py` is empty (no render or schema change).
- [ ] The gate is green, and `uv run coverage report | tail -1` shows 97% or more.

---

## PR B - Text-to-sketch diffusion adapter, offline half (#29)

Branch `plan-11-b-diffusion-adapter`, started from `main` after PR A merges. The PR body says no model was downloaded or run (R3), lists follow-ups D35.3 and D35.4 with their numbers, and ends with `Closes #29`.

### Task B1: `DiffusionStyle` and the channel pins

**Files:**
- Modify: `src/narratoflow/channels/config.py`, `src/narratoflow/channels/keys.py` (`FREE_PROVIDERS` += `"lineart-diffusion"`)
- Test: `tests/test_channels.py`

- [ ] **Step 1: Write the failing tests.**
  - `test_diffusion_style_defaults`: the D10 defaults hold (steps 4, guidance 0.0, seed 0, size 1024, `lora_scale` 1.0).
  - `test_lora_needs_licence_and_source`: each of the two missing gives a `ChannelConfigError` naming the field.
  - `test_lineart_provider_needs_diffusion_table`: `image_provider = "lineart-diffusion"` without `[style.diffusion]` is refused.
  - `test_fingerprint_unchanged_without_diffusion`: the literal value pinned before the change (D11).
  - `test_fingerprint_changes_with_seed_and_lora`.
  - `test_lineart_diffusion_is_free`: `prices.image("lineart-diffusion", "x") == 0`, and no key is required.
- [ ] **Step 2: Run.** Expected: failures. **Step 3: Implement.** **Step 4: Gate.**
- [ ] **Step 5: Commit** `feat(channels): per-channel diffusion model, LoRA and seed pins (#29)`.

### Task B2: `LineArtDiffusionProvider` with an injected pipeline

**Files:**
- Modify: `src/narratoflow/visuals/diffusion.py`, `src/narratoflow/images/factory.py`
- Test: `tests/visuals/test_diffusion.py`; `tests/fakes/diffusion.py` (`FakePipeline`, `fake_torch_diffusers(monkeypatch) -> Recorder`)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class DiffusionSpec`, with `model`, `revision` and the D10 fields, plus `DiffusionSpec.from_style(style: StyleConfig) -> DiffusionSpec`;
  - `TextToImagePipeline`, `PipelineFactory` and `default_pipeline_factory(spec) -> TextToImagePipeline`;
  - `effective_seed(base: int, request_seed: int | None) -> int`;
  - `LineArtDiffusionProvider`.
- Consumes: `images.ImageRequest`, `images.GeneratedImage`, `images.prompt.NEGATIVE_TERMS`.

- [ ] **Step 1: Write the failing tests.**
  - `test_same_request_same_bytes`: two providers with the same spec and `FakePipeline` give byte-identical PNGs for the same `ImageRequest(seed=7)`.
  - `test_seed_varies_by_scene_and_base`: request seeds 7 and 8 differ. Base seeds 0 and 1 differ, and are passed to the pipeline as `effective_seed(...)`.
  - `test_effective_seed_pinned`: `effective_seed(0, 7)` equals a literal value, so the formula is frozen.
  - `test_reference_png_ignored`: the result is the same with and without `reference_png`.
  - `test_pipeline_built_once_lazily`: the factory is called 0 times after `__init__` and once after two `generate` calls.
  - `test_default_factory_uses_cpu_generator_and_lora`, with `fake_torch_diffusers`:
    - it records `from_pretrained(model, revision=...)`, then `load_lora_weights(lora, weight_name=..., revision=...)`, then `fuse_lora(lora_scale=...)`;
    - the call has `generator` from `Generator("cpu").manual_seed(seed)`, `num_inference_steps=4`, `guidance_scale=0.0`, `width=height=1024`;
    - `negative_prompt` is passed only when the fake pipeline's signature has it (both cases are tested).
  - `test_extra_missing_names_install_command`: with `torch` blocked in `sys.modules`, the error is `ExtraMissingError` whose message contains `uv sync --extra diffusion`.
  - `test_pipeline_error_is_provider_error`: a pipeline raising `RuntimeError` gives a `ProviderError("lineart diffusion failed (RuntimeError)")`.
  - `test_png_has_no_metadata`: the output PNG has no `tEXt` chunks.
  - `test_factory_builds_lineart_provider`: `build_image_provider` with `image_provider = "lineart-diffusion"` returns a `LineArtDiffusionProvider` without importing torch.
- [ ] **Step 2: Run.** Expected: failures. **Step 3: Implement** per D9, D14 and D16. **Step 4: Gate.**
- [ ] **Step 5: Commit** `feat(visuals): line-art diffusion provider with injected pipeline (#29)`.

### Task B3: Licences in provenance, the default path, docs and dropping controlnet-aux

**Files:**
- Modify: `src/narratoflow/sketching/generate.py` (`_TERMS`/`_provenance`), `src/narratoflow/publish/provenance.py`, `src/narratoflow/data/platform_rules.toml`
- Modify: `pyproject.toml` and `uv.lock` (remove `controlnet-aux`), `README.md` (the "Text-to-sketch (diffusion)" section: config example, licences, gated-model note, VRAM table labelled unmeasured, CPU note)
- Test: `tests/sketching/test_generate.py`, `tests/publish/test_provenance.py`, `tests/publish/test_rules.py`

- [ ] **Step 1: Write the failing tests.**
  - `test_lineart_provenance_records_licences`: a `generate_missing` run with an injected `LineArtDiffusionProvider(spec, factory=FakePipeline)` stores provenance with `license == "Apache-2.0+MIT"` (LoRA declared MIT), `provider == "lineart-diffusion"`, and `usage_terms` containing the model revision and the LoRA source.
  - `test_generated_composite_licence_verdicts`:
    - `"Apache-2.0"` passes;
    - `"Apache-2.0+OpenRAIL++-M"` warns;
    - `"Apache-2.0+FLUX-1-dev-non-commercial"` FAILs, naming the unknown id.
  - `test_default_path_never_imports_diffusers`: D16.
  - `test_rules_version_2026_11_13_2`.
- [ ] **Step 2: Run.** **Step 3: Implement** per D12, D15 and D17. `uv lock` runs with the network, and `uv lock --check` is green afterwards.
- [ ] **Step 4:** File follow-ups D35.3 and D35.4, and put their numbers in the README VRAM table.
- [ ] **Step 5: Gate. Step 6: Commit** `feat(sketching): record diffusion model and LoRA licences; drop unused controlnet-aux (#29)`.

**Done means (PR B):**
- [ ] `uv run pytest tests/visuals/test_diffusion.py tests/test_channels.py -q -o addopts=""` is green, with no torch installed (`uv run python -c "import importlib.util as u; assert u.find_spec('torch') is None"`).
- [ ] `grep -c controlnet pyproject.toml` prints 0, and `uv lock --check` passes.
- [ ] `grep -n "FLUX.1-schnell" README.md` and `grep -n "not measured" README.md` both match.
- [ ] `gh issue list --search "Diffusion: real-model determinism" --state open` and `gh issue list --search "line-art LoRA for FLUX.1-schnell" --state open` each list one issue (gh needs the sandbox disabled).
- [ ] `git diff main -- src/narratoflow/models.py src/narratoflow/render/` is empty.
- [ ] The gate is green, with coverage of 97% or more.

---

## PR C - Diacritizer benchmark harness (#8)

Branch `plan-11-c-tashkeel-bench`. The PR body states R3 (LLM not run), the CATT reason, follow-ups D35.1 and D35.2, and ends with `Closes #8`.

### Task C1: Scoring library

**Files:**
- Create: `src/narratoflow/text/tashkeel/bench.py`
- Test: `tests/text/test_tashkeel_bench.py` (all Arabic as `\u` escapes)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Counts: letters: int; letter_errors: int; words: int; word_errors: int`;
  - `@dataclass(frozen=True) class LineScore: with_case: Counts; without_case: Counts; rejected: bool`;
  - `score_line(gold: str, predicted: str) -> LineScore`;
  - `@dataclass(frozen=True) class EngineResult: engine: str; lines: int; rejected_lines: int; der: float; der_no_case: float; wer: float; wer_no_case: float; seconds: float; not_run: str | None = None`;
  - `run_engine(name: str, diacritize: Callable[[str], str], gold: Sequence[str], *, clock: Callable[[], float] = time.perf_counter) -> EngineResult`, which catches `SkeletonMismatchError` and `ProviderError` per line (D21);
  - `not_run(name: str, reason: str) -> EngineResult`;
  - `@dataclass(frozen=True) class GoldManifest`, with `source_url`, `commit`, `file_sha256`, `licence_notes`, `rule` and `lines: list[tuple[int, str]]`;
  - `load_manifest(path: Path) -> GoldManifest`;
  - `select_gold(text: str, manifest: GoldManifest) -> list[str]`, which raises `ValueError` naming the line number on a per-line sha256 mismatch;
  - `make_manifest(text: str, *, source_url: str, commit: str, licence_notes: str, count: int = 300, min_words: int = 5, max_words: int = 60) -> GoldManifest`;
  - `results_markdown(results: Sequence[EngineResult]) -> str`.

- [ ] **Step 1: Write the failing tests.**
  - `test_perfect_prediction_zero_error`.
  - `test_undiacritized_prediction`: for a fully vocalized 3-word line, DER is 100 and WER is 100. A gold letter with no harakat, predicted bare, is correct.
  - `test_case_ending_variant_ignores_last_letter`: a wrong mark only on the last letter of each word gives `without_case.letter_errors == 0` and `with_case.letter_errors == words`.
  - `test_non_arabic_excluded`: digits, Latin and punctuation change no count.
  - `test_shadda_order_insensitive`.
  - `test_superscript_alef_and_tatweel_stripped`.
  - `test_changed_base_letters_rejected`: all letters and words are wrong, and `rejected` is True.
  - `test_run_engine_catches_skeleton_mismatch` and `test_run_engine_catches_provider_error`.
  - `test_not_run_row_has_reason` and `test_results_markdown_lists_not_run_reason`.
  - `test_select_gold_checks_hashes`: one altered line raises `ValueError("gold line 7: sha256 mismatch")`.
  - `test_make_manifest_rule`: on a 10-line synthetic text, the selection follows the token bounds and the count, with 1-based line numbers.
  - `test_committed_manifest_is_well_formed`, which loads `docs/benchmarks/tashkeel-gold.toml`:
    - 300 lines, strictly ascending, with 64-hex hashes;
    - `file_sha256 == "4e851ff8...aa12"` (full value from D18) and the commit from D18;
    - `licence_notes` mentions "MIT" and "GPLv2".
- [ ] **Step 2: Run.** Expected: ImportError. **Step 3: Implement** per D20 and D21. **Step 4: Gate**, with `bench.py` at 100% branch coverage.
- [ ] **Step 5: Commit** `feat(text): DER/WER scoring for diacritizers (#8)`.

### Task C2: Fetch script, manifest, benchmark script and run

**Files:**
- Create: `scripts/fetch_gold_tashkeel.py`, a `fetch_quran.py`-style script with pinned URL and sha256, an atomic write to `--out` (default `.cache/bench/tashkeel`), and `--make-manifest PATH`
- Create: `scripts/bench_tashkeel.py`, with `--gold DIR`, `--manifest PATH`, `--engines identity,mishkal,llm,catt` and `--out JSON`; `llm` and `catt` always produce `not_run` rows with the D22 reasons
- Create: `docs/benchmarks/tashkeel-gold.toml`, `docs/benchmarks/2026-11-13-tashkeel-results.json`
- Test: `tests/text/test_bench_scripts.py`, which loads both scripts with `importlib` as `tests/render/test_assets.py` does, and checks the argument parsing and that `llm`/`catt` are never executed (offline)

- [ ] **Step 1: Write the failing script tests.**
  - `test_bench_script_llm_and_catt_not_run`: with a 3-line synthetic gold and `--engines identity,llm,catt`, the JSON has 3 rows and two `not_run` reasons, containing "paid" and "onnxruntime-gpu".
  - `test_fetch_script_refuses_bad_sha`: `httpx` is patched to return wrong bytes, and the script exits non-zero writing nothing.
- [ ] **Step 2: Implement.**
- [ ] **Step 3: Run with the network.** `uv run python scripts/fetch_gold_tashkeel.py`, then `uv run python scripts/fetch_gold_tashkeel.py --make-manifest docs/benchmarks/tashkeel-gold.toml`. The manifest's 300th line number must be 427 (D19); if not, stop and report.
- [ ] **Step 4: Run the benchmark on CPU (about 10 s).** `uv run --with mishkal==0.4.1 python scripts/bench_tashkeel.py --gold .cache/bench/tashkeel --manifest docs/benchmarks/tashkeel-gold.toml --engines identity,mishkal,llm,catt --out docs/benchmarks/2026-11-13-tashkeel-results.json`. Keep the printed Markdown for C3.
- [ ] **Step 5: Gate. Step 6: Commit** `feat(scripts): pinned gold set fetch and diacritizer benchmark (#8)`.

### Task C3: ADR 0005 and follow-ups

**Files:**
- Create: `docs/adr/0005-diacritizer-benchmark.md` (D24)
- Modify: `docs/adr/0001-record-architecture-decisions.md` (link 0005)

- [ ] **Step 1:** File follow-ups D35.1 and D35.2.
- [ ] **Step 2:** Write the ADR with the C2 table copied verbatim, the issue numbers, and the exact commands used.
- [ ] **Step 3: Commit** `docs(adr): 0005 diacritizer benchmark results (#8)`.

**Done means (PR C):**
- [ ] `uv run pytest tests/text/test_tashkeel_bench.py tests/text/test_bench_scripts.py -q -o addopts=""` is green offline.
- [ ] `python3 -c "import tomllib;m=tomllib.load(open('docs/benchmarks/tashkeel-gold.toml','rb'));assert len(m['lines'])>=200"` exits 0.
- [ ] `python3 -c "import pathlib,re,sys;sys.exit(any(re.search('[؀-ۿ]',p.read_text(encoding='utf-8')) for p in pathlib.Path('docs/benchmarks').glob('*tashkeel*')))"` exits 0 (no Arabic text is committed).
- [ ] `python3 -c "import json;r={x['engine']:x for x in json.load(open('docs/benchmarks/2026-11-13-tashkeel-results.json'))};assert r['mishkal']['not_run'] is None and r['llm']['not_run'] and r['catt']['not_run']"` exits 0.
- [ ] `grep -n "not run" docs/adr/0005-diacritizer-benchmark.md` shows the llm and catt rows with issue numbers.
- [ ] The gate is green, with coverage of 97% or more.

---

## PR D - Gradio review station (#30)

Branch `plan-11-d-review-station`, started from `main` after PR A merges. The PR body states D29 (what "only that scene" means), D31 (pillarbox) and follow-up D35.5, and ends with `Closes #30`.

### Task D1: Shared seams

**Files:**
- Modify: `src/narratoflow/audio/build.py` (`check_scene_text(scene: Scene) -> None` made public; `_check_texts` calls it)
- Modify: `src/narratoflow/sketching/resolve.py` (`regenerate(..., require_quarantine: bool = True)` and `regenerate_estimate(idea_key, channel, prices) -> Decimal`, moved from the CLI)
- Modify: `src/narratoflow/cli.py` (uses `regenerate_estimate`)
- Test: `tests/audio/test_build.py`, `tests/sketching/test_resolve.py`

- [ ] **Step 1: Write the failing tests.**
  - `test_check_scene_text_messages_unchanged`: the two existing error strings are unchanged.
  - `test_regenerate_unflagged_scene_when_allowed`: with `require_quarantine=False`, an attached scene is regenerated with `sketch_attempt + 1`. With the default, it is refused as before.
  - `test_regenerate_estimate_matches_cli`.
- [ ] **Step 2: Implement. Step 3: Gate. Step 4: Commit** `refactor: public text check and regenerate seams for the station (#30)`.

### Task D2: WITHDRAWN by D37 — do not build (skip to D3)

**Files:**
- Modify: `src/narratoflow/render/build.py` (D30)
- Test: `tests/render/test_segment_cache.py`, using `fixtures.parts` (`make_series`, `add_sketch`, `write_timelines`, `render`) and `render_tools` (fail-in-CI)

- [ ] **Step 1: Write the failing tests.** Use a canvas at 1/6 and `workers=1`. A spy wraps `compose.render_segment` and records `job.scene_id`.
  - `test_second_render_all_segments_cached`: with `force` False and the part's report removed, the second render redraws nothing, `_run_jobs` is not called, and `result.redrawn == {1: []}`.
  - `test_edit_one_sketch_redraws_only_that_scene`: three scenes a, b, c. b gets a new sketch with the same slot, and its timelines are rewritten. Only `["b"]` is redrawn, part 1 is not in `skipped`, and a second part stays in `skipped`.
  - `test_cached_segments_give_identical_frames`: the `framemd5` of the cached render equals that of a `force=True` render of the same inputs.
  - `test_moved_slot_redraws_later_scenes`: b's slot is longer by 1 s, and `["b", "c"]` is redrawn.
  - `test_failed_encode_clears_cache`: with `encode_part` patched to raise once, `out/.segments/part-01` no longer exists, and the next run redraws everything.
  - `test_cache_pruned_to_current_keys`: after the b edit, the segment directory holds exactly 3 `.mkv` files.
  - `test_force_ignores_cache`: everything is redrawn.
- [ ] **Step 2: Run.** **Step 3: Implement.** **Step 4: Gate**, and the existing `tests/render/test_build.py` stays green unchanged.
- [ ] **Step 5: Commit** `feat(render): per-scene segment cache so one edited scene is redrawn (#30)`.

### Task D3: Pillarbox 16:9 export

**Files:**
- Create: `src/narratoflow/publish/landscape.py`
- Test: `tests/publish/test_landscape.py` (`render_tools`)

**Interfaces:**
- Produces:
  - `LANDSCAPE_SIZE = (1920, 1080)`;
  - `pillarbox_export(tools: RenderTools, src: Path, out: Path, *, frames: int, duration_ms: int) -> Path`;
  - `landscape_name(index: int) -> str`, which returns `"part-NN.16x9.mp4"`.

- [ ] **Step 1: Write the failing tests.**
  - `test_pillarbox_is_1920x1080_same_frames`: from a small 9:16 MP4 rendered by the fixtures at 1/6 scale (180x320), the output is 1920x1080 with the frame count and duration within `verify_mp4` tolerance. The left and right columns are white (decode one frame, mean > 250 in the first 100 px).
  - `test_pillarbox_failure_leaves_no_file`: a missing src raises `RenderError`, and `out` does not exist.
- [ ] **Step 2: Implement** per D31. **Step 3: Gate. Step 4: Commit** `feat(publish): pillarboxed 16:9 copy of an exported part (#30)`.

### Task D4: Station functions

**Files:**
- Create: `src/narratoflow/ui/station.py`
- Modify: `src/narratoflow/ui/__init__.py` (docstring only)
- Test: `tests/ui/__init__.py`, `tests/ui/test_station.py` (fakes: `FakeVoice`, `FakeImages`, `fixtures.parts`)

**Interfaces:**
- Produces:
  - `StationPaths`;
  - `SceneRow(scene_id, part, kind, visual, flags)`;
  - `SceneView`;
  - `RebuildReport(scene_id: str, part: int, generated: GenerationResult, audio_parts: list[int], rendered_parts: list[int], spent_usd: Decimal)`;
  - `series_sha256(path) -> str`;
  - `list_scenes`, `scene_view`, `edit_text`, `edit_timing`, `upload_visual`, `regenerate_visual`, `use_text_card`, `rebuild_scene`, `export`, as in D28.
  - Each of `rebuild_scene`, `regenerate_visual` and `export` takes optional injected `settings`, `images`, `vision`, `voices`, `aligner`, `tools` and `library`, which pass through to the stage functions.

- [ ] **Step 1: Write the failing tests.**
  - `test_edit_text_persists_to_series_json`: `Series.load(paths.series)` shows the new `tts_text`, and the other scenes are byte-equal in `model_dump`.
  - `test_edit_text_rejects_changed_words`: a skeleton change raises `PlanningError`, and the file sha256 is unchanged.
  - `test_edit_timing_bounds`: 999 ms and a `draw_fraction` of 0.05 are refused.
  - `test_edit_refuses_stale_view`: the file is modified between view and edit, and `PlanningError` says "reload".
  - `test_station_holds_no_state`: there is no module-level mutable state (`station.__dict__` holds only functions, classes and constants). Two calls after an external `Series.save` see the external change.
  - `test_rebuild_one_scene_only`: two parts. The sketch of `p1-b` is replaced via `upload_visual`, then `rebuild_scene` runs:
    - `report.rendered_parts == [1]`, and part 2's MP4 and report are byte-identical with unchanged mtimes (D37);
    - `FakeVoice` got no new synthesis call (TTS cache);
    - `FakeImages.requests == []`.
  - `test_paid_rebuild_needs_confirm`: with an uncached voice whose price is above 0 and `confirm=False`, it raises before any call, and the message has the estimate.
  - `test_export_portrait_and_landscape`:
    - `export(..., PORTRAIT)` returns `[export/part-01.mp4]`;
    - `export(..., LANDSCAPE)` returns `[export/part-01.16x9.mp4]` at 1920x1080 (`render_tools`);
    - a part with a FAIL produces neither.
  - `test_scene_view_audio_clip_cut_to_slot`: the clip duration equals the scene slot within 30 ms, and a second call reuses the file (same mtime).
- [ ] **Step 2: Run.** **Step 3: Implement.** **Step 4: Gate**, with `station.py` at 100% branch coverage.
- [ ] **Step 5: Commit** `feat(ui): review-station functions that edit series.json (#30)`.

### Task D5: Thin Gradio layer and `narratoflow ui`

**Files:**
- Create: `src/narratoflow/ui/app.py`, `tests/fakes/gradio.py`, `tests/ui/test_app.py`
- Modify: `src/narratoflow/cli.py` (`ui SERIES_JSON --channel C [--audio DIR] [--export DIR] [--host 127.0.0.1] [--port 7860]`), `README.md` ("Review station")

**Interfaces:**
- Produces: `build_app(paths: StationPaths, settings: Settings, gr: Any) -> Any`. The layout:
  - a scene dropdown;
  - Audio, tts and caption Textboxes, an Image, two Numbers for timing;
  - Buttons: Save text, Save timing, Upload (File plus licence and source Textboxes), Regenerate (estimate Markdown plus confirm Checkbox), Text card, Rebuild scene, Export 9:16 and Export 16:9;
  - a hidden Textbox carrying `expected_sha256` (D33).
- Every callback wraps `NarratoFlowError` into `gr.Error(str(e))`. This is the only catch outside `cli.main`, and it is allowed because it re-raises as the UI's error type.

- [ ] **Step 1: Write the failing tests.**
  - `test_app_wires_every_button`: with `FakeGradio`, the 9 buttons are registered with the expected input and output component names.
  - `test_export_buttons_call_both_ratios`: the two recorded callbacks call `station.export` with PORTRAIT and LANDSCAPE.
  - `test_callback_errors_become_gr_error`.
  - `test_cli_ui_without_extra`: with `gradio` blocked in `sys.modules`, the exit code is 1 and the message contains `uv sync --extra ui`.
  - `test_real_gradio_builds` (`pytest.importorskip("gradio")`): `build_app` returns a `gr.Blocks` and is not launched.
- [ ] **Step 2: Implement.** The CLI calls `app.launch(server_name=host, server_port=port)`, which is the only line marked `# pragma: no cover - starts a server`.
- [ ] **Step 3:** File follow-up D35.5.
- [ ] **Step 4: Gate. Step 5: Commit** `feat(ui): Gradio review station and narratoflow ui (#30)`.

**Done means (PR D):**
- [ ] `uv run pytest tests/ui tests/publish/test_landscape.py -q -o addopts=""` is green without gradio installed. The importorskip test is the only skip.
- [ ] `uv run pytest tests/ui/test_station.py -k "rebuild_one_scene_only or persists" -q -o addopts=""` passes.
- [ ] `uv run narratoflow ui --help` lists `--channel` and `--port`.
- [ ] `grep -n 'RENDER_INPUTS_VERSION = "2"' src/narratoflow/render/build.py` matches.
- [ ] `uv run --extra ui python -c "import gradio"` succeeds locally, then `uv run --extra ui pytest tests/ui/test_app.py -k real_gradio -q -o addopts=""` passes. This is a local check; the network is allowed for the extra.
- [ ] The gate is green, with coverage of 97% or more.

---

## PR E - PRD/SRD v1.1 (#1)

Branch `plan-11-e-prd-srd-v1-1`, started from `main` after A to D merge. Docs only. The PR body lists each acceptance criterion with the section that meets it, and ends with `Closes #1`.

### Task E1: PRD v1.1

**Files:** Modify `docs/PRD.md`.

Sections, each with the evidence it cites:
1. **Status and changes from v1.0:** a table mapping C1 to C10 to sections.
2. **Vision (corrected).**
   - The open-source core is Apache-2.0 (`pyproject.toml`).
   - "Open-weights", not "FOSS", for models (C8).
   - Paid providers are optional and budget-capped: `ChannelConfig.spending_limit_usd`, `SpendTracker`.
   - Remotion is removed (C1, ADR 0002).
3. **Personas:** an Islamic-studies educator (verse scenes, reciters, `riwaya`; `quran/`), and a history explainer channel (`tests/fixtures/channels/history.toml`). Each is marked "Recommendation, owner to confirm" where the code is silent.
4. **Goals and success metrics.** Each metric is tied to an artifact that measures it:
   - time from text to video: stage wall times, and the render benchmark `docs/benchmarks/2026-11-13-render-cpu.md`;
   - % of scenes accepted without edits: `GenerationResult.quarantined` and unchecked counts, plus station edits;
   - a retention proxy: hook, recap, teaser and end card exist (`Part` fields); no measurement is built, so this is a Recommendation.
5. **Non-goals:**
   - no dialects (flag only, C9; `check-dialect`);
   - no real-time;
   - no cloud SaaS;
   - no voice cloning (C3);
   - no native 16:9 layout in v1 (D31, follow-up D35.5).
6. **Linguistic policy (corrected):** dialect is flagged by CAMeL DID, never rejected (C9). The diacritizer is chosen by ADR 0005. Protected spans for the Qur'an.
7. **Workflow and phasing:** M0 to M4 per REVIEW §4, with each milestone mapped to its CLI commands (`plan`, `approve`, `sketch generate/resolve/upload`, `audio build`, `render`, `review`, `publish`, `ui`) and its GitHub milestone.
8. **Open questions answered (§5):** D26, verbatim structure, with the "Recommendation, owner to confirm" lines.

### Task E2: SRD v1.1

**Files:** Modify `docs/SRD.md`.

Sections, each with the evidence it cites:
1. **Architecture:** a staged CLI over `series.json` (ADR 0004). The diagram is redrawn from `cli.py` commands.
2. **IR schema:**
   - the `Series`, `Part`, `Scene`, `Narration`, `Visual` and `VerseRef` fields from `models.py`;
   - per-artifact versions (ADR 0004);
   - the stroke schema (`docs/strokes-schema.md`).
3. **Interfaces:** a table of protocol, implementations and file:
   - `Diacritizer` (`interfaces.py`, `text/tashkeel/engines.py`);
   - `SpeechProvider` (`tts/`);
   - `Aligner` (`align/`);
   - `ImageProvider` (`images/`, `visuals/diffusion.py`);
   - `LLMClient` (`llm/`);
   - `VisualProvider` (`interfaces.py`);
   - the compositor (`render/compose.py`).
4. **Modules (corrected):**
   - tashkeel guard (§2.1);
   - TTS chain (ADR 0003, C2, C3);
   - alignment (C5: provider spans, else CTC, else estimate);
   - visuals: XDoG uploads (PR A), centerline strokes (C7, `visuals/strokes.py`), diffusion (C8, PR B, with the VRAM table);
   - compositor (C1, ADR 0002);
   - captions: logical-order Unicode into libass, no reshaper (C6, `captions/ass.py`);
   - review station (PR D, D29 to D31).
5. **NFRs:**
   - render NFR: 600 s for a 180 s part, with the measured 84.5 s and 181.8 s (ADR 0002);
   - determinism: seeds, `test_workers_do_not_change_frames`, D14;
   - caching: the TTS cache, render skip, segment cache (D30);
   - disk: the segment cache is bounded per part;
   - minimum hardware: D26.4.
6. **Failure modes:** a table of trigger, behaviour, error type and test, drawn from `errors.py` and ADR 0003:
   - edge outage falls back to Piper (`fallback_reasons`);
   - invalid LLM output is retried, then the stage fails;
   - an icon miss leads to generation, quarantine or a text card;
   - a budget overrun raises `BudgetExceededError`, keeping progress;
   - a skeleton mismatch sets the `tashkeel_rejected` flag;
   - a stale timeline raises `StaleTimelineError`;
   - a missing extra raises `ExtraMissingError`.
7. **Dependency policy (C10):**
   - extras as in `pyproject.toml` (`arabic`, `tts-edge`, `tts-local`, `align`, `diffusion`, `ui`);
   - `uv.lock`, and `uv sync --locked` in CI;
   - no unused dependencies (controlnet-aux removed);
   - MoviePy, faster-whisper, streamlit, cairosvg, arabic-reshaper and python-bidi are not used.
8. **Licence matrix:** every runtime dependency, extra, model and bundled asset pack:
   - `assets/LICENSES.md`, fonts, hand, SFX, icons, Tanzil, skimage samples, the gold set;
   - the D27 rows;
   - columns: component, version, licence, commercial-safe (yes / warn / no), source URL checked, date.

   The builder re-checks each row with `curl -s https://pypi.org/pypi/<pkg>/json` or the `gh api`/HF API calls used in this plan, and corrects any row that differs.

### Task E3: REVIEW status line and follow-up

- [ ] File follow-up D35.6, and cite its number in SRD §8.
- [ ] Add the D25 status line to `docs/REVIEW.md`.
- [ ] **Commit** `docs: PRD and SRD v1.1 with review corrections and licence matrix (#1)`.

**Done means (PR E):**
- [ ] `grep -n "^## " docs/PRD.md` lists Personas, Success metrics, Non-goals, Phasing and Open questions.
- [ ] `grep -n "^## " docs/SRD.md` lists IR schema, Interfaces, Non-functional requirements, Failure modes and Licence matrix.
- [ ] `grep -nE "Remotion|XTTS|F5-TTS|Farasa" docs/PRD.md docs/SRD.md` shows each only in corrected form (source-available or non-commercial), never as a FOSS default.
- [ ] `grep -c "owner to confirm" docs/PRD.md` is 1 or more.
- [ ] `grep -n "CC-BY-NC-4.0" docs/SRD.md` shows the aligner row with the D35.6 issue number.
- [ ] `grep -nE "arabic.reshaper|python-bidi" docs/SRD.md` appears only in the "not used" line.
- [ ] `git diff main --stat` lists only `docs/PRD.md`, `docs/SRD.md` and `docs/REVIEW.md`, and the gate is green (unchanged code).
