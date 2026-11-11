# Plan 3: Sketch Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For an approved series, fill every scene that is `match == "new"` with a checked, consistently styled sketch, and give every scene that has a sketch the ordered centerline strokes the hand will draw. The work for each sketch is: generate → binarize → density check → vision check (likeness, logo, outlier, mirror-safe) → store as passed, unchecked or quarantined. The creator resolves a quarantined scene without any automatic spend.

**Architecture:**
- **Image providers** sit behind an `ImageProvider` Protocol. Paid (default) is OpenAI `gpt-image-1` over REST, with the channel's reference sketch sent as the style image. Free is local diffusion through the `diffusion` extra.
- **Line-art pipeline.** Everything after generation is pure, deterministic and CPU-only, in three modules:
  - `visuals/lineart.py` binarizes and runs the density check;
  - `visuals/strokes.py` builds the skeleton and the ordered strokes;
  - `visuals/marker.py` re-renders the strokes in the channel's marker.
- **Vision checks** reuse the Plan 2 `LLMClient`, extended with image input.
- **Orchestration.** `sketching/generate.py` runs the whole loop under the Plan 2 `SpendTracker`. The CLI adds `sketch generate` and `sketch resolve`.

**Tech Stack:** Python 3.12, httpx (OpenAI Images REST), opencv-python-headless and scikit-image (already dependencies), Pillow, numpy, the `anthropic` SDK image blocks, and optional `diffusers` (the `diffusion` extra).

**Spec:** `docs/superpowers/specs/2026-11-10-sketch-library-pipeline-design.md`. This plan implements §2.3, §3 steps 2 and 4, §4.1 (the image rows), §4.3, §4.4, the §4.5 mirror-safe decision, and the generation-time rows and quarantine rules of §6.3.

---

## Global Constraints

- **No human figures, logos or likeness [spec §2.2a, §2.2c, §6.3].**
  - Every generated sketch gets a vision check.
  - **Allowed:** standalone symbolic organs and limbs (eye, a stylized hand holding a pen or balance, footprints, paths).
  - **FAIL:** faces, recognizable likenesses, heads, torsos, stick figures, full-body silhouettes, logos, trademarks, copyrighted characters.
- **Negative prompt [spec §4.3].** The forbidden list is exactly `shading, fills, grayscale, textures, double lines, gradients, hatching, 3d`.
  - Where a provider has no negative-prompt parameter, the adapter writes these as explicit "no …" instructions in the positive prompt.
  - Prompts also carry the §2.2c rule as positive wording: "no people, no faces, no figures, no text, no logos".
- **Density check runs before the paid vision check [spec §4.3].** A density reject costs nothing further. It stores the sketch as FAILED with the reason `density:<metric>`.
- **When no vision check is possible** (free mode with no vision model configured), the sketch is stored as `UNCHECKED` with the reason `vision: not checked`. This is never a silent PASS [spec §6.3].
- **A failed sketch is quarantined and never reused.**
  - Nothing is regenerated automatically.
  - The creator chooses one of three resolutions: regenerate (paid, with confirmation), swap the idea key, or use a text card.
  - A FAIL has no override [spec §6.3].
- **Callers must check status.** Every caller of `add_sketch` or `copy_to` checks `.status` (Plan 1 carry-over). A FAILED result is never attached to a scene.
- **Spending:**
  - Every image and vision call goes through the `SpendTracker` (`ensure_room` before, charge after).
  - The limit is per part (`ChannelConfig.spending_limit_usd`).
  - Generation refuses to start on a series that is not approved (`Series.approved`).
- **One locked style per channel [spec §4.4].** Every paid request carries the same `style.instructions` and `style.reference_sketch`. The marker re-render uses `style.line_weight_px`, `style.ink` and a wobble seed.
- **Determinism.**
  - Binarization, strokes and marker rendering are pure functions of their inputs.
  - Stroke order is deterministic.
  - Wobble and variation seeds come from `choose_variant`-style hashing of `video_id:scene_id` (Plan 1).
- **Secrets** are `SecretStr` and never logged. Add `image_api_key` usage (it already exists in Settings); a missing key fails fast (`check_provider_keys`).
- **Tests never touch the network or a GPU.**
  - Providers are tested with `httpx.MockTransport` and fakes.
  - Image fixtures are generated in code (PIL drawings), so there are no binary blobs in git except one small reference PNG if needed.
- Same tooling gate as Plans 1–2. Sources are pure ASCII with `\u` escapes.

## Review Focus

1. **A provider returns a filled silhouette or crosshatched image.** Expected: the density check rejects it before any vision spend, and it is stored FAILED with the metric. → Task 3 and Task 7 tests.
2. **The vision model flags a face in an `eye|generic` sketch, or a stick figure in `road|two-horizons`.** Expected: FAILED, quarantined, the scene is left without a visual, and `sketch resolve` is required. A valid `eye|generic` or `hand|generic` verdict is not a false FAIL, because the calibration prompt allows these. → Task 6 and Task 7 tests.
3. **The spending limit is hit mid-series.** Expected:
   - generation stops before the call that would exceed the limit;
   - sketches already stored stay stored and are reused on the next run (this happens naturally, since the next run finds them as `reused`);
   - `BudgetExceededError` names the estimate and the limit.

   → Task 7 test.
4. **A bundled icon is used for a generic idea.** Expected: it is copied into the channel library on first use (`origin=copied`), its copy's `.status` is checked, and the next use is `reused`. → Task 7 test.
5. **A sketch whose skeleton has thousands of tiny spurs** (noisy JPEG-like edges). Expected: spurs shorter than the pruning threshold are removed, and the stroke count stays bounded (≤ `MAX_STROKES`). → Task 4 test.

---

## File Structure

```
src/narratoflow/
  images/__init__.py           # NEW: ImageProvider protocol, ImageRequest, GeneratedImage
  images/prompt.py             # NEW: build_prompt(style, idea_key, description) with negative "no ..." wording
  images/openai_images.py      # NEW: OpenAIImageProvider (REST, httpx)
  images/local_diffusion.py    # NEW: LocalDiffusionProvider (lazy diffusers import)
  images/factory.py            # NEW: make_image_provider(channel, settings, tracker, prices) + TrackedImageProvider
  llm/base.py                  # MODIFY: LLMClient.complete_structured gains images: Sequence[bytes] = ()
  llm/anthropic_client.py      # MODIFY: image content blocks
  llm/ollama.py                # MODIFY: images field
  costs/tracker.py             # MODIFY: TrackedLLM forwards images; vision estimate helper
  channels/config.py           # MODIFY: VisionConfig (provider, model) optional, defaults to llm
  visuals/lineart.py           # REPLACE stub: binarize(), DensityReport, check_density()
  visuals/strokes.py           # REPLACE stub: Stroke, StrokeSet, extract_strokes()
  visuals/marker.py            # NEW: render_marker(strokes, size, line_weight, ink, seed)
  visuals/bundled.py           # MODIFY: hardened index validation
  sketching/__init__.py        # NEW
  sketching/vision.py          # NEW: VisionVerdict, VISION_SYSTEM, check_sketch()
  sketching/generate.py        # NEW: generate_missing(series, ...) orchestration
  sketching/resolve.py         # NEW: regenerate / swap / text-card
  library/repository.py        # MODIFY: default_library engine cache; strokes blob helpers
  cli.py                       # MODIFY: `sketch generate`, `sketch resolve`
pyproject.toml                 # MODIFY: wheel force-include of assets/icons
tests/fixtures/images.py       # NEW: code-drawn line art / silhouette / hatching / spur images
```

---

### Task 1: Image provider protocol, prompt builder, OpenAI adapter

**Files:** Create `images/__init__.py`, `images/prompt.py`, `images/openai_images.py`. Tests go in `tests/images/`.

**Interfaces — Produces:**
- `@dataclass(frozen=True) class ImageRequest: prompt: str; reference_png: bytes | None; size: Literal["1024x1024"] = "1024x1024"; seed: int | None = None`
- `@dataclass(frozen=True) class GeneratedImage: png: bytes; provider: str; model: str; revised_prompt: str | None`
- `class ImageProvider(Protocol): name: str; model: str; def generate(self, req: ImageRequest) -> GeneratedImage: ...`
- `NEGATIVE_TERMS: tuple[str, ...] = ("shading", "fills", "grayscale", "textures", "double lines", "gradients", "hatching", "3d")`. These are verbatim from the spec.
- `build_prompt(style: StyleConfig, idea_key: str, description: str) -> str` builds a deterministic string containing:
  - the style instructions;
  - `Subject: <description>` (or the key's object and context when the description is empty);
  - `Black ink line drawing on a plain white background, single clean outlines.`;
  - one "No …" sentence listing every `NEGATIVE_TERMS` item plus `people, faces, human figures, text, letters, logos`.
- `OpenAIImageProvider(api_key: SecretStr, model: str, http: httpx.Client | None = None)`:
  - When `reference_png` is set, it sends `POST https://api.openai.com/v1/images/edits` as multipart with `image[]=reference`, `prompt`, `model` and `size`. Otherwise it sends `POST /v1/images/generations` as JSON.
  - It reads `data[0].b64_json` and returns PNG bytes, and validates the PNG with Pillow.
  - Errors map to `ProviderError`: non-2xx, timeout, connect failure, a bad body, or a non-image payload. The key is never echoed. The timeout is 180 s.
  - **Verify the current Images API request and response shape** against OpenAI's API reference with WebFetch before coding. Record the URL and date in the module docstring. If edits no longer accept `image[]`, use the documented field and note it.
- The OpenAI key is `Settings.image_api_key`.

- [ ] **Step 1: Write the failing tests** (`httpx.MockTransport`):
  - `test_prompt_contains_every_negative_term_and_style`
  - `test_prompt_is_deterministic`
  - `test_generation_request_shape_without_reference`
  - `test_edit_request_is_multipart_with_reference`
  - `test_bad_b64_raises_provider_error`
  - `test_401_does_not_echo_key`
- [ ] **Steps 2–5:** Red, implement, run the gate, then commit `feat: image provider protocol, prompt builder and OpenAI adapter`.

### Task 2: Local diffusion and the tracked provider factory

**Files:** Create `images/local_diffusion.py` and `images/factory.py`. Modify `costs/tracker.py` if needed. Tests go in `tests/images/test_factory.py`.

**Interfaces — Produces:**
- `LocalDiffusionProvider(model: str, device: str = "auto")`:
  - lazily imports `diffusers` and `torch`, and raises `ChannelConfigError("install the 'diffusion' extra …")` when they are missing;
  - `generate` runs a text-to-image pipeline, or img2img with the reference when one is given, with the negative prompt built from `NEGATIVE_TERMS`;
  - it is free of charge.
- `TrackedImageProvider(inner, tracker, prices)`. Before the call it does `ensure_room(prices.image(provider, model), "image <key>")`. After a success it charges that amount, and a failed call charges nothing.
- `make_image_provider(channel, settings, tracker, prices) -> ImageProvider` dispatches on `channel.style.image_provider`, case-insensitively:
  - `"openai"` → `OpenAIImageProvider` (`MissingApiKeyError` without a key);
  - `"local-diffusion"` → `LocalDiffusionProvider`;
  - anything else → `ChannelConfigError`.

- [ ] **Step 1: Write the failing tests:**
  - factory branches;
  - a missing extra explains itself (monkeypatched import);
  - the tracked provider blocks before the call (fake provider calls == 0) and charges after.
- [ ] **Steps 2–5:** Red, implement, run the gate, then commit `feat: local diffusion option and spend-tracked image providers`.

### Task 3: Binarization and the density check

**Files:** Replace `visuals/lineart.py`. Create `tests/fixtures/images.py` with drawing helpers, and `tests/visuals/test_lineart.py`.

**Interfaces — Produces:**
- `binarize(png: bytes, *, method: Literal["otsu", "adaptive"] = "otsu") -> np.ndarray`:
  - output is a bool array where ink is True;
  - converts to grayscale, flattens alpha onto white, applies Otsu (`cv2.threshold` with `THRESH_OTSU`), then removes specks with a 3×3 morphological open;
  - an image with no ink raises `SketchRejectedError("empty")`, a new `NarratoFlowError` subclass.
- `@dataclass(frozen=True) class DensityReport: ink_ratio: float; branch_points_per_kpx: float; endpoint_ratio: float; ok: bool; reason: str | None`
- `check_density(ink: np.ndarray) -> DensityReport`. It skeletonizes with `skimage.morphology.skeletonize`.
  - A branch point is a skeleton pixel with ≥ 3 skeleton neighbours (8-connectivity).
  - The report is not ok when `ink_ratio > MAX_INK_RATIO` (filled silhouettes) or `branch_points_per_kpx > MAX_BRANCH_DENSITY` (hatching).
  - The thresholds are module constants, calibrated in Step 1 on the code-drawn fixtures and documented with their calibration values.
- Fixtures in `tests/fixtures/images.py`. Each returns 512×512 PNG bytes drawn with PIL:
  - `line_drawing_scroll()`: outlines only;
  - `filled_silhouette()`: a solid black shape covering about 30% of the image;
  - `crosshatched_box()`: a dense 45° hatch;
  - `noisy_line_drawing()`: outlines plus salt noise;
  - `blank()`.

- [ ] **Step 1: Write the failing tests:**
  - `test_line_drawing_passes`
  - `test_filled_silhouette_rejected_by_ink_ratio`
  - `test_crosshatching_rejected_by_branch_density`
  - `test_noise_specks_removed_by_open`
  - `test_blank_raises`
  - `test_alpha_png_flattened_on_white`
  - `test_deterministic`
- [ ] **Steps 2–5:** Red, implement, run the gate, then commit `feat: binarization and density check before paid vision`.

### Task 4: Centerline strokes

**Files:** Replace `visuals/strokes.py`. Test: `tests/visuals/test_strokes.py`.

**Interfaces — Produces:**
- `class Stroke(BaseModel): points: list[tuple[float, float]]`, with points normalized to [0, 1] and `min_length=2`.
- `class StrokeSet(BaseModel): version: int = 1; width: int; height: int; strokes: list[Stroke]`, with `to_json() -> bytes` and `from_json(bytes) -> StrokeSet`.
- `extract_strokes(ink: np.ndarray, *, rtl: bool = False) -> StrokeSet`. The steps:
  1. Skeletonize.
  2. Build an 8-connected pixel graph.
  3. Split it at junctions and endpoints into paths.
  4. Prune spur paths shorter than `MIN_SPUR_PX` (default 8) that end at an endpoint.
  5. Simplify each path with Douglas–Peucker (`cv2.approxPolyDP`, epsilon 1.0 px).
  6. Order the strokes greedily by nearest start point. The first stroke is the one whose start is closest to the top-left corner, or top-right when `rtl=True` (Arabic channels).
  7. Reverse a stroke when its end is nearer to the pen's current position.
  8. Cap the result at `MAX_STROKES = 400` by dropping the shortest strokes.

  The output is deterministic and has no random tie-breaks; ties go to the lowest (y, x).

- [ ] **Step 1: Write the failing tests:**
  - a square outline → one closed-ish stroke or ≤ 4 strokes covering its perimeter;
  - an "L" → 1 stroke;
  - a "T" → 2–3 strokes split at the junction;
  - spurs pruned (Review Focus 5): `noisy_line_drawing` gives strokes ≤ MAX_STROKES, with no stroke shorter than MIN_SPUR_PX;
  - `rtl` changes the first stroke;
  - JSON roundtrip;
  - deterministic.
- [ ] **Steps 2–5:** Red, implement, run the gate, then commit `feat: centerline stroke extraction and ordering`.

### Task 5: Marker re-render

**Files:** Create `visuals/marker.py`. Test: `tests/visuals/test_marker.py`.

**Interfaces — Produces:**
- `render_marker(strokes: StrokeSet, size: tuple[int, int], *, line_weight_px: float, ink: str, seed: int, upto: float = 1.0) -> Image.Image`:
  - draws on white with PIL `ImageDraw.line` (`joint="curve"`), using a round line cap made by drawing circles at the joints;
  - applies a wobble by offsetting each point with seeded Gaussian noise of σ = 0.6 px (`numpy.random.default_rng(seed)`);
  - varies the line weight by ±10% per stroke from the same RNG;
  - `upto` in [0, 1] draws only that fraction of the total path length, which the reveal animation needs in Plan 5;
  - is deterministic for the same inputs.
- `marker_seed(video_id: str, scene_id: str) -> int` uses a sha256-based value (reuse the hashing approach of `choose_variant`).

- [ ] **Step 1: Write the failing tests:**
  - same seed → identical bytes;
  - a different seed → different pixels but the same ink coverage within ±15%;
  - `upto=0` gives a blank image;
  - `upto=0.5` gives roughly half the ink;
  - the ink colour is respected;
  - line weight changes the coverage monotonically.
- [ ] **Steps 2–5:** Red, implement, run the gate, then commit `feat: channel marker re-render with seeded wobble`.

### Task 6: Image input for the LLM and the vision check

**Files:**
- Modify `llm/base.py`, `llm/anthropic_client.py`, `llm/ollama.py`, `costs/tracker.py` (TrackedLLM forwards images), `channels/config.py`, and `tests/llm/fakes.py`.
- Create `sketching/__init__.py` and `sketching/vision.py`.
- Tests: extend `tests/llm/*`, and add `tests/sketching/test_vision.py`.

**Interfaces — Produces:**
- `LLMClient.complete_structured(*, system, user, output, images: Sequence[bytes] = ())`, where the images are PNG bytes.
  - **Anthropic:** the user content becomes a list of `{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": …}}` blocks followed by a text block. Verify this against the claude-api skill (`python/claude-api/README.md`, vision section) and the installed SDK.
  - **Ollama:** `messages[0].images = [base64…]`.
- `class VisionConfig(BaseModel): provider: str | None = None; model: str | None = None`. `ChannelConfig.vision: VisionConfig = VisionConfig()`. When this is None, the LLM provider and model are used. Vision pricing uses `prices.vision(provider, model)`.
- `class VisionVerdict(BaseModel): human_figure: bool; face: bool; logo_or_text: bool; style_outlier: bool; mirror_safe: bool; reason: str = Field(max_length=200)`
- `VISION_SYSTEM`. A frozen prompt in `planning/prompts.py` that holds the §6.3 calibration verbatim in meaning:
  - **allowed:** standalone eyes, hands (including a hand holding an object), footprints, paths;
  - **reject:** faces, heads, torsos, stick figures, silhouettes, recognizable likeness, logos, trademarks, copyrighted characters, and any written text;
  - **style outlier:** shading, fills, colour, texture or excess detail compared with the style instructions;
  - **mirror_safe:** false for maps, anything containing text or symbols, and direction-dependent objects.
- `check_sketch(png: bytes, style: StyleConfig, idea_key: str, llm: LLMClient | None) -> tuple[SketchStatus, str | None, bool, VisionVerdict | None]`:
  - `llm is None` → `(UNCHECKED, "vision: not checked", False, None)`. Mirror-safe defaults to False when unchecked.
  - `human_figure`, `face` or `logo_or_text` → `(FAILED, "vision: <fields>; <reason>", mirror_safe, verdict)`.
  - `style_outlier` alone → `(PASSED, "warn: style outlier; <reason>", …)`. The WARN is recorded in `status_reason` and does not block (§6.3).
  - Otherwise → `(PASSED, None, verdict.mirror_safe, verdict)`.
  - The reference sketch is sent as a second image, so the outlier judgment has the style anchor.

- [ ] **Step 1: Write the failing tests:**
  - the Anthropic request body has image blocks before the text block (MockTransport);
  - the Ollama `images` field;
  - TrackedLLM forwards images and estimates their cost with `prices.vision`;
  - `check_sketch` maps each verdict as described above;
  - `VISION_SYSTEM` contains the allowed and rejected lists (Review Focus 2);
  - the no-vision path gives UNCHECKED.
- [ ] **Steps 2–5:** Red, implement, run the gate, then commit `feat: image input for LLM clients and calibrated vision check`.

### Task 7: Generation orchestration

**Files:**
- Create `sketching/generate.py`.
- Modify `library/repository.py`:
  - cache the engine per data dir in `default_library` (Plan 1 carry-over);
  - add `put_strokes(sketch_id, StrokeSet)` and `get_strokes(sketch_id) -> StrokeSet | None`. These are stored as a content-addressed blob with the suffix `.strokes.json`, in a new nullable column `sketch.strokes_hash`, which needs migration 0003.
- Modify `models.py`: `Scene.sketch_id: int | None = None`.
- Test: `tests/sketching/test_generate.py`.

**Interfaces — Produces:**
- `@dataclass class GenerationResult: generated: int; reused: int; bundled: int; quarantined: list[str]; unchecked: list[str]; spent_usd: Decimal`
- `generate_missing(series: Series, channel: ChannelConfig, settings: Settings, *, images: ImageProvider | None = None, vision: LLMClient | None | Literal["auto"] = "auto", library: SketchLibrary | None = None, prices: PriceTable | None = None) -> tuple[Series, GenerationResult]`

**Algorithm, per part and per scene in order:**
1. A series that is not approved raises `PlanningError("approve the series first")`.
2. Run `check_provider_keys`. Create a new `SpendTracker(limit)` for each part, since the limit is per part.
3. Skip VERSE scenes.
4. Re-match each scene with `match_idea`, because the library may have grown since planning:
   - **Reused:** set `scene.sketch_id` to the first passed record, `match="reused"`, and make sure the strokes exist (extract them if missing).
   - **Bundled:** `copy_to` or add the icon into the channel library with `origin=copied`, then check `.status`. If it is FAILED, treat the scene as new. Otherwise set `match="reused"` and `sketch_id`.
   - **New:** `ensure_room(image price + vision price)`, then:
     1. `generate`;
     2. `binarize`;
     3. `check_density`. If that fails, store the sketch FAILED with `density:<reason>`, append the scene to `quarantined` and stop here, with no vision call.
     4. `check_sketch`.
     5. `add_sketch(status, mirror_safe, provenance=Provenance(source="generated", license=<provider terms id>, provider, model, created_at=now UTC, usage_terms=...))`. Check the returned `.status`; a dedupe can return an existing FAILED record.
     6. If it is PASSED or UNCHECKED, extract the strokes (`rtl = channel.language is AR`), call `put_strokes`, and set `scene.sketch_id`.
     7. If it is FAILED, the scene is quarantined: `sketch_id` stays None and the flag `"sketch_quarantined"` is added.
   - `BudgetExceededError` propagates after everything already stored has been kept. Return what was done so far by saving the partially updated series first; the CLI saves before re-raising.
5. The result counts what happened. The returned series is a new object; inputs are not mutated.

- [ ] **Step 1: Write the failing tests** (fake providers, a fake vision LLM, the `library` fixture):
  - happy path: 3 new scenes → 3 passed with strokes stored;
  - silhouette → FAILED with `density:`, and the vision fake has 0 calls (Review Focus 1);
  - a face verdict → quarantined, flagged, no sketch_id (Review Focus 2);
  - a budget that allows exactly 2 → the 3rd raises, the first 2 are in the library, and a re-run reuses them (Review Focus 3);
  - a bundled icon is copied on first use and reused on the second (Review Focus 4);
  - an unapproved series is refused;
  - verse scenes are untouched;
  - unchecked mode, where vision is None.
- [ ] **Steps 2–5:** Red, implement, run the gate, then commit `feat: generate missing sketches with density and vision gates`.

### Task 8: Hardened bundled icons and packaging

**Files:** Modify `visuals/bundled.py`, `pyproject.toml` and `assets/icons/index.toml` (add documentation comments only). Test: `tests/visuals/test_bundled.py`.

**Interfaces — Produces:** `BundledIcons.from_index(path)` validates:
- every key parses with `parse_idea_key` and has context `generic`;
- every file path is relative, has no `..`, and exists under the index's directory;
- there are no duplicate keys;
- every entry has `license` and `source`.

Any violation raises `ChannelConfigError` naming the entry. Package `assets/icons` into the wheel with hatch `force-include` under `narratoflow/assets/icons`. `default_bundled_icons()` looks in the package first, then the dev checkout.

- [ ] **Step 1: Write the failing tests:** each validation error case; a valid index loads; and the wheel built with `uv build` contains `narratoflow/assets/icons/index.toml` (run in a tmp dir and mark the test slow but keep it in CI).
- [ ] **Steps 2–5:** Red, implement, run the gate, then commit `feat: validated bundled icon index packaged in the wheel`.

### Task 9: Quarantine resolution

**Files:** Create `sketching/resolve.py`. Test: `tests/sketching/test_resolve.py`.

**Interfaces — Produces:**
- `regenerate(series, scene_id, channel, settings, **deps) -> Series` runs the Task 7 new-sketch path for one scene. This is a paid call, so the caller confirms.
- `swap_idea(series, scene_id, new_key, description) -> Series` validates the key with `validate_idea` (Plan 2), sets `idea_key` and `match`, clears the quarantine flag, and leaves generation to the next `sketch generate` run.
- `use_text_card(series, scene_id) -> Series` sets `scene.visual.source = VisualSource.TEXT_CARD`, `sketch_id=None`, and clears the flag. The text card shows the scene title, or the first words of the narration, in Plan 5.
- An unknown scene id or a scene that is not quarantined raises `PlanningError`. `swap_idea` to a key that violates the rules raises `PlanningError` with the violations.

- [ ] **Step 1: Write the failing tests:** each path; validation errors; inputs are not mutated.
- [ ] **Steps 2–5:** Red, implement, run the gate, then commit `feat: quarantine resolution (regenerate, swap idea, text card)`.

### Task 10: CLI

**Files:** Modify `cli.py` and `README.md`. Test: `tests/test_cli.py`.

**Interfaces — Produces:**
- `narratoflow sketch generate SERIES_JSON --channel PATH`:
  - runs `generate_missing` and saves the series atomically, also on `BudgetExceededError`, before exiting 1;
  - prints a summary line (generated, reused, quarantined, unchecked, spent);
  - prints one line per quarantined scene with the resolve hint.
- `narratoflow sketch resolve SERIES_JSON SCENE_ID (--regenerate [--yes] | --swap KEY --description TEXT | --text-card) --channel PATH`. `--regenerate` asks `typer.confirm` with the estimated price unless `--yes` is given.
- Errors go through `main()` as before.

- [ ] **Step 1: Write the failing tests** (patch `generate_missing` and the resolvers): the summary output; the series is saved on a budget error; regenerate asks for confirmation; the option groups are mutually exclusive.
- [ ] **Steps 2–5:** Red, implement, run the gate, update the README Quick start, then commit `feat: sketch generate and resolve commands`.

---

## Done when

- The gate is green in CI.
- With fakes, `sketch generate` on the Plan 2 reference series stores passed sketches with strokes. A silhouette is rejected before any vision spend. A face verdict quarantines the scene. A budget stop keeps the work already done.
- Bundled icons are validated and packaged.
- Every `add_sketch` and `copy_to` caller checks `.status`.
