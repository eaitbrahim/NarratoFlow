# Technical Review of PRD v1.0 & SRD v1.0

**Date:** 2026-11-10
**Scope:** [`PRD.md`](PRD.md), [`SRD.md`](SRD.md)
**Verdict:** The vision is clear and the stack is mostly well chosen. Before building, fix several factual and licensing errors (§1) and fill in the missing architecture pieces (§3). Without them the "100% FOSS / zero-cost" promise and the "authentic whiteboard" promise won't hold.

**Status:** Corrections C1–C10 folded into PRD/SRD v1.1 (#1); §5 answered in PRD §8.

---

## 1. Corrections (factual / licensing errors)

These claims are wrong as written. Each one would either break the FOSS promise or lead to a design that doesn't work.

| # | Claim in docs | Reality | Fix |
|---|---|---|---|
| C1 | **Remotion is FOSS** (PRD §1, §2) | Remotion is *source-available*. Companies above a small headcount need a paid company license. | Drop Remotion from the core. Use **FFmpeg + NumPy/OpenCV frame rendering** (or **Manim CE**, MIT) as the compositor. |
| C2 | **edge-tts is FOSS / zero lock-in** | The `edge-tts` *client* is open source (LGPL/GPL), but it calls an **undocumented Microsoft endpoint**. That endpoint needs a network connection, can be rate-limited or changed without notice, and has unclear terms for commercial reuse. It is the biggest vendor dependency in the system. | Keep it as the default engine, but put it behind a `TTSEngine` interface. Make a local engine (Piper) a first-class fallback, cache all synthesized audio, and document the risk. |
| C3 | **XTTS-v2 / F5-TTS** are open-source alternatives | XTTS-v2 weights use the **Coqui Public Model License (non-commercial)**, and Coqui the company has shut down. The official F5-TTS weights are **CC-BY-NC**. | List them as *non-commercial optional* engines only. Use **Piper** (permissive license, has an Arabic voice `ar_JO-kareem`) as the commercial-safe offline option, and check the license of each voice model. |
| C4 | **Farasa** as an open diacritizer | Farasa is distributed under a research/non-commercial license (Java). | Remove it, or mark it non-commercial. |
| C5 | **Whisper gives "sub-millisecond word alignment"** | Whisper word timestamps are typically off by tens to hundreds of milliseconds. | Use edge-tts boundaries when they are available. Otherwise use **CTC forced alignment** (e.g. `ctc-forced-aligner` / MMS, or WhisperX). Run alignment on the *undiacritized* text. |
| C6 | **`arabic-reshaper` + `python-bidi` before writing `.ass`** (SRD §2.5) | libass already shapes text with HarfBuzz + FriBidi. Pre-reshaping converts letters to *presentation forms* and reverses their order, so libass ends up processing the text twice. You get mangled joins and misplaced harakat, which is exactly the bug this step was meant to prevent. This also contradicts PRD §4.5. | Write **logical-order Unicode** into `.ass` and let libass shape it. Burned-in text drawn with Pillow should use the **raqm** layout engine (`ImageFont.Layout.RAQM`). Drop `arabic-reshaper` and `python-bidi`. |
| C7 | **Potrace output can be animated with `stroke-dashoffset`** | Potrace traces the *outline* of filled regions. A 3px marker line becomes a thin closed shape, so a dash animation draws both edges of every line and looks wrong. | Use **centerline** vectorization (skeletonize, then trace and order the paths, or `autotrace -centerline`). This is what makes the "hand draws the line" effect possible. See §2.3. |
| C8 | **SDXL Turbo + LineArt ControlNet for text-to-sketch** | ControlNet needs a *control image*, so on its own it doesn't fit text-to-image. SDXL in 6 GB of VRAM is tight. SDXL / SD1.5 weights are OpenRAIL (open-weights, not OSI-approved). | For text-to-sketch, use a **line-art LoRA** on SDXL or Flux.1-schnell (Apache-2.0; note that Flux.1-dev is non-commercial). Use ControlNet only for the image-to-sketch path. Describe the weights as "open-weights" rather than FOSS. |
| C9 | **Mishkal "validates Classical Arabic grammar and excludes colloquial forms"** | Mishkal is a rule-based diacritizer. It does not detect dialect, and its accuracy is modest on modern prose. | Add a separate **dialect-identification** step (CAMeL Tools DID) that *flags* text rather than rejecting it silently. Benchmark diacritizers before choosing one (see §2.1). |
| C10 | **Requirements list** | `moviepy>=1.0.3` allows MoviePy 2.x, which has a breaking API. `faster-whisper`, `streamlit`, and `cairosvg` are listed but either unused or duplicated, while `diffusers`, `torch`, and `ollama` are missing. | Use `pyproject.toml` with **optional extras** (`[tts-local]`, `[diffusion]`, `[ui]`, `[align]`) and a lock file (`uv.lock`). |

---

## 2. Enhancements by module

### 2.1 Arabic NLP and Tashkeel
- **Skeleton-preservation guard (must-have).** Any diacritizer, and LLMs in particular, can quietly *change words*. After diacritization, strip the harakat and assert that the consonantal skeleton is byte-identical to the input. If the check fails, reject the result or fall back. *(Implemented in `narratoflow.text.tashkeel.verify`.)*
- **Pluggable diacritizers with a benchmark.** Offer Mishkal (baseline), CATT and other open neural diacritizers, and an LLM option. Score them on a small gold set using DER/WER, then choose by evidence.
- **Protected spans.** Never auto-diacritize **Qur'anic verses** or **hadith**. Detect them, or let the author mark them (`«…»` or `{{quran:2:255}}`), and insert verified text (e.g. Tanzil Uthmani/Simple). For a Classical Arabic audience this matters for both reputation and correctness.
- **Pronunciation lexicon.** Keep a per-project dictionary for names, loanwords, and numbers, applied before TTS. Normalize numerals to words in Arabic, since TTS engines misread `١٢٣` and `123` differently.
- **Partial-diacritics mode.** Fully vocalized captions are hard to read. Keep two versions of the text: a *TTS text* (full tashkeel) and a *caption text* (diacritics only where words are ambiguous, or none).

### 2.2 Script and scene planning
- **2–4 seconds is too short for whiteboard scenes.** Drawing a doodle convincingly takes roughly 60–70% of a 5–10 second scene, and the rest is a hold. Let narration length set scene duration, with configurable minimum and maximum values.
- **Structured LLM output.** Use JSON-schema/`format` output in Ollama, validated with Pydantic and retried on failure. Don't parse free text.
- **Make the model configurable.** "Qwen2.5:7b" is already dated. Name the model in config, not in requirements.
- **Choose a visual source per scene.** The planner decides between `icon` (library match), `diffusion`, `image-to-sketch` (user upload), and `text-card`. Icon-first is cheaper, faster, and more consistent.
- **Hook and retention structure.** The PRD promises "high-retention" but specifies nothing for it. Add a 3-second hook scene, pattern interrupts (zoom or colour accents) every N scenes, and an end-card/CTA scene.

### 2.3 Visuals (the "authentic whiteboard" core)
- **Stroke-ordered reveal instead of a diagonal wipe.** A diagonal alpha wipe looks like a cheap transition, and viewers can tell the difference. Pipeline: binarize → **skeletonize** → extract polylines → **order the strokes** (outer contours first, then by spatial proximity) → animate the stroke tip with the hand sprite following it. Pace the drawing by stroke length.
- **Better line extraction than dodge-divide.** The dodge/divide filter produces grey *pencil* shading, which doesn't trace well. Use **XDoG** or the line-art annotators from `controlnet_aux`, then binarize.
- **Asset library with clear licensing.** The "~500 icons" need sources with known licenses: Tabler (MIT), Lucide (ISC), Phosphor (MIT). OpenMoji is CC-BY-SA, which requires attribution and share-alike. Keep a `LICENSES.md` per asset pack and search the library by embeddings so the planner can match concepts to icons.
- **Style consistency.** Pin one LoRA, seed, and stroke width per project, so that scene 12 looks like scene 1.

### 2.4 Compositor and layout
- **One storyboard, two layouts.** Describe scenes as a scene graph in *normalized coordinates*, and give each aspect ratio a **layout strategy** (side-by-side panorama for 16:9, vertical stack with camera pan for 9:16). Don't build two separate pipelines.
- **Platform safe zones.** TikTok, Shorts, and Reels overlay UI on the bottom ~20% and the right edge. Keep captions and key doodles inside the safe area for each layout.
- **Render engine.** Rendering every frame in MoviePy is slow. Render frames with NumPy/OpenCV (or Skia/Cairo for vectors), **pipe raw frames into FFmpeg**, and run the per-scene renders in parallel before concatenating.
- **Camera system.** Add a virtual camera (pan/zoom keyframes with easing) so movement isn't hard-coded per layout.

### 2.5 Audio
- Do loudness normalization with **FFmpeg `loudnorm` to −14 LUFS** (the YouTube and TikTok target) instead of ad-hoc pydub gain. Use `sidechaincompress` for ducking.
- Drive SFX from stroke events (marker squeak while drawing, silence on pen-up). Randomize sample choice to avoid obvious looping.
- Optional background music bed from CC0 sources. Store attribution metadata even for CC0.

### 2.6 Captions
- **Karaoke highlighting in RTL text is a known libass edge case.** Prototype it early (spike issue). The fallback is per-word ASS events instead of `\k` tags.
- Keep captions to 1–2 lines in a fixed max-width box. Arabic lines break differently than English, so wrap on word boundaries using the aligned word list.

### 2.7 Review station
- **Choose Gradio.** It handles audio, image, and video components better and suits a pipeline UI. Drop Streamlit to keep a single UI dependency.
- The UI should *edit the storyboard file* and re-run only the stages that changed. It must not hold state of its own.

---

## 3. Architecture gaps (missing from both docs)

1. **Intermediate representation (most important).** The docs pass loose files (`voice.mp3`, `timing.vtt`, `sketch.png`) between stages. Define a single versioned **`storyboard.json`** (Pydantic models: `Project → Scene → Narration/WordTiming/Visual/Layout`). Each stage reads and writes it, and the HITL UI edits it. *(Scaffolded in `narratoflow.models`.)*
2. **Content-addressed caching and resumability.** Key each stage's output by a hash of its inputs and config. Changing one scene's narration should re-synthesize only that scene. *(Scaffolded in `narratoflow.pipeline.cache`.)*
3. **Determinism.** Fix seeds, version models, and pin voice IDs, all recorded in the storyboard, so a render can be reproduced.
4. **Plugin interfaces.** Use `Protocol`s for `Diacritizer`, `TTSEngine`, `Aligner`, `VisualSource`, `LLM`, and `Renderer`. Select implementations from config, which makes testing straightforward with fakes.
5. **Non-functional requirements.** Add targets for render time (e.g. a 3-minute video renders in under 10 minutes on CPU-only for the icon path), minimum hardware (CPU-only path vs GPU path), and disk/caching budgets.
6. **Error handling.** Define behaviour for edge-tts outage (fall back to Piper), LLM invalid output (retry, then fail the scene), and missing icon matches (fall back to a text card).
7. **Testing strategy.** Unit tests for text, timing, and layout math. Golden-image tests on single frames. One end-to-end smoke test that renders a 5-second video in CI with the fake engines.
8. **Compliance.** YouTube requires AI-content disclosure for realistic synthetic media, and TikTok has labeling rules for AI-generated content. Add a disclosure flag and auto-generated description text. Keep license attribution for all bundled assets.

---

## 4. Document structure suggestions

- **PRD** should state *what and why*: personas (e.g. Islamic-studies educator, history explainer channel), goals and **success metrics** (time from text to video, % of scenes accepted without edits, retention proxy), **non-goals** (no dialects, no real-time, no cloud SaaS), and **MVP phasing**. It currently contains code-level detail that belongs in the SRD.
- **SRD** should state *how*: the IR schema, interfaces, NFRs, failure modes, dependency policy, and a license matrix.
- Record key decisions as ADRs in `docs/adr/` (compositor choice, TTS strategy, IR format).

### Proposed phasing
| Phase | Deliverable |
|---|---|
| **M0 – Foundations** | Repo, CI, storyboard IR, cache, CLI, fake engines, E2E smoke test |
| **M1 – Talking Whiteboard (MVP)** | Text ingest → tashkeel (+guard) → edge-tts with word timings → icon-library visuals → stroke-ordered reveal → captions → 16:9 MP4 |
| **M2 – Vertical and Polish** | 9:16 layout + safe zones + camera, SFX + loudnorm, Piper fallback, karaoke |
| **M3 – Generative Visuals** | LLM scene planner, image-to-sketch, diffusion LoRA path |
| **M4 – Review Station** | Gradio HITL UI with partial re-render |

---

## 5. Open questions for the product owner
1. Is **commercial use** a requirement? The answer decides whether XTTS-v2, F5-TTS, Flux-dev, and Farasa are allowed at all.
2. Is the edge-tts network dependency acceptable for v1, or is fully offline a hard requirement?
3. Will content include Qur'an or hadith? If yes, protected spans are an M1 requirement.
4. What is the minimum target hardware (CPU-only laptop vs GPU)?
