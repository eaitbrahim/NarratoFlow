# Product Requirements Document (PRD)
## Project: Open-Source Whiteboard & Sketch AI Video Engine ("NarratoFlow")

> **Status:** v1.1, 2026-11-13; supersedes v1.0 (see git history). Folds in the corrections C1 to C10 of [`REVIEW.md`](REVIEW.md) and answers its section 5 (here in section 8). The PRD says what and why; how it is built is in [`SRD.md`](SRD.md) and `docs/adr/`.
>
> A line marked **Recommendation, owner to confirm** is a product decision the code does not make; it is the cheapest answer to undo (one docs edit).

---

## 1. Status and changes from v1.0

| # | v1.0 claim | Correction in v1.1 | Section |
| :--- | :--- | :--- | :--- |
| C1 | Remotion is FOSS | Remotion is source-available and was dropped; the compositor is NumPy/Pillow frames piped into ffmpeg (ADR 0002) | 2, 6.5 |
| C2 | edge-tts is free with no lock-in | edge is the default free fallback behind a voice chain, with Piper as a local fallback and a cache (ADR 0003); its terms are a publish warning | 2, 6.3 |
| C3 | XTTS-v2 and F5-TTS are open-source voice cloning | Their weights are non-commercial; neither is implemented; voice cloning is a non-goal | 5, 6.3 |
| C4 | Farasa is an open diacritizer | Farasa is research and non-commercial; it is not used | 6.7 |
| C5 | Whisper gives sub-millisecond word alignment | Word times come from provider spans, else a forced aligner, else an estimate; precision is not claimed | 6.3 |
| C6 | Reshape and reorder Arabic before writing subtitles | Logical-order Unicode goes straight to libass, which shapes it; no reshaper | 6.4 |
| C7 | Potrace output animates with a dash offset | Centerline strokes are extracted and ordered for the reveal | 6.2 |
| C8 | SDXL Turbo with a ControlNet for text-to-sketch | A line-art diffusion provider with pinned model, LoRA and seed; weights are "open-weights", not FOSS | 2, 6.2 |
| C9 | Mishkal rejects colloquial Arabic | Dialect is flagged, never rejected; the diacritizer is chosen by benchmark (ADR 0005) | 6.7 |
| C10 | Loose requirements list | Optional extras and a lock file | 2 (SRD 7) |

---

## 2. Vision (corrected)

**NarratoFlow** turns a written script into **whiteboard animation videos**: a hand draws each sketch while a narrator reads, with captions burned in. It targets **TikTok and YouTube Shorts (9:16)** first; a 16:9 file is a pillarboxed copy of the 9:16 video (6.5).

- **The code is open source.** The core is Apache-2.0 (`pyproject.toml`). Models are "open-weights" where they are used, never "FOSS": SDXL is OpenRAIL++-M and FLUX.1-schnell is Apache-2.0, but some model weights carry use-based or non-commercial terms. The SRD licence matrix lists every one.
- **Free by default, paid by choice.** Bundled icons, uploads, text cards, Piper, Mishkal and local diffusion cost nothing. Paid providers (an LLM, OpenAI images, ElevenLabs) are optional and capped per part by `ChannelConfig.spending_limit_usd` and `SpendTracker`; nothing spends past the cap or without an estimate.
- **Classical Arabic and English** with pristine elocution; dialect is flagged, not rewritten.
- **Authentic whiteboard look:** strokes are drawn in order by a hand sprite, not wiped in.
- **No browser or scene-graph renderer.** Remotion is source-available, not free software, and was removed (REVIEW C1, ADR 0002). Frames are drawn in Python and piped to ffmpeg.

### Benchmark mapping (corrected)

| Commercial reference | NarratoFlow replacement | Caveat |
| :--- | :--- | :--- |
| Voice SaaS (ElevenLabs) | `edge` (free Microsoft service) and Piper (local); ElevenLabs itself as an optional paid primary | edge has no explicit commercial licence (publish warns); Piper voices carry their own licences; voice cloning engines with non-commercial weights are not offered |
| Image SaaS (Ideogram, Midjourney) | Bundled icons, uploads (XDoG line art), OpenAI images (paid), local line-art diffusion (FLUX.1-schnell, Apache-2.0) | SDXL is open-weights with use-based terms; FLUX.1-dev is non-commercial and fails publish |
| Canva, VideoScribe | NumPy/Pillow frames piped to ffmpeg with a hand sprite and centerline strokes | 9:16 native; 16:9 is pillarboxed |
| Commercial Arabic diacritizers | An LLM (default), Mishkal (free, offline, measured), behind a skeleton guard | Mishkal DER 25.45, WER 65.03 on the gold set (ADR 0005); other engines not yet measured |
| Auto-captions | Provider word spans, else forced alignment, libass for shaping | Alignment precision is not claimed; the `align` extra is not commercial-safe |

---

## 3. Personas

Marked **Recommendation, owner to confirm** where the code is silent about who the user is.

1. **The Islamic-studies educator.** Writes Classical Arabic scripts that quote the Qur'an and want verified verse text, a chosen reciter and `riwaya`, and no invented vowels in a verse. The code is built for this person: verse scenes (`SceneKind.VERSE`, `VerseRef`), protected narration (`Narration.protected`), the Tanzil corpus (`quran/corpus.py`), the reciter allow-list (`quran/reciters.py`), `riwaya` and `reciter` in the channel file. Recommendation, owner to confirm: a solo creator or small institution publishing short videos on a monetised channel, who reviews every part before it goes out.
2. **The history explainer channel.** Writes narration-only Arabic series, uses scene planning and the sketch library to keep one style across many parts. Evidence: `tests/fixtures/channels/history.toml` (an Arabic channel with a local LLM and OpenAI images). Recommendation, owner to confirm: runs a series with a monthly spending cap and wants the same look from the first part to the twentieth.

Both are **creators, not developers**: they edit a channel file and a script, run commands, and review in a page (section 7). Neither is expected to write code.

---

## 4. Success metrics

### Goals
- Turn a script into publishable 9:16 whiteboard videos with the creator editing as little as possible.
- Keep spend explicit: an estimate before every paid action, a cap per part, and a per-part record of what was spent.
- Keep output reproducible: the same inputs give the same frames (seeds are pinned, the worker count does not change frames).
- Give the creator what a platform asks for: an AI-disclosure flag, licence provenance, and publish checks.

### Success metrics
Each metric is tied to an artifact that measures it today. Where none exists the line says so.

| Metric | Target | Measured by |
| :--- | :--- | :--- |
| Time from text to video | A 3-minute part renders in under 600 s on CPU; measured 84.5 s | `docs/benchmarks/2026-11-13-render-cpu.md`; stage wall times in the logs |
| Scenes accepted without edits | Share of scenes with no quarantine, no flag and no station edit | `GenerationResult.quarantined` and the part `flags`; edits are visible as changes to `series.json` through the review station. No aggregate is computed yet; this is a Recommendation, owner to confirm: a target of 80% accepted. |
| Retention proxy | Every part has a hook, a recap, a teaser (non-final parts) and an end card (final part) | The `Part` fields `hook_text`, `recap_text`, `teaser_text`, `end_card_text`. No viewer retention is measured; building that is out of scope. |
| Diacritization quality | DER and WER on a pinned gold set | `scripts/bench_tashkeel.py`, ADR 0005 |
| Cost per part | At or under the channel's `spending_limit_usd` | `Part.spent_usd`, `Part.spend`, the publish report |

## 5. Non-goals

- **No dialects.** Dialect is flagged (`check-dialect`, REVIEW C9), never converted or rejected.
- **No real-time** generation or live video.
- **No cloud SaaS.** NarratoFlow runs on the creator's machine; there is no hosted service, account or multi-user store.
- **No voice cloning** (REVIEW C3): the cloning engines' weights are non-commercial, so none is built.
- **No native 16:9 layout in v1.** The 16:9 file is a pillarboxed copy of the 9:16 video; a native panorama layout is follow-up #163.
- **No viewer-retention analytics.**

---

## 6. Functional requirements

What the product does, at the level of behaviour. How is in the SRD.

### 6.1 Script ingestion and planning
- Takes a text or Markdown script and rejects corrupt input (copy-pasted PDF text) before any spend (`check-input`).
- Declares the work's author and rights; a third-party work still in copyright is blocked before any spend (`ScriptSource`, `ScriptBlockedError`).
- Splits the script into parts of a configured length (default 60 to 90 s) and scenes, with the author's words kept exactly; an LLM proposes the split and the result is verified against the script.
- Detects Qur'anic quotations against the verified corpus and makes them verse scenes. The creator approves the plan, with its cost estimate, before anything is generated (`approve`).
- Each scene gets a concept key; a visual source is chosen per scene: bundled icon, library reuse, generated sketch, uploaded image or text card. Icon-first is cheaper and more consistent.
- Every part has a hook, a recap, a teaser and (final part) an end card.

### 6.2 Visuals
- **Bundled icons first** (Tabler, MIT), then library reuse by concept, then generation.
- **Generation** is either OpenAI images (paid) or a local line-art diffusion provider (`lineart-diffusion`: the channel names the model, FLUX.1-schnell being the documented one, with an optional LoRA, pinned model, seed and licences). A sketch must be clean line art; a failed check quarantines it and the creator resolves it.
- **Upload your own image** (`sketch upload`): the image becomes line art through XDoG, with the creator's licence and source recorded. It is free, and an uploaded sketch is never reused for another scene or regenerated by accident.
- **Reveal:** strokes are extracted as centerlines, ordered, and drawn by a hand sprite (REVIEW C7).
- Style consistency: one reference sketch, one line weight and one seed pin per channel; a change gives a new style version so old sketches are not reused.

### 6.3 Voice and audio
- A chain of voices: the channel's primary voice, then fallbacks (edge, then Piper). A part is always voiced by one voice.
- Word times come from the provider when it reports them, else a forced aligner, else an estimate that says so.
- **Verse scenes are recited, not synthesized**: a verse plays an allow-listed reciter's recording. No reciter is bundled (the terms of the common sources do not allow monetised use), so verse scenes fail publish checks until a licensed reciter is added.
- Marker sound effects, ducking and loudness mastering to -14 LUFS.
- **Hadith** is ordinary narration. The code gives it no protected handling: it is diacritized like any other sentence and checked only by the skeleton guard. Recommendation, owner to confirm: this stays true for v1, with protected handling for hadith deferred until a verified hadith corpus is chosen.

### 6.4 Captions and typography
- Captions are burned in with logical-order Arabic that libass shapes (REVIEW C6); fonts are Amiri (Arabic) and Noto Sans (Latin), both SIL OFL.
- Karaoke-style word highlighting follows the word times.

### 6.5 Layout and rendering
- Output is 1080x1920 at 30 fps, H.264 and AAC, one MP4 per part, with a camera (pan and zoom) and platform safe zones.
- A part whose inputs did not change is not re-rendered.
- 16:9 is a pillarboxed copy of the 9:16 export, made from the review station (section 7).

### 6.6 Publishing
- `publish` checks each part against the platform rules (duration, size, codecs, loudness, licences, AI-content label, content policy, verse integrity) and writes a report. A FAIL blocks the part; a WARN is recorded.
- The creator is told what each platform wants for AI disclosure, and every sketch carries licence provenance.

---

### 6.7 Linguistic policy (corrected)

- **Dialect is flagged, never rejected** (REVIEW C9). `check-dialect` marks sentences that read as dialect, using CAMeL Tools, and always exits 0.
- **The diacritizer is chosen on evidence** (ADR 0005). The default stays an LLM, because no row can show it worse (it was not run: paid); Mishkal is the measured free, offline option (DER 25.45, WER 65.03; DER 18.83, WER 38.79 without case endings). Follow-ups #155 (LLM) and #156 (CATT).
- **A skeleton guard is mandatory** for every engine: after vocalizing, the consonants must be identical to the input, or the result is rejected and the source text is kept.
- **Protected spans for the Qur'an.** Verse text comes verified from the Tanzil corpus and is never auto-diacritized. Hadith is not protected (6.3).
- **Two texts per scene:** a vocalized text for the voice and a lighter caption text, so captions stay readable.
- **English** narration is supported with the same planning and rendering.

---

## 7. Phasing and workflow

The pipeline is a set of commands over one file, `series.json` (SRD 1). Each milestone adds commands; the GitHub milestone is named in the last column.

| Phase | Deliverable | Commands | GitHub milestone |
| :--- | :--- | :--- | :--- |
| **M0 Foundations** | Repo and CI, the series IR, the stage cache, the CLI, fake engines, an end-to-end smoke test, the sketch library, these documents | `validate`, `check-input`, `channel check`, `library keys`, `library icons`, `library copy` | M0 – Foundations |
| **M1 Talking Whiteboard (MVP)** | Script to planned series, vocalized text, voice with word times, icon visuals, stroke reveal, captions | `plan`, `approve`, `sketch generate`, `sketch resolve`, `audio build`, `render` | M1 – Talking Whiteboard (MVP) |
| **M2 Vertical and Polish** | 9:16 layout, safe zones, camera, SFX, loudness, Piper fallback, karaoke, publish checks | `review` (content review), `publish` | M2 – Vertical & Polish |
| **M3 Generative Visuals** | LLM scene planner, image to sketch, diffusion provider, dialect check, the diacritizer benchmark | `sketch upload`, `check-dialect`, `lineart-diffusion` in the channel file | M3 – Generative Visuals |
| **M4 Review Station** | A local Gradio page that edits `series.json` and re-runs only what an edit needs | `ui` | M4 – Review Station |

### Review station (shipped in #165)
`narratoflow ui` opens a local page over `series.json`. It holds no state of its own and refuses a stale page. Per scene the creator can hear the clip, edit the vocalized and caption text, set the minimum duration and draw fraction, upload an image, regenerate the sketch, switch to a text card and rebuild the scene. Rebuilding a scene re-runs only that scene's sketch, voice (the cache serves the other scenes) and the render of its part; other parts are untouched. Paid actions show an estimate and need a confirm. A per-scene render cache inside a part is follow-up #164. Export 9:16 checks and exports every part like `publish`; Export 16:9 writes the pillarboxed copy.

### Workflow
1. Write the script and a channel file (voice, style, spending cap, reciter).
2. `plan` splits it into parts and scenes; `approve` shows the estimate and locks the plan.
3. `sketch generate` draws or reuses a sketch per scene; the creator resolves any quarantined scene, or uploads an image.
4. `audio build` vocalizes, voices and times each part.
5. `render` draws the video; `review` (optional) runs a content check; `publish` checks and exports.
6. The review station can do steps 3 to 5 scene by scene, with confirmation before any spend.

---

## 8. Open questions answered

REVIEW section 5 asked four questions. Each is answered from what the code and ADRs already decide; an answer the code does not decide says so.

1. **Is commercial use a requirement?** The code decides that output is meant for monetised platforms: `platform_rules.toml` warns on `edge-tts-terms` ("no explicit commercial licence"), `elevenlabs-terms` ("commercial use needs a paid ElevenLabs plan") and `local-model-license`, and fails FLUX.1-dev, whose licence is non-commercial; the project is Apache-2.0 (`pyproject.toml`). The non-commercial voice cloning engines and the non-commercial Arabic diacritizer have no code (ADR 0003, alternatives). **Answer: commercial-safe defaults are a requirement. Recommendation, owner to confirm.** One gap is known: the optional `align` extra is documented non-commercial only, and publish warns (`ctc-forced-aligner-terms`) when word timings come from it (SRD 8, #166).
2. **Is the edge-tts network dependency acceptable for v1, or is offline a hard requirement?** The code decides: **offline is not a hard v1 requirement.** The default fallback voice is edge, which needs the network (ADR 0003); the `anthropic` LLM provider, when a channel selects it, needs the network (`LlmConfig.provider` is required and has no default); the verse recording fetch needs the network (`audio/build.py`). An offline-capable path exists, with gaps: Piper (voice), Mishkal (diacritizer), Ollama (LLM), local diffusion (sketches) and the `.cache` TTS cache. Verse scenes need a recording and so the network, and no reciter is shipped.
3. **Will content include Qur'an or hadith?** The code decides: **Qur'an, yes.** Verse scenes (`SceneKind.VERSE`, `VerseRef`), `Narration.protected`, the Tanzil corpus (`quran/corpus.py`, `scripts/fetch_quran.py`), the reciter allow-list (`quran/reciters.py`) and the verse render checks (`render/build.py`, `_check_verse_pages`) are built, so protected spans are an M1 feature already in place. **Hadith is ordinary narration** with no protected handling. Recommendation, owner to confirm.
4. **What is the minimum target hardware?** The code decides **CPU-only for the default path**: ADR 0002 has no GPU path, and a 180 s part renders in 84.5 s with the default workers on a 16-core arm machine (181.8 s on one worker), against a 600 s target. A GPU is needed only for `lineart-diffusion` and `local-diffusion`; its VRAM is not measured (#160). **The minimum core count and RAM are not decided by the code. Recommendation, owner to confirm:** a 4-core CPU with 8 GB RAM and an ffmpeg built with libass, with the render target re-measured there.
