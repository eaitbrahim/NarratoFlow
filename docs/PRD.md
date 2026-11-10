# Product Requirements Document (PRD)
## Project: Open-Source Whiteboard & Sketch AI Video Engine ("NarratoFlow")

> **Status:** v1.0 (original draft). See [`REVIEW.md`](REVIEW.md) for the technical review and proposed v1.1 amendments.

---

## 1. Executive Summary & Vision
**NarratoFlow** is a **100% Free and Open-Source Software (FOSS)** automated and semi-automated pipeline that converts text documents into high-retention **Whiteboard Animation Videos** for **YouTube (16:9 landscape)** and **TikTok / YouTube Shorts (9:16 vertical)**.

The platform eliminates recurring SaaS subscription costs and proprietary vendor lock-in (such as ElevenLabs, Ideogram, or Canva) by employing state-of-the-art open-source libraries, open-weight models, and self-hostable pipelines.

### Core Pillars
1. **Classical Arabic (العربية الفصحى) & English**: Pristine elocution without colloquial dialects, powered by open diacritization tools and free high-grade neural TTS.
2. **Authentic Whiteboard Animation**: Hand-drawn stroke reveals, vector doodles, and line-art sketches powered by open-source sketch generation and SVG path tracing.
3. **Zero-Cost Operation**: Built entirely on free Python libraries, open-weights models (via Hugging Face / Ollama / Diffusers), and open video composition frameworks (Remotion / MoviePy / FFmpeg).

---

## 2. Open-Source vs. Proprietary Benchmark Mapping

| Commercial Reference | NarratoFlow Free & Open-Source Replacement | Key Advantage |
| :--- | :--- | :--- |
| **ElevenLabs** | **`edge-tts`** (free neural Microsoft voices) + **`XTTS-v2` / `F5-TTS` / `Piper`** (local open-source voice cloning). | Zero API costs; native Classical Arabic support (`ar-SA-HamedNeural`, `ar-SA-ZariyahNeural`) with perfect diacritics and no token limits. |
| **Ideogram / Midjourney** | **SDXL / Flux.1 Schnell with LineArt ControlNet / LoRA** + **OpenCV & Potrace** vectorization. | Fully offline, customizable sketch styles, black & white line art output, and zero generation fees. |
| **Canva / VideoScribe** | **Remotion (FOSS)** or **Python MoviePy + OpenCV hand-draw mask engine**. | Full programmatic control over hand-drawing wipes (`stroke-dashoffset`), dynamic camera zoom/pan, and dual 16:9 & 9:16 layout exports. |
| **Commercial Arabic Diacritizers** | **`Mishkal` / `Farasa`** or local open LLM (**Qwen 2.5 / Llama 3 via Ollama**). | Privacy-first, local text vocalization (*Tashkeel*) guaranteeing grammatically sound Classical Arabic. |
| **Commercial Auto-Captions** | **`faster-whisper`** (local Whisper engine) + **`FriBidi` / `HarfBuzz` / `libass`**. | Sub-millisecond word alignment; open classical fonts (Amiri, Scheherazade New) with zero BiDi rendering bugs. |

---

## 3. Linguistic Boundaries: Pure Classical Arabic (الفصحى)

- **Strict Non-Dialect Policy**: Rejects Egyptian, Levantine, Gulf, and Maghrebi colloquial words.
- **Automated Open Tashkeel (التشكيل)**:
  - Undiacritized classical text is enriched with full grammatical vowels (*Fatha, Damma, Kasra, Sukun, Tanwin, Shadda*) using `mishkal-tashkeel`.
  - This ensures open neural TTS engines pronounce case endings (*I'rab*) with classical accuracy.
- **English Language Support**: Clear educational, documentary, and storytelling pacing.

---

## 4. Key Functional Requirements

### 4.1 Document Ingestion & Scene Scripting
- Ingests raw text, Markdown, or PDF documents.
- Uses local LLMs (e.g., `Qwen2.5:7b` via Ollama) or free-tier APIs to segment text into 2–4 second visual concepts.
- Generates line-art doodle prompts formatted specifically for sketch diffusion models.

### 4.2 Open-Source Sketch & Graphic Generation
- **Line-Art Diffusion Engine**:
  - Stable Diffusion XL (SDXL) or Flux.1 Schnell running locally via `diffusers` or ComfyUI, paired with LineArt/Scribble LoRAs.
  - Generates crisp black ink illustrations on pure white backgrounds (`#FFFFFF`).
- **Image-to-Sketch Converter (OpenCV + Potrace)**:
  - Built-in algorithm: Converts any raster image or diagram into a pencil/marker sketch using adaptive edge filtering, then vectorizes it into SVG paths with `potrace`.
- **Open Vector Asset Library**:
  - Bundled offline library of common whiteboard doodles (arrows, callouts, lightbulbs, gears, scrolls, icons).

### 4.3 Free Voice Synthesis & Audio Engineering
- **Primary Voice Engine (`edge-tts`)**:
  - Free Python wrapper for Microsoft's high-fidelity neural voices.
  - Native Classical Arabic: `ar-SA-HamedNeural` (authoritative, clear classical tone) and `ar-SA-ZariyahNeural` (articulate female classical tone).
  - Native English: `en-US-GuyNeural`, `en-US-JennyNeural`, `en-GB-RyanNeural`.
  - Outputs MP3 audio with native **word-level subtitle metadata** at zero cost.
- **Secondary Local Voice Engine (`XTTS-v2` / `Piper`)**:
  - 100% offline neural TTS for air-gapped or customized voice cloning.
- **Open-Source Sound Effects (SFX)**:
  - CC0-licensed library of dry-erase marker squeaks, chalk strokes, eraser wipes, and paper page turns.
  - Automatic audio ducking and volume normalization using `pydub` and FFmpeg.

### 4.4 Whiteboard Hand-Drawn Animation Mechanics
- **Simulated Hand-Drawing Effect**:
  - Renders an animated drawing hand cursor following the contours or a diagonal progressive wipe (`stroke-dashoffset` in SVG/Remotion or alpha mask gradient in OpenCV).
- **Dual Aspect Ratio Support**:
  - **YouTube (16:9)**: Horizontal expansive whiteboard with connecting doodle paths.
  - **TikTok / Shorts (9:16)**: Vertical stacked doodles with dynamic camera zoom-in/out and downward pan.

### 4.5 Subtitle & Classical Typography Studio
- **Free Google Fonts**:
  - Classical Arabic: **Amiri** (traditional Naskh calligraphy) and **Cairo** (clean modern whiteboard font).
  - English: **Montserrat** or **Caveat** (handwritten style).
- **BiDi Integrity**:
  - Fully formatted Right-to-Left subtitles using `fribidi` and `harfbuzz` to prevent disconnected letters or inverted punctuation.
  - Dynamic karaoke-style word highlighting synchronized to the audio.

---

## 5. User Workflow (Automated / Semi-Automated)

1. **Input**: User supplies text or script file.
2. **Pre-processing**: System detects language, applies Classical Arabic Tashkeel via `mishkal`, and breaks script into visual scenes.
3. **Audio Generation**: `edge-tts` produces narration and word-level timestamps.
4. **Visual Generation**: SDXL/Flux line-art engine or OpenCV sketch pipeline produces doodles for each scene.
5. **Preview Station (HITL - Optional)**: User can review scene sketches, edit narration lines, or swap doodles in an open-source web UI (Streamlit / Gradio).
6. **Rendering**: Headless Remotion or MoviePy/FFmpeg composites sketches, hand-drawing masks, marker SFX, and BiDi subtitles into MP4.
