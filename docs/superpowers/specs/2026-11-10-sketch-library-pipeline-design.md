# Design: Text-to-Whiteboard Pipeline with a Reusable Sketch Library

**Date:** 2026-11-10
**Status:** Ready for owner approval (builder check done; owner questions answered)
**Supersedes, where they conflict:** `docs/PRD.md` v1.0 and `docs/SRD.md` v1.0 (see §9 for the changes).

Labels used below: **[owner]** means the product owner decided it in conversation. **[design call]** means the designer chose it, and the owner can override it. Unlabeled text is description.

---

## 1. Intent

A creator provides a script, for example a **history explainer**. NarratoFlow breaks it into ideas, draws each idea as a whiteboard sketch, animates a hand drawing it, and narrates over it with a professional voice. The output is a finished video for YouTube and/or TikTok.

**Why this design exists:** history explainers use the same visual vocabulary over and over: scrolls, maps, crowns, castles, ships, armies. Paying an AI service to draw "a scroll" in every video is wasted money. **Every generated sketch is stored and reused, so cost per video falls as the library grows. [owner]**

### What success looks like
- After a channel's first few videos, most scenes are served from the library at no cost.
- All sketches in a channel look like they came from the same hand.
- Nothing gets published that a platform would reject for licensing, missing AI disclosure, or content-policy reasons that an automated check can catch.
- The creator always knows what a video will cost before paying, and afterwards what it did cost.

### Anti-goals (failures even if everything technically works)
- **Inconsistent-looking sketches** within a channel. **[owner]** This is the most important quality bar.
- Paying to generate something the library already has.
- Exporting a video that failed the publishing check.
- Surprise spending.

### Non-goals
- Real-time or live rendering (OBS integration was considered and rejected **[owner]**).
- Recording human narration. Narration is AI voice only, in this design.
- Sharing libraries automatically between channels.
- Guaranteeing legal compliance. The check reduces risk. Once a video passes, the creator still makes the final call to publish (§6.3). A FAIL can't be overridden (§6.1).

---

## 2. Core concepts

### 2.1 Channel [owner]
A **channel** is a named, persistent setup, for example "History Explainers". Each channel has:
- **its own sketch style and its own sketch library [owner];**
- a **style definition**: a reference sketch plus style instructions used in every paid request;
- *[design call]* default voice, language (Classical Arabic or English), output formats (16:9, 9:16), spending limit, and approval-list toggle.

Starting a video needs only a script and a channel name.

**Style changes [design call]:** each style definition has a version. **The image provider and model are part of the style version.** Changing the reference sketch, the instructions or the image model creates a new version. For this reason **image generation has no automatic fallback**: if the image provider fails, the run stops and asks, rather than silently mixing models. Voice and LLM may fall back automatically, and the fallback is recorded in the report. Library entries remember which version they were made in, and only entries from the *current* version are reused. Old entries are kept, not deleted, but never mixed into new videos.

**Cross-channel copy [design call]:** the creator can copy a sketch from one channel to another by hand. Its stored strokes are re-rendered with the target channel's marker (§4.4), which is **free, with no regeneration**. The copy keeps its original provenance plus a "copied from" note. Nothing is shared automatically.

### 2.2 Idea
An **idea** is the thing a scene shows. Ideas are matched **specifically, not loosely [owner]**:
- "a scroll", "an old scroll" and "a rolled parchment scroll" are the **same** idea (wording differs, drawing doesn't);
- "a Roman scroll" and "an Egyptian papyrus" are **different** ideas, so the papyrus gets generated.

*[design call]* The working rule is: **same object + same culture/period = same idea**. Each idea gets a canonical key, e.g. `scroll|roman` or `papyrus|ancient-egypt`.

*[design call, from the reference script §10]* **Abstract ideas are keyed by the visual metaphor that gets drawn, not by the concept.**
- Examples: "the present world vs the hereafter" → `road|two-horizons`; "seen by the eye" → `eye|generic`.
- The planner first picks a concrete, drawable metaphor, then keys it.
- Most ideas in reflective or literary scripts end up `|generic`, which is good for reuse.

**Matching & scale [owner]:** to prevent key drift (`scroll|roman` vs `scroll|ancient-rome`) and prompt bloat as the library grows, matching uses a **two-tier lookup**:
1. *Candidate keys in the prompt:*
   - **Up to 300 keys:** the planner gets the channel's full key catalog.
   - **Above 300 keys:** it gets the **top-5 candidates per scene**, retrieved by embedding similarity.
2. *Exact DB verification:* the chosen key must match an existing library entry exactly. Otherwise it is marked as a new idea.

Retrieval details:
- Every key is stored with a short **English description** (e.g. `road|two-horizons` → "a road running between two horizons"). The **description** is what gets embedded, not the bare key, so Arabic and English scenes both retrieve correctly. **The embedding model must be multilingual.**
- The embedding model follows the provider pattern (§4.1): paid by default, with a local free option. It needs its own key and its own line in the cost estimate.

### 2.2a People are always drawn indirectly [owner]
Prophets, famous historical figures and any real person are **never drawn as a likeness**. Instead they appear **indirectly**, as one of these (silhouettes are excluded by §2.2c):
- a symbol (a crown, a sword, a pen, a ship);
- a name card.

How this is enforced:
- The planner turns every person-idea into one of these forms, keyed by the form, e.g. `crown|abbasid` or `name-card|harun-al-rashid`.
- The likeness check (§6.3) **FAILs any face or recognisable figure**, whoever it is.
- *[design call]* **Prophets get only a name card or a non-figurative symbol, never a silhouette.** In Islamic tradition even a silhouette of a prophet is widely considered unacceptable.
- *[design call]* **God is never depicted in any form.** Only the name may appear, as text or calligraphy.
- *[design call]* **Angels and the unseen** (resurrection, Paradise, Hell, the Bridge) are shown **only through abstract symbols**: light, a gate, scales, an open book, a horizon. They are never shown as figures or literal scenes.

### 2.2c No human figures at all [owner]
**No human figure is ever drawn, not even an anonymous or faceless silhouette.** People in the text become objects or symbols:
- a driver → a steering wheel;
- a team → connected nodes;
- a parent and child → two separated paths.

**Allowed body parts [owner]:**
- the **drawing-hand sprite** that animates every sketch;
- **isolated** hands, eyes and footprints used as symbols, e.g. a hand holding a pen or a balance, `eye|generic`.

**Never allowed:** faces, heads, torsos, stick figures, silhouettes, or any recognisable likeness. The vision check (§6.3) enforces exactly this list.

Because of this, the silhouette option in §2.2a is removed. Prophets and famous people are shown **only as a name card or a non-figurative symbol**.

### 2.2d Narration is the author's exact words [owner]
- The narration is the script **verbatim**. The planner splits it and picks visuals, but **never rewrites, condenses or adds words**.
- Hooks, "Part N of M", recaps and teasers (§2.2b) are **on-screen text only**, never spoken.
- Diacritization adds harakat only, and the skeleton guard enforces that.
- **No dead air at the start [owner]:** short-form retention drops sharply if the first 2–3 s are silent.
  - Narration (or the recitation) **starts at 0 s**, with the on-screen hook text shown at the same time.
  - If a part has to open without speech, the only allowed sound is **marker-stroke SFX**. No rising cue or other music-like sound, because background music is off by default.
  - **Precedence:** if a part opens with a Qur'anic verse, the recitation is the opening audio and **no SFX** plays (§2.2e).

### 2.2e Qur'anic verses [owner: real reciter, Hafs by default]
- Verses are **recited by a real reciter's recording**, never by the AI voice. The riwaya is **Hafs 'an 'Asim** by default and is configurable per channel.
- *[design call]* **Verse detection:**
  - The text is matched against a verified Qur'an corpus (Tanzil), not only by looking for ﴿…﴾. The reference scripts quote verses inside "…" or with no marks at all.
  - Each match is identified as surah:ayah (or a partial ayah range).
  - Uncertain matches are shown in the approval list for confirmation.
- *[design call]* **On screen:** the verified Uthmani text from the corpus, never the pasted text. The verse is **font-rendered exactly** (never converted to strokes or redrawn, since that would distort joins, harakat and Uthmani marks and break the exact-text FAIL rule). It is **revealed word by word, right to left**, timed to the recitation:
- using per-ayah word-segment timing data for the chosen reciter;
- that timing data's licence is recorded like the audio's.

If no timing data is available, the whole ayah is revealed at the start of its recitation.
- *[design call]* **Recitation audio** comes from per-ayah recordings with a recorded licence and source. **Unlicensed recitation is a FAIL.** Recitations are a common source of YouTube Content ID claims, so only reciters on an allow-list with clear reuse terms are used.
- Narration pauses during a verse. There is no marker SFX and no music under verses. *[owner]* Splicing between TTS narration and reciter audio includes acoustic smoothing (LUFS loudness match and 300–500 ms buffer pauses) to eliminate jarring transitions.

### 2.2b Series [owner]
The target is **short vertical videos for YouTube Shorts and TikTok**. A long script is **split into a series of short videos**.
- *[design call]* Each part has a configurable target length, 60–90 s by default. That fits both platforms' short-form limits and the norms for retention.
- The planner splits at idea boundaries, never mid-scene. Each part gets:
  - a 2–3 s hook;
  - a "Part N of M" marker;
  - a brief recap opening from Part 2 onward;
  - a teaser ending, except on the final part.
- *[design call]* One approval list covers the whole series, with an estimated cost per part and a total. The spending limit applies per video, meaning per part. Every part gets its own report, and the series gets a summary report.
- Sketches drawn in Part 1 are reused in later parts automatically through the library.

### 2.3 Sketch (library entry)
A stored drawing for one idea in one channel and style version. It holds the centerline strokes (§4.3), the original image, and its provenance (§6.2). One idea can have several sketches, but **any existing sketch is always reused**. A new one is generated only when the idea has **no** sketch in the channel's current style. **[owner]**

---

## 3. Pipeline overview

```
Script ─► 1 Plan ideas ─► 2 Match library ─► [3 Approval list] ─► 4 Generate missing sketches
                                                  (optional)            │
        ┌─────────────────────────────────────────────────────────────────┘
        ▼
   5 Narrate ─► 6 Animate & render ─► 7 Publishing check ─► 8 Export + Report
                                          (FAIL blocks)
```

1. **Plan ideas.** The script AI splits the text into scenes. For each scene it produces the narration text and one idea (object, culture/period, and a short description for drawing).
1a. **Input quality gate** *[design call, from the reference scripts §10]*.
   - Before anything is paid for, the pasted text is checked for copy-paste corruption. Signs include:
     - stretched letters (`ييييدعى`);
     - broken font encoding (`يومن رررا`);
     - lines in reversed or scrambled order;
     - page numbers mixed into the text;
     - a high ratio of non-words.
   - Corrupt text is **rejected with a report of where**. It is never "repaired" by AI, since narration must be the exact words (§2.2d).
2. **Match library.** For each idea, look in this order: (a) the channel library in the current style, (b) the bundled free icon library, (c) otherwise mark it **new (paid)**.

   *[design call]* Bundled icons have **generic keys** (`arrow|generic`, `lightbulb|generic`, `map-pin|generic`). They only match ideas that the planner marks as culture-agnostic, never `scroll|roman`. An icon is copied into the channel library the first time it is used. Icons go through the same marker re-render and the same outlier check (§4.4), and an icon that looks too geometric next to the channel's style is flagged.
3. **Approval list (optional).** See §5.1.
4. **Generate missing sketches** with the chosen image provider, store them in the library, and run the per-sketch checks (§6.3) at that point, once.
5. **Narrate** each scene with the chosen voice provider and get word timings.
6. **Animate & render.** Free variations are applied (§4.5), strokes are drawn by the hand in narration-paced order, captions and sound effects are added, and the video is rendered with FFmpeg for each output format.
7. **Publishing check.** See §6.3. A **FAIL blocks export; a WARN does not. [owner]**
8. **Export + report.** The MP4(s), the production report (§5.2), and the disclosure text are ready to paste.

Each stage is cached by its inputs, so re-running after editing one scene redoes only that scene (this is already scaffolded in `pipeline/cache.py`).

---

## 4. Sketches

### 4.1 Providers: paid by default, free option [owner]
| Job | Default | Free option |
|---|---|---|
| Script processing (split into ideas, review text) | Paid LLM: the creator specifies the model and API key | Local model via Ollama |
| Sketch generation | Paid image AI: the creator specifies the model and API key | Local diffusion model (needs a GPU) |
| Narration | Paid voice AI: the creator specifies the model and API key | edge-tts (Microsoft, free, online) or Piper (offline) |

**Left to the builder:** which paid providers to support first. They must meet these requirements:
- **Image:** can follow a style reference or style instructions, and has commercial-use terms for its outputs.
- **Voice:** good Classical Arabic (Fusha) with diacritics honored. It either returns word timestamps or its audio is aligned afterwards (CTC aligner, issue #25).
- **LLM:** supports structured JSON output.

Each provider sits behind an adapter that implements the existing Protocols in `src/narratoflow/interfaces.py`.

### 4.2 Idea matching [design call]
- The script AI receives the channel's **existing idea keys** and is told to reuse a key when the idea is the same (§2.2), and to propose a new key only when it truly isn't. This keeps keys from drifting ("scroll" vs "parchment") without loose matching.
- Matching itself is an **exact key lookup**. There is no fuzzy or "close enough" similarity, because the owner wants specific matching.
- In the approval list the creator can override a match, either "use this existing sketch" or "generate new".

### 4.3 Turning any image into strokes
Every sketch, whatever its source (paid AI, local model or bundled icon), goes through the same process: clean line art → centerline skeleton → ordered strokes (issue #15). This is what lets the hand draw it, and it is also the main tool for consistency.
- **Negative prompt & binarization guard [owner]:** generative prompts strictly forbid shading, solid fills, gradients, drop shadows, and 3D textures (`shading, fills, grayscale, textures, double lines, gradients, hatching, 3d`). Where a provider has no negative-prompt parameter, the **provider adapter** puts these as explicit "no …" instructions in the positive prompt.
- **Density check [owner]:** after binarization (Otsu global thresholding; the builder may switch to adaptive Gaussian if the calibration set shows it's better) and morphological thinning, sketches whose skeleton graph exhibits excessive branch points or stroke density (signs of filled silhouettes or crosshatching) are flagged before storage.
  - It runs **before** the paid vision check, so rejects cost nothing.
  - The thresholds are left to the builder, calibrated on a labelled sample set.

### 4.4 Consistency (the top quality bar) [design call]
- **One locked style per channel.** The same style instructions and reference sketch go into every paid request.
- **Every sketch is redrawn by the same "marker".** Strokes are rendered with the channel's line weight, ink colour and hand wobble, so a bundled-icon scroll and an AI castle look alike.
- **Outlier check.** Newly generated sketches that are too detailed, shaded, coloured or textured get a warning. The creator can regenerate them or accept them.

### 4.5 Free variations [owner: fresh-looking variations, always reusing stored versions]
On reuse, each appearance of a sketch is varied at no cost:
- small rotation and scale;
- a different wobble seed;
- slight line-weight change;
- a different stroke order;
- horizontal mirroring, **only for sketches marked mirror-safe [design call]**. Maps, anything with text or symbols, and asymmetric objects whose direction matters are never mirrored. Whether a sketch is mirror-safe is decided when it is generated and stored with it.

Variation parameters are derived from the video and scene, so re-rendering the same video gives the same result, while different videos get different variations.

### 4.6 Storage [design call]
- One library database (SQLite) under the user's data directory, with channel as a column.
- Sketch files are stored content-addressed (by hash) alongside it.
- Narration is **not** in the shared library, since each script's words are unique. It is cached per project.

---

## 5. Creator-facing outputs

### 5.1 Approval list [owner]
- A **toggle** per channel, which can be overridden per run. When it's **on**, the pipeline pauses after planning and shows one line per scene: narration text, idea, **reused / bundled icon / new (paid)**, and an **estimated total cost** (sketches + voice + script processing).
- The creator can approve, edit a scene's text or idea, or swap the match. Nothing paid runs before approval.
- When it's **off**, the pipeline runs straight through.

### 5.2 Spending limit [owner]
- An optional per-video limit, set on the channel and overridable per run.
- If the estimate goes over the limit, the pipeline **stops and shows the approval list even when the toggle is off**.
- If actual spending passes the limit mid-run (for example after retries), the pipeline stops before the next paid call.

**Cost estimates [design call]:** prices come from a creator-editable price table per provider and model, because providers change prices. The report records actual costs from provider responses where the provider gives them, and otherwise from the table, marked as estimated.

### 5.3 Production report [owner asked for it; contents are a design call]
Produced at the end of every run, as both a readable file and a machine-readable file:
- scenes, duration, and output formats;
- per scene: idea, whether it was reused, a bundled icon or newly generated, and the variation applied;
- cost per provider, the total, the estimate vs actual, and the **savings from reuse**;
- models, voices and style version used;
- provenance and license for every asset;
- the publishing-check result with every WARN/FAIL and its reason;
- the AI-disclosure text.

---

## 6. Publishing check

### 6.1 Outcome [owner]
Each check returns **PASS / WARN / FAIL** with a reason. Any FAIL **blocks export**. WARNs go into the report and don't block.

### 6.2 Provenance
Every asset (generated sketch, bundled icon, sound effect, font, music) records its source, license, and for generated assets the provider, model, date and the provider's output-usage terms. Missing provenance is a **FAIL**.

### 6.3 Checks [design call for the list and the severity defaults]
| Check | When | Default severity |
|---|---|---|
| Asset has no recorded source/license, or a license that doesn't allow this use | Render | FAIL |
| **Script source and rights.** Author, book/edition, and the basis for using it: own text, public domain, or permission. | After planning | WARN if not recorded; **FAIL if declared as third-party in-copyright with no permission** |
| Qur'anic text in the video doesn't match the verified source exactly (§10) | Render | FAIL |
| AI-content disclosure not set | Export | FAIL |
| Generated sketch shows a logo, trademark, copyrighted character, or **any face or recognisable human figure** (§2.2a) | Once, at generation (result stored with the sketch) | FAIL |
| Script content vs YouTube/TikTok content rules (e.g. graphic violence in battle scenes, likely misinformation claims) | After planning | WARN, or FAIL for clear violations |
| Technical requirements per platform (length, resolution, aspect ratio, loudness) | Export | FAIL |
| Inconsistent-style sketch (§4.4) | At generation | WARN |

**How generation-time checks run [design call]:**
- The likeness, logo and outlier checks use a vision-capable model. In paid mode this is a paid call that is **included in the estimate and the spending limit**. In free mode a local vision model is used. If none is configured, the check is a WARN saying "not checked", never a silent PASS.
- **Likeness check prompt calibration [owner]:** matches §2.2c. To avoid false-positive rejections on valid abstract symbols, the vision check explicitly permits standalone symbolic organs/limbs (e.g. `eye|generic`, a stylized hand holding a pen or balance, footprints/paths), while strictly rejecting faces, recognizable likenesses, torsos, heads, stick figures, and full-body silhouettes.
- **A sketch that fails is quarantined.** It is stored with status `failed` and **never reused** (an exception to §2.3, which applies only to sketches that passed). The scene isn't regenerated automatically. The creator chooses between regenerating it (a paid call), swapping the idea, or using a text card, so no money is spent without being asked.
- There is **no override of a FAIL [design call based on the owner's "block the export on fail"]**. A FAIL is resolved by fixing the scene, not by accepting it.

**Honest limit:** these checks catch the common problems but cannot guarantee there are no legal issues. Platform rules change, and copyright is decided by people. The platform rules the check uses are stored in a **dated, versioned rules file**, and the report states which version was used. Updating the rules file is a manual maintenance task.

---

## 7. Engineering standards for the SRD (python-best-practices) [owner: apply this skill]

The SRD v1.1 must adopt these. Each one names the failure it prevents.

### 7.1 Environment and tooling
- **uv only.** `requires-python` and `.python-version` must agree. Today they don't (`>=3.11,<3.14` vs `3.12`). Pin both to the same minor version, the newest that every heavy dependency (torch, opencv) ships wheels for. This prevents "works on my machine".
- `uv.lock` is committed (done). CI uses `uv sync --locked` and a **pinned `setup-uv` version**. The audit flagged both as missing in `.github/workflows/ci.yml`.
- Dev tools stay in `[dependency-groups] dev`. Heavy engines stay optional extras (done).
- **Ruff rules:** add `S` (security), `G` (logging format) and `ASYNC` (blocking calls in async) to the existing set, with `"tests/**" = ["S101"]`. In CI use `ruff check --no-cache` and `ruff format --check`, never a rewriting format.
- **mypy strict on `src` and `tests`.** Today only the package is checked.
- **Branch coverage** scoped to `src/narratoflow`. The gate order is `ruff check` → `ruff format --check` → `mypy` → `pytest` under coverage, identical locally and in CI.

### 7.2 Secrets and configuration
- **API keys come from the environment, never from code or the channel config files.** A `Settings(BaseSettings)` class (pydantic-settings) holds them as `SecretStr`, unwrapped only at the call site of the provider adapter. That prevents keys showing up in logs, tracebacks or the report.
- `.env` is git-ignored; `.gitignore` currently lacks it, so it must be added before any key exists. A committed `.env.example` lists every variable.
- **A key has no default.** If a paid provider is selected and its key is missing, NarratoFlow fails at startup with a clear message instead of failing mid-run after money has been spent.
- Channel settings (style, voice, limits) are non-secret config files validated with Pydantic. A cached `get_settings()` accessor is passed in as a parameter, not imported as a global, so tests can override it.

### 7.3 Boundaries and errors
- **Validate every trust boundary with Pydantic, once:** provider API responses, LLM JSON output, `storyboard.json`, channel config, the rules file, and library rows read back.
- **Never swallow exceptions.** Define a small typed error hierarchy (`ProviderError`, `BudgetExceeded`, `PublishCheckFailed`, …). Catch only to recover (retry, fallback) or to add context (`raise … from e`). A provider fallback is always recorded in the report, never silent.
- **No blocking calls inside `async def`.** Provider calls use `httpx`. If the pipeline goes async for parallel generation, every I/O call is awaited and CPU work (rendering, skeletonizing) runs in processes or threads.

### 7.4 Logging
- `logging.getLogger(__name__)` per module, configured once in the CLI entrypoint with the level from settings, using lazy `%s` arguments.
- CLI output to the user goes through `rich`. Package code never uses `print`.
- Never log keys, full prompts containing user scripts at INFO level, or raw provider responses.

### 7.5 Data layer (sketch library)
- **SQLAlchemy 2.0** typed models on SQLite, with **Alembic** migrations. Every schema change is a reviewed migration, and an autogenerated one is only a draft. The library outlives individual versions of the app, so its schema has to be upgradable.
- The database enforces integrity: unique `(channel, style_version, idea_key, sketch_hash)`, foreign keys, NOT NULL.
- Timestamps are timezone-aware UTC.
- Queries live in one repository module. One session per unit of work, using a context manager.

### 7.6 Testing
- **TDD:** write the failing test first.
- **No network and no paid calls in tests.** Fake providers implement the same Protocols (issue #5).
- Tests cover failure paths: missing key, budget exceeded, provider outage → fallback recorded, publishing-check FAIL blocks export, mirror-unsafe sketch never mirrored.
- Library tests use a fresh in-memory SQLite per test.
- The clock is injected, so tests never depend on wall-clock time.

---

## 8. First version (v1) [owner]
- **Output: 9:16 short videos only** (YouTube Shorts + TikTok). 16:9 comes later.
  - *[owner]* **Vertical composition zones & safe zones:**
    - **Top 20%:** series title, part indicator ("Part N of M") and topic badge. Kept below the platforms' top tab bar, roughly the top 8%.
    - **Middle 50%:** the active drawing stage. A virtual camera pans and zooms to follow the hand and keep sketches legible on mobile screens without crowding.
    - **Bottom 30%:** narration captions, placed in the **upper part** of the band (roughly 65–78% of the height). The very bottom is left clear for the username and description.
    - **Right edge:** about 12–15% of the width, across the lower ~60% of the screen, is kept clear of sketches and captions because the like, comment, share and sound buttons sit there.
    - These percentages are approximate. Pin exact 1080×1920 pixel values in Plan 5, verified against the official TikTok and YouTube Shorts safe-zone templates.
- **Long scripts become a series** (§2.2b).
- **Languages: Classical Arabic and English, both in v1 [owner].**
- **Input: pasted text only [owner].** No OCR in v1. For scanned or PDF books, the owner turns screenshots into text with Claude outside NarratoFlow, then pastes the result.
  - *[design call]* An AI transcription can produce **plausible but wrong words**: silently "corrected" spelling, or verses quoted from memory. The input quality gate can't catch those.
  - Mitigations:
    - verses are always replaced by the verified corpus text (§2.2e);
    - the approval list shows the full text for proofreading;
    - the report records that the script came from an AI transcription.
- **In-copyright third-party books are blocked [owner].** The script-source check FAILs them, with no permission path in v1.

## 8a. Left to the builder
- Which paid image, voice and LLM providers to support first (within the §4.1 requirements).
- Exact variation ranges (rotation degrees, wobble amplitude).
- Report file formats (suggested: Markdown + JSON).
- Exact wording of the approval-list and report screens in the CLI.
- How the content-rules review prompt is written and evaluated.

---

## 10. Reference-script walkthrough (dry run by hand)

**Input:** three screenshots of a printed Arabic book page, about 230 words. It opens with Qur'an 31:33 in ﴿…﴾ and continues as reflective literary prose about the "scenes" of the hereafter. The prose style resembles Sayyid Qutb's *مشاهد القيامة في القرآن*. That attribution is not confirmed, and it is exactly what the script-source check (§6.3) exists to record.

**Split:** at a Classical Arabic TTS pace of roughly 110–130 words/min, the excerpt runs about 2 minutes, so it becomes **2 parts**:
- Part 1 is the verse plus paragraphs 1–2;
- Part 2 is paragraphs 3–5.

The excerpt **ends mid-thought**, on "في استطراد عجيب:". The planner must detect a trailing colon or an unfinished argument and either hold that part for the next chunk of the script or warn.

**Ideas (about 14), and what they test:**

| Text | Drawn as | Path |
|---|---|---|
| The verse | Verse card: calligraphic text, no figures | Protected text, no sketch |
| Parent and child cannot help each other | Two paths diverging (no figures, §2.2c) | People rule |
| One simple truth in many images | A prism splitting light into frames | New (paid) |
| A whole world full of scenes | A globe made of picture frames | New (paid) |
| Imagination follows | A thought bubble | Bundled `|generic` |
| Unique pages added to literature | An open book with pages flying | `book|generic`, reused often |
| Living scenes, not dead lines | A wilting line drawing beside a growing tree | New (paid) |
| Distances measured by feelings | A ruler ending in a heart | New (paid) |
| Present today, seen by the eye | An eye | Bundled `eye|generic` |
| The two worlds are near | Two horizons joined by a short bridge | New; **unseen rule**, abstract only |
| The hereafter present, the world a distant past | An hourglass reversed | `hourglass|generic` |
| Humanity crossing the journey | A long road | `road|two-horizons`, reused often |

**Assumptions tested:**
- ✅ **Reuse holds:** eye, book, road, hourglass, scales and horizon will recur across a series like this.
- ⚠️ **Wrong guess corrected:** I assumed history objects (`scroll|roman`). This content is abstract, so ideas become *visual metaphors*, mostly `|generic` (§2.2). The bundled icons and the consistency check (§4.4) matter more than expected.
- ⚠️ **Gap found: script copyright.** The design checked assets but not the script itself. Added a WARN check (§6.3).
- ⚠️ **Gap found: input format.** The script arrived as images of a printed page. Ingestion needs Arabic OCR (see the questions to the owner).
- ⚠️ **Gap found: Qur'anic verses in audio and visuals** (see the questions to the owner).
- *[design call]* **No marker SFX or music under Qur'anic verses.** The channel setting for background music defaults to **off**.
- ✅ **Diacritics:** the source is partly vocalised (شيئاً، وأياً). The skeleton guard (§2.1 in REVIEW) protects it. The verse comes from the verified source, never from OCR plus diacritization.

### Reference script 1 follow-ups
- The verse (Luqman 31:33) is recited by a Hafs reciter. The narration pauses during it, and the verse card shows the corpus text.

### Reference script 2: Arabic, on Yahya (John the Baptist), pasted from a PDF
- ❌ **The text is corrupted by the PDF copy.** Examples: `ييييدعى`, `يومن رررا` for «يوحنا»?, `المنرران` for «الحنان», a broken Maryam 19:12–13, reversed line order, page numbers inline. It can't be narrated as-is, and AI repair would violate exact words. → Input quality gate (step 1a). The creator must paste from a clean source such as a .docx or an online library.
- ✅ **People rules hold.** Yahya and Zakariyya are prophets, so they get name cards. Baptism in the Jordan → a river with water drops. Sabians/Mandaeans → a name card. No human figures.
- ✅ **Verses are quoted with "…" and no ﴿﴾** (Maryam 19:7, 19:12–13), which confirms corpus-based detection.
- ⚠️ **Comparative-religion claims.** The text discusses the Gospels and the Sabians, so the content-rules review (§6.3) must handle religious discussion as a WARN at most, not a FAIL.

### Reference script 3: English, a business/AI book chapter
- **Length:** about 1,500 words, roughly 10 minutes at 150 words/min → **about 8–10 parts** of 60–90 s. A clear case for series splitting.
- ✅ **No-human-figures works.** Drivers → steering wheel; mechanics in the F1 pit → wrench and tyre; procurement teams → two documents with arrows; CFO and salesperson → shared chart.
- ✅ **Brands are named in narration but never drawn.** Shein, Stripe, Google Maps, Netflix and F1 become generic objects (hanger, card, map pin, play button, race flag). The logo check backs this up.
- ✅ **Reuse is high:** network nodes, shipping container, map, gears, puzzle pieces, documents.
- ❌ **Copyright risk is high.** It reads like a recently published, in-copyright book. Narrating its **exact words** across 8–10 videos is reproduction of a substantial part of a copyrighted work, the riskiest combination there is. *[design call]* The script-source check (§6.3) becomes a **FAIL** when the declared source is a third-party, in-copyright work with no recorded permission. That is the gate doing its job, so this example is **blocked**, as decided by the owner.

## 9. Changes to existing documents and backlog

**PRD v1.1** (issue #1 absorbs these):
- Replace "100% FOSS / zero-cost operation" with: **open-source software, paid AI providers by default with a free/local mode, and a reusable sketch library that drives cost toward zero over time.**
- Add channels, the approval list, the spending limit, the production report, and the publishing check as product features.
- Add the anti-goal "inconsistent sketches" and the success measures in §1.

**SRD v1.1:** add §7 in full, and add provider adapters, the library data model and the publishing-check architecture.

**New backlog issues** (to be created after approval):
- channels & style versions
- sketch library DB + Alembic
- idea keys & matching
- paid image provider adapter
- paid voice provider adapter
- paid LLM adapter
- settings & secrets
- approval list
- spending limit & cost table
- free variations incl. mirror-safety
- consistency/outlier check
- production report
- publishing-check framework + rules file
- provenance tracking
- CI hardening per §7.1

**Existing issues to adjust:**
- #12 edge-tts becomes the free fallback rather than the primary voice.
- #14 the bundled icons become the second lookup after the channel library.
- #27 the scene planner must receive the existing idea keys.
- #29 local diffusion is now the free option.
