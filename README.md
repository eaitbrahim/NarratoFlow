# NarratoFlow

Open-source pipeline that turns Classical Arabic (الفصحى) or English text into whiteboard animation videos for YouTube (16:9) and TikTok/Shorts (9:16).

> Status: early scaffold (M0). See [`docs/`](docs/) for the PRD, SRD and the technical review.

## Quick start

```bash
uv sync --group dev              # core + dev tools
uv sync --extra arabic --extra tts-edge   # add engines as needed
uv run narratoflow --help
# Prices: a packaged default table is used; copy it to $NARRATOFLOW_DATA_DIR/prices.toml to override.
uv run pytest

cp .env.example .env             # then fill in keys for any paid providers
                                 # (NARRATOFLOW_EMBED_API_KEY: Voyage embeddings default; or set [embed] provider = "ollama")
uv run narratoflow channel check channel.toml          # fails fast naming a missing key
uv run narratoflow library keys channel.toml           # reusable sketch idea keys
uv run narratoflow library copy SKETCH_ID other.toml   # reuse a sketch in another channel
uv run narratoflow check-input script.txt --language ar   # reject corrupt PDF copy-paste

uv run python scripts/fetch_quran.py                   # one-time: Tanzil Qur'an text for verse detection
uv run narratoflow plan script.txt --channel channel.toml   # writes series.json next to the script
uv run narratoflow approve series.json                 # review the list, then confirm (--yes skips the prompt)
uv run narratoflow sketch generate series.json --channel channel.toml   # draw or reuse a sketch per scene; keeps progress on errors
uv run narratoflow sketch resolve series.json s003 --regenerate --channel channel.toml   # or --swap KEY --description TEXT, or --text-card
uv run narratoflow audio build series.json --channel channel.toml   # narrate, mix and master each part; --part N (repeatable), --out DIR
uv run narratoflow render series.json --channel channel.toml   # draw and encode each part to an MP4; --part N, --audio DIR, --out DIR, --workers N
```

### Script front matter

`narratoflow plan` reads a script that starts with a `+++` block declaring its source.
Scripts marked `third_party_in_copyright` are refused before any paid call, and
`permission` needs a `permission_note`:

```
+++
author = "Your Name"
work = "Title of the work"        # also the series title
rights = "own"                    # own | public_domain | permission
permission_note = ""
+++
The narration text follows, exactly as it should be spoken.
```

`plan` checks the API keys and the estimated planning cost against the channel's
`spending_limit_usd` before the first LLM call. It prints the approval list (each scene's
idea, whether its sketch is reused, bundled or new and paid, detected verses, flags and
per-part cost estimates) when `approval_list = true`, when a part's estimate is over the
limit, or when something needs a human check. Otherwise it approves the series and
prints a one-line summary.

`sketch generate` checks every new sketch with a vision model (no people, faces, logos or
text; style matches the reference sketch). Set it in the channel TOML with
`[vision] provider = "anthropic"` and `model = "..."`. Without a `[vision]` table the channel's
`[llm]` is used only when its provider can see images (`anthropic`). Otherwise, or with
`[vision] provider = "none"`, vision is off. New sketches are then stored unchecked
(`vision: not checked`), the command prints a warning, and no vision cost is estimated or charged.

`audio build` writes `part-NN.flac` (48 kHz mono, mastered to -14 LUFS) and `part-NN.timeline.json`
(scene, word and verse timings, SFX hits, the voice and reciter used, licences, spend and warnings)
per part, into `audio/` next to the series file unless `--out` is given. It refuses an unapproved
series, a changed narration text or a missing reciter before any spend, keeps finished parts when a
later one fails, and counts its spend against the same per-part limit as `sketch generate`.
Nothing but the scenes' narration text is ever spoken.

`render` turns each part's FLAC and timeline into `part-NN.mp4` (1080x1920, 30 fps, H.264 High
yuv420p, AAC 48 kHz) plus `part-NN.render.json` (frames, scenes, fonts and hand licences, warnings),
next to the audio unless `--out` is given. Render twice, or with another `--workers`, and the decoded
frames are identical. It needs an ffmpeg built with libass; on macOS the Homebrew `ffmpeg` has none,
so run `brew install ffmpeg-full` and set `NARRATOFLOW_FFMPEG=$(brew --prefix ffmpeg-full)/bin/ffmpeg`.
`render` refuses a stale timeline (swapped sketch, text-card swap, edited verse): after
`sketch resolve` run `audio build` again before `render`. A text card says what `--card-text`
gave it (`sketch resolve ... --text-card --card-text TEXT`).

### Voices

The channel's `[voice]` table picks the narrator: `provider` (`elevenlabs`, `edge` or `piper`),
`voice_id`, `model` and, for `piper`, `piper_model` (path to the `.onnx` file). `fallback` (default
`["edge"]`) re-voices a whole part with the next voice when the first one is down;
the timeline and the command output record the fallback, so there is one voice per part and it is
never silent. `edge_voice` names the edge voice the fallback uses (an edge short name such as
`ar-SA-HamedNeural`; unset, the channel language's default voice). The `tts-edge` and `tts-local`
extras provide `edge-tts` and `piper-tts`; the default edge fallback works only with `tts-edge`
installed (`uv sync --extra tts-edge`), otherwise an outage ends the build with "no voice could
narrate part N". A rejected ElevenLabs key (HTTP 401/403) or an unknown voice (HTTP 404) is a
configuration error: the build stops (finished parts are kept) instead of falling back.

### Alignment

Word timings come from the voice when it reports them. The optional `align` extra
(`ctc-forced-aligner`) refines or supplies them: it downloads a model on first use and, being
source-only on PyPI, needs a C++ compiler to install. Without it the timeline falls back to
estimated word timings and each part warns once.

### Qur'an recitation

No reciter ships with NarratoFlow: no source's terms allowed use in published videos (quran.com and
quranicaudio.com are non-commercial, everyayah.com states no terms, alquran.cloud is
non-commercial). `audio build` therefore refuses the whole run, before any spend, if any selected
part contains a verse, and verses are never voiced by text to speech. Build the verse-free parts
with `--part N`. To enable verses the owner must add a reciter whose licence permits this use to
`assets/reciters/index.toml` (see the entry format there, and record the licence in
`assets/LICENSES.md`). A non-editable install (e.g. `pip install` of a wheel) reads the packaged
copy of that index inside the installed `narratoflow` package, not the repository file.

System requirement: `ffmpeg` on `PATH`.

## Layout

```
docs/                PRD, SRD, REVIEW (suggestions), adr/
configs/             default.toml: engine selection, voices, render settings
src/narratoflow/
  models.py          storyboard.json IR (Project → Scene → Narration / Visual / Layout)
  interfaces.py      Protocols: Diacritizer, TTSEngine, Aligner, LLM, VisualProvider, Renderer
  pipeline/          orchestrator + content-addressed stage cache
  text/              ingest, language/dialect, protected spans, tashkeel (+ skeleton guard), scene planner
  llm/               Ollama client (structured JSON output)
  tts/               edge-tts, Piper
  align/             CTC forced alignment
  visuals/           icon library, line-art extraction, stroke ordering, diffusion
  audio/             SFX, ducking, loudness
  render/            layouts + safe zones, camera, stroke reveal, FFmpeg encoder
  captions/          ASS subtitles (libass shapes Arabic natively)
  ui/                Gradio review station
assets/              fonts, sfx, icons, hand sprites; every pack listed in assets/LICENSES.md
tests/
```

Heavy engines are optional extras (`arabic`, `tts-edge`, `tts-local`, `align`, `diffusion`, `ui`), so the core installs and tests without a GPU.
