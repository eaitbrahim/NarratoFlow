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
uv run narratoflow channel check channel.toml          # fails fast naming a missing key
uv run narratoflow library keys channel.toml           # reusable sketch idea keys
uv run narratoflow library copy SKETCH_ID other.toml   # reuse a sketch in another channel
uv run narratoflow check-input script.txt --language ar   # reject corrupt PDF copy-paste

uv run python scripts/fetch_quran.py                   # one-time: Tanzil Qur'an text for verse detection
uv run narratoflow plan script.txt --channel channel.toml   # writes series.json next to the script
uv run narratoflow approve series.json                 # review the list, then confirm (--yes skips the prompt)
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
