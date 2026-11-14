# NarratoFlow

Open-source pipeline that turns Classical Arabic (الفصحى) or English text into whiteboard animation videos for YouTube (16:9) and TikTok/Shorts (9:16).

> Status: early scaffold (M0). See [`docs/`](docs/) for the PRD, SRD and the technical review.

Stroke data format: [`docs/strokes-schema.md`](docs/strokes-schema.md).

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
uv run narratoflow library icons light   # bundled icon keys, e.g. for sketch resolve --swap on a quarantined scene
uv run narratoflow library copy SKETCH_ID other.toml   # reuse a sketch in another channel
uv run narratoflow check-input script.txt --language ar   # reject corrupt PDF copy-paste
uv run narratoflow check-dialect script.txt            # flag dialect sentences (needs the arabic extra; see Dialect check)

uv run python scripts/fetch_quran.py                   # one-time: Tanzil Qur'an text for verse detection
uv run narratoflow plan script.txt --channel channel.toml   # writes series.json next to the script
uv run narratoflow approve series.json                 # review the list, then confirm (--yes skips the prompt)
uv run narratoflow sketch generate series.json --channel channel.toml   # draw or reuse a sketch per scene; keeps progress on errors
uv run narratoflow sketch resolve series.json s003 --regenerate --channel channel.toml   # or --swap KEY --description TEXT, or --text-card
uv run narratoflow audio build series.json --channel channel.toml   # narrate, mix and master each part; --part N (repeatable), --out DIR
uv run narratoflow render series.json --channel channel.toml   # draw and encode each part to an MP4; --part N, --audio DIR, --out DIR, --workers N, --force
uv run narratoflow review series.json --channel channel.toml   # optional, paid, asks first: LLM check of each part's text against the content rules; --part N, --yes
uv run narratoflow publish series.json --channel channel.toml   # free and offline: check each part, export the ones that pass; --part N, --audio DIR, --render DIR, --export DIR
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
The marker SFX duck 8 dB under narration (10 ms frames, 30 ms attack, 250 ms release).
Only the spoken form of the scenes' narration text is ever voiced: numbers spelled out and the
channel's pronunciation entries applied (see Pronunciation).

`render` turns each part's FLAC and timeline into `part-NN.mp4` (1080x1920, 30 fps, H.264 High
yuv420p, AAC 48 kHz) plus `part-NN.render.json` (frames, scenes, fonts and hand licences, warnings),
next to the audio unless `--out` is given. Render twice, or with another `--workers`, and the decoded
frames are identical. A re-run skips each part whose render inputs are unchanged (it prints
`unchanged, kept`): the series text, the part's timeline and FLAC bytes, its sketches' strokes, the
channel's ink and line weight, the canvas, the fonts and hand, and the ffmpeg version, with an intact
MP4 and captions file on disk. Pass `--force` to re-render anyway.
It needs an ffmpeg built with libass; on macOS the Homebrew `ffmpeg` has none,
so run `brew install ffmpeg-full` and set `NARRATOFLOW_FFMPEG=$(brew --prefix ffmpeg-full)/bin/ffmpeg`.
`render` refuses a stale timeline (swapped sketch, text-card swap, edited verse): after
`sketch resolve` run `audio build` again before `render`. A text card says what `--card-text`
gave it (`sketch resolve ... --text-card --card-text TEXT`); run the same command again to change
the text of a scene that is already a text card.

### Uploading your own image

Instead of a drawn sketch, a scene can use your own picture, turned into line art:

```sh
uv run narratoflow sketch upload series.json s003 photo.jpg --channel channel.toml \
  --license own-work --source "my photo"   # CC-BY-4.0 also needs --attribution "credit line"
```

It is free: no vision call and no spend. The image is traced into black lines on white (XDoG),
must pass the same density check as a drawn sketch, and is refused with the reason otherwise
(undecodable, under 32 px or over 8192 px on a side, empty, too dense). The line art is stored as
an unchecked `uploaded` sketch, which `sketch generate` keeps whatever the channel style, never
redraws and never reuses for another scene. `--license` must be one the platform rules allow
for assets (`own-work` means you made the image); it is checked again at publish. A scene with no
usable upload is listed as quarantined; `sketch resolve --regenerate` or `--swap` gives the scene
back to the styled generator.

### Publish

`publish` checks each rendered part against the platform rules (technical, licences and provenance,
script source, exact verse text, AI disclosure, and the content review if you ran one) and writes
`part-NN.publish.json` and `.md` plus `series.publish.json` and `.md` next to the render outputs.
A part with no FAIL is copied to `export/` (default: next to `series.json`) as `part-NN.mp4`,
`part-NN.disclosure.txt` and `part-NN.publish.md`, with `series.publish.md`. A FAIL blocks that
part's export and cannot be overridden: there is no flag for it. Fix the cause and run `publish`
again. `publish` exits 1 when any selected part is blocked, never spends money and never uses the
network; it does not change `series.json`.

`review` is optional and paid: it asks the channel's LLM to flag clear violations in each part's
text, shows the estimated cost and asks before spending (`--yes` skips the question), and counts the
spend against the part's limit. Without it `publish` only notes that no content review was run;
a part edited after its review gets a "stale" warning. Run `review` again after editing.
The review sees the `[spoken]` line, which is exactly what the voice says: numbers, signs,
currency and pronunciation entries already applied. After upgrading, parts with numbers or
pronunciation entries show the stale-review WARN once, until `narratoflow review` is run again.

The platform rules live in `src/narratoflow/data/platform_rules.toml` (copy it to
`$NARRATOFLOW_DATA_DIR/platform_rules.toml` to override). Updating it is a manual task: every value
not fixed by the spec has a `verify` note with a source; re-check each one, then bump `version` and
`checked`. A rules file checked more than 180 days ago gives a warning.

### Voices

The channel's `[voice]` table picks the narrator: `provider` (`elevenlabs`, `edge` or `piper`),
`voice_id`, `model` and, for `piper`, `piper_model` (path to the `.onnx` file) with the model's
licence: `piper_model_license` (for example `MIT`, `CC-BY-4.0`), `piper_model_source` (where the model
came from; required with a licence) and `piper_model_attribution` (required for `CC-BY-4.0`). The
licence goes into the publish provenance check: an undeclared licence warns, an attribution-only
licence without an attribution fails, and a licence the platform rules do not allow
(`CC-BY-NC-4.0`) fails. Example:

```toml
piper_model = "voices/ar_JO-kareem-medium.onnx"
piper_model_license = "CC-BY-4.0"
piper_model_source = "https://huggingface.co/rhasspy/piper-voices"
piper_model_attribution = "Voice: Kareem (Piper voices), CC-BY-4.0"
```

`fallback` (default
`["edge"]`) re-voices a whole part with the next voice when the first one is down;
the timeline and the command output record the fallback, so there is one voice per part and it is
never silent. `edge_voice` names the edge voice the fallback uses (an edge short name such as
`ar-SA-HamedNeural`; unset, the channel language's default voice). The `tts-edge` and `tts-local`
extras provide `edge-tts` and `piper-tts`; the default edge fallback works only with `tts-edge`
installed (`uv sync --extra tts-edge`), otherwise an outage ends the build with "no voice could
narrate part N". A rejected ElevenLabs key (HTTP 401/403) or an unknown voice (HTTP 404) is a
configuration error: the build stops (finished parts are kept) instead of falling back.

Temporary failures retry inside one voice before any fallback: edge-tts errors and timeouts, and
ElevenLabs connection failures, HTTP 429 and HTTP 5xx, are tried up to 3 times in all, waiting 1 s
and then 4 s. A request that may already have been billed (an ElevenLabs read timeout or an unusable
200 response) is never retried. Piper is local and is not retried. `edge-tts` drives an unofficial,
undocumented Microsoft endpoint that may break or be rate-limited without notice; the fallback chain
(for example ElevenLabs, then Piper) is the mitigation, so keep a second voice configured. Worst
case before a fallback, per part: a hung edge request can take about 9 minutes (3 attempts of up
to 180 s plus 5 s of waits); an unreachable ElevenLabs about 6 minutes (3 attempts of up to 120 s
plus 5 s). Every part restarts at the primary voice, so with edge hanging a 10-part series waits
about 90 minutes before its fallbacks.

### Pronunciation

A channel's `[pronunciation]` table tells the voice how to say words it gets wrong; the caption
keeps the word as written:

```toml
[pronunciation]
"NASA" = "nasa"
"سقراط" = "سُقْراط"
```

Keys are single words. A key with no uppercase letter and no punctuation inside it (`"nasa"`, an
Arabic name) matches ignoring case, tashkeel and edge punctuation, so an Arabic name matches with
or without its vowel marks. A key with an uppercase letter or punctuation inside it (`"NASA"`,
`"WHO"`, `"U.S."`) matches only a word spelled exactly like it (case counts; tashkeel and
punctuation around the word are ignored), so `"WHO"` leaves "who" alone and `"U.S." = "you ess"`
leaves "us" alone. A key's own final period is dropped when a lowercase word follows ("the U.S.
economy" is voiced "the you ess economy") and kept before a capital or at the end, where it may
end the sentence. If both kinds collide, the exact key wins; two keys that are the same after
this reduction are rejected. Numbers are spoken as words while captions keep the digits:

- plain integers up to 999,999,999,999 (thousands separators allowed) are spelled as cardinals;
- in English, a bare four-digit number from 1100 to 1999 is read as a year ("1905" is "nineteen oh
  five"), and ordinals such as "22nd" are spelled as ordinals ("twenty-second");
- decimals read the fraction digit by digit after "point" in English, or after the Arabic word for
  "comma";
- a `-` or `+` touching the number is read as "minus" or "plus" ("-5" is "minus five"; Arabic
  "-٥" is "سالب خمسة"); a `%` or `٪` after it is "percent"
  ("50%" is "fifty percent"; Arabic "٥٠٪" is "خمسون بالمائة");
- `$`, `€` and `£` before or after a number name the unit after it ("$5" is "five dollars",
  "$1.50" is "one dollar and fifty cents"; Arabic "$5" is "خمسة دولارات"). These
  amounts never use the year reading ("$1453" is "one thousand four hundred and fifty-three
  dollars"). With a spaced sign ("- 5"), two currencies, two signs or a currency and a percent, the
  symbols are not read; the number is still spelled out;
- anything else (larger numbers, mixed digit scripts, numbers joined to letters such as `1453م`,
  `٨٥٧هـ` or `1990s`) is left as digits for the voice to read.

Changing an entry re-voices the scenes it touches. After upgrading to this version, scenes that
contain numbers are voiced again once, and a paid voice bills them again; scenes without numbers
are reused. The spend can then differ from an estimate approved before the upgrade.
Scenes containing a sign, currency or percent next to a number are likewise voiced again once after
upgrading, and a paid voice bills them once more.

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

## Text-to-sketch (diffusion)

Optional: a channel can draw new sketches with a local diffusion model instead of OpenAI. Without
a GPU and without the `diffusion` extra nothing changes: bundled icons, library reuse, uploads,
text cards and the OpenAI provider all run as before, and no run imports torch unless the channel
selects `lineart-diffusion` and a scene needs a new sketch. Install with
`uv sync --extra diffusion`.

```toml
[style]
reference_sketch = "ref.png"
instructions = "Black ink whiteboard sketch, minimal, no shading."
image_provider = "lineart-diffusion"
image_model = "black-forest-labs/FLUX.1-schnell"

[style.diffusion]
model_license = "Apache-2.0"
model_source = "https://huggingface.co/black-forest-labs/FLUX.1-schnell"
model_revision = "741f7c3ce8b383c54771c7003378a50191e9efe9"
seed = 0              # the channel's pin; each drawing also mixes in video, scene and attempt
steps = 4             # FLUX.1-schnell is distilled: 4 steps, guidance 0.0
guidance_scale = 0.0
device = "auto"       # cuda, then mps, then cpu; or "cpu" / "cuda" / "mps"
# optional LoRA; a licence and a source are then mandatory
# lora = "owner/name"
# lora_weight_name = "file.safetensors"
# lora_revision = "..."
# lora_license = "Apache-2.0"
# lora_source = "https://huggingface.co/owner/name"
# lora_scale = 1.0
```

Licences. FLUX.1-schnell is Apache-2.0, but its Hugging Face repository is gated: accept the terms
on the model page and log in to Hugging Face before the first download. The licence ids
you declare are copied into every sketch's provenance and checked at publish against
`[licences.generated]` in the platform rules: `Apache-2.0` and `MIT` pass, `OpenRAIL++-M` (SDXL)
warns because it carries use-based restrictions, and anything else, such as FLUX.1-dev's
non-commercial licence, fails. With a LoRA the recorded licence is `<model> AND <lora>` and each
part is checked. No LoRA ships by default: no line-art LoRA with a commercial-safe licence for
schnell was found (follow-up #161). Changing the seed, model or LoRA gives the channel a new style
version, so old sketches are not reused for the new look.

Determinism. The seed reaches the generator on a CPU generator, so the same channel, video, scene
and attempt draw the same image, and a regenerate draws a new one. This is proved offline with an
injected pipeline; whether the real model is bit-identical on a GPU is not measured, see #160.

VRAM. All figures come from model cards and the diffusers memory guide and are not measured; see
#160.

| Model | Weights | VRAM | Status |
|---|---|---|---|
| FLUX.1-schnell, fully on GPU | about 12B parameters; bf16 weights plus T5-XXL over 30 GB on disk | about 24 GB or more | not measured; see #160 |
| FLUX.1-schnell with `enable_model_cpu_offload()` | same | lower peak, slower | not measured; see #160 |
| SDXL fp16 (OpenRAIL++-M, warns at publish) | about 7 GB | about 8 GB typical, 6 GB tight | not measured; see #160 |
| CPU only (`device = "cpu"`) | same as above | none | works, impractically slow |

`local-diffusion` (image to image, `image_provider = "local-diffusion"`) is a separate provider
and is unchanged; use `lineart-diffusion` for line art drawn from the text alone.

## Re-running after an error

- `plan` refuses an existing series file; pass `--replace` to re-plan (this discards its approval, attached sketches and recorded spend).
- `approve` is idempotent.
- `sketch generate` keeps attached sketches and pays only for the missing ones.
- `audio build` reuses each cached scene. After upgrading to the version with spoken numbers,
  scenes containing numbers, signs, currency or percent are voiced again once and a paid voice bills them again, so the spend
  can differ from an estimate approved before the upgrade. Changing a pronunciation entry likewise
  re-voices the scenes it touches.
- `render` re-renders the selected parts deterministically at no cost, skipping the ones whose inputs are unchanged unless `--force` (after an error it prints the finished parts; pass `--part` for the others).
- `review` shows an estimate and asks before spending, unless `--yes` is given.
- `publish` is free.

## Dialect check

`narratoflow check-dialect` flags Arabic sentences that read as a dialect rather than Modern Standard Arabic, using CAMeL Tools. It is free, separate from `plan`, and it flags, never rejects: it exits 0 even when sentences are flagged.

Install once (NarratoFlow never downloads the 274 MB model data itself). Add `--extra arabic` to the extras you already sync (for example `uv sync --extra arabic --extra tts-edge`); a plain `uv sync --extra arabic` would drop your other extras.

```
uv sync --extra arabic --extra tts-edge   # your usual extras plus arabic
uv run camel_data -i dialectid-model26
```

```
uv run narratoflow check-dialect script.txt [--threshold 0.5] [--report PATH]
```

A sentence is flagged when the model's top label is not `MSA` and its probability is at least `--threshold` (default 0.5). Lines with no Arabic letters are never checked, and Qur'anic verses in ﴿…﴾ are skipped (they are classical, not dialect). Labels are CAMeL city or variety codes, for example `CAI` Cairo, `RAB` Rabat, `BEI` Beirut, `DOH` Doha, `TUN` Tunis and `MSA` Modern Standard Arabic. Each flagged sentence prints as `line N: LABEL 0.87 «excerpt»`, followed by a summary. The full report (every Arabic sentence with its label and confidence) is written to `FILE.dialect.json`, or to `--report PATH`. Without the extra or its data, the command exits 1 with one line naming both install steps.

## End card

The final part of a series closes with a 3 second on-screen end card (never spoken), set by the
channel's `end_card` key. Left out, it is "Follow for more" for an English channel and
"تابعونا للمزيد" for an Arabic one;
`end_card = ""` turns it off. A single-part series gets it too. A final part too short to show it after the hook
shows none. It applies to newly planned series: a series already planned keeps the text it was
planned with. The end card shows in the `review` text, so a part gets a fresh review only when
it has one.

## Where settings live

| Kind | Lives in | Examples |
|---|---|---|
| Secrets, machine paths, log level | environment or `.env` (`Settings`, `NARRATOFLOW_*`) | `*_API_KEY`, `DATA_DIR`, `FFMPEG`, `LOG_LEVEL`, `OLLAMA_URL` |
| Channel choices | channel TOML (`ChannelConfig`; defaults in code) | style, voice, llm, embed, vision, `spending_limit_usd`, `approval_list`, part lengths, `end_card`, `riwaya`, `diacritizer` |
| Per-run choices | CLI options | `--part`, `--out`, `--workers`, `--yes`, `--replace`, `--threshold` (dialect check) |
| Prices / platform rules | `data/prices.toml` / `data/platform_rules.toml`, each overridable in `NARRATOFLOW_DATA_DIR` | |
| Output contract | code constants pinned by Plans 4-6 | 1080x1920, 30 fps, -14 LUFS |

## Layout

```
docs/                PRD, SRD, REVIEW (suggestions), adr/
src/narratoflow/
  models.py          storyboard.json IR (Project → Scene → Narration / Visual / Layout)
  interfaces.py      Protocols: Diacritizer, TTSEngine, Aligner, LLM, VisualProvider, Renderer
  pipeline/          content-addressed stage cache
  text/              ingest, language/dialect, protected spans, tashkeel (+ skeleton guard), scene planner
  llm/               Ollama client (structured JSON output)
  tts/               edge-tts, Piper
  align/             CTC forced alignment
  visuals/           icon library, line-art extraction, stroke ordering, diffusion
  audio/             SFX, ducking, loudness
  publish/           platform checks, content review, export and publish reports
  render/            layouts + safe zones, camera, stroke reveal, FFmpeg encoder
  captions/          ASS subtitles (libass shapes Arabic natively)
  ui/                Gradio review station
assets/              fonts, sfx, icons, hand sprites; every pack listed in assets/LICENSES.md
tests/
```

Heavy engines are optional extras (`arabic`, `tts-edge`, `tts-local`, `align`, `diffusion`, `ui`), so the core installs and tests without a GPU.
