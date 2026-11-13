# Plan 10 - Close-out and Word-Timing Ground Truth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship four PRs:
- **A** (#133 step 1, and #25's measurement if the data exists): offline tools that measure word timing against the waveform. Production timings do not change.
- **B** (#5, #20): one shared fakes package, a CLI smoke test from text to a verified MP4, and a `render` re-run that skips unchanged parts.
- **C** (#15, #16, #17, #27, #24): a close-out pack of goldens, a benchmark, the end card and the Piper licence record.
- **D** (#2): ADRs 0002 to 0004, describing the system as built.

**Architecture:**
- **PR A.** Two new modules sit in `audio/`:
  - `onsets.py` is an RMS-envelope detector of voiced regions.
  - `mp3info.py` reads the MP3 frame header and the Xing/Info, LAME and VBRI tags.

  No production module imports either one in this plan. `scripts/spike_word_timing.py` gains `--save DIR`, which runs once with network, and `--from DIR`, which is offline. `--from` compares the edge and CTC first-word, post-pause and last-word boundaries with energy onsets, as signed values, and scores an optional hand-made `labels.json`.
- **PR B.**
  - The fakes move to `tests/fakes/`, a top-level test package importable as `fakes`, the same way `fixtures` is.
  - A `CliRunner` test drives plan, approve, sketch generate, audio build, render and publish on `examples/sample_ar.md`, with fakes patched in at the factory seams.
  - `render` hashes each part's render inputs into `part-NN.render.json`. It skips the part when the hash matches and the MP4 still verifies.
- **PR C.** Each item stays local to its module: a strokes golden plus a schema doc, PNG reveal goldens, a benchmark script plus a results doc, `Part.end_card_text`, and Piper licence fields from the channel through `VoiceUsed` into provenance.
- **PR D.** Docs only.

**Tech Stack:** Python 3.12, numpy, pydantic, typer, pytest, ffmpeg (libmp3lame, libass). No new dependencies.

**Spec:**
- `.superpowers/orchestrate/2026-11-12-plan-10/assessment.md`: per-issue evidence of what is built and what is missing;
- the texts of issues #133, #25, #5, #20, #15, #16, #17, #27, #24 and #2;
- `docs/spikes/2026-11-12-arabic-word-timing.md`, the measured disagreement this plan's PR A must explain;
- `docs/adr/0001-record-architecture-decisions.md`, the ADR format.

Out of scope: #30, #28, #8, #29, #1, #22 and #26.

## Decisions

- D1: The onset and MP3 tools live in `src/narratoflow/audio/` (`onsets.py`, `mp3info.py`), not inside the spike script. Why: #133 steps 2 and 3 will reuse them in production, and `src/` is under the coverage gate while `scripts/` is not. Cost if wrong: two modules that nothing in production imports until a later plan.
- D2: The onset rule:
  - Take RMS over 5 ms windows. A window is voiced when it is above -45 dBFS, which is `mixer.ONSET_THRESHOLD_DBFS`.
  - Voiced runs shorter than 20 ms are dropped.
  - Gaps shorter than `min_gap_ms` are merged. The default is 60 ms; the spike passes 150 ms to find pauses.
  - Resolution is ±5 ms, well below the 46 ms and 115 ms effects being measured.

  Why: it is the same threshold production already trusts for the first onset. Cost if wrong: quiet fricative onsets are detected late, which biases "edge earlier". The labels in D5 cross-check this.
- D3: The `--save DIR` layout is one set of files per sentence `NN` (01 to 11):
  - `NN.mp3`: the raw edge bytes;
  - `NN.edge.json`: edge spans as `[[text, start_ms, end_ms], ...]`;
  - `NN.words.json`: the caption words from `tokens(sentence)`;
  - `NN.ctc.json`: CTC spans, written only when the `align` extra is installed.

  A `manifest.json` holds `{"version": 1, "voice", "edge_tts_version", "captured_at", "sentences": [...]}`. `--from DIR` uses `NN.ctc.json` when it is present. Otherwise it runs CTC live when the extra is installed, and otherwise it omits the CTC columns. Why: one networked run, and every analysis after it is offline. Cost if wrong: a re-capture, which is free.
- D4: The sign convention is estimate minus waveform (or minus label). Positive means the estimate is later. This matches the existing "native minus reference" columns. Cost if wrong: readers misread the direction. Every table header repeats the rule.
- D5: `labels.json` is `{"version": 1, "labels": [{"sentence": 1, "word": 3, "onset_ms": 812}, ...]}`. `word` is a 0-based index into `NN.words.json`, and `onset_ms` is on the decoded-audio clock. Why: Audacity label tracks export start times, and the word index avoids matching Arabic text. Cost if wrong: re-labelling.
- D6: #25's "median word boundary error" is defined as two numbers:
  - CTC minus energy at the boundaries energy can see: the first word's start, post-pause word starts and the last word's end;
  - CTC minus labels for mid-sentence words, when `labels.json` exists.

  PR A closes #25 only if the results section holds real numbers from a capture. Otherwise it writes `Refs #25`. PR A never closes #133, whose steps 2 to 4 remain.
- D7: The priming numbers were measured on ffmpeg 9.0.2 with libmp3lame at 24 kHz mono, 48 kbit/s, written to a seekable file:
  - With the Info tag (`-write_xing 1`), the tag declares encoder delay 576 and padding 960 with the encoder string `Lavc63.1.`. `decode()` puts a burst authored at 200 ms at 200.02 ms, so ffmpeg trims the priming.
  - With no tag (`-write_xing 0`), the burst lands at 246.06 ms. That is (576 + 529) / 24000 s ≈ 46.04 ms late.
  - ffmpeg writes no Info tag to a pipe.

  The tests assert ±3 ms around these values. Cost if wrong: an ffmpeg build that differs fails the characterization test loudly, which is the intended signal.
- D8: The fakes are moved, not re-exported. The modules are `tests/fakes/{__init__,voice,align,llm,planning,images}.py`. `tests/llm/fakes.py` and `tests/planning/fakes.py` are deleted, and `FakeImages` leaves `tests/sketching/support.py`. The builders (`clean`, `scene`, `series`, `PAID_VISION`, `PRICES`) stay in `support.py`, because they are builders, not fakes. Cost if wrong: an import-path churn of about 20 lines.
- D9: There is one `FakeVoice`. Its defaults are the "plain" profile used by `fixtures/parts.py` and `audio/test_build.py`: no spans, `speechlike(n, 300, 80, lead_ms=150)`. `fail_on=""` replaces `fail=True`, because the empty string is in every text. `test_narrate.py` uses `TimedVoice(FakeVoice)`, which only changes the defaults to `spans=True, lead_ms=200`. Why: each test's audio stays byte-identical, so no expected value changes. Cost if wrong: one subclass more than strictly needed.
- D10: The smoke test runs the real Typer app on a copy of `examples/sample_ar.md` (Arabic), which gains a front-matter block. It patches only the factory seams:
  - `planning.pipeline.make_llm` becomes `ScriptedLLM`;
  - `sketching.generate.build_image_provider` raises `AssertionError`, because "scroll|generic" is bundled;
  - `audio.build.build_voice_chain` becomes `[VoiceChoice(FakeVoice("edge", "edge-tts", spans=True), "ar-SA-HamedNeural")]`;
  - `audio.build.CtcAligner.available` becomes `False`.

  The Qur'an fixture files are copied into `data_dir/quran`. It renders at full 1080x1920 size, because publish's technical check needs that size. Cost if wrong: a slower test (it is about 6 s of video), measured in the PR.
- D11: The render fingerprint is `RenderReport.inputs_key: str | None = None`, a `cache_key("render", RENDER_INPUTS_VERSION, payload)` over the fields listed in Task B3, with no paths and no worker count. `REPORT_SCHEMA_VERSION` stays 1, because the field is optional.
  - A part is skipped only when all of these hold: the report loads, `inputs_key` matches, the MP4 and the `.ass` exist, and `verify_mp4` passes.
  - `--force` and `build_render(force=True)` re-render.
  - `RENDER_INPUTS_VERSION = "1"` must be bumped by any change to frame or encode output.

  Cost if wrong: a renderer change without a bump serves stale videos until `--force`. Review Focus 1 and the ADR say so.
- D12: A skipped part still calls `on_part`. It is listed in `RenderResult.skipped: list[int]`, and the CLI prints `part n/m: unchanged, kept <mp4>`.
- D13: The #15 golden is `lightbulb|generic`, stored compact at `tests/visuals/golden/lightbulb-generic.strokes.json`.
  - The comparison is exact on width, height, stroke count, point count per stroke and `width` presence, and `abs <= 1e-6` per coordinate.
  - Regenerate it only with `uv run python tests/visuals/test_strokes_golden.py --update`, as a reviewed diff.
- D14: The #16 goldens are PNGs at 0%, 50% and 100% for the `_set()` stroke set already in `tests/render/test_reveal.py`, with seed 1, 200x120, ink `#112233` and `draw_schedule(_set(), 0, 3000)`. The times are t = 0, 1500 and `events[-1].end_ms`.
  - **Comparison (tolerance-based):** a frame passes when at most 0.5% of its pixels differ from the golden by more than 32 in any channel, and its ink-pixel count is within 2% of the golden's.
  - Why: this absorbs Pillow anti-aliasing drift between versions, but catches order, pacing or a missing stroke.
  - Cost if wrong: a sub-pixel regression passes.
- D15: The #17 benchmark is one Arabic part of 180 s: 18 scenes of 10 s each, each drawing a bundled icon.
  - It uses a written timeline and a sine FLAC, with no TTS.
  - It renders at full 1080x1920, timed once with the default workers and once with `--workers 1`.
  - The NFR passes when the default-worker wall time is under 600 s.
  - The script and a results slot ship. It is not a CI test.
- D16: The end card:
  - **Where it lives:** `Part.end_card_text: str = ""`. There is no series schema bump, because the field is additive with a default.
  - **Who sets it:** the planner, on the final part only, from `ChannelConfig.end_card: str | None = None`. None gives the language default: "Follow for more" in English, "تابعونا للمزيد" in Arabic. An empty string turns it off.
  - **Display:** it is burned as a `Headline` over the final part's last `END_CARD_MS = 3000`. That is the slot the teaser uses on other parts, so the two never overlap.
  - **Review:** the content review sees it as an `[end card]` line.

  Series planned before this change have no end card until they are re-planned or edited. Cost if wrong: the owner dislikes the default wording, which is one config line.
- D17: The Piper licence:
  - **Config:** `VoiceConfig` gains `piper_model_license: str | None`, `piper_model_source: str | None` and `piper_model_attribution: str = ""`. A licence without a source is a config error.
  - **Carried by:** `PiperProvider` keyword arguments, then three optional `VoiceUsed` fields.
  - **Provenance:** the voice record uses the declared licence, looked up in `[licences.voice]` first, then `[licences.assets]`. So MIT passes, CC-BY-4.0 needs an attribution, and CC-BY-NC-4.0 fails.
  - **Undeclared:** it stays `piper-model-license` WARN, with a note naming the config key.
  - **Cache:** the licence is not in the TTS cache key, so nothing is re-voiced.
- D18: The ADRs are numbered 0002 compositor, 0003 TTS and 0004 IR. They are "Accepted, 2026-11-13" and record what was built, citing code paths. 0004 records the move from `storyboard.json` to `series.json`, with the real versioning rule: exact-match refusal, no migrations, and additive optional fields do not bump. ADR 0001's pending list is replaced with links.
- D19: #20 is closed by PR B as "superseded by the staged CLI" (smoke test plus render skip). Its one-command `render examples/sample_ar.md` criterion is dropped, and the PR body says so.
- D20: #5's "CI < 2 min" is recorded, not enforced. The PR body quotes the wall time of the gate's pytest run.
- D21 (cold read): the render payload also carries `line_weight_px` and each sketch's `mirror_safe`; a changed line weight or mirror flag must re-render (D11 forbids silent reuse).
- D22 (cold read): `_run_jobs` is called once per part; tests count calls per part.
- D23 (cold read): on the final part the end card caps the recap like the teaser does; the two Headline events never overlap.
- D24 (cold read): B2's smoke test does `monkeypatch.setenv("NARRATOFLOW_FFMPEG", str(render_tools.ffmpeg))` (and the ffprobe variable if `Settings` has one) after the autouse env isolation, so the CLI uses the same ffmpeg as the fixture. Fix test setup, never production code.
- D25 (cold read): the benchmark's `run` passes `force=True` and every run uses a fresh `out_dir` under the work root, so B3's skip can never record a skipped run as a time.
- D26 (cold read): platform rules become `version = "2026-11-12.1"`, `checked = 2026-11-12`; C5 updates the pinned version in `tests/publish/test_rules.py` and `tests/publish/test_provenance.py`.
- D27 (cold read, non-blocking defaults): `ffmpeg_bin` moves from `tests/audio/conftest.py` to `tests/conftest.py`; a buffer ending right after the Xing fields means "no LAME extension" (None), a buffer cut inside a declared field raises `AudioError`; spike helpers are widened to `float`; `edge_tts_version` comes from `importlib.metadata.version("edge-tts")` or "unknown"; labels live at `DIR/labels.json`; `BadAudioVoice` subclasses `TimedVoice`; the B1 grep output is the binding file list; `make_llm` is patched with a 4-argument lambda returning `ScriptedLLM()`; any other test that renders twice into one out dir and expects fresh frames passes `force=True`; a skipped part prints only the "unchanged, kept" line plus its warnings; the stroke golden compares coordinates within 1e-6 and, if CI's Linux differs, the tolerance is widened to 1e-3 with a comment, never the golden regenerated in CI; C2 uses a fresh `RevealCanvas` per time and copies each frame, `line_weight_px=4.0`, ink pixel = the existing `_ink` helper; `models.py` line 5's "add a migration" sentence is rewritten to the exact-match, no-migration rule; coverage TOTAL >= 97% is checked by the coordinator from `coverage report`.

## Global Constraints

- `.py` and `.toml` sources are pure ASCII. Arabic literals in source use `\u` escapes. The Arabic in this plan is for reading only.
- mypy strict and Ruff clean, with no `# type: ignore` added.
- Tests never use the network and never sleep for real. A test that needs an optional extra, ffmpeg or downloaded model data uses the existing skip-locally, fail-in-CI fixtures (`ffmpeg_bin`, `probe_tools`, `render_tools`).
- `NarratoFlowError` is caught only in `cli.main`.
- Secrets never appear in logs or error messages. Library errors are reported by exception type only, as in `align/ctc.py`.
- Production timings do not change in this plan:
  - PR A does not modify `audio/narrate.py`, `audio/mixer.py`, `audio/io.py`, `audio/words.py` or `align/ctc.py`;
  - no module under `src/` imports `audio.onsets` or `audio.mp3info`.
- Stage files by name in commits; never `git add -A`.
- Every code-writing agent follows the ub-mode skill: failing test first, then the smallest edit.
- Gate: `uv lock --check && uv run ruff check --no-cache && uv run ruff format --check && uv run mypy && uv run coverage run -m pytest -o addopts="" -q && uv run coverage report`.
- Coverage must not drop below 97%.
- PR order is A, B, C, D. Inside PR C, run C4 and then C5 (both touch `channels/config.py` and `README.md`). C1 to C3 are independent of each other.

## Review Focus

1. **A changed input must re-render.** Each of these must re-render the part:
   - a caption word, a timeline byte, the FLAC, a sketch's strokes, `style.ink` or the canvas scale;
   - an MP4 that was deleted or no longer verifies;
   - a report written before this plan (with no `inputs_key`).

   A second run with nothing changed must not call `_run_jobs`. Tests: `test_rerun_unchanged_skips_part` and the parametrized `test_changed_input_rerenders`, Task B3.
2. **A crash mid-render must not leave a report that a later run trusts.** The old report is unlinked before the work starts, so a failure in `encode_part` leaves no report, and the next run renders. Test: `test_failed_rerender_is_not_skipped_next_time`, Task B3.
3. **Malformed MP3 bytes.** An empty input, an ID3 tag with nothing after it, no frame sync, a non-Layer-III frame and a truncated Xing block must raise `AudioError` naming the problem, never `IndexError` or `struct.error`. An Info tag with no LAME extension returns `encoder_delay=None`. Test: `test_malformed_inputs_raise_audio_error`, Task A2.
4. **Piper licence edge cases.**
   - No declared licence keeps today's WARN.
   - A declared licence that is not allowed (CC-BY-NC-4.0) FAILs.
   - CC-BY-4.0 without an attribution FAILs.
   - A licence without a source is refused at channel load.

   Tests: `test_piper_declared_licence_verdicts` and `test_piper_licence_needs_source`, Task C5.
5. **End-card edge cases.**
   - A single-part series gets the end card.
   - A final part shorter than `HOOK_MS + MIN_OVERLAY_MS` shows no end card and never overlaps the hook.
   - `end_card = ""` turns it off.
   - Non-final parts are unchanged, still showing the teaser.

   Tests: `test_end_card_on_final_part_only`, `test_end_card_skipped_when_part_too_short` and `test_end_card_disabled`, Task C4.

---

## PR A - Word-timing ground truth tooling (#133 step 1, #25 measurement)

Branch `plan-10-a-timing-ground-truth`. The PR body says production timings are unchanged and ends with `Refs #133`, plus `Closes #25` or `Refs #25` according to D6.

### Task A1: Energy onset/offset detector (#133)

**Files:**
- Create: `src/narratoflow/audio/onsets.py`
- Test: `tests/audio/test_onsets.py`

**Interfaces:**
- Consumes: `audio.io.Samples`; `fixtures.audio.tone(ms, hz, dbfs, lead_ms)` in tests.
- Produces:
  - `WINDOW_MS = 5`, `THRESHOLD_DBFS = -45.0`, `MIN_VOICED_MS = 20`, `MIN_GAP_MS = 60`;
  - `@dataclass(frozen=True) class VoicedRegion: start_ms: float; end_ms: float`;
  - `voiced_regions(samples: Samples, sample_rate: int, *, threshold_dbfs: float = THRESHOLD_DBFS, window_ms: int = WINDOW_MS, min_voiced_ms: int = MIN_VOICED_MS, min_gap_ms: int = MIN_GAP_MS) -> list[VoicedRegion]`;
  - `first_onset_ms(samples, sample_rate, **same keywords) -> float | None`;
  - `last_offset_ms(samples, sample_rate, **same keywords) -> float | None`.

- [ ] **Step 1: Write the failing tests** in `tests/audio/test_onsets.py`. Build the signal at 48 kHz with `np.concatenate`: 200 ms of zeros, a 300 ms tone, 100 ms of zeros, a 300 ms tone, 400 ms of zeros, a 200 ms tone and 300 ms of zeros, with the tones at -20 dBFS.
  - `test_regions_match_bursts`: the default call gives 3 regions, each start and end within 5 ms of (200, 500), (600, 900) and (1300, 1500).
  - `test_min_gap_merges_short_pauses`: `min_gap_ms=150` gives 2 regions, about (200, 900) and (1300, 1500).
  - `test_first_and_last`: `first_onset_ms` is about 200 and `last_offset_ms` is about 1500, each ±5.
  - `test_short_click_dropped`: a lone 5 ms burst in silence gives `[]`, and `first_onset_ms` is None.
  - `test_silent_and_empty`: zeros and an empty array give `[]`, None and None.
  - `test_voiced_from_sample_zero`: a tone starting at t = 0 gives an onset of 0.0.
  - `test_quiet_tone_below_threshold`: a -60 dBFS tone gives `[]`.
  - `test_bad_arguments`: `sample_rate <= 0`, or a `window_ms` that gives fewer than 1 sample, raises ValueError.

- [ ] **Step 2: Run** `uv run pytest tests/audio/test_onsets.py -q -o addopts=""`. Expected: ImportError.

- [ ] **Step 3: Implement.**
  - Window RMS uses the same reshape-and-pad approach as `io.trim_silence`. The last window is divided by its real size.
  - A window is voiced when its RMS is above `10 ** (threshold_dbfs / 20)`.
  - Runs are converted to ms as `index * window_ms` for a start and `(last + 1) * window_ms` for an end, capped at the clip length.
  - Gaps shorter than `min_gap_ms` are merged first, then runs shorter than `min_voiced_ms` are dropped.
  - The module docstring states that energy onsets are reliable at the sentence start, after pauses and at the final word's end, and not reliable inside connected speech.

- [ ] **Step 4: Run the gate.** Expected: green, with `onsets.py` at 100% branch coverage.

- [ ] **Step 5: Commit**

```bash
git add src/narratoflow/audio/onsets.py tests/audio/test_onsets.py
git commit -m "feat(audio): energy onset and offset detector for timing ground truth (#133)"
```

---

### Task A2: MP3 header reader and decoder-priming characterization (#133)

**Files:**
- Create: `src/narratoflow/audio/mp3info.py`
- Test: `tests/audio/test_mp3info.py`

**Interfaces:**
- Consumes: `errors.AudioError`; `audio.io.decode`; `audio.onsets.first_onset_ms` (Task A1).
- Produces:
  - `@dataclass(frozen=True) class Mp3Info`, with these fields:
    - `mpeg_version: Literal["1", "2", "2.5"]`
    - `sample_rate: int`
    - `channels: int`
    - `samples_per_frame: int` (1152 or 576)
    - `tag: Literal["Xing", "Info", "VBRI"] | None`
    - `encoder: str | None` (the 9-byte LAME/Lavc string, stripped)
    - `encoder_delay: int | None`
    - `encoder_padding: int | None`
    - `frames: int | None`
  - `read_mp3_info(data: bytes) -> Mp3Info`.
  - `DECODER_DELAY_SAMPLES = 529`.

- [ ] **Step 1: Write the failing tests** in `tests/audio/test_mp3info.py`. Write a local builder `_mp3(*, id3: bool, version, rate_index, mono, tag, flags, lame: tuple[int, int] | None, vbri_delay: int | None) -> bytes`. It emits an optional 10-byte ID3v2 header with a syncsafe size, then a 4-byte frame header, zeroed side info, and the tag fields in their standard offsets:
  - Xing/Info side-info offset: 17 for MPEG1 mono, 32 for MPEG1 stereo, 9 for MPEG2/2.5 mono, 17 for MPEG2/2.5 stereo;
  - VBRI at byte 36 after the header;
  - LAME extension: 9-byte encoder string, then delay and padding as two 12-bit fields at extension bytes 21 to 23.

  The tests:
  - `test_info_tag_with_lame_delay`: MPEG2 at 24 kHz, mono, `Info`, flags 15, `Lavc63.1.`, delay 576 and padding 960 give `Mp3Info("2", 24000, 1, 576, "Info", "Lavc63.1.", 576, 960, <frames>)`.
  - `test_xing_without_lame_extension`: `encoder`, `encoder_delay` and `encoder_padding` are None.
  - `test_vbri_delay`: `tag == "VBRI"` and `encoder_delay` comes from VBRI.
  - `test_no_tag`: a plain frame gives `tag is None`.
  - `test_id3_is_skipped`: the same result with an ID3v2 prefix.
  - `test_malformed_inputs_raise_audio_error` (Review Focus 3), parametrized over: `b""`, an ID3 header with nothing after it, 1 KiB of zeros, a Layer II header, and an Info tag cut after its flags. Each raises `AudioError`.
  - `test_ffmpeg_honours_info_tag_delay(ffmpeg_bin, tmp_path)`:
    - build a 2 s, 24 kHz click train with 30 ms 1 kHz bursts at 200, 700 and 1200 ms;
    - encode it with `ffmpeg -f f32le -ar 24000 -ac 1 -i pipe:0 -c:a libmp3lame -b:a 48k -write_xing 1` into a file in `tmp_path`. It must be a file, not a pipe (D7).
    - `read_mp3_info` gives `tag == "Info"`, `encoder.startswith(("Lavc", "LAME"))` and `encoder_delay == 576`;
    - `first_onset_ms(decode(bytes), 48000)` is within 3 ms of 200.
  - `test_ffmpeg_without_tag_keeps_priming(ffmpeg_bin, tmp_path)`: the same with `-write_xing 0`. `tag is None`, and the onset minus 200 is within 3 ms of `(576 + DECODER_DELAY_SAMPLES) * 1000 / 24000` (46.04).

- [ ] **Step 2: Run** `uv run pytest tests/audio/test_mp3info.py -q -o addopts=""`. Expected: ImportError.

- [ ] **Step 3: Implement `read_mp3_info`.**
  - Skip an ID3v2 tag: a 10-byte header with a 28-bit syncsafe size, plus 10 more bytes when the footer flag is set.
  - Find the first `0xFFE` sync and check that the frame is Layer III.
  - Read the version, the sample-rate table and the channel mode.
  - Look for `Xing`/`Info` at the side-info offset. Read the flag-gated fields: frames (bit 1), bytes (bit 2), TOC (bit 4) and quality (bit 8).
  - Read the LAME extension when its first 4 bytes are `LAME` or `Lavc`. Otherwise look for `VBRI` at offset 36.
  - Every out-of-range read raises `AudioError("mp3 header: <what> is truncated")`.

- [ ] **Step 4: Run the gate.** Expected: green.

- [ ] **Step 5: Commit**

```bash
git add src/narratoflow/audio/mp3info.py tests/audio/test_mp3info.py
git commit -m "feat(audio): MP3 Xing/LAME/VBRI header reader and priming test (#133)"
```

---

### Task A3: Spike capture/analyse split, labels and procedure doc (#133, #25)

**Files:**
- Modify: `scripts/spike_word_timing.py`. Add `main(argv)`, argparse, capture I/O, the waveform analysis and labels. The legacy no-argument run is unchanged.
- Create: `docs/spikes/2026-11-13-word-onset-ground-truth.md`
- Modify: `docs/spikes/2026-11-12-arabic-word-timing.md` (one line under Recommendation linking the new doc)
- Test: `tests/align/test_spike_word_timing.py`

**Interfaces:**
- Consumes:
  - `audio.onsets.voiced_regions` (A1) and `audio.mp3info.read_mp3_info` (A2);
  - `audio.io.decode` and `SAMPLE_RATE`;
  - `audio.words.timings_from_spans` and `tokens`;
  - `align.ctc.CtcAligner`;
  - the existing `signed_deltas` and `offset_stats`.
- Produces, all inside the script:
  - `@dataclass(frozen=True) class Capture`: `index: int`, `sentence: str`, `audio: bytes`, `words: list[str]`, `edge: list[TimedSpan]`, `ctc: list[TimedSpan] | None`.
  - `save_capture(directory: Path, capture: Capture) -> None`, which writes the D3 files.
  - `write_manifest(directory: Path, *, voice: str, edge_tts_version: str, captured_at: str) -> None`.
  - `load_captures(directory: Path) -> list[Capture]`. It raises ValueError naming the file when a file is missing, when its JSON is malformed, or when `words` differs from `tokens(manifest sentence)`.
  - `@dataclass(frozen=True) class WaveTruth`: `first_onset_ms: float`, `last_offset_ms: float`, `pause_onsets_ms: list[float]`.
  - `wave_truth(samples: Samples, sample_rate: int, *, pause_gap_ms: int = 150) -> WaveTruth | None`. The pause onsets are the starts of regions 2 to n. It returns None for a silent clip.
  - `@dataclass(frozen=True) class WaveDeltas`: `first_start_ms: float`, `last_end_ms: float`, `pause_starts_ms: list[tuple[int, float]]`, the last holding `(word index, signed delta)`.
  - `wave_deltas(timings: Sequence[WordTiming], truth: WaveTruth, *, window_ms: int = 300) -> WaveDeltas`. Each pause onset is matched to the word whose start is nearest within `window_ms`, skipping word 0. Unmatched onsets are dropped.
  - `load_labels(path: Path) -> dict[int, list[tuple[int, int]]]`: sentence to `[(word, onset_ms)]`. It raises ValueError on a bad version, an unknown sentence or a negative onset.
  - `label_deltas(timings: Sequence[WordTiming], labels: Sequence[tuple[int, int]]) -> list[int]`. It raises ValueError on a word index out of range.
  - `analyse(directory: Path, out: TextIO) -> int`, the `--from` body.
  - `main(argv: Sequence[str] | None = None) -> int`:
    - no arguments: the legacy live report;
    - `--save DIR`: needs `edge_tts`, else exits 2 naming `uv sync --extra tts-edge`;
    - `--from DIR`: needs no extra;
    - `--save` and `--from` together: a usage error.

- [ ] **Step 1: Write the failing tests** in `tests/align/test_spike_word_timing.py`, using the existing `_script` loader:
  - `test_capture_round_trip(tmp_path)`: `save_capture` and `write_manifest`, then `load_captures`, return an equal `Capture`, with and without `ctc`.
  - `test_load_captures_names_the_bad_file`: deleting `01.edge.json` raises ValueError matching `01.edge.json`, and so does corrupting `01.words.json`.
  - `test_wave_truth_on_bursts`: three bursts, with gaps of 80 ms and 400 ms, give `first_onset_ms` about 200, `last_offset_ms` about the last burst end, and exactly one pause onset (the 400 ms gap, since 80 < 150). A silent clip gives None.
  - `test_wave_deltas_signed`:
    - timings with word 0 at 90 ms against truth 200 give `first_start_ms == -110`;
    - a last word ending at 2400 against an offset of 1540 gives `+860`;
    - a pause onset at 1000 with word 3 starting at 960 gives `(3, -40)`;
    - an onset 400 ms from any word is dropped.
  - `test_labels_round_trip_and_errors`: a valid file loads. Version 2, sentence 99 and `onset_ms: -1` each raise ValueError. `label_deltas` with word 3 at 1000 against a label of 940 gives `[60]`, and an out-of-range index raises.
  - `test_from_runs_offline(tmp_path, ffmpeg_bin, monkeypatch, capsys)`:
    - build a two-sentence capture with `fixtures.audio.to_mp3_bytes(speechlike(...))` and spans at known times, with no `ctc` files;
    - monkeypatch `importlib.util.find_spec` to return None for `edge_tts` and `ctc_forced_aligner`;
    - `main(["--from", str(tmp_path)])` returns 0;
    - stdout contains the MP3-header table, the edge-vs-waveform table and `CTC: not available`, with no CTC columns.
  - `test_from_scores_labels`: the same, with a `labels.json`. Stdout contains `labelled words: 2` and a signed median.
  - `test_save_without_edge_extra_exits_2`, and `test_save_and_from_together_is_usage_error`, which expects `SystemExit` code 2 from argparse.

- [ ] **Step 2: Run** `uv run pytest tests/align/test_spike_word_timing.py -q -o addopts=""`. Expected: the new tests fail with AttributeError.

- [ ] **Step 3: Implement** the script changes as the Interfaces block says.
  - `--from` prints, in order:
    1. per sentence: `tag`, `encoder`, `encoder_delay` and `encoder_padding`;
    2. one line on what that means for `decode()`, with D7's two measured cases quoted;
    3. a table per sentence of `edge first - onset`, `ctc first - onset`, `edge last end - offset`, `ctc last end - offset`, and the post-pause deltas for edge and CTC;
    4. pooled medians for each column;
    5. when labels exist, labelled-word signed deltas for edge and CTC, with the median, the absolute median and the absolute p95.
  - Every header repeats D4's sign rule.
  - The extras are imported inside the functions that need them only.

- [ ] **Step 4: Run the gate.** Expected: green.

- [ ] **Step 5: Write `docs/spikes/2026-11-13-word-onset-ground-truth.md`** with these sections:
  - **Purpose:** #133 step 1, and #25's measurement definition (D6).
  - **Decoder priming:** D7's measured table, and what the edge MP3's header must show to settle it.
  - **Capture:** `uv sync --extra tts-edge --extra align && uv run python scripts/spike_word_timing.py --save "$DIR"`, then `uv sync` to restore the environment. This is free but networked, and sends only the 11 fixed sentences.
  - **Offline analysis:** `uv run python scripts/spike_word_timing.py --from "$DIR"`.
  - **Manual labelling:**
    1. Open each `NN.mp3` in Audacity, after Tracks > Resample to 48 kHz, so the clock matches `decode()`.
    2. Add a label track.
    3. Label 2 or 3 mid-sentence word onsets per sentence at the first visible energy rise of the word's first consonant, checked by ear with the selection-play.
    4. Export the labels, then convert them into the D5 `labels.json` by hand, with word indices from `NN.words.json`.
  - **What energy can and cannot decide**, from the A1 docstring.
  - **Results:** a slot for the `--from` output. If no capture has been run, write "not yet run: needs one networked `--save`". Do not invent numbers.
  - **Next steps:** #133 steps 2 to 4, unchanged.

- [ ] **Step 6 (conductor only, when network is allowed for this run):** run the capture and analysis commands above, and paste the `--from` output into Results with the date and the edge-tts version. Skip this step without network. Do not commit the capture directory.

- [ ] **Step 7: Commit**

```bash
git add scripts/spike_word_timing.py tests/align/test_spike_word_timing.py docs/spikes/2026-11-13-word-onset-ground-truth.md docs/spikes/2026-11-12-arabic-word-timing.md
git commit -m "spike(align): offline capture and waveform ground truth for word timings (#133, #25)"
```

---

## PR B - Shared fakes, CLI smoke test, render skip (#5, #20)

Branch `plan-10-b-fakes-smoke-skip`. The PR body ends with `Closes #5` and `Closes #20`, states D19, and quotes the measured pytest wall time (D20).

### Task B1: Shared `tests/fakes/` package (#5)

**Files:**
- Create: `tests/fakes/__init__.py`. Its docstring says: "Test doubles for every provider protocol. Import as `from fakes.voice import FakeVoice`."
- Create: `tests/fakes/voice.py`, holding `FakeVoice`, `word_spans`, `WORD_MS = 300`, `GAP_MS = 80`, `PLAIN_LEAD_MS = 150` and `TIMED_LEAD_MS = 200`.
- Create: `tests/fakes/align.py` (`FakeAligner`), `tests/fakes/llm.py` (`FakeLLM`, `FAKE_USAGE`), `tests/fakes/planning.py` (`ScriptedLLM`, `ConstantEmbedder`, `FATHA`, `SCRIPTED_IDEA`, `add_fatha`) and `tests/fakes/images.py` (`FakeImages`).
- Delete: `tests/llm/fakes.py`, `tests/planning/fakes.py`
- Modify:
  - `tests/sketching/support.py`: remove `FakeImages`, and import `FakeLLM` from `fakes.llm`;
  - `tests/fixtures/parts.py`: remove `FakeVoice`;
  - `tests/audio/test_narrate.py`: remove `FakeVoice`, `FakeAligner` and `word_spans`, and add `TimedVoice`;
  - `tests/audio/test_build.py`: remove `FakeVoice`, and change `fail=True` to `fail_on=""`.
- Modify the imports in every file that `grep -rlE "llm\.fakes|planning\.fakes|\.fakes import|FakeImages|FakeVoice" tests` lists. Today that includes `tests/test_icons_e2e.py`, `tests/test_cli.py`, `tests/costs/test_tracker.py`, `tests/sketching/*.py`, `tests/planning/*.py`, `tests/llm/test_fakes.py`, `tests/publish/test_content.py`, `tests/publish/test_build.py` and `tests/render/test_build.py`.
- Test: `tests/llm/test_fakes.py` (it keeps testing `FakeLLM`, now imported from `fakes.llm`), plus `tests/test_fakes_voice.py`.

**Interfaces:**
- Produces:
  - `FakeVoice(name: str = "paid", model: str = "m1", *, spans: bool = False, fail_on: str | None = None, error: Exception | None = None, billed: bool = True, word_ms: int = WORD_MS, gap_ms: int = GAP_MS, lead_ms: int = PLAIN_LEAD_MS)`.
    - `synthesize(req) -> Speech` appends `req` to `.requests` and raises `error` (default `ProviderError("HTTP 500")`) when `fail_on is not None and fail_on in req.text`.
    - Otherwise its audio is `to_wav_bytes(speechlike(len(req.text.split()), word_ms, gap_ms, lead_ms=lead_ms))`, with spans `word_spans(req.text, lead_ms, word_ms, gap_ms)` when `spans` is set, else `()`, and `billed_chars = len(req.text) if billed else None`.
  - `word_spans(text: str, lead_ms: int = TIMED_LEAD_MS, word_ms: int = WORD_MS, gap_ms: int = GAP_MS) -> tuple[TimedSpan, ...]`.
  - `FakeAligner(error: Exception | None = None, *, shift_ms: int = 10)`. It returns `word_spans(text, lead_ms=TIMED_LEAD_MS + shift_ms)`, as today.
  - In `test_narrate.py`: `class TimedVoice(FakeVoice)`, whose `__init__` only changes the defaults to `spans=True, lead_ms=TIMED_LEAD_MS`. Every former `FakeVoice(` in that file becomes `TimedVoice(`, including `class Mismatch(TimedVoice)`.

- [ ] **Step 1: Write the failing test** `tests/test_fakes_voice.py`:
  - `test_plain_profile_matches_old_parts_fake`: `FakeVoice().synthesize(SpeechRequest("a b c", "v", Language.ENGLISH))` gives audio equal to `to_wav_bytes(speechlike(3, lead_ms=150))`, with `spans == ()`, `billed_chars == 5`, and provider and model `("paid", "m1")`.
  - `test_timed_profile_matches_old_narrate_fake`: `FakeVoice(spans=True, lead_ms=200)` gives audio equal to `to_wav_bytes(speechlike(3, 300, 80, lead_ms=200))`, with spans `[("a", 200, 500), ("b", 580, 880), ("c", 960, 1260)]`.
  - `test_fail_on_empty_always_fails` and `test_fail_on_substring`. The request is recorded in both cases.

- [ ] **Step 2: Run** `uv run pytest tests/test_fakes_voice.py -q -o addopts=""`. Expected: ImportError (no module `fakes`).

- [ ] **Step 3: Move the code.**
  - Move the bodies verbatim. Only `FakeVoice` is merged, following D9.
  - Update the imports, and delete the old modules.
  - `tests/publish/test_build.py:657` has `class _ElevenVoice(FakeVoice): name = "elevenlabs"`. The shared `__init__` sets `self.name`, which would silently shadow that class attribute. Delete the subclass and construct `FakeVoice("elevenlabs")` where `_ElevenVoice()` was used.
  - Do not edit any test's assertions or expected values. If a test's result changes, the merge is wrong: fix the fake, not the test.

- [ ] **Step 4: Run the gate.** Expected: green, with exactly as many tests passed and skipped as on `main`, plus the 4 new ones. Compare the summary line from `uv run pytest -o addopts="" -q` on `main` and on the branch.

- [ ] **Step 5: Commit.** Stage every file listed above by name, including the `git rm` of the two deleted modules.

```bash
git commit -m "test: consolidate fakes into tests/fakes with one FakeVoice (#5)"
```

---

### Task B2: CLI smoke test from text to a verified MP4 and disclosure (#5, #20)

**Files:**
- Modify: `examples/sample_ar.md`. Prepend a front-matter block: `+++`, `author = "NarratoFlow"`, `work = "Sample"`, `rights = "own"`, `+++`. The body is unchanged.
- Create: `tests/test_smoke_e2e.py`

**Interfaces:**
- Consumes:
  - `fakes.voice.FakeVoice` and `fakes.planning.ScriptedLLM` (B1);
  - `cli.app`, through `typer.testing.CliRunner`;
  - the `channel_file`, `data_dir` and `render_tools` fixtures from `tests/conftest.py`;
  - `render.encoder.verify_mp4` and `render.ffmpeg.probe`;
  - `quran.corpus.SIMPLE_NAME` and `UTHMANI_NAME`.

- [ ] **Step 1: Write the test** `test_text_to_verified_mp4_and_disclosure(channel_file, data_dir, render_tools, tmp_path, monkeypatch)`:
  - **Setup:**
    - Copy `examples/sample_ar.md` into `tmp_path`.
    - Copy `tests/fixtures/quran/quran-simple-clean.txt` and `quran-uthmani.txt` into `data_dir / "quran"`.
    - `monkeypatch.setenv("NARRATOFLOW_IMAGE_API_KEY", "img")`.
    - Apply D10's four patches with `monkeypatch.setattr` on the module objects: `narratoflow.planning.pipeline`, `narratoflow.sketching.generate` and `narratoflow.audio.build`.
  - **Run** these in order, each asserting `exit_code == 0` and printing `result.output` on failure:
    1. `plan <md> --channel <toml>`
    2. `approve <series> --yes`
    3. `sketch generate <series> --channel <toml>`
    4. `audio build <series> --channel <toml>`
    5. `render <series> --channel <toml> --workers 2`
    6. `publish <series> --channel <toml>`
  - **Assert:**
    - `tmp_path / "audio" / "part-01.mp4"` passes `verify_mp4` at size (1080, 1920), with the frames and duration from `part-01.render.json`;
    - `probe(render_tools, ...)` reports `h264` and `aac`;
    - `tmp_path / "export" / "part-01.disclosure.txt"` exists and is non-empty;
    - the image provider was never built;
    - the fake voice received at least one request.

  If the plan yields several parts, assert on every part.

- [ ] **Step 2: Run** `uv run pytest tests/test_smoke_e2e.py -q -o addopts=""`.
  - Before the front matter is added, `plan` fails with "declare the script source". Add it.
  - Expected after that: PASS. If any stage fails, the cause is a missing seam or fixture: fix the test setup, never production code, unless the failure is a real bug. A real bug gets its own failing unit test first.

- [ ] **Step 3: Time it.** Run `uv run pytest tests/test_smoke_e2e.py -q -o addopts="" --durations=1` and note the time for the PR body.

- [ ] **Step 4: Run the gate.** Expected: green.

- [ ] **Step 5: Commit**

```bash
git add examples/sample_ar.md tests/test_smoke_e2e.py
git commit -m "test: CLI smoke test from Arabic text to verified MP4 and disclosure (#5, #20)"
```

---

### Task B3: `render` skips parts whose inputs are unchanged (#20)

**Files:**
- Modify: `src/narratoflow/render/report.py` (add `inputs_key: str | None = None` to `RenderReport`)
- Modify: `src/narratoflow/render/build.py`:
  - add `RENDER_INPUTS_VERSION = "1"`;
  - add `_Run.inputs_key(plan) -> str` and the skip check at the top of `_Run.part`;
  - add the `force` parameter;
  - add `RenderResult.skipped`.
- Modify: `src/narratoflow/cli.py` (`render --force`, and the skipped line in `_print_render`)
- Modify: `README.md` (render section: re-runs skip unchanged parts, `--force`, and what counts as an input)
- Test: `tests/render/test_build.py` (new tests, and `force=True` on the second render of `test_rerender_identical_frames` so it still re-renders), plus `tests/test_cli.py`

**Interfaces:**
- Consumes: `pipeline.cache.cache_key(stage, version, payload) -> str` (24 hex characters); `AssDocument.to_text()`; `StrokeSet.to_json()`; `SketchRecord.blob_hash`; `HandSprite.id`, `.sha256` and `.tip`.
- Produces:
  - `build_render(..., force: bool = False)`;
  - `RenderResult.skipped: list[int]`, the part indexes that were kept unchanged;
  - `RenderReport.inputs_key`.
- The payload for `cache_key("render", RENDER_INPUTS_VERSION, payload)`. Every value is JSON, with no filesystem paths and no worker count:
  - `series`: `{title, channel_slug, language}`
  - `part`: `{index, count}`
  - `scenes`: `[s.model_dump(mode="json") for s in part.scenes]`
  - `captions_sha256`: the sha256 of `plan.doc.to_text()`
  - `timeline_sha256` and `audio_sha256`: the sha256 of the timeline file's bytes and of the FLAC's bytes
  - `sketches`: `{scene_id: {"blob": record.blob_hash, "strokes": sha256(strokes.to_json()), "mirror_safe": record.mirror_safe}}`
  - `verses`: `[v.model_dump(mode="json") for v in plan.verses]`
  - `assets`: `[a.model_dump(mode="json") for a in self.assets(plan.doc)]`
  - `hand`: `{id, sha256, tip}`
  - `ink`: `channel.style.ink`
  - `line_weight_px`: `channel.style.line_weight_px` (render passes it into every `SceneJob`, `render/build.py:392`)
  - `canvas`: `[width, height]`
  - `crf`: `CRF`
  - `ffmpeg`: `tools.version`
  - `warnings`: the warnings list the report would store.

- [ ] **Step 1: Write the failing tests** in `tests/render/test_build.py`. Wrap `build_module._run_jobs` in a spy that counts calls and delegates to the real function.
  - `test_rerun_unchanged_skips_part`:
    - Render `_one_part(library)` once, then again into the same out dir.
    - The second run makes no `_run_jobs` call, gives `result.skipped == [1]`, an equal report and an unchanged MP4 mtime, and calls `on_part` once.
    - The report's `inputs_key` is a 24-hex-character string.
  - `test_changed_input_rerenders`, parametrized over these changes:
    - edit a word in the series;
    - rewrite the timeline with different word times;
    - re-encode the FLAC with `tone(..., hz=330)`;
    - `library.put_strokes` with one stroke fewer, after rewriting the timeline with the new `stroke_count`;
    - change the channel to `style.ink = "#222222"`;
    - change `canvas=Canvas(Fraction(1, 3))`;
    - delete the MP4;
    - truncate the MP4 to 100 bytes;
    - rewrite the report with `inputs_key=None`.

    Each makes exactly one `_run_jobs` call (it is called once per part, `render/build.py:453`) and gives `skipped == []`. Add a `line_weight_px` change and a `mirror_safe` flip to the parameter list.
  - `test_force_rerenders`: `force=True` re-renders an unchanged part.
  - `test_inputs_key_ignores_paths_and_workers`: renders into `out="a"` with `workers=1` and into `out="b"` with `workers=2` give equal `inputs_key`.
  - `test_failed_rerender_is_not_skipped_next_time` (Review Focus 2):
    - render once;
    - change a word;
    - monkeypatch `build_module.encode_part` to raise `RenderError`, and render, which raises;
    - `part-01.render.json` no longer exists;
    - restore `encode_part`, revert the word, and render: the part is re-rendered, not skipped.

  In `tests/test_cli.py`, add `test_render_force_flag_and_skipped_line`. `_patch_render` records `force=True`, and a result with `skipped=[1]` prints `part 1/1: unchanged, kept`.

- [ ] **Step 2: Run** `uv run pytest tests/render/test_build.py tests/test_cli.py -q -o addopts=""`. Expected: the new tests fail.

- [ ] **Step 3: Implement.**
  - At the top of `_Run.part`, compute `key = self.inputs_key(plan)`. Unless `force` is set, skip when:
    - `report_path` loads (any load error means not skipped);
    - `report.inputs_key == key`;
    - the video and captions files exist;
    - `verify_mp4(...)` does not raise.

    A skipped part returns `PartVideo(n, video, report_path, report)` and sets a flag the caller adds to `skipped`.
  - Otherwise the current path runs unchanged, and `RenderReport` gets `inputs_key=key`.
  - Thread `force` through `_Run` and `build_render`. Add the CLI `--force` option, with help text "Re-render even when the inputs are unchanged."
  - Document D11's bump rule next to the `RENDER_INPUTS_VERSION` constant.

- [ ] **Step 4: Run the gate.** Expected: green.

- [ ] **Step 5: Update `README.md`** in the render section: a re-run skips parts whose inputs are unchanged; what the inputs are (a one-line summary of the payload); and that `--force` re-renders.

- [ ] **Step 6: Commit**

```bash
git add src/narratoflow/render/report.py src/narratoflow/render/build.py src/narratoflow/cli.py tests/render/test_build.py tests/test_cli.py README.md
git commit -m "feat(render): skip parts whose render inputs are unchanged (#20)"
```

---

## PR C - Close-out pack (#15, #16, #17, #27, #24)

Branch `plan-10-c-close-out`. The PR body ends with `Closes #15`, `Closes #16`, `Closes #17`, `Closes #27` and `Closes #24`. Write `Refs #17` instead if Task C3's result slot stays "not yet run".

### Task C1: Stroke golden for a shipped icon and the schema doc (#15)

**Files:**
- Create: `tests/visuals/test_strokes_golden.py`, `tests/visuals/golden/lightbulb-generic.strokes.json`
- Create: `docs/strokes-schema.md`
- Modify: `README.md` (one line linking `docs/strokes-schema.md`)

**Interfaces:**
- Consumes: `visuals.bundled.default_bundled_icons().get(parse_idea_key("lightbulb|generic"))`, whose `.file` is a PNG path; `visuals.lineart.binarize(png: bytes)`; `visuals.strokes.extract_strokes(ink, rtl=False)`; `StrokeSet.from_json` and `.to_json`.
- Produces: `extract_icon(key: str) -> StrokeSet`, and a `__main__` block that writes the golden when it is given `--update`. The module imports only `narratoflow`, so it runs as a plain script.

- [ ] **Step 1: Write the test** `test_lightbulb_strokes_match_golden`. Compare as D13 says, with a helper `_assert_same(got, want)` whose failure message names the first differing stroke index. Add `test_golden_is_valid_schema`: `StrokeSet.from_json(golden bytes)` has `version == 1`.
- [ ] **Step 2: Run it.** Expected: FAIL with FileNotFoundError on the golden.
- [ ] **Step 3: Generate the golden** with `uv run python tests/visuals/test_strokes_golden.py --update`. Check that it has more than one stroke and that every point is in [0, 1]. Run the test again. Expected: PASS.
- [ ] **Step 4: Write `docs/strokes-schema.md`.** It states the fields from `visuals/strokes.py`:
  - `version` (always 1), `width` and `height` (pixels of the source raster), and `strokes[]`;
  - each stroke's `points` as normalized `(x, y)` in [0, 1], in drawing order;
  - `width`, present only for dot "taps", as the dot diameter divided by the image width;
  - the serialization: compact, sorted keys, UTF-8;
  - the ordering rule, from the module docstring's steps 7 and 8;
  - a 2-stroke example;
  - when to bump `version`.
- [ ] **Step 5: Run the gate.** Expected: green.
- [ ] **Step 6: Commit**

```bash
git add tests/visuals/test_strokes_golden.py tests/visuals/golden/lightbulb-generic.strokes.json docs/strokes-schema.md README.md
git commit -m "test(visuals): golden stroke JSON for a shipped icon; document the schema (#15)"
```

---

### Task C2: Reveal goldens at 0%, 50% and 100% (#16)

**Files:**
- Create: `tests/render/test_reveal_golden.py`, plus `tests/render/golden/reveal-000.png`, `reveal-050.png` and `reveal-100.png`

**Interfaces:**
- Consumes: `render.reveal.RevealCanvas`, `visuals.marker.prepare_marker`, `visuals.schedule.draw_schedule` and `visuals.strokes.Stroke`/`StrokeSet`. Copy the `_set()` literal from `tests/render/test_reveal.py` into the new file, so neither file imports the other.
- Produces: `frames() -> dict[str, Image.Image]` keyed `"000"`, `"050"` and `"100"` (D14 times); `close_enough(got: Image.Image, want: Image.Image) -> str | None`, which returns None or a reason; and a `__main__` block with `--update`.

- [ ] **Step 1: Write the tests:**
  - `test_reveal_matches_golden`, parametrized over the three keys: `close_enough(...) is None`.
  - `test_close_enough_rejects_a_missing_stroke`: the 100% frame against the 50% golden returns a reason.
  - `test_close_enough_accepts_small_drift`: shifting the ink of 0.2% of the pixels by 20 levels returns None.
  - `test_zero_percent_is_nearly_blank`: the 0% golden has at most `2 * 4 * 4` ink pixels, which matches `test_reveal_monotonic_ink`.
- [ ] **Step 2: Run them.** Expected: FAIL, because the PNGs are missing.
- [ ] **Step 3: Generate the goldens** with `uv run python tests/render/test_reveal_golden.py --update`, saved with `optimize=True`. Look at the three PNGs (Read tool) and confirm they show 0, about half, and all of the strokes. Run the tests again. Expected: PASS.
- [ ] **Step 4: Run the gate.** Expected: green.
- [ ] **Step 5: Commit**

```bash
git add tests/render/test_reveal_golden.py tests/render/golden/reveal-000.png tests/render/golden/reveal-050.png tests/render/golden/reveal-100.png
git commit -m "test(render): stored 0/50/100% reveal goldens with tolerance compare (#16)"
```

---

### Task C3: CPU render benchmark for a 3-minute video (#17)

**Files:**
- Create: `scripts/bench_render.py`
- Create: `docs/benchmarks/2026-11-13-render-cpu.md`
- Test: `tests/render/test_bench_render.py` (loads the script with the `_script` loader pattern from `tests/align/test_spike_word_timing.py`)

**Interfaces:**
- Consumes:
  - `render.build.build_render`, `render.layout.Canvas`, `render.ffmpeg.find_render_tools`;
  - `library.repository.default_library`, `SketchLibrary.ensure_style_version`, `.add_sketch` and `.put_strokes`;
  - `channels.config.ChannelConfig` and `style_fingerprint`;
  - `visuals.bundled.default_bundled_icons`, `lineart.binarize` and `strokes.extract_strokes`;
  - `audio.timeline.PartTimeline`, `SceneTiming` and `TimelineWord`;
  - `audio.io.encode_flac`;
  - `settings.Settings(data_dir=...)`.
- Produces, inside the script:
  - `@dataclass(frozen=True) class BenchInputs`: `series: Series`, `channel: ChannelConfig`, `settings: Settings`, `audio_dir: Path`, `out_dir: Path`.
  - `prepare(root: Path, *, duration_ms: int = 180_000, scene_ms: int = 10_000) -> BenchInputs`. It writes the channel's `ref.png`, the library with one bundled icon per scene (cycling `default_bundled_icons().catalog()` in order), `part-01.timeline.json` and `part-01.flac`.
    - The narration is `"العلم نور"` ("العلم نور") repeated at 2 words per second.
    - Each scene draws during its first 60%.
  - `@dataclass(frozen=True) class BenchResult`: `wall_s: float`, `frames: int`, `duration_ms: int`, `workers: int | None`.
  - `run(inputs: BenchInputs, *, workers: int | None, canvas: Canvas | None = None, tools: RenderTools | None = None) -> BenchResult`, timed with `time.perf_counter`.
  - `main(argv) -> int`, with options `--minutes` (default 3), `--workers` and `--keep DIR`. It prints one Markdown row: `| date | platform.processor() / os.cpu_count() | ffmpeg version | workers | duration s | frames | wall s | NFR (< 600 s) pass/fail |`.

- [ ] **Step 1: Write the test** `test_bench_renders_a_short_series(render_tools, tmp_path)`. `prepare(tmp_path, duration_ms=3000, scene_ms=1500)`, then `run(..., workers=1, canvas=Canvas(Fraction(1, 6)), tools=render_tools)`, gives 90 frames and a positive `wall_s`, and the MP4 exists. Add `test_prepare_covers_duration`: the timeline scenes tile [0, 180000) with 18 scenes.
- [ ] **Step 2: Run them.** Expected: FAIL, because the script is missing.
- [ ] **Step 3: Implement** the script as the Interfaces block says.
- [ ] **Step 4: Run the gate.** Expected: green.
- [ ] **Step 5: Write `docs/benchmarks/2026-11-13-render-cpu.md`.** It covers:
  - the method (D15);
  - the commands `uv run python scripts/bench_render.py` and `uv run python scripts/bench_render.py --workers 1`;
  - the NFR;
  - a results table with the header row and one row per run.
- [ ] **Step 6: Run the benchmark once (offline, minutes)** with both commands, and paste the rows. If the run cannot finish in this environment, write "not yet run" and say why. Do not estimate.
- [ ] **Step 7: Commit**

```bash
git add scripts/bench_render.py tests/render/test_bench_render.py docs/benchmarks/2026-11-13-render-cpu.md
git commit -m "bench(render): 3-minute icon-path CPU benchmark script and result (#17)"
```

---

### Task C4: End card on the final part (#27)

**Files:**
- Modify: `src/narratoflow/models.py` (`Part.end_card_text: str = ""`)
- Modify: `src/narratoflow/channels/config.py` (`ChannelConfig.end_card: str | None = None`)
- Modify: `src/narratoflow/planning/series.py`:
  - `split_into_parts(..., end_card: str = "")` sets the text on the last part;
  - `END_CARD_DEFAULTS: dict[Language, str]`;
  - `end_card_text(channel_end_card: str | None, language: Language) -> str`.
- Modify: `src/narratoflow/planning/pipeline.py:183` (pass `end_card=end_card_text(channel.end_card, channel.language)`)
- Modify: `src/narratoflow/captions/overlays.py` (`END_CARD_MS = 3000`; the end-card window in `_headline`; the module docstring)
- Modify: `src/narratoflow/publish/content.py` (`review_text` adds `[end card] <text>` after the teaser line when non-empty; docstring)
- Modify: `README.md` (channel settings table: `end_card`, its defaults, `""` turns it off, and that it applies to newly planned series)
- Test: `tests/planning/test_series.py`, `tests/captions/test_overlays.py`, `tests/publish/test_content.py`, `tests/test_channels.py`

**Interfaces:**
- `END_CARD_DEFAULTS = {Language.ENGLISH: "Follow for more", Language.ARABIC: "تابعونا للمزيد"}`
- `end_card_text(None, lang)` returns the default, `end_card_text("", lang)` returns `""`, and any other string is returned stripped.

- [ ] **Step 1: Write the failing tests:**
  - `tests/planning/test_series.py`:
    - `test_end_card_on_final_part_only`: three parts with `end_card="X"` give `end_card_text` values `["", "", "X"]`, and a single part gets `"X"`.
    - `test_end_card_defaults`: the three `end_card_text` cases above, for both languages.
  - `tests/captions/test_overlays.py`:
    - `test_end_card_headline_at_end_of_final_part`: a final part of 10 000 ms with `end_card_text="Bye"` has a Headline event `(7000, 10000, "Bye")`. A non-final part with a teaser is unchanged from today.
    - `test_end_card_skipped_when_part_too_short`: d = 3000 (under `HOOK_MS + MIN_OVERLAY_MS`) gives no end-card event, and the hook stays at (0, 2500).
    - `test_end_card_disabled`: `end_card_text=""` gives no event, and the series title fills the gap as today.
  - `tests/publish/test_content.py`:
    - `test_end_card_in_review_text`: the line `[end card] Bye` is present.
    - `test_review_hash_unchanged_without_end_card`: equal to the hash of the same part before this task, using a literal digest recorded in the test with a comment naming the commit.
  - `tests/test_channels.py`:
    - `test_end_card_config`: `end_card = ""` loads as `""`, and an omitted key is None.

- [ ] **Step 2: Run** `uv run pytest tests/planning tests/captions/test_overlays.py tests/publish/test_content.py tests/test_channels.py -q -o addopts=""`. Expected: the new tests fail.

- [ ] **Step 3: Implement.** In `_headline`, when `part.index == part.count`, `part.end_card_text.strip()` is non-empty and `d - h >= MIN_OVERLAY_MS`, append `(max(h, d - END_CARD_MS), d, part.end_card_text)`. The text is truncated and escaped like the other headline text. The end card start also caps the recap exactly as the teaser does: the recap window becomes `(h, min(h + RECAP_MS, end_card_start))`, so the two never overlap (D16). Add `test_end_card_clips_recap_on_short_final_part`: final part 2 of 2 with d = 6000 has no two Headline events overlapping in time. The teaser branch is otherwise untouched.

- [ ] **Step 4: Run the gate.** Expected: green.

- [ ] **Step 5: Update `README.md`** as the Files list says.

- [ ] **Step 6: Commit**

```bash
git add src/narratoflow/models.py src/narratoflow/channels/config.py src/narratoflow/planning/series.py src/narratoflow/planning/pipeline.py src/narratoflow/captions/overlays.py src/narratoflow/publish/content.py README.md tests/planning/test_series.py tests/captions/test_overlays.py tests/publish/test_content.py tests/test_channels.py
git commit -m "feat(captions): end card on the final part, planned from the channel (#27)"
```

---

### Task C5: Record the real Piper model licence (#24)

**Files:**
- Modify: `src/narratoflow/channels/config.py` (the three `VoiceConfig` fields, and a validator requiring `piper_model_source` when `piper_model_license` is set)
- Modify: `src/narratoflow/tts/piper.py` (`PiperProvider.__init__(model_path, *, license: str | None = None, source: str | None = None, attribution: str = "")`, stored as `model_license`, `model_source` and `model_attribution`)
- Modify: `src/narratoflow/tts/factory.py` (`_piper` passes them)
- Modify: `src/narratoflow/audio/timeline.py` (`VoiceUsed.model_license: str | None = None`, `model_source: str | None = None`, `model_attribution: str = ""`)
- Modify: `src/narratoflow/audio/narrate.py:378` (fill them when `isinstance(choice.provider, PiperProvider)`)
- Modify: `src/narratoflow/publish/provenance.py`:
  - `_voice_records` uses the declared licence, its source and its attribution;
  - `_licence_table`: a voice record's licence is looked up in `licences.voice`, then `licences.assets`.
- Modify: `src/narratoflow/data/platform_rules.toml`. The `piper-model-license` note becomes "set voice.piper_model_license (and piper_model_source) to the voice model's licence". Bump `version` to `2026-11-12.1` and `checked = 2026-11-12` (never a future date: `publish/rules.py:154` WARNs on one). Update every pinned `"2026-11-11.1"` in `tests/publish/test_rules.py` and `tests/publish/test_provenance.py` (add both to Files and to the `git add` line).
- Modify: `README.md` (Piper section: the three keys, with a CC-BY-4.0 attribution example)
- Test: `tests/test_channels.py`, `tests/tts/test_factory.py`, `tests/audio/test_narrate.py`, `tests/publish/test_provenance.py`

**Interfaces:**
- Consumes: `fakes.voice.FakeVoice` (B1). For narrate, use a `PiperProvider` subclass whose `synthesize` delegates to a `FakeVoice`, so the `isinstance` check holds without the extra.

- [ ] **Step 1: Write the failing tests:**
  - `tests/test_channels.py`:
    - `test_piper_licence_needs_source`: setting `piper_model_license` without `piper_model_source` raises `ChannelConfigError`.
    - `test_piper_licence_loads`: with both set, the fields round-trip.
  - `tests/tts/test_factory.py`:
    - `test_piper_carries_declared_licence`: `build_voice_chain` gives a `PiperProvider` with the three attributes from the channel.
  - `tests/audio/test_narrate.py`:
    - `test_piper_voice_records_model_licence`: `VoiceUsed` has `model_license == "MIT"`, `model_source` set, and `terms == "piper-model-license"`.
    - `test_licence_not_in_cache_key`: changing the licence re-uses the cached clips, with no new request.
  - `tests/publish/test_provenance.py`:
    - `test_piper_declared_licence_verdicts`, parametrized:
      - MIT gives no voice finding;
      - CC-BY-4.0 with an attribution gives no finding;
      - CC-BY-4.0 without one gives a FAIL with "needs an attribution";
      - CC-BY-NC-4.0 gives a FAIL with "not allowed";
      - undeclared gives a WARN whose reason contains `voice.piper_model_license`.
    - `test_edge_voice_unchanged`: the edge WARN text is identical to before.

- [ ] **Step 2: Run** the four test files. Expected: the new tests fail.

- [ ] **Step 3: Implement** as the Files list says. Old timelines without the fields still load, because the fields have defaults and `_STRICT` only forbids unknown keys.

- [ ] **Step 4: Run the gate.** Expected: green.

- [ ] **Step 5: Update `README.md`** as the Files list says.

- [ ] **Step 6: Commit**

```bash
git add src/narratoflow/channels/config.py src/narratoflow/tts/piper.py src/narratoflow/tts/factory.py src/narratoflow/audio/timeline.py src/narratoflow/audio/narrate.py src/narratoflow/publish/provenance.py src/narratoflow/data/platform_rules.toml README.md tests/test_channels.py tests/tts/test_factory.py tests/audio/test_narrate.py tests/publish/test_provenance.py
git commit -m "feat(publish): record the declared Piper voice-model licence in provenance (#24)"
```

---

## PR D - ADRs (#2)

Branch `plan-10-d-adrs`, opened after PRs B and C are merged, so the ADRs describe the render skip and the Piper licence as built. The PR body ends with `Closes #2`.

### Task D1: ADRs 0002 compositor, 0003 TTS, 0004 IR (#2)

**Files:**
- Create: `docs/adr/0002-compositor-ffmpeg-frame-pipe.md`, `docs/adr/0003-tts-strategy.md`, `docs/adr/0004-series-ir-and-versioning.md`
- Modify: `docs/adr/0001-record-architecture-decisions.md`. Replace the "Pending decisions" sentence with links to 0002 to 0004.

Each ADR follows 0001's format: a `# ADR 000N: <title>` heading, a `**Status:** Accepted, 2026-11-13` line, then the sections **Context**, **Decision**, **Consequences** and **Alternatives considered**. Every claim cites a code path. No claim goes beyond what the code does. Read each cited file before writing its claim.

- [ ] **Step 1: Write 0002 (compositor).**
  - **Context:** REVIEW C1, that Remotion is source-available.
  - **Decision:** NumPy/Pillow frames are piped as raw RGB into ffmpeg FFV1 segments, in parallel per scene (`render/encoder.py` FrameSink, `render/build.py` `_run_jobs`, `--workers`). Then concat, burn the libass captions, mux, and encode with x264/AAC constants (`encoder.py` `X264_ARGS`, `AAC_ARGS`), checked by `verify_mp4`. Manim CE was not used. The output is 9:16 only (`LANDSCAPE_WARNING`).
  - **Consequences:**
    - determinism (`test_workers_do_not_change_frames`);
    - the render-input skip and its `RENDER_INPUTS_VERSION` bump rule (Task B3);
    - the CPU benchmark (Task C3 doc);
    - the libass dependency.
- [ ] **Step 2: Write 0003 (TTS).**
  - **Decision:**
    - the chain is the channel's primary voice, then the `voice.fallback` entries (`tts/factory.py` `build_voice_chain`);
    - edge and ElevenLabs retry (`tts/retry.py`);
    - paid voices are budget-tracked (`TrackedSpeechProvider`);
    - a failure re-voices the whole part with the next voice (`audio/narrate.py`);
    - the TTS cache is keyed on provider, model, voice id, spoken text and the local model file (`narrate._key`, `pipeline/cache.py`);
    - word timings come from the provider's spans, else the CTC aligner (`align/ctc.py`);
    - licence terms per provider go into `VoiceUsed`, including the declared Piper licence (Task C5).
  - **Consequences:** the open Arabic offset question (#133, both spike docs), and that edge has no explicit commercial licence (the rules WARN).
- [ ] **Step 3: Write 0004 (IR).**
  - **Decision:** `storyboard.json` (`Project`, schema v1) was superseded by `series.json` (`Series` then `Part` then `Scene`, `SCHEMA_VERSION = 2` in `models.py`). `Project` remains only for `narratoflow validate`. The per-stage artifacts are each versioned on their own:
    - `PartTimeline` (`TIMELINE_SCHEMA_VERSION = 1`);
    - `RenderReport` (`REPORT_SCHEMA_VERSION = 1`);
    - `StrokeSet` (`version = 1`, `docs/strokes-schema.md`).
  - **Versioning rule as built:**
    - a loader refuses any other `schema_version` with a message naming both versions;
    - there are no migrations, so re-planning or re-building is the upgrade path;
    - additive optional fields with defaults (`Part.end_card_text`, `RenderReport.inputs_key`, the `VoiceUsed` licence fields) do not bump the version;
    - a removed, renamed or re-typed field bumps it.
  - **Consequences:** the review UI (#30) edits `series.json`, and the `models.py` docstring still says "Every pipeline stage reads and writes a `Project`".
- [ ] **Step 4: Correct the `models.py` docstring.** Lines 3 and 4 become: "`Series` (schema v2) is the IR every stage reads and writes; `Project` (v1) remains only for `validate`. See docs/adr/0004." This is a docstring-only change, so no test changes.
- [ ] **Step 5: Run the gate.** Expected: green.
- [ ] **Step 6: Commit**

```bash
git add docs/adr/0001-record-architecture-decisions.md docs/adr/0002-compositor-ffmpeg-frame-pipe.md docs/adr/0003-tts-strategy.md docs/adr/0004-series-ir-and-versioning.md src/narratoflow/models.py
git commit -m "docs(adr): 0002 compositor, 0003 TTS strategy, 0004 series IR and versioning (#2)"
```
