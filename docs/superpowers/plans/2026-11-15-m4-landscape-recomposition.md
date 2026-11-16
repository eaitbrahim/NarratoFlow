# M4 - 16:9 Landscape Re-composition (#193, #194) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The 16:9 file `part-NN.16x9.mp4` becomes a side-by-side re-composition of the checked 9:16 export. The stage, top bar and captions zones are carried whole at 1:1 into the wide frame, and an ink guard fails closed. The pillarbox is gone.

**PRs (in order):**
- **PR A = #193** (slice 1): `panorama_layout`, `landscape_size`, `_check_ink` and `panorama_export` in `publish/landscape.py`. The station keeps calling `pillarbox_export`, so shipped behaviour is unchanged.
- **PR B = #194** (slice 2): the station calls `panorama_export`, `LANDSCAPE_NOTE` replaces `PILLARBOX_NOTE`, `pillarbox_export` and its tests are deleted, ADR 0006 is added and the docs are amended. **Start PR B only after PR A is merged into `main`**, and branch it from the new `origin/main`.

**Architecture:** One ffmpeg pass in `publish/landscape.py`. It crops three bled zones out of the 9:16 MP4 and places them in a white frame `landscape_size(scale)` big. It encodes with `X264_ARGS` and copies the audio. Before that pass, a separate ffmpeg pass (the ink guard) masks the same three bled source zones on every decoded frame and fails if any luma below 195 remains. Nothing under `render/`, `captions/` or `publish/verse.py` changes.

**Tech Stack:** Python 3.12, numpy, pytest, ffmpeg-full (libx264, `signalstats`, `drawbox`, `crop`, `pad`, `overlay`).

**Spec:**
- `.superpowers/orchestrate/2026-11-15-m1-m3-m4/issue-163/design.md` (D1-D9, slices, §3 doc changes, §4 scope);
- issues #193 and #194 (`gh issue view 193`, `gh issue view 194`). Their acceptance criteria are the requirements and are quoted verbatim below;
- controls registers `.superpowers/orchestrate/2026-11-15-m1-m3-m4/issue-193/controls.md` and `issue-194/controls.md`;
- `.superpowers/orchestrate/project.md` (proof commands, conventions).

**Accepted rulings (verbatim; do not reopen):**
- R5: #163 'native' = re-composition of the checked 9:16 MP4: three 9:16 zones cropped whole at 1:1 and placed side by side in 1920x1080 (stage (48,115,828,850), top bar (912,115,960,200), captions (978,450,828,180); bottom 108 px clear) in one ffmpeg pass in publish/landscape.py; no second render, no frame-exact-core change.
- R6: no 16:9 ASS and no second shown_sha256; verse pixels are the certified 9:16 burn; a per-frame ink guard (luma < 195 outside the bled zones) fails closed (RenderError, no file written, also when the guard cannot run or parse).
- R7: the 16:9 file stays a derivative made only from the part this export call just checked; not in the publish report.
- R8: no RENDER_INPUTS_VERSION bump, Canvas unchanged, LANDSCAPE_WARNING unchanged; output size = 1920x1080 times the source scale (320x180 at 1/6).
- R9: pillarbox_export removed with no fallback; PILLARBOX_NOTE becomes LANDSCAPE_NOTE; no migration of old files.
- R10: two slices; the guard's full-scale cost is measured on the 180 s benchmark part (scripts/bench_render.py prepares it) and recorded in the PR body; slice 1 uses the cheapest guard recipe meeting its criteria (signalstats preferred if it meets them).

**Proof commands** (from `.superpowers/orchestrate/project.md`; always with ffmpeg-full, otherwise the render and publish tests skip):

```bash
export NARRATOFLOW_FFMPEG=/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg
# focused
uv run pytest -o addopts="" -q tests/publish/test_landscape.py
uv run pytest -o addopts="" -q tests/ui/test_station.py tests/ui/test_app.py
# local gate (must exit 0, coverage TOTAL >= 97 %)
uv lock --check && uv run ruff check --no-cache . && uv run ruff format --check . && uv run mypy && uv run coverage run -m pytest -o addopts="" -q && uv run coverage report
# certification path untouched (must print nothing)
git diff --name-only origin/main...HEAD -- src/narratoflow/render src/narratoflow/captions src/narratoflow/publish/verse.py
```

**Touches and seams (from the issues):**

Touches #193: `src/narratoflow/publish/landscape.py`, `tests/publish/test_landscape.py`.
Seams #193: `publish.landscape.panorama_layout` / `landscape_size` (pure) — catches wrong rectangles, overlap, a zone outside the frame or in `LANDSCAPE_BOTTOM`, and drift from the 9:16 zone sizes; misses how libass or x264 fill the zones.
Seams #193: `publish.landscape.panorama_export` on real MP4s made by `tests/fixtures/parts.render` (ffmpeg-full) — catches wrong placement, lost or cut ink, a seam in the background, a stale or temp file left behind, and a guard that does not fire; misses full-scale timing (runs at 1/6) and inks paler than the guard threshold.

Touches #194: `src/narratoflow/ui/station.py`, `src/narratoflow/ui/app.py`, `src/narratoflow/publish/landscape.py`, `tests/ui/test_station.py`, `tests/ui/test_app.py`, `tests/publish/test_landscape.py`, `docs/adr/0006-landscape-recomposition.md`, `docs/adr/0002-compositor-ffmpeg-frame-pipe.md`, `docs/PRD.md`, `docs/SRD.md`, `README.md`.
Seams #194: `ui.station.export` with the `publishable` fixture (real render, publish and ffmpeg at 1/6) — catches a 16:9 file made from an unchecked part, a 9:16 file or render report changed by a 16:9 export, and a stale copy surviving; misses the Gradio wiring and full-scale output.
Seams #194: `ui.app` handlers through `FakeGradio` — catches the wrong note or ratio; misses the real gradio module, which the existing `importorskip` test covers.

## Global Constraints

- No file under `src/narratoflow/render/`, `src/narratoflow/captions/` or `src/narratoflow/publish/verse.py` is edited by either PR. `render.encoder` and `render.ffmpeg` are imported only: `X264_ARGS`, `FINAL_TIMEOUT_S`, `verify_mp4`, `probe`, `run_tool`, `RenderTools`; `render.layout`: `Canvas`, `Rect`, `STAGE`, `TOP_BAR`, `CAPTIONS`.
- `RENDER_INPUTS_VERSION` stays `"2"`; `LANDSCAPE_WARNING` text unchanged; `Canvas` unchanged; `PublishReport.outputs` unchanged.
- File name `landscape_name(index)` = `part-NN.16x9.mp4`, button text "Export 16:9", the `station.export` signature and return value: unchanged.
- Every failure of the 16:9 producer is a `narratoflow.errors.RenderError`; on any failure no temp file stays and any older target file is removed.
- Exact constants (full scale, `Rect(x, y, w, h)`): `STAGE_16X9 = Rect(48, 115, 828, 850)`, `TOP_BAR_16X9 = Rect(912, 115, 960, 200)`, `CAPTIONS_16X9 = Rect(978, 450, 828, 180)`, `LANDSCAPE_BOTTOM = Rect(0, 972, 1920, 108)`, `BLEED_PX = 8`, `INK_LUMA_MAX = 195`, `LANDSCAPE_SIZE = (1920, 1080)` (kept).
- `LANDSCAPE_NOTE = "16:9 is the checked 9:16 video re-arranged side by side"`.
- Commit trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; PR bodies end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`; squash merge.
- Tests never touch the network; render tests use `tests/fixtures/parts.render` (canvas 1/6, so sources are 180x320 and outputs 320x180).

## Review Focus

1. A source with a video stream but no audio stream: expect `RenderError` and no file (the `-map 0:a:0` fails or `verify_mp4` refuses). Pinned by `tests/publish/test_landscape.py::test_source_without_audio_is_refused` (Task A4).
2. A 9:16 source replaced while the export runs (the guard read one file and the encode read another): expect `RenderError`. Pinned by `test_source_changed_during_export_is_refused` (Task A4).
3. Guard output that is present but incomplete (fewer frame records than `frames`, a non-integer value): expect `RenderError`, never a pass. Pinned by `test_first_ink_rejects_unreadable_output` (Task A2).
4. A verse page whose ink touches a zone edge: the glyphs must be carried, and the guard must not false-fire on codec ringing. Pinned by `test_verse_ink_is_carried_whole` (Task A3), because a false fire raises there.
5. A target directory that does not exist yet: it is created and only the target file is left in it. Pinned by `test_panorama_has_landscape_size_and_frames` (Task A3), which writes into a fresh `export/` directory.

---

## PR A — #193: panorama producer with the ink guard

Branch: `feat/193-panorama-producer` from `origin/main` (a8bdc5f or later).

### File structure (PR A)

- Modify `src/narratoflow/publish/landscape.py`: add the constants above, `CANVAS_SCALES`, `Zone`, `PanoramaLayout`, `panorama_layout`, `landscape_size`, `_ink_filter`, `_first_ink`, `_check_ink`, `_compose_graph`, `_source_scale`, `_identity`, `panorama_export`. Keep `pillarbox_export`, `_PILLARBOX_VF` and `landscape_name` as they are. Update the module docstring (text in Task A3).
- Modify `tests/publish/test_landscape.py`: keep the three pillarbox tests and `test_landscape_name_is_zero_padded`, and add the tests below. Update the module docstring to `"""The 16:9 producers: the panorama re-composition (and the pillarbox until #194)."""`.

### Shared test helpers (written in Task A2, used by A2-A4)

All helpers go in `tests/publish/test_landscape.py`:

- `_part(channel, library, tmp_path, tools, scenes: list[Scene]) -> BuiltPart`: `write_timelines(series, tmp_path / "audio", library)`, then `render(series, channel, Settings(data_dir=tmp_path / "data"), tmp_path, tools, library=library)`, and returns the single part. `series = make_series(scenes)`. The fixture renders at `Canvas(Fraction(1, 6))`, so the source is 180x320 with 30 frames per 1000 ms scene. `BuiltPart` is whatever `render(...).parts[0]` is; annotate it with the type `build_render` returns (`narratoflow.render.build`). Importing a type from `render/` is allowed; editing is not.
- `_sketch_part(...)`: `_part(..., [narr("s1", "alpha beta", sketch=add_sketch(library)), narr("s2", "gamma")])`. That gives 60 frames, with drawing during frames 0-17 of s1 and captions in both scenes.
- `_verse_part(...)`: `_part(..., [verse("v1"), narr("s1", "alpha")])`. The verse is 31:33 in frames 0-29.
- `_frames(tools, video, pix: str = "rgb24") -> np.ndarray`: probe the size, then run `run_tool(tools.ffmpeg, ["-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", pix, "-"])` and reshape to `(n, h, w, 3)` for `rgb24` or `(n, h, w)` for `gray`.
- `_reencode(tools, src, dst, vf: str) -> Path`: run `run_tool(tools.ffmpeg, ["-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a:0", "-vf", vf, *X264_ARGS, "-c:a", "copy", str(dst)])`. It builds the bad sources: the planted mark, the wrong size.
- `MARK = Rect(165, 100, 3, 3)`. It sits at 1/6 scale, right of the stage and below the top bar. The marked source is `_reencode(tools, part.video_path, tmp_path / "src" / "marked.mp4", "drawbox=x=165:y=100:w=3:h=3:color=black:t=fill")`. Create `tmp_path / "src"` first; it is not the target directory. Each test using it first asserts `not any(z.src_bled.intersects(MARK) for z in panorama_layout(Fraction(1, 6)).zones)`, so the mark is provably outside every bled zone.
- `def _crop(a: np.ndarray, r: Rect) -> np.ndarray: return a[..., r.y:r.bottom, r.x:r.right, :]` for rgb, with a gray variant that drops the channel axis.

### Task A1: Pure geometry: `panorama_layout` and `landscape_size`

**Files:**
- Modify: `src/narratoflow/publish/landscape.py`
- Test: `tests/publish/test_landscape.py`

**Interfaces:**
- Produces:
  - `CANVAS_SCALES: tuple[Fraction, ...] = (Fraction(1), Fraction(1, 2), Fraction(1, 3), Fraction(1, 6))`.
  - `@dataclass(frozen=True) class Zone: name: str; src: Rect; dst: Rect; bleed: int`, with properties `src_bled` and `dst_bled`. Each is `Rect(r.x - bleed, r.y - bleed, r.w + 2 * bleed, r.h + 2 * bleed)`.
  - `@dataclass(frozen=True) class PanoramaLayout: scale: Fraction; source: tuple[int, int]; size: tuple[int, int]; zones: tuple[Zone, Zone, Zone]`. Zone order and names: `"stage"`, `"top_bar"`, `"captions"`.
  - `panorama_layout(scale: Fraction) -> PanoramaLayout`.
  - `landscape_size(scale: Fraction) -> tuple[int, int]`.

- [ ] **Step 1: Write the failing tests** (AC-193.1 to AC-193.4, control C1)

```python
def test_layout_scale_1_matches_the_design_table() -> None:
    layout = panorama_layout(Fraction(1))
    got = {z.name: (z.src, z.dst) for z in layout.zones}
    assert got == {
        "stage": (STAGE, Rect(48, 115, 828, 850)),
        "top_bar": (TOP_BAR, Rect(912, 115, 960, 200)),
        "captions": (CAPTIONS, Rect(978, 450, 828, 180)),
    }
    assert all((z.dst.w, z.dst.h) == (z.src.w, z.src.h) for z in layout.zones)

@pytest.mark.parametrize("scale", CANVAS_SCALES)
def test_layout_bled_rects_are_disjoint_and_inside(scale: Fraction) -> None:
    layout = panorama_layout(scale)
    canvas = Canvas(scale)
    assert layout.source == (canvas.width, canvas.height)
    src_frame = Rect(0, 0, canvas.width, canvas.height)
    dst_frame = Rect(0, 0, *landscape_size(scale))
    for z in layout.zones:
        assert src_frame.contains(z.src_bled) and dst_frame.contains(z.dst_bled)
        assert (z.dst.w, z.dst.h) == (z.src.w, z.src.h)  # 1:1 at every scale
        assert z.bleed == max(1, round(8 * scale))
    for a, b in itertools.combinations(layout.zones, 2):
        assert not a.src_bled.intersects(b.src_bled)
        assert not a.dst_bled.intersects(b.dst_bled)

def test_layout_keeps_the_bottom_and_side_margins_clear() -> None:
    for z in panorama_layout(Fraction(1)).zones:   # unbled zones (see planner ruling 3)
        assert not z.dst.intersects(LANDSCAPE_BOTTOM)
        assert z.dst.x >= 48 and 1920 - z.dst.right >= 48

def test_landscape_size_follows_the_canvas_scale() -> None:
    assert landscape_size(Fraction(1)) == (1920, 1080) == LANDSCAPE_SIZE
    assert landscape_size(Fraction(1, 6)) == (320, 180)
    with pytest.raises(ValueError, match="scale"):
        landscape_size(Fraction(1, 4))
```

Imports to add: `itertools`, `Fraction`, `Canvas`, `Rect`, `STAGE`, `TOP_BAR`, `CAPTIONS` from `narratoflow.render.layout`, and `CANVAS_SCALES`, `LANDSCAPE_BOTTOM`, `landscape_size`, `panorama_layout` from the module.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -o addopts="" -q tests/publish/test_landscape.py -k "layout or landscape_size"`
Expected: collection ERROR, `ImportError: cannot import name 'CANVAS_SCALES'`.

- [ ] **Step 3: Implement in `src/narratoflow/publish/landscape.py`**

- `landscape_size(scale)` calls `Canvas(scale)`, which raises `ValueError("unsupported canvas scale ...")` for a bad scale. It then returns `(int(1920 * scale), int(1080 * scale))`.
- `panorama_layout(scale)`: let `canvas = Canvas(scale)` and `b = max(1, round(BLEED_PX * scale))`. The pairs are `(canvas.stage, STAGE_16X9)`, `(canvas.top_bar, TOP_BAR_16X9)` and `(canvas.captions, CAPTIONS_16X9)`. For each pair the destination is `Rect(round(D.x * scale), round(D.y * scale), src.w, src.h)`: the destination origin is scaled, and the size is the scaled source zone's size. Do not use `D.scaled(scale)`; planner ruling 1 says why.

- [ ] **Step 4: Run them to verify they pass**

Run: same command. Expected: 7 passed (the parametrized test counts 4).

- [ ] **Step 5: Commit** `feat(publish): panorama_layout and landscape_size for the 16:9 re-composition (#193)`

### Task A2: The ink guard (`_first_ink`, `_check_ink`)

**Files:**
- Modify: `src/narratoflow/publish/landscape.py`
- Test: `tests/publish/test_landscape.py`, which also gets the shared helpers above

**Interfaces:**
- Consumes: `PanoramaLayout`, `Zone.src_bled` (A1).
- Produces:
  - `_ink_filter(layout: PanoramaLayout) -> str`.
  - `_first_ink(text: str, frames: int) -> tuple[int, int] | None`: the first `(frame, ymin)` with `ymin < INK_LUMA_MAX`, or None. It raises `RenderError` when the text is unreadable.
  - `_check_ink(tools: RenderTools, src: Path, layout: PanoramaLayout, *, frames: int) -> None`.

**Guard recipe (exact).** It is the `signalstats` variant, which R10 prefers. It was checked on ffmpeg-full while this plan was written: `drawbox` on `yuv420p` masks luma pixel-exactly at odd offsets, so no format conversion is needed.

```
drawbox=x={x}:y={y}:w={w}:h={h}:color=white:t=fill   (once per zone, using zone.src_bled, joined by ",")
,signalstats,metadata=mode=print:key=lavfi.signalstats.YMIN:file=ink.txt
```

- The filter masks the three **bled source** rectangles: the same rectangles that the compose pass crops. After masking, `YMIN` is the darkest luma left outside the zones. A frame has ink when `YMIN < INK_LUMA_MAX` (195, tv range; paper is 235, and white masks write 235).
- Command: `run_tool(tools.ffmpeg, ["-hide_banner", "-nostdin", "-v", "error", "-xerror", "-i", str(src.absolute()), "-map", "0:v:0", "-vf", _ink_filter(layout), "-f", "null", "-"], cwd=work, timeout_s=FINAL_TIMEOUT_S)`. `work` is a `tempfile.TemporaryDirectory(prefix="narratoflow-ink-")` in the system temp dir. The metadata file is named relatively (`ink.txt`), so no path is ever escaped inside a filtergraph, as in `encoder._FINAL_VF`. The target directory is never touched.
- Parse `work / "ink.txt"`. If the file is missing, raise `RenderError("ink guard wrote no output for {src.name}")`. `metadata=print` writes, for each frame, a line `frame:N    pts:...    pts_time:...` followed by `lavfi.signalstats.YMIN=V`.
- `_first_ink` is strict. Every non-blank line must match `^frame:(\d+)\s` or `^lavfi\.signalstats\.YMIN=(\d+)$`, and they must alternate, starting with a frame line. Frame numbers must be exactly `0..frames-1` in order. `V` must be an int in 0..255. Any deviation raises `RenderError(f"ink guard output is unreadable ({reason})")`. That is fail-closed: a short or garbled file is never a pass.
- `_check_ink` raises `RenderError(f"{src.name}: ink outside the 16:9 zones at frame {n} (luma {v} < {INK_LUMA_MAX}); the 9:16 layout lets text or drawing leave its zone")` for the first hit. An ffmpeg that cannot run or exits non-zero is already a `RenderError` from `run_tool` (fail-closed, C3).

- [ ] **Step 1: Write the failing tests** (controls C2 and C3)

```python
@pytest.mark.parametrize(
    ("text", "frames"),
    [
        ("", 2),                                                     # nothing written
        ("frame:0 pts:0 pts_time:0\nlavfi.signalstats.YMIN=235\n", 2),  # a frame missing
        ("frame:0 pts:0 pts_time:0\nlavfi.signalstats.YMIN=abc\n", 1),  # not an int
        ("frame:0 pts:0 pts_time:0\n", 1),                           # no value
        ("frame:1 pts:0 pts_time:0\nlavfi.signalstats.YMIN=235\n", 1),  # wrong numbering
        ("garbage\n", 1),
    ],
)
def test_first_ink_rejects_unreadable_output(text: str, frames: int) -> None:
    with pytest.raises(RenderError, match="unreadable"):
        _first_ink(text, frames)

def test_first_ink_reports_the_first_dark_frame() -> None:
    rec = "frame:{0} pts:{0} pts_time:0\nlavfi.signalstats.YMIN={1}\n"
    text = rec.format(0, 235) + rec.format(1, 194) + rec.format(2, 16)
    assert _first_ink(text, 3) == (1, 194)
    assert _first_ink(rec.format(0, 195), 1) is None              # 195 is not ink

def test_check_ink_passes_a_rendered_part_and_flags_a_planted_mark(
    channel, library, tmp_path, render_tools
) -> None:
    part = _sketch_part(channel, library, tmp_path, render_tools)
    layout = panorama_layout(Fraction(1, 6))
    _check_ink(render_tools, part.video_path, layout, frames=part.report.frames)  # no raise
    marked = _marked(render_tools, part.video_path, tmp_path)
    with pytest.raises(RenderError, match=r"at frame \d+ \(luma \d+ < 195\)"):
        _check_ink(render_tools, marked, layout, frames=part.report.frames)

def test_ink_guard_that_cannot_run_is_an_error(render_tools, tmp_path) -> None:
    broken = dataclasses.replace(render_tools, ffmpeg=tmp_path / "no-ffmpeg")
    src = tmp_path / "x.mp4"
    src.write_bytes(b"")
    with pytest.raises(RenderError, match="cannot run"):
        _check_ink(broken, src, panorama_layout(Fraction(1, 6)), frames=1)
```

Fill in the fixture parameter types (`ChannelConfig`, `SketchLibrary`, `Path`, `RenderTools`) as the existing tests do. `_marked` wraps `_reencode` with the `MARK` drawbox and the outside-zones assertion.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -o addopts="" -q tests/publish/test_landscape.py -k "ink"`
Expected: collection ERROR, `ImportError: cannot import name '_first_ink'`.

- [ ] **Step 3: Implement `_ink_filter`, `_first_ink` and `_check_ink`** as specified above.

- [ ] **Step 4: Run them to verify they pass.** Expected: 9 passed.

- [ ] **Step 5: Commit** `feat(publish): fail-closed ink guard for the 16:9 re-composition (#193)`

### Task A3: `panorama_export` happy path (compose graph, encode, verify, rename)

**Files:**
- Modify: `src/narratoflow/publish/landscape.py`, including the module docstring
- Test: `tests/publish/test_landscape.py`

**Interfaces:**
- Consumes: `panorama_layout`, `_check_ink` (A1, A2).
- Produces:
  - `_compose_graph(layout: PanoramaLayout) -> str`.
  - `_source_scale(src: Path, width: int | None, height: int | None) -> Fraction`: raises `RenderError(f"{src.name}: size {w}x{h} is not a 9:16 canvas size")`.
  - `_identity(path: Path) -> tuple[int, int, int, int]`: `(st_dev, st_ino, st_size, st_mtime_ns)`; `OSError` becomes `RenderError(f"cannot read {path}: {exc}")`.
  - `panorama_export(tools: RenderTools, src: Path, out: Path, *, frames: int, duration_ms: int) -> Path`. It has the same signature as `pillarbox_export` and no other parameter (no bypass, #194 C3).

**Compose filtergraph (exact shape).** Let S, T and C be the zones' `src_bled`, SD, TD and CD the zones' `dst_bled`, and (W, H) = `layout.size`. Join with no whitespace:

```
[0:v]format=yuv444p,split=3[s][t][c];
[s]crop={S.w}:{S.h}:{S.x}:{S.y}:exact=1,pad={W}:{H}:{SD.x}:{SD.y}:color=white[p0];
[t]crop={T.w}:{T.h}:{T.x}:{T.y}:exact=1[t1];[p0][t1]overlay={TD.x}:{TD.y}:format=yuv444[p1];
[c]crop={C.w}:{C.h}:{C.x}:{C.y}:exact=1[c1];[p1][c1]overlay={CD.x}:{CD.y}:format=yuv444[p2];
[p2]format=yuv420p,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv[v]
```

The `yuv444p` round trip plus `exact=1` makes odd offsets at 1/6 safe. The `setparams` tail matches `encoder._FINAL_VF`, so the H.264 VUI is bt709/tv, as with `X264_ARGS`.

**Encode command:**

```python
["-hide_banner", "-nostdin", "-v", "error", "-xerror", "-y",
 "-i", str(src.absolute()),
 "-filter_complex", _compose_graph(layout),
 "-map", "[v]", "-map", "0:a:0",
 *X264_ARGS, "-c:a", "copy",
 "-movflags", "+faststart",
 str(tmp)]
```

Run it with `run_tool(tools.ffmpeg, args, timeout_s=FINAL_TIMEOUT_S)`.

**Order inside `panorama_export` (temp file and atomic replace):**

1. `out.parent.mkdir(parents=True, exist_ok=True)`; `fd, name = tempfile.mkstemp(dir=out.parent, prefix=f".{out.stem}.", suffix=".mp4")`; `os.close(fd)`. An `OSError` becomes `RenderError(f"cannot write {out}: {exc}")`. This is the pillarbox discipline, unchanged.
2. `try:`
   - `before = _identity(src)`;
   - `found = probe(tools, src)`;
   - `layout = panorama_layout(_source_scale(src, found.width, found.height))`;
   - `if found.frames != frames: raise RenderError(f"{src.name}: frames is {found.frames}, expected {frames}")`;
   - `_check_ink(tools, src, layout, frames=frames)`, the guard, before the encode;
   - run the encode;
   - `verify_mp4(tools, tmp, size=layout.size, frames=frames, duration_ms=duration_ms)`;
   - `if _identity(src) != before: raise RenderError(f"{src.name} changed during the 16:9 export")`. This proves the guard and the encode read the same file;
   - `tmp.chmod(0o644)`; `os.replace(tmp, out)`, with `OSError` becoming `RenderError(f"cannot write {out}: {exc}")`.
3. `except BaseException:` unlink `tmp` and `out` with `contextlib.suppress(OSError)`, then re-raise. Return `out`.

Module docstring (PR A):

```
"""The 16:9 export, made from an exported 9:16 part.

`panorama_export` re-composes the part for the wide frame (#163): the stage, top bar and
captions zones are cropped whole at 1:1 with a bleed and placed side by side in a frame
1920x1080 times the source canvas scale, in one ffmpeg pass with the project's H.264 settings
and the audio copied. An ink guard refuses a source with ink outside those zones, so verse
pixels are always the certified 9:16 burn. `pillarbox_export` (D31) is the station's producer
until #194 switches it.
"""
```

- [ ] **Step 1: Write the failing tests** (AC-193.5, AC-193.6, AC-193.7)

```python
def test_compose_graph_at_scale_1_is_the_design_graph() -> None:
    assert _compose_graph(panorama_layout(Fraction(1))) == (
        "[0:v]format=yuv444p,split=3[s][t][c];"
        "[s]crop=844:866:52:382:exact=1,pad=1920:1080:40:107:color=white[p0];"
        "[t]crop=976:216:52:162:exact=1[t1];[p0][t1]overlay=904:107:format=yuv444[p1];"
        "[c]crop=844:196:52:1252:exact=1[c1];[p1][c1]overlay=970:442:format=yuv444[p2];"
        "[p2]format=yuv420p,setparams=color_primaries=bt709:color_trc=bt709"
        ":colorspace=bt709:range=tv[v]"
    )

def test_panorama_has_landscape_size_and_frames(channel, library, tmp_path, render_tools):
    part = _sketch_part(channel, library, tmp_path, render_tools)
    out = tmp_path / "export" / landscape_name(1)
    got = panorama_export(render_tools, part.video_path, out,
                          frames=part.report.frames, duration_ms=part.report.duration_ms)
    found = probe(render_tools, out)
    assert got == out
    assert (found.width, found.height) == landscape_size(Fraction(1, 6))
    assert found.frames == part.report.frames
    assert sorted(p.name for p in out.parent.iterdir()) == [out.name]

def test_panorama_carries_stage_and_captions_on_paper(channel, library, tmp_path, render_tools):
    part = _sketch_part(channel, library, tmp_path, render_tools)
    out = panorama_export(..., tmp_path / "export" / landscape_name(1), ...)
    k = 9                                              # 300 ms: mid-draw of s1 (0-600 ms)
    src = _frames(render_tools, part.video_path)[k].astype(np.int16)
    dst = _frames(render_tools, out)[k].astype(np.int16)
    layout = panorama_layout(Fraction(1, 6))
    zones = {z.name: z for z in layout.zones}
    for name in ("stage", "captions"):
        z = zones[name]
        assert np.abs(_crop(dst, z.dst) - _crop(src, z.src)).mean() < 4
    assert _crop(src, zones["stage"].src).min() < 128     # the frame really is mid-draw
    outside = np.ones(dst.shape[:2], dtype=bool)
    for z in layout.zones:
        r = z.dst_bled
        outside[r.y:r.bottom, r.x:r.right] = False
    assert dst[outside].mean() > 250

def test_verse_ink_is_carried_whole(channel, library, tmp_path, render_tools):
    part = _verse_part(channel, library, tmp_path, render_tools)
    out = panorama_export(..., tmp_path / "export" / landscape_name(1), ...)
    stage = next(z for z in panorama_layout(Fraction(1, 6)).zones if z.name == "stage")
    src = _frames(render_tools, part.video_path, "gray")
    dst = _frames(render_tools, out, "gray")
    counts = [(_crop_gray(f, stage.src) < 128).sum() for f in src[:30]]   # verse slot
    k = int(np.argmax(counts))                          # the fully lit page
    s_ink = _crop_gray(src[k], stage.src) < 128
    d_ink = _crop_gray(dst[k], stage.dst) < 128
    assert s_ink.sum() > 0
    assert np.abs(np.array(_box(d_ink)) - np.array(_box(s_ink))).max() <= 2
    assert abs(int(d_ink.sum()) - int(s_ink.sum())) <= 0.02 * s_ink.sum()
```

`_box(mask) -> tuple[int, int, int, int]` returns `(x0, y0, x1, y1)` from `np.nonzero`. Both boxes are relative to their own zone origin, so the comparison is "translated" by construction. `...` above stands for the same `render_tools, part.video_path, out, frames=..., duration_ms=...` arguments as in the first test; write them out.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -o addopts="" -q tests/publish/test_landscape.py -k "compose or panorama or verse_ink"`
Expected: `ImportError: cannot import name 'panorama_export'`.

- [ ] **Step 3: Implement `_compose_graph`, `_source_scale`, `_identity` and `panorama_export`** as specified above. `_source_scale` maps `(Canvas(s).width, Canvas(s).height)` to `s` for every `s` in `CANVAS_SCALES`.

- [ ] **Step 4: Run them to verify they pass.** Expected: 4 passed. If `test_verse_ink_is_carried_whole` raises `RenderError ... ink outside` (a guard false-fire), stop. That is a 9:16 zone escape (design §4, out of scope), and the threshold must not be loosened. Record it in the PR body and file an issue.

- [ ] **Step 5: Commit** `feat(publish): panorama_export re-composes a 9:16 part side by side (#193)`

### Task A4: `panorama_export` failure paths

**Files:**
- Test: `tests/publish/test_landscape.py`. `src/narratoflow/publish/landscape.py` changes only if a test exposes a gap.

**Interfaces:**
- Consumes: `panorama_export`, `_check_ink`, `landscape` module (A3).

- [ ] **Step 1: Write the tests** (AC-193.8, AC-193.9, controls C2, C4, C5, Review Focus 1-2)

Each test pre-writes an older target, `out.parent.mkdir(parents=True); out.write_bytes(b"old")`, where `out = tmp_path / "export" / landscape_name(1)`. Each ends with `assert not out.exists() and list(out.parent.iterdir()) == []`, so no target and no temp file are left.

- `test_marked_source_is_refused_and_leaves_nothing`: source = `_marked(...)` (MARK outside every bled zone, asserted); `pytest.raises(RenderError, match=r"frame \d+")`.
- `test_non_canvas_source_is_refused`: source = `_reencode(tools, part.video_path, tmp_path / "src" / "odd.mp4", "scale=200:356")`; `match="not a 9:16 canvas size"`.
- `test_missing_source_is_refused`: `src = tmp_path / "missing.mp4"`; `match="missing.mp4"`.
- `test_wrong_frame_count_is_refused`: `frames=part.report.frames + 5`; `match="frames"`.
- `test_wrong_duration_is_refused_by_verify`: `duration_ms=part.report.duration_ms + 500`; `match="video_ms"`. This proves `verify_mp4` runs before `os.replace` (C5).
- `test_source_without_audio_is_refused`: source = `run_tool(ffmpeg, ["-v", "error", "-y", "-i", src, "-map", "0:v:0", "-c:v", "copy", dst])`. Assert only `RenderError`.
- `test_source_changed_during_export_is_refused`: `monkeypatch.setattr(landscape, "_check_ink", wrapper)`, where `wrapper` calls the real `_check_ink` and then `os.utime(src, ns=(1, 1))`; `match="changed during"`. Copy the part to `tmp_path / "src" / "a.mp4"` first, so the render output is not touched.

- [ ] **Step 2: Run them.** Command: `uv run pytest -o addopts="" -q tests/publish/test_landscape.py`. Expected: all pass. Any test that passes before the matching code exists is acceptable here only because Task A3 already wrote that code. To prove each test bites, temporarily comment out its check in `panorama_export` (the frames comparison, the `_check_ink` call, the identity check) and watch the test fail. Then restore the check. Do not commit the mutation.

- [ ] **Step 3: Commit** `test(publish): panorama_export fails closed and leaves no file (#193)`

### Task A5: Guard cost at full scale, gate, PR

No committed code. The measurement script is throwaway and lives outside the repo.

- [ ] **Step 1: Prepare the 180 s benchmark part** (about 85 s of render):

```bash
export NARRATOFLOW_FFMPEG=/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg
SCRATCH=$(mktemp -d)
uv run python scripts/bench_render.py --keep "$SCRATCH/bench"
SRC=$(find "$SCRATCH/bench" -name 'part-01.mp4' -not -path '*/.*' | head -1); echo "$SRC"
```

- [ ] **Step 2: Write `$SCRATCH/guard_cost.py`.** It is throwaway and is never committed:

```python
import sys, time
from fractions import Fraction
from pathlib import Path
from narratoflow.publish import landscape
from narratoflow.render.ffmpeg import find_render_tools, probe
from narratoflow.settings import Settings

src = Path(sys.argv[1]); tools = find_render_tools(Settings())
p = probe(tools, src); layout = landscape.panorama_layout(Fraction(1))
t0 = time.perf_counter(); landscape._check_ink(tools, src, layout, frames=p.frames)
guard = time.perf_counter() - t0
t0 = time.perf_counter()
landscape.panorama_export(tools, src, src.with_name("part-01.16x9.mp4"),
                          frames=p.frames, duration_ms=p.video_ms)
total = time.perf_counter() - t0
print(f"| signalstats | {p.frames} | {guard:.1f} | {guard * 1000 / p.frames:.1f} "
      f"| {total:.1f} | {'pass' if total < 600 else 'fail'} |")
```

Run: `uv run python "$SCRATCH/guard_cost.py" "$SRC"`. `p.video_ms` comes from the probe and is within `verify_mp4`'s ±50 ms tolerance by construction. If the export raises `RenderError` on `video_ms`, read `duration_ms` from the render report next to `$SRC` instead.

- [ ] **Step 3: Decide.** If the signalstats guard is under 600 s in total and every Task A1-A4 test passes, keep signalstats (R10). Otherwise switch `_ink_filter` to the design's bbox recipe (`extractplanes=y,drawbox x3,negate,bbox=min_val=60,metadata=mode=print:key=lavfi.bbox.w:file=ink.txt`, where an empty file means clean), re-run A2-A4 and measure again. Record both rows.

- [ ] **Step 4: Run the local gate** (Proof commands). Expected: exit 0, coverage TOTAL >= 97 %. Also run the certification-path diff command. Expected: no output.

- [ ] **Step 5: Open PR A** with `gh pr create` (the sandbox must be disabled for gh). Title: `16:9 panorama producer with the ink guard (#193)`. The body holds: `Closes #193`; the criterion → test table below; the controls table with test ids; a "Guard cost (180 s benchmark part, full scale)" table with header `| recipe | frames | guard s | ms/frame | export total s | NFR (< 600 s) |`, the row(s) from Step 2, the CPU, and the ffmpeg version from `ffmpeg -version | head -1`; the gate result line; and the attribution line.

### PR A: criterion → test id

| # | Acceptance criterion (#193, verbatim) | Task | Test id |
|---|---|---|---|
| AC-193.1 | `panorama_layout`: at scale 1, the unbled destination rectangles equal the table above, and each zone's size equals `render.layout.STAGE`, `TOP_BAR` and `CAPTIONS`. | A1 | `tests/publish/test_landscape.py::test_layout_scale_1_matches_the_design_table` |
| AC-193.2 | `panorama_layout`: at each of the four canvas scales, the bled source rectangles are pairwise disjoint and inside the source frame. The bled destination rectangles are pairwise disjoint and inside `landscape_size(scale)`. | A1 | `tests/publish/test_landscape.py::test_layout_bled_rects_are_disjoint_and_inside` |
| AC-193.3 | `panorama_layout`: at scale 1, no destination zone intersects `LANDSCAPE_BOTTOM`, and every zone is at least 48 px from the left and right edges. | A1 | `tests/publish/test_landscape.py::test_layout_keeps_the_bottom_and_side_margins_clear` |
| AC-193.4 | `landscape_size`: scale 1 gives (1920, 1080) and scale 1/6 gives (320, 180). | A1 | `tests/publish/test_landscape.py::test_landscape_size_follows_the_canvas_scale` |
| AC-193.5 | `panorama_export`: for a rendered part with a sketch scene and captions, the output has the `landscape_size` of the source's scale and the source's frame count, and no other file is left in the target directory. | A3 | `tests/publish/test_landscape.py::test_panorama_has_landscape_size_and_frames` |
| AC-193.6 | `panorama_export`: on a mid-draw frame, the output's stage rectangle and captions rectangle each differ from the source's same zone by a mean absolute difference under 4 (0-255 scale). The area outside the bled destination zones has a mean above 250. | A3 | `tests/publish/test_landscape.py::test_panorama_carries_stage_and_captions_on_paper` |
| AC-193.7 | `panorama_export`: for a rendered part with a verse scene, on a frame where the page is fully lit, the ink bounding box inside the output's stage equals the source's (translated) within 2 px, and the ink pixel count is within 2 %. | A3 | `tests/publish/test_landscape.py::test_verse_ink_is_carried_whole` |
| AC-193.8 | `panorama_export`: a source with a dark mark outside every bled zone raises `RenderError` naming a frame. It creates no target file, removes an older target file, and leaves no temp file. | A4 | `tests/publish/test_landscape.py::test_marked_source_is_refused_and_leaves_nothing` |
| AC-193.9 | `panorama_export`: a source whose size is not a canvas size raises `RenderError` and leaves no target file. So does a missing source, and so does a wrong `frames` value. | A4 | `tests/publish/test_landscape.py::test_non_canvas_source_is_refused`, `::test_missing_source_is_refused`, `::test_wrong_frame_count_is_refused` |
| AC-193.10 | The guard's cost is measured on the 180 s benchmark part at full scale and recorded in the PR body. The guard uses the cheapest recipe that meets the criteria above: `signalstats` (about 10 ms per frame, estimated) is preferred over the 24 ms per frame prototype. | A5 | PR body table (measurement, Task A5 Steps 1-3); recipe pinned by `::test_check_ink_passes_a_rendered_part_and_flags_a_planted_mark` |
| AC-193.11 | The local gate in `.superpowers/orchestrate/project.md` exits 0 with `NARRATOFLOW_FFMPEG=/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg`. | A5 | local gate command (Proof commands) |

### PR A: controls → test id (`issue-193/controls.md`)

| C# | Control | Proof (test id) |
|---|---|---|
| C1 | Zones carried whole and unscaled; sizes are the 9:16 zones | `::test_layout_scale_1_matches_the_design_table`, `::test_layout_bled_rects_are_disjoint_and_inside`, `::test_compose_graph_at_scale_1_is_the_design_graph` |
| C2 | No ink (luma < 195) outside the bled zones, every frame | `::test_check_ink_passes_a_rendered_part_and_flags_a_planted_mark`, `::test_first_ink_reports_the_first_dark_frame`, `::test_marked_source_is_refused_and_leaves_nothing` |
| C3 | Guard cannot run or parse → error | `::test_ink_guard_that_cannot_run_is_an_error`, `::test_first_ink_rejects_unreadable_output` (6 cases) |
| C4 | Source is a canvas-sized 9:16 file | `::test_non_canvas_source_is_refused` |
| C5 | Output size, frames, duration checked before the rename | `::test_wrong_duration_is_refused_by_verify`, `::test_panorama_has_landscape_size_and_frames` |
| C6 | Certification path unchanged (review control) | `git diff --name-only origin/main...HEAD -- src/narratoflow/render src/narratoflow/captions src/narratoflow/publish/verse.py` prints nothing (Task A5 Step 4) |

(Every test id above lives in `tests/publish/test_landscape.py`.)

---

## PR B — #194: station exports the panorama; pillarbox removed; docs

Branch: `feat/194-station-panorama` from `origin/main` **after PR A is merged**.

### File structure (PR B)

- `src/narratoflow/ui/station.py`: line 59 import, the call at :766 and the `export` docstring (:718-720).
- `src/narratoflow/ui/app.py`: :35 constant, :245 handler binding.
- `src/narratoflow/publish/landscape.py`: delete `pillarbox_export` and `_PILLARBOX_VF`, and the last sentence of the module docstring.
- `tests/ui/test_station.py`, `tests/ui/test_app.py`, `tests/publish/test_landscape.py`.
- Docs: create `docs/adr/0006-landscape-recomposition.md`, and modify `docs/adr/0002-compositor-ffmpeg-frame-pipe.md:28-29`, `docs/PRD.md` (:29, :43, :85, :123, :155), `docs/SRD.md` (:142, :155) and `README.md:356`.

### Task B1: The station exports with `panorama_export`

**Files:**
- Modify: `src/narratoflow/ui/station.py:59`, `:718-720`, `:766`
- Test: `tests/ui/test_station.py`

**Interfaces:**
- Consumes: `panorama_export(tools, src, out, *, frames, duration_ms) -> Path` and `landscape_size(scale) -> tuple[int, int]` from #193.

- [ ] **Step 1: Write the failing tests**

- Edit `test_export_portrait_and_landscape` (AC-194.1). Replace `assert (found.width, found.height) == (1920, 1080)` with:

```python
    assert (found.width, found.height) == landscape_size(TINY.scale)
    raw = run_tool(kit["tools"].ffmpeg, ["-v", "error", "-i", str(landscape[0]),
                   "-vf", "select=eq(n\\,9),format=gray", "-frames:v", "1",
                   "-f", "rawvideo", "-"])   # frame 9 = 300 ms, mid-draw of s1
    w, h = landscape_size(TINY.scale)
    left = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)[:, : w // 4]
    assert left.min() < 128   # the stage is there, not a bar of paper
```

- New `test_landscape_export_leaves_the_portrait_file_and_render_report_unchanged` (AC-194.2, C4): `export(paths, AspectRatio.PORTRAIT, **kit)`, then `before = {p: sha256(p) for p in (paths.export_dir / "part-01.mp4", paths.render_dir / "part-01.render.json")}`, then `export(paths, AspectRatio.LANDSCAPE, **kit)`; assert the same hashes after (`hashlib.sha256(p.read_bytes()).hexdigest()`).
- New fixture `verse_publishable`: a copy of `publishable` whose series is `make_series([verse("v1"), narr("s1", "alpha beta gamma", sketch=sid, title="One")])`.
- New `test_landscape_export_of_an_altered_verse_part_writes_nothing` (AC-194.4, C1):
  - Call `export(paths, AspectRatio.PORTRAIT, **kit)` to render; the result is not asserted.
  - Let `ass = paths.render_dir / "part-01.ass"` and `word = display_words(CORPUS.ayah(31, 33).uthmani)[1]`; assert `word in ass.read_text(encoding="utf-8")`.
  - Write the text back with the first `word` replaced by `"الناس"`.
  - Pre-write `(paths.export_dir / landscape_name(1)).write_bytes(b"old")`, with `mkdir(parents=True, exist_ok=True)` first.
  - Assert `export(paths, AspectRatio.LANDSCAPE, **kit) == []` and `not (paths.export_dir / landscape_name(1)).exists()`.
  - Load `report = PublishReport.load(paths.render_dir / "part-01.publish.json")` and assert `any(c.check is CheckId.VERSE_TEXT and c.severity is Severity.FAIL for c in report.checks)`.
  - Imports: `display_words` from `narratoflow.audio.recitation`; `PublishReport`; `CheckId` and `Severity` from `narratoflow.publish.checks`; `verse` and `CORPUS` from `fixtures.parts`.
- New `test_panorama_export_has_no_bypass_argument` (C3): `assert list(inspect.signature(panorama_export).parameters) == ["tools", "src", "out", "frames", "duration_ms"]`.
- Keep unchanged and green: `test_export_of_a_failing_part_produces_neither_file` (AC-194.3), `test_portrait_export_removes_the_now_stale_landscape_copy` and `test_portrait_export_of_a_failing_part_removes_the_landscape_copy` (AC-194.5, C2), and `test_portrait_export_cannot_remove_a_landscape_copy_is_a_plain_error`.

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -o addopts="" -q tests/ui/test_station.py -k "export"`
Expected: `test_export_portrait_and_landscape` fails because the pillarbox is 1920x1080, not (320, 180). The new tests pass or fail according to the current pillarbox. The AC-194.4 test already passes, because a FAIL already blocks; it guards against regression.

- [ ] **Step 3: Implement.**
  - In `station.py`, import `landscape_name, panorama_export` and replace the `pillarbox_export(` call with `panorama_export(` (same arguments).
  - Docstring lines 718-720: `"""Render every part (unchanged parts are kept), check and export them, and return the exported files of `ratio`: `part-NN.mp4`, or for 16:9 the `part-NN.16x9.mp4` side-by-side re-compositions (ADR 0006). A part that FAILs its check is exported in neither ratio."""`.

- [ ] **Step 4: Run them to verify they pass.** Same command; expected all pass.

- [ ] **Step 5: Commit** `feat(ui): Export 16:9 writes the side-by-side re-composition (#194)`

### Task B2: `LANDSCAPE_NOTE`

**Files:**
- Modify: `src/narratoflow/ui/app.py:35`, `:245`
- Test: `tests/ui/test_app.py:250`

- [ ] **Step 1: Edit the test** `test_export_buttons_call_both_ratios` (AC-194.6). Change the last line to:

```python
    assert files == [] and "nothing" in status and app_module.LANDSCAPE_NOTE in status
    assert [n for n in dir(app_module) if n.endswith("_NOTE")] == ["LANDSCAPE_NOTE"]
```

The second line proves the old note is gone without spelling its name, which would break the AC-194.7 grep.

- [ ] **Step 2: Run** `uv run pytest -o addopts="" -q tests/ui/test_app.py -k export_buttons`. Expected: FAIL, `AttributeError: module ... has no attribute 'LANDSCAPE_NOTE'`.
- [ ] **Step 3: Implement.** Set `LANDSCAPE_NOTE = "16:9 is the checked 9:16 video re-arranged side by side"` in place of the old constant, and use `bind(_on_export, AspectRatio.LANDSCAPE, LANDSCAPE_NOTE)` at :245.
- [ ] **Step 4: Run the test again.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(ui): the 16:9 status names the re-composition (#194)`

### Task B3: Delete the pillarbox producer and its tests

**Files:**
- Modify: `src/narratoflow/publish/landscape.py`, `tests/publish/test_landscape.py`

- [ ] **Step 1: Delete** `test_pillarbox_is_1920x1080_same_frames`, `test_pillarbox_failure_leaves_no_file` and `test_pillarbox_wrong_frame_count_is_refused_and_removed`, and the `pillarbox_export` import. Set the test module docstring to `"""panorama_export: the checked 9:16 part re-composed side by side, guarded and verified."""`. `test_landscape_name_is_zero_padded` stays.
- [ ] **Step 2: Delete** `pillarbox_export` and `_PILLARBOX_VF`, and drop the last sentence of the module docstring (the one naming D31's producer). The final docstring is the PR A docstring without that sentence.
- [ ] **Step 3: Run** `grep -ri pillarbox src/ tests/`. Expected: no output. Then run `uv run pytest -o addopts="" -q tests/publish/test_landscape.py tests/ui`. Expected: all pass.
- [ ] **Step 4: Commit** `refactor(publish): remove the pillarbox 16:9 producer (#194)`

### Task B4: ADR 0006 and the doc amendments (design §3)

**Files:**
- Create: `docs/adr/0006-landscape-recomposition.md`
- Modify: `docs/adr/0002-compositor-ffmpeg-frame-pipe.md:28-29`, `docs/PRD.md:29,43,85,123,155`, `docs/SRD.md:142,155`, `README.md:356`

Write `docs/adr/0006-landscape-recomposition.md` in the house style of `docs/adr/0005-diacritizer-benchmark.md`:

```
# ADR 0006: The 16:9 export is a re-composition of the checked 9:16 video

**Status:** Accepted, 2026-11-15
```

Its sections:

1. **Context.** D31 made the 16:9 file a pillarbox. #163 asks for a layout composed for the wide frame, with only the producer behind `publish/landscape.py` changing. REVIEW 2.4 asks for one pipeline with two layouts.
2. **Decision.**
   - The stage, top bar and captions zones of the 9:16 frame are carried whole and unscaled into 1920x1080. Include D1's table: stage (60, 390, 828, 850) → (48, 115, 828, 850); top bar (60, 170, 960, 200) → (912, 115, 960, 200); captions (60, 1260, 828, 180) → (978, 450, 828, 180).
   - Add the ASCII sketch from design D1.
   - An 8 px bleed (`max(1, round(8 * scale))`) travels with each zone.
   - `LANDSCAPE_BOTTOM` (0, 972, 1920, 108) stays clear for the player controls, with a 48 px edge margin.
   - The output size is 1920x1080 times the source canvas scale.
   - One ffmpeg pass in `publish/landscape.py`, with `X264_ARGS`, audio copied, and `verify_mp4` before the rename.
3. **Verse integrity.**
   - There is no second subtitle document and no second `shown_sha256`; the verse pixels are the certified 9:16 burn.
   - The ink guard masks the three bled source zones on every frame and fails when any luma below 195 (tv range; paper 235) remains. It also fails when it cannot run or its output cannot be parsed. On failure it writes nothing and removes an older file. State the recipe used (signalstats `YMIN`, or bbox if Task A5 switched) and the measured cost from PR A's body.
4. **Consequences.**
   - 9:16 output, `inputs_key` and `RENDER_INPUTS_VERSION` are unchanged.
   - `LANDSCAPE_WARNING` keeps its text ("16:9 is not rendered in v1") because it is part of `inputs_key`, and it stays true: render does not draw 16:9.
   - The 16:9 file is a second-generation encode (51 dB PSNR measured on the carried zones).
   - It is a derivative, not in the publish report, and is rebuilt on every click.
   - The layout follows any future change to `STAGE`, `TOP_BAR` or `CAPTIONS`, and the geometry test fails if they stop fitting.
   - No migration: an old pillarbox file is overwritten by the next Export 16:9 and deleted by the next Export 9:16.
5. **Limits.**
   - Text is not re-typeset for the wide frame; caption lines keep the 9:16 length.
   - A verse scene leaves the right column almost empty.
   - The guard does not see ink paler than luma 195 (`ChannelStyle.ink` defaults to `#111111`).
6. **Alternatives considered.**
   - A second render with a landscape `Canvas` and its own ASS document and verse certificate (design §6; the path to take if re-typesetting is wanted).
   - Rendering both ratios in `build.py`.
   - Keeping the previous centred-on-bars copy as a fallback. Rejected: a guard failure must be an error. Do not write the word that the AC-194.7 grep looks for; the grep does not cover `docs/adr/`, but keep the ADR clean anyway.

Amendments (exact replacement sentences):

| File:line | Replace | With |
|---|---|---|
| `docs/adr/0002-compositor-ffmpeg-frame-pipe.md:28-29` | the bullet `- Output is 9:16 only. A channel that lists … (`render/build.py`).` | the same two lines, followed by the sentence `The 16:9 export is re-composed from the 9:16 video by the review station (ADR 0006).` (wrapped at 100 columns, two-space continuation indent). No other line changes. |
| `docs/PRD.md:29` | `a 16:9 file is a pillarboxed copy of the 9:16 video (6.5).` | `a 16:9 file is the 9:16 video re-arranged side by side (sketch left, text right) (6.5).` |
| `docs/PRD.md:43` | `9:16 native; 16:9 is pillarboxed` | `9:16 native; 16:9 re-composed from it` |
| `docs/PRD.md:85` | the whole bullet | `- **No separately rendered 16:9 in v1.** The 16:9 file is re-composed from the checked 9:16 video (ADR 0006).` |
| `docs/PRD.md:123` | `- 16:9 is a pillarboxed copy of the 9:16 export, made from the review station (section 7).` | `- A 16:9 file is the 9:16 video re-arranged side by side (sketch left, text right), made from the review station (section 7).` |
| `docs/PRD.md:155` | `Export 16:9 writes the pillarboxed copy.` | `Export 16:9 writes the 9:16 video re-arranged side by side (ADR 0006).` |
| `docs/SRD.md:142` | `The 16:9 file is a pillarboxed copy made by the review station (4.6); a native layout is #163.` | `The 16:9 file is re-composed from the 9:16 export by the review station (4.6, ADR 0006).` |
| `docs/SRD.md:155` | `Export 16:9 writes … renamed into place.` (the `pillarbox_export` sentence) | `Export 16:9 writes part-NN.16x9.mp4 with panorama_export (publish/landscape.py): the stage, top bar and captions zones of the 9:16 video are carried whole at 1:1 into a 1920x1080 frame (sketch left, text right), after an ink guard has refused any ink outside those zones on every frame; it is re-encoded with the project's x264 arguments, audio copied, and verified before it is renamed into place.` (keep the backticks on code names as in the line today). Keep the following sentence `It is a derivative file, listed in the station's result and not recorded in the publish report.` unchanged. |
| `README.md:356` | `a pillarboxed copy of the 9:16 video (the compositor draws 9:16 only); a native 16:9 layout is follow-up #163.` | `the 9:16 video re-arranged side by side (sketch left, text right), made from the checked 9:16 file (ADR 0006).` |

Not edited: `docs/superpowers/plans/*` and `docs/REVIEW.md`.

- [ ] **Step 1: Write the check that fails now:** `grep -ri pillarbox src/ tests/ README.md docs/PRD.md docs/SRD.md; test -f docs/adr/0006-landscape-recomposition.md`. Expected before the edits: matches in README, PRD and SRD, and `test -f` exits 1.
- [ ] **Step 2: Make the edits above.**
- [ ] **Step 3: Run the same command.** Expected: no grep output and `test -f` exits 0 (AC-194.7).
- [ ] **Step 4: Commit** `docs: ADR 0006 and the 16:9 re-composition in PRD, SRD, README, ADR 0002 (#194)`

### Task B5: Gate, render-diff check, PR

- [ ] **Step 1:** Run `git diff --name-only origin/main...HEAD -- src/narratoflow/render/`. Expected: no output (AC-194.8). Also run `git diff --name-only origin/main...HEAD -- src/narratoflow/captions src/narratoflow/publish/verse.py`. Expected: no output.
- [ ] **Step 2:** Run the local gate with ffmpeg-full. Expected: exit 0, TOTAL >= 97 % (AC-194.9).
- [ ] **Step 3:** Open PR B with `gh pr create` (sandbox disabled). Title: `Station exports the 16:9 panorama; pillarbox removed; ADR 0006 (#194)`. The body holds `Closes #194` and `Closes #163` (the last slice), the criterion → test table, the controls table, the rollback line ("revert the PR; the next export rewrites the file as before"), and the attribution line.

### PR B: criterion → test id

| # | Acceptance criterion (#194, verbatim) | Task | Test id / proof |
|---|---|---|---|
| AC-194.1 | `station.export`: a 16:9 export returns `[export_dir / "part-01.16x9.mp4"]`, whose size is `landscape_size(TINY.scale)` and whose left quarter is not uniform paper on a mid-draw frame. | B1 | `tests/ui/test_station.py::test_export_portrait_and_landscape` |
| AC-194.2 | `station.export`: the sha256 of `export_dir/part-01.mp4` and of `render_dir/part-01.render.json` is the same before and after a 16:9 export that follows a 9:16 export. | B1 | `tests/ui/test_station.py::test_landscape_export_leaves_the_portrait_file_and_render_report_unchanged` |
| AC-194.3 | `station.export`: a part that FAILs its publish check yields no 16:9 file, and an older one is removed (the existing test, kept green). | B1 | `tests/ui/test_station.py::test_export_of_a_failing_part_produces_neither_file` |
| AC-194.4 | `station.export`: for a part with a verse scene whose shipped `.ass` was altered so that `verse_text` FAILs, a 16:9 export returns `[]` and no `part-NN.16x9.mp4` exists. | B1 | `tests/ui/test_station.py::test_landscape_export_of_an_altered_verse_part_writes_nothing` |
| AC-194.5 | `station.export`: a 9:16 export removes an existing 16:9 file (the existing tests, kept green). | B1 | `tests/ui/test_station.py::test_portrait_export_removes_the_now_stale_landscape_copy`, `::test_portrait_export_of_a_failing_part_removes_the_landscape_copy` |
| AC-194.6 | `ui.app`: clicking "Export 16:9" puts `LANDSCAPE_NOTE` in the status, and the module no longer has `PILLARBOX_NOTE`. | B2 | `tests/ui/test_app.py::test_export_buttons_call_both_ratios` |
| AC-194.7 | `grep -ri pillarbox src/ tests/ README.md docs/PRD.md docs/SRD.md` finds nothing, and `docs/adr/0006-landscape-recomposition.md` exists. | B3, B4 | the grep + `test -f` command (B4 Step 3) |
| AC-194.8 | `git diff origin/main...HEAD` shows no change under `src/narratoflow/render/`, so `RENDER_INPUTS_VERSION == "2"` and `LANDSCAPE_WARNING` are unchanged. | B5 | the diff command (B5 Step 1); `tests/render/test_build.py` stays green in the gate |
| AC-194.9 | The local gate in `.superpowers/orchestrate/project.md` exits 0 with ffmpeg-full. | B5 | local gate command |

### PR B: controls → test id (`issue-194/controls.md`)

| C# | Control | Proof (test id) |
|---|---|---|
| C1 | 16:9 made only from a part exported by this call's publish check | `tests/ui/test_station.py::test_export_of_a_failing_part_produces_neither_file`, `::test_landscape_export_of_an_altered_verse_part_writes_nothing` |
| C2 | A stale 16:9 file never sits beside a newer 9:16 file | `tests/ui/test_station.py::test_portrait_export_removes_the_now_stale_landscape_copy`, `::test_portrait_export_of_a_failing_part_removes_the_landscape_copy` |
| C3 | #193's guard and verify run on every 16:9 export (no bypass argument) | `tests/ui/test_station.py::test_panorama_export_has_no_bypass_argument`; guard and verify proven in #193 (`tests/publish/test_landscape.py::test_marked_source_is_refused_and_leaves_nothing`, `::test_wrong_duration_is_refused_by_verify`) |
| C4 | 9:16 output and render cache untouched | `tests/ui/test_station.py::test_landscape_export_leaves_the_portrait_file_and_render_report_unchanged` + the B5 Step 1 diff command |

---

## Self-review

- **Coverage.** All 11 #193 criteria and all 9 #194 criteria map to a task and a test id or command (tables above). All 6 #193 controls and all 4 #194 controls map to a test id or command. D9's items are covered: `PublishReport.outputs` is untouched (no task edits `publish/report.py`) and `LANDSCAPE_WARNING` is untouched (AC-194.8).
- **Placeholders.** None. Every constant, message fragment matched by a test, filtergraph, command and replacement sentence is written out. `...` appears only in Task A3's test sketch, where the sentence after it says what it stands for. The ADR body is given as required facts per section, from design §3.
- **Types.** `panorama_layout(scale: Fraction) -> PanoramaLayout`, `Zone.src/dst/src_bled/dst_bled`, `landscape_size(scale) -> tuple[int, int]`, `_check_ink(tools, src, layout, *, frames)`, `_first_ink(text, frames) -> tuple[int, int] | None` and `panorama_export(tools, src, out, *, frames, duration_ms) -> Path` are used with the same names and types in A2-A5 and B1.
- **Forbidden paths.** No task edits anything under `src/narratoflow/render/`, `src/narratoflow/captions/` or `src/narratoflow/publish/verse.py`. They are only imported from, and A5 Step 4 and B5 Step 1 check this with `git diff`. `scripts/` is not edited; the guard-cost script lives in a `mktemp -d` directory.
- **Proportion.** Code blocks are limited to the tests' assertions, the exact filtergraphs and commands, and the throwaway measurement script. Function bodies are left to the implementer, except the order of steps inside `panorama_export`, which the controls depend on.

## Planner rulings

1. Destination rectangle at scale f = `(round(D.x * f), round(D.y * f))` with the scaled source zone's own w and h, not `D.scaled(f)` — `Rect.scaled` rounds the two edges separately, so at 1/6 the top bar would come out 33 px high at the destination against 34 px at the source, breaking the 1:1 crop; computed for all four scales while planning (bled rects disjoint and inside at each) — cost if wrong: a 1 px shift of a zone at test scales only (scale 1 is exact) — rejected: `D.scaled(f)` (size drift); reversal: low.
2. Guard recipe: three `drawbox ... color=white:t=fill` on the decoded `yuv420p` frame, then `signalstats` with `metadata=print` of `YMIN` to a relative `ink.txt`; no `format` or `extractplanes` step — R10 prefers signalstats, and drawbox was checked to mask luma exactly at odd offsets on ffmpeg-full; the bbox recipe is the fallback only if Task A5's measurement or a criterion fails — cost if wrong: the guard is slower than the alternative or misses a case; A5 measures it and the tests pin it — rejected: the 24 ms/frame bbox prototype as default; reversal: low.
3. AC-193.3 is checked on the unbled destination zones — at scale 1 the bled stage reaches y = 973 and touches `LANDSCAPE_BOTTOM` (y 972) by 1 px of paper-coloured bleed; the criterion says "destination zone", and the design's own 10 % band refers to the content zones — cost if wrong: one px of bleed sits in the controls band (it is paper in practice) — rejected: shrinking the bleed or moving the stage (contradicts R5's rectangles); reversal: low.
4. The guard is a separate ffmpeg pass before the encode, not a `split` in one invocation. The "same file" property is enforced by a `(st_dev, st_ino, st_size, st_mtime_ns)` identity of `src`, taken before the probe and compared after `verify_mp4`, before `os.replace` — it matches the controls' wording "called by panorama_export before the encode", and no output file is ever started for a guarded-out source — cost if wrong: one extra decode of the source per export (measured in A5) — rejected: a single split invocation (the encode starts before the verdict and needs a cleanup race), and hashing the source (slower, a second full read); reversal: low.
5. The guard's error names the first ink frame and its luma, not a bounding box — signalstats gives no box; #193 AC-193.8 requires only "naming a frame" (design D3 mentioned a box) — cost if wrong: a creator gets less help finding the escape; a bbox could be added with the bbox recipe — rejected: a second pass just to compute a box; reversal: low.
6. The guard's working directory is a system-temp `TemporaryDirectory`, and the metadata file name is relative (`cwd=`) — no path is escaped inside a filtergraph, as in `encoder._FINAL_VF`, and nothing extra ever lands in the target directory (AC-193.5) — cost if wrong: none known — rejected: a file beside `out` (would appear in the target directory on a crash); reversal: low.
7. `panorama_export` checks the source's probed frame count against `frames` before the guard (`RenderError` "frames is X, expected Y") — fails fast on AC-193.9's wrong `frames` without running the guard and encode; `verify_mp4` still checks the output — cost if wrong: one `-count_frames` probe of the source per export (included in A5's measurement) — rejected: relying on `verify_mp4` after the encode (wastes a full encode); reversal: low.
8. `CANVAS_SCALES` is declared in `publish/landscape.py` — `render.layout._ALLOWED_SCALES` is private and `render/` is off limits; each value is passed through `Canvas(...)`, which still validates — cost if wrong: two lists to keep in sync if a scale is added (the parametrized test fails if `Canvas` rejects one) — rejected: importing the private name; reversal: low.
9. The output file mode stays `0o644`, as in `pillarbox_export` — keeps the existing producer's discipline, which the design says to keep as is — cost if wrong: differs from `encode_part`'s umask-derived mode — rejected: copying the umask helper; reversal: low.
10. In test helpers, "ink" for AC-193.7 is gray < 128, and "the frame where the page is fully lit" is the frame in the verse slot (frames 0-29) with the most stage ink in the source — the criterion fixes neither; the max-ink frame is fully lit by definition and does not depend on fade timings — cost if wrong: a looser or stricter test than intended — rejected: a hard-coded frame number; reversal: low.
11. AC-194.4's test asserts a `verse_text` FAIL in the saved `render_dir/part-01.publish.json` and does not require a passing 16:9 export of the unaltered verse part first — the `publishable` fixture has no recitation assets, and a passing verse publish needs the reciter and audio setup of `tests/publish/test_build.py::test_reference_verse_part_checks_reciter`, which is out of this slice's scope; asserting the FAIL in the report keeps the test from passing for an unrelated reason — cost if wrong: the test does not prove the same part would otherwise export — rejected: building the full recitation fixture in `test_station.py`; reversal: low.
12. AC-194.6's "no longer has `PILLARBOX_NOTE`" is asserted as `[n for n in dir(app_module) if n.endswith("_NOTE")] == ["LANDSCAPE_NOTE"]` — spelling the old name in a test would make the AC-194.7 grep over `tests/` fail — cost if wrong: an unrelated new `*_NOTE` constant would need the test updated — rejected: `not hasattr(app_module, "PILLARBOX_NOTE")`; reversal: low.
13. `docs/PRD.md:155` (`Export 16:9 writes the pillarboxed copy.`) is also edited, although design §3 lists only :29, :43, :85 and :123 — AC-194.7's grep over `docs/PRD.md` would otherwise fail — cost if wrong: none (it is the same wording change) — rejected: leaving it; reversal: low.
14. The guard-cost measurement is a throwaway script in a `mktemp -d` directory, built on `scripts/bench_render.py --keep`; no new file in `scripts/` — R10 asks for the number in the PR body, not for a tool; `bench_render.py` already prepares the part — cost if wrong: re-measuring later needs the script from the PR body — rejected: a committed `scripts/bench_landscape.py` (adds coverage and maintenance for a one-off number); reversal: low.
15. PR A keeps the pillarbox tests and only adds to `tests/publish/test_landscape.py`; PR B deletes them — the station still calls `pillarbox_export` in PR A (#193 "stays in place") — cost if wrong: none — rejected: deleting them in PR A; reversal: low.

## Cold-read decisions (2026-11-15, binding on both PRs)

These override any contrary text above.

- **CR1 (overrides the bbox fallback in A5 Step 3).** There is no bbox fallback. The guard recipe is the `drawbox` masks plus `signalstats` YMIN recipe of A2, with its strict parse: frame numbers exactly `0..frames-1`, and an empty or short file is unreadable, so it fails closed. If that recipe cannot meet an AC-193 criterion, stop. Return `stalled: guard recipe — <criterion, evidence>`, do not switch recipes, and open no PR. A slow guard is not a failure: record its cost in the PR body (R10). The 600 s figure in A5 is the render NFR (ADR 0002). Report it beside the export total for context only; it is not a gate.
- **CR2 (overrides A3 Step 4).** If `test_verse_ink_is_carried_whole` or any real-fixture test shows ink outside the bled zones, do not mark the test xfail or skip, and do not loosen the threshold. Commit nothing that leaves the gate red. Return `blocked: 9:16 zone escape — <test, frame, luma, region>` with the evidence. The conductor files the issue and decides. PR B is blocked until that is settled.
- **CR3.** The AC-194.7 grep is run as `grep -ri --exclude-dir=__pycache__ pillarbox src/ tests/ README.md docs/PRD.md docs/SRD.md`, so stale `.pyc` files are not counted.
- **CR4.** The AC-194.4 test also asserts that the `verse_text` FAIL's reason is the subtitle mismatch (`publish/verse.py`: "subtitles in ... differ"), so an unrelated FAIL cannot satisfy it.
- **CR5.** Coverage: `coverage report` does not enforce 97 % (there is no `fail_under`). Each builder reads TOTAL from the output and reports it; below 97 % is a failed gate.
- **CR6.** The `BuiltPart` annotation is `PartVideo` (`render/build.py`), not `RenderResult`.
- **CR7.** Whether `_first_ink` reads to the first ink or the whole file is your choice. Both fail closed.
- **CR8.** The ADR 0002 sentence is a new continuation line at the end of the bullet on :28-29, with no re-wrap of other lines. SRD:155 keeps backticks on the code names it already has, plus `panorama_export`.
- **CR9.** In PR B, delete `_row` in `tests/publish/test_landscape.py` with the pillarbox tests if nothing else uses it. Add the imports the edited station test needs.
