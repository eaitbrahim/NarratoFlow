"""Render build: every selected part's timeline and FLAC to ``part-NN.mp4`` and its report.

Everything that can be checked is checked before any frame is drawn, and those errors
are raised as they are: the series, the channel, the tools, the part indexes, the
bundled fonts and hand, and per part its timeline (`check_timeline`), its sketches,
its verses and its subtitle document. Then, per part, in a fresh work directory
(always deleted): one FFV1 segment per scene (inline or in a spawn process pool), the
single libass/H.264 pass, `verify_mp4`, the ``part-NN.ass`` sidecar and
``part-NN.render.json``. A failure leaves the parts already finished on disk, and never
an unverified MP4 at a part's final path. Render reads audio and spends nothing, so the
series is never modified.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Literal

import cv2

from narratoflow.audio.timeline import PartTimeline, SceneTiming
from narratoflow.captions.ass import AssDocument, style_font_id
from narratoflow.captions.part import build_part_ass
from narratoflow.captions.verse import VERSE_FONT, check_verse_text, shown_text, verse_pages
from narratoflow.channels.config import ChannelConfig
from narratoflow.errors import LibraryError, PlanningError, RenderError, StaleTimelineError
from narratoflow.library.repository import SketchLibrary, default_library
from narratoflow.library.schemas import SketchRecord, SketchStatus
from narratoflow.models import AspectRatio, Language, Part, Scene, SceneKind, Series, VisualSource
from narratoflow.quran.corpus import QuranCorpus, default_corpus
from narratoflow.render import compose
from narratoflow.render.assets import FontPack, HandSprite, default_fonts, default_hand
from narratoflow.render.compose import SceneJob
from narratoflow.render.encoder import X264_ARGS, encode_part, verify_mp4, write_concat_list
from narratoflow.render.ffmpeg import RenderTools, check_arabic_shaping, find_render_tools, run_tool
from narratoflow.render.layout import Canvas, frame_of, frame_range
from narratoflow.render.report import RenderAsset, RenderReport, SceneRender, VerseRender
from narratoflow.render.variation import Variation, choose_variation
from narratoflow.settings import Settings
from narratoflow.sketching.generate import require_series_channel
from narratoflow.visuals.marker import marker_seed, video_id
from narratoflow.visuals.strokes import StrokeSet

logger = logging.getLogger(__name__)

REBUILD_AUDIO = "rebuild audio: narratoflow audio build"
LANDSCAPE_WARNING = "16:9 is not rendered in v1"
AUDIO_TOLERANCE_MS = 30
CRF = int(X264_ARGS[X264_ARGS.index("-crf") + 1])


@dataclass(frozen=True)
class PartVideo:
    index: int
    video_path: Path
    report_path: Path
    report: RenderReport


@dataclass
class RenderResult:
    parts: list[PartVideo] = field(default_factory=list)


def _stale(message: str) -> StaleTimelineError:
    return StaleTimelineError(f"{message}; {REBUILD_AUDIO}")


def _drawn(scene: Scene) -> bool:
    return scene.kind is SceneKind.NARRATION and scene.visual.source is not VisualSource.TEXT_CARD


def _is_card(scene: Scene) -> bool:
    return scene.kind is SceneKind.NARRATION and scene.visual.source is VisualSource.TEXT_CARD


def audio_duration_ms(tools: RenderTools, path: Path) -> int | None:
    """The audio stream's length from its packets (a piped FLAC carries no duration field)."""
    raw = run_tool(
        tools.ffprobe,
        [
            "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=time_base:packet=pts,duration", "-of", "json", str(path),
        ],
    )  # fmt: skip
    try:
        data = json.loads(raw)
        base = Fraction(data["streams"][0]["time_base"])
        end = max(int(p["pts"]) + int(p["duration"]) for p in data["packets"])
    except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError):
        return None
    return round(end * base * 1000)


def check_timeline(
    series: Series, part: Part, timeline: PartTimeline, *, audio_dir: Path, tools: RenderTools
) -> None:
    """Raise `StaleTimelineError` unless `timeline` still describes `part` and its FLAC."""
    n = part.index
    want = (series.title, series.channel_slug, part.index, part.count)
    got = (timeline.series_title, timeline.channel_slug, timeline.part_index, timeline.part_count)
    if got != want:
        raise _stale(
            f"part {n}: the timeline is for {got[1]}/{got[0]} part {got[2]} of {got[3]}, "
            f"not {want[1]}/{want[0]} part {want[2]} of {want[3]}"
        )
    scenes = [(s.id, s.kind.value) for s in part.scenes]
    timed = [(t.scene_id, t.kind.value) for t in timeline.scenes]
    if scenes != timed:
        raise _stale(f"part {n}: the scenes changed since the audio build ({timed} -> {scenes})")
    for scene, timing in zip(part.scenes, timeline.scenes, strict=True):
        if _is_card(scene):
            if timing.draw_start_ms is not None or timing.sketch_id is not None:
                raise _stale(f"scene {scene.id} is now a text card but the timeline draws a sketch")
        elif _drawn(scene):
            if scene.sketch_id is None:
                raise PlanningError(
                    f"scene {scene.id} has no sketch yet; "
                    "run narratoflow sketch generate / resolve first"
                )
            if timing.draw_start_ms is None:
                raise _stale(f"scene {scene.id} is drawn but the timeline has no draw times")
            if timing.sketch_id is None or timing.sketch_id != scene.sketch_id:
                raise _stale(
                    f"scene {scene.id}: the timeline drew sketch {timing.sketch_id}, "
                    f"the scene now has sketch {scene.sketch_id}"
                )
    audio = audio_dir / timeline.audio_file
    if not audio.is_file():
        raise _stale(f"part {n}: {audio.name} is missing")
    audio_ms = audio_duration_ms(tools, audio)
    if audio_ms is None or abs(audio_ms - timeline.duration_ms) > AUDIO_TOLERANCE_MS:
        raise _stale(
            f"part {n}: {audio.name} lasts {audio_ms} ms, the timeline {timeline.duration_ms} ms"
        )


@dataclass
class _PartPlan:
    part: Part
    timeline: PartTimeline
    doc: AssDocument
    ass_warnings: list[str]
    sketches: dict[str, tuple[SketchRecord, StrokeSet]]
    verses: list[VerseRender]


@dataclass
class _Sources:
    """The library and corpus, opened on first need."""

    settings: Settings
    library: SketchLibrary | None
    corpus: QuranCorpus | None

    def get_library(self) -> SketchLibrary:
        if self.library is None:
            self.library = default_library(self.settings)
        return self.library

    def get_corpus(self) -> QuranCorpus:
        if self.corpus is None:
            self.corpus = default_corpus(self.settings)
        return self.corpus


def _selected(series: Series, parts: Sequence[int] | None) -> list[Part]:
    if parts is None:
        return list(series.parts)
    known = {p.index for p in series.parts}
    for n in parts:
        if n not in known:
            raise PlanningError(f"no part {n}; the series has parts {sorted(known)}")
    wanted = set(parts)
    return [p for p in series.parts if p.index in wanted]


def _sketch(scene: Scene, timing: SceneTiming, sources: _Sources) -> tuple[SketchRecord, StrokeSet]:
    sid = scene.sketch_id
    if sid is None:  # pragma: no cover - check_timeline refused it
        raise PlanningError(f"scene {scene.id} has no sketch yet")
    library = sources.get_library()
    record = library.get_sketch(sid)
    if record is None or record.status is SketchStatus.FAILED:
        raise RenderError(f"scene {scene.id}: sketch {sid} is quarantined; resolve it")
    strokes = library.get_strokes(sid)
    if strokes is None:
        raise LibraryError(f"scene {scene.id}: sketch {sid} has no strokes; re-run sketch generate")
    if len(strokes.strokes) != timing.stroke_count:
        raise _stale(
            f"scene {scene.id}: sketch {sid} has {len(strokes.strokes)} strokes, "
            f"the timeline drew {timing.stroke_count}"
        )
    return record, strokes


def _verse(scene: Scene, timing: SceneTiming, doc: AssDocument) -> VerseRender:
    """The verse's report entry; `shown_sha256` certifies the Verse events in the burned
    document that fall in the scene's slot."""
    burned = [e for e in doc.events if timing.start_ms <= e.start_ms < timing.end_ms]
    shown = shown_text(burned)
    return VerseRender(
        scene_id=scene.id,
        ayat=[f"{a.surah}:{a.ayah}" for a in timing.ayat],
        shown_sha256=hashlib.sha256(shown.encode("utf-8")).hexdigest(),
        font_id=VERSE_FONT,
        timing_source=timing.timing_source,
    )


def _plan_part(
    series: Series,
    channel: ChannelConfig,
    part: Part,
    *,
    audio_dir: Path,
    tools: RenderTools,
    fonts: FontPack,
    sources: _Sources,
) -> _PartPlan:
    n = part.index
    path = audio_dir / f"part-{n:02d}.timeline.json"
    if not path.is_file():
        raise StaleTimelineError(f"part {n} has no timeline; run narratoflow audio build")
    timeline = PartTimeline.load(path)
    check_timeline(series, part, timeline, audio_dir=audio_dir, tools=tools)
    if frame_of(timeline.duration_ms) <= 0:
        raise _stale(f"part {n}: the timeline is empty ({timeline.duration_ms} ms)")
    sketches: dict[str, tuple[SketchRecord, StrokeSet]] = {}
    verse_slots: list[tuple[Scene, SceneTiming]] = []
    corpus: QuranCorpus | None = None
    for scene, timing in zip(part.scenes, timeline.scenes, strict=True):
        if _drawn(scene):
            sketches[scene.id] = _sketch(scene, timing, sources)
        elif scene.kind is SceneKind.VERSE:
            corpus = sources.get_corpus()
            check_verse_text(scene, timing, corpus, fonts)
            _check_verse_pages(part, scene, timing, corpus)
            verse_slots.append((scene, timing))
    doc, warnings = build_part_ass(
        series, part, timeline, corpus=corpus, fonts=fonts, ink=channel.style.ink
    )
    verses = [_verse(scene, timing, doc) for scene, timing in verse_slots]
    return _PartPlan(part, timeline, doc, warnings, sketches, verses)


def _check_verse_pages(part: Part, scene: Scene, timing: SceneTiming, corpus: QuranCorpus) -> None:
    """Refuse a verse page with no time on screen: its corpus words would be dropped silently."""
    if scene.verse is None:
        raise RenderError(f"FAIL: verse scene {scene.id} has no verse reference")
    for page in verse_pages(timing, corpus, scene.verse):
        if page.end_ms <= page.start_ms:
            raise RenderError(
                f"FAIL: part {part.index}, scene {scene.id}, verse {page.surah}:{page.ayah}: "
                f"a verse page would last {page.end_ms - page.start_ms} ms "
                f"(starts at {page.start_ms} ms); rebuild audio: narratoflow audio build"
            )


def _init_worker() -> None:
    """Pool initializer: one OpenCV thread per worker (N workers, not N x cores)."""
    cv2.setNumThreads(1)


def _checked(job: SceneJob, written: int) -> None:
    if written != len(job.frames):
        raise RenderError(
            f"scene {job.scene_id}: wrote {written} frames, expected {len(job.frames)}"
        )


def _run_jobs(jobs: list[SceneJob], workers: int | None, part: int) -> None:
    """Render every segment; a scene's failure is `RenderError("scene <id>: ...")`.

    A dead worker process breaks the whole pool: every unfinished future then raises
    `BrokenProcessPool`, so which scene killed it is unknown and none is blamed.
    """
    if workers == 1:
        for job in jobs:
            try:
                written = compose.render_segment(job)
            except Exception as exc:
                raise RenderError(f"scene {job.scene_id}: {exc}") from exc
            _checked(job, written)
        return
    with ProcessPoolExecutor(
        max_workers=workers or min(os.cpu_count() or 1, len(jobs)),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
    ) as pool:
        futures = [pool.submit(compose.render_segment, job) for job in jobs]
        try:
            for job, future in zip(jobs, futures, strict=True):
                try:
                    written = future.result()
                except BrokenProcessPool as exc:
                    unfinished = [
                        j.scene_id
                        for j, f in zip(jobs, futures, strict=True)
                        if not f.done() or f.exception() is not None
                    ]
                    raise RenderError(
                        f"part {part}: a render worker died (scene unknown; "
                        f"unfinished scenes: {', '.join(unfinished)})"
                    ) from exc
                except Exception as exc:
                    raise RenderError(f"scene {job.scene_id}: {exc}") from exc
                _checked(job, written)
        except BaseException:
            # Cancel what has not started; the running jobs finish (their FrameSinks
            # close their ffmpeg) before the pool exits, so nothing is orphaned.
            pool.shutdown(wait=True, cancel_futures=True)
            raise


def visual_kind(scene: Scene) -> Literal["sketch", "text_card", "verse"]:
    if scene.kind is SceneKind.VERSE:
        return "verse"
    return "sketch" if _drawn(scene) else "text_card"


@dataclass
class _Run:
    series: Series
    channel: ChannelConfig
    tools: RenderTools
    fonts: FontPack
    hand: HandSprite
    canvas: Canvas
    audio_dir: Path
    out_dir: Path
    workers: int | None

    def jobs(self, plan: _PartPlan, work: Path) -> tuple[list[SceneJob], list[SceneRender]]:
        vid = video_id(self.series)
        style = self.channel.style
        jobs: list[SceneJob] = []
        scenes: list[SceneRender] = []
        for i, (scene, timing) in enumerate(
            zip(plan.part.scenes, plan.timeline.scenes, strict=True)
        ):
            frames = frame_range(timing.start_ms, timing.end_ms)
            visual = visual_kind(scene)
            variation: Variation | None = None
            strokes: StrokeSet | None = None
            draw: tuple[int, int] | None = None
            if visual == "sketch":
                record, strokes = plan.sketches[scene.id]
                variation = choose_variation(
                    vid,
                    scene.id,
                    mirror_safe=record.mirror_safe,
                    stroke_count=len(strokes.strokes),
                )
                if timing.draw_start_ms is None or timing.draw_end_ms is None:  # pragma: no cover
                    raise _stale(f"scene {scene.id} has no draw times")
                draw = (timing.draw_start_ms, timing.draw_end_ms)
            scenes.append(
                SceneRender(
                    scene_id=scene.id,
                    kind=scene.kind,
                    visual=visual,
                    sketch_id=scene.sketch_id if visual == "sketch" else None,
                    frames=len(frames),
                    variation=variation,
                )
            )
            if len(frames) == 0:
                continue
            jobs.append(
                SceneJob(
                    scene_id=scene.id,
                    kind="paper" if visual == "verse" else visual,
                    frames=frames,
                    slot=(timing.start_ms, timing.end_ms),
                    draw=draw,
                    strokes=strokes,
                    variation=variation,
                    wobble_seed=marker_seed(vid, scene.id),
                    line_weight_px=style.line_weight_px,
                    ink=style.ink,
                    canvas=self.canvas,
                    hand=self.hand,
                    segment=work / f"seg-{i:03d}.mkv",
                    ffmpeg=self.tools.ffmpeg,
                )
            )
        return jobs, scenes

    def assets(self, doc: AssDocument) -> list[RenderAsset]:
        language = self.series.language
        families = {self.fonts.get(style_font_id(language, e.style)).family for e in doc.events}
        out: list[RenderAsset] = []
        for fid in self.fonts.ids():
            font = self.fonts.get(fid)
            if font.family in families:
                out.append(
                    RenderAsset(
                        kind="font",
                        asset_id=fid,
                        source=font.source,
                        license=font.license,
                        sha256=font.sha256,
                    )
                )
        hand = self.hand
        out.append(
            RenderAsset(
                kind="hand",
                asset_id=hand.id,
                source=hand.source,
                license=hand.license,
                sha256=hand.sha256,
                attribution=hand.attribution,
            )
        )
        return out

    def part(self, plan: _PartPlan) -> PartVideo:
        n = plan.part.index
        stem = f"part-{n:02d}"
        video = self.out_dir / f"{stem}.mp4"
        report_path = self.out_dir / f"{stem}.render.json"
        captions = self.out_dir / f"{stem}.ass"
        timeline = plan.timeline
        frames = frame_of(timeline.duration_ms)
        work = self.out_dir / f".render-{stem}"
        try:
            try:
                # A failed re-render leaves nothing made from the old audio without its report.
                for old in (report_path, video, captions):
                    old.unlink(missing_ok=True)
                if work.exists():
                    shutil.rmtree(work)
                work.mkdir()
                self.fonts.copy_to(work / "fonts")
            except OSError as exc:
                raise RenderError(f"cannot prepare the work directory {work}: {exc}") from exc
            plan.doc.write(work / "captions.ass")
            jobs, scenes = self.jobs(plan, work)
            _run_jobs(jobs, self.workers, n)
            write_concat_list([j.segment for j in jobs], work / "segments.txt")
            encode_part(
                self.tools,
                work,
                frames=frames,
                audio=self.audio_dir / timeline.audio_file,
                out=video,
            )
            try:
                verify_mp4(
                    self.tools,
                    video,
                    size=(self.canvas.width, self.canvas.height),
                    frames=frames,
                    duration_ms=timeline.duration_ms,
                )
                plan.doc.write(captions)
                warnings = [*self._warnings(), *plan.ass_warnings]
                report = RenderReport(
                    series_title=self.series.title,
                    channel_slug=self.series.channel_slug,
                    part_index=n,
                    part_count=plan.part.count,
                    video_file=video.name,
                    timeline_file=f"{stem}.timeline.json",
                    audio_file=timeline.audio_file,
                    captions_file=captions.name,
                    width=self.canvas.width,
                    height=self.canvas.height,
                    frames=frames,
                    duration_ms=timeline.duration_ms,
                    crf=CRF,
                    ffmpeg_version=self.tools.version,
                    integrated_lufs=timeline.integrated_lufs,
                    true_peak_dbtp=timeline.true_peak_dbtp,
                    scenes=scenes,
                    verses=plan.verses,
                    assets=self.assets(plan.doc),
                    warnings=warnings,
                )
                report.save(report_path)
            except BaseException:
                video.unlink(missing_ok=True)  # never leave an unverified MP4 behind
                captions.unlink(missing_ok=True)  # nor captions without their report
                raise
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return PartVideo(n, video, report_path, report)

    def _warnings(self) -> list[str]:
        return [LANDSCAPE_WARNING] if AspectRatio.LANDSCAPE in self.channel.outputs else []


def build_render(
    series: Series,
    channel: ChannelConfig,
    settings: Settings,
    *,
    audio_dir: Path,
    out_dir: Path,
    parts: Sequence[int] | None = None,
    workers: int | None = None,
    canvas: Canvas | None = None,
    tools: RenderTools | None = None,
    library: SketchLibrary | None = None,
    corpus: QuranCorpus | None = None,
    fonts: FontPack | None = None,
    hand: HandSprite | None = None,
    on_part: Callable[[PartVideo], None] | None = None,
) -> RenderResult:
    """Render every selected part (`parts`: part indexes; None = all) to `out_dir`.

    `on_part` is called with each part as soon as it is finished (so a caller still
    knows which parts were kept when a later part fails or is interrupted).

    `workers`: 1 renders inline; None uses one process per core (at most one per
    scene). `canvas` scales the frame (tests use 1/6); the output is 9:16 only.
    """
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    if not series.approved:
        raise PlanningError("approve the series first")
    require_series_channel(series, channel)
    tools = tools if tools is not None else find_render_tools(settings)
    selected = _selected(series, parts)
    fonts = fonts if fonts is not None else default_fonts()
    hand = hand if hand is not None else default_hand()
    sources = _Sources(settings, library, corpus)
    plans = [
        _plan_part(
            series, channel, p, audio_dir=audio_dir, tools=tools, fonts=fonts, sources=sources
        )
        for p in selected
    ]
    if series.language is Language.ARABIC or any(p.verses for p in plans):
        with tempfile.TemporaryDirectory(prefix="narratoflow-shaping-") as tmp:
            check_arabic_shaping(tools, fonts, Path(tmp))

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RenderError(f"cannot create output directory {out_dir}: {exc}") from exc
    run = _Run(
        series=series,
        channel=channel,
        tools=tools,
        fonts=fonts,
        hand=hand,
        canvas=canvas if canvas is not None else Canvas(),
        audio_dir=audio_dir,
        out_dir=out_dir,
        workers=workers,
    )
    result = RenderResult()
    for plan in plans:
        built = run.part(plan)
        result.parts.append(built)
        logger.info("render part %d: %d frames", built.index, built.report.frames)
        if on_part is not None:
            on_part(built)
    return result
