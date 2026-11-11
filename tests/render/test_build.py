"""build_render: preflight refusals, per-part MP4 + report, workers, determinism, cleanup."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import shutil
import subprocess
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from dataclasses import replace
from datetime import date
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, ClassVar

import cv2
import httpx
import pytest
from conftest import FIXTURES
from fixtures import images as img
from fixtures.audio import speechlike, to_wav_bytes, tone

from narratoflow.audio import io as aio
from narratoflow.audio.build import build_audio
from narratoflow.audio.recitation import display_words
from narratoflow.audio.sfx import SfxAsset, SfxPack
from narratoflow.audio.timeline import (
    AyahTiming,
    PartTimeline,
    SceneTiming,
    TimelineWord,
    TimingSource,
)
from narratoflow.captions.part import build_part_ass
from narratoflow.captions.verse import VERSE_PAGE_MAX_WORDS, shown_text
from narratoflow.channels.config import ChannelConfig, load_channel
from narratoflow.costs.prices import PriceTable
from narratoflow.errors import LibraryError, PlanningError, RenderError, StaleTimelineError
from narratoflow.library.repository import SketchLibrary
from narratoflow.library.schemas import Provenance, SketchOrigin, SketchStatus
from narratoflow.models import (
    AspectRatio,
    Language,
    Narration,
    Part,
    RightsBasis,
    Scene,
    SceneKind,
    ScriptSource,
    Series,
    VerseRef,
    Visual,
    VisualSource,
)
from narratoflow.quran.corpus import QuranCorpus
from narratoflow.quran.reciters import Reciter, ReciterIndex
from narratoflow.render import build as build_module
from narratoflow.render import compose
from narratoflow.render.assets import default_fonts
from narratoflow.render.build import RenderResult, build_render, check_timeline
from narratoflow.render.encoder import verify_mp4
from narratoflow.render.ffmpeg import RenderTools
from narratoflow.render.layout import Canvas, frame_of, frame_range
from narratoflow.render.report import RenderReport
from narratoflow.settings import Settings
from narratoflow.tts import Speech, SpeechRequest, VoiceChoice
from narratoflow.visuals.strokes import Stroke, StrokeSet

CORPUS = QuranCorpus.from_files(
    FIXTURES / "quran/quran-simple-clean.txt", FIXTURES / "quran/quran-uthmani.txt"
)
SLOT_MS = 1000


class _Stop(Exception):
    pass


# -- fixtures -----------------------------------------------------------------


@pytest.fixture
def probe_tools() -> RenderTools:
    """Plain ffmpeg/ffprobe (no libass needed): enough for every check before a frame."""
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        if os.environ.get("CI"):
            pytest.fail("ffmpeg is required in CI")
        pytest.skip("ffmpeg not installed")
    return RenderTools(Path(ffmpeg), Path(ffprobe), "ffmpeg version test")


@pytest.fixture
def channel(channel_file: Path) -> ChannelConfig:
    return load_channel(channel_file)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def never_render(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Patch `compose.render_segment` (the module attribute build.py calls) to fail."""
    calls: list[str] = []

    def boom(job: compose.SceneJob) -> int:
        calls.append(job.scene_id)
        raise AssertionError("a frame was rendered")

    monkeypatch.setattr(compose, "render_segment", boom)
    yield calls
    assert calls == []


# -- builders -----------------------------------------------------------------


def _strokes(n: int = 6) -> StrokeSet:
    return StrokeSet(
        width=100,
        height=100,
        strokes=[Stroke(points=[(0.1, 0.1 + 0.12 * i), (0.9, 0.14 + 0.12 * i)]) for i in range(n)],
    )


def add_sketch(library: SketchLibrary, *, strokes: bool = True, key: str = "scroll|roman") -> int:
    version = library.ensure_style_version("history", "fp")
    record = library.add_sketch(
        version,
        key,
        img.line_drawing_scroll(),
        ".png",
        SketchOrigin.GENERATED,
        SketchStatus.PASSED,
        True,
        Provenance.model_validate(
            {"source": "t", "license": "t", "created_at": "2026-01-01T00:00:00Z"}
        ),
    )
    if strokes:
        library.put_strokes(record.id, _strokes())
    return record.id


def narr(sid: str, text: str, *, sketch: int | None = None, title: str = "") -> Scene:
    visual = Visual() if sketch is not None else Visual(source=VisualSource.TEXT_CARD)
    return Scene(
        id=sid,
        title=title,
        narration=Narration(source_text=text),
        visual=visual,
        sketch_id=sketch,
        min_duration_ms=0,
    )


def verse(sid: str) -> Scene:
    return Scene(
        id=sid,
        narration=Narration(source_text="verse words never voiced", protected=True),
        kind=SceneKind.VERSE,
        verse=VerseRef(surah=31, ayah_start=33, ayah_end=33, certain=True, uthmani_text="x"),
    )


def make_series(
    *parts: list[Scene], approved: bool = True, language: Language = Language.ARABIC
) -> Series:
    return Series(
        title="Scenes",
        channel_slug="history",
        language=language,
        source=ScriptSource(author="a", work="Scenes", rights=RightsBasis.OWN),
        parts=[
            Part(index=i + 1, count=len(parts), hook_text="Hook", scenes=scenes)
            for i, scenes in enumerate(parts)
        ],
        approved=approved,
    )


def _words(text: str, lo: int, hi: int) -> list[TimelineWord]:
    words = text.split()
    step = (hi - lo) // max(len(words), 1)
    return [
        TimelineWord(text=w, start_ms=lo + i * step, end_ms=lo + (i + 1) * step)
        for i, w in enumerate(words)
    ]


def _timing(scene: Scene, start: int, library: SketchLibrary | None) -> SceneTiming:
    end = start + SLOT_MS
    if scene.kind is SceneKind.VERSE:
        return SceneTiming(
            scene_id=scene.id,
            kind=SceneKind.VERSE,
            start_ms=start,
            end_ms=end,
            speech_start_ms=start,
            speech_end_ms=end,
            words=[],
            timing_source=TimingSource.AYAH,
            ayat=[AyahTiming(surah=31, ayah=33, start_ms=start, end_ms=end)],
        )
    drawn = scene.visual.source is not VisualSource.TEXT_CARD
    count: int | None = None
    if drawn and scene.sketch_id is not None and library is not None:
        found = library.get_strokes(scene.sketch_id)
        count = None if found is None else len(found.strokes)
    return SceneTiming(
        scene_id=scene.id,
        kind=SceneKind.NARRATION,
        start_ms=start,
        end_ms=end,
        speech_start_ms=start + 100,
        speech_end_ms=end - 100,
        words=_words(scene.narration.source_text, start + 100, end - 100),
        timing_source=TimingSource.ESTIMATED,
        draw_start_ms=start if drawn else None,
        draw_end_ms=start + 600 if drawn else None,
        sketch_id=scene.sketch_id if drawn else None,
        stroke_count=count,
    )


def write_timelines(
    series: Series,
    audio_dir: Path,
    library: SketchLibrary | None = None,
    *,
    lufs: float | None = -14.0,
    flac: bool = True,
) -> list[PartTimeline]:
    """The timelines (and FLACs) audio build would write for `series`, made directly."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    out: list[PartTimeline] = []
    for part in series.parts:
        scenes = [_timing(s, i * SLOT_MS, library) for i, s in enumerate(part.scenes)]
        stem = f"part-{part.index:02d}"
        timeline = PartTimeline(
            series_title=series.title,
            channel_slug=series.channel_slug,
            part_index=part.index,
            part_count=part.count,
            audio_file=f"{stem}.flac",
            duration_ms=scenes[-1].end_ms,
            integrated_lufs=lufs,
            true_peak_dbtp=None if lufs is None else -1.5,
            scenes=scenes,
            sfx=[],
            voice=None,
            recitations=[],
            assets=[],
            spent_usd=Decimal(0),
            cost_estimated=False,
            warnings=["a timeline warning"],
        )
        timeline.save(audio_dir / f"{stem}.timeline.json")
        if flac:
            aio.encode_flac(tone(timeline.duration_ms), audio_dir / timeline.audio_file)
        out.append(timeline)
    return out


def render(
    series: Series,
    channel: ChannelConfig,
    settings: Settings,
    tmp_path: Path,
    tools: RenderTools,
    *,
    out: str = "video",
    **kw: Any,
) -> RenderResult:
    options: dict[str, Any] = {
        "workers": 1,
        "canvas": Canvas(Fraction(1, 6)),
        "corpus": CORPUS,
        **kw,
    }
    return build_render(
        series,
        channel,
        settings,
        audio_dir=tmp_path / "audio",
        out_dir=tmp_path / out,
        tools=tools,
        **options,
    )


def framemd5(tools: RenderTools, path: Path) -> list[str]:
    proc = subprocess.run(  # noqa: S603
        [str(tools.ffmpeg), "-v", "error", "-i", str(path), "-map", "0:v", "-f", "framemd5", "-"],
        capture_output=True,
        check=True,
        timeout=120,
    )
    lines = proc.stdout.decode().splitlines()
    return [ln for ln in lines if not ln.startswith("#")]


def _one_part(library: SketchLibrary, language: Language = Language.ARABIC) -> Series:
    sid = add_sketch(library)
    return make_series(
        [narr("s1", "alpha beta gamma", sketch=sid, title="One"), narr("s2", "delta epsilon")],
        language=language,
    )


# -- end to end (libass) ------------------------------------------------------


def test_renders_mp4_and_report_per_part(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
    tiny_canvas: Canvas,
) -> None:
    sid = add_sketch(library)
    series = make_series(
        [narr("s1", "alpha beta", sketch=sid), narr("s2", "gamma delta")],
        [narr("s3", "epsilon zeta", sketch=sid)],
    )
    timelines = write_timelines(series, tmp_path / "audio", library)
    result = render(series, channel, settings, tmp_path, render_tools, library=library)

    out_dir = tmp_path / "video"
    assert [p.index for p in result.parts] == [1, 2]
    for got, timeline in zip(result.parts, timelines, strict=True):
        stem = f"part-{got.index:02d}"
        assert got.video_path == out_dir / f"{stem}.mp4"
        assert got.report_path == out_dir / f"{stem}.render.json"
        report = RenderReport.load(got.report_path)
        assert report == got.report
        frames = frame_of(timeline.duration_ms)
        assert report.frames == frames and report.duration_ms == timeline.duration_ms
        assert (report.width, report.height) == (tiny_canvas.width, tiny_canvas.height)
        verify_mp4(
            render_tools,
            got.video_path,
            size=(tiny_canvas.width, tiny_canvas.height),
            frames=frames,
            duration_ms=timeline.duration_ms,
        )
        assert (report.video_file, report.timeline_file) == (f"{stem}.mp4", f"{stem}.timeline.json")
        assert (report.audio_file, report.captions_file) == (f"{stem}.flac", f"{stem}.ass")
        assert (
            (out_dir / report.captions_file).read_text(encoding="utf-8").startswith("[Script Info]")
        )
        assert (report.series_title, report.channel_slug) == ("Scenes", "history")
        assert (report.part_index, report.part_count) == (got.index, 2)
        assert (report.crf, report.ffmpeg_version) == (18, render_tools.version)
        assert (report.integrated_lufs, report.true_peak_dbtp) == (-14.0, -1.5)
        assert "a timeline warning" not in report.warnings  # timeline warnings stay there
        assert report.verses == []
        assert {a.asset_id for a in report.assets if a.kind == "font"} == {
            "amiri-bold",
            "amiri-regular",
        }
        [hand] = [a for a in report.assets if a.kind == "hand"]
        assert (hand.asset_id, hand.license) == ("right", "Apache-2.0")
        assert sum(s.frames for s in report.scenes) == frames
    first = result.parts[0].report.scenes
    assert [(s.scene_id, s.visual, s.sketch_id) for s in first] == [
        ("s1", "sketch", sid),
        ("s2", "text_card", None),
    ]
    assert first[0].variation is not None and first[0].variation.wobble_seed > 0
    assert first[1].variation is None
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "part-01.ass",
        "part-01.mp4",
        "part-01.render.json",
        "part-02.ass",
        "part-02.mp4",
        "part-02.render.json",
    ]  # no work dir is left behind


def _reciter() -> Reciter:
    return Reciter(
        id="files-hafs",
        name="Files",
        riwaya="hafs",
        kind="files",
        audio_url="https://audio.example.org/r/{surah:03d}{ayah:03d}.mp3",
        license="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        source="https://audio.example.org/r/",
        checked=date(2026, 11, 11),
    )


class FakeVoice:
    name = "paid"
    model = "m1"

    def synthesize(self, req: SpeechRequest) -> Speech:
        audio = to_wav_bytes(speechlike(len(req.text.split()), lead_ms=150))
        return Speech(
            audio=audio, spans=(), provider=self.name, model=self.model, billed_chars=len(req.text)
        )


def test_reference_series_31_33_end_to_end(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    sid = add_sketch(library)
    series = make_series([verse("v1"), narr("s1", "alpha beta gamma", sketch=sid)])
    recitation = to_wav_bytes(speechlike(3, seed=7))

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/031033.mp3"):
            return httpx.Response(200, content=recitation)
        return httpx.Response(404)  # pragma: no cover - only 31:33 is asked for

    path = tmp_path / "pen.flac"
    aio.encode_flac(tone(200, hz=2500, dbfs=-9), path)
    pack = SfxPack(
        {"pen": SfxAsset(id="pen", file=path, license="CC0-1.0", source="https://sfx.example/")}
    )
    _, audio = build_audio(
        series,
        channel,
        settings,
        out_dir=tmp_path / "audio",
        cache_root=tmp_path / ".cache",
        voices=[VoiceChoice(FakeVoice(), "v1")],
        aligner=None,
        reciters=ReciterIndex([_reciter()]),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        corpus=CORPUS,
        library=library,
        sfx=pack,
        prices=PriceTable({"voice": {"paid/m1": (Decimal("0.30"),)}}),
    )
    [part_audio] = audio.parts
    timeline = part_audio.timeline

    result = render(series, channel, settings, tmp_path, render_tools, library=library)
    [part] = result.parts
    report = RenderReport.load(part.report_path)
    [v] = report.verses
    uthmani = CORPUS.ayah(31, 33).uthmani
    expected = hashlib.sha256(" ".join(uthmani.split()).encode("utf-8")).hexdigest()
    assert (v.scene_id, v.ayat, v.font_id) == ("v1", ["31:33"], "amiri-quran")
    assert v.shown_sha256 == expected
    assert v.timing_source is TimingSource.AYAH
    assert report.scenes[0].visual == "verse" and report.scenes[0].variation is None
    assert "amiri-quran" in {a.asset_id for a in report.assets}
    verify_mp4(
        render_tools,
        part.video_path,
        size=(180, 320),
        frames=frame_of(timeline.duration_ms),
        duration_ms=timeline.duration_ms,
    )


def test_none_loudness_fields_pass_through(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    series = _one_part(library)
    write_timelines(series, tmp_path / "audio", library, lufs=None)
    [part] = render(series, channel, settings, tmp_path, render_tools, library=library).parts
    report = RenderReport.load(part.report_path)
    assert report.integrated_lufs is None and report.true_peak_dbtp is None
    assert report.warnings == []


def test_landscape_output_warns(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    series = _one_part(library)
    write_timelines(series, tmp_path / "audio", library)
    wide = channel.model_copy(update={"outputs": [AspectRatio.PORTRAIT, AspectRatio.LANDSCAPE]})
    [part] = render(series, wide, settings, tmp_path, render_tools, library=library).parts
    assert part.report.warnings[0] == "16:9 is not rendered in v1"
    assert (part.report.width, part.report.height) == (180, 320)  # 9:16 only


def test_workers_do_not_change_frames(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    sid = add_sketch(library)
    series = make_series(
        [narr("s1", "alpha beta", sketch=sid), narr("s2", "gamma"), narr("s3", "delta", sketch=sid)]
    )
    write_timelines(series, tmp_path / "audio", library)
    one = render(series, channel, settings, tmp_path, render_tools, library=library, out="a")
    two = render(
        series, channel, settings, tmp_path, render_tools, library=library, out="b", workers=2
    )
    assert framemd5(render_tools, one.parts[0].video_path) == framemd5(
        render_tools, two.parts[0].video_path
    )
    assert one.parts[0].report == two.parts[0].report


def test_rerender_identical_frames(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    series = _one_part(library)
    write_timelines(series, tmp_path / "audio", library)
    first = render(series, channel, settings, tmp_path, render_tools, library=library)
    before = framemd5(render_tools, first.parts[0].video_path)
    (tmp_path / "video" / ".render-part-01").mkdir()  # a leftover work dir is replaced
    (tmp_path / "video" / ".render-part-01" / "junk").write_text("x")
    again = render(series, channel, settings, tmp_path, render_tools, library=library)
    assert framemd5(render_tools, again.parts[0].video_path) == before
    assert again.parts[0].report == first.parts[0].report
    assert not (tmp_path / "video" / ".render-part-01").exists()


def test_unverified_mp4_removed_and_earlier_parts_kept(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = make_series([narr("s1", "alpha")], [narr("s2", "beta")])
    write_timelines(series, tmp_path / "audio", library)
    real = verify_mp4

    def flaky(tools: RenderTools, path: Path, **kw: Any) -> Any:
        if path.name == "part-02.mp4":
            assert path.is_file()  # encode_part renamed it into place
            raise RenderError("part-02.mp4: frames is 1, expected 60")
        return real(tools, path, **kw)

    monkeypatch.setattr(build_module, "verify_mp4", flaky)
    with pytest.raises(RenderError, match="frames is 1"):
        render(series, channel, settings, tmp_path, render_tools, library=library)
    assert sorted(p.name for p in (tmp_path / "video").iterdir()) == [
        "part-01.ass",
        "part-01.mp4",
        "part-01.render.json",
    ]


def test_render_spends_nothing_and_keeps_series(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    series = _one_part(library)
    before = series.model_copy(deep=True)
    write_timelines(series, tmp_path / "audio", library)
    render(series, channel, settings, tmp_path, render_tools, library=library, parts=[1])
    assert series == before


# -- refusals before any frame (no libass needed) -----------------------------


def test_stale_sketch_id_refused_before_any_frame(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    probe_tools: RenderTools,
    never_render: list[str],
) -> None:
    series = _one_part(library)
    write_timelines(series, tmp_path / "audio", library)
    other = add_sketch(library, key="lamp|roman")
    series.parts[0].scenes[0].sketch_id = other  # `sketch resolve` after the audio build
    with pytest.raises(StaleTimelineError, match=r"rebuild audio: narratoflow audio build$"):
        render(series, channel, settings, tmp_path, probe_tools, library=library)
    assert not (tmp_path / "video").exists()


def test_stroke_count_mismatch_refused(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    probe_tools: RenderTools,
    never_render: list[str],
) -> None:
    series = _one_part(library)
    write_timelines(series, tmp_path / "audio", library)
    sid = series.parts[0].scenes[0].sketch_id
    assert sid is not None
    library.put_strokes(sid, _strokes(4))  # strokes re-extracted after the audio build
    with pytest.raises(StaleTimelineError, match="4 strokes, the timeline drew 6"):
        render(series, channel, settings, tmp_path, probe_tools, library=library)


def test_text_card_swap_after_audio_refused(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    probe_tools: RenderTools,
    never_render: list[str],
) -> None:
    series = _one_part(library)
    write_timelines(series, tmp_path / "audio", library)
    card = series.model_copy(deep=True)
    card.parts[0].scenes[0].visual = Visual(source=VisualSource.TEXT_CARD)
    with pytest.raises(StaleTimelineError, match="s1 is now a text card"):
        render(card, channel, settings, tmp_path, probe_tools, library=library)
    drawn = series.model_copy(deep=True)
    drawn.parts[0].scenes[1].visual = Visual()
    drawn.parts[0].scenes[1].sketch_id = series.parts[0].scenes[0].sketch_id
    with pytest.raises(StaleTimelineError, match="s2 is drawn but the timeline has no draw"):
        render(drawn, channel, settings, tmp_path, probe_tools, library=library)
    unsketched = series.model_copy(deep=True)
    unsketched.parts[0].scenes[0].sketch_id = None
    with pytest.raises(PlanningError, match="scene s1 has no sketch yet"):
        render(unsketched, channel, settings, tmp_path, probe_tools, library=library)


def test_scene_list_changed_refused(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    probe_tools: RenderTools,
    never_render: list[str],
) -> None:
    series = _one_part(library)
    write_timelines(series, tmp_path / "audio", library)
    added = series.model_copy(deep=True)
    added.parts[0].scenes.append(narr("s3", "zeta"))
    with pytest.raises(StaleTimelineError, match="the scenes changed"):
        render(added, channel, settings, tmp_path, probe_tools, library=library)
    swapped = series.model_copy(deep=True)
    swapped.parts[0].scenes.reverse()
    with pytest.raises(StaleTimelineError, match="the scenes changed"):
        render(swapped, channel, settings, tmp_path, probe_tools, library=library)
    renamed = series.model_copy(update={"title": "Other"})
    with pytest.raises(StaleTimelineError, match="history/Scenes part 1 of 1, not history/Other"):
        render(renamed, channel, settings, tmp_path, probe_tools, library=library)


def test_missing_timeline_and_flac_refused(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    probe_tools: RenderTools,
    never_render: list[str],
) -> None:
    series = _one_part(library)
    with pytest.raises(StaleTimelineError, match=r"^part 1 has no timeline; run narratoflow audio"):
        render(series, channel, settings, tmp_path, probe_tools, library=library)
    write_timelines(series, tmp_path / "audio", library, flac=False)
    with pytest.raises(StaleTimelineError, match=r"part-01.flac is missing; rebuild audio"):
        render(series, channel, settings, tmp_path, probe_tools, library=library)
    aio.encode_flac(tone(SLOT_MS), tmp_path / "audio" / "part-01.flac")  # half the length
    with pytest.raises(StaleTimelineError, match="lasts 1000 ms, the timeline 2000 ms"):
        render(series, channel, settings, tmp_path, probe_tools, library=library)
    (tmp_path / "audio" / "part-01.flac").write_bytes(img.line_drawing_scroll())  # a PNG
    with pytest.raises(StaleTimelineError, match="lasts None ms"):  # no audio stream at all
        render(series, channel, settings, tmp_path, probe_tools, library=library)


def test_check_timeline_accepts_within_30_ms(
    library: SketchLibrary, tmp_path: Path, probe_tools: RenderTools
) -> None:
    series = _one_part(library)
    [timeline] = write_timelines(series, tmp_path / "audio", library, flac=False)
    flac = tmp_path / "audio" / "part-01.flac"
    aio.encode_flac(tone(2 * SLOT_MS + 25), flac)
    check_timeline(
        series, series.parts[0], timeline, audio_dir=tmp_path / "audio", tools=probe_tools
    )
    aio.encode_flac(tone(2 * SLOT_MS + 40), flac)
    with pytest.raises(StaleTimelineError, match="lasts 2040 ms"):
        check_timeline(
            series, series.parts[0], timeline, audio_dir=tmp_path / "audio", tools=probe_tools
        )


def test_quarantined_sketch_refused(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    probe_tools: RenderTools,
    never_render: list[str],
) -> None:
    series = _one_part(library)
    write_timelines(series, tmp_path / "audio", library)
    sid = series.parts[0].scenes[0].sketch_id
    assert sid is not None
    library.quarantine(sid, "bad")
    with pytest.raises(RenderError, match=f"^scene s1: sketch {sid} is quarantined; resolve it$"):
        render(series, channel, settings, tmp_path, probe_tools, library=library)


def test_missing_strokes_refused(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    probe_tools: RenderTools,
    never_render: list[str],
) -> None:
    sid = add_sketch(library, strokes=False)
    series = make_series([narr("s1", "alpha", sketch=sid)])
    [timeline] = write_timelines(series, tmp_path / "audio", library)
    timing = timeline.scenes[0].model_copy(update={"stroke_count": 6})
    timeline.model_copy(update={"scenes": [timing]}).save(
        tmp_path / "audio" / "part-01.timeline.json"
    )
    with pytest.raises(LibraryError, match=f"scene s1: sketch {sid} has no strokes"):
        render(series, channel, settings, tmp_path, probe_tools, library=library)


def test_verse_mismatch_fails_before_encode(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    probe_tools: RenderTools,
    never_render: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = make_series([verse("v1"), narr("s1", "alpha")])
    [timeline] = write_timelines(series, tmp_path / "audio", library)
    v1 = timeline.scenes[0]
    wrong = v1.model_copy(
        update={"ayat": [AyahTiming(surah=31, ayah=34, start_ms=v1.start_ms, end_ms=v1.end_ms)]}
    )
    timeline.model_copy(update={"scenes": [wrong, *timeline.scenes[1:]]}).save(
        tmp_path / "audio" / "part-01.timeline.json"
    )
    shaping: list[object] = []
    monkeypatch.setattr(build_module, "check_arabic_shaping", lambda *a: shaping.append(a))
    with pytest.raises(RenderError, match=r"^FAIL: verse 31:33: the timeline has ayat 31:34"):
        render(series, channel, settings, tmp_path, probe_tools, library=library)
    assert shaping == []  # refused before the shaping check and before any frame
    assert not (tmp_path / "video").exists()


def test_unapproved_refused(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    probe_tools: RenderTools,
    never_render: list[str],
) -> None:
    series = make_series([narr("s1", "alpha")], approved=False)
    with pytest.raises(PlanningError, match=r"^approve the series first$"):
        render(series, channel, settings, tmp_path, probe_tools, library=library)
    other = make_series([narr("s1", "alpha")]).model_copy(update={"channel_slug": "other"})
    with pytest.raises(PlanningError, match="belongs to channel 'other'"):
        render(other, channel, settings, tmp_path, probe_tools, library=library)
    with pytest.raises(PlanningError, match=r"no part 3; the series has parts \[1\]"):
        render(make_series([narr("s1", "a")]), channel, settings, tmp_path, probe_tools, parts=[3])
    with pytest.raises(ValueError, match="workers"):
        render(make_series([narr("s1", "a")]), channel, settings, tmp_path, probe_tools, workers=0)


def test_missing_tools_refused_before_timelines(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary, tmp_path: Path
) -> None:
    series = _one_part(library)
    missing = settings.model_copy(update={"ffmpeg": tmp_path / "no-ffmpeg"})
    with pytest.raises(RenderError, match="ffmpeg not found at"):
        build_render(series, channel, missing, audio_dir=tmp_path / "audio", out_dir=tmp_path / "v")


def test_empty_timeline_refused(
    channel: ChannelConfig,
    settings: Settings,
    tmp_path: Path,
    probe_tools: RenderTools,
    never_render: list[str],
) -> None:
    series = make_series([narr("s1", "alpha")])
    [timeline] = write_timelines(series, tmp_path / "audio", flac=False)
    empty = timeline.scenes[0].model_copy(
        update={"end_ms": 0, "speech_start_ms": 0, "speech_end_ms": 0, "words": []}
    )
    timeline.model_copy(update={"scenes": [empty], "duration_ms": 0}).save(
        tmp_path / "audio" / "part-01.timeline.json"
    )
    aio.encode_flac(tone(10), tmp_path / "audio" / "part-01.flac")
    with pytest.raises(StaleTimelineError, match="the timeline is empty"):
        render(series, channel, settings, tmp_path, probe_tools)


def test_shaping_check_runs_only_for_arabic_or_verse(
    channel: ChannelConfig,
    settings: Settings,
    tmp_path: Path,
    probe_tools: RenderTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(
        build_module, "check_arabic_shaping", lambda tools, fonts, work: calls.append(work)
    )

    def stop(job: compose.SceneJob) -> int:
        raise _Stop("stopped after preflight")

    monkeypatch.setattr(compose, "render_segment", stop)

    def attempt(series: Series) -> None:
        if (tmp_path / "audio").exists():
            shutil.rmtree(tmp_path / "audio")
        write_timelines(series, tmp_path / "audio")
        with pytest.raises(RenderError, match=r"^scene .*: stopped after preflight$"):
            render(series, channel, settings, tmp_path, probe_tools)

    attempt(make_series([narr("s1", "alpha")], [narr("s2", "beta")], language=Language.ENGLISH))
    assert calls == []
    attempt(make_series([narr("s1", "alpha")], [narr("s2", "beta")]))  # Arabic: once, not per part
    assert len(calls) == 1
    attempt(make_series([verse("v1"), narr("s1", "alpha")], language=Language.ENGLISH))
    assert len(calls) == 2
    assert not calls[0].exists()  # its scratch directory is removed


def test_work_dir_removed_on_failure(
    channel: ChannelConfig,
    settings: Settings,
    tmp_path: Path,
    probe_tools: RenderTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = make_series([narr("s1", "alpha"), narr("s2", "beta")], language=Language.ENGLISH)
    write_timelines(series, tmp_path / "audio")
    seen: list[Path] = []

    def fail_second(job: compose.SceneJob) -> int:
        seen.append(job.segment)
        assert (job.segment.parent / "captions.ass").is_file()
        assert (job.segment.parent / "fonts" / "NotoSans-Bold.ttf").is_file()
        if job.scene_id == "s2":
            raise _Stop("segment writer died")
        job.segment.write_bytes(b"partial")
        return len(job.frames)

    monkeypatch.setattr(compose, "render_segment", fail_second)
    with pytest.raises(RenderError, match=r"^scene s2: segment writer died$") as info:
        render(series, channel, settings, tmp_path, probe_tools)
    assert isinstance(info.value.__cause__, _Stop)
    assert [p.name for p in seen] == ["seg-000.mkv", "seg-001.mkv"]
    assert seen[0].parent == tmp_path / "video" / ".render-part-01"
    assert list((tmp_path / "video").iterdir()) == []


def test_wrong_segment_frame_count_refused(
    channel: ChannelConfig,
    settings: Settings,
    tmp_path: Path,
    probe_tools: RenderTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = make_series([narr("s1", "alpha")], language=Language.ENGLISH)
    write_timelines(series, tmp_path / "audio")
    monkeypatch.setattr(compose, "render_segment", lambda job: len(job.frames) - 1)
    with pytest.raises(RenderError, match="scene s1: wrote 29 frames, expected 30"):
        render(series, channel, settings, tmp_path, probe_tools)


class _InlinePool:
    """A ProcessPoolExecutor stand-in that runs the initializer and jobs in-process."""

    made: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kw: Any) -> None:
        self.made.append(kw)
        self.shutdowns: list[dict[str, Any]] = []
        kw["initializer"]()

    def __enter__(self) -> _InlinePool:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def submit(self, fn: Callable[[compose.SceneJob], int], job: compose.SceneJob) -> Future[int]:
        future: Future[int] = Future()
        try:
            future.set_result(fn(job))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **kw: Any) -> None:
        self.made[-1]["shutdown"] = kw


def test_pool_initializer_sets_cv2_single_thread(
    channel: ChannelConfig,
    settings: Settings,
    tmp_path: Path,
    probe_tools: RenderTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    threads: list[int] = []
    monkeypatch.setattr(cv2, "setNumThreads", threads.append)
    monkeypatch.setattr(build_module, "ProcessPoolExecutor", _InlinePool)
    _InlinePool.made.clear()

    def stop(job: compose.SceneJob) -> int:
        raise _Stop("stop")

    monkeypatch.setattr(compose, "render_segment", stop)
    series = make_series(
        [narr("s1", "a"), narr("s2", "b"), narr("s3", "c")], language=Language.ENGLISH
    )
    write_timelines(series, tmp_path / "audio")
    with pytest.raises(RenderError, match=r"^scene s1: stop$"):
        render(series, channel, settings, tmp_path, probe_tools, workers=None)
    [made] = _InlinePool.made
    assert threads == [1]
    assert made["initializer"] is build_module._init_worker
    assert isinstance(made["mp_context"], type(multiprocessing.get_context("spawn")))
    assert made["mp_context"].get_start_method() == "spawn"
    assert made["max_workers"] == min(os.cpu_count() or 1, 3)
    assert made["shutdown"] == {"wait": True, "cancel_futures": True}


def test_worker_exception_becomes_render_error(
    channel: ChannelConfig, settings: Settings, tmp_path: Path, probe_tools: RenderTools
) -> None:
    """A real spawn pool: the worker's FrameSink cannot start its ffmpeg."""
    series = make_series([narr("s1", "alpha"), narr("s2", "beta")], language=Language.ENGLISH)
    write_timelines(series, tmp_path / "audio")
    broken = RenderTools(tmp_path / "no-ffmpeg", probe_tools.ffprobe, probe_tools.version)
    with pytest.raises(RenderError, match=r"^scene s1: cannot run .*no-ffmpeg") as info:
        render(series, channel, settings, tmp_path, broken, workers=2)
    assert isinstance(info.value.__cause__, RenderError)
    assert list((tmp_path / "video").iterdir()) == []


def test_zero_length_scene_gets_no_segment(
    channel: ChannelConfig,
    settings: Settings,
    tmp_path: Path,
    probe_tools: RenderTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = make_series([narr("s1", "alpha"), narr("s2", "beta")], language=Language.ENGLISH)
    [timeline] = write_timelines(series, tmp_path / "audio", flac=False)
    a, b = timeline.scenes
    short = a.model_copy(
        update={"end_ms": 10, "speech_start_ms": 0, "speech_end_ms": 10, "words": []}
    )
    later = b.model_copy(update={"start_ms": 10})
    timeline.model_copy(update={"scenes": [short, later]}).save(
        tmp_path / "audio" / "part-01.timeline.json"
    )
    aio.encode_flac(tone(timeline.duration_ms), tmp_path / "audio" / "part-01.flac")
    assert len(frame_range(0, 10)) == 0
    jobs: list[str] = []

    def stop(job: compose.SceneJob) -> int:
        jobs.append(job.scene_id)
        raise _Stop("stop")

    monkeypatch.setattr(compose, "render_segment", stop)
    with pytest.raises(RenderError, match=r"^scene s2: stop$"):
        render(series, channel, settings, tmp_path, probe_tools)
    assert jobs == ["s2"]


def test_interrupt_is_not_wrapped_and_cleans_up(
    channel: ChannelConfig,
    settings: Settings,
    tmp_path: Path,
    probe_tools: RenderTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = make_series([narr("s1", "alpha")], language=Language.ENGLISH)
    write_timelines(series, tmp_path / "audio")

    def interrupted(job: compose.SceneJob) -> int:
        job.segment.write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(compose, "render_segment", interrupted)
    with pytest.raises(KeyboardInterrupt):
        render(series, channel, settings, tmp_path, probe_tools)
    assert list((tmp_path / "video").iterdir()) == []


# -- final-review fixes -------------------------------------------------------


def test_failed_rerender_removes_the_old_part_files(
    channel: ChannelConfig,
    settings: Settings,
    tmp_path: Path,
    probe_tools: RenderTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = make_series([narr("s1", "alpha")], language=Language.ENGLISH)
    write_timelines(series, tmp_path / "audio")
    out = tmp_path / "video"
    out.mkdir()
    for name in ("part-01.mp4", "part-01.ass", "part-01.render.json"):
        (out / name).write_bytes(b"from the old audio")

    def stop(job: compose.SceneJob) -> int:
        raise _Stop("stop")

    monkeypatch.setattr(compose, "render_segment", stop)
    with pytest.raises(RenderError, match=r"^scene s1: stop$"):
        render(series, channel, settings, tmp_path, probe_tools)
    assert list(out.iterdir()) == []


def test_failed_report_save_removes_mp4_and_captions(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = make_series([narr("s1", "alpha")], language=Language.ENGLISH)
    write_timelines(series, tmp_path / "audio", library)

    def disk_full(self: RenderReport, path: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(RenderReport, "save", disk_full)
    with pytest.raises(OSError, match="disk full"):
        render(series, channel, settings, tmp_path, render_tools, library=library)
    assert list((tmp_path / "video").iterdir()) == []


def test_verse_shown_sha256_comes_from_the_burned_document(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    probe_tools: RenderTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = make_series([verse("v1"), narr("s1", "alpha")])
    write_timelines(series, tmp_path / "audio", library)
    real = build_part_ass

    def marked(*a: Any, **kw: Any) -> Any:
        doc, warnings = real(*a, **kw)
        for i, e in enumerate(doc.events):
            if e.style == "Verse":
                doc.events[i] = replace(e, text=e.text + " MARK")
        return doc, warnings

    monkeypatch.setattr(build_module, "build_part_ass", marked)
    plan = build_module._plan_part(
        series,
        channel,
        series.parts[0],
        audio_dir=tmp_path / "audio",
        tools=probe_tools,
        fonts=default_fonts(),
        sources=build_module._Sources(settings, library, CORPUS),
    )
    [v] = plan.verses
    burned = shown_text([e for e in plan.doc.events if e.style == "Verse"])
    assert burned.endswith("MARK")
    assert v.shown_sha256 == hashlib.sha256(burned.encode("utf-8")).hexdigest()


def test_dead_worker_does_not_blame_a_scene(
    channel: ChannelConfig,
    settings: Settings,
    tmp_path: Path,
    probe_tools: RenderTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(build_module, "ProcessPoolExecutor", _InlinePool)
    _InlinePool.made.clear()

    def die_after_first(job: compose.SceneJob) -> int:
        if job.scene_id == "s1":
            return len(job.frames)
        raise BrokenProcessPool("A process in the process pool was terminated abruptly")

    monkeypatch.setattr(compose, "render_segment", die_after_first)
    series = make_series(
        [narr("s1", "a"), narr("s2", "b"), narr("s3", "c")], language=Language.ENGLISH
    )
    write_timelines(series, tmp_path / "audio")
    with pytest.raises(RenderError) as info:
        render(series, channel, settings, tmp_path, probe_tools, workers=2)
    assert str(info.value) == (
        "part 1: a render worker died (scene unknown; unfinished scenes: s2, s3)"
    )
    assert isinstance(info.value.__cause__, BrokenProcessPool)


def test_on_part_reports_each_finished_part(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = make_series([narr("s1", "alpha")], [narr("s2", "beta")], language=Language.ENGLISH)
    write_timelines(series, tmp_path / "audio", library)
    real = compose.render_segment

    def fail_part_two(job: compose.SceneJob) -> int:
        if job.scene_id == "s2":
            raise _Stop("stop")
        return real(job)

    monkeypatch.setattr(compose, "render_segment", fail_part_two)
    done: list[int] = []
    with pytest.raises(RenderError, match="scene s2"):
        render(
            series,
            channel,
            settings,
            tmp_path,
            render_tools,
            library=library,
            on_part=lambda p: done.append(p.index),
        )
    assert done == [1]


def test_work_dir_oserror_becomes_render_error(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    probe_tools: RenderTools,
    never_render: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = make_series([narr("s1", "alpha")], language=Language.ENGLISH)
    write_timelines(series, tmp_path / "audio", library)
    real_mkdir = Path.mkdir

    def mkdir(self: Path, *a: Any, **kw: Any) -> None:
        if self.name.startswith(".render-"):
            raise OSError("read-only")
        real_mkdir(self, *a, **kw)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    with pytest.raises(RenderError, match=r"cannot prepare the work directory .*read-only"):
        render(series, channel, settings, tmp_path, probe_tools, library=library)


def test_out_dir_oserror_becomes_render_error(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    probe_tools: RenderTools,
    never_render: list[str],
) -> None:
    series = make_series([narr("s1", "alpha")], language=Language.ENGLISH)
    write_timelines(series, tmp_path / "audio", library)
    (tmp_path / "video").write_text("a file, not a directory")
    with pytest.raises(RenderError, match="cannot create output directory"):
        render(series, channel, settings, tmp_path, probe_tools, library=library)


def test_sources_open_defaults_on_first_need(
    settings: Settings, library: SketchLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(build_module, "default_library", lambda s: library)
    monkeypatch.setattr(build_module, "default_corpus", lambda s: CORPUS)
    sources = build_module._Sources(settings, None, None)
    assert sources.get_library() is library
    assert sources.get_corpus() is CORPUS
    assert sources.library is library and sources.corpus is CORPUS


def test_verse_page_without_duration_refused_naming_part_scene_ayah(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    probe_tools: RenderTools,
    never_render: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = make_series([verse("v1"), narr("s1", "alpha")])
    [timeline] = write_timelines(series, tmp_path / "audio", library)
    v1 = timeline.scenes[0]
    texts = display_words(CORPUS.ayah(31, 33).uthmani)
    assert len(texts) > VERSE_PAGE_MAX_WORDS
    # Every word starts together, so page 2 starts where page 1 starts: page 1 has no time.
    words = [TimelineWord(text=t, start_ms=v1.start_ms, end_ms=v1.end_ms) for t in texts]
    timed = v1.model_copy(
        update={
            "timing_source": TimingSource.SEGMENTS,
            "ayat": [
                AyahTiming(surah=31, ayah=33, start_ms=v1.start_ms, end_ms=v1.end_ms, words=words)
            ],
        }
    )
    timeline.model_copy(update={"scenes": [timed, *timeline.scenes[1:]]}).save(
        tmp_path / "audio" / "part-01.timeline.json"
    )
    monkeypatch.setattr(build_module, "check_arabic_shaping", lambda *a: None)
    with pytest.raises(RenderError, match=r"FAIL: part 1, scene v1, verse 31:33"):
        render(series, channel, settings, tmp_path, probe_tools, library=library)
    assert not (tmp_path / "video").exists()
