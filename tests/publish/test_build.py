"""build_publish: preflight refusals, per-part blocking export, reports and the summary."""

from __future__ import annotations

import socket
import sys
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from fixtures.audio import speechlike, to_wav_bytes, tone
from fixtures.parts import (
    CORPUS,
    GENERATED_PROVENANCE,
    FakeVoice,
    add_sketch,
    fake_render_report,
    make_series,
    narr,
    reciter,
    render,
    verse,
    write_timelines,
)
from fixtures.publish import write_rules

from narratoflow.audio import io as aio
from narratoflow.audio.build import build_audio
from narratoflow.audio.sfx import SfxAsset, SfxPack
from narratoflow.audio.timeline import AssetUse, PartTimeline, VoiceUsed
from narratoflow.channels.config import ChannelConfig
from narratoflow.costs.prices import PriceTable
from narratoflow.costs.tracker import TrackedLLM
from narratoflow.errors import PlanningError, PublishError, RenderError
from narratoflow.library.repository import SketchLibrary
from narratoflow.models import Series
from narratoflow.publish import build as build_module
from narratoflow.publish.build import PublishResult, build_publish
from narratoflow.publish.checks import CheckId, Severity
from narratoflow.publish.report import EARLIER_RUN, NOT_CHECKED, PublishReport, SeriesSummary
from narratoflow.publish.rules import Rules, load_rules
from narratoflow.publish.technical import MediaFacts, inspect_video
from narratoflow.quran.reciters import ReciterIndex
from narratoflow.render.ffmpeg import RenderTools
from narratoflow.settings import Settings
from narratoflow.tts import VoiceChoice

EDGE = VoiceUsed(provider="edge", model="", voice_id="ar-SA-HamedNeural", terms="edge-tts-terms")
MARKER = AssetUse(
    kind="sfx", asset_id="marker-1", source="scripts/make_sfx.py", license="Apache-2.0"
)
NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)
EXPORTS = ("part-01.mp4", "part-01.disclosure.txt", "part-01.publish.md")


def _now() -> datetime:
    return NOW


def _tiny_rules(tmp_path: Path) -> Rules:
    changes: dict[str, object] = {
        "platforms.youtube-shorts.width": 180,
        "platforms.youtube-shorts.height": 320,
        "platforms.tiktok.width": 180,
        "platforms.tiktok.height": 320,
    }
    return load_rules(write_rules(tmp_path, **changes))


def _pipeline(
    tmp_path: Path,
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tools: RenderTools,
    *,
    parts: int = 1,
) -> Series:
    """Sketch, timelines (mastered, edge voice, marker SFX) and a tiny real render."""
    sid = add_sketch(library, provenance=GENERATED_PROVENANCE)
    scenes = [narr("s1", "alpha beta gamma", sketch=sid, title="One"), narr("s2", "delta")]
    series = make_series(*[scenes for _ in range(parts)])
    write_timelines(series, tmp_path / "audio", library, master=True, voice=EDGE, assets=[MARKER])
    render(series, channel, settings, tmp_path, tools, library=library)
    return series


def _publish(
    series: Series,
    channel: ChannelConfig,
    settings: Settings,
    tmp_path: Path,
    tools: RenderTools | None,
    library: SketchLibrary | None,
    **kw: Any,
) -> PublishResult:
    options: dict[str, Any] = {
        "tools": tools,
        "library": library,
        "corpus": CORPUS,
        "rules": _tiny_rules(tmp_path),
        "now": _now,
        **kw,
    }
    return build_publish(
        series,
        channel,
        settings,
        render_dir=tmp_path / "video",
        audio_dir=tmp_path / "audio",
        export_dir=tmp_path / "export",
        **options,
    )


def _summary(tmp_path: Path) -> SeriesSummary:
    return SeriesSummary.load(tmp_path / "video" / "series.publish.json")


@pytest.fixture
def one_part(
    tmp_path: Path,
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    render_tools: RenderTools,
) -> Series:
    return _pipeline(tmp_path, channel, settings, library, render_tools)


@pytest.fixture
def two_parts(
    tmp_path: Path,
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    render_tools: RenderTools,
) -> Series:
    return _pipeline(tmp_path, channel, settings, library, render_tools, parts=2)


# -- export -------------------------------------------------------------------


def test_clean_part_exported_with_disclosure_and_report(
    one_part: Series,
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    result = _publish(one_part, channel, settings, tmp_path, render_tools, library)
    [done] = result.parts
    report = done.report
    export = tmp_path / "export"
    assert result.blocked == []
    assert all(c.severity is not Severity.FAIL for c in report.checks)
    assert report.verdict is Severity.WARN
    warned = {c.check for c in report.checks if c.severity is Severity.WARN}
    assert {CheckId.CONTENT_RULES, CheckId.ASSET_PROVENANCE} <= warned
    assert report.exported is True and report.export_file == "part-01.mp4"
    assert report.video_file == "part-01.mp4"
    assert report.created_at == NOW
    assert done.export_path == export / "part-01.mp4"
    assert done.export_path.read_bytes() == (tmp_path / "video" / "part-01.mp4").read_bytes()
    disclosure = (export / "part-01.disclosure.txt").read_text(encoding="utf-8")
    assert disclosure == report.disclosure_text
    assert (export / "part-01.publish.md").read_text(encoding="utf-8") == report.to_markdown()
    assert done.report_path == tmp_path / "video" / "part-01.publish.json"
    assert PublishReport.load(done.report_path) == report
    md = (tmp_path / "video" / "part-01.publish.md").read_text(encoding="utf-8")
    assert md == report.to_markdown()
    summary = _summary(tmp_path)
    [line] = summary.parts
    assert (line.verdict, line.exported, line.note) == (Severity.WARN, True, "")
    assert result.summary_path == tmp_path / "video" / "series.publish.json"
    summary_md = (tmp_path / "video" / "series.publish.md").read_text(encoding="utf-8")
    assert summary_md == summary.to_markdown()
    assert (export / "series.publish.md").read_text(encoding="utf-8") == summary_md
    assert sorted(p.name for p in export.iterdir()) == sorted([*EXPORTS, "series.publish.md"])


def test_fail_blocks_export_and_no_override(
    one_part: Series,
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    undisclosed = one_part.model_copy(update={"ai_disclosure": False})
    result = _publish(undisclosed, channel, settings, tmp_path, render_tools, library)
    assert result.blocked == [1]
    [done] = result.parts
    assert done.export_path is None
    assert done.report.exported is False and done.report.export_file is None
    assert done.report.verdict is Severity.FAIL
    assert PublishReport.load(done.report_path).exported is False
    export = tmp_path / "export"
    assert not export.exists() or list(export.iterdir()) == []
    assert _summary(tmp_path).parts[0].verdict is Severity.FAIL


def test_fail_removes_earlier_export(
    one_part: Series,
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    first = _publish(one_part, channel, settings, tmp_path, render_tools, library)
    assert first.blocked == []
    export = tmp_path / "export"
    assert all((export / name).is_file() for name in EXPORTS)
    sid = one_part.parts[0].scenes[0].sketch_id
    assert sid is not None
    library.quarantine(sid, "a face")
    second = _publish(one_part, channel, settings, tmp_path, render_tools, library)
    assert second.blocked == [1]
    assert not any((export / name).exists() for name in EXPORTS)
    [line] = _summary(tmp_path).parts
    assert (line.verdict, line.exported) == (Severity.FAIL, False)
    likeness = [c for c in second.parts[0].report.checks if c.check is CheckId.SKETCH_LIKENESS]
    assert likeness[0].severity is Severity.FAIL


def test_default_rules_fail_tiny_render_resolution(
    one_part: Series,
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    result = build_publish(
        one_part,
        channel,
        settings,
        render_dir=tmp_path / "video",
        audio_dir=tmp_path / "audio",
        export_dir=tmp_path / "export",
        tools=render_tools,
        library=library,
        now=_now,
    )
    assert result.blocked == [1]
    fails = [
        c
        for c in result.parts[0].report.checks
        if c.check is CheckId.TECHNICAL and c.severity is Severity.FAIL
    ]
    assert {c.platform for c in fails} == {"youtube-shorts", "tiktok"}
    assert result.parts[0].report.rules_origin == "packaged"


# -- preflight ----------------------------------------------------------------


_NO_TOOLS = RenderTools(Path("/nonexistent/ffmpeg"), Path("/nonexistent/ffprobe"), "none")


def _unrendered(tmp_path: Path, library: SketchLibrary) -> Series:
    sid = add_sketch(library, provenance=GENERATED_PROVENANCE)
    series = make_series([narr("s1", "alpha beta", sketch=sid)])
    write_timelines(series, tmp_path / "audio", library, flac=False, voice=EDGE)
    (tmp_path / "video").mkdir()
    return series


def test_missing_render_report_or_timeline_is_error(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary, tmp_path: Path
) -> None:
    series = _unrendered(tmp_path, library)
    with pytest.raises(PublishError, match=r"part 1 has no render report; run narratoflow render"):
        _publish(series, channel, settings, tmp_path, _NO_TOOLS, library)
    timeline = tmp_path / "audio" / "part-01.timeline.json"
    report = fake_render_report(series, series.parts[0], PartTimeline.load(timeline))
    report.save(tmp_path / "video" / "part-01.render.json")
    timeline.unlink()
    with pytest.raises(
        PublishError,
        match=r"part 1: part-01\.timeline\.json is missing; run narratoflow audio build",
    ):
        _publish(series, channel, settings, tmp_path, _NO_TOOLS, library)
    assert sorted(p.name for p in (tmp_path / "video").iterdir()) == ["part-01.render.json"]
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("export", ["video", "video/export", "."])
def test_export_dir_overlapping_render_dir_is_refused(
    one_part: Series,
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
    export: str,
) -> None:
    video_dir = tmp_path / "video"
    before = sorted(p.name for p in video_dir.iterdir())
    undisclosed = one_part.model_copy(update={"ai_disclosure": False})
    with pytest.raises(PublishError, match="must be outside the render directory"):
        build_publish(
            undisclosed,
            channel,
            settings,
            render_dir=video_dir,
            audio_dir=tmp_path / "audio",
            export_dir=tmp_path / export,
            tools=render_tools,
            library=library,
            rules=_tiny_rules(tmp_path),
            now=_now,
        )
    assert (video_dir / "part-01.mp4").is_file()
    assert sorted(p.name for p in video_dir.iterdir()) == before
    assert not (video_dir / "export").exists()


def test_unknown_part_index_is_error(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary, tmp_path: Path
) -> None:
    series = _unrendered(tmp_path, library)
    with pytest.raises(PlanningError, match=r"no part 3; the series has parts \[1\]"):
        _publish(series, channel, settings, tmp_path, _NO_TOOLS, library, parts=[3])


def test_unapproved_and_wrong_channel_refused(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary, tmp_path: Path
) -> None:
    series = _unrendered(tmp_path, library)
    unapproved = series.model_copy(update={"approved": False})
    with pytest.raises(PlanningError, match="approve the series first"):
        _publish(unapproved, channel, settings, tmp_path, _NO_TOOLS, library)
    other = series.model_copy(update={"channel_slug": "other"})
    with pytest.raises(PlanningError, match="belongs to channel 'other'"):
        _publish(other, channel, settings, tmp_path, _NO_TOOLS, library)


def test_tools_are_found_when_not_given(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    series = _unrendered(tmp_path, library)
    timeline = tmp_path / "audio" / "part-01.timeline.json"
    report = fake_render_report(series, series.parts[0], PartTimeline.load(timeline))
    report.save(tmp_path / "video" / "part-01.render.json")
    seen: list[Settings] = []

    def no_tools(s: Settings) -> RenderTools:
        seen.append(s)
        raise RenderError("no libass")

    monkeypatch.setattr(build_module, "find_render_tools", no_tools)
    with pytest.raises(RenderError, match="no libass"):
        _publish(series, channel, settings, tmp_path, None, library)
    assert seen == [settings]


# -- no spend -----------------------------------------------------------------


def test_publish_spends_nothing_and_keeps_series(
    one_part: Series,
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_a: object, **_k: object) -> None:
        raise AssertionError("publish must not build or call an LLM")

    for target in (
        "narratoflow.llm.factory.build_llm_client",
        "narratoflow.llm.factory.make_llm",
        "narratoflow.publish.review.build_llm_client",
        "narratoflow.sketching.generate.build_llm_client",
    ):
        monkeypatch.setattr(target, refuse)
    # ...and every other module that binds a factory name.
    bound = 0
    for name, module in list(sys.modules.items()):
        if not name.startswith("narratoflow"):
            continue
        for attr in ("build_llm_client", "make_llm"):
            if hasattr(module, attr):
                monkeypatch.setattr(module, attr, refuse)
                bound += 1
    assert bound >= 4  # factory (2), publish.review, sketching.generate
    monkeypatch.setattr(TrackedLLM, "complete_structured", refuse)

    def no_network(*_a: object, **_k: object) -> None:
        raise AssertionError("publish must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    spent = one_part.model_copy(
        update={
            "parts": [
                p.model_copy(update={"spent_usd": Decimal("0.25"), "spend": {"x": Decimal("0.2")}})
                for p in one_part.parts
            ]
        }
    )
    before = spent.model_dump_json()
    _publish(spent, channel, settings, tmp_path, render_tools, library)
    assert spent.model_dump_json() == before
    assert [(p.spent_usd, p.spend) for p in spent.parts] == [
        (Decimal("0.25"), {"x": Decimal("0.2")})
    ]


# -- export failures ----------------------------------------------------------


def test_export_hash_mismatch_raises(
    one_part: Series,
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = inspect_video

    def wrong_sha(tools: RenderTools, video: Path) -> MediaFacts:
        return replace(real(tools, video), sha256="0" * 64)

    monkeypatch.setattr(build_module, "inspect_video", wrong_sha)
    with pytest.raises(PublishError, match=r"part 1: .*sha256"):
        _publish(one_part, channel, settings, tmp_path, render_tools, library)
    report = PublishReport.load(tmp_path / "video" / "part-01.publish.json")
    assert report.exported is False and report.export_file is None
    assert not (tmp_path / "video" / "series.publish.json").exists()
    export = tmp_path / "export"
    assert [p.name for p in export.iterdir()] == []  # no temp file, no export


def test_export_copy_failure_raises(
    one_part: Series,
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    (tmp_path / "export").write_text("not a directory")
    with pytest.raises(PublishError, match="cannot create the export directory"):
        _publish(one_part, channel, settings, tmp_path, render_tools, library)
    report = PublishReport.load(tmp_path / "video" / "part-01.publish.json")
    assert report.exported is False
    assert not (tmp_path / "video" / "series.publish.json").exists()

    (tmp_path / "export").unlink()
    (tmp_path / "export").mkdir()
    (tmp_path / "export").chmod(0o555)
    try:
        with pytest.raises(PublishError, match=r"part 1: cannot copy part-01\.mp4"):
            _publish(one_part, channel, settings, tmp_path, render_tools, library)
    finally:
        (tmp_path / "export").chmod(0o755)
    assert list((tmp_path / "export").iterdir()) == []
    assert PublishReport.load(tmp_path / "video" / "part-01.publish.json").exported is False


def test_old_export_that_cannot_be_removed_is_error(
    one_part: Series,
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    (tmp_path / "export" / "part-01.mp4").mkdir(parents=True)
    (tmp_path / "export" / "part-01.mp4" / "keep").write_text("x")
    undisclosed = one_part.model_copy(update={"ai_disclosure": False})
    with pytest.raises(PublishError, match="part 1: cannot remove the old export"):
        _publish(undisclosed, channel, settings, tmp_path, render_tools, library)


def test_library_opens_only_for_sketch_scenes(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = add_sketch(library, provenance=GENERATED_PROVENANCE)
    cards = make_series([narr("s1", "alpha beta"), narr("s2", "gamma delta")])
    write_timelines(cards, tmp_path / "audio", library, master=True, voice=EDGE)
    render(cards, channel, settings, tmp_path, render_tools, library=library)
    opened: list[Settings] = []

    def no_library(s: Settings) -> SketchLibrary:
        opened.append(s)
        return library

    monkeypatch.setattr(build_module, "default_library", no_library)
    result = _publish(cards, channel, settings, tmp_path, render_tools, None)
    assert opened == [] and result.blocked == []

    drawn = make_series([narr("s1", "alpha beta", sketch=sid), narr("s2", "gamma delta")])
    write_timelines(drawn, tmp_path / "audio", library, master=True, voice=EDGE)
    render(drawn, channel, settings, tmp_path, render_tools, library=library)
    result = _publish(drawn, channel, settings, tmp_path, render_tools, None)
    assert opened == [settings] and result.blocked == []


def test_unreadable_video_is_technical_fail_and_run_continues(
    two_parts: Series,
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    video_dir = tmp_path / "video"
    (video_dir / "part-01.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42 not really an mp4")
    result = _publish(two_parts, channel, settings, tmp_path, render_tools, library)
    assert result.blocked == [1]
    first, second = result.parts
    technical = [c for c in first.report.checks if c.check is CheckId.TECHNICAL]
    assert [c.severity for c in technical] == [Severity.FAIL]
    reason = technical[0].reason
    assert reason.startswith("part-01.mp4 cannot be read: ffprobe failed (exit ")
    assert "\n" not in reason and "0x" not in reason and str(tmp_path) not in reason
    for path in (first.report_path, video_dir / "part-01.publish.md"):
        assert str(video_dir) not in path.read_text(encoding="utf-8")
    assert first.report.video_sha256 is None and first.export_path is None
    assert second.report.exported is True and second.export_path is not None
    assert second.export_path.is_file()


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("", "RenderError"),
        ("moov atom not found", "moov atom not found"),
        (
            "ffprobe failed (exit 1):\n\n[mov,mp4 @ 0x957088000] moov atom not found\n"
            "/private/var/x/video/part-01.mp4: Invalid data",
            "ffprobe failed (exit 1): [mov,mp4] moov atom not found",
        ),
        ("cannot read /tmp/a/b/part-01.mp4: denied", "cannot read part-01.mp4: denied"),
    ],
)
def test_short_error(message: str, expected: str) -> None:
    video = Path("/elsewhere/video/part-01.mp4")
    assert build_module._short_error(RenderError(message), video) == expected


def test_missing_video_is_technical_fail(
    one_part: Series,
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    (tmp_path / "video" / "part-01.mp4").unlink()
    result = _publish(one_part, channel, settings, tmp_path, render_tools, library)
    assert result.blocked == [1]
    technical = [c for c in result.parts[0].report.checks if c.check is CheckId.TECHNICAL]
    assert [c.severity for c in technical] == [Severity.FAIL]


# -- the summary --------------------------------------------------------------


def test_summary_uses_one_run_clock(
    two_parts: Series,
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    calls: list[datetime] = []

    def ticking() -> Iterator[datetime]:
        t = NOW
        while True:
            yield t
            t += timedelta(minutes=1)

    clock = ticking()

    def now() -> datetime:
        calls.append(next(clock))
        return calls[-1]

    result = _publish(two_parts, channel, settings, tmp_path, render_tools, library, now=now)
    assert calls == [NOW]
    assert [p.report.created_at for p in result.parts] == [NOW, NOW]
    summary = _summary(tmp_path)
    assert summary.created_at == NOW
    assert [(p.note, p.checked_at) for p in summary.parts] == [("", NOW), ("", NOW)]


def test_part_filter_and_summary_reads_earlier_reports(
    two_parts: Series,
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    first = _publish(two_parts, channel, settings, tmp_path, render_tools, library, parts=[1])
    assert [p.index for p in first.parts] == [1]
    one, two = _summary(tmp_path).parts
    assert (one.verdict, one.note) == (Severity.WARN, "")
    assert (two.verdict, two.note, two.exported) == (None, NOT_CHECKED, False)
    assert not (tmp_path / "video" / "part-02.publish.json").exists()

    later = lambda: NOW + timedelta(hours=1)  # noqa: E731
    _publish(two_parts, channel, settings, tmp_path, render_tools, library, parts=[2], now=later)
    one, two = _summary(tmp_path).parts
    assert (one.verdict, one.note, one.checked_at) == (Severity.WARN, EARLIER_RUN, NOW)
    assert (two.verdict, two.note) == (Severity.WARN, "")

    # A report for another series, or one that cannot be read, is not counted.
    path = tmp_path / "video" / "part-01.publish.json"
    other = PublishReport.load(path).model_copy(update={"series_title": "Other"})
    other.save(path)
    _publish(two_parts, channel, settings, tmp_path, render_tools, library, parts=[2])
    assert _summary(tmp_path).parts[0].note == NOT_CHECKED
    path.write_text("{not json", encoding="utf-8")
    _publish(two_parts, channel, settings, tmp_path, render_tools, library, parts=[2])
    assert _summary(tmp_path).parts[0].note == NOT_CHECKED


# -- a verse part -------------------------------------------------------------


class _ElevenVoice(FakeVoice):
    name = "elevenlabs"


def test_reference_verse_part_checks_reciter(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    tmp_path: Path,
    render_tools: RenderTools,
) -> None:
    sid = add_sketch(library, provenance=GENERATED_PROVENANCE)
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
    build_audio(
        series,
        channel,
        settings,
        out_dir=tmp_path / "audio",
        cache_root=tmp_path / ".cache",
        voices=[VoiceChoice(_ElevenVoice(), "v1")],
        aligner=None,
        reciters=ReciterIndex([reciter()]),
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        corpus=CORPUS,
        library=library,
        sfx=pack,
        prices=PriceTable({"voice": {"elevenlabs/m1": (Decimal("0.30"),)}}),
    )
    render(series, channel, settings, tmp_path, render_tools, library=library)

    allowed = _publish(
        series,
        channel,
        settings,
        tmp_path,
        render_tools,
        library,
        reciters=ReciterIndex([reciter()]),
    )
    checks = allowed.parts[0].report.checks
    verse_text = [c for c in checks if c.check is CheckId.VERSE_TEXT]
    assert [c.severity for c in verse_text] == [Severity.PASS]
    provenance = [c for c in checks if c.check is CheckId.ASSET_PROVENANCE]
    assert provenance and all(c.severity is Severity.WARN for c in provenance)

    refused = _publish(
        series, channel, settings, tmp_path, render_tools, library, reciters=ReciterIndex([])
    )
    assert refused.blocked == [1]
    fails = [
        c
        for c in refused.parts[0].report.checks
        if c.check is CheckId.ASSET_PROVENANCE and c.severity is Severity.FAIL
    ]
    assert fails and any("files-hafs" in c.reason for c in fails)
