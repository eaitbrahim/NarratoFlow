"""Series, timeline and render-report builders shared by render, audio and publish tests."""

from __future__ import annotations

import hashlib
from datetime import date
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from fixtures import images as img
from fixtures.audio import speechlike, to_wav_bytes, tone
from narratoflow.audio import io as aio
from narratoflow.audio.timeline import (
    AssetUse,
    AyahTiming,
    PartTimeline,
    SceneTiming,
    TimelineWord,
    TimingSource,
    VoiceUsed,
)
from narratoflow.captions.ass import style_font_id
from narratoflow.captions.part import build_part_ass
from narratoflow.captions.verse import VERSE_FONT
from narratoflow.channels.config import ChannelConfig
from narratoflow.library.repository import SketchLibrary
from narratoflow.library.schemas import Provenance, SketchOrigin, SketchStatus
from narratoflow.models import (
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
from narratoflow.publish.verse import expected_verse_text
from narratoflow.quran.corpus import QuranCorpus
from narratoflow.quran.reciters import Reciter
from narratoflow.render.assets import default_fonts, default_hand
from narratoflow.render.build import RenderResult, build_render
from narratoflow.render.ffmpeg import RenderTools
from narratoflow.render.layout import Canvas, frame_of, frame_range
from narratoflow.render.report import RenderAsset, RenderReport, SceneRender, VerseRender
from narratoflow.settings import Settings
from narratoflow.tts import Speech, SpeechRequest
from narratoflow.visuals.strokes import Stroke, StrokeSet

FIXTURES = Path(__file__).parent

CORPUS = QuranCorpus.from_files(
    FIXTURES / "quran/quran-simple-clean.txt", FIXTURES / "quran/quran-uthmani.txt"
)
SLOT_MS = 1000

GENERATED_PROVENANCE = Provenance.model_validate(
    {
        "source": "generated",
        "license": "openai-terms",
        "provider": "openai",
        "model": "gpt-image-1",
        "created_at": "2026-01-01T00:00:00Z",
        "usage_terms": "https://openai.com/policies/terms-of-use",
    }
)


def _strokes(n: int = 6) -> StrokeSet:
    return StrokeSet(
        width=100,
        height=100,
        strokes=[Stroke(points=[(0.1, 0.1 + 0.12 * i), (0.9, 0.14 + 0.12 * i)]) for i in range(n)],
    )


def add_sketch(
    library: SketchLibrary,
    *,
    strokes: bool = True,
    key: str = "scroll|roman",
    provenance: Provenance | None = None,
) -> int:
    version = library.ensure_style_version("history", "fp")
    record = library.add_sketch(
        version,
        key,
        img.line_drawing_scroll(),
        ".png",
        SketchOrigin.GENERATED,
        SketchStatus.PASSED,
        True,
        provenance
        or Provenance.model_validate(
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
    master: bool = False,
    voice: VoiceUsed | None = None,
    assets: list[AssetUse] | None = None,
) -> list[PartTimeline]:
    """The timelines (and FLACs) audio build would write for `series`, made directly.

    `master` passes the tone through `aio.master` (-14 LUFS) before encoding."""
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
            voice=voice,
            recitations=[],
            assets=list(assets or []),
            spent_usd=Decimal(0),
            cost_estimated=False,
            warnings=["a timeline warning"],
        )
        timeline.save(audio_dir / f"{stem}.timeline.json")
        if flac:
            samples = tone(timeline.duration_ms)
            if master:
                samples, _ = aio.master(samples)
            aio.encode_flac(samples, audio_dir / timeline.audio_file)
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


def reciter() -> Reciter:
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


def _verse_sha256(timing: SceneTiming) -> str:
    text = expected_verse_text(CORPUS, [(a.surah, a.ayah) for a in timing.ayat])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _scene_visual(scene: Scene) -> Literal["sketch", "text_card", "verse"]:
    if scene.kind is SceneKind.VERSE:
        return "verse"
    return "text_card" if scene.visual.source is VisualSource.TEXT_CARD else "sketch"


def _report_assets(series: Series, part: Part, timeline: PartTimeline) -> list[RenderAsset]:
    """The font and hand entries render would record, chosen as `build_render` does."""
    fonts = default_fonts()
    doc, _ = build_part_ass(series, part, timeline, corpus=CORPUS, fonts=fonts, ink="#000000")
    families = {fonts.get(style_font_id(series.language, e.style)).family for e in doc.events}
    out = [
        RenderAsset(
            kind="font", asset_id=fid, source=font.source, license=font.license, sha256=font.sha256
        )
        for fid in fonts.ids()
        if (font := fonts.get(fid)).family in families
    ]
    hand = default_hand()
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


def fake_render_report(
    series: Series,
    part: Part,
    timeline: PartTimeline,
    *,
    video_file: str | None = None,
    assets: list[RenderAsset] | None = None,
) -> RenderReport:
    """A report consistent with `timeline`, built without rendering a frame."""
    stem = f"part-{part.index:02d}"
    scenes: list[SceneRender] = []
    verses: list[VerseRender] = []
    for scene, timing in zip(part.scenes, timeline.scenes, strict=True):
        visual = _scene_visual(scene)
        scenes.append(
            SceneRender(
                scene_id=scene.id,
                kind=scene.kind,
                visual=visual,
                sketch_id=scene.sketch_id if visual == "sketch" else None,
                frames=len(frame_range(timing.start_ms, timing.end_ms)),
                variation=None,
            )
        )
        if scene.kind is SceneKind.VERSE:
            verses.append(
                VerseRender(
                    scene_id=scene.id,
                    ayat=[f"{a.surah}:{a.ayah}" for a in timing.ayat],
                    shown_sha256=_verse_sha256(timing),
                    font_id=VERSE_FONT,
                    timing_source=timing.timing_source,
                )
            )
    return RenderReport(
        series_title=series.title,
        channel_slug=series.channel_slug,
        part_index=part.index,
        part_count=part.count,
        video_file=video_file or f"{stem}.mp4",
        timeline_file=f"{stem}.timeline.json",
        audio_file=timeline.audio_file,
        captions_file=f"{stem}.ass",
        width=1080,
        height=1920,
        frames=frame_of(timeline.duration_ms),
        duration_ms=timeline.duration_ms,
        crf=18,
        ffmpeg_version="ffmpeg version test",
        integrated_lufs=timeline.integrated_lufs,
        true_peak_dbtp=timeline.true_peak_dbtp,
        scenes=scenes,
        verses=verses,
        assets=_report_assets(series, part, timeline) if assets is None else assets,
        warnings=[],
    )
