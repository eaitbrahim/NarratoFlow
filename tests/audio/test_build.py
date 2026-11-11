"""build_audio: pre-spend checks, per-part narration + mix, files, and cross-stage spend."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from conftest import FIXTURES
from fixtures import images as img
from fixtures.audio import speechlike, to_wav_bytes, tone

from narratoflow.align import ctc
from narratoflow.audio import build as build_module
from narratoflow.audio import io as aio
from narratoflow.audio import mixer
from narratoflow.audio.build import (
    AudioBuildResult,
    AudioInterruptedError,
    build_audio,
)
from narratoflow.audio.io import Loudness, Samples
from narratoflow.audio.narrate import ALIGN_EXTRA_WARNING
from narratoflow.audio.sfx import SfxAsset, SfxPack
from narratoflow.audio.timeline import PartTimeline, TimingSource
from narratoflow.channels.config import ChannelConfig, load_channel
from narratoflow.costs.prices import PriceTable
from narratoflow.errors import (
    AudioError,
    BudgetExceededError,
    ChannelConfigError,
    LibraryError,
    MissingApiKeyError,
    PlanningError,
    ProviderError,
    ReciterError,
)
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
from narratoflow.money import format_usd
from narratoflow.quran.corpus import QuranCorpus
from narratoflow.quran.reciters import Reciter, ReciterIndex
from narratoflow.settings import Settings
from narratoflow.tts import Speech, SpeechRequest, VoiceChoice
from narratoflow.visuals.strokes import Stroke, StrokeSet

pytestmark = pytest.mark.usefixtures("ffmpeg_bin")

PER_KCHAR = Decimal("0.30")
PRICES = PriceTable({"voice": {"paid/m1": (PER_KCHAR,), "other/m2": (Decimal("0.10"),)}})
CORPUS = QuranCorpus.from_files(
    FIXTURES / "quran/quran-simple-clean.txt", FIXTURES / "quran/quran-uthmani.txt"
)
PRICES_WITH_ELEVEN = PriceTable({"voice": {"elevenlabs/eleven_multilingual_v2": (PER_KCHAR,)}})
RECITATION = to_wav_bytes(speechlike(5, seed=7))
Handler = Callable[[httpx.Request], httpx.Response]


class FakeVoice:
    def __init__(self, name: str = "paid", model: str = "m1", *, fail: bool = False) -> None:
        self.name = name
        self.model = model
        self.fail = fail
        self.requests: list[SpeechRequest] = []

    def synthesize(self, req: SpeechRequest) -> Speech:
        self.requests.append(req)
        if self.fail:
            raise ProviderError("HTTP 500")
        audio = to_wav_bytes(speechlike(len(req.text.split()), lead_ms=150))
        return Speech(
            audio=audio, spans=(), provider=self.name, model=self.model, billed_chars=len(req.text)
        )


def files_reciter() -> Reciter:
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


def reciter_server(seen: list[str], *, missing: bool = False) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if missing or not str(request.url).endswith("/031033.mp3"):
            return httpx.Response(404)
        return httpx.Response(200, content=RECITATION)

    return httpx.Client(transport=httpx.MockTransport(handler))


# -- fixtures and builders ----------------------------------------------------


@pytest.fixture
def channel(channel_file: Path) -> ChannelConfig:
    return load_channel(channel_file)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def pack(tmp_path: Path) -> SfxPack:
    path = tmp_path / "pen.flac"
    aio.encode_flac(tone(200, hz=2500, dbfs=-9), path)
    return SfxPack(
        {"pen": SfxAsset(id="pen", file=path, license="CC0-1.0", source="https://sfx.example/")}
    )


def _strokes() -> StrokeSet:
    return StrokeSet(
        width=100,
        height=100,
        strokes=[Stroke(points=[(0.0, 0.1 * i), (1.0, 0.1 * i)]) for i in range(6)],
    )


def _sketch(library: SketchLibrary, *, strokes: bool = True, key: str = "scroll|roman") -> int:
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


def narr(
    sid: str,
    text: str,
    *,
    sketch: int | None = None,
    tts: str | None = None,
    caption: str | None = None,
) -> Scene:
    visual = Visual() if sketch is not None else Visual(source=VisualSource.TEXT_CARD)
    return Scene(
        id=sid,
        narration=Narration(source_text=text, tts_text=tts, caption_text=caption),
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


def make_series(*parts: list[Scene], approved: bool = True) -> Series:
    return Series(
        title="Scenes",
        channel_slug="history",
        language=Language.ARABIC,
        source=ScriptSource(author="a", work="Scenes", rights=RightsBasis.OWN),
        parts=[
            Part(index=i + 1, count=len(parts), hook_text="HOOK never spoken", scenes=scenes)
            for i, scenes in enumerate(parts)
        ],
        approved=approved,
    )


def run(
    series: Series,
    channel: ChannelConfig,
    settings: Settings,
    tmp_path: Path,
    *,
    voices: list[FakeVoice],
    library: SketchLibrary,
    pack: SfxPack,
    **kw: Any,
) -> tuple[Series, AudioBuildResult]:
    options: dict[str, Any] = {
        "aligner": None,
        "reciters": ReciterIndex([]),
        "corpus": CORPUS,
        "prices": PRICES,
        **kw,
    }
    return build_audio(
        series,
        channel,
        settings,
        out_dir=tmp_path / "out",
        cache_root=tmp_path / "series" / ".cache",
        voices=[VoiceChoice(v, f"v{i}") for i, v in enumerate(voices, start=1)],
        library=library,
        sfx=pack,
        **options,
    )


def cost(*texts: str) -> Decimal:
    return sum((Decimal(len(t)) / 1000 * PER_KCHAR for t in texts), Decimal(0))


def _flac_ms(path: Path) -> float:
    samples: Samples = aio.decode(path.read_bytes())
    return len(samples) * 1000 / aio.SAMPLE_RATE


# -- tests --------------------------------------------------------------------


def test_builds_flac_and_timeline_per_part(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    sid = _sketch(library)
    original = make_series(
        [narr("s1", "alpha beta gamma delta", sketch=sid), narr("s2", "epsilon zeta")],
        [narr("s3", "eta theta iota")],
    )
    voice = FakeVoice()
    out, result = run(
        original, channel, settings, tmp_path, voices=[voice], library=library, pack=pack
    )

    assert [r.text for r in voice.requests] == [
        "alpha beta gamma delta",
        "epsilon zeta",
        "eta theta iota",
    ]
    assert [p.index for p in result.parts] == [1, 2]
    spent = [cost("alpha beta gamma delta", "epsilon zeta"), cost("eta theta iota")]
    assert result.spent_usd == sum(spent)
    assert [p.spent_usd for p in out.parts] == spent
    assert all(p.spent_usd == 0 for p in original.parts)  # the input is not mutated
    for got, expected_spent in zip(result.parts, spent, strict=True):
        n = f"{got.index:02d}"
        assert got.audio_path == tmp_path / "out" / f"part-{n}.flac"
        assert got.timeline_path == tmp_path / "out" / f"part-{n}.timeline.json"
        timeline = PartTimeline.load(got.timeline_path)
        assert timeline == got.timeline
        audio = got.timeline_path.parent / timeline.audio_file
        assert audio == got.audio_path and audio.is_file()
        assert abs(_flac_ms(audio) - timeline.duration_ms) <= 30
        assert timeline.spent_usd == expected_spent
        assert (timeline.series_title, timeline.channel_slug) == ("Scenes", "history")
        assert (timeline.part_index, timeline.part_count) == (got.index, 2)
        assert timeline.voice is not None and timeline.voice.provider == "paid"
        assert timeline.voice.fallback_from is None
        assert timeline.recitations == [] and timeline.cost_estimated is False
    first = result.parts[0].timeline
    assert first.scenes[0].draw_start_ms is not None  # a sketch scene is drawn
    assert first.scenes[1].draw_start_ms is None  # a text card is not
    # which sketch the draw schedule and SFX refer to (Plan 5 checks these)
    assert (first.scenes[0].sketch_id, first.scenes[0].stroke_count) == (sid, 6)
    assert (first.scenes[1].sketch_id, first.scenes[1].stroke_count) == (None, None)
    assert all(h.stroke_index < 6 for h in first.sfx)
    assert first.sfx and {h.scene_id for h in first.sfx} == {"s1"}
    [asset] = first.assets
    assert (asset.kind, asset.asset_id, asset.license) == ("sfx", "pen", "CC0-1.0")
    assert asset.source == "https://sfx.example/"
    assert result.parts[1].timeline.assets == []


def test_verse_part_records_recitation_and_warnings(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    sid = _sketch(library)
    seen: list[str] = []
    voice = FakeVoice()
    s = make_series([verse("v1"), narr("s1", "alpha beta gamma", sketch=sid)])
    out, result = run(
        s,
        channel,
        settings,
        tmp_path,
        voices=[voice],
        library=library,
        pack=pack,
        reciters=ReciterIndex([files_reciter()]),
        http=reciter_server(seen),
    )
    assert seen == ["https://audio.example.org/r/031033.mp3"]
    assert [r.text for r in voice.requests] == ["alpha beta gamma"]  # never the verse
    cached = settings.data_dir / "reciters" / "files-hafs" / "031033.mp3"
    assert cached.is_file()
    [part] = result.parts
    timeline = part.timeline
    [rec] = timeline.recitations
    assert (rec.reciter_id, rec.riwaya, rec.license, rec.ayat) == (
        "files-hafs",
        "hafs",
        "CC-BY-4.0",
        ["31:33"],
    )
    assert rec.license_url and rec.source and rec.segments_source is None
    v1 = timeline.scenes[0]
    assert v1.kind is SceneKind.VERSE and v1.timing_source is TimingSource.AYAH
    assert all(a.words == [] for a in v1.ayat)
    assert "31:33: no word timings; whole ayah revealed" in timeline.warnings
    assert all(h.at_ms >= v1.end_ms for h in timeline.sfx)
    assert out.parts[0].spent_usd == cost("alpha beta gamma")


def test_skeleton_mismatch_refused_before_spend(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    voice = FakeVoice()
    s = make_series([narr("s1", "alpha beta", tts="alpha gamma")])
    with pytest.raises(PlanningError, match="s1"):
        run(s, channel, settings, tmp_path, voices=[voice], library=library, pack=pack)
    assert voice.requests == []
    assert not (tmp_path / "out").exists()


def test_caption_token_count_mismatch_refused_before_spend(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    voice = FakeVoice()
    s = make_series([narr("s1", "alpha beta gamma", caption="alpha beta")])
    with pytest.raises(PlanningError, match="s1"):
        run(s, channel, settings, tmp_path, voices=[voice], library=library, pack=pack)
    assert voice.requests == []


def test_reciter_unavailable_fails_before_tts_spend(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    voice = FakeVoice()
    seen: list[str] = []
    s = make_series([narr("s1", "alpha beta"), verse("v1")])
    with pytest.raises(
        ReciterError,
        match=r"^scene v1 \(31:33\): no allow-listed reciter .*--part N$",
    ):
        run(
            s,
            channel,
            settings,
            tmp_path,
            voices=[voice],
            library=library,
            pack=pack,
            reciters=ReciterIndex([]),
            http=reciter_server(seen),
        )
    assert voice.requests == [] and seen == []

    # the recording itself is missing upstream: the part fails before its TTS
    with pytest.raises(AudioInterruptedError) as info:
        run(
            s,
            channel,
            settings,
            tmp_path,
            voices=[voice],
            library=library,
            pack=pack,
            reciters=ReciterIndex([files_reciter()]),
            http=reciter_server(seen, missing=True),
        )
    assert isinstance(info.value.__cause__, ReciterError)
    assert voice.requests == [] and len(seen) == 1
    partial, so_far = info.value.partial
    assert so_far.parts == [] and so_far.spent_usd == 0
    assert partial.parts[0].spent_usd == 0


def test_recitation_provider_error_never_falls_back_to_tts(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    primary, backup = FakeVoice(), FakeVoice("other", "m2")
    s = make_series([verse("v1"), narr("s1", "alpha beta")])
    with pytest.raises(AudioInterruptedError) as info:
        run(
            s,
            channel,
            settings,
            tmp_path,
            voices=[primary, backup],
            library=library,
            pack=pack,
            reciters=ReciterIndex([files_reciter()]),
            http=httpx.Client(transport=httpx.MockTransport(handler)),
        )
    assert isinstance(info.value.__cause__, ProviderError)
    assert primary.requests == [] and backup.requests == []


def test_unapproved_and_missing_sketch_refused(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    voice = FakeVoice()
    unapproved = make_series([narr("s1", "alpha beta")], approved=False)
    with pytest.raises(PlanningError, match="approve"):
        run(unapproved, channel, settings, tmp_path, voices=[voice], library=library, pack=pack)
    no_sketch = make_series([narr("s1", "alpha beta").model_copy(update={"visual": Visual()})])
    with pytest.raises(PlanningError) as info:
        run(no_sketch, channel, settings, tmp_path, voices=[voice], library=library, pack=pack)
    assert str(info.value) == (
        "scene s1 has no sketch yet; run narratoflow sketch generate / resolve first"
    )
    other = make_series([narr("s1", "alpha beta")]).model_copy(update={"channel_slug": "other"})
    with pytest.raises(PlanningError, match="channel"):
        run(other, channel, settings, tmp_path, voices=[voice], library=library, pack=pack)
    assert voice.requests == []


def test_background_music_refused(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    voice = FakeVoice()
    music = channel.model_copy(update={"background_music": True})
    s = make_series([narr("s1", "alpha beta")])
    with pytest.raises(ChannelConfigError) as info:
        run(s, music, settings, tmp_path, voices=[voice], library=library, pack=pack)
    assert str(info.value) == ("no licensed music pack is bundled; set background_music = false")
    assert voice.requests == []


def test_missing_ffmpeg_refused_before_spend(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> Path:
        raise AudioError("ffmpeg not found")

    monkeypatch.setattr(build_module, "require_ffmpeg", missing)
    voice = FakeVoice()
    with pytest.raises(AudioError, match="ffmpeg"):
        run(
            make_series([narr("s1", "alpha beta")]),
            channel,
            settings,
            tmp_path,
            voices=[voice],
            library=library,
            pack=pack,
        )
    assert voice.requests == []


def test_limit_spans_stages(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    limited = channel.model_copy(update={"spending_limit_usd": Decimal("0.10")})
    s = make_series([narr("s1", "alpha beta")], [narr("s2", "gamma delta")])
    s.parts[1].spent_usd = Decimal("0.099")  # sketch generation spent this already
    voice = FakeVoice()
    with pytest.raises(AudioInterruptedError) as info:
        run(s, limited, settings, tmp_path, voices=[voice], library=library, pack=pack)
    cause = info.value.__cause__
    assert isinstance(cause, BudgetExceededError)
    assert str(info.value) == f"{cause} (progress so far is kept)"
    assert str(cause) == (
        f"voice paid/m1 part 2: estimated cost {format_usd(cost('gamma delta'))} exceeds the "
        "remaining $0.0010 (spent $0.0990 of the $0.10 per-part limit; "
        "raise spending_limit_usd in the channel file)"
    )
    assert [r.text for r in voice.requests] == ["alpha beta"]  # part 2 never voiced
    partial, so_far = info.value.partial
    assert [p.index for p in so_far.parts] == [1]
    assert so_far.spent_usd == cost("alpha beta")
    assert [p.spent_usd for p in partial.parts] == [cost("alpha beta"), Decimal("0.099")]
    assert s.parts[0].spent_usd == 0


def test_spent_written_back_and_cache_rerun_spends_zero(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    s = make_series([narr("s1", "alpha beta gamma")])
    s.parts[0].spent_usd = Decimal("0.50")  # from sketch generation
    voice = FakeVoice()
    out, result = run(s, channel, settings, tmp_path, voices=[voice], library=library, pack=pack)
    paid = cost("alpha beta gamma")
    assert result.spent_usd == paid  # this run's delta
    assert out.parts[0].spent_usd == Decimal("0.50") + paid  # cumulative
    assert result.parts[0].timeline.spent_usd == Decimal("0.50") + paid

    again = FakeVoice()
    out2, rerun = run(out, channel, settings, tmp_path, voices=[again], library=library, pack=pack)
    assert again.requests == []
    assert rerun.spent_usd == 0
    assert out2.parts[0].spent_usd == Decimal("0.50") + paid
    assert PartTimeline.load(rerun.parts[0].timeline_path).spent_usd == Decimal("0.50") + paid


def test_missing_strokes_refused_before_spend(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    voice = FakeVoice()
    bare = _sketch(library, strokes=False)
    s = make_series([narr("s1", "alpha beta", sketch=bare)])
    with pytest.raises(LibraryError) as info:
        run(s, channel, settings, tmp_path, voices=[voice], library=library, pack=pack)
    assert str(info.value) == (f"scene s1: sketch {bare} has no strokes; re-run sketch generate")
    unknown = make_series([narr("s1", "alpha beta", sketch=9999)])
    with pytest.raises(LibraryError):
        run(unknown, channel, settings, tmp_path, voices=[voice], library=library, pack=pack)
    assert voice.requests == []


def test_only_voice_key_required(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    # the history channel generates images with openai, but audio needs no image key
    assert channel.style.image_provider == "openai" and settings.image_api_key is None
    voice = FakeVoice()
    s = make_series([narr("s1", "alpha beta")])
    _, result = run(s, channel, settings, tmp_path, voices=[voice], library=library, pack=pack)
    assert len(result.parts) == 1

    paid = channel.model_copy(
        update={
            "voice": channel.voice.model_copy(
                update={"provider": "elevenlabs", "model": "eleven_multilingual_v2"}
            )
        }
    )
    with pytest.raises(MissingApiKeyError, match="NARRATOFLOW_VOICE_API_KEY") as info:
        build_audio(
            s,
            paid,
            settings,
            out_dir=tmp_path / "out2",
            cache_root=tmp_path / "series" / ".cache",
            library=library,
            sfx=pack,
            prices=PRICES_WITH_ELEVEN,
            reciters=ReciterIndex([]),
            corpus=CORPUS,
            aligner=None,
        )
    assert "IMAGE" not in str(info.value)


class ClosingVoice(FakeVoice):
    def __init__(self) -> None:
        super().__init__()
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


@pytest.mark.parametrize("refused", [False, True])
def test_built_voice_chain_is_closed(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refused: bool,
) -> None:
    voice = ClosingVoice()
    monkeypatch.setattr(
        build_module, "build_voice_chain", lambda channel, settings: [VoiceChoice(voice, "v")]
    )
    scenes = [narr("s1", "alpha beta"), verse("v1")] if refused else [narr("s1", "alpha beta")]
    kwargs: dict[str, Any] = {
        "out_dir": tmp_path / "out",
        "cache_root": tmp_path / "series" / ".cache",
        "library": library,
        "sfx": pack,
        "prices": PRICES,
        "reciters": ReciterIndex([]),
        "corpus": CORPUS,
        "aligner": None,
    }
    if refused:  # a pre-spend refusal after the chain is built still closes it
        with pytest.raises(ReciterError):
            build_audio(make_series(scenes), channel, settings, **kwargs)
    else:
        build_audio(make_series(scenes), channel, settings, **kwargs)
        assert len(voice.requests) == 1
    assert voice.closed == 1


def test_injected_voices_are_not_closed(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    voice = ClosingVoice()
    run(
        make_series([narr("s1", "alpha beta")]),
        channel,
        settings,
        tmp_path,
        voices=[voice],
        library=library,
        pack=pack,
    )
    assert voice.closed == 0  # the caller owns injected providers


def test_unpriced_voice_refused_before_spend(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    voice = FakeVoice("unpriced", "m9")
    with pytest.raises(PlanningError, match="unpriced/m9"):
        run(
            make_series([narr("s1", "alpha beta")]),
            channel,
            settings,
            tmp_path,
            voices=[voice],
            library=library,
            pack=pack,
        )
    assert voice.requests == []


def test_part_selection_builds_only_that_part(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    voice = FakeVoice()
    s = make_series([narr("s1", "alpha beta")], [narr("s2", "gamma delta")])
    out, result = run(
        s, channel, settings, tmp_path, voices=[voice], library=library, pack=pack, parts=[2]
    )
    assert [r.text for r in voice.requests] == ["gamma delta"]
    assert [p.index for p in result.parts] == [2]
    assert not (tmp_path / "out" / "part-01.flac").exists()
    assert (tmp_path / "out" / "part-02.flac").is_file()
    assert [p.spent_usd for p in out.parts] == [Decimal(0), cost("gamma delta")]

    with pytest.raises(PlanningError, match="3"):
        run(s, channel, settings, tmp_path, voices=[voice], library=library, pack=pack, parts=[3])
    assert len(voice.requests) == 1


def test_selected_parts_only_are_checked(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    voice = FakeVoice()
    unsketched = narr("s1", "alpha beta").model_copy(update={"visual": Visual()})
    s = make_series([unsketched], [narr("s2", "gamma delta")])
    _, result = run(
        s, channel, settings, tmp_path, voices=[voice], library=library, pack=pack, parts=[2]
    )
    assert [p.index for p in result.parts] == [2]


def test_fallback_recorded_in_timeline(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    primary, backup = FakeVoice(fail=True), FakeVoice("other", "m2")
    s = make_series([narr("s1", "alpha beta")])
    _, result = run(
        s, channel, settings, tmp_path, voices=[primary, backup], library=library, pack=pack
    )
    voice = result.parts[0].timeline.voice
    assert voice is not None
    assert (voice.provider, voice.model, voice.voice_id) == ("other", "m2", "v2")
    assert voice.fallback_from == "paid/v1"
    assert voice.fallback_reasons == ["paid/v1: HTTP 500"]
    assert len(primary.requests) == 1 and len(backup.requests) == 1


def test_auto_aligner_missing_warns_once_per_part(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ctc.CtcAligner, "available", staticmethod(lambda: False))
    s = make_series([narr("s1", "alpha beta"), narr("s2", "gamma")], [narr("s3", "delta")])
    _, result = run(
        s,
        channel,
        settings,
        tmp_path,
        voices=[FakeVoice()],
        library=library,
        pack=pack,
        aligner="auto",
    )
    for part in result.parts:
        assert part.timeline.warnings.count(ALIGN_EXTRA_WARNING) == 1


def test_unmeasurable_loudness_is_none_with_warnings(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_master(samples: Samples, **_: float) -> tuple[Samples, Loudness]:
        return samples, Loudness(None, None, 0.0, 0.0, 0.0)

    monkeypatch.setattr(mixer, "master", fake_master)
    s = make_series([narr("s1", "alpha beta gamma")])
    _, result = run(
        s, channel, settings, tmp_path, voices=[FakeVoice()], library=library, pack=pack
    )
    timeline = PartTimeline.load(result.parts[0].timeline_path)
    assert timeline.integrated_lufs is None and timeline.true_peak_dbtp is None
    assert "loudness not measurable; master not normalized" in timeline.warnings
    assert "true peak not measurable" in timeline.warnings


def test_sources_are_ascii() -> None:
    for path in (Path(build_module.__file__), Path(__file__)):
        path.read_bytes().decode("ascii")


def test_auto_aligner_unavailable_at_run_time_warns_once(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from narratoflow.errors import AlignerUnavailableError

    class BrokenAligner:
        made = 0

        def __init__(self) -> None:
            BrokenAligner.made += 1

        @staticmethod
        def available() -> bool:
            return True

        def align(self, *_: object) -> list[object]:
            raise AlignerUnavailableError("install the 'align' extra")

    monkeypatch.setattr(build_module, "CtcAligner", BrokenAligner)
    s = make_series([narr("s1", "alpha beta"), narr("s2", "gamma")])
    _, result = run(
        s,
        channel,
        settings,
        tmp_path,
        voices=[FakeVoice()],
        library=library,
        pack=pack,
        aligner="auto",
    )
    assert BrokenAligner.made == 1
    assert result.parts[0].timeline.warnings.count(ALIGN_EXTRA_WARNING) == 1


def test_default_library_is_opened_for_sketched_scenes(
    channel: ChannelConfig,
    settings: Settings,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    voice = FakeVoice()
    s = make_series([narr("s1", "alpha beta", sketch=1)])
    with pytest.raises(LibraryError, match="unknown sketch id"):
        build_audio(
            s,
            channel,
            settings,
            out_dir=tmp_path / "out",
            cache_root=tmp_path / ".cache",
            voices=[VoiceChoice(voice, "v1")],
            aligner=None,
            sfx=pack,
            prices=PRICES,
        )
    assert voice.requests == [] and (settings.data_dir / "library.db").is_file()


def test_verse_scene_without_reference_refused(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    broken = verse("v1").model_copy(update={"verse": None})
    with pytest.raises(PlanningError, match="v1"):
        run(
            make_series([broken]),
            channel,
            settings,
            tmp_path,
            voices=[FakeVoice()],
            library=library,
            pack=pack,
            reciters=ReciterIndex([files_reciter()]),
        )


def test_empty_voice_chain_refused(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    with pytest.raises(ChannelConfigError, match="no voice"):
        run(
            make_series([narr("s1", "alpha beta")]),
            channel,
            settings,
            tmp_path,
            voices=[],
            library=library,
            pack=pack,
        )


def test_cached_part_is_not_refused_when_the_limit_was_lowered(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    s = make_series([narr("s1", "alpha beta gamma")])
    out, first = run(
        s, channel, settings, tmp_path, voices=[FakeVoice()], library=library, pack=pack
    )
    assert first.spent_usd > 0
    lowered = channel.model_copy(update={"spending_limit_usd": Decimal("0.0001")})
    again = FakeVoice()
    _, rerun = run(out, lowered, settings, tmp_path, voices=[again], library=library, pack=pack)
    assert again.requests == [] and rerun.spent_usd == 0


def test_timeline_records_voice_terms_and_reciter_check_date(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    sid = _sketch(library)
    s = make_series([verse("v1"), narr("s1", "alpha beta gamma", sketch=sid)])
    _, result = run(
        s,
        channel,
        settings,
        tmp_path,
        voices=[FakeVoice()],
        library=library,
        pack=pack,
        reciters=ReciterIndex([files_reciter()]),
        http=reciter_server([]),
    )
    timeline = PartTimeline.load(result.parts[0].timeline_path)
    assert timeline.voice is not None and timeline.voice.terms == "paid-terms"
    assert timeline.recitations[0].checked == date(2026, 11, 11)


def test_ayat_label_for_single_ayah_and_range() -> None:
    one = VerseRef(surah=31, ayah_start=33, ayah_end=33, certain=True, uthmani_text="x")
    many = VerseRef(surah=2, ayah_start=255, ayah_end=257, certain=True, uthmani_text="x")
    assert build_module._ayat(one) == "31:33"
    assert build_module._ayat(many) == "2:255-257"


def test_reciter_refusal_names_the_first_verse_scene(
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    pack: SfxPack,
    tmp_path: Path,
) -> None:
    s = make_series([narr("s1", "alpha beta")], [verse("v2"), verse("v3")])
    with pytest.raises(ReciterError, match=r"^scene v2 \(31:33\): "):
        run(s, channel, settings, tmp_path, voices=[FakeVoice()], library=library, pack=pack)
