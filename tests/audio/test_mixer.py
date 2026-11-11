"""mix_part: scene slots, loudness matching, verse buffers, stroke SFX and the master."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from fixtures.audio import speechlike, tone

from narratoflow.audio import io as aio
from narratoflow.audio import mixer
from narratoflow.audio.io import Loudness, Samples
from narratoflow.audio.mixer import (
    FADE_MS,
    FIRST_ONSET_MAX_MS,
    INTER_SCENE_GAP_MS,
    MATCH_TARGET_LUFS,
    MAX_MATCH_GAIN_DB,
    VERSE_BUFFER_MS,
    MixResult,
    mix_part,
)
from narratoflow.audio.narrate import NarrationClip, NarrationResult
from narratoflow.audio.recitation import AYAH_GAP_MS, VerseClip
from narratoflow.audio.sfx import SFX_GAIN_DB, SFX_MIN_SPACING_MS, SfxAsset, SfxPack
from narratoflow.audio.timeline import AyahTiming, PartTimeline, TimelineWord, TimingSource
from narratoflow.audio.words import estimate_timings
from narratoflow.errors import AudioError
from narratoflow.models import Narration, Part, Scene, SceneKind, VerseRef, WordTiming
from narratoflow.visuals.strokes import Stroke, StrokeSet

pytestmark = pytest.mark.usefixtures("ffmpeg_bin")

TEXT = "one two three four"
VIDEO = "chan/Series"


# -- builders -----------------------------------------------------------------


def _ms(samples: Samples) -> int:
    return math.ceil(len(samples) * 1000 / aio.SAMPLE_RATE)


def _scene(
    sid: str,
    kind: SceneKind = SceneKind.NARRATION,
    *,
    min_ms: int = 0,
    draw_fraction: float = 0.65,
) -> Scene:
    verse = None
    if kind is SceneKind.VERSE:
        verse = VerseRef(surah=31, ayah_start=33, ayah_end=33, certain=True, uthmani_text="x")
    return Scene(
        id=sid,
        narration=Narration(source_text=TEXT),
        kind=kind,
        verse=verse,
        min_duration_ms=min_ms,
        draw_fraction=draw_fraction,
    )


def _part(*scenes: Scene) -> Part:
    return Part(index=1, count=1, hook_text="hook", scenes=list(scenes))


def _clip(samples: Samples, lead_ms: int = 0) -> NarrationClip:
    words = estimate_timings(TEXT.split(), lead_ms, _ms(samples))
    return NarrationClip(samples, words, TimingSource.ESTIMATED)


def _narr(clips: dict[str, NarrationClip], warnings: list[str] | None = None) -> NarrationResult:
    return NarrationResult(clips=clips, voice=None, cost_estimated=False, warnings=warnings or [])


def _verse(samples: Samples, lead_ms: int = 0, *, words: bool = True) -> VerseClip:
    end = _ms(samples)
    timed = (
        [TimelineWord.model_validate(w) for w in estimate_timings(["a", "b"], lead_ms, end)]
        if words
        else []
    )
    ayah = AyahTiming(surah=31, ayah=33, start_ms=lead_ms, end_ms=end, words=timed)
    source = TimingSource.SEGMENTS if words else TimingSource.AYAH
    return VerseClip(samples=samples, ayat=[ayah], timing_source=source)


def _strokes(count: int = 6) -> StrokeSet:
    return StrokeSet(
        width=100,
        height=100,
        strokes=[Stroke(points=[(0.0, 0.1 * (i % 9)), (1.0, 0.1 * (i % 9))]) for i in range(count)],
    )


@pytest.fixture
def pack(tmp_path: Path) -> SfxPack:
    assets: dict[str, SfxAsset] = {}
    for sid, hz in (("pen-a", 2000.0), ("pen-b", 3000.0)):
        path = tmp_path / f"{sid}.flac"
        aio.encode_flac(tone(200, hz=hz, dbfs=-9), path)  # -6 dBFS peak
        assets[sid] = SfxAsset(id=sid, file=path, license="CC0-1.0", source="test")
    return SfxPack(assets)


def _rms_windows(samples: Samples) -> np.ndarray:
    win = aio.ms_to_samples(10)
    count = len(samples) // win
    blocks = samples[: count * win].astype(np.float64).reshape(count, win)
    return np.asarray(np.sqrt(np.mean(blocks**2, axis=1)))


def _loud(samples: Samples) -> np.ndarray:
    return np.asarray(_rms_windows(samples) > 10 ** (-45 / 20))


def _onset_ms(samples: Samples) -> int:
    return int(np.flatnonzero(_loud(samples))[0]) * 10


def _slice(samples: Samples, start_ms: int, end_ms: int) -> Samples:
    return samples[aio.ms_to_samples(start_ms) : aio.ms_to_samples(end_ms)]


def _timeline(res: MixResult) -> PartTimeline:
    return PartTimeline(
        series_title="S",
        channel_slug="chan",
        part_index=1,
        part_count=1,
        audio_file="part-01.flac",
        duration_ms=res.scenes[-1].end_ms,
        integrated_lufs=res.loudness.integrated_lufs,
        true_peak_dbtp=res.loudness.true_peak_dbtp,
        scenes=res.scenes,
        sfx=res.sfx,
        voice=None,
        recitations=[],
        assets=[],
        spent_usd=Decimal(0),
        cost_estimated=False,
        warnings=res.warnings,
    )


def _identity_master(loud: Loudness) -> Callable[..., tuple[Samples, Loudness]]:
    def fake(samples: Samples, **_: float) -> tuple[Samples, Loudness]:
        return samples, loud

    return fake


ON_TARGET = Loudness(-14.0, -2.0, 5.0, -24.0, 0.0)


# -- tests --------------------------------------------------------------------


def test_constants() -> None:
    assert (VERSE_BUFFER_MS, INTER_SCENE_GAP_MS, FIRST_ONSET_MAX_MS, FADE_MS) == (400, 250, 100, 10)
    assert (MATCH_TARGET_LUFS, MAX_MATCH_GAIN_DB, mixer.LOUDNESS_TOLERANCE_LU) == (-18.0, 12.0, 1.0)


def test_first_onset_within_100ms_despite_lead_silence(pack: SfxPack) -> None:
    tts = speechlike(6, lead_ms=300)
    res = mix_part(
        _part(_scene("n1")), _narr({"n1": _clip(tts, 300)}), {}, {}, pack, video_id=VIDEO
    )
    assert _onset_ms(res.samples) <= FIRST_ONSET_MAX_MS
    s = res.scenes[0]
    assert s.start_ms == s.speech_start_ms == 0
    assert s.words[0].start_ms <= 20  # shifted by the 300 ms lead

    recited = speechlike(6, lead_ms=700, seed=3)
    res = mix_part(
        _part(_scene("v1", SceneKind.VERSE), _scene("n1")),
        _narr({"n1": _clip(speechlike(4, seed=1))}),
        {"v1": _verse(recited, 700)},
        {},
        pack,
        video_id=VIDEO,
    )
    assert _onset_ms(res.samples) <= FIRST_ONSET_MAX_MS
    verse = res.scenes[0]
    assert verse.speech_start_ms == 0
    assert verse.ayat[0].start_ms <= 10  # shifted by the re-trimmed 700 ms lead
    assert verse.ayat[0].words[0].start_ms <= 10
    assert verse.ayat[-1].end_ms <= verse.speech_end_ms


def test_verse_first_part_has_no_sfx_before_verse_end(pack: SfxPack) -> None:
    res = mix_part(
        _part(_scene("v1", SceneKind.VERSE), _scene("n1", draw_fraction=1.0)),
        _narr({"n1": _clip(speechlike(10))}),
        {"v1": _verse(speechlike(5, seed=2))},
        {"v1": _strokes(), "n1": _strokes()},
        pack,
        video_id=VIDEO,
    )
    assert res.sfx
    verse_end = res.scenes[0].end_ms
    assert all(h.at_ms >= verse_end for h in res.sfx)
    assert res.scenes[0].draw_start_ms is None


def test_buffers_only_at_narration_verse_boundaries(pack: SfxPack) -> None:
    res = mix_part(
        _part(_scene("n1"), _scene("v1", SceneKind.VERSE), _scene("n2")),
        _narr({"n1": _clip(speechlike(5)), "n2": _clip(speechlike(5, seed=4), 200)}),
        {"v1": _verse(speechlike(5, seed=2))},
        {},
        pack,
        video_id=VIDEO,
    )
    n1, v1, n2 = res.scenes
    assert n1.start_ms == n1.speech_start_ms == 0  # no buffer at the part start
    assert n1.end_ms == n1.speech_end_ms  # no inter-scene gap before a verse
    assert v1.speech_start_ms - v1.start_ms == VERSE_BUFFER_MS
    assert v1.end_ms - v1.speech_end_ms == VERSE_BUFFER_MS
    assert n2.speech_start_ms == n2.start_ms  # none on narration after a verse
    assert n2.end_ms == n2.speech_end_ms + INTER_SCENE_GAP_MS
    loud = _loud(res.samples)
    last_n1 = int(np.flatnonzero(loud[: n1.end_ms // 10])[-1])
    first_v1 = v1.start_ms // 10 + int(np.flatnonzero(loud[v1.start_ms // 10 :])[0])
    assert (first_v1 - last_n1 - 1) * 10 <= 420


def test_consecutive_verse_scenes_join_with_ayah_gap(pack: SfxPack) -> None:
    res = mix_part(
        _part(_scene("v1", SceneKind.VERSE), _scene("v2", SceneKind.VERSE)),
        _narr({}),
        {"v1": _verse(speechlike(4, seed=2)), "v2": _verse(speechlike(4, seed=5))},
        {},
        pack,
        video_id=VIDEO,
    )
    v1, v2 = res.scenes
    assert v1.speech_start_ms == 0
    assert v1.end_ms - v1.speech_end_ms == AYAH_GAP_MS
    assert v2.speech_start_ms == v2.start_ms == v1.end_ms
    assert v2.end_ms == v2.speech_end_ms
    _timeline(res)


def test_ayah_timed_verse_words_are_cleared(pack: SfxPack) -> None:
    clip = _verse(speechlike(4, seed=2))
    clip.timing_source = TimingSource.AYAH  # partial segments: words present, ayah-timed
    res = mix_part(
        _part(_scene("v1", SceneKind.VERSE)), _narr({}), {"v1": clip}, {}, pack, video_id=VIDEO
    )
    assert res.scenes[0].timing_source is TimingSource.AYAH
    assert all(not a.words for a in res.scenes[0].ayat)


def test_words_clamped_inside_speech_span(pack: SfxPack) -> None:
    samples = speechlike(4, lead_ms=300)
    over = _ms(samples) + 900
    words = [
        WordTiming(text="one", start_ms=0, end_ms=350),
        WordTiming(text="two", start_ms=700, end_ms=900),
        WordTiming(text="three", start_ms=900, end_ms=over),
        WordTiming(text="four", start_ms=over, end_ms=over + 100),
    ]
    clip = NarrationClip(samples, words, TimingSource.PROVIDER)
    res = mix_part(
        _part(_scene("n0"), _scene("n1")),
        _narr({"n0": _clip(speechlike(3, seed=9)), "n1": clip}),
        {},
        {},
        pack,
        video_id=VIDEO,
    )
    s = res.scenes[1]
    assert s.timing_source is TimingSource.PROVIDER
    assert [w.text for w in s.words] == ["one", "two", "three", "four"]
    for w in s.words:
        assert s.speech_start_ms <= w.start_ms <= w.end_ms <= s.speech_end_ms
    assert s.words[0].start_ms == s.speech_start_ms
    assert s.words[-1].end_ms == s.speech_end_ms


def test_no_sfx_inside_verse_slots(pack: SfxPack) -> None:
    res = mix_part(
        _part(
            _scene("n1", draw_fraction=1.0),
            _scene("v1", SceneKind.VERSE),
            _scene("n2", draw_fraction=1.0),
        ),
        _narr({"n1": _clip(speechlike(8)), "n2": _clip(speechlike(8, seed=4))}),
        {"v1": _verse(speechlike(5, seed=2))},
        {"n1": _strokes(10), "n2": _strokes(10)},
        pack,
        video_id=VIDEO,
    )
    v1 = res.scenes[1]
    assert {h.scene_id for h in res.sfx} == {"n1", "n2"}
    for h in res.sfx:
        assert not (h.at_ms < v1.end_ms and h.at_ms + h.duration_ms > v1.start_ms)
    _timeline(res)


def test_slot_extended_to_min_duration(pack: SfxPack) -> None:
    res = mix_part(
        _part(_scene("n1", min_ms=5000), _scene("n2")),
        _narr({"n1": _clip(speechlike(2)), "n2": _clip(speechlike(2, seed=1))}),
        {},
        {},
        pack,
        video_id=VIDEO,
    )
    n1, n2 = res.scenes
    assert n1.speech_end_ms + INTER_SCENE_GAP_MS < 5000
    assert n1.end_ms - n1.start_ms == 5000
    assert n2.end_ms == n2.speech_end_ms + INTER_SCENE_GAP_MS
    assert len(res.samples) == aio.ms_to_samples(n2.end_ms)


def test_draw_window_uses_draw_fraction(pack: SfxPack) -> None:
    res = mix_part(
        _part(_scene("n1", draw_fraction=0.5), _scene("n2", draw_fraction=0.65), _scene("n3")),
        _narr({sid: _clip(speechlike(4, seed=i)) for i, sid in enumerate(("n1", "n2", "n3"))}),
        {},
        {"n1": _strokes(), "n2": _strokes()},
        pack,
        video_id=VIDEO,
    )
    for s, frac in zip(res.scenes[:2], (0.5, 0.65), strict=True):
        assert s.draw_start_ms == s.start_ms
        assert s.draw_end_ms == s.start_ms + round(frac * (s.end_ms - s.start_ms))
    assert res.scenes[1].start_ms > 0
    assert res.scenes[2].draw_start_ms is None  # text card: no strokes
    assert res.scenes[2].draw_end_ms is None
    assert all(h.scene_id != "n3" for h in res.sfx)


def _at_lufs(samples: Samples, lufs: float) -> Samples:
    got = aio.measure(samples).integrated_lufs
    assert got is not None
    return aio.apply_gain(samples, lufs - got)


def test_loudness_matched_between_quiet_recitation_and_loud_narration(pack: SfxPack) -> None:
    loud_n = _at_lufs(speechlike(12), -12.0)
    quiet_v = _at_lufs(speechlike(12, seed=7), -30.0)
    res = mix_part(
        _part(_scene("n1"), _scene("v1", SceneKind.VERSE)),
        _narr({"n1": _clip(loud_n)}),
        {"v1": _verse(quiet_v)},
        {},
        pack,
        video_id=VIDEO,
    )
    n1, v1 = res.scenes
    a = aio.measure(_slice(res.samples, n1.speech_start_ms, n1.speech_end_ms)).integrated_lufs
    b = aio.measure(_slice(res.samples, v1.speech_start_ms, v1.speech_end_ms)).integrated_lufs
    assert a is not None
    assert b is not None
    assert abs(a - b) <= 1.5
    assert res.loudness.integrated_lufs is not None
    assert abs(res.loudness.integrated_lufs + 14) <= 1.0


def test_short_part_true_peak_and_warning(pack: SfxPack) -> None:
    res = mix_part(
        _part(_scene("n1", draw_fraction=1.0)),
        _narr({"n1": _clip(speechlike(4, word_ms=300, dbfs=-6))}),
        {},
        {"n1": _strokes(12)},
        pack,
        video_id=VIDEO,
    )
    assert res.scenes[0].end_ms <= 1800
    assert res.sfx
    for a, b in itertools.pairwise(res.sfx):
        assert b.at_ms - a.at_ms >= SFX_MIN_SPACING_MS
    assert float(np.abs(res.samples).max()) <= 10 ** (-1 / 20) + 1e-3
    loud = res.loudness
    assert loud.true_peak_dbtp is not None
    assert loud.true_peak_dbtp <= -1.0 + 0.1
    assert loud.integrated_lufs is not None
    off = abs(loud.integrated_lufs + 14) > 1.0
    assert off == any(w.startswith("loudness off target") for w in res.warnings)
    _timeline(res)


def test_off_target_and_true_peak_warnings(pack: SfxPack, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mixer, "master", _identity_master(Loudness(-17.25, -0.4, 5, -24, 0)))
    res = mix_part(
        _part(_scene("n1")),
        _narr({"n1": _clip(speechlike(4))}, warnings=["n1: word timings estimated"]),
        {},
        {},
        pack,
        video_id=VIDEO,
    )
    assert res.warnings == [
        "n1: word timings estimated",
        "loudness off target: -17.2 LUFS",
        "true peak above -1.0 dBTP: -0.4 dBTP",
    ]


def test_unmeasurable_narration_group_not_matched(pack: SfxPack) -> None:
    # 250 ms of speech is too short to gate, so the narration group gets 0 dB.
    res = mix_part(
        _part(_scene("n1")), _narr({"n1": _clip(tone(250))}), {}, {}, pack, video_id=VIDEO
    )
    assert "narration: loudness not measurable; not matched" in res.warnings


def test_unmeasurable_master_records_warning(
    pack: SfxPack, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mixer, "master", _identity_master(Loudness(None, None, 0, 0, 0)))
    res = mix_part(
        _part(_scene("n1")), _narr({"n1": _clip(speechlike(4))}), {}, {}, pack, video_id=VIDEO
    )
    assert res.loudness.integrated_lufs is None
    assert res.warnings == ["loudness not measurable; master not normalized"]
    assert bool(np.all(np.isfinite(res.samples)))


def test_unmeasurable_verse_group_named_by_scene(
    pack: SfxPack, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mixer, "master", _identity_master(ON_TARGET))
    res = mix_part(
        _part(_scene("n1"), _scene("v1", SceneKind.VERSE)),
        _narr({"n1": _clip(speechlike(6))}),
        {"v1": _verse(tone(250))},
        {},
        pack,
        video_id=VIDEO,
    )
    assert res.warnings == ["v1: loudness not measurable; not matched"]


def test_match_gain_clamped(pack: SfxPack, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mixer, "master", _identity_master(ON_TARGET))
    quiet = _at_lufs(speechlike(10), -40.0)  # needs +22 dB; clamped to +12
    res = mix_part(_part(_scene("n1")), _narr({"n1": _clip(quiet)}), {}, {}, pack, video_id=VIDEO)
    got = aio.measure(_slice(res.samples, 0, res.scenes[0].speech_end_ms)).integrated_lufs
    assert got is not None
    assert abs(got - (-40.0 + MAX_MATCH_GAIN_DB)) <= 0.5


def test_sfx_hits_cut_faded_and_gained(pack: SfxPack, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mixer, "master", _identity_master(ON_TARGET))
    gappy = np.concatenate([tone(300, dbfs=-18), aio.silence(4000), tone(300, dbfs=-18)])
    res = mix_part(
        _part(_scene("n1", draw_fraction=1.0)),
        _narr({"n1": _clip(gappy)}),
        {},
        {"n1": _strokes(8)},
        pack,
        video_id=VIDEO,
    )
    hits = [h for h in res.sfx if h.at_ms >= 500 and h.at_ms + h.duration_ms <= 4200]
    assert hits
    expected_peak = 10 ** (-6 / 20) * 10 ** (SFX_GAIN_DB / 20)
    for h in hits:
        seg = _slice(res.samples, h.at_ms, h.at_ms + h.duration_ms)
        assert abs(float(np.abs(seg).max()) - expected_peak) <= 0.02
        assert abs(float(seg[-1])) <= 1e-3  # faded out
        after = _slice(res.samples, h.at_ms + h.duration_ms, h.at_ms + h.duration_ms + 5)
        assert float(np.abs(after).max(initial=0.0)) <= 1e-6  # cut to duration_ms


def test_mix_deterministic(pack: SfxPack) -> None:
    def run() -> MixResult:
        return mix_part(
            _part(_scene("n1", draw_fraction=1.0), _scene("v1", SceneKind.VERSE), _scene("n2")),
            _narr({"n1": _clip(speechlike(8)), "n2": _clip(speechlike(5, seed=4))}),
            {"v1": _verse(speechlike(5, seed=2))},
            {"n1": _strokes(10), "n2": _strokes(5)},
            pack,
            video_id=VIDEO,
        )

    a, b = run(), run()
    assert np.array_equal(a.samples, b.samples)
    assert a.sfx == b.sfx
    assert a.scenes == b.scenes
    assert a.loudness == b.loudness


def test_scene_timings_validate_as_parttimeline(pack: SfxPack) -> None:
    res = mix_part(
        _part(
            _scene("n1"),
            _scene("v1", SceneKind.VERSE),
            _scene("v2", SceneKind.VERSE),
            _scene("n2", min_ms=4000),
        ),
        _narr({"n1": _clip(speechlike(8), 120), "n2": _clip(speechlike(5, seed=4), 90)}),
        {
            "v1": _verse(speechlike(5, seed=2), 20),
            "v2": _verse(speechlike(3, seed=6), 20, words=False),
        },
        {"n1": _strokes(10), "n2": _strokes(5)},
        pack,
        video_id=VIDEO,
    )
    timeline = _timeline(res)
    assert timeline.duration_ms * aio.SAMPLE_RATE // 1000 == len(res.samples)
    assert res.samples.dtype == np.float32
    assert PartTimeline.model_validate_json(timeline.to_json()) == timeline


def test_dead_air_guard(pack: SfxPack, monkeypatch: pytest.MonkeyPatch) -> None:
    def late(samples: Samples, **_: float) -> tuple[Samples, Loudness]:
        out = np.zeros_like(samples)
        out[aio.ms_to_samples(300) :] = samples[: len(samples) - aio.ms_to_samples(300)]
        return out, ON_TARGET

    monkeypatch.setattr(mixer, "master", late)
    with pytest.raises(AudioError, match="dead air at part start"):
        mix_part(
            _part(_scene("n1")), _narr({"n1": _clip(speechlike(4))}), {}, {}, pack, video_id=VIDEO
        )


@pytest.mark.parametrize(
    ("scenes", "clips", "verses", "needle"),
    [
        ([], {}, {}, "no scenes"),
        (["n1"], {}, {}, "n1"),
        (["v1"], {}, {}, "v1"),
    ],
)
def test_missing_inputs_raise(
    pack: SfxPack,
    scenes: list[str],
    clips: dict[str, NarrationClip],
    verses: dict[str, VerseClip],
    needle: str,
) -> None:
    built = [
        _scene(s, SceneKind.VERSE if s.startswith("v") else SceneKind.NARRATION) for s in scenes
    ]
    with pytest.raises(AudioError, match=needle):
        mix_part(_part(*built), _narr(clips), verses, {}, pack, video_id=VIDEO)


def test_silent_clips_raise(pack: SfxPack) -> None:
    quiet = _clip(aio.silence(500))
    with pytest.raises(AudioError, match="n1: narration clip is all silence"):
        mix_part(_part(_scene("n1")), _narr({"n1": quiet}), {}, {}, pack, video_id=VIDEO)
    silent_verse = _verse(aio.silence(500), words=False)
    with pytest.raises(AudioError, match="v1: verse clip is all silence"):
        mix_part(
            _part(_scene("v1", SceneKind.VERSE)),
            _narr({}),
            {"v1": silent_verse},
            {},
            pack,
            video_id=VIDEO,
        )


@pytest.mark.parametrize(("breath_db", "speech_db"), [(-44, -12), (-43, -12), (-42, -8), (-40, -6)])
@pytest.mark.parametrize("kind", [SceneKind.NARRATION, SceneKind.VERSE])
def test_soft_lead_in_before_loud_first_clip_is_trimmed(
    pack: SfxPack, breath_db: float, speech_db: float, kind: SceneKind
) -> None:
    rng = np.random.default_rng(1)
    breath = (rng.standard_normal(aio.ms_to_samples(200)) * 10 ** (breath_db / 20)).astype(
        np.float32
    )
    samples = np.concatenate([breath, speechlike(10, dbfs=speech_db)]).astype(np.float32)
    if kind is SceneKind.NARRATION:
        res = mix_part(
            _part(_scene("n1")), _narr({"n1": _clip(samples)}), {}, {}, pack, video_id=VIDEO
        )
    else:
        res = mix_part(
            _part(_scene("v1", SceneKind.VERSE)),
            _narr({}),
            {"v1": _verse(samples, 200)},
            {},
            pack,
            video_id=VIDEO,
        )
    assert _onset_ms(res.samples) <= FIRST_ONSET_MAX_MS
    _timeline(res)


def test_first_trim_threshold_has_master_headroom() -> None:
    assert mixer.FIRST_TRIM_DBFS == -46.0
    assert mixer.FIRST_TRIM_DBFS > mixer.ONSET_THRESHOLD_DBFS - (
        mixer.MASTER_TARGET_LUFS - MATCH_TARGET_LUFS
    )
