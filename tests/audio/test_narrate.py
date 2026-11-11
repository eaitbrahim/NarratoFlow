"""narrate_part: scene TTS with the per-project cache, spend, voice fallback and timings."""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
from fixtures.audio import speechlike, to_wav_bytes

from narratoflow.audio.io import SAMPLE_RATE, Samples, trim_silence
from narratoflow.audio.narrate import (
    ADAPTER_VERSION,
    ALIGN_EXTRA_WARNING,
    NarrationResult,
    narrate_part,
    uncached_chars,
)
from narratoflow.audio.timeline import TimingSource, VoiceUsed
from narratoflow.audio.words import TimedSpan
from narratoflow.costs.prices import PriceTable
from narratoflow.costs.tracker import SpendTracker
from narratoflow.errors import (
    AlignerUnavailableError,
    AudioError,
    BilledProviderError,
    BudgetExceededError,
    ChannelConfigError,
    ExtraMissingError,
    MissingApiKeyError,
    PlanningError,
    ProviderError,
)
from narratoflow.models import Language, Narration, Part, Scene, SceneKind, VerseRef
from narratoflow.pipeline.cache import StageCache, cache_key
from narratoflow.tts import Speech, SpeechRequest, VoiceChoice
from narratoflow.tts.piper import PiperProvider

pytestmark = pytest.mark.usefixtures("ffmpeg_bin")

PRICES = PriceTable({"voice": {"paid/m1": (Decimal("0.30"),), "other/m2": (Decimal("0.10"),)}})
WORD_MS = 300
GAP_MS = 80
LEAD_MS = 200


def word_spans(text: str, lead_ms: int = LEAD_MS) -> tuple[TimedSpan, ...]:
    out = []
    t = lead_ms
    for w in text.split():
        out.append(TimedSpan(w, t, t + WORD_MS))
        t += WORD_MS + GAP_MS
    return tuple(out)


class FakeVoice:
    def __init__(
        self,
        name: str = "paid",
        model: str = "m1",
        *,
        spans: bool = True,
        fail_on: str | None = None,
        error: Exception | None = None,
        billed: bool = True,
    ) -> None:
        self.name = name
        self.model = model
        self.spans = spans
        self.fail_on = fail_on
        self.error = error if error is not None else ProviderError("HTTP 500")
        self.billed = billed
        self.requests: list[SpeechRequest] = []

    def synthesize(self, req: SpeechRequest) -> Speech:
        self.requests.append(req)
        if self.fail_on is not None and self.fail_on in req.text:
            raise self.error
        n = len(req.text.split())
        audio = to_wav_bytes(speechlike(n, WORD_MS, GAP_MS, lead_ms=LEAD_MS))
        return Speech(
            audio=audio,
            spans=word_spans(req.text) if self.spans else (),
            provider=self.name,
            model=self.model,
            billed_chars=len(req.text) if self.billed else None,
        )


class FakeAligner:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[int, int, str, Language]] = []

    def align(
        self, samples: Samples, sample_rate: int, text: str, language: Language
    ) -> list[TimedSpan]:
        self.calls.append((len(samples), sample_rate, text, language))
        if self.error is not None:
            raise self.error
        return list(word_spans(text, lead_ms=LEAD_MS + 10))


def narration(sid: str, text: str, tts: str | None = None, caption: str | None = None) -> Scene:
    return Scene(id=sid, narration=Narration(source_text=text, tts_text=tts, caption_text=caption))


def verse(sid: str) -> Scene:
    return Scene(
        id=sid,
        narration=Narration(source_text="verse words here"),
        kind=SceneKind.VERSE,
        verse=VerseRef(surah=31, ayah_start=33, ayah_end=33, certain=True, uthmani_text="x"),
    )


def make_part(*scenes: Scene) -> Part:
    return Part(
        index=2,
        count=3,
        hook_text="HOOK never spoken",
        recap_text="RECAP never spoken",
        teaser_text="TEASER never spoken",
        scenes=list(scenes),
    )


def three() -> Part:
    return make_part(
        narration("s1", "alpha beta"),
        narration("s2", "gamma delta epsilon"),
        narration("s3", "zeta eta"),
    )


@pytest.fixture
def cache(tmp_path: Path) -> StageCache:
    return StageCache(tmp_path / ".cache")


def run(
    part: Part,
    chain: list[VoiceChoice],
    cache: StageCache,
    *,
    aligner: FakeAligner | None = None,
    tracker: SpendTracker | None = None,
) -> tuple[NarrationResult, SpendTracker]:
    tracker = tracker if tracker is not None else SpendTracker(limit_usd=None)
    result = narrate_part(
        part,
        chain,
        language=Language.ENGLISH,
        tracker=tracker,
        prices=PRICES,
        cache=cache,
        aligner=aligner,
    )
    return result, tracker


def voice_of(result: NarrationResult) -> VoiceUsed:
    assert result.voice is not None
    return result.voice


def test_only_scene_text_is_spoken(cache: StageCache) -> None:
    voice = FakeVoice()
    part = make_part(
        narration("s1", "plain words", tts="plain wordz"),
        verse("v1"),
        narration("s2", "second scene"),
    )
    result, _ = run(part, [VoiceChoice(voice, "v-1")], cache)
    texts = [r.text for r in voice.requests]
    assert texts == ["plain wordz", "second scene"]
    assert all(r.voice_id == "v-1" and r.language is Language.ENGLISH for r in voice.requests)
    for r in voice.requests:
        assert "HOOK" not in r.text and "RECAP" not in r.text and "TEASER" not in r.text
    assert set(result.clips) == {"s1", "s2"}


def test_verse_only_part_has_no_voice(cache: StageCache) -> None:
    voice = FakeVoice()
    result, _ = run(make_part(verse("v1")), [VoiceChoice(voice, "v")], cache)
    assert result.clips == {}
    assert result.voice is None
    assert voice.requests == []


def test_primary_failure_revoices_whole_part_and_records_fallback(cache: StageCache) -> None:
    primary = FakeVoice(fail_on="zeta")
    backup = FakeVoice("edge", "edge-tts")
    result, _ = run(three(), [VoiceChoice(primary, "p"), VoiceChoice(backup, "b")], cache)
    assert [r.text for r in primary.requests] == ["alpha beta", "gamma delta epsilon", "zeta eta"]
    assert [r.text for r in backup.requests] == ["alpha beta", "gamma delta epsilon", "zeta eta"]
    voice = voice_of(result)
    assert (voice.provider, voice.model, voice.voice_id) == ("edge", "edge-tts", "b")
    assert voice.fallback_from == "paid/p"
    assert voice.fallback_reasons == ["paid/p: HTTP 500"]


def test_no_fallback_records_primary(cache: StageCache) -> None:
    result, _ = run(three(), [VoiceChoice(FakeVoice(), "p")], cache)
    voice = voice_of(result)
    assert (voice.provider, voice.model, voice.voice_id) == ("paid", "m1", "p")
    assert voice.fallback_from is None
    assert voice.fallback_reasons == []


def test_billed_provider_error_falls_back(cache: StageCache) -> None:
    primary = FakeVoice(fail_on="alpha", error=BilledProviderError("bad body"))
    backup = FakeVoice("edge", "edge-tts")
    result, tracker = run(three(), [VoiceChoice(primary, "p"), VoiceChoice(backup, "b")], cache)
    assert voice_of(result).fallback_reasons == ["paid/p: bad body"]
    assert tracker.spent > 0  # the billed failure was charged
    assert result.cost_estimated is True


@pytest.mark.parametrize(
    "error",
    [
        BudgetExceededError(Decimal("2"), Decimal("1")),
        MissingApiKeyError("Missing API key"),
    ],
)
def test_budget_error_does_not_fall_back(cache: StageCache, error: Exception) -> None:
    primary = FakeVoice(fail_on="gamma", error=error)
    backup = FakeVoice("edge", "edge-tts")
    with pytest.raises(type(error)):
        run(three(), [VoiceChoice(primary, "p"), VoiceChoice(backup, "b")], cache)
    assert backup.requests == []


def test_tracker_budget_exceeded_does_not_fall_back(cache: StageCache) -> None:
    primary = FakeVoice()
    backup = FakeVoice("edge", "edge-tts")
    tracker = SpendTracker(limit_usd=Decimal("0.001"))
    with pytest.raises(BudgetExceededError):
        run(three(), [VoiceChoice(primary, "p"), VoiceChoice(backup, "b")], cache, tracker=tracker)
    assert primary.requests == [] and backup.requests == []


def test_all_voices_fail_raises_audio_error(cache: StageCache) -> None:
    a = FakeVoice(fail_on="alpha", error=ProviderError("down"))
    b = FakeVoice("edge", "edge-tts", fail_on="gamma", error=ProviderError("offline"))
    with pytest.raises(AudioError) as info:
        run(three(), [VoiceChoice(a, "a"), VoiceChoice(b, "b")], cache)
    assert str(info.value) == "no voice could narrate part 2: paid/a: down; edge/b: offline"


def test_missing_extra_counts_as_failed_voice(cache: StageCache) -> None:
    hint = "install the 'tts-edge' extra: uv sync --extra tts-edge"
    a = FakeVoice("edge", "edge-tts", fail_on="alpha", error=ExtraMissingError(hint))
    b = FakeVoice("piper", "model")
    result, _ = run(three(), [VoiceChoice(a, "a"), VoiceChoice(b, "piper")], cache)
    voice = voice_of(result)
    assert voice.provider == "piper"
    assert voice.fallback_from == "edge/a"
    assert voice.fallback_reasons == [f"edge/a: {hint}"]


def test_other_channel_config_error_propagates_without_fallback(cache: StageCache) -> None:
    # Same message prefix as a missing extra, but the type decides.
    error = ChannelConfigError("install the 'x' extra -- but really a config error")
    a = FakeVoice(fail_on="alpha", error=error)
    b = FakeVoice("edge", "edge-tts")
    with pytest.raises(ChannelConfigError) as info:
        run(three(), [VoiceChoice(a, "a"), VoiceChoice(b, "b")], cache)
    assert info.value is error
    assert b.requests == []


def test_aligner_provider_error_estimates_and_keeps_voice(cache: StageCache) -> None:
    voice = FakeVoice(spans=False)
    backup = FakeVoice("edge", "edge-tts")
    aligner = FakeAligner(error=ProviderError("ctc crashed"))
    result, _ = run(
        three(), [VoiceChoice(voice, "p"), VoiceChoice(backup, "b")], cache, aligner=aligner
    )
    assert voice_of(result).provider == "paid"
    assert voice_of(result).fallback_from is None
    assert backup.requests == []
    assert len(aligner.calls) == 3  # a failed alignment does not disable the aligner
    for clip in result.clips.values():
        assert clip.timing_source is TimingSource.ESTIMATED
    assert result.warnings == [
        f"{sid}: alignment failed; word timings estimated" for sid in ("s1", "s2", "s3")
    ]


def test_cache_entry_needs_both_files(cache: StageCache) -> None:
    part = make_part(narration("s1", "alpha beta"))
    voice = FakeVoice()
    run(part, [VoiceChoice(voice, "p")], cache)
    key = cache_key(
        "tts",
        ADAPTER_VERSION,
        {"provider": "paid", "model": "m1", "voice_id": "p", "text": "alpha beta"},
    )
    audio = cache.root / "tts" / f"{key}.audio"
    meta = cache.root / "tts" / f"{key}.json"
    assert audio.is_file() and meta.is_file()
    assert json.loads(meta.read_text(encoding="utf-8"))["billed_chars"] == len("alpha beta")
    assert not list((cache.root / "tts").glob(".*"))  # no temp files left behind

    meta.unlink()
    choice = VoiceChoice(voice, "p")
    assert uncached_chars(part, choice, cache) == len("alpha beta")
    run(part, [choice], cache)
    assert len(voice.requests) == 2
    assert meta.is_file()

    audio.unlink()
    assert uncached_chars(part, choice, cache) == len("alpha beta")
    run(part, [choice], cache)
    assert len(voice.requests) == 3

    meta.write_text("{not json", encoding="utf-8")
    run(part, [choice], cache)
    assert len(voice.requests) == 4  # a corrupt entry is a miss and is rewritten
    run(part, [choice], cache)
    assert len(voice.requests) == 4


def test_cost_estimated_read_from_wrapper(cache: StageCache) -> None:
    billed, _ = run(three(), [VoiceChoice(FakeVoice(billed=True), "p")], cache)
    assert billed.cost_estimated is False
    unbilled, _ = run(three(), [VoiceChoice(FakeVoice("other", "m2", billed=False), "p")], cache)
    assert unbilled.cost_estimated is True
    free, tracker = run(
        three(), [VoiceChoice(FakeVoice("edge", "edge-tts", billed=False), "e")], cache
    )
    assert free.cost_estimated is False
    assert tracker.spent == 0


def test_cache_hit_makes_no_call_and_no_charge(cache: StageCache) -> None:
    voice = FakeVoice()
    first, tracker1 = run(three(), [VoiceChoice(voice, "p")], cache)
    assert tracker1.spent == Decimal(len("alpha betagamma delta epsilonzeta eta")) / 1000 * Decimal(
        "0.30"
    )
    assert len(voice.requests) == 3
    assert uncached_chars(three(), VoiceChoice(voice, "p"), cache) == 0
    second, tracker2 = run(three(), [VoiceChoice(voice, "p")], cache)
    assert len(voice.requests) == 3
    assert tracker2.spent == 0
    assert tracker2.entries == []
    for sid, clip in first.clips.items():
        again = second.clips[sid]
        assert np.array_equal(clip.samples, again.samples)
        assert clip.words == again.words
        assert again.timing_source is TimingSource.PROVIDER
    # A different voice id is a different key.
    assert uncached_chars(three(), VoiceChoice(voice, "other"), cache) == len(
        "alpha betagamma delta epsilonzeta eta"
    )


def test_uncached_chars_counts_repeated_text_once(cache: StageCache) -> None:
    part = make_part(narration("s1", "same words"), narration("s2", "same words"), verse("v"))
    assert uncached_chars(part, VoiceChoice(FakeVoice(), "p"), cache) == len("same words")


def test_provider_spans_used_when_they_map(cache: StageCache) -> None:
    aligner = FakeAligner()
    result, _ = run(three(), [VoiceChoice(FakeVoice(), "p")], cache, aligner=aligner)
    clip = result.clips["s2"]
    assert clip.timing_source is TimingSource.PROVIDER
    assert [(w.text, w.start_ms, w.end_ms) for w in clip.words] == [
        (s.text, s.start_ms, s.end_ms) for s in word_spans("gamma delta epsilon")
    ]
    assert aligner.calls == []
    assert result.warnings == []


def test_aligner_used_when_no_spans(cache: StageCache) -> None:
    aligner = FakeAligner()
    result, _ = run(three(), [VoiceChoice(FakeVoice(spans=False), "p")], cache, aligner=aligner)
    clip = result.clips["s1"]
    assert clip.timing_source is TimingSource.ALIGNED
    assert [(w.start_ms, w.end_ms) for w in clip.words] == [
        (s.start_ms, s.end_ms) for s in word_spans("alpha beta", lead_ms=LEAD_MS + 10)
    ]
    n, rate, text, language = aligner.calls[0]
    assert (n, rate, text, language) == (
        len(clip.samples),
        SAMPLE_RATE,
        "alpha beta",
        Language.ENGLISH,
    )
    assert result.warnings == []


def test_aligner_used_when_provider_spans_do_not_map(cache: StageCache) -> None:
    class Mismatch(FakeVoice):
        def synthesize(self, req: SpeechRequest) -> Speech:
            speech = super().synthesize(req)
            return Speech(
                speech.audio, (TimedSpan("nope", 0, 10),), speech.provider, speech.model, None
            )

    aligner = FakeAligner()
    result, _ = run(three(), [VoiceChoice(Mismatch(), "p")], cache, aligner=aligner)
    assert result.clips["s1"].timing_source is TimingSource.ALIGNED


def test_estimated_when_no_aligner_with_warnings(cache: StageCache) -> None:
    result, _ = run(three(), [VoiceChoice(FakeVoice(spans=False), "p")], cache)
    assert result.warnings == [f"{sid}: word timings estimated" for sid in ("s1", "s2", "s3")]
    assert result.aligner_unavailable is False
    clip = result.clips["s2"]
    assert clip.timing_source is TimingSource.ESTIMATED
    trimmed, lead = trim_silence(clip.samples)
    assert clip.words[0].start_ms == lead
    assert clip.words[-1].end_ms == lead + len(trimmed) * 1000 // SAMPLE_RATE
    assert lead > 0
    assert [w.text for w in clip.words] == ["gamma", "delta", "epsilon"]


def test_estimated_when_aligner_output_does_not_map(cache: StageCache) -> None:
    class Wrong(FakeAligner):
        def align(
            self, samples: Samples, sample_rate: int, text: str, language: Language
        ) -> list[TimedSpan]:
            return [TimedSpan("other", 0, 5)]

    result, _ = run(
        make_part(narration("s1", "alpha beta")),
        [VoiceChoice(FakeVoice(spans=False), "p")],
        cache,
        aligner=Wrong(),
    )
    assert result.clips["s1"].timing_source is TimingSource.ESTIMATED
    assert result.warnings == ["s1: word timings estimated"]


def test_aligner_unavailable_warns_once(cache: StageCache) -> None:
    aligner = FakeAligner(error=AlignerUnavailableError(ALIGN_EXTRA_WARNING))
    result, _ = run(three(), [VoiceChoice(FakeVoice(spans=False), "p")], cache, aligner=aligner)
    assert len(aligner.calls) == 1  # dropped for the rest of the part
    assert result.aligner_unavailable is True
    assert ALIGN_EXTRA_WARNING == "install the 'align' extra for word timings"
    # The once-per-part extra warning is the caller's (Task 10); clips are estimated.
    assert result.warnings == [f"{sid}: word timings estimated" for sid in ("s1", "s2", "s3")]


def test_word_texts_are_caption_tokens(cache: StageCache) -> None:
    src = "\u0642\u0627\u0644 \u0627\u0644\u0644\u0647"
    tts = "\u0642\u064e\u0627\u0644\u064e \u0627\u0644\u0644\u0651\u064e\u0647\u064f"
    cap = "\u0642\u0627\u0644\u064e \u0627\u0644\u0644\u0647"
    part = make_part(narration("s1", src, tts=tts, caption=cap), narration("s2", src))
    voice = FakeVoice("edge", "edge-tts")
    result, _ = run(part, [VoiceChoice(voice, "v")], cache)
    assert voice.requests[0].text == tts
    assert [w.text for w in result.clips["s1"].words] == cap.split()
    assert [w.text for w in result.clips["s2"].words] == src.split()


def test_caption_token_mismatch_raises_before_spend(cache: StageCache) -> None:
    part = make_part(narration("s7", "one two", caption="one two three"))
    voice = FakeVoice()
    with pytest.raises(PlanningError, match="s7"):
        run(part, [VoiceChoice(voice, "p")], cache)
    assert voice.requests == []


def test_one_wrapper_per_voice_per_part(cache: StageCache) -> None:
    primary = FakeVoice(fail_on="zeta", billed=False)
    backup = FakeVoice("edge", "edge-tts", billed=False)
    result, tracker = run(three(), [VoiceChoice(primary, "p"), VoiceChoice(backup, "b")], cache)
    # The failed paid primary charged unbilled estimates for two scenes.
    assert tracker.spent > 0
    assert result.cost_estimated is True
    assert [e.what for e in tracker.entries] == ["voice paid/m1"] * 2 + ["voice edge/edge-tts"] * 3


@pytest.mark.parametrize(
    "meta",
    [
        "[]",
        '{"spans": "x"}',
        '{"spans": [["a", 1]]}',
        '{"spans": [[1, 0, 5]]}',
        '{"spans": [["a", true, 5]]}',
        '{"spans": [["a", -1, 5]]}',
        '{"spans": [["a", 9, 5]]}',
    ],
)
def test_malformed_cache_meta_is_a_miss(cache: StageCache, meta: str) -> None:
    part = make_part(narration("s1", "alpha beta"))
    voice = FakeVoice()
    choice = VoiceChoice(voice, "p")
    run(part, [choice], cache)
    (meta_path,) = (cache.root / "tts").glob("*.json")
    meta_path.write_text(meta, encoding="utf-8")
    run(part, [choice], cache)
    assert len(voice.requests) == 2  # the malformed entry was a miss and is rewritten


def test_empty_cached_audio_is_a_miss(cache: StageCache) -> None:
    part = make_part(narration("s1", "alpha beta"))
    choice = VoiceChoice(FakeVoice(), "p")
    run(part, [choice], cache)
    (audio_path,) = (cache.root / "tts").glob("*.audio")
    audio_path.write_bytes(b"")
    assert uncached_chars(part, choice, cache) == len("alpha beta")


def test_cache_write_failure_raises_audio_error(
    cache: StageCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(src: str, dst: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("narratoflow.audio.narrate.os.replace", refuse)
    with pytest.raises(AudioError, match="cannot write tts cache"):
        run(make_part(narration("s1", "alpha beta")), [VoiceChoice(FakeVoice(), "p")], cache)
    assert not list((cache.root / "tts").iterdir())  # temp files cleaned up


class BadAudioVoice(FakeVoice):
    def __init__(self, audio: bytes, bad_on: str, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        self.audio = audio
        self.bad_on = bad_on

    def synthesize(self, req: SpeechRequest) -> Speech:
        speech = super().synthesize(req)
        if self.bad_on in req.text:
            return Speech(self.audio, speech.spans, speech.provider, speech.model, None)
        return speech


def key_for(choice: VoiceChoice, text: str) -> str:
    return cache_key(
        "tts",
        ADAPTER_VERSION,
        {
            "provider": choice.provider.name,
            "model": choice.provider.model,
            "voice_id": choice.voice_id,
            "text": text,
        },
    )


@pytest.mark.parametrize(
    ("audio", "why"),
    [
        (b"this is not audio at all", "does not decode"),
        (to_wav_bytes(np.zeros(SAMPLE_RATE // 2, dtype=np.float32)), "all silence"),
    ],
)
def test_bad_voice_audio_fails_the_voice(cache: StageCache, audio: bytes, why: str) -> None:
    primary = BadAudioVoice(audio, "gamma", billed=False)
    backup = FakeVoice("edge", "edge-tts")
    p, b = VoiceChoice(primary, "p"), VoiceChoice(backup, "b")
    result, tracker = run(three(), [p, b], cache)
    voice = voice_of(result)
    assert voice.provider == "edge" and voice.fallback_from == "paid/p"
    (reason,) = voice.fallback_reasons
    assert reason.startswith("paid/p: scene s2: audio") and why in reason
    assert [r.text for r in backup.requests] == ["alpha beta", "gamma delta epsilon", "zeta eta"]
    # The bad clip's entry is gone; the good one stays; the paid spend stays charged.
    tts = cache.root / "tts"
    bad, good = key_for(p, "gamma delta epsilon"), key_for(p, "alpha beta")
    assert not (tts / f"{bad}.audio").exists() and not (tts / f"{bad}.json").exists()
    assert (tts / f"{good}.audio").exists() and (tts / f"{good}.json").exists()
    assert tracker.spent == Decimal(len("alpha betagamma delta epsilon")) / 1000 * Decimal("0.30")
    assert result.cost_estimated is True


def test_bad_cached_audio_fails_the_voice_and_is_dropped(cache: StageCache) -> None:
    part = make_part(narration("s1", "alpha beta"))
    p = VoiceChoice(FakeVoice(), "p")
    run(part, [p], cache)
    (cache.root / "tts" / f"{key_for(p, 'alpha beta')}.audio").write_bytes(b"junk")
    backup = FakeVoice("edge", "edge-tts")
    result, _ = run(part, [p, VoiceChoice(backup, "b")], cache)
    assert voice_of(result).provider == "edge"
    assert not (cache.root / "tts" / f"{key_for(p, 'alpha beta')}.json").exists()


def test_failed_voice_cached_clips_are_reused_on_rerun(cache: StageCache) -> None:
    failing = FakeVoice(fail_on="zeta")
    backup = FakeVoice("edge", "edge-tts", fail_on="alpha", error=ProviderError("off"))
    with pytest.raises(AudioError):
        run(three(), [VoiceChoice(failing, "p"), VoiceChoice(backup, "b")], cache)
    fixed = FakeVoice()  # same provider/model/voice id, now healthy
    choice = VoiceChoice(fixed, "p")
    assert uncached_chars(three(), choice, cache) == len("zeta eta")
    result, tracker = run(three(), [choice], cache)
    assert [r.text for r in fixed.requests] == ["zeta eta"]
    assert tracker.spent == Decimal(len("zeta eta")) / 1000 * Decimal("0.30")
    assert set(result.clips) == {"s1", "s2", "s3"}


def test_store_drops_old_json_before_writing_new_audio(
    cache: StageCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = make_part(narration("s1", "alpha beta"))
    p = VoiceChoice(FakeVoice(), "p")
    run(part, [p], cache)
    tts = cache.root / "tts"
    key = key_for(p, "alpha beta")
    (tts / f"{key}.audio").unlink()  # a miss with old spans still on disk
    assert (tts / f"{key}.json").exists()

    def crash(src: str, dst: str) -> None:
        raise OSError("power cut")

    monkeypatch.setattr("narratoflow.audio.narrate.os.replace", crash)
    with pytest.raises(AudioError):
        run(part, [p], cache)
    assert not (tts / f"{key}.json").exists()  # old spans never pair with new audio


def test_uncached_chars_is_stat_only(cache: StageCache, monkeypatch: pytest.MonkeyPatch) -> None:
    part = make_part(narration("s1", "alpha beta"))
    p = VoiceChoice(FakeVoice(), "p")
    run(part, [p], cache)

    def no_read(self: Path) -> bytes:
        raise AssertionError("uncached_chars must not read the audio")

    monkeypatch.setattr(Path, "read_bytes", no_read)
    assert uncached_chars(part, p, cache) == 0
    (cache.root / "tts" / f"{key_for(p, 'alpha beta')}.json").write_text("", encoding="utf-8")
    assert uncached_chars(part, p, cache) == len("alpha beta")


class FakePiper(PiperProvider):
    def synthesize(self, req: SpeechRequest) -> Speech:
        audio = to_wav_bytes(speechlike(len(req.text.split()), WORD_MS, GAP_MS, lead_ms=LEAD_MS))
        return Speech(audio, (), self.name, self.model, None)


def test_piper_key_tracks_the_model_file(cache: StageCache, tmp_path: Path) -> None:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model-v1")
    part = make_part(narration("s1", "alpha beta"))
    choice = VoiceChoice(FakePiper(model), "piper")
    run(part, [choice], cache)
    assert uncached_chars(part, choice, cache) == 0

    model.write_bytes(b"model-v2-longer")  # replaced, different size
    assert uncached_chars(part, choice, cache) == len("alpha beta")
    run(part, [choice], cache)
    assert uncached_chars(part, choice, cache) == 0

    st = model.stat()
    model.write_bytes(b"model-v3-longer")  # same size, same name
    os.utime(model, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert uncached_chars(part, choice, cache) == len("alpha beta")

    other = tmp_path / "other"
    other.mkdir()
    twin = other / "voice.onnx"  # same stem, different file
    twin.write_bytes(b"model-v3-longer")
    assert uncached_chars(part, VoiceChoice(FakePiper(twin), "piper"), cache) == len("alpha beta")


def test_piper_key_without_model_file(cache: StageCache, tmp_path: Path) -> None:
    choice = VoiceChoice(FakePiper(tmp_path / "missing.onnx"), "piper")
    assert uncached_chars(make_part(narration("s1", "a b")), choice, cache) == len("a b")


def test_unremovable_old_json_raises_audio_error(cache: StageCache) -> None:
    part = make_part(narration("s1", "alpha beta"))
    p = VoiceChoice(FakeVoice(), "p")
    tts = cache.root / "tts"
    tts.mkdir(parents=True)
    (tts / f"{key_for(p, 'alpha beta')}.json").mkdir()  # cannot be unlinked as a file
    with pytest.raises(AudioError, match="cannot write tts cache"):
        run(part, [p], cache)
