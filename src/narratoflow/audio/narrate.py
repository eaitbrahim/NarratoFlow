"""Narration synthesis: one voice per part, cached per project, with word timings.

Only the `tts_text or source_text` of NARRATION scenes is ever spoken. Each voice of
the chain is tried in order for the *whole* part: a `ProviderError` (including a
`BilledProviderError`) or an `ExtraMissingError` on any scene abandons that voice
and the next one starts again from the first scene. Budget, key and other config
errors propagate. Word timings come from the provider's spans, else the aligner,
else an estimate over the clip's non-silent span; the voice never falls back
because of timing.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from narratoflow.align import Aligner
from narratoflow.audio.io import SAMPLE_RATE, Samples, decode, trim_silence
from narratoflow.audio.timeline import TimingSource, VoiceUsed
from narratoflow.audio.words import TimedSpan, estimate_timings, timings_from_spans, tokens
from narratoflow.costs.prices import PriceTable
from narratoflow.costs.tracker import SpendTracker
from narratoflow.errors import (
    AlignerUnavailableError,
    AudioError,
    ExtraMissingError,
    PlanningError,
    ProviderError,
)
from narratoflow.models import Language, Part, Scene, SceneKind, WordTiming
from narratoflow.pipeline.cache import StageCache, cache_key
from narratoflow.tts import SpeechRequest, VoiceChoice
from narratoflow.tts.factory import TrackedSpeechProvider

ADAPTER_VERSION = "1"
STAGE = "tts"
ALIGN_EXTRA_WARNING = "install the 'align' extra for word timings"
_FILE_MODE = 0o644


@dataclass
class NarrationClip:
    """One scene's untrimmed clip; `words` are caption tokens on the clip's clock."""

    samples: Samples
    words: list[WordTiming]
    timing_source: TimingSource


@dataclass
class NarrationResult:
    clips: dict[str, NarrationClip]
    voice: VoiceUsed | None  # None: verse-only part
    cost_estimated: bool
    warnings: list[str]
    aligner_unavailable: bool = False
    """The aligner raised `AlignerUnavailableError`; the caller warns once per part."""


@dataclass(frozen=True)
class _Voiced:
    audio: bytes
    spans: tuple[TimedSpan, ...]


@dataclass
class _Timer:
    aligner: Aligner | None
    language: Language
    warnings: list[str] = field(default_factory=list)
    unavailable: bool = False


def _narration_scenes(part: Part) -> list[Scene]:
    return [s for s in part.scenes if s.kind is SceneKind.NARRATION]


def _spoken(scene: Scene) -> str:
    return scene.narration.tts_text or scene.narration.source_text


def _caption(scene: Scene) -> str:
    return scene.narration.caption_text or scene.narration.source_text


def _key(choice: VoiceChoice, text: str) -> str:
    payload = {
        "provider": choice.provider.name,
        "model": choice.provider.model,
        "voice_id": choice.voice_id,
        "text": text,
    }
    return cache_key(STAGE, ADAPTER_VERSION, payload)


def _span(raw: object) -> TimedSpan | None:
    if not (isinstance(raw, list) and len(raw) == 3):
        return None
    text, start, end = raw
    ok = isinstance(text, str) and all(
        isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (start, end)
    )
    if not ok or end < start:
        return None
    return TimedSpan(text, start, end)


def _read(cache: StageCache, key: str) -> _Voiced | None:
    """A hit needs both `<key>.audio` and a valid `<key>.json`; anything else misses."""
    audio_path = cache.get(STAGE, key, ".audio")
    meta_path = cache.get(STAGE, key, ".json")
    if audio_path is None or meta_path is None:
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        audio = audio_path.read_bytes()
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        return None
    raw_spans = meta.get("spans") if isinstance(meta, dict) else None
    if not isinstance(raw_spans, list) or not audio:
        return None
    spans = [_span(s) for s in raw_spans]
    if any(s is None for s in spans):
        return None
    return _Voiced(audio, tuple(s for s in spans if s is not None))


def _write_atomic(path: Path, data: bytes) -> None:
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as tmp:
            tmp_name = tmp.name
            tmp.write(data)
            tmp.flush()
            os.fchmod(tmp.fileno(), _FILE_MODE)
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
    except OSError as exc:
        if tmp_name is not None:
            Path(tmp_name).unlink(missing_ok=True)
        raise AudioError(f"cannot write tts cache {path}: {exc}") from exc


def _store(cache: StageCache, key: str, voiced: _Voiced, billed_chars: int | None) -> None:
    """Write the audio, then the `.json` last, so a hit never sees half an entry."""
    meta = {
        "spans": [[s.text, s.start_ms, s.end_ms] for s in voiced.spans],
        "billed_chars": billed_chars,
    }
    _write_atomic(cache.path_for(STAGE, key, ".audio"), voiced.audio)
    _write_atomic(cache.path_for(STAGE, key, ".json"), json.dumps(meta).encode("utf-8"))


def uncached_chars(part: Part, choice: VoiceChoice, cache: StageCache) -> int:
    """Characters of the part's narration whose cache entry misses for `choice`.

    A text repeated within the part is counted once: the first call caches it.
    """
    seen: set[str] = set()
    total = 0
    for scene in _narration_scenes(part):
        text = _spoken(scene)
        key = _key(choice, text)
        if key in seen:
            continue
        seen.add(key)
        if _read(cache, key) is None:
            total += len(text)
    return total


def _check_tokens(scenes: list[Scene]) -> None:
    for scene in scenes:
        spoken, shown = len(tokens(_spoken(scene))), len(tokens(_caption(scene)))
        if spoken != shown:
            raise PlanningError(
                f"scene {scene.id}: caption has {shown} words but the voiced text has {spoken}"
            )


def _voice_all(
    scenes: list[Scene],
    choice: VoiceChoice,
    tracked: TrackedSpeechProvider,
    cache: StageCache,
    language: Language,
) -> dict[str, _Voiced]:
    out: dict[str, _Voiced] = {}
    for scene in scenes:
        text = _spoken(scene)
        key = _key(choice, text)
        hit = _read(cache, key)
        if hit is None:
            speech = tracked.synthesize(SpeechRequest(text, choice.voice_id, language))
            hit = _Voiced(speech.audio, tuple(speech.spans))
            _store(cache, key, hit, speech.billed_chars)
        out[scene.id] = hit
    return out


def _timings(
    timer: _Timer, scene: Scene, samples: Samples, spans: tuple[TimedSpan, ...]
) -> tuple[list[WordTiming], TimingSource]:
    text = _spoken(scene)
    words = tokens(text)
    if spans:
        found = timings_from_spans(words, spans)
        if found is not None:
            return found, TimingSource.PROVIDER
    note = f"{scene.id}: word timings estimated"
    if timer.aligner is not None:
        try:
            aligned = timer.aligner.align(samples, SAMPLE_RATE, text, timer.language)
        except AlignerUnavailableError:
            timer.aligner = None
            timer.unavailable = True
        except ProviderError:
            note = f"{scene.id}: alignment failed; word timings estimated"
        else:
            found = timings_from_spans(words, aligned)
            if found is not None:
                return found, TimingSource.ALIGNED
    trimmed, lead_ms = trim_silence(samples)
    timer.warnings.append(note)
    offset_ms = lead_ms + len(trimmed) * 1000 // SAMPLE_RATE
    return estimate_timings(words, lead_ms, offset_ms), TimingSource.ESTIMATED


def _caption_words(scene: Scene, timed: list[WordTiming]) -> list[WordTiming]:
    """Caption tokens, index-wise; `_check_tokens` guaranteed equal counts up front."""
    shown = tokens(_caption(scene))
    return [
        WordTiming(text=t, start_ms=w.start_ms, end_ms=w.end_ms)
        for t, w in zip(shown, timed, strict=True)
    ]


def narrate_part(
    part: Part,
    chain: list[VoiceChoice],
    *,
    language: Language,
    tracker: SpendTracker,
    prices: PriceTable,
    cache: StageCache,
    aligner: Aligner | None,
) -> NarrationResult:
    """Voice every narration scene of `part` with the first voice that succeeds."""
    scenes = _narration_scenes(part)
    if not scenes:
        return NarrationResult(clips={}, voice=None, cost_estimated=False, warnings=[])
    _check_tokens(scenes)
    reasons: list[str] = []
    wrappers: list[TrackedSpeechProvider] = []
    for index, choice in enumerate(chain):
        tracked = TrackedSpeechProvider(choice.provider, tracker, prices)
        wrappers.append(tracked)
        try:
            voiced = _voice_all(scenes, choice, tracked, cache, language)
        except (ProviderError, ExtraMissingError) as exc:
            reasons.append(f"{choice.provider.name}: {exc}")
            continue
        voice = VoiceUsed(
            provider=choice.provider.name,
            model=choice.provider.model,
            voice_id=choice.voice_id,
            fallback_from=chain[0].provider.name if index > 0 else None,
            fallback_reasons=reasons,
        )
        timer = _Timer(aligner, language)
        clips: dict[str, NarrationClip] = {}
        for scene in scenes:
            samples = decode(voiced[scene.id].audio)
            timed, source = _timings(timer, scene, samples, voiced[scene.id].spans)
            clips[scene.id] = NarrationClip(samples, _caption_words(scene, timed), source)
        estimated = any(w.cost_estimated and prices.voice(w.name, w.model) > 0 for w in wrappers)
        return NarrationResult(
            clips=clips,
            voice=voice,
            cost_estimated=estimated,
            warnings=timer.warnings,
            aligner_unavailable=timer.unavailable,
        )
    raise AudioError(f"no voice could narrate part {part.index}: {'; '.join(reasons)}")
