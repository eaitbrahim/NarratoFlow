"""Narration synthesis: one voice per part, cached per project, with word timings.

Only the spoken form of the `tts_text or source_text` of NARRATION scenes is ever
voiced (numbers spelled out, the channel's pronunciation lexicon applied); captions
keep the author's tokens and take the merged timings of the words each became. Each voice of
the chain is tried in order for the *whole* part: a `ProviderError` (including a
`BilledProviderError`), an `ExtraMissingError`, or audio that does not decode or
is all silence, on any scene abandons that voice and the next one starts again
from the first scene. Bad audio also drops that clip's cache entry; any spend
already charged for it stays charged. Bad audio found in the cache is dropped and
re-voiced with the same voice instead. Budget, key and other config
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
from narratoflow.audio.io import SAMPLE_RATE, Samples, decode, require_ffmpeg, trim_silence
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
from narratoflow.text.spoken import Lexicon, SpokenText, spoken_text
from narratoflow.tts import SpeechRequest, VoiceChoice
from narratoflow.tts.factory import TrackedSpeechProvider, voice_terms
from narratoflow.tts.piper import PiperProvider

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


@dataclass(frozen=True)
class _Clip:
    """A voiced scene: decoded samples, provider spans and the non-silent span."""

    samples: Samples
    spans: tuple[TimedSpan, ...]
    lead_ms: int
    speech_ms: int


class _BadVoiceAudio(AudioError):
    """The voice returned audio that does not decode or is all silence."""


@dataclass
class _Timer:
    aligner: Aligner | None
    language: Language
    warnings: list[str] = field(default_factory=list)
    unavailable: bool = False


def _narration_scenes(part: Part) -> list[Scene]:
    return [s for s in part.scenes if s.kind is SceneKind.NARRATION]


def _spoken(scene: Scene, language: Language, lexicon: Lexicon) -> SpokenText:
    return spoken_text(scene.narration.tts_text or scene.narration.source_text, language, lexicon)


def _caption(scene: Scene) -> str:
    return scene.narration.caption_text or scene.narration.source_text


def _label(choice: VoiceChoice) -> str:
    return f"{choice.provider.name}/{choice.voice_id}"


def _model_file(provider: object) -> dict[str, object] | None:
    """The identity of a local model file, so a replaced model never reuses clips."""
    if not isinstance(provider, PiperProvider):
        return None
    path = provider.model_path.resolve()
    try:
        st = path.stat()
    except OSError:
        return {"path": str(path)}
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _key(choice: VoiceChoice, text: str) -> str:
    payload: dict[str, object] = {
        "provider": choice.provider.name,
        "model": choice.provider.model,
        "voice_id": choice.voice_id,
        "text": text,
    }
    model_file = _model_file(choice.provider)
    if model_file is not None:
        payload["model_file"] = model_file
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


def _present(cache: StageCache, key: str) -> bool:
    """A cheap stat-only check: both files exist and neither is empty."""
    return all(_size(cache.root / STAGE / f"{key}{suffix}") > 0 for suffix in (".audio", ".json"))


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _drop(cache: StageCache, key: str) -> None:
    for suffix in (".json", ".audio"):
        (cache.root / STAGE / f"{key}{suffix}").unlink(missing_ok=True)


def _store(cache: StageCache, key: str, voiced: _Voiced, billed_chars: int | None) -> None:
    """Drop any old `.json`, write the audio, then the `.json` last.

    A hit needs both files, so a crash at any point never pairs old spans with
    new audio and never leaves half an entry.
    """
    meta = {
        "spans": [[s.text, s.start_ms, s.end_ms] for s in voiced.spans],
        "billed_chars": billed_chars,
    }
    meta_path = cache.path_for(STAGE, key, ".json")
    try:
        meta_path.unlink(missing_ok=True)
    except OSError as exc:
        raise AudioError(f"cannot write tts cache {meta_path}: {exc}") from exc
    _write_atomic(cache.path_for(STAGE, key, ".audio"), voiced.audio)
    _write_atomic(meta_path, json.dumps(meta).encode("utf-8"))


def uncached_chars(
    part: Part,
    choice: VoiceChoice,
    cache: StageCache,
    *,
    language: Language,
    lexicon: Lexicon,
) -> int:
    """Characters of the part's narration whose cache entry misses for `choice`.

    A text repeated within the part is counted once: the first call caches it.
    The check is stat-only (both files present, neither empty); `narrate_part`
    still validates each entry before using it.
    """
    seen: set[str] = set()
    total = 0
    for scene in _narration_scenes(part):
        text = _spoken(scene, language, lexicon).text
        key = _key(choice, text)
        if key in seen:
            continue
        seen.add(key)
        if not _present(cache, key):
            total += len(text)
    return total


def _check_tokens(scenes: list[Scene], language: Language, lexicon: Lexicon) -> None:
    for scene in scenes:
        spoken = len(_spoken(scene, language, lexicon).groups)
        shown = len(tokens(_caption(scene)))
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
    lexicon: Lexicon,
) -> dict[str, _Clip]:
    """Voice (or reuse) every scene.

    A bad cached clip is dropped and re-voiced with the same voice (the normal paid
    path); only freshly synthesized bad audio fails the voice.
    """
    out: dict[str, _Clip] = {}
    for scene in scenes:
        text = _spoken(scene, language, lexicon).text
        key = _key(choice, text)
        hit = _read(cache, key)
        if hit is not None:
            try:
                out[scene.id] = _checked_clip(cache, key, scene, hit)
                continue
            except _BadVoiceAudio:
                pass  # entry dropped; synthesize it again below
        speech = tracked.synthesize(SpeechRequest(text, choice.voice_id, language))
        fresh = _Voiced(speech.audio, tuple(speech.spans))
        _store(cache, key, fresh, speech.billed_chars)
        out[scene.id] = _checked_clip(cache, key, scene, fresh)
    return out


def _checked_clip(cache: StageCache, key: str, scene: Scene, voiced: _Voiced) -> _Clip:
    try:
        samples = decode(voiced.audio)
    except AudioError as exc:
        _drop(cache, key)
        raise _BadVoiceAudio(f"scene {scene.id}: audio does not decode ({exc})") from exc
    trimmed, lead_ms = trim_silence(samples)
    if trimmed.size == 0:
        _drop(cache, key)
        raise _BadVoiceAudio(f"scene {scene.id}: audio is all silence")
    return _Clip(samples, voiced.spans, lead_ms, len(trimmed) * 1000 // SAMPLE_RATE)


def _timings(
    timer: _Timer, text: str, scene: Scene, clip: _Clip
) -> tuple[list[WordTiming], TimingSource]:
    words = tokens(text)
    if clip.spans:
        found = timings_from_spans(words, clip.spans)
        if found is not None:
            return found, TimingSource.PROVIDER
    note = f"{scene.id}: word timings estimated"
    if timer.aligner is not None:
        try:
            aligned = timer.aligner.align(clip.samples, SAMPLE_RATE, text, timer.language)
        except AlignerUnavailableError:
            timer.aligner = None
            timer.unavailable = True
        except ProviderError:
            note = f"{scene.id}: alignment failed; word timings estimated"
        else:
            found = timings_from_spans(words, aligned)
            if found is not None:
                return found, TimingSource.ALIGNED
    timer.warnings.append(note)
    # Spread the words over the clip's non-silent span [speech_start, speech_end].
    speech_start_ms = clip.lead_ms
    speech_end_ms = clip.lead_ms + clip.speech_ms
    return estimate_timings(words, speech_start_ms, speech_end_ms), TimingSource.ESTIMATED


def _caption_words(
    scene: Scene, timed: list[WordTiming], groups: tuple[int, ...]
) -> list[WordTiming]:
    """Caption tokens, each spanning the spoken words of its group.

    `_check_tokens` guaranteed one group per caption token up front.
    """
    shown = tokens(_caption(scene))
    out: list[WordTiming] = []
    index = 0
    for text, size in zip(shown, groups, strict=True):
        first, last = timed[index], timed[index + size - 1]
        out.append(WordTiming(text=text, start_ms=first.start_ms, end_ms=last.end_ms))
        index += size
    return out


def _voice_used(choice: VoiceChoice, first: VoiceChoice | None, reasons: list[str]) -> VoiceUsed:
    """What the timeline records about the voice that narrated, with a Piper model's licence."""
    declared: dict[str, str | None] = {}
    provider = choice.provider
    if isinstance(provider, PiperProvider):
        declared = {
            "model_license": provider.model_license,
            "model_source": provider.model_source,
            "model_attribution": provider.model_attribution,
        }
    return VoiceUsed.model_validate(
        {
            "provider": provider.name,
            "model": provider.model,
            "voice_id": choice.voice_id,
            "terms": voice_terms(provider.name),
            "fallback_from": _label(first) if first is not None else None,
            "fallback_reasons": reasons,
            **declared,
        }
    )


def narrate_part(
    part: Part,
    chain: list[VoiceChoice],
    *,
    language: Language,
    tracker: SpendTracker,
    prices: PriceTable,
    cache: StageCache,
    aligner: Aligner | None,
    lexicon: Lexicon = Lexicon.EMPTY,
) -> NarrationResult:
    """Voice every narration scene of `part` with the first voice that succeeds."""
    scenes = _narration_scenes(part)
    if not scenes:
        return NarrationResult(clips={}, voice=None, cost_estimated=False, warnings=[])
    _check_tokens(scenes, language, lexicon)
    require_ffmpeg()  # a missing ffmpeg is a setup error, never a voice failure
    reasons: list[str] = []
    wrappers: list[TrackedSpeechProvider] = []
    for index, choice in enumerate(chain):
        tracked = TrackedSpeechProvider(choice.provider, tracker, prices)
        wrappers.append(tracked)
        try:
            voiced = _voice_all(scenes, choice, tracked, cache, language, lexicon)
        except (ProviderError, ExtraMissingError, _BadVoiceAudio) as exc:
            reasons.append(f"{_label(choice)}: {exc}")
            continue
        voice = _voice_used(choice, chain[0] if index > 0 else None, reasons)
        timer = _Timer(aligner, language)
        clips: dict[str, NarrationClip] = {}
        for scene in scenes:
            clip = voiced[scene.id]
            spoken = _spoken(scene, language, lexicon)
            timed, source = _timings(timer, spoken.text, scene, clip)
            clips[scene.id] = NarrationClip(
                clip.samples, _caption_words(scene, timed, spoken.groups), source
            )
        estimated = any(w.cost_estimated and prices.voice(w.name, w.model) > 0 for w in wrappers)
        return NarrationResult(
            clips=clips,
            voice=voice,
            cost_estimated=estimated,
            warnings=timer.warnings,
            aligner_unavailable=timer.unavailable,
        )
    raise AudioError(f"no voice could narrate part {part.index}: {'; '.join(reasons)}")
