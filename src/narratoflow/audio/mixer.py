"""The part mix: scene slots on one clock, loudness-matched clips, stroke SFX, the master.

Scenes are laid out in order from t = 0:

- a NARRATION slot starts its speech at the slot start and ends at
  ``max(speech_end + INTER_SCENE_GAP_MS, start + min_duration_ms)``; before a
  VERSE scene the inter-scene gap is dropped (the verse brings its own buffer);
- a VERSE slot is ``leading + clip + trailing``, with ``VERSE_BUFFER_MS`` only on
  a side that meets a NARRATION scene, and ``AYAH_GAP_MS`` before a following
  VERSE scene. The part start never gets a buffer.

The first clip's lead silence is trimmed to 0 ms, and the master's first onset is
checked against ``FIRST_ONSET_MAX_MS``. No SFX lands in a verse slot (buffers
included). The SFX bus is ducked under the voice bus before mastering, and
``SfxHit.gain_db`` stays the pre-duck gain. Every float the mix reports is finite.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from narratoflow.audio import io as aio
from narratoflow.audio.duck import duck
from narratoflow.audio.io import Loudness, Samples, master
from narratoflow.audio.narrate import NarrationResult
from narratoflow.audio.recitation import AYAH_GAP_MS, VerseClip
from narratoflow.audio.sfx import SFX_FADE_MS, SfxPack, place_sfx
from narratoflow.audio.timeline import AyahTiming, SceneTiming, SfxHit, TimelineWord, TimingSource
from narratoflow.audio.words import shift
from narratoflow.errors import AudioError
from narratoflow.models import Part, Scene, SceneKind, WordTiming
from narratoflow.visuals.marker import marker_seed
from narratoflow.visuals.schedule import draw_schedule
from narratoflow.visuals.strokes import StrokeSet

VERSE_BUFFER_MS = 400
INTER_SCENE_GAP_MS = 250
FIRST_ONSET_MAX_MS = 100
MATCH_TARGET_LUFS = -18.0
MAX_MATCH_GAIN_DB = 12.0
FADE_MS = 10
LOUDNESS_TOLERANCE_LU = 1.0

MASTER_TARGET_LUFS = -14.0
MASTER_TRUE_PEAK_DBTP = -1.0
ONSET_THRESHOLD_DBFS = -45.0
TRIM_KEEP_MS = 20
FIRST_TRIM_MARGIN_DB = 3.0
# The first clip is head-trimmed after its match gain at this level: the master
# lifts the matched mix by about MASTER_TARGET_LUFS - MATCH_TARGET_LUFS, so a lead
# quieter than this would still sit under ONSET_THRESHOLD_DBFS after mastering.
FIRST_TRIM_DBFS = (
    ONSET_THRESHOLD_DBFS - (MASTER_TARGET_LUFS - MATCH_TARGET_LUFS) + FIRST_TRIM_MARGIN_DB
)


@dataclass
class MixResult:
    samples: np.ndarray
    scenes: list[SceneTiming]
    sfx: list[SfxHit]
    loudness: Loudness
    warnings: list[str]


@dataclass
class _Clip:
    """One scene's trimmed clip; times are relative to the clip start."""

    scene: Scene
    samples: Samples
    timing_source: TimingSource
    words: list[WordTiming]  # narration only
    ayat: list[AyahTiming]  # verse only

    @property
    def length_ms(self) -> int:
        return math.ceil(len(self.samples) * 1000 / aio.SAMPLE_RATE)

    def trim_head(self, threshold_dbfs: float) -> None:
        """Cut the lead below `threshold_dbfs` to 0 ms; the tail is never touched."""
        loud, lead = aio.trim_silence(
            self.samples, threshold_dbfs=threshold_dbfs, keep_ms=0, keep_lead_ms=0
        )
        if loud.size == 0 or lead == 0:
            return  # nothing loud enough to trim to: the onset guard decides
        self.samples = self.samples[aio.ms_to_samples(lead) :]
        self.shift(-lead)

    def shift(self, delta_ms: int) -> None:
        """Move every time by `delta_ms`, clamped to [0, length_ms]."""
        limit = self.length_ms
        self.words = shift(self.words, delta_ms, limit_ms=limit)
        self.ayat = [
            a.model_copy(
                update={
                    "start_ms": _clamp(a.start_ms, delta_ms, limit),
                    "end_ms": _clamp(a.end_ms, delta_ms, limit),
                    "words": _timeline_words(a.words, delta_ms, limit),
                }
            )
            for a in self.ayat
        ]


@dataclass
class _Slot:
    """One scene's slot on the part clock."""

    start_ms: int
    end_ms: int
    speech_start_ms: int
    draw_start_ms: int | None = None
    draw_end_ms: int | None = None
    sketch_id: int | None = None
    stroke_count: int | None = None


def _timeline_words(words: Sequence[WordTiming], delta_ms: int, limit: int) -> list[TimelineWord]:
    return [TimelineWord.model_validate(w) for w in shift(list(words), delta_ms, limit_ms=limit)]


def _clamp(ms: int, delta: int, limit: int) -> int:
    return min(max(ms + delta, 0), limit)


def _narration_clip(scene: Scene, narration: NarrationResult) -> _Clip:
    clip = narration.clips.get(scene.id)
    if clip is None:
        raise AudioError(f"scene {scene.id}: no narration clip")
    trimmed, lead = aio.trim_silence(clip.samples, keep_ms=TRIM_KEEP_MS)
    if trimmed.size == 0:
        raise AudioError(f"scene {scene.id}: narration clip is all silence")
    out = _Clip(scene, trimmed, clip.timing_source, list(clip.words), [])
    out.shift(-lead)
    return out


def _verse_clip(scene: Scene, verses: dict[str, VerseClip]) -> _Clip:
    clip = verses.get(scene.id)
    if clip is None:
        raise AudioError(f"scene {scene.id}: no verse clip")
    if not np.any(np.abs(clip.samples) > 0):
        raise AudioError(f"scene {scene.id}: verse clip is all silence")
    ayah_timed = clip.timing_source is TimingSource.AYAH
    ayat = [a.model_copy(update={"words": []}) if ayah_timed else a for a in clip.ayat]
    return _Clip(scene, clip.samples, clip.timing_source, [], ayat)


def _match_gain(samples: Samples, label: str, warnings: list[str]) -> float:
    measured = aio.measure(samples).integrated_lufs
    if measured is None:
        warnings.append(f"{label}: loudness not measurable; not matched")
        return 0.0
    return max(-MAX_MATCH_GAIN_DB, min(MAX_MATCH_GAIN_DB, MATCH_TARGET_LUFS - measured))


def _layout(clips: list[_Clip]) -> list[_Slot]:
    """Scene slots on the part clock, draw times unset."""
    slots: list[_Slot] = []
    t = 0
    for i, clip in enumerate(clips):
        prev = clips[i - 1].scene.kind if i > 0 else None
        nxt = clips[i + 1].scene.kind if i + 1 < len(clips) else None
        if clip.scene.kind is SceneKind.VERSE:
            leading = VERSE_BUFFER_MS if prev is SceneKind.NARRATION else 0
            if nxt is SceneKind.NARRATION:
                trailing = VERSE_BUFFER_MS
            elif nxt is SceneKind.VERSE:
                trailing = AYAH_GAP_MS
            else:
                trailing = 0
            speech_start = t + leading
            end = speech_start + clip.length_ms + trailing
        else:
            speech_start = t
            gap = 0 if nxt is SceneKind.VERSE else INTER_SCENE_GAP_MS
            end = max(t + clip.length_ms + gap, t + clip.scene.min_duration_ms)
        slots.append(_Slot(start_ms=t, end_ms=end, speech_start_ms=speech_start))
        t = end
    return slots


def _timing(clip: _Clip, slot: _Slot) -> SceneTiming:
    at = slot.speech_start_ms
    speech_end = at + clip.length_ms
    words = _timeline_words(clip.words, at, speech_end)
    ayat = [
        a.model_copy(
            update={
                "start_ms": a.start_ms + at,
                "end_ms": a.end_ms + at,
                "words": _timeline_words(a.words, at, speech_end),
            }
        )
        for a in clip.ayat
    ]
    return SceneTiming(
        scene_id=clip.scene.id,
        kind=clip.scene.kind,
        start_ms=slot.start_ms,
        end_ms=slot.end_ms,
        speech_start_ms=at,
        speech_end_ms=speech_end,
        words=words,
        timing_source=clip.timing_source,
        ayat=ayat,
        draw_start_ms=slot.draw_start_ms,
        draw_end_ms=slot.draw_end_ms,
        sketch_id=slot.sketch_id,
        stroke_count=slot.stroke_count,
    )


def _add(bus: Samples, samples: Samples, at_ms: int) -> None:
    start = aio.ms_to_samples(at_ms)
    room = max(0, min(len(samples), len(bus) - start))
    bus[start : start + room] += samples[:room]


def _fade_out(samples: Samples, ms: int) -> Samples:
    out = samples.copy()
    n = min(aio.ms_to_samples(ms), len(out))
    if n > 0:
        out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


def _sfx_bus(hits: list[SfxHit], pack: SfxPack, total: int) -> Samples:
    bus = np.zeros(total, dtype=np.float32)
    for hit in hits:
        cut = pack.samples(hit.asset_id)[: aio.ms_to_samples(hit.duration_ms)]
        _add(bus, aio.apply_gain(_fade_out(cut, SFX_FADE_MS), hit.gain_db), hit.at_ms)
    return bus


def _master_warnings(loudness: Loudness) -> list[str]:
    out: list[str] = []
    lufs = loudness.integrated_lufs
    if lufs is None:
        out.append("loudness not measurable; master not normalized")
    elif abs(lufs - MASTER_TARGET_LUFS) > LOUDNESS_TOLERANCE_LU:
        out.append(f"loudness off target: {lufs:.1f} LUFS")
    peak = loudness.true_peak_dbtp
    if peak is not None and peak > MASTER_TRUE_PEAK_DBTP:
        out.append(f"true peak above {MASTER_TRUE_PEAK_DBTP:.1f} dBTP: {peak:.1f} dBTP")
    return out


def _fit(samples: Samples, total: int) -> Samples:
    if len(samples) >= total:
        return samples[:total].astype(np.float32)
    pad = np.zeros(total - len(samples), dtype=np.float32)
    return np.concatenate([samples, pad]).astype(np.float32)


def mix_part(
    part: Part,
    narration: NarrationResult,
    verses: dict[str, VerseClip],
    strokes: dict[str, StrokeSet],
    pack: SfxPack,
    *,
    video_id: str,
) -> MixResult:
    """Lay out, loudness-match, add stroke SFX and master one part."""
    if not part.scenes:
        raise AudioError(f"part {part.index} has no scenes")
    clips = [
        _verse_clip(s, verses) if s.kind is SceneKind.VERSE else _narration_clip(s, narration)
        for s in part.scenes
    ]
    warnings: list[str] = []

    # Loudness match: one gain for all narration, one per verse scene.
    narr = [c for c in clips if c.scene.kind is SceneKind.NARRATION]
    narr_gain = 0.0
    if narr:
        narr_gain = _match_gain(np.concatenate([c.samples for c in narr]), "narration", warnings)
    for c in clips:
        if c.scene.kind is SceneKind.VERSE:
            gain = _match_gain(c.samples, c.scene.id, warnings)
        else:
            gain = narr_gain
        c.samples = aio.apply_gain(c.samples, gain)
    # The first clip's lead goes to 0 ms after its gain, so a soft lead-in that the
    # master would leave under the onset threshold is trimmed too.
    clips[0].trim_head(FIRST_TRIM_DBFS)
    for c in clips:
        c.samples = aio.fade(c.samples, FADE_MS)

    slots = _layout(clips)
    blocked = [
        (slot.start_ms, slot.end_ms)
        for clip, slot in zip(clips, slots, strict=True)
        if clip.scene.kind is SceneKind.VERSE
    ]
    hits: list[SfxHit] = []
    for clip, slot in zip(clips, slots, strict=True):
        scene = clip.scene
        stroke_set = strokes.get(scene.id)
        if scene.kind is not SceneKind.NARRATION or stroke_set is None:
            continue
        start, end = slot.start_ms, slot.end_ms
        draw_end = start + round(scene.draw_fraction * (end - start))
        slot.draw_start_ms, slot.draw_end_ms = start, draw_end
        slot.sketch_id, slot.stroke_count = scene.sketch_id, len(stroke_set.strokes)
        hits.extend(
            place_sfx(
                draw_schedule(stroke_set, start, draw_end),
                scene_id=scene.id,
                seed=marker_seed(video_id, scene.id),
                pack=pack,
                blocked=blocked,
            )
        )

    duration_ms = slots[-1].end_ms
    total = aio.ms_to_samples(duration_ms)
    voice_bus = np.zeros(total, dtype=np.float32)
    for clip, slot in zip(clips, slots, strict=True):
        _add(voice_bus, clip.samples, slot.speech_start_ms)
    mixed = (voice_bus + duck(_sfx_bus(hits, pack, total), voice_bus)).astype(np.float32)

    mastered, loudness = master(
        mixed, target_lufs=MASTER_TARGET_LUFS, true_peak_dbtp=MASTER_TRUE_PEAK_DBTP
    )
    mastered = _fit(mastered, total)
    head, onset = aio.trim_silence(
        mastered, threshold_dbfs=ONSET_THRESHOLD_DBFS, keep_ms=0, keep_lead_ms=0
    )
    if head.size == 0 or onset > FIRST_ONSET_MAX_MS:
        raise AudioError("dead air at part start")

    return MixResult(
        samples=mastered,
        scenes=[_timing(c, s) for c, s in zip(clips, slots, strict=True)],
        sfx=hits,
        loudness=loudness,
        warnings=[*narration.warnings, *warnings, *_master_warnings(loudness)],
    )
