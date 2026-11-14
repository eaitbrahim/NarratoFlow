"""Audio build: every part of an approved series to a mastered FLAC plus its timeline.

Everything that can be checked is checked before any spend or network call, and
those errors are raised as they are. Then, per selected part: the verse clips
(free, but network) come first, the up-front budget check of the part's uncached
narration, the narration, the mix, and the two files ``part-NN.flac`` and
``part-NN.timeline.json``. The part's `SpendTracker` starts at `Part.spent_usd`,
so the spending limit spans the sketch and audio stages; `Part.spent_usd` and
`PartTimeline.spent_usd` are cumulative, `AudioBuildResult.spent_usd` is this
run's delta. A `NarratoFlowError` during a part is re-raised as
`AudioInterruptedError` carrying the progress so far.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Literal

import httpx

from narratoflow.align import Aligner
from narratoflow.align.ctc import CtcAligner
from narratoflow.audio import io as aio
from narratoflow.audio.io import require_ffmpeg
from narratoflow.audio.mixer import MixResult, mix_part
from narratoflow.audio.narrate import ALIGN_EXTRA_WARNING, narrate_part, uncached_chars
from narratoflow.audio.recitation import RecitationSource, VerseClip, verse_clip
from narratoflow.audio.sfx import SfxPack, default_sfx
from narratoflow.audio.timeline import AssetUse, PartTimeline, RecitationUsed
from narratoflow.audio.words import tokens
from narratoflow.channels.config import ChannelConfig
from narratoflow.costs.prices import PriceTable, default_prices
from narratoflow.costs.tracker import SpendTracker, add_spend, estimate_speech_cost
from narratoflow.errors import (
    AudioError,
    ChannelConfigError,
    LibraryError,
    NarratoFlowError,
    PlanningError,
)
from narratoflow.library.repository import SketchLibrary, default_library
from narratoflow.models import Part, Scene, SceneKind, Series, VerseRef, VisualSource
from narratoflow.pipeline.cache import StageCache
from narratoflow.quran.corpus import QuranCorpus, default_corpus
from narratoflow.quran.reciters import Reciter, ReciterIndex, default_reciters, select_reciter
from narratoflow.settings import Settings
from narratoflow.sketching.generate import require_series_channel
from narratoflow.text.spoken import Lexicon, channel_lexicon
from narratoflow.text.tashkeel.verify import SkeletonMismatchError, assert_skeleton_preserved
from narratoflow.tts import VoiceChoice
from narratoflow.tts.factory import build_voice_chain
from narratoflow.visuals.marker import video_id
from narratoflow.visuals.strokes import StrokeSet

logger = logging.getLogger(__name__)

PEAK_UNMEASURABLE_WARNING = "true peak not measurable"


@dataclass(frozen=True)
class PartAudio:
    index: int
    audio_path: Path
    timeline_path: Path
    timeline: PartTimeline


@dataclass
class AudioBuildResult:
    parts: list[PartAudio] = field(default_factory=list)
    spent_usd: Decimal = Decimal(0)
    """This run's spend; `PartTimeline.spent_usd` is the cumulative part spend."""


class AudioInterruptedError(NarratoFlowError):
    """A `NarratoFlowError` stopped the build mid-part; finished parts are kept.

    Raised `from` the cause (`__cause__`); the message is the cause's plus
    " (progress so far is kept)". `partial` is the series with each part's spend
    so far, and the result with the parts already written. The caller saves
    `partial[0]` before reporting the error.
    """

    def __init__(self, cause: NarratoFlowError, partial: tuple[Series, AudioBuildResult]) -> None:
        super().__init__(f"{cause} (progress so far is kept)")
        self.partial = partial


@dataclass
class _Plan:
    """Everything the pre-spend checks resolved."""

    parts: list[Part]
    chain: list[VoiceChoice]
    prices: PriceTable
    strokes: dict[str, StrokeSet]
    reciter: Reciter | None
    corpus: QuranCorpus | None
    lexicon: Lexicon


def _selected(series: Series, parts: Sequence[int] | None) -> list[Part]:
    if parts is None:
        return list(series.parts)
    known = {p.index for p in series.parts}
    for n in parts:
        if n not in known:
            raise PlanningError(f"no part {n}; the series has parts {sorted(known)}")
    wanted = set(parts)
    return [p for p in series.parts if p.index in wanted]


def _narration(part: Part) -> list[Scene]:
    return [s for s in part.scenes if s.kind is SceneKind.NARRATION]


def _drawn(scene: Scene) -> bool:
    return scene.visual.source is not VisualSource.TEXT_CARD


def _check_sketches(parts: list[Part]) -> None:
    for part in parts:
        for scene in _narration(part):
            if scene.sketch_id is None and _drawn(scene):
                raise PlanningError(
                    f"scene {scene.id} has no sketch yet; "
                    "run narratoflow sketch generate / resolve first"
                )


def check_scene_text(scene: Scene) -> None:
    """The author's words: tts_text keeps the skeleton, captions keep the word count.

    `PlanningError` otherwise. A scene without narration texts passes.
    """
    narration = scene.narration
    if narration.tts_text is not None:
        try:
            assert_skeleton_preserved(narration.source_text, narration.tts_text)
        except SkeletonMismatchError as exc:
            raise PlanningError(
                f"scene {scene.id}: tts_text changes the author's words ({exc})"
            ) from exc
    if narration.caption_text is not None:
        shown = len(tokens(narration.caption_text))
        written = len(tokens(narration.source_text))
        if shown != written:
            raise PlanningError(
                f"scene {scene.id}: caption has {shown} words but the source text has {written}"
            )


def _check_texts(parts: list[Part]) -> None:
    for part in parts:
        for scene in _narration(part):
            check_scene_text(scene)


def _verses(part: Part) -> list[Scene]:
    return [s for s in part.scenes if s.kind is SceneKind.VERSE]


def _ayat(verse: VerseRef) -> str:
    """ "31:33" or "2:255-257"."""
    if verse.ayah_end == verse.ayah_start:
        return f"{verse.surah}:{verse.ayah_start}"
    return f"{verse.surah}:{verse.ayah_start}-{verse.ayah_end}"


def _load_strokes(parts: list[Part], library: SketchLibrary | None) -> dict[str, StrokeSet]:
    out: dict[str, StrokeSet] = {}
    for part in parts:
        for scene in _narration(part):
            if scene.sketch_id is None or not _drawn(scene):
                continue
            if library is None:  # pragma: no cover - the caller opens one when needed
                raise LibraryError("no sketch library")
            found = library.get_strokes(scene.sketch_id)
            if found is None:
                raise LibraryError(
                    f"scene {scene.id}: sketch {scene.sketch_id} has no strokes; "
                    "re-run sketch generate"
                )
            out[scene.id] = found
    return out


def _preflight(
    series: Series,
    channel: ChannelConfig,
    settings: Settings,
    *,
    parts: Sequence[int] | None,
    voices: list[VoiceChoice] | None,
    reciters: ReciterIndex | None,
    corpus: QuranCorpus | None,
    library: SketchLibrary | None,
    prices: PriceTable | None,
    stack: contextlib.ExitStack,
) -> _Plan:
    """Checks 1-11, in order, before any spend or network call; errors raise raw.

    A voice chain built here (not injected) is closed when `stack` exits.
    """
    if not series.approved:
        raise PlanningError("approve the series first")
    require_series_channel(series, channel)
    if channel.background_music:
        raise ChannelConfigError("no licensed music pack is bundled; set background_music = false")
    require_ffmpeg()
    prices = prices if prices is not None else default_prices(settings)
    prices.voice(channel.voice.provider, channel.voice.model)
    if voices is None:
        chain = build_voice_chain(channel, settings)
        for choice in chain:
            close = getattr(choice.provider, "close", None)
            if callable(close):
                stack.callback(close)
    else:
        chain = voices
    if not chain:
        raise ChannelConfigError("no voice to narrate with")
    for choice in chain:
        prices.voice(choice.provider.name, choice.provider.model)
    selected = _selected(series, parts)
    _check_sketches(selected)
    _check_texts(selected)
    reciter: Reciter | None = None
    verse_scenes = [s for p in selected for s in _verses(p)]
    if verse_scenes:
        first: str | None = None
        for scene in verse_scenes:
            if scene.verse is None:
                raise PlanningError(f"verse scene {scene.id} has no verse reference")
            if first is None:
                first = f"scene {scene.id} ({_ayat(scene.verse)})"
        reciter = select_reciter(
            channel,
            reciters if reciters is not None else default_reciters(),
            needed_by=first,
        )
        corpus = corpus if corpus is not None else default_corpus(settings)
    needs_library = any(
        s.sketch_id is not None and _drawn(s) for p in selected for s in _narration(p)
    )
    if needs_library and library is None:
        library = default_library(settings)
    strokes = _load_strokes(selected, library)
    return _Plan(selected, chain, prices, strokes, reciter, corpus, channel_lexicon(channel))


def _resolve_aligner(aligner: Aligner | Literal["auto"] | None) -> tuple[Aligner | None, bool]:
    """The aligner, and whether "auto" found none (each part then warns once)."""
    if aligner != "auto":
        return aligner, False
    if CtcAligner.available():
        return CtcAligner(), False
    return None, True


def _recitations(reciter: Reciter | None, verses: list[Scene]) -> list[RecitationUsed]:
    if reciter is None or not verses:
        return []
    ayat = [
        f"{s.verse.surah}:{n}"
        for s in verses
        if s.verse is not None
        for n in range(s.verse.ayah_start, s.verse.ayah_end + 1)
    ]
    return [
        RecitationUsed(
            reciter_id=reciter.id,
            name=reciter.name,
            riwaya=reciter.riwaya,
            license=reciter.license,
            license_url=reciter.license_url,
            source=reciter.source,
            segments_source=reciter.segments_source,
            checked=reciter.checked,
            ayat=ayat,
        )
    ]


def _assets(mix: MixResult, pack: SfxPack) -> list[AssetUse]:
    out: list[AssetUse] = []
    for sid in sorted({hit.asset_id for hit in mix.sfx}):
        asset = pack.asset(sid)
        out.append(
            AssetUse(
                kind="sfx",
                asset_id=sid,
                source=asset.source,
                license=asset.license,
                attribution=asset.attribution,
            )
        )
    return out


@dataclass
class _Run:
    series: Series
    plan: _Plan
    cache: StageCache
    aligner: Aligner | None
    align_missing: bool
    pack: SfxPack
    out_dir: Path
    source: RecitationSource | None

    def part(self, part: Part, tracker: SpendTracker) -> PartAudio:
        plan = self.plan
        verse_warnings: list[str] = []
        clips: dict[str, VerseClip] = {}
        for scene in _verses(part):  # recitations first: free, before any paid TTS
            if scene.verse is None or self.source is None or plan.corpus is None:
                raise PlanningError(f"verse scene {scene.id} has no reciter")  # pragma: no cover
            clip, warnings = verse_clip(scene.verse, plan.corpus, self.source)
            clips[scene.id] = clip
            verse_warnings.extend(warnings)
        primary = plan.chain[0]
        name, model = primary.provider.name, primary.provider.model
        chars = uncached_chars(
            part, primary, self.cache, language=self.series.language, lexicon=plan.lexicon
        )
        estimate = estimate_speech_cost(plan.prices, name, model, chars)
        if estimate > 0:  # verse-only, fully cached or free: nothing to refuse
            tracker.ensure_room(estimate, f"voice {name}/{model} part {part.index}")
        narration = narrate_part(
            part,
            plan.chain,
            language=self.series.language,
            tracker=tracker,
            prices=plan.prices,
            cache=self.cache,
            aligner=self.aligner,
            lexicon=plan.lexicon,
        )
        strokes = {s.id: plan.strokes[s.id] for s in part.scenes if s.id in plan.strokes}
        mix = mix_part(
            part,
            narration,
            clips,
            strokes,
            self.pack,
            video_id=video_id(self.series),
        )
        warnings = [*mix.warnings, *verse_warnings]
        if self.align_missing or narration.aligner_unavailable:
            warnings.append(ALIGN_EXTRA_WARNING)
        if mix.loudness.true_peak_dbtp is None:
            warnings.append(PEAK_UNMEASURABLE_WARNING)
        stem = f"part-{part.index:02d}"
        audio_path = self.out_dir / f"{stem}.flac"
        timeline_path = self.out_dir / f"{stem}.timeline.json"
        timeline = PartTimeline(
            series_title=self.series.title,
            channel_slug=self.series.channel_slug,
            part_index=part.index,
            part_count=part.count,
            audio_file=audio_path.name,
            duration_ms=mix.scenes[-1].end_ms,
            integrated_lufs=mix.loudness.integrated_lufs,
            true_peak_dbtp=mix.loudness.true_peak_dbtp,
            scenes=mix.scenes,
            sfx=mix.sfx,
            voice=narration.voice,
            recitations=_recitations(plan.reciter, _verses(part)),
            assets=_assets(mix, self.pack),
            spent_usd=tracker.spent,
            cost_estimated=narration.cost_estimated,
            warnings=warnings,
        )
        aio.encode_flac(mix.samples, audio_path)
        timeline.save(timeline_path)
        return PartAudio(part.index, audio_path, timeline_path, timeline)


def build_audio(
    series: Series,
    channel: ChannelConfig,
    settings: Settings,
    *,
    out_dir: Path,
    cache_root: Path,
    parts: Sequence[int] | None = None,
    voices: list[VoiceChoice] | None = None,
    aligner: Aligner | Literal["auto"] | None = "auto",
    reciters: ReciterIndex | None = None,
    http: httpx.Client | None = None,
    corpus: QuranCorpus | None = None,
    library: SketchLibrary | None = None,
    sfx: SfxPack | None = None,
    prices: PriceTable | None = None,
) -> tuple[Series, AudioBuildResult]:
    """Narrate, mix and write every selected part (`parts`: part indexes; None = all).

    `voices` replaces the channel's voice chain (tests inject fakes); providers
    must be untracked. `aligner="auto"` uses `CtcAligner` when its extra is
    installed, else none, and every part's timeline warns once. TTS clips are
    cached in `StageCache(cache_root)` (stage "tts"); recitations under
    `settings.data_dir / "reciters"`. The input series is not mutated.
    """
    result = AudioBuildResult()
    trackers: list[tuple[SpendTracker, Decimal]] = []

    def delta() -> Decimal:
        return sum((t.spent - seed for t, seed in trackers), Decimal(0))

    with contextlib.ExitStack() as stack:
        plan = _preflight(
            series,
            channel,
            settings,
            parts=parts,
            voices=voices,
            reciters=reciters,
            corpus=corpus,
            library=library,
            prices=prices,
            stack=stack,
        )
        chosen_aligner, align_missing = _resolve_aligner(aligner)
        pack = sfx if sfx is not None else default_sfx()
        out = series.model_copy(deep=True)
        by_index = {p.index: p for p in out.parts}
        source: RecitationSource | None = None
        if plan.reciter is not None:
            source = stack.enter_context(
                RecitationSource(plan.reciter, settings.data_dir / "reciters", http=http)
            )
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AudioError(f"cannot create output directory {out_dir}: {exc}") from exc
        run = _Run(
            series=series,
            plan=plan,
            cache=StageCache(cache_root),
            aligner=chosen_aligner,
            align_missing=align_missing,
            pack=pack,
            out_dir=out_dir,
            source=source,
        )
        for selected in plan.parts:
            part = by_index[selected.index]
            tracker = SpendTracker(channel.spending_limit_usd, spent=part.spent_usd)
            trackers.append((tracker, part.spent_usd))
            try:
                built = run.part(part, tracker)
            except NarratoFlowError as exc:
                part.spent_usd = tracker.spent
                add_spend(part.spend, tracker)
                result.spent_usd = delta()
                partial = AudioBuildResult(parts=list(result.parts), spent_usd=result.spent_usd)
                raise AudioInterruptedError(exc, (out.model_copy(deep=True), partial)) from exc
            part.spent_usd = tracker.spent
            add_spend(part.spend, tracker)
            result.parts.append(built)
            logger.info("audio part %d: %d ms", part.index, built.timeline.duration_ms)
    result.spent_usd = delta()
    return out, result
