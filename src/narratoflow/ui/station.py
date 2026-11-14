"""The review station's functions: read a scene, edit `series.json`, rebuild, export.

Pure functions keyed on file paths. Nothing is kept in memory: each call reloads
`series.json` and writes it back through `Series.save`. The only value a page keeps is
the file's sha256 when the scene was shown; every handler that writes takes it back as
`expected_sha256` and refuses (`PlanningError`, nothing written) when the file has
changed since, so two tabs or the CLI never overwrite each other (D33). Paid actions
estimate first and refuse without `confirm=True` (D34).

"Editing one scene re-renders only that scene" (D29, D37): `rebuild_scene` runs the
sketch stage for that scene only, the audio stage for its part (the TTS cache serves
every other scene, so no provider call for them) and the render for its part only. Other
parts are never passed to a stage, so their files are untouched.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import wave
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

import httpx
import numpy as np

from narratoflow.align import Aligner
from narratoflow.audio import io as aio
from narratoflow.audio.build import (
    AudioInterruptedError,
    build_audio,
    check_scene_text,
    estimate_voice_cost,
)
from narratoflow.audio.timeline import PartTimeline
from narratoflow.channels.config import ChannelConfig, load_channel
from narratoflow.costs.prices import PriceTable, default_prices
from narratoflow.errors import PlanningError
from narratoflow.images import ImageProvider
from narratoflow.library.repository import SketchLibrary, default_library
from narratoflow.llm.base import LLMClient
from narratoflow.models import AspectRatio, Scene, SceneKind, Series
from narratoflow.money import format_usd
from narratoflow.pipeline.paths import (
    default_audio_dir,
    default_cache_root,
    default_export_dir,
    default_render_dir,
)
from narratoflow.publish.build import build_publish
from narratoflow.publish.landscape import landscape_name, pillarbox_export
from narratoflow.publish.rules import Rules, default_rules
from narratoflow.quran.corpus import QuranCorpus
from narratoflow.quran.reciters import ReciterIndex
from narratoflow.render.build import build_render
from narratoflow.render.ffmpeg import RenderTools, find_render_tools
from narratoflow.render.layout import Canvas
from narratoflow.settings import Settings, get_settings
from narratoflow.sketching import resolve
from narratoflow.sketching.generate import (
    GenerationInterruptedError,
    GenerationResult,
    generate_missing,
    idea_to_draw,
    require_series_channel,
)
from narratoflow.sketching.resolve import (
    check_regenerable,
    regenerate,
    regenerate_estimate,
)
from narratoflow.sketching.upload import attach_upload, check_upload_licence, find_scene
from narratoflow.tts import VoiceChoice

SERIES_CHANGED = "series.json changed on disk since this view was loaded; reload"
MIN_DURATION_MS = 1000
MAX_DURATION_MS = 60000
MIN_DRAW_FRACTION = 0.1
MAX_DRAW_FRACTION = 1.0
CLIP_DIR = ".ui"


@dataclass(frozen=True)
class StationPaths:
    """Where one series lives; the render directory and the cache root derive from it
    exactly as the CLI derives them (`pipeline.paths`)."""

    series: Path
    channel: Path
    audio_dir: Path
    export_dir: Path

    @classmethod
    def derive(
        cls,
        series: Path,
        channel: Path,
        *,
        audio_dir: Path | None = None,
        export_dir: Path | None = None,
    ) -> StationPaths:
        return cls(
            series,
            channel,
            audio_dir if audio_dir is not None else default_audio_dir(series),
            export_dir if export_dir is not None else default_export_dir(series),
        )

    @property
    def render_dir(self) -> Path:
        return default_render_dir(self.audio_dir)

    @property
    def cache_root(self) -> Path:
        return default_cache_root(self.series)


@dataclass(frozen=True)
class SceneRow:
    scene_id: str
    part: int
    kind: str
    visual: str
    flags: list[str]


@dataclass(frozen=True)
class SceneView:
    """One scene as the page shows it; `sha256` is `series.json`'s hash when read."""

    scene_id: str
    part: int
    source_text: str
    tts_text: str
    caption_text: str
    min_duration_ms: int
    draw_fraction: float
    sketch_png: bytes | None
    audio_clip: Path | None
    sha256: str


@dataclass(frozen=True)
class UploadReport:
    sha256: str
    sketch_id: int


@dataclass(frozen=True)
class RegenerateReport:
    sha256: str
    generated: GenerationResult
    spent_usd: Decimal


@dataclass(frozen=True)
class RebuildReport:
    scene_id: str
    part: int
    generated: GenerationResult
    audio_parts: list[int]
    rendered_parts: list[int]
    spent_usd: Decimal
    sha256: str


def series_sha256(path: Path) -> str:
    """The hex sha256 of the series file's bytes (`PlanningError` when unreadable)."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise PlanningError(f"cannot read series file {path}: {exc}") from exc


def _load(paths: StationPaths, expected_sha256: str | None = None) -> Series:
    if expected_sha256 is not None and series_sha256(paths.series) != expected_sha256:
        raise PlanningError(SERIES_CHANGED)
    return Series.load(paths.series)


def _narration(series: Series, scene_id: str) -> tuple[int, int, Scene]:
    """Where a scene the station may edit sits; verses are the reciter's, not editable."""
    located = find_scene(series, scene_id)
    if located[2].kind is SceneKind.VERSE:
        raise PlanningError(f"scene {scene_id} is a verse; the station does not edit verses")
    return located


def _edit(
    paths: StationPaths,
    scene_id: str,
    expected_sha256: str,
    change: Callable[[Scene], Scene],
) -> str:
    """Apply `change` to one narration scene of the file and save; the new sha256."""
    series = _load(paths, expected_sha256)
    p, i, scene = _narration(series, scene_id)
    series.parts[p].scenes[i] = change(scene)
    series.save(paths.series)
    return series_sha256(paths.series)


def _confirmed(estimate: Decimal, confirm: bool) -> None:
    if estimate > 0 and not confirm:
        raise PlanningError(f"estimated cost {format_usd(estimate)}; confirm to continue")


# -- read ----------------------------------------------------------------------


def list_scenes(paths: StationPaths) -> list[SceneRow]:
    """Every scene of the series, in order, with its part."""
    return [
        SceneRow(s.id, part.index, s.kind.value, s.visual.source.value, list(s.flags))
        for part in _load(paths).parts
        for s in part.scenes
    ]


def _write_wav(samples: aio.Samples, path: Path) -> None:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as handle, wave.open(handle, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(aio.SAMPLE_RATE)
            out.writeframes(pcm.tobytes())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _audio_clip(paths: StationPaths, part: int, scene_id: str) -> Path | None:
    """The scene's slot cut from its part FLAC into `audio_dir/.ui/` (a derived cache)."""
    stem = f"part-{part:02d}"
    timeline_path, flac = (
        paths.audio_dir / f"{stem}.timeline.json",
        paths.audio_dir / f"{stem}.flac",
    )
    if not (timeline_path.is_file() and flac.is_file()):
        return None
    slot = next(
        (t for t in PartTimeline.load(timeline_path).scenes if t.scene_id == scene_id), None
    )
    if slot is None:
        return None
    stat = flac.stat()
    key = f"{part}:{scene_id}:{stat.st_size}:{stat.st_mtime_ns}:{slot.start_ms}:{slot.end_ms}"
    out = paths.audio_dir / CLIP_DIR / f"{hashlib.sha256(key.encode()).hexdigest()}.wav"
    if not out.is_file():
        samples = aio.decode(flac.read_bytes())
        _write_wav(samples[aio.ms_to_samples(slot.start_ms) : aio.ms_to_samples(slot.end_ms)], out)
    return out


def scene_view(
    paths: StationPaths,
    scene_id: str,
    *,
    settings: Settings | None = None,
    library: SketchLibrary | None = None,
) -> SceneView:
    """Texts, timing, sketch image and audio clip of one scene."""
    sha = series_sha256(paths.series)
    series = _load(paths)
    p, _, scene = find_scene(series, scene_id)
    png: bytes | None = None
    if scene.sketch_id is not None:
        lib = library if library is not None else default_library(settings or get_settings())
        record = lib.get_sketch(scene.sketch_id)
        png = None if record is None else lib.sketch_bytes(record)
    narration = scene.narration
    return SceneView(
        scene_id=scene.id,
        part=series.parts[p].index,
        source_text=narration.source_text,
        tts_text=narration.tts_text or "",
        caption_text=narration.caption_text or "",
        min_duration_ms=scene.min_duration_ms,
        draw_fraction=scene.draw_fraction,
        sketch_png=png,
        audio_clip=_audio_clip(paths, series.parts[p].index, scene.id),
        sha256=sha,
    )


# -- edit ----------------------------------------------------------------------


def edit_text(
    paths: StationPaths,
    scene_id: str,
    *,
    tts_text: str,
    caption_text: str,
    expected_sha256: str,
) -> str:
    """Set the scene's vocalized and caption texts (blank clears one); the new sha256.

    The author's words stay read-only: `check_scene_text` refuses a `tts_text` that
    changes the skeleton and a caption whose word count differs.
    """

    def change(scene: Scene) -> Scene:
        narration = scene.narration.model_copy(
            update={
                "tts_text": tts_text.strip() or None,
                "caption_text": caption_text.strip() or None,
            }
        )
        edited = scene.model_copy(update={"narration": narration})
        check_scene_text(edited)
        return edited

    return _edit(paths, scene_id, expected_sha256, change)


def edit_timing(
    paths: StationPaths,
    scene_id: str,
    *,
    min_duration_ms: int,
    draw_fraction: float,
    expected_sha256: str,
) -> str:
    """Set the scene's minimum duration (1000..60000 ms) and draw fraction (0.1..1.0)."""
    if not MIN_DURATION_MS <= min_duration_ms <= MAX_DURATION_MS:
        raise PlanningError(
            f"minimum duration must be {MIN_DURATION_MS}..{MAX_DURATION_MS} ms, "
            f"got {min_duration_ms}"
        )
    if not MIN_DRAW_FRACTION <= draw_fraction <= MAX_DRAW_FRACTION:
        raise PlanningError(
            f"draw fraction must be {MIN_DRAW_FRACTION}..{MAX_DRAW_FRACTION}, got {draw_fraction}"
        )
    return _edit(
        paths,
        scene_id,
        expected_sha256,
        lambda s: s.model_copy(
            update={"min_duration_ms": min_duration_ms, "draw_fraction": draw_fraction}
        ),
    )


# -- visual --------------------------------------------------------------------


def upload_visual(
    paths: StationPaths,
    scene_id: str,
    image: bytes,
    *,
    licence: str,
    source: str,
    attribution: str = "",
    expected_sha256: str,
    settings: Settings | None = None,
    library: SketchLibrary | None = None,
) -> UploadReport:
    """Attach the creator's own image to the scene as line art (free: no spend)."""
    settings = settings if settings is not None else get_settings()
    check_upload_licence(default_rules(settings), licence, attribution)
    series = _load(paths, expected_sha256)
    updated, record = attach_upload(
        series,
        scene_id,
        load_channel(paths.channel),
        image,
        licence=licence,
        source=source,
        attribution=attribution,
        library=library if library is not None else default_library(settings),
    )
    updated.save(paths.series)
    return UploadReport(series_sha256(paths.series), record.id)


def _regenerate_cost(
    series: Series, channel: ChannelConfig, scene_id: str, prices: PriceTable
) -> Decimal:
    idea_key = check_regenerable(series, scene_id, require_quarantine=False)
    require_series_channel(series, channel)
    return regenerate_estimate(idea_key, channel, prices)


def regenerate_cost(
    paths: StationPaths,
    scene_id: str,
    *,
    settings: Settings | None = None,
    prices: PriceTable | None = None,
) -> Decimal:
    """What `regenerate_visual` would spend on this scene (`PlanningError` if it cannot)."""
    settings = settings if settings is not None else get_settings()
    prices = prices if prices is not None else default_prices(settings)
    return _regenerate_cost(_load(paths), load_channel(paths.channel), scene_id, prices)


def regenerate_visual(
    paths: StationPaths,
    scene_id: str,
    *,
    confirm: bool,
    expected_sha256: str,
    settings: Settings | None = None,
    images: ImageProvider | None = None,
    vision: LLMClient | Literal["auto"] | None = "auto",
    library: SketchLibrary | None = None,
    prices: PriceTable | None = None,
) -> RegenerateReport:
    """Draw a new sketch for the scene (paid): refuses without `confirm` when it costs."""
    settings = settings if settings is not None else get_settings()
    series = _load(paths, expected_sha256)
    channel = load_channel(paths.channel)
    prices = prices if prices is not None else default_prices(settings)
    _confirmed(_regenerate_cost(series, channel, scene_id, prices), confirm)
    try:
        updated, result = regenerate(
            series,
            scene_id,
            channel,
            settings,
            images=images,
            vision=vision,
            library=library,
            prices=prices,
            require_quarantine=False,
        )
    except GenerationInterruptedError as exc:
        exc.partial[0].save(paths.series)
        raise
    updated.save(paths.series)
    return RegenerateReport(series_sha256(paths.series), result, result.spent_usd)


def use_text_card(
    paths: StationPaths, scene_id: str, card_text: str, *, expected_sha256: str
) -> str:
    """Show a text card instead of a sketch; the new sha256."""
    series = _load(paths, expected_sha256)
    _narration(series, scene_id)
    resolve.use_text_card(series, scene_id, card_text, require_quarantine=False).save(paths.series)
    return series_sha256(paths.series)


# -- pipeline ------------------------------------------------------------------


def rebuild_scene(
    paths: StationPaths,
    scene_id: str,
    *,
    confirm: bool,
    expected_sha256: str,
    settings: Settings | None = None,
    images: ImageProvider | None = None,
    vision: LLMClient | Literal["auto"] | None = "auto",
    voices: list[VoiceChoice] | None = None,
    aligner: Aligner | Literal["auto"] | None = "auto",
    tools: RenderTools | None = None,
    library: SketchLibrary | None = None,
    prices: PriceTable | None = None,
    canvas: Canvas | None = None,
    workers: int | None = None,
    corpus: QuranCorpus | None = None,
) -> RebuildReport:
    """Re-run the stages an edit to one scene needs, for that scene's part only.

    Sketch for this scene, audio and render for its part (D29, D37). The estimate (a
    sketch the scene lacks, plus the voice characters its part's TTS cache lacks) is
    checked before any call; each stage's progress is saved even when a later one fails.
    """
    settings = settings if settings is not None else get_settings()
    series = _load(paths, expected_sha256)
    channel = load_channel(paths.channel)
    p, _, scene = find_scene(series, scene_id)
    part = series.parts[p].index
    prices = prices if prices is not None else default_prices(settings)
    estimate = estimate_voice_cost(
        series, channel, settings, [part], cache_root=paths.cache_root, voices=voices, prices=prices
    )
    key = idea_to_draw(scene)
    if scene.sketch_id is None and key is not None:
        estimate += regenerate_estimate(key, channel, prices)
    _confirmed(estimate, confirm)
    try:
        series, generated = generate_missing(
            series,
            channel,
            settings,
            images=images,
            vision=vision,
            library=library,
            prices=prices,
            only=frozenset({scene_id}),
        )
    except GenerationInterruptedError as exc:
        exc.partial[0].save(paths.series)
        raise
    series.save(paths.series)
    try:
        series, audio = build_audio(
            series,
            channel,
            settings,
            out_dir=paths.audio_dir,
            cache_root=paths.cache_root,
            parts=[part],
            voices=voices,
            aligner=aligner,
            corpus=corpus,
            library=library,
            prices=prices,
        )
    except AudioInterruptedError as exc:
        exc.partial[0].save(paths.series)
        raise
    series.save(paths.series)
    rendered = build_render(
        series,
        channel,
        settings,
        audio_dir=paths.audio_dir,
        out_dir=paths.render_dir,
        parts=[part],
        workers=workers,
        canvas=canvas,
        tools=tools,
        library=library,
        corpus=corpus,
    )
    return RebuildReport(
        scene_id=scene_id,
        part=part,
        generated=generated,
        audio_parts=[b.index for b in audio.parts],
        rendered_parts=[b.index for b in rendered.parts if b.index not in rendered.skipped],
        spent_usd=generated.spent_usd + audio.spent_usd,
        sha256=series_sha256(paths.series),
    )


def export(
    paths: StationPaths,
    ratio: AspectRatio,
    *,
    settings: Settings | None = None,
    tools: RenderTools | None = None,
    library: SketchLibrary | None = None,
    corpus: QuranCorpus | None = None,
    reciters: ReciterIndex | None = None,
    rules: Rules | None = None,
    prices: PriceTable | None = None,
    canvas: Canvas | None = None,
    workers: int | None = None,
    now: Callable[[], datetime] | None = None,
    http: httpx.Client | None = None,
) -> list[Path]:
    """Render every part (unchanged ones are kept), check and export them, and return
    the exported files of `ratio`: `part-NN.mp4`, or for 16:9 the `part-NN.16x9.mp4`
    pillarbox copies (D31). A part that FAILs its check is exported in neither ratio."""
    settings = settings if settings is not None else get_settings()
    series = _load(paths)
    channel = load_channel(paths.channel)
    tools = tools if tools is not None else find_render_tools(settings)
    rendered = build_render(
        series,
        channel,
        settings,
        audio_dir=paths.audio_dir,
        out_dir=paths.render_dir,
        workers=workers,
        canvas=canvas,
        tools=tools,
        library=library,
        corpus=corpus,
    )
    published = build_publish(
        series,
        channel,
        settings,
        render_dir=paths.render_dir,
        audio_dir=paths.audio_dir,
        export_dir=paths.export_dir,
        tools=tools,
        library=library,
        corpus=corpus,
        reciters=reciters,
        rules=rules,
        prices=prices,
        now=now,
    )
    exported = {p.index: p.export_path for p in published.parts}
    if ratio is AspectRatio.PORTRAIT:
        return [path for path in exported.values() if path is not None]
    reports = {b.index: b.report for b in rendered.parts}
    out: list[Path] = []
    for index, path in exported.items():
        target = paths.export_dir / landscape_name(index)
        if path is None:
            target.unlink(missing_ok=True)  # a failing part keeps no older copy
            continue
        report = reports[index]
        out.append(
            pillarbox_export(
                tools, path, target, frames=report.frames, duration_ms=report.duration_ms
            )
        )
    return out
