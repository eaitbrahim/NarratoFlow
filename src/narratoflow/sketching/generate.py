"""Turn an approved series into scenes with library sketches (spec 4.3, 6.3).

Per narration scene, in order: keep the sketch already attached by an earlier
run, reuse a stored sketch, copy a bundled icon into the channel library, or
generate a new one. A new sketch (or a bundled icon on its first copy) passes the
free density check before the paid vision check; a failure is stored FAILED
(quarantined) and never attached to a scene. Every image and vision call is
budget-checked on the part's own `SpendTracker`, which starts at the part's
cumulative `spent_usd` (the limit spans stages) and writes it back.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Literal

import numpy as np
from PIL import Image, UnidentifiedImageError

from narratoflow.channels.config import (
    ChannelConfig,
    StyleConfig,
    style_fingerprint,
    vision_model,
)
from narratoflow.channels.keys import check_provider_keys
from narratoflow.costs.prices import PriceTable, default_prices
from narratoflow.costs.tracker import (
    SpendTracker,
    TrackedLLM,
    add_spend,
    require_generation_prices,
)
from narratoflow.errors import (
    ChannelConfigError,
    NarratoFlowError,
    PlanningError,
    SketchRejectedError,
)
from narratoflow.images import GeneratedImage, ImageProvider, ImageRequest
from narratoflow.images.factory import TrackedImageProvider, build_image_provider
from narratoflow.images.prompt import build_prompt
from narratoflow.library.keys import IdeaKey, parse_idea_key
from narratoflow.library.matching import MatchKind, match_idea
from narratoflow.library.repository import SketchLibrary, default_library
from narratoflow.library.schemas import Provenance, SketchOrigin, SketchRecord, SketchStatus
from narratoflow.llm.base import LLMClient
from narratoflow.llm.factory import build_llm_client, vision_llm_config
from narratoflow.models import Language, Scene, SceneKind, Series, VisualSource
from narratoflow.settings import Settings
from narratoflow.sketching.vision import check_sketch, estimate_check_cost
from narratoflow.visuals.bundled import BundledIcon, BundledIcons, default_bundled_icons
from narratoflow.visuals.lineart import binarize, check_density
from narratoflow.visuals.marker import marker_seed, video_id
from narratoflow.visuals.strokes import extract_strokes

logger = logging.getLogger(__name__)

QUARANTINED_FLAG = "sketch_quarantined"
# Anthropic limits the base64 payload to 5 MB; 3.5 MB raw encodes to about 4.7 MB
MAX_REFERENCE_BYTES = 3_500_000
MAX_REFERENCE_SIDE_PX = 2048
REFERENCE_LONG_SIDE_PX = 1024
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

# (license id, usage terms) recorded in the provenance of a generated sketch
_TERMS: dict[str, tuple[str, str]] = {
    "openai": ("openai-terms", "Output use is subject to the OpenAI terms of use."),
    "local-diffusion": (
        "local-model-license",
        "Output use is subject to the license of the local diffusion model.",
    ),
}


@dataclass
class GenerationResult:
    """What happened in this run; every counter counts this run's attachments only."""

    generated: int = 0
    """New sketches attached to a scene (PASSED or UNCHECKED)."""
    reused: int = 0
    """Scenes this run attached to an existing library sketch."""
    bundled: int = 0
    """Scenes this run attached to a bundled icon copied into the channel library."""
    kept: int = 0
    """Scenes whose sketch, attached by an earlier run, is still usable (unchanged)."""
    quarantined: list[str] = field(default_factory=list)
    """Scene ids left without a sketch because theirs FAILED or the scene is flagged."""
    unchecked: list[str] = field(default_factory=list)
    """Scene ids whose sketch (new, bundled, reused or kept) is UNCHECKED (no vision)."""
    spent_usd: Decimal = Decimal(0)
    """This run's spend; each `Part.spent_usd` holds the part's cumulative spend."""


class GenerationInterruptedError(NarratoFlowError):
    """A `NarratoFlowError` stopped generation; everything stored so far is kept.

    Raised `from` the cause (`__cause__`, e.g. `BudgetExceededError` or a
    `ProviderError`); the message is the cause's plus " (progress so far is kept)".
    `partial` is the series with the scenes updated so far, and the result so far.
    The caller saves `partial[0]` before reporting the error.
    """

    def __init__(self, cause: NarratoFlowError, partial: tuple[Series, GenerationResult]) -> None:
        super().__init__(f"{cause} (progress so far is kept)")
        self.partial = partial


def load_reference(style: StyleConfig) -> bytes:
    """The channel's reference sketch as PNG bytes, read once per run.

    A PNG within `MAX_REFERENCE_BYTES` and `MAX_REFERENCE_SIDE_PX` is used as is;
    anything else is re-encoded as a PNG of at most `REFERENCE_LONG_SIDE_PX` on its
    long side. A missing or undecodable file is a `ChannelConfigError`.
    """
    path = style.reference_sketch
    try:
        data = path.read_bytes()
    except OSError as e:
        raise ChannelConfigError(f"cannot read reference sketch {path}: {e}") from e
    try:
        with Image.open(io.BytesIO(data)) as image:
            if (
                data.startswith(_PNG_MAGIC)
                and len(data) <= MAX_REFERENCE_BYTES
                and max(image.size) <= MAX_REFERENCE_SIDE_PX
            ):
                return data
            alpha = "A" in image.getbands() or "transparency" in image.info
            converted = image.convert("RGBA" if alpha else "RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise ChannelConfigError(f"reference sketch {path} is not a readable image") from e
    converted.thumbnail((REFERENCE_LONG_SIDE_PX, REFERENCE_LONG_SIDE_PX))
    buffer = io.BytesIO()
    converted.save(buffer, format="PNG")
    return buffer.getvalue()


def _now() -> datetime:
    return datetime.now(UTC)


def _provenance(image: GeneratedImage) -> Provenance:
    provider = image.provider.strip().lower()
    license_id, terms = _TERMS.get(
        provider, (f"{provider}-terms", "Output use is subject to the provider's terms.")
    )
    return Provenance(
        source="generated",
        license=license_id,
        provider=image.provider,
        model=image.model,
        created_at=_now(),
        usage_terms=terms,
    )


def _ensure_strokes(
    library: SketchLibrary, record: SketchRecord, *, rtl: bool, ink: np.ndarray | None = None
) -> None:
    """Extract and store the stroke set of a usable sketch unless it already has one."""
    if record.strokes_hash is not None:
        return
    if ink is None:
        ink = binarize(library.sketch_bytes(record))
    library.put_strokes(record.id, extract_strokes(ink, rtl=rtl))


def _tracked_vision(
    channel: ChannelConfig, vision: LLMClient | None, tracker: SpendTracker, prices: PriceTable
) -> TrackedLLM | None:
    """`vision` charged to `tracker` at the vision model's prices; None when either the
    client is None or the channel turns vision off (`vision_model` is None)."""
    ident = vision_model(channel)
    if vision is None or ident is None:
        return None
    return TrackedLLM(vision, tracker, *ident, prices)


def _require_untracked(images: ImageProvider | None, vision: LLMClient | None) -> None:
    """Injected clients are wrapped on the part's tracker; a tracked one would pay twice."""
    if isinstance(images, TrackedImageProvider) or isinstance(vision, TrackedLLM):
        raise TypeError("pass untracked image and vision clients; they are tracked here")


def generate_one(
    scene: Scene,
    channel: ChannelConfig,
    *,
    library: SketchLibrary,
    style_version_id: int,
    images: ImageProvider,
    vision: LLMClient | None,
    tracker: SpendTracker,
    reference_png: bytes,
    prices: PriceTable,
    video_id: str,
) -> SketchRecord:
    """Generate, check and store one new sketch for `scene`; the caller checks `.status`.

    `images` and `vision` must be untracked clients (an already tracked one raises
    `TypeError`): they are wrapped here so every call is checked against and charged
    to `tracker` at the channel's configured prices. `vision` is not used when the
    channel turns vision off (`vision_model` is None). The image seed is
    `marker_seed(video_id, scene.id)`, so a re-run asks for the same image. When the
    image dedupes onto a stored record (same style version, key and bytes), that
    record is returned as is, even FAILED, with no vision call. A density reject is
    stored FAILED with `density: <reason>` and skips the vision call. A vision call
    that raises stores the paid image FAILED with `vision: error (<type>)`, then the
    error propagates. A usable record (PASSED or UNCHECKED) has its strokes stored.
    """
    return _generate_one(
        scene,
        channel,
        library=library,
        style_version_id=style_version_id,
        images=images,
        vision=vision,
        tracker=tracker,
        reference_png=reference_png,
        prices=prices,
        video_id=video_id,
        stored=[],
    )


def _generate_one(
    scene: Scene,
    channel: ChannelConfig,
    *,
    library: SketchLibrary,
    style_version_id: int,
    images: ImageProvider,
    vision: LLMClient | None,
    tracker: SpendTracker,
    reference_png: bytes,
    prices: PriceTable,
    video_id: str,
    stored: list[SketchRecord],
) -> SketchRecord:
    """`generate_one`; every record it stores is appended to `stored` as it is stored."""
    _require_untracked(images, vision)
    if scene.idea_key is None:
        raise PlanningError(f"scene {scene.id} has no idea key")
    key = str(parse_idea_key(scene.idea_key))
    style = channel.style
    image_provider = style.image_provider.strip().lower()
    tracked_images = TrackedImageProvider(
        images, tracker, prices, image_provider, style.image_model
    )
    tracked_vision = _tracked_vision(channel, vision, tracker, prices)
    estimate = prices.image(image_provider, style.image_model)
    if tracked_vision is not None:
        estimate += estimate_check_cost(
            style, key, prices, tracked_vision.provider, tracked_vision.model, with_reference=True
        )
    tracker.ensure_room(estimate, f"sketch {key}")

    image = tracked_images.generate(
        ImageRequest(
            prompt=build_prompt(style, key, scene.idea_description),
            reference_png=reference_png,
            seed=marker_seed(video_id, scene.id, scene.sketch_attempt),
        )
    )
    provenance = _provenance(image)

    def store(status: SketchStatus, reason: str | None, mirror_safe: bool) -> SketchRecord:
        record = library.add_sketch(
            style_version_id,
            key,
            image.png,
            ".png",
            SketchOrigin.GENERATED,
            status,
            mirror_safe,
            provenance,
            reason,
            description=scene.idea_description,
        )
        stored.append(record)
        return record

    rtl = channel.language is Language.ARABIC
    try:
        ink = binarize(image.png)
    except SketchRejectedError as e:
        record = store(SketchStatus.FAILED, f"density: {e}", False)
        logger.info("sketch %s for scene %s: density reject", key, scene.id)
        return record
    existing = library.find_by_content(style_version_id, key, image.png)
    if existing is not None:
        # already judged: its status is authoritative, no second vision call
        if existing.status is not SketchStatus.FAILED:
            _ensure_strokes(library, existing, rtl=rtl, ink=ink)
        logger.info("sketch %s for scene %s: dedupe onto %d", key, scene.id, existing.id)
        return existing
    report = check_density(ink, line_weight_px=style.line_weight_px)
    if not report.ok:
        record = store(SketchStatus.FAILED, f"density: {report.reason}", False)
        logger.info("sketch %s for scene %s: density reject", key, scene.id)
        return record
    try:
        status, reason, mirror_safe, _ = check_sketch(
            image.png, style, key, tracked_vision, reference_png=reference_png
        )
    except Exception as e:
        # the image is paid for: keep it, quarantined, so a re-run does not pay again
        store(SketchStatus.FAILED, f"vision: error ({type(e).__name__})", False)
        raise
    record = store(status, reason, mirror_safe)
    if record.status is not SketchStatus.FAILED:
        _ensure_strokes(library, record, rtl=rtl, ink=ink)
    logger.info("sketch %s for scene %s: %s", key, scene.id, record.status.value)
    return record


def _add_bundled(
    library: SketchLibrary,
    style_version_id: int,
    scene: Scene,
    key: IdeaKey,
    icon: BundledIcon,
    channel: ChannelConfig,
    *,
    vision: TrackedLLM | None,
    reference_png: bytes,
) -> SketchRecord:
    """Copy a bundled icon into the channel library; the caller checks `.status`.

    The first copy passes the same gates as a generated sketch (spec 4.4): the free
    density check, then the vision/outlier check with the reference sketch (paid,
    budget-checked by `vision`; UNCHECKED without vision). It is stored with
    origin BUNDLED and the icon's license and source. A vision error propagates and
    stores nothing: the icon cost nothing, so it stays available for a re-run.
    """
    try:
        data = icon.file.read_bytes()
    except OSError as e:
        raise ChannelConfigError(f"cannot read bundled icon {icon.file}: {e}") from e
    existing = library.find_by_content(style_version_id, key, data)
    if existing is not None:
        return existing
    ink: np.ndarray | None = None
    status: SketchStatus
    reason: str | None
    mirror_safe = False
    try:
        ink = binarize(data)
        rejected = check_density(ink, line_weight_px=channel.style.line_weight_px).reason
    except SketchRejectedError as e:
        rejected = str(e)
    if rejected is not None:
        status, reason = SketchStatus.FAILED, f"density: {rejected}"
    else:
        status, reason, mirror_safe, _ = check_sketch(
            data, channel.style, str(key), vision, reference_png=reference_png
        )
    record = library.add_sketch(
        style_version_id,
        key,
        data,
        icon.file.suffix.lower() or ".png",
        SketchOrigin.BUNDLED,
        status,
        mirror_safe,
        Provenance(source=icon.source, license=icon.license, created_at=_now()),
        reason,
        description=scene.idea_description,
    )
    if record.status is not SketchStatus.FAILED:
        _ensure_strokes(library, record, rtl=channel.language is Language.ARABIC, ink=ink)
    return record


def _attached(scene: Scene, record: SketchRecord, match: Literal["reused", "new"]) -> Scene:
    flags = [f for f in scene.flags if f != QUARANTINED_FLAG]
    return scene.model_copy(update={"sketch_id": record.id, "match": match, "flags": flags})


def _quarantined(scene: Scene) -> Scene:
    flags = scene.flags if QUARANTINED_FLAG in scene.flags else [*scene.flags, QUARANTINED_FLAG]
    return scene.model_copy(update={"sketch_id": None, "match": "new", "flags": flags})


@dataclass
class _Run:
    channel: ChannelConfig
    library: SketchLibrary
    version: int
    bundled: BundledIcons
    images: Callable[[], ImageProvider]
    vision: LLMClient | None
    prices: PriceTable
    reference: bytes
    video_id: str
    result: GenerationResult
    stored: list[SketchRecord] = field(default_factory=list)
    """Records stored for the scene in progress (read when that scene raises)."""

    def tracked_vision(self, tracker: SpendTracker) -> TrackedLLM | None:
        return _tracked_vision(self.channel, self.vision, tracker, self.prices)

    def interrupted(self, scene: Scene) -> Scene:
        """The scene as it stands when its processing raised mid-way."""
        if any(r.status is SketchStatus.FAILED for r in self.stored):
            # a paid image whose vision call raised was stored FAILED
            self.result.quarantined.append(scene.id)
            return _quarantined(scene)
        return scene

    def scene(self, scene: Scene, tracker: SpendTracker) -> Scene:
        self.stored = []
        if scene.kind is SceneKind.VERSE or scene.idea_key is None:
            return scene
        if scene.visual.source is VisualSource.TEXT_CARD:
            # the creator chose a text card: nothing to draw, nothing to spend
            return scene
        if QUARANTINED_FLAG in scene.flags:
            # the creator resolves a quarantined scene; nothing is attached or
            # regenerated automatically (spec 6.3)
            self.result.quarantined.append(scene.id)
            return scene
        rtl = self.channel.language is Language.ARABIC
        key = parse_idea_key(scene.idea_key)
        current = self._attached_record(scene, key)
        if current is not None:
            if current.status is SketchStatus.FAILED:
                # quarantined after it was attached: no regeneration, no spend
                self.result.quarantined.append(scene.id)
                return _quarantined(scene)
            # stable re-runs: keep the sketch and the scene's match as they are
            _ensure_strokes(self.library, current, rtl=rtl)
            self.result.kept += 1
            self._note_unchecked(scene, current)
            return scene
        found = match_idea(self.library, self.version, key, self.bundled)
        if found.kind is MatchKind.REUSED:
            passed = [r for r in found.candidates if r.status is SketchStatus.PASSED]
            record = (passed or list(found.candidates))[0]
            _ensure_strokes(self.library, record, rtl=rtl)
            self.result.reused += 1
            self._note_unchecked(scene, record)
            return _attached(scene, record, "reused")
        # `found.icon is not None` only narrows the type: match_idea sets it for BUNDLED
        if found.kind is MatchKind.BUNDLED and found.icon is not None:
            record = _add_bundled(
                self.library,
                self.version,
                scene,
                key,
                found.icon,
                self.channel,
                vision=self.tracked_vision(tracker),
                reference_png=self.reference,
            )
            if record.status is not SketchStatus.FAILED:
                self.result.bundled += 1
                self._note_unchecked(scene, record)
                return _attached(scene, record, "reused")
            logger.info("bundled icon %s failed its check; generating instead", key)
        record = _generate_one(
            scene,
            self.channel,
            library=self.library,
            style_version_id=self.version,
            images=self.images(),
            vision=self.vision,
            tracker=tracker,
            reference_png=self.reference,
            prices=self.prices,
            video_id=self.video_id,
            stored=self.stored,
        )
        if record.status is SketchStatus.FAILED:
            self.result.quarantined.append(scene.id)
            return _quarantined(scene)
        self.result.generated += 1
        self._note_unchecked(scene, record)
        return _attached(scene, record, "new")

    def _note_unchecked(self, scene: Scene, record: SketchRecord) -> None:
        if record.status is SketchStatus.UNCHECKED:
            self.result.unchecked.append(scene.id)

    def _attached_record(self, scene: Scene, key: IdeaKey) -> SketchRecord | None:
        """The record an earlier run attached, unless it is stale (unknown id, or
        another style version or idea key), in which case the scene is matched again."""
        if scene.sketch_id is None:
            return None
        record = self.library.get_sketch(scene.sketch_id)
        if record is None or record.style_version_id != self.version or record.idea_key != str(key):
            logger.info("scene %s: stale sketch id %d; matching again", scene.id, scene.sketch_id)
            return None
        return record


def require_series_channel(series: Series, channel: ChannelConfig) -> None:
    """`PlanningError` unless `series` belongs to `channel` (same slug). No spend."""
    if series.channel_slug != channel.slug:
        raise PlanningError(
            f"series belongs to channel {series.channel_slug!r}, not {channel.slug!r}; "
            "pass that channel's TOML with --channel"
        )


def generate_missing(
    series: Series,
    channel: ChannelConfig,
    settings: Settings,
    *,
    images: ImageProvider | None = None,
    vision: LLMClient | Literal["auto"] | None = "auto",
    library: SketchLibrary | None = None,
    prices: PriceTable | None = None,
    only: frozenset[str] | None = None,
) -> tuple[Series, GenerationResult]:
    """Attach a library sketch to every narration scene of an approved series.

    `images` and `vision` must be untracked clients (tests inject fakes; a tracked
    one raises `TypeError`); they are charged at the channel's configured image and
    vision prices. `vision="auto"` builds the vision model of `vision_model(channel)`;
    `None`, or a channel whose vision is off (`[vision] provider = "none"`, or no
    `[vision]` with a text-only LLM provider), stores new sketches UNCHECKED. The
    series must belong to `channel` (same slug). Keys, prices and the reference sketch are
    checked before any spend. The spending limit applies per part. A scene whose
    attached sketch is still usable is kept as is; one quarantined since is
    flagged again without spend. Any `NarratoFlowError` raised mid-run (budget,
    provider, library) is re-raised as `GenerationInterruptedError` carrying the
    partial series and result. `only` restricts the run to those scene ids; the
    others are left untouched and not counted. Text-card scenes are skipped. The
    input series is not mutated.
    """
    if not series.approved:
        raise PlanningError("approve the series first")
    require_series_channel(series, channel)
    _require_untracked(images, None if vision == "auto" else vision)
    check_provider_keys(channel, settings)
    prices = prices if prices is not None else default_prices(settings)
    require_generation_prices(channel, prices, vision=vision is not None)
    reference = load_reference(channel.style)
    library = library if library is not None else default_library(settings)
    version = library.ensure_style_version(channel.slug, style_fingerprint(channel.style))
    bundled = default_bundled_icons()
    vision_client: LLMClient | None
    if vision == "auto":
        cfg = vision_llm_config(channel)
        vision_client = None if cfg is None else build_llm_client(cfg, settings)
    else:
        vision_client = vision
    built: list[ImageProvider] = [] if images is None else [images]

    def image_provider() -> ImageProvider:
        # built on the first new sketch: a local model is not loaded for a reuse-only run
        if not built:
            built.append(build_image_provider(channel, settings))
        return built[0]

    result = GenerationResult()
    run = _Run(
        channel=channel,
        library=library,
        version=version,
        bundled=bundled,
        images=image_provider,
        vision=vision_client,
        prices=prices,
        reference=reference,
        video_id=video_id(series),
        result=result,
    )
    out = series.model_copy(deep=True)
    # (tracker, seed): each part's tracker starts at the part's cumulative spend, so
    # the limit spans stages; the result reports this run's delta only
    trackers: list[tuple[SpendTracker, Decimal]] = []

    def delta() -> Decimal:
        return sum((t.spent - seed for t, seed in trackers), Decimal(0))

    for part in out.parts:
        tracker = SpendTracker(channel.spending_limit_usd, spent=part.spent_usd)
        trackers.append((tracker, part.spent_usd))
        for i, scene in enumerate(part.scenes):
            if only is not None and scene.id not in only:
                continue
            try:
                part.scenes[i] = run.scene(scene, tracker)
            except NarratoFlowError as e:
                part.scenes[i] = run.interrupted(scene)
                part.spent_usd = tracker.spent
                add_spend(part.spend, tracker)
                result.spent_usd = delta()
                raise GenerationInterruptedError(e, (out.model_copy(deep=True), result)) from e
        part.spent_usd = tracker.spent
        add_spend(part.spend, tracker)
    result.spent_usd = delta()
    logger.info(
        "sketches: %d generated, %d reused, %d bundled, %d kept, %d quarantined",
        result.generated,
        result.reused,
        result.bundled,
        result.kept,
        len(result.quarantined),
    )
    return out, result
