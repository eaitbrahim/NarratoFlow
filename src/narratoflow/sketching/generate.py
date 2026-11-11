"""Turn an approved series into scenes with library sketches (spec 4.3, 6.3).

Per narration scene, in order: reuse a stored sketch, copy a bundled icon into
the channel library, or generate a new one. A new sketch passes the free density
check before the paid vision check; a failure is stored FAILED (quarantined) and
never attached to a scene. Every image and vision call is budget-checked on the
part's own `SpendTracker`.
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
from narratoflow.costs.tracker import SpendTracker, TrackedLLM, require_generation_prices
from narratoflow.errors import (
    BudgetExceededError,
    ChannelConfigError,
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
from narratoflow.models import Language, Scene, SceneKind, Series
from narratoflow.settings import Settings
from narratoflow.sketching.vision import NOT_CHECKED, check_sketch, estimate_check_cost
from narratoflow.visuals.bundled import BundledIcon, BundledIcons, default_bundled_icons
from narratoflow.visuals.lineart import binarize, check_density
from narratoflow.visuals.marker import marker_seed
from narratoflow.visuals.strokes import extract_strokes

logger = logging.getLogger(__name__)

QUARANTINED_FLAG = "sketch_quarantined"
MAX_REFERENCE_BYTES = 4 * 1024 * 1024
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
    generated: int = 0
    """New sketches attached to a scene (PASSED or UNCHECKED)."""
    reused: int = 0
    bundled: int = 0
    quarantined: list[str] = field(default_factory=list)
    """Scene ids left without a sketch because theirs FAILED."""
    unchecked: list[str] = field(default_factory=list)
    """Scene ids whose newly generated sketch is UNCHECKED (no vision check)."""
    spent_usd: Decimal = Decimal(0)


class GenerationInterruptedError(BudgetExceededError):
    """The spending limit stopped generation; everything stored so far is kept.

    `partial` is the series with the scenes updated so far, and the result so far.
    The caller saves `partial[0]` before reporting the error.
    """

    def __init__(
        self, cause: BudgetExceededError, partial: tuple[Series, GenerationResult]
    ) -> None:
        super().__init__(cause.estimate_usd, cause.limit_usd)
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

    `images` and `vision` are untracked clients: they are wrapped here so every call
    is checked against and charged to `tracker` at the channel's configured prices.
    The image seed is `marker_seed(video_id, scene.id)`, so a re-run asks for the same
    image. A density reject is stored FAILED with `density: <reason>` and skips the
    vision call. The returned record may be an existing one (dedupe), even FAILED.
    A usable record (PASSED or UNCHECKED) has its strokes stored.
    """
    if scene.idea_key is None:
        raise PlanningError(f"scene {scene.id} has no idea key")
    key = str(parse_idea_key(scene.idea_key))
    style = channel.style
    image_provider = style.image_provider.strip().lower()
    vision_provider, vision_name = vision_model(channel)
    tracked_images = TrackedImageProvider(
        images, tracker, prices, image_provider, style.image_model
    )
    tracked_vision = (
        None
        if vision is None
        else TrackedLLM(vision, tracker, vision_provider, vision_name, prices)
    )
    estimate = prices.image(image_provider, style.image_model)
    if vision is not None:
        estimate += estimate_check_cost(
            style, key, prices, vision_provider, vision_name, with_reference=True
        )
    tracker.ensure_room(estimate, f"sketch {key}")

    image = tracked_images.generate(
        ImageRequest(
            prompt=build_prompt(style, key, scene.idea_description),
            reference_png=reference_png,
            seed=marker_seed(video_id, scene.id),
        )
    )
    provenance = _provenance(image)

    def store(status: SketchStatus, reason: str | None, mirror_safe: bool) -> SketchRecord:
        return library.add_sketch(
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

    try:
        ink = binarize(image.png)
    except SketchRejectedError as e:
        record = store(SketchStatus.FAILED, f"density: {e}", False)
        logger.info("sketch %s for scene %s: density reject", key, scene.id)
        return record
    report = check_density(ink, line_weight_px=style.line_weight_px)
    if not report.ok:
        record = store(SketchStatus.FAILED, f"density: {report.reason}", False)
        logger.info("sketch %s for scene %s: density reject", key, scene.id)
        return record
    status, reason, mirror_safe, _ = check_sketch(
        image.png, style, key, tracked_vision, reference_png=reference_png
    )
    record = store(status, reason, mirror_safe)
    if record.status is not SketchStatus.FAILED:
        _ensure_strokes(library, record, rtl=channel.language is Language.ARABIC, ink=ink)
    logger.info("sketch %s for scene %s: %s", key, scene.id, record.status.value)
    return record


def _add_bundled(
    library: SketchLibrary,
    style_version_id: int,
    key: IdeaKey,
    icon: BundledIcon,
    channel: ChannelConfig,
) -> SketchRecord:
    """Copy a bundled icon into the channel library; the caller checks `.status`.

    Icons are curated free assets: they pass the free density check and are stored
    UNCHECKED (no paid vision call that the approved estimate does not include).
    """
    try:
        data = icon.file.read_bytes()
    except OSError as e:
        raise ChannelConfigError(f"cannot read bundled icon {icon.file}: {e}") from e
    ink: np.ndarray | None = None
    try:
        ink = binarize(data)
        report = check_density(ink, line_weight_px=channel.style.line_weight_px)
        rejected = report.reason
    except SketchRejectedError as e:
        rejected = str(e)
    status, reason = (
        (SketchStatus.UNCHECKED, NOT_CHECKED)
        if rejected is None
        else (SketchStatus.FAILED, f"density: {rejected}")
    )
    record = library.add_sketch(
        style_version_id,
        key,
        data,
        icon.file.suffix.lower() or ".png",
        SketchOrigin.COPIED,
        status,
        False,
        Provenance(source=icon.source, license=icon.license, created_at=_now()),
        reason,
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

    def scene(self, scene: Scene, tracker: SpendTracker) -> Scene:
        if scene.kind is SceneKind.VERSE or scene.idea_key is None:
            return scene
        rtl = self.channel.language is Language.ARABIC
        key = parse_idea_key(scene.idea_key)
        found = match_idea(self.library, self.version, key, self.bundled)
        if found.kind is MatchKind.REUSED:
            passed = [r for r in found.candidates if r.status is SketchStatus.PASSED]
            record = (passed or list(found.candidates))[0]
            _ensure_strokes(self.library, record, rtl=rtl)
            self.result.reused += 1
            return _attached(scene, record, "reused")
        if found.kind is MatchKind.BUNDLED and found.icon is not None:
            record = _add_bundled(self.library, self.version, key, found.icon, self.channel)
            if record.status is not SketchStatus.FAILED:
                self.result.bundled += 1
                return _attached(scene, record, "reused")
            logger.info("bundled icon %s failed its check; generating instead", key)
        if QUARANTINED_FLAG in scene.flags:
            # a quarantined scene is never regenerated automatically (spec 6.3)
            self.result.quarantined.append(scene.id)
            return scene
        record = generate_one(
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
        )
        if record.status is SketchStatus.FAILED:
            self.result.quarantined.append(scene.id)
            return _quarantined(scene)
        self.result.generated += 1
        if record.status is SketchStatus.UNCHECKED:
            self.result.unchecked.append(scene.id)
        return _attached(scene, record, "new")


def generate_missing(
    series: Series,
    channel: ChannelConfig,
    settings: Settings,
    *,
    images: ImageProvider | None = None,
    vision: LLMClient | Literal["auto"] | None = "auto",
    library: SketchLibrary | None = None,
    prices: PriceTable | None = None,
) -> tuple[Series, GenerationResult]:
    """Attach a library sketch to every narration scene of an approved series.

    `images` and `vision` are untracked clients (tests inject fakes); they are
    charged at the channel's configured image and vision prices. `vision="auto"`
    builds the vision model of `channel.vision` (else the channel LLM); `None`
    stores new sketches UNCHECKED. Keys, prices and the reference sketch are
    checked before any spend. The spending limit applies per part; when it stops
    a run, `GenerationInterruptedError` carries the partial series and result.
    The input series is not mutated.
    """
    if not series.approved:
        raise PlanningError("approve the series first")
    check_provider_keys(channel, settings)
    prices = prices if prices is not None else default_prices(settings)
    require_generation_prices(channel, prices, vision=vision is not None)
    reference = load_reference(channel.style)
    library = library if library is not None else default_library(settings)
    version = library.ensure_style_version(channel.slug, style_fingerprint(channel.style))
    bundled = default_bundled_icons()
    vision_client = (
        build_llm_client(vision_llm_config(channel), settings) if vision == "auto" else vision
    )
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
        video_id=f"{series.channel_slug}/{series.title}",
        result=result,
    )
    out = series.model_copy(deep=True)
    trackers: list[SpendTracker] = []
    for part in out.parts:
        tracker = SpendTracker(channel.spending_limit_usd)
        trackers.append(tracker)
        for i, scene in enumerate(part.scenes):
            try:
                part.scenes[i] = run.scene(scene, tracker)
            except BudgetExceededError as e:
                result.spent_usd = sum((t.spent for t in trackers), Decimal(0))
                raise GenerationInterruptedError(e, (out.model_copy(deep=True), result)) from e
    result.spent_usd = sum((t.spent for t in trackers), Decimal(0))
    logger.info(
        "sketches: %d generated, %d reused, %d bundled, %d quarantined",
        result.generated,
        result.reused,
        result.bundled,
        len(result.quarantined),
    )
    return out, result
