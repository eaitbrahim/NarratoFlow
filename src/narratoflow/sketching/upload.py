"""Attach a creator's own image to a scene as XDoG line art (spec 4.3, issue #28).

The line art (not the original upload) is stored as an `UPLOADED` library sketch,
UNCHECKED because no vision call is made (free: no spend). It passes the same free
density check as a generated sketch before anything is stored.
"""

from __future__ import annotations

from datetime import UTC, datetime

from narratoflow.channels.config import ChannelConfig, style_fingerprint
from narratoflow.errors import PlanningError, SketchRejectedError
from narratoflow.library.repository import SketchLibrary
from narratoflow.library.schemas import Provenance, SketchOrigin, SketchRecord, SketchStatus
from narratoflow.models import Language, Scene, SceneKind, Series, VisualSource
from narratoflow.publish.rules import Rules
from narratoflow.sketching.generate import QUARANTINED_FLAG, require_series_channel
from narratoflow.visuals.lineart import check_density, ink_to_png, xdog_lineart
from narratoflow.visuals.strokes import extract_strokes

UPLOAD_REASON = "uploaded: vision not run"


def check_upload_licence(rules: Rules, licence: str, attribution: str) -> None:
    """`PlanningError` unless `licence` is in the rules' `[licences.assets]` (an upload
    is checked as an icon at publish) and has the attribution that licence needs."""
    rule = rules.file.licences.assets.get(licence.strip())
    if rule is None:
        raise PlanningError(
            f"licence {licence} is not allowed by platform rules {rules.file.version}; "
            "pass the licence of your image (use own-work for your own)"
        )
    if rule.attribution and not attribution.strip():
        raise PlanningError(f"{licence} needs an attribution; pass --attribution")


def _find(series: Series, scene_id: str) -> tuple[int, int, Scene]:
    for p, part in enumerate(series.parts):
        for i, scene in enumerate(part.scenes):
            if scene.id == scene_id:
                return p, i, scene
    raise PlanningError(f"unknown scene {scene_id}")


def attach_upload(
    series: Series,
    scene_id: str,
    channel: ChannelConfig,
    image: bytes,
    *,
    licence: str,
    source: str,
    attribution: str = "",
    library: SketchLibrary,
) -> tuple[Series, SketchRecord]:
    """The series with `image` attached to scene `scene_id` as line art, and the record.

    The scene becomes `IMAGE_TO_SKETCH` with the new `sketch_id` and no quarantine
    flag. Refuses, storing nothing: a scene that is unknown, a verse, a text card or
    has no idea key, or a series of another channel (`PlanningError`); a blank
    `source`; an image that is undecodable, too small or large, empty or fails the
    density check (`SketchRejectedError`). The series need not be approved. The input
    series is not mutated.
    """
    require_series_channel(series, channel)
    p, i, scene = _find(series, scene_id)
    if (
        scene.kind is SceneKind.VERSE
        or scene.idea_key is None
        or scene.visual.source is VisualSource.TEXT_CARD
    ):
        raise PlanningError(f"scene {scene_id} has nothing to draw; it cannot take an upload")
    if not source.strip():
        raise PlanningError("say where the image came from (source)")
    ink = xdog_lineart(image)
    report = check_density(ink, line_weight_px=channel.style.line_weight_px)
    if not report.ok:
        raise SketchRejectedError(f"density: {report.reason}")
    strokes = extract_strokes(ink, rtl=channel.language is Language.ARABIC)
    version = library.ensure_style_version(channel.slug, style_fingerprint(channel.style))
    record = library.add_sketch(
        version,
        scene.idea_key,
        ink_to_png(ink),
        ".png",
        SketchOrigin.UPLOADED,
        SketchStatus.UNCHECKED,
        False,
        Provenance(
            source=source,
            license=licence,
            created_at=datetime.now(UTC),
            attribution=attribution,
        ),
        UPLOAD_REASON,
        description=scene.idea_description,
    )
    if record.status is SketchStatus.FAILED:
        raise SketchRejectedError(f"this image is quarantined: {record.status_reason}")
    library.put_strokes(record.id, strokes)
    out = series.model_copy(deep=True)
    out.parts[p].scenes[i] = scene.model_copy(
        update={
            "visual": scene.visual.model_copy(update={"source": VisualSource.IMAGE_TO_SKETCH}),
            "sketch_id": record.id,
            "match": "new",
            "flags": [f for f in scene.flags if f != QUARANTINED_FLAG],
        }
    )
    return out, record
