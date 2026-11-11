"""The creator's three ways to settle a scene whose sketch was quarantined (spec 6.3).

Regenerate (a paid call, the caller confirms), swap the idea key, or use a text
card. A FAIL has no override. Each function returns a new `Series`; the input is
never mutated.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from narratoflow.channels.config import ChannelConfig
from narratoflow.costs.prices import PriceTable
from narratoflow.errors import PlanningError
from narratoflow.images import ImageProvider
from narratoflow.library.keys import parse_idea_key
from narratoflow.library.repository import SketchLibrary
from narratoflow.llm.base import LLMClient
from narratoflow.models import Scene, SceneKind, Series, VisualSource
from narratoflow.planning.ideas import IdeaOut, validate_idea
from narratoflow.settings import Settings
from narratoflow.sketching.generate import (
    QUARANTINED_FLAG,
    GenerationInterruptedError,
    GenerationResult,
    generate_missing,
)


def _resolved(series: Series, scene_id: str, change: Callable[[Scene], dict[str, Any]]) -> Series:
    """A copy of `series` whose quarantined scene `scene_id` is updated by `change`
    (the quarantine flag is always cleared)."""
    out = series.model_copy(deep=True)
    for part in out.parts:
        for i, scene in enumerate(part.scenes):
            if scene.id != scene_id:
                continue
            if QUARANTINED_FLAG not in scene.flags:
                raise PlanningError(f"scene {scene_id} is not quarantined")
            flags = [f for f in scene.flags if f != QUARANTINED_FLAG]
            part.scenes[i] = scene.model_copy(update={**change(scene), "flags": flags})
            return out
    raise PlanningError(f"unknown scene {scene_id}")


def _requarantined(partial: Series, scene_id: str) -> Series:
    """`partial` with the scene flagged again if it ended up without a sketch."""
    fixed = partial.model_copy(deep=True)
    for part in fixed.parts:
        for i, scene in enumerate(part.scenes):
            if scene.id == scene_id and scene.sketch_id is None:
                flags = [f for f in scene.flags if f != QUARANTINED_FLAG] + [QUARANTINED_FLAG]
                part.scenes[i] = scene.model_copy(update={"flags": flags})
    return fixed


def _drawable_key(scene: Scene) -> str:
    """The idea key of a scene `generate_missing` would draw, else `PlanningError`."""
    if (
        scene.kind is SceneKind.VERSE
        or scene.idea_key is None
        or scene.visual.source is VisualSource.TEXT_CARD
    ):
        raise PlanningError(f"scene {scene.id} has nothing to generate")
    return scene.idea_key


def _regenerated(scene: Scene) -> dict[str, Any]:
    _drawable_key(scene)
    return {"sketch_id": None, "match": "new", "sketch_attempt": scene.sketch_attempt + 1}


def check_regenerable(series: Series, scene_id: str) -> str:
    """The scene's idea key if `regenerate` accepts the scene (it exists, is
    quarantined and has an idea to draw), else `PlanningError`. No spend; the
    series is not changed."""
    keys: list[str] = []

    def change(scene: Scene) -> dict[str, Any]:
        keys.append(_drawable_key(scene))
        return {}

    _resolved(series, scene_id, change)
    return keys[0]


def regenerate(
    series: Series,
    scene_id: str,
    channel: ChannelConfig,
    settings: Settings,
    *,
    images: ImageProvider | None = None,
    vision: LLMClient | Literal["auto"] | None = "auto",
    library: SketchLibrary | None = None,
    prices: PriceTable | None = None,
) -> tuple[Series, GenerationResult]:
    """Generate a new sketch for one quarantined scene (a paid call: confirm first).

    The scene's attempt counter goes up so the image seed differs from the earlier
    one. Returns the new series and the run's result: when the new sketch fails too
    the scene is quarantined again (listed in `result.quarantined`). `images`,
    `vision`, `library` and `prices` are `generate_missing`'s. Refuses what
    `check_regenerable` refuses. May raise `GenerationInterruptedError` like
    `generate_missing`.
    """
    prepared = _resolved(series, scene_id, _regenerated)
    try:
        out, result = generate_missing(
            prepared,
            channel,
            settings,
            images=images,
            vision=vision,
            library=library,
            prices=prices,
            only=frozenset({scene_id}),
        )
    except GenerationInterruptedError as e:
        # the creator confirmed this attempt only: a scene left without a sketch goes
        # back to quarantine so the next `sketch generate` does not spend on it
        fixed, result = e.partial
        fixed = _requarantined(fixed, scene_id)
        flagged = any(
            s.id == scene_id and QUARANTINED_FLAG in s.flags for p in fixed.parts for s in p.scenes
        )
        if flagged and scene_id not in result.quarantined:
            result.quarantined.append(scene_id)
        e.partial = (fixed, result)
        raise
    return out, result


def swap_idea(series: Series, scene_id: str, new_key: str, description: str) -> Series:
    """Give a quarantined scene another idea; the next `sketch generate` matches it."""
    violations = validate_idea(
        IdeaOut(
            idea_key=new_key,
            description=description,
            depicts_person=False,
            depicts_logo=False,
        )
    )
    if violations:
        raise PlanningError("idea rejected: " + "; ".join(violations))
    key = str(parse_idea_key(new_key))

    def change(_: Scene) -> dict[str, Any]:
        return {
            "idea_key": key,
            "idea_description": description,
            "match": None,
            "sketch_id": None,
        }

    return _resolved(series, scene_id, change)


def use_text_card(series: Series, scene_id: str) -> Series:
    """Show a text card instead of a sketch (the card is drawn in Plan 5)."""

    def change(scene: Scene) -> dict[str, Any]:
        return {
            "sketch_id": None,
            "visual": scene.visual.model_copy(update={"source": VisualSource.TEXT_CARD}),
        }

    return _resolved(series, scene_id, change)
