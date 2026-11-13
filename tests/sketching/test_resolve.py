from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from fakes.images import FakeImages
from fakes.llm import FakeLLM
from fixtures import images as img

from narratoflow.channels.config import ChannelConfig
from narratoflow.errors import BudgetExceededError, PlanningError, ProviderError
from narratoflow.library.repository import SketchLibrary
from narratoflow.models import Series, VisualSource
from narratoflow.settings import Settings
from narratoflow.sketching.generate import (
    QUARANTINED_FLAG,
    GenerationInterruptedError,
    generate_missing,
)
from narratoflow.sketching.resolve import (
    check_regenerable,
    regenerate,
    swap_idea,
    use_text_card,
)
from narratoflow.visuals.marker import marker_seed
from sketching.support import (
    PER_SKETCH,
    PRICES,
    clean,
    scene,
    scenes_of,
    series,
    verdict,
)


def quarantined() -> Series:
    return series([scene(1, flags=[QUARANTINED_FLAG]), scene(2)])


def test_regenerate_makes_a_new_sketch_with_a_different_seed(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    first_images = FakeImages()
    first, _ = generate_missing(
        series([scene(1)]),
        channel,
        settings,
        images=first_images,
        vision=FakeLLM([verdict(face=True)] * 2),
        library=library,
        prices=PRICES,
    )
    [sc] = scenes_of(first)
    assert QUARANTINED_FLAG in sc.flags
    before = first.model_copy(deep=True)
    images = FakeImages([img.scroll(size=256)])
    out, _ = regenerate(
        first,
        "s001",
        channel,
        settings,
        images=images,
        vision=FakeLLM(clean(2)),
        library=library,
        prices=PRICES,
    )
    assert first == before
    [new] = scenes_of(out)
    assert new.sketch_attempt == 1
    assert images.requests[0].seed != first_images.requests[0].seed
    assert images.requests[0].seed == marker_seed("history/Scenes", "s001", 1)
    assert new.sketch_id is not None and QUARANTINED_FLAG not in new.flags


def test_regenerate_only_touches_the_chosen_scene(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    s = quarantined()
    images = FakeImages()
    out, _ = regenerate(
        s,
        "s001",
        channel,
        settings,
        images=images,
        vision=FakeLLM(clean(2)),
        library=library,
        prices=PRICES,
    )
    assert len(images.requests) == 1
    first, second = scenes_of(out)
    assert first.sketch_id is not None
    assert second == scenes_of(s)[1]


def test_regenerate_that_fails_again_stays_quarantined(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    out, result = regenerate(
        quarantined(),
        "s001",
        channel,
        settings,
        images=FakeImages(),
        vision=FakeLLM([verdict(face=True)] * 2),
        library=library,
        prices=PRICES,
    )
    [first, _] = scenes_of(out)
    assert QUARANTINED_FLAG in first.flags and first.sketch_id is None
    assert result.quarantined == ["s001"] and result.generated == 0
    assert result.spent_usd == PER_SKETCH


def test_regenerate_needs_an_approved_series(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    s = series([scene(1, flags=[QUARANTINED_FLAG])], approved=False)
    with pytest.raises(PlanningError):
        regenerate(
            s,
            "s001",
            channel,
            settings,
            images=FakeImages(),
            vision=None,
            library=library,
            prices=PRICES,
        )


def test_regenerate_refuses_unknown_or_unflagged_scenes(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    images = FakeImages()
    for sid in ("s999", "s002"):
        with pytest.raises(PlanningError):
            regenerate(
                quarantined(),
                sid,
                channel,
                settings,
                images=images,
                vision=None,
                library=library,
                prices=PRICES,
            )
    assert images.requests == []


def test_swap_idea_sets_the_new_key_and_clears_the_flag() -> None:
    s = series([scene(1, flags=[QUARANTINED_FLAG, "other"], sketch_attempt=2)])
    before = s.model_copy(deep=True)
    out = swap_idea(s, "s001", "Lamp|Roman", "an oil lamp")
    assert s == before
    [sc] = scenes_of(out)
    assert sc.idea_key == "lamp|roman" and sc.idea_description == "an oil lamp"
    assert sc.match is None and sc.sketch_id is None
    assert sc.flags == ["other"] and sc.sketch_attempt == 2


def test_swap_idea_rejects_rule_violations_and_bad_keys() -> None:
    s = series([scene(1, flags=[QUARANTINED_FLAG])])
    with pytest.raises(PlanningError, match="forbidden"):
        swap_idea(s, "s001", "man|roman", "a man")
    with pytest.raises(PlanningError):
        swap_idea(s, "s001", "not a key", "x")


def test_swap_idea_refuses_unknown_or_unflagged_scenes() -> None:
    s = quarantined()
    for sid in ("s999", "s002"):
        with pytest.raises(PlanningError):
            swap_idea(s, sid, "lamp|roman", "an oil lamp")


def test_use_text_card_sets_the_source_and_clears_the_flag() -> None:
    s = series([scene(1, flags=[QUARANTINED_FLAG])])
    before = s.model_copy(deep=True)
    out = use_text_card(s, "s001")
    assert s == before
    [sc] = scenes_of(out)
    assert sc.visual.source is VisualSource.TEXT_CARD and sc.sketch_id is None
    assert QUARANTINED_FLAG not in sc.flags


def test_use_text_card_sets_card_text_and_default_empty() -> None:
    s = series([scene(1, flags=[QUARANTINED_FLAG]), scene(2, flags=[QUARANTINED_FLAG])])
    [sc, _] = scenes_of(use_text_card(s, "s001", "Hello"))
    assert sc.card_text == "Hello"
    [_, other] = scenes_of(use_text_card(s, "s002"))
    assert other.card_text == ""


def test_use_text_card_refuses_unknown_or_unflagged_scenes() -> None:
    s = quarantined()
    for sid in ("s999", "s002"):
        with pytest.raises(PlanningError):
            use_text_card(s, sid)


def test_use_text_card_updates_the_text_of_an_existing_card() -> None:
    s = series([scene(1, flags=[QUARANTINED_FLAG]), scene(2)])
    card = use_text_card(s, "s001", "First")
    [sc, _] = scenes_of(use_text_card(card, "s001", "Second"))
    assert sc.card_text == "Second" and sc.visual.source is VisualSource.TEXT_CARD
    with pytest.raises(PlanningError, match="not quarantined"):
        use_text_card(card, "s001")  # nothing to change without a text
    with pytest.raises(PlanningError, match="not quarantined"):
        use_text_card(card, "s002", "x")  # an unflagged sketch scene is still refused


def test_sources_are_ascii() -> None:
    root = Path(__file__).parents[2] / "src" / "narratoflow" / "sketching"
    (root / "resolve.py").read_bytes().decode("ascii")


def test_interrupted_regenerate_keeps_the_scene_quarantined(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    limited = channel.model_copy(update={"spending_limit_usd": Decimal("0.01")})
    images = FakeImages()
    with pytest.raises(GenerationInterruptedError) as info:
        regenerate(
            quarantined(),
            "s001",
            limited,
            settings,
            images=images,
            vision=FakeLLM([]),
            library=library,
            prices=PRICES,
        )
    assert images.requests == []
    assert isinstance(info.value.__cause__, BudgetExceededError)
    [first, second] = scenes_of(info.value.partial[0])
    assert QUARANTINED_FLAG in first.flags and first.sketch_id is None
    assert first.sketch_attempt == 1
    assert second == scenes_of(quarantined())[1]
    assert "progress so far is kept" in str(info.value)
    assert info.value.partial[1].quarantined == ["s001"]


def test_interrupted_regenerate_after_a_failed_check_lists_the_scene_once(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    boom = ProviderError("vision down")
    with pytest.raises(GenerationInterruptedError) as info:
        regenerate(
            quarantined(),
            "s001",
            channel,
            settings,
            images=FakeImages(),
            vision=FakeLLM([boom]),
            library=library,
            prices=PRICES,
        )
    partial, result = info.value.partial
    [first, _] = scenes_of(partial)
    assert QUARANTINED_FLAG in first.flags and first.sketch_id is None
    assert result.quarantined == ["s001"]


def test_regenerate_refuses_scenes_generate_missing_would_skip(
    channel: ChannelConfig, settings: Settings, library: SketchLibrary
) -> None:
    cases = [
        scene(1, flags=[QUARANTINED_FLAG], kind="verse"),
        scene(1, flags=[QUARANTINED_FLAG], idea_key=None),
        scene(1, flags=[QUARANTINED_FLAG], visual={"source": "text_card"}),
    ]
    for sc in cases:
        s = series([sc])
        before = s.model_copy(deep=True)
        images = FakeImages()
        with pytest.raises(PlanningError, match="nothing to generate"):
            regenerate(
                s,
                "s001",
                channel,
                settings,
                images=images,
                vision=None,
                library=library,
                prices=PRICES,
            )
        assert s == before and images.requests == []


def test_check_regenerable_accepts_only_a_quarantined_drawable_scene() -> None:
    assert check_regenerable(quarantined(), "s001") == "thing-1|roman"
    with pytest.raises(PlanningError, match="unknown scene"):
        check_regenerable(quarantined(), "s999")
    with pytest.raises(PlanningError, match="not quarantined"):
        check_regenerable(quarantined(), "s002")
    text_card = scene(1, flags=[QUARANTINED_FLAG], visual={"source": "text_card"})
    with pytest.raises(PlanningError, match="nothing to generate"):
        check_regenerable(series([text_card]), "s001")
