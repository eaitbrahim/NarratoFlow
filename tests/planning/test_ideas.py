from pathlib import Path

import pytest
from llm.fakes import FakeLLM

from narratoflow.errors import PlanningError
from narratoflow.models import Narration, Scene, SceneKind
from narratoflow.planning import ideas as ideas_module
from narratoflow.planning.ideas import (
    ALLOWED_COMPOUNDS,
    FIGURE_WORDS,
    FORBIDDEN_OBJECTS,
    IdeaOut,
    assign_ideas,
    validate_idea,
)
from narratoflow.planning.prompts import IDEAS_SYSTEM


def _idea(
    key: str, description: str = "an eye", *, person: bool = False, logo: bool = False
) -> IdeaOut:
    return IdeaOut(idea_key=key, description=description, depicts_person=person, depicts_logo=logo)


def _scene(sid: str, text: str = "العلم نور.", kind: SceneKind = SceneKind.NARRATION) -> Scene:
    return Scene(id=sid, narration=Narration(source_text=text), kind=kind)


def test_forbidden_objects_are_the_brief_set() -> None:
    expected = {"man", "woman", "person", "people", "child", "boy", "girl", "prophet"}
    expected |= {"figure", "face", "silhouette", "angel", "god", "allah", "logo"}
    assert expected == FORBIDDEN_OBJECTS


def test_name_card_allowed() -> None:
    assert (
        validate_idea(
            IdeaOut(
                idea_key="name-card|yahya",
                description="name",
                depicts_person=False,
                depicts_logo=False,
            )
        )
        == []
    )


@pytest.mark.parametrize("word", ["logos", "trademark", "brand", "brands", "mascot", "logo"])
def test_brand_words_rejected_in_object_and_description(word: str) -> None:
    assert validate_idea(_idea(f"{word}-sign|generic"))
    assert validate_idea(_idea("sign|generic", f"a sign with a {word} on it"))


def test_brand_word_object_reports_once() -> None:
    [violation] = validate_idea(_idea("mascot|generic"))
    assert "mascot" in violation


def test_logo_rejected() -> None:
    violations = validate_idea(_idea("phone|generic", logo=True))
    assert len(violations) == 1
    assert "logo" in violations[0]
    assert validate_idea(_idea("logo|generic"))


def test_person_flag_rejected() -> None:
    violations = validate_idea(_idea("eye|generic", person=True))
    assert len(violations) == 1
    assert "person" in violations[0]


@pytest.mark.parametrize(
    "key",
    [
        "man|generic",
        "old-man|generic",
        "angel-wings|generic",
        "angels|generic",
        "faces|generic",
        "women|generic",
        "children|generic",
        "God|generic",
        "allah|generic",
        "prophet|generic",
        "stick-figure|generic",
        "head|generic",
        "torso|generic",
    ],
)
def test_forbidden_object_rejected(key: str) -> None:
    violations = validate_idea(_idea(key))
    assert len(violations) == 1
    assert key.lower() in violations[0]


@pytest.mark.parametrize(
    "key",
    [
        "hand|generic",
        "eye|generic",
        "footprints|desert",
        "road|two-horizons",
        "crown|abbasid",
        "manuscript|generic",
        "scales|generic",
    ],
)
def test_symbols_and_limbs_allowed(key: str) -> None:
    assert validate_idea(_idea(key)) == []


@pytest.mark.parametrize(
    "key",
    [
        "human|generic",
        "stickman|generic",
        "stick-figure|generic",
        "crowd|generic",
        "soldier|roman",
        "kings|abbasid",
        "praying-worshipper|generic",
        "marble-statue|greek",
        "idol|generic",
        "bust|roman",
        "baby|generic",
    ],
)
def test_figure_words_rejected(key: str) -> None:
    violations = validate_idea(_idea(key, "a drawing"))
    assert len(violations) == 1
    assert key in violations[0]


def test_figure_words_cover_the_review_list() -> None:
    required = {"human", "humans", "stickman", "stick", "body", "crowd", "skeleton"}
    required |= {"portrait", "mannequin", "soldier", "soldiers", "king", "queen"}
    required |= {"mother", "father", "baby", "warrior", "worshipper", "worshippers"}
    required |= {"scholar", "saint", "statue", "idol", "profile", "bust"}
    assert required <= FIGURE_WORDS
    assert not {"crown", "throne", "sword"} & FIGURE_WORDS


@pytest.mark.parametrize("key", ["crown|abbasid", "throne|generic", "sword|generic"])
def test_royal_symbols_allowed(key: str) -> None:
    assert validate_idea(_idea(key, "a golden crown on a cushion")) == []


def test_person_in_description_rejected() -> None:
    out = IdeaOut(
        idea_key="eye|generic",
        description="a bearded man praying",
        depicts_person=False,
        depicts_logo=False,
    )
    violations = validate_idea(out)
    assert len(violations) == 1
    assert "'man'" in violations[0]
    assert validate_idea(_idea("pen|generic", "Two Women, walking"))
    assert validate_idea(_idea("pen|generic", "angels around a gate"))


def test_allowed_body_words_in_description_pass() -> None:
    assert validate_idea(_idea("hand|generic", "a hand holding a pen")) == []
    assert validate_idea(_idea("eye|generic", "two open eyes")) == []
    assert validate_idea(_idea("footprints|desert", "footprints in the sand")) == []


@pytest.mark.parametrize("key", sorted(ALLOWED_COMPOUNDS))
def test_allowed_compounds(key: str) -> None:
    assert validate_idea(_idea(f"{key}|generic", f"a {key.replace('-', ' ')}")) == []


def test_allowed_compounds_are_the_review_list() -> None:
    assert {"clock-face", "arrow-head", "figure-eight"} == ALLOWED_COMPOUNDS
    assert validate_idea(_idea("clock-face|generic", "a face of a clock"))
    assert validate_idea(_idea("old-face|generic", "a clock"))


def test_unparseable_key_is_a_violation() -> None:
    violations = validate_idea(_idea("no pipe here"))
    assert len(violations) == 1
    assert "invalid idea key" in violations[0]


def test_person_idea_rejected_then_fixed() -> None:
    llm = FakeLLM([_idea("man|generic", "a man"), _idea("Eye | Generic", "  an open eye  ")])
    [scene] = assign_ideas([_scene("s001")], [[]], llm)

    assert scene.idea_key == "eye|generic"
    assert scene.idea_description == "an open eye"
    assert len(llm.calls) == 2
    assert all(call[0] == IDEAS_SYSTEM and call[2] is IdeaOut for call in llm.calls)
    first_user, retry_user = llm.calls[0][1], llm.calls[1][1]
    assert retry_user.startswith(first_user)
    assert "man|generic" in retry_user[len(first_user) :]


def test_person_idea_rejected_twice_raises() -> None:
    llm = FakeLLM([_idea("man|generic"), _idea("woman|generic", person=True)])
    with pytest.raises(PlanningError, match="s007") as info:
        assign_ideas([_scene("s007")], [[]], llm)
    assert "woman|generic" in str(info.value)
    assert "person" in str(info.value)
    assert len(llm.calls) == 2


def test_bad_key_retried_not_crash() -> None:
    llm = FakeLLM([_idea("a|b|c"), _idea("pen|generic", "a pen")])
    [scene] = assign_ideas([_scene("s001")], [[]], llm)
    assert scene.idea_key == "pen|generic"
    assert "invalid idea key" in llm.calls[1][1]


def test_candidates_rendered_in_prompt() -> None:
    llm = FakeLLM([_idea("road|two-horizons", "a road")])
    candidates = [("road|two-horizons", "a road"), ("eye|generic", "an open eye")]
    [scene] = assign_ideas([_scene("s001", "الدنيا والآخرة")], [candidates], llm)
    user = llm.calls[0][1]
    assert "road|two-horizons \u2014 a road" in user
    assert "eye|generic \u2014 an open eye" in user
    assert "الدنيا والآخرة" in user
    assert "reuse" in user.lower()
    assert scene.idea_key == "road|two-horizons"


def test_no_candidates_rendered_as_none() -> None:
    llm = FakeLLM([_idea("eye|generic")])
    assign_ideas([_scene("s001")], [[]], llm)
    assert "(none)" in llm.calls[0][1]


def test_verse_scenes_skipped_and_order_kept() -> None:
    scenes = [
        _scene("s001"),
        _scene("s002", "نص الآية", SceneKind.VERSE),
        _scene("s003"),
    ]
    llm = FakeLLM([_idea("eye|generic"), _idea("pen|generic", "a pen")])
    out = assign_ideas(scenes, [[], [("x|generic", "x")], []], llm)
    assert [s.id for s in out] == ["s001", "s002", "s003"]
    assert [s.idea_key for s in out] == ["eye|generic", None, "pen|generic"]
    assert out[1].idea_description == ""
    assert len(llm.calls) == 2
    assert scenes[0].idea_key is None  # inputs are not mutated


def test_long_description_truncated_to_15_words() -> None:
    words = [f"w{n}" for n in range(20)]
    llm = FakeLLM([_idea("eye|generic", " ".join(words))])
    [scene] = assign_ideas([_scene("s001")], [[]], llm)
    assert scene.idea_description == " ".join(words[:15])


def test_candidate_list_length_must_match() -> None:
    with pytest.raises(PlanningError, match="candidate lists"):
        assign_ideas([_scene("s001")], [], FakeLLM([]))


def test_ideas_system_carries_the_drawing_rules() -> None:
    text = IDEAS_SYSTEM.lower()
    for needle in (
        "no human figure",
        "silhouette",
        "name-card|",
        "god is never",
        "angels",
        "unseen",
        "brand",
        "object|context",
        "generic",
        "kebab-case",
        "reuse",
        "15 words",
        "eye|generic",
        "footprints",
        "stick figure",
    ):
        assert needle in text, needle


def test_sources_are_ascii() -> None:
    assert Path(ideas_module.__file__ or "").read_bytes().isascii()
