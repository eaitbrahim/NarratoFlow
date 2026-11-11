from pathlib import Path

import pytest
from conftest import FIXTURES

from narratoflow.models import Language, Narration, Scene, SceneKind
from narratoflow.planning import series as series_module
from narratoflow.planning.series import estimate_scene_seconds, split_into_parts


def narr(
    text: str, title: str = "", sid: str = "s", kind: SceneKind = SceneKind.NARRATION
) -> Scene:
    return Scene(id=sid, narration=Narration(source_text=text), title=title, kind=kind)


def words(n: int, end: str = ".") -> str:
    return " ".join(["word"] * n) + end


def test_estimate_narration_en_and_ar() -> None:
    s = narr(words(30))
    assert estimate_scene_seconds(s, Language.ENGLISH) == pytest.approx(30 / 150 * 60 + 0.4)
    assert estimate_scene_seconds(s, Language.ARABIC) == pytest.approx(30 / 120 * 60 + 0.4)


def test_estimate_ignores_punctuation_tokens_and_diacritics() -> None:
    plain = narr("كتاب قيم")
    marked = narr("كِتَاب - … قَيِّم")
    a = estimate_scene_seconds(plain, Language.ARABIC)
    assert estimate_scene_seconds(marked, Language.ARABIC) == a


def test_verse_duration_slower_than_narration() -> None:
    text = words(20)
    verse = narr(text, kind=SceneKind.VERSE)
    assert estimate_scene_seconds(verse, Language.ARABIC) == pytest.approx(20 / 80 * 60 + 0.8)
    assert estimate_scene_seconds(verse, Language.ARABIC) > estimate_scene_seconds(
        narr(text), Language.ARABIC
    )


def test_empty_input() -> None:
    assert split_into_parts([], Language.ENGLISH, min_s=60, max_s=90) == []


def test_parts_respect_bounds_and_inputs_not_mutated() -> None:
    scenes = [narr(words(20), title=f"t{i}", sid=f"s{i}") for i in range(20)]
    parts = split_into_parts(scenes, Language.ENGLISH, min_s=60, max_s=90)
    assert all(s.est_duration_s == 0 for s in scenes)
    assert len(parts) > 1
    for p in parts:
        assert p.est_duration_s <= 90
        assert p.index == parts.index(p) + 1 and p.count == len(parts)
    for p in parts[:-1]:
        assert p.est_duration_s >= 60
    assert [s.id for p in parts for s in p.scenes] == [s.id for s in scenes]
    assert all(s.est_duration_s > 0 for p in parts for s in p.scenes)


def test_scene_never_split_oversized_flagged() -> None:
    big = narr(words(400), title="big", sid="big")
    small = narr(words(10), title="small", sid="small")
    parts = split_into_parts([big, small, small], Language.ENGLISH, min_s=60, max_s=90)
    assert [s.id for s in parts[0].scenes] == ["big"]
    assert "over_max" in parts[0].flags
    assert "over_max" not in parts[1].flags


def test_tiny_tail_merged_without_over_max() -> None:
    # each scene: 20 words EN = 8.4 s. 10 scenes = 84 s, tail of 1 scene = 8.4 s < 30
    scenes = [narr(words(20), title=f"t{i}", sid=f"s{i}") for i in range(11)]
    parts = split_into_parts(scenes, Language.ENGLISH, min_s=60, max_s=90)
    assert len(parts) == 1
    assert len(parts[0].scenes) == 11
    assert parts[0].est_duration_s > 90
    assert "over_max" not in parts[0].flags


def test_tiny_tail_not_merged_when_too_long() -> None:
    scenes = [narr(words(20), title=f"t{i}", sid=f"s{i}") for i in range(10)]
    scenes.append(narr(words(60), title="tail", sid="tail"))  # 24.4 s, 84 + 24.4 > 103.5
    parts = split_into_parts(scenes, Language.ENGLISH, min_s=60, max_s=90)
    assert len(parts) == 2
    assert [s.id for s in parts[1].scenes] == ["tail"]


def test_hook_recap_teaser_chain() -> None:
    scenes = [narr(words(100), title=f"title{i}", sid=f"s{i}") for i in range(6)]
    parts = split_into_parts(scenes, Language.ENGLISH, min_s=60, max_s=90)
    assert len(parts) == 3
    assert [p.hook_text for p in parts] == ["title0", "title2", "title4"]
    assert parts[0].recap_text == ""
    assert parts[1].recap_text == parts[0].hook_text
    assert parts[1].teaser_text == parts[2].hook_text
    assert parts[2].teaser_text == ""
    assert parts[0].teaser_text == parts[1].hook_text


def test_single_part_has_no_recap_or_teaser() -> None:
    parts = split_into_parts([narr(words(50), title="only")], Language.ENGLISH, min_s=60, max_s=90)
    assert len(parts) == 1
    assert parts[0].recap_text == "" and parts[0].teaser_text == ""
    assert parts[0].hook_text == "only"


def test_hook_falls_back_to_first_nonempty_title() -> None:
    scenes = [narr(words(10), title="", sid="a"), narr(words(10), title="second", sid="b")]
    parts = split_into_parts(scenes, Language.ENGLISH, min_s=60, max_s=90)
    assert parts[0].hook_text == "second"


def test_hook_falls_back_to_first_eight_words() -> None:
    text = "one two three four five six seven eight nine ten."
    scenes = [narr(text, sid="a")]
    parts = split_into_parts(scenes, Language.ENGLISH, min_s=60, max_s=90)
    assert parts[0].hook_text == "one two three four five six seven eight…"


def test_hook_fallback_skips_verse_scenes() -> None:
    verse = narr("alpha beta", kind=SceneKind.VERSE, sid="v")
    n = narr("x y z.", sid="n")
    parts = split_into_parts([verse, n], Language.ENGLISH, min_s=60, max_s=90)
    assert parts[0].hook_text == "x y z.…"


def test_unfinished_ending_flagged_from_file() -> None:
    text = (FIXTURES / "input" / "clean_ar.txt").read_text(encoding="utf-8")
    unfinished = text[: text.index(":") + 1]
    parts = split_into_parts([narr(unfinished)], Language.ARABIC, min_s=60, max_s=90)
    assert "unfinished_ending" in parts[-1].flags


def test_finished_last_paragraph_not_flagged() -> None:
    text = (FIXTURES / "input" / "clean_ar.txt").read_text(encoding="utf-8")
    parts = split_into_parts([narr(text.strip())], Language.ARABIC, min_s=60, max_s=90)
    assert "unfinished_ending" not in parts[-1].flags


@pytest.mark.parametrize(
    ("ending", "flagged"),
    [
        (":", True),
        ("،", True),
        (",", True),
        ("؛", True),
        (";", True),
        ("word", True),
        (".", False),
        ("!", False),
        ("?", False),
        ("؟", False),
        ("…", False),
        ("»", False),
        ('"', False),
        ("”", False),
        (")", False),
        ("\ufd3e", False),
        (".  \n", False),
    ],
)
def test_unfinished_ending_rules(ending: str, flagged: bool) -> None:
    parts = split_into_parts([narr("hello there" + ending)], Language.ENGLISH, min_s=60, max_s=90)
    assert ("unfinished_ending" in parts[-1].flags) is flagged


def test_unfinished_ending_uses_last_narration_scene() -> None:
    scenes = [narr("done.", sid="a"), narr("an ayah", kind=SceneKind.VERSE, sid="v")]
    parts = split_into_parts(scenes, Language.ENGLISH, min_s=60, max_s=90)
    assert "unfinished_ending" not in parts[-1].flags
    scenes = [narr("not done", sid="a"), narr("an ayah", kind=SceneKind.VERSE, sid="v")]
    parts = split_into_parts(scenes, Language.ENGLISH, min_s=60, max_s=90)
    assert "unfinished_ending" in parts[-1].flags


def test_verse_only_last_part_not_flagged() -> None:
    parts = split_into_parts(
        [narr("an ayah", kind=SceneKind.VERSE)], Language.ENGLISH, min_s=60, max_s=90
    )
    assert parts[-1].flags == []


def test_only_last_part_checked() -> None:
    scenes = [narr(words(180, end=":"), title=f"t{i}", sid=f"s{i}") for i in range(2)]
    scenes[1] = narr(words(180), title="t1", sid="s1")
    parts = split_into_parts(scenes, Language.ENGLISH, min_s=60, max_s=90)
    assert len(parts) == 2
    assert all("unfinished_ending" not in p.flags for p in parts)


def test_sources_are_ascii() -> None:
    assert Path(series_module.__file__ or "").read_bytes().isascii()
