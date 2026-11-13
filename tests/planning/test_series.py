from pathlib import Path

import pytest
from conftest import FIXTURES

from narratoflow.models import Language, Narration, Part, Scene, SceneKind
from narratoflow.planning import series as series_module
from narratoflow.planning.series import (
    END_CARD_DEFAULTS,
    end_card_text,
    estimate_scene_seconds,
    split_into_parts,
)


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


def test_end_card_on_final_part_only() -> None:
    scenes = [narr(words(100), title=f"title{i}", sid=f"s{i}") for i in range(6)]
    parts = split_into_parts(scenes, Language.ENGLISH, min_s=60, max_s=90, end_card="X")
    assert [p.end_card_text for p in parts] == ["", "", "X"]
    single = split_into_parts(scenes[:1], Language.ENGLISH, min_s=60, max_s=90, end_card="X")
    assert [p.end_card_text for p in single] == ["X"]


def test_end_card_defaults() -> None:
    for language in Language:
        assert end_card_text(None, language) == END_CARD_DEFAULTS[language]
        assert end_card_text("", language) == ""
        assert end_card_text("  Thanks ", language) == "Thanks"
    assert END_CARD_DEFAULTS[Language.ENGLISH] == "Follow for more"
    assert END_CARD_DEFAULTS[Language.ARABIC] == (
        "\u062a\u0627\u0628\u0639\u0648\u0646\u0627 \u0644\u0644\u0645\u0632\u064a\u062f"
    )


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
    assert parts[0].hook_text == "x y z\u2026"


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


def test_scene_landing_exactly_on_max_stays() -> None:
    # 24 English words = 24 / 150 * 60 + 0.4 = 10.0 s; nine scenes = 90 s.
    scenes = [narr(words(24), title=f"t{i}", sid=f"s{i}") for i in range(9)]
    parts = split_into_parts(scenes, Language.ENGLISH, min_s=60, max_s=90)
    assert len(parts) == 1
    assert len(parts[0].scenes) == 9


def test_oversized_scene_in_middle_gets_own_part() -> None:
    scenes = [narr(words(100), title="a", sid="a")]
    scenes.append(narr(words(400), title="big", sid="big"))
    scenes.append(narr(words(100), title="c", sid="c"))
    parts = split_into_parts(scenes, Language.ENGLISH, min_s=60, max_s=90)
    assert [[s.id for s in p.scenes] for p in parts] == [["a"], ["big"], ["c"]]
    assert "over_max" in parts[1].flags
    assert "over_max" not in parts[0].flags
    assert "over_max" not in parts[2].flags


def test_short_part_still_closes_when_next_does_not_fit() -> None:
    # 40.4 s then 100 words (40.4 s) then 150 words (60.4 s): first part stays under min_s.
    scenes = [
        narr(words(100), title="a", sid="a"),
        narr(words(200), title="b", sid="b"),
        narr(words(100), title="c", sid="c"),
    ]
    parts = split_into_parts(scenes, Language.ENGLISH, min_s=60, max_s=90)
    assert [s.id for s in parts[0].scenes] == ["a"]
    assert parts[0].est_duration_s < 60


def test_unfinished_ending_ignores_invisible_marks() -> None:
    finished = "hello there.\u200f\u200b\ufeff"
    parts = split_into_parts([narr(finished)], Language.ENGLISH, min_s=60, max_s=90)
    assert "unfinished_ending" not in parts[-1].flags
    unfinished = "hello there:\u200f\u2069"
    parts = split_into_parts([narr(unfinished)], Language.ENGLISH, min_s=60, max_s=90)
    assert "unfinished_ending" in parts[-1].flags


@pytest.mark.parametrize("text", ["", "   \n", "\u200b\u200f", " \u200b \ufeff"])
def test_blank_text_counts_as_unfinished(text: str) -> None:
    assert series_module._is_unfinished(text) is True


def _stub_estimates(monkeypatch: pytest.MonkeyPatch, table: dict[str, float]) -> None:
    """Exact, dyadic per-scene seconds so boundary cases do not depend on float rounding."""
    monkeypatch.setattr(series_module, "estimate_scene_seconds", lambda s, lang: table[s.id])
    monkeypatch.setattr(series_module, "TAIL_MERGE_SLACK", 1.25)


def _ids(parts: list[Part]) -> list[list[str]]:
    return [[s.id for s in p.scenes] for p in parts]


def test_lone_scene_exactly_at_max_is_not_over_max(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_estimates(monkeypatch, {"a": 40.0, "b": 40.5})
    exact = split_into_parts([narr("x.", sid="a")], Language.ENGLISH, min_s=20, max_s=40)
    assert exact[0].flags == []
    over = split_into_parts([narr("x.", sid="b")], Language.ENGLISH, min_s=20, max_s=40)
    assert over[0].flags == ["over_max"]


def test_tail_of_exactly_half_min_is_kept_but_shorter_is_merged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_estimates(monkeypatch, {"p": 35.0, "t10": 10.0, "t9": 9.5})
    kept = split_into_parts(
        [narr("x.", sid="p"), narr("y.", sid="t10")], Language.ENGLISH, min_s=20, max_s=40
    )
    assert _ids(kept) == [["p"], ["t10"]]
    merged = split_into_parts(
        [narr("x.", sid="p"), narr("y.", sid="t9")], Language.ENGLISH, min_s=20, max_s=40
    )
    assert _ids(merged) == [["p", "t9"]]


def test_tail_merge_allowed_up_to_exactly_the_slack(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_estimates(monkeypatch, {"p": 35.0, "fits": 15.0, "over": 15.5})
    fits = split_into_parts(
        [narr("x.", sid="p"), narr("y.", sid="fits")], Language.ENGLISH, min_s=40, max_s=40
    )
    assert _ids(fits) == [["p", "fits"]]
    over = split_into_parts(
        [narr("x.", sid="p"), narr("y.", sid="over")], Language.ENGLISH, min_s=40, max_s=40
    )
    assert _ids(over) == [["p"], ["over"]]


def test_tail_merges_into_the_part_before_it_not_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_estimates(monkeypatch, {"a": 38.0, "b": 38.0, "c": 5.0})
    parts = split_into_parts(
        [narr("x.", sid="a"), narr("y.", sid="b"), narr("z.", sid="c")],
        Language.ENGLISH,
        min_s=20,
        max_s=40,
    )
    assert _ids(parts) == [["a"], ["b", "c"]]
    assert parts[-1].count == 2


def test_unfinished_ending_judges_the_last_narration_of_the_last_part() -> None:
    first_open = [narr("Opening line,", sid="a"), narr("Closing line.", sid="b")]
    flags = split_into_parts(first_open, Language.ENGLISH, min_s=60, max_s=90)[-1].flags
    assert "unfinished_ending" not in flags
    last_open = [narr("Opening line.", sid="a"), narr("Closing line,", sid="b")]
    flags = split_into_parts(last_open, Language.ENGLISH, min_s=60, max_s=90)[-1].flags
    assert "unfinished_ending" in flags
