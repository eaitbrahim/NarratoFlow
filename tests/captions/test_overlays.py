"""Overlays: part label, topic badges, the hook/recap/teaser headline and text cards."""

from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import pytest

from narratoflow.audio.timeline import PartTimeline, SceneTiming, TimingSource
from narratoflow.captions.ass import AssEvent
from narratoflow.captions.overlays import (
    LABELS,
    OVERLAY_MAX_CHARS,
    card_events,
    card_text,
    overlay_events,
    part_label,
    truncate,
)
from narratoflow.errors import RenderError
from narratoflow.models import (
    Language,
    Narration,
    Part,
    RightsBasis,
    Scene,
    SceneKind,
    ScriptSource,
    Series,
    VerseRef,
)
from narratoflow.render.ffmpeg import RenderTools
from narratoflow.render.layout import STAGE, TOP_BAR, Rect

from .stills import assert_in_zone, bbox, ink, still

ELL = "\u2026"
# Words of the widest Arabic letters, sad and dad (23.0 px each in Amiri Bold).
AR_WIDE = ["\u0635\u0635\u0635\u0635", "\u0636\u0636\u0636\u0636"]
DAD = "\u0636"
EN, AR = Language.ENGLISH, Language.ARABIC
# "ar-rajulu alladhi kataba kulla kalimatin fi al-layli": prose with full tashkeel.
AR_PROSE = [
    "\u0627\u0644\u0631\u0651\u064e\u062c\u064f\u0644\u064f",
    "\u0627\u0644\u0651\u064e\u0630\u0650\u064a",
    "\u0643\u064e\u062a\u064e\u0628\u064e",
    "\u0643\u064f\u0644\u0651\u064e",
    "\u0643\u064e\u0644\u0650\u0645\u064e\u0629\u064d",
    "\u0641\u0650\u064a",
    "\u0627\u0644\u0644\u0651\u064e\u064a\u0652\u0644\u0650",
]


EN_PROSE = "The scholar who wrote down every word of the long night"


def _fill(words: list[str], n: int) -> str:
    """Repeat `words` into a string of exactly `n` characters (cut inside the last word)."""
    text = ""
    i = 0
    while len(text) < n:
        text = f"{text} {words[i % len(words)]}" if text else words[0]
        i += 1
    text = text[:n]
    return text[:-1] + words[0][0] if text.endswith(" ") else text


def _scene(sid: str, title: str = "", *, text: str = "a b c", card_text: str = "") -> Scene:
    return Scene(id=sid, narration=Narration(source_text=text), title=title, card_text=card_text)


def _series(language: Language, part: Part, title: str = "The Series") -> Series:
    return Series(
        title=title,
        channel_slug="chan",
        language=language,
        source=ScriptSource(author="a", work="w", rights=RightsBasis.OWN),
        parts=[part],
    )


def _part(index: int, count: int, *, hook: str = "Hook", recap: str = "", teaser: str = "",
          end_card: str = "", scenes: list[Scene] | None = None) -> Part:  # fmt: skip
    return Part(
        index=index,
        count=count,
        hook_text=hook,
        recap_text=recap,
        teaser_text=teaser,
        end_card_text=end_card,
        scenes=scenes or [_scene("s1")],
    )


def _slot(sid: str, start: int, end: int) -> SceneTiming:
    return SceneTiming(
        scene_id=sid,
        kind=SceneKind.NARRATION,
        start_ms=start,
        end_ms=end,
        speech_start_ms=start,
        speech_end_ms=start,
        words=[],
        timing_source=TimingSource.ESTIMATED,
    )


def _timeline(part: Part, slots: list[tuple[str, int, int]]) -> PartTimeline:
    return PartTimeline(
        series_title="S",
        channel_slug="chan",
        part_index=part.index,
        part_count=part.count,
        audio_file="part-01.flac",
        duration_ms=slots[-1][2],
        integrated_lufs=None,
        true_peak_dbtp=None,
        scenes=[_slot(*s) for s in slots],
        sfx=[],
        voice=None,
        recitations=[],
        assets=[],
        spent_usd=Decimal(0),
        cost_estimated=False,
        warnings=[],
    )


def _headlines(part: Part, d: int, language: Language = Language.ENGLISH) -> list[AssEvent]:
    events = overlay_events(_series(language, part), part, _timeline(part, [("s1", 0, d)]))
    return [e for e in events if e.style == "Headline"]


def _windows(events: list[AssEvent]) -> list[tuple[int, int, str]]:
    return [(e.start_ms, e.end_ms, e.text) for e in events]


def test_truncate_cuts_at_word_boundary() -> None:
    assert truncate("short") == "short"
    assert truncate("  a   b  ") == "a b"
    assert truncate("alpha beta gamma", 12) == "alpha beta" + ELL
    assert truncate("alpha beta gamma", 11) == "alpha beta" + ELL
    assert truncate("abcdefghijkl", 6) == "abcde" + ELL
    long = " ".join(["word"] * 30)
    assert len(truncate(long)) <= OVERLAY_MAX_CHARS
    assert truncate(long).endswith("word" + ELL)


def test_truncate_never_splits_a_grapheme() -> None:
    # beh + shadda + fatha: both marks stay on every kept letter (marks take no width).
    tashkeel = truncate("\u0628\u0651\u064e" * 30, 24)
    assert tashkeel == "\u0628\u0651\u064e" * 23 + ELL
    # An emoji ZWJ family is kept whole or dropped whole.
    family = "\U0001f468\u200d\U0001f469\u200d\U0001f467"
    cut = truncate(family * 10, 24)
    assert cut.endswith(ELL)
    assert cut[:-1] == family * (len(cut[:-1]) // len(family))
    assert truncate(family * 10, 2) == ELL
    # e + combining acute: the accent stays with its e.
    acute = truncate("e\u0301" * 30, 24)
    assert acute == "e\u0301" * 23 + ELL
    # A tatweel stays with the letter it stretches; ZWNJ is never cut off.
    assert truncate("\u0628\u0640\u0640" * 4, 5) == "\u0628\u0640\u0640" + ELL
    assert truncate("\u0628\u200c" * 8, 5) == "\u0628\u200c" * 4 + ELL


def test_truncate_cuts_tokens_wider_than_a_line() -> None:
    assert truncate("W" * 80, 60, line=30) == "W" * 29 + ELL
    assert truncate("ab " + "W" * 80 + " cd", 60, line=30) == "ab " + "W" * 29 + ELL
    assert truncate("W" * 40, 32, line=30) == "W" * 29 + ELL
    # Near the end of the budget the long token is cut to what remains.
    words = " ".join(["aaaa"] * 10)
    assert truncate(words + " " + "W" * 40, 60, line=30) == words + " " + "W" * 9 + ELL
    # No room left for any of it: a word-boundary cut.
    two = "a" * 29 + " " + "a" * 29
    assert truncate(two + " " + "W" * 40, 60, line=30) == two + ELL
    # In an Arabic style a Latin letter counts 5/4 of an Arabic one.
    assert truncate(DAD * 45, 40, language=AR) == DAD * 39 + ELL
    assert truncate("W" * 80, 40, language=AR) == "W" * 31 + ELL
    assert truncate("W" * 32, 40, language=AR) == "W" * 32


def test_hook_at_zero_and_teaser_at_end() -> None:
    part = _part(1, 2, teaser="What comes")
    assert _windows(_headlines(part, 10_000)) == [
        (0, 2_500, "Hook"),
        (2_500, 7_000, "The Series"),
        (7_000, 10_000, "Next: What comes"),
    ]
    # A short part: the teaser never starts before the hook ends.
    assert _windows(_headlines(part, 4_000)) == [
        (0, 2_500, "Hook"),
        (2_500, 4_000, "Next: What comes"),
    ]
    assert _windows(_headlines(part, 2_000)) == [(0, 2_000, "Hook")]
    # Under MIN_OVERLAY_MS after the hook: no teaser flash.
    assert _windows(_headlines(part, 2_600)) == [(0, 2_500, "Hook")]
    assert _windows(_headlines(part, 3_500))[-1] == (2_500, 3_500, "Next: What comes")


def test_end_card_headline_at_end_of_final_part() -> None:
    final = _part(2, 2, end_card="Bye")
    assert _windows(_headlines(final, 10_000)) == [
        (0, 2_500, "Hook"),
        (2_500, 7_000, "The Series"),
        (7_000, 10_000, "Bye"),
    ]
    # A non-final part keeps the teaser and never shows the end card.
    first = _part(1, 2, teaser="What comes", end_card="Bye")
    assert _windows(_headlines(first, 10_000))[-1] == (7_000, 10_000, "Next: What comes")


def test_end_card_skipped_when_part_too_short() -> None:
    final = _part(2, 2, end_card="Bye")
    assert _windows(_headlines(final, 3_000)) == [(0, 2_500, "Hook")]


def test_end_card_disabled() -> None:
    final = _part(2, 2, end_card="")
    assert _windows(_headlines(final, 10_000)) == [
        (0, 2_500, "Hook"),
        (2_500, 10_000, "The Series"),
    ]


def test_end_card_clips_recap_on_short_final_part() -> None:
    final = _part(2, 2, recap="Before", end_card="Bye")
    events = _windows(_headlines(final, 6_000))
    assert (3_000, 6_000, "Bye") in events
    for (_, a_end, _), (b_start, _, _) in pairwise(events):
        assert a_end <= b_start


def test_recap_only_from_part_two_and_dropped_when_no_room() -> None:
    last = _part(2, 2, recap="Before")
    assert _windows(_headlines(last, 8_000)) == [
        (0, 2_500, "Hook"),
        (2_500, 5_000, "Previously: Before"),
        (5_000, 8_000, "The Series"),
    ]
    # No teaser: the recap window may run to D.
    assert _windows(_headlines(last, 4_000)) == [
        (0, 2_500, "Hook"),
        (2_500, 4_000, "Previously: Before"),
    ]
    # Under MIN_OVERLAY_MS of room: no recap, and no title in a sub-second gap.
    assert _windows(_headlines(last, 3_200)) == [(0, 2_500, "Hook")]
    middle = _part(2, 3, recap="Before", teaser="After")
    assert _windows(_headlines(middle, 6_000)) == [
        (0, 2_500, "Hook"),
        (3_000, 6_000, "Next: After"),
    ]
    assert _windows(_headlines(middle, 7_000)) == [
        (0, 2_500, "Hook"),
        (2_500, 4_000, "Previously: Before"),
        (4_000, 7_000, "Next: After"),
    ]


def test_recap_text_ignored_on_part_one_and_teaser_ignored_on_last_part() -> None:
    first = _part(1, 1, recap="Before", teaser="After")
    assert _windows(_headlines(first, 8_000)) == [
        (0, 2_500, "Hook"),
        (2_500, 8_000, "The Series"),
    ]
    last = _part(3, 3, recap="", teaser="After")
    assert _windows(_headlines(last, 8_000)) == [
        (0, 2_500, "Hook"),
        (2_500, 8_000, "The Series"),
    ]


def test_empty_hook_and_title_leave_gaps() -> None:
    part = _part(1, 1, hook=" ")
    series = _series(Language.ENGLISH, part, title="")
    events = overlay_events(series, part, _timeline(part, [("s1", 0, 5_000)]))
    assert [e for e in events if e.style == "Headline"] == []
    assert _windows(_headlines(part, 5_000)) == [(0, 5_000, "The Series")]


def test_headline_text_is_escaped_and_truncated() -> None:
    part = _part(1, 2, hook="{\\pos(1,1)} " + "word " * 30)
    hook = _headlines(part, 3_000)[0].text
    assert "{" not in hook and "\\" not in hook
    assert len(hook) <= OVERLAY_MAX_CHARS
    teaser = _headlines(_part(1, 2, teaser="x " * 40), 9_000)[-1].text
    assert teaser.startswith("Next: ") and len(teaser) <= OVERLAY_MAX_CHARS


def test_topright_title_truncated_to_24_chars() -> None:
    scenes = [
        _scene("s1", "A rather long scene title for badges"),
        _scene("s2"),
        _scene("s3", "Short"),
    ]
    part = _part(1, 1, scenes=scenes)
    tl = _timeline(part, [("s1", 0, 1_000), ("s2", 1_000, 2_000), ("s3", 2_000, 3_000)])
    badges = [
        e
        for e in overlay_events(_series(Language.ENGLISH, part), part, tl)
        if e.style == "TopRight"
    ]
    assert _windows(badges) == [(0, 1_000, "A rather long scene" + ELL), (2_000, 3_000, "Short")]
    assert all(len(e.text) <= 24 for e in badges)


def test_verse_scene_has_no_topright_badge() -> None:
    # The verse reference line under the verse already shows the (localized) reference.
    verse = Scene(
        id="s1",
        kind=SceneKind.VERSE,
        title="31:33",
        narration=Narration(source_text="x"),
        verse=VerseRef(surah=31, ayah_start=33, ayah_end=33, certain=True, uthmani_text="x"),
    )
    part = _part(1, 1, scenes=[verse, _scene("s2", "Topic")])
    tl = _timeline(part, [("s1", 0, 1_000), ("s2", 1_000, 2_000)])
    for language in (Language.ARABIC, Language.ENGLISH):
        badges = [
            e for e in overlay_events(_series(language, part), part, tl) if e.style == "TopRight"
        ]
        assert _windows(badges) == [(1_000, 2_000, "Topic")]


def test_unknown_timeline_scene_is_an_error() -> None:
    part = _part(1, 1)
    with pytest.raises(RenderError, match="s9"):
        overlay_events(_series(Language.ENGLISH, part), part, _timeline(part, [("s9", 0, 1_000)]))


def test_single_part_has_no_part_label() -> None:
    assert part_label(1, 1, Language.ENGLISH) is None
    part = _part(1, 1)
    events = overlay_events(
        _series(Language.ENGLISH, part), part, _timeline(part, [("s1", 0, 3_000)])
    )
    assert not [e for e in events if e.style == "TopLeft"]
    two = _part(2, 3)
    events = overlay_events(_series(Language.ENGLISH, two), two, _timeline(two, [("s1", 0, 3_000)]))
    assert _windows([e for e in events if e.style == "TopLeft"]) == [(0, 3_000, "Part 2 of 3")]


def test_arabic_part_label_digits() -> None:
    label = part_label(2, 12, Language.ARABIC)
    assert label == "\u0627\u0644\u062c\u0632\u0621 \u0662 \u0645\u0646 \u0661\u0662"
    assert LABELS[Language.ARABIC][1] == "\u0633\u0627\u0628\u0642\u064b\u0627: "
    assert LABELS[Language.ARABIC][2] == "\u0627\u0644\u062a\u0627\u0644\u064a: "


def test_card_text_fallbacks() -> None:
    assert card_text(_scene("s", "Title", card_text="Card  words")) == "Card words"
    assert card_text(_scene("s", "Title")) == "Title"
    assert card_text(_scene("s", text="one two three four five six seven")) == (
        "one two three four five six" + ELL
    )
    assert card_text(_scene("s", text="one two")) == "one two"
    captioned = _scene("s").model_copy(
        update={"narration": Narration(source_text="x y", caption_text="shown words")}
    )
    assert card_text(captioned) == "shown words"


def test_card_events_over_the_slot() -> None:
    scene = _scene("s", "A {title}")
    assert card_events(scene, _slot("s", 1_000, 4_000), EN) == [
        AssEvent(1_000, 4_000, "Card", "{\\an5\\pos(474,815)\\fad(150,200)}A (title)")
    ]
    assert card_events(_scene("s", text=" "), _slot("s", 0, 1_000), EN) == []
    assert card_events(scene, _slot("s", 1_000, 1_000), EN) == []


def _wide(language: Language, n: int) -> str:
    return _fill(AR_WIDE if language is Language.ARABIC else ["WWWWW", "MMMM"], n)


def _prose(language: Language, n: int) -> str:
    if language is Language.ARABIC:
        return _fill(AR_PROSE, n)
    return _fill(EN_PROSE.split(), n)


def test_headline_60_chars_inside_top_bar(render_tools: RenderTools, tmp_path: Path) -> None:
    # The widest 60-char headline, a 60-char prose headline, a 24-char badge of wide
    # letters and a two-digit part label: each inside TOP_BAR, and no two collide.
    # Arabic headlines are cut to OVERLAY_MAX_CHARS_AR (one line) before rendering, so
    # the Arabic case checks that cut of the 60-char text.
    for language in Language:
        for kind, make in (("wide", _wide), ("prose", _prose)):
            headline = make(language, OVERLAY_MAX_CHARS)
            assert len(headline) == OVERLAY_MAX_CHARS
            badge = truncate(make(language, 40), 24)
            part = _part(12, 12, hook=headline, scenes=[_scene("s1", badge)])
            tl = _timeline(part, [("s1", 0, 3_000)])
            events = overlay_events(_series(language, part), part, tl)
            boxes = {}
            for style in ("TopLeft", "TopRight", "Headline"):
                event = next(e for e in events if e.style == style and e.start_ms == 0)
                work = tmp_path / f"{language.value}-{kind}-{style}"
                boxes[style] = bbox(ink(still(render_tools, work, language, [event], 500)))
                assert_in_zone(boxes[style], TOP_BAR, f"{language.value} {kind} {style}")
            left, right, head = boxes["TopLeft"], boxes["TopRight"], boxes["Headline"]
            assert left.right < right.x, (left, right)
            assert head.y > max(left.bottom, right.bottom), (left, right, head)


def test_card_inside_stage(render_tools: RenderTools, tmp_path: Path) -> None:
    for language in Language:
        for kind, make in (("wide", _wide), ("prose", _prose)):
            scene = _scene("s1", card_text=make(language, 80))
            event = card_events(scene, _slot("s1", 0, 3_000), language)[0]
            frame = still(
                render_tools, tmp_path / f"{language.value}-{kind}", language, [event], 1_500
            )
            assert_in_zone(bbox(ink(frame)), STAGE, f"{language.value} {kind} card")


def _headline_box(tools: RenderTools, work: Path, language: Language, hook: str) -> Rect:
    part = _part(1, 1, hook=hook)
    events = overlay_events(_series(language, part), part, _timeline(part, [("s1", 0, 3_000)]))
    event = next(e for e in events if e.style == "Headline" and e.start_ms == 0)
    return bbox(ink(still(tools, work, language, [event], 500)))


def test_arabic_headline_worst_cases_inside_top_bar(
    render_tools: RenderTools, tmp_path: Path
) -> None:
    # sad/dad words (the widest Arabic letters) and Latin capitals on an Arabic channel,
    # both well past the Arabic cap: one line, inside TOP_BAR.
    cases = {
        "sad": " ".join(["\u0635" * 4] * 20),
        "dad": " ".join([DAD * 4] * 20),
        "sad-dad": _wide(AR, 80),
        "latin-caps": "WWWWW MMMMM " * 8,
        "latin-long-words": "W" * 20 + " " + "M" * 20 + " " + "W" * 20,
    }
    for name, hook in cases.items():
        box = _headline_box(render_tools, tmp_path / name, AR, hook)
        assert_in_zone(box, TOP_BAR, f"ar {name} headline")
        assert box.h < 60, f"ar {name} headline wraps: {box}"


def test_unbroken_strings_stay_inside_their_zones(
    render_tools: RenderTools, tmp_path: Path
) -> None:
    for language in Language:
        for name, text in (("w80", "W" * 80), ("dad45", DAD * 45)):
            if language is EN and name == "dad45":
                continue  # Noto Sans has no Arabic; the glyph warning covers it
            work = tmp_path / f"{language.value}-{name}"
            box = _headline_box(render_tools, work / "headline", language, text)
            assert_in_zone(box, TOP_BAR, f"{language.value} {name} headline")
            part = _part(1, 1, scenes=[_scene("s1", text)])
            tl = _timeline(part, [("s1", 0, 3_000)])
            events = overlay_events(_series(language, part), part, tl)
            badge = next(e for e in events if e.style == "TopRight")
            box = bbox(ink(still(render_tools, work / "badge", language, [badge], 500)))
            assert_in_zone(box, TOP_BAR, f"{language.value} {name} badge")
            card = card_events(_scene("s1", card_text=text), _slot("s1", 0, 3_000), language)
            frame = still(render_tools, work / "card", language, card, 1_500)
            assert_in_zone(bbox(ink(frame)), STAGE, f"{language.value} {name} card")
