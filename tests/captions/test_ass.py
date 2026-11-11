import re
from pathlib import Path

import pytest

from narratoflow.audio.timeline import AyahTiming, SceneTiming, TimelineWord, TimingSource
from narratoflow.captions.ass import (
    CAPTION_GAP_MS,
    CAPTION_HOLD_MS,
    CAPTION_MAX_CHARS,
    CAPTION_MAX_CHARS_AR,
    CAPTION_MAX_WORDS,
    SENTENCE_END,
    AssDocument,
    AssEvent,
    AssStyle,
    ass_color,
    ass_round_ms,
    ass_time,
    build_styles,
    caption_events,
    caption_max_chars,
    caption_pages,
    escape_text,
    karaoke,
    parse_events,
)
from narratoflow.errors import RenderError
from narratoflow.models import Language, SceneKind, WordTiming
from narratoflow.render.assets import default_fonts

SEEN3 = "\u0633\u0633\u0633"
QMARK_AR = "\u061f"
_K = re.compile(r"\{\\k(\d+)\}([^{]*)")


def _reveals(text: str, event_start_ms: int) -> list[tuple[str, int]]:
    """Rebuild (word, reveal_ms) from the \\k sums; the whole text must be \\k syllables."""
    assert "".join(m.group(0) for m in _K.finditer(text)) == text
    out: list[tuple[str, int]] = []
    cs = 0
    for m in _K.finditer(text):
        out.extend((word, event_start_ms + cs * 10) for word in m.group(2).split())
        cs += int(m.group(1))
    return out


def _scene_ar(words: list[TimelineWord]) -> list[AssEvent]:
    return caption_events(_scene(words), Language.ARABIC)


def _durations(text: str) -> list[int]:
    return [int(m.group(1)) for m in _K.finditer(text)]


def _w(text: str, start: int, end: int | None = None) -> TimelineWord:
    return TimelineWord(text=text, start_ms=start, end_ms=start if end is None else end)


def _scene(words: list[TimelineWord], end_ms: int = 10_000) -> SceneTiming:
    return SceneTiming(
        scene_id="s1",
        kind=SceneKind.NARRATION,
        start_ms=0,
        end_ms=end_ms,
        speech_start_ms=0,
        speech_end_ms=max([w.end_ms for w in words], default=0),
        words=words,
        timing_source=TimingSource.PROVIDER,
    )


def _texts(pages: list[list[WordTiming]]) -> list[list[str]]:
    return [[w.text for w in page] for page in pages]


def test_ass_time_and_color_format() -> None:
    assert ass_color("#112233") == "&H00332211"
    assert ass_color("#aabbcc", 0x99) == "&H99CCBBAA"
    assert ass_color("#FFFFFF", 255) == "&HFFFFFFFF"
    for bad_hex in ("112233", "#12345", "#GGGGGG", "#1122334"):
        with pytest.raises(RenderError, match="colour"):
            ass_color(bad_hex)
    for bad_alpha in (-1, 256):
        with pytest.raises(RenderError, match="alpha"):
            ass_color("#000000", bad_alpha)

    assert ass_time(0) == "0:00:00.00"
    assert ass_time(4) == "0:00:00.00"
    assert ass_time(5) == "0:00:00.01"  # half up
    assert ass_time(1234) == "0:00:01.23"
    assert ass_time(1235) == "0:00:01.24"
    assert ass_time(59_999) == "0:01:00.00"
    assert ass_time(3_661_005) == "1:01:01.01"
    with pytest.raises(RenderError, match="negative"):
        ass_time(-1)


def test_escape_braces_and_backslash() -> None:
    assert escape_text("a{\\b}c") == "a(/b)c"
    assert escape_text("one\ntwo\r\nthree\rfour\u2028five") == "one two three four five"
    assert escape_text("\\N\\h{\\fs200}") == "/N/h(/fs200)"
    assert escape_text(SEEN3 + ".") == SEEN3 + "."
    for raw in ("{\\pos(0,0)}x", "a\\Nb", "}{"):
        assert not set(escape_text(raw)) & {"{", "}", "\\", "\n", "\r"}


def test_karaoke_escapes_each_word() -> None:
    text = karaoke([("{\\fs300}big", 0), ("a\\Nb", 100)], 0)
    assert text == "{\\k10}(/fs300)big {\\k1}a/Nb"


def test_karaoke_cumulative_rounding_no_drift() -> None:
    for event_start in (0, 7, 1_005, 1_000):
        words = [(f"w{i}", event_start + i * 333) for i in range(200)]
        # The event itself starts at its centisecond-rounded time (ass_time).
        rounded_start = (event_start * 2 + 10) // 20 * 10
        rebuilt = _reveals(karaoke(words, event_start), rounded_start)
        assert [w for w, _ in rebuilt] == [w for w, _ in words]
        for (_, want), (_, got) in zip(words, rebuilt, strict=True):
            assert abs(got - want) <= 5


def test_karaoke_overlapping_and_zero_length_words() -> None:
    # Plan 4 contract: words may overlap or be zero-length; only starts are non-decreasing.
    words = [("a", 100), ("b", 100), ("c", 104), ("d", 105), ("e", 105), ("f", 2_000)]
    text = karaoke(words, 100)
    assert all(d >= 0 for d in _durations(text))
    assert [t for _, t in _reveals(text, 100)] == [100, 100, 100, 110, 110, 2_000]
    # A reveal before the event start clamps to the start rather than going negative.
    early = karaoke([("x", 50), ("y", 90), ("z", 300)], 100)
    assert all(d >= 0 for d in _durations(early))
    assert _reveals(early, 100) == [("x", 100), ("y", 100), ("z", 300)]


def test_karaoke_rejects_decreasing_starts() -> None:
    with pytest.raises(RenderError, match="non-decreasing"):
        karaoke([("a", 500), ("b", 400)], 0)


def test_karaoke_leading_offset_when_first_word_after_event_start() -> None:
    text = karaoke([("one", 250), ("two", 600)], 0)
    assert text.startswith("{\\k25}{\\k35}one")
    assert _reveals(text, 0) == [("one", 250), ("two", 600)]
    at_start = karaoke([("one", 0), ("two", 600)], 0)
    assert at_start == "{\\k60}one {\\k1}two"
    assert karaoke([], 0) == ""


def test_karaoke_no_trailing_space_single_space_joins() -> None:
    text = karaoke([("a", 0), ("b", 100), ("c", 200)], 0)
    assert text == "{\\k10}a {\\k10}b {\\k1}c"
    assert not text.endswith(" ")
    assert "  " not in text
    assert karaoke([("solo", 30)], 0) == "{\\k3}{\\k1}solo"


def test_pages_respect_limits_sentence_and_gap() -> None:
    assert CAPTION_MAX_WORDS == 6
    assert CAPTION_MAX_CHARS == 28
    assert CAPTION_GAP_MS == 700
    assert CAPTION_HOLD_MS == 400
    assert SENTENCE_END == (".", "!", "?", QMARK_AR, "\u2026")

    eight = [_w(str(i), i * 100) for i in range(8)]
    assert [len(p) for p in caption_pages(eight)] == [6, 2]

    # 28 characters including the 5 joining spaces fit; one more character does not.
    fits = [_w("aaaa", i * 100) for i in range(5)] + [_w("aaa", 500)]
    assert [len(p) for p in caption_pages(fits)] == [6]
    over = [_w("aaaa", i * 100) for i in range(6)]  # 24 letters, 29 with spaces
    assert [len(p) for p in caption_pages(over)] == [5, 1]

    sentences = [_w("Hi.", 0), _w("there", 100), _w("why?", 200), _w("so", 300)]
    assert _texts(caption_pages(sentences)) == [["Hi."], ["there", "why?"], ["so"]]
    arabic = [_w(SEEN3 + QMARK_AR, 0), _w(SEEN3, 100), _w(SEEN3 + "\u2026", 200), _w("x!", 300)]
    assert [len(p) for p in caption_pages(arabic)] == [1, 2, 1]

    gaps = [_w("a", 0), _w("b", 700), _w("c", 1_401), _w("d", 1_500)]
    assert _texts(caption_pages(gaps)) == [["a", "b"], ["c", "d"]]

    long_word = "x" * 30
    lone = [_w("a", 0), _w(long_word, 100), _w("b", 200), _w("c", 300)]
    assert _texts(caption_pages(lone)) == [["a"], [long_word], ["b", "c"]]
    assert _texts(caption_pages([_w(long_word, 0)])) == [[long_word]]
    assert caption_pages([]) == []


def test_caption_events_one_per_page_with_karaoke() -> None:
    words = [_w("Hello", 100, 400), _w("world.", 450, 900), _w("Next", 1_000, 1_300)]
    events = caption_events(_scene(words), Language.ENGLISH)
    assert [(e.start_ms, e.end_ms, e.style, e.layer) for e in events] == [
        (100, 1_000, "Caption", 0),
        (1_000, 1_700, "Caption", 0),
    ]
    assert events[0].text == karaoke([("Hello", 100), ("world.", 450)], 100)
    assert events[1].text == karaoke([("Next", 1_000)], 1_000)


def test_caption_events_escape_author_text() -> None:
    events = caption_events(
        _scene([_w("{\\an7}x", 0, 100), _w("y\\N", 100, 200)]), Language.ENGLISH
    )
    assert len(events) == 1
    assert "\\an7" not in events[0].text
    assert "\\N" not in events[0].text
    assert re.sub(r"\{\\k\d+\}", "", events[0].text).count("{") == 0


def test_last_page_held_but_not_past_scene() -> None:
    # Overlapping words: the hold runs from the latest end, not the last word's end.
    words = [_w("a", 0, 900), _w("b", 100, 800)]
    en = Language.ENGLISH
    assert caption_events(_scene(words, end_ms=5_000), en)[-1].end_ms == 900 + CAPTION_HOLD_MS
    assert caption_events(_scene(words, end_ms=1_000), en)[-1].end_ms == 1_000
    zero = [_w("a", 300), _w("b", 300, 310)]
    held = caption_events(_scene(zero, end_ms=310), en)
    assert [(e.start_ms, e.end_ms) for e in held] == [(300, 310)]


def test_zero_duration_pages_are_dropped() -> None:
    # A sentence end splits pages between two words with the same start: the first page
    # would last 0 ms and never show, so it is dropped rather than written.
    words = [_w("Done.", 300), _w("Next", 300, 600)]
    events = caption_events(_scene(words), Language.ENGLISH)
    assert [(e.start_ms, e.end_ms) for e in events] == [(300, 1_000)]
    assert "Next" in events[0].text
    assert "Done." not in events[0].text
    # A last page whose hold is clipped to a scene ending at its start is dropped too.
    assert caption_events(_scene([_w("a", 0)], end_ms=0), Language.ENGLISH) == []


def test_karaoke_never_emits_k0_and_groups_equal_reveals() -> None:
    # libass does not start a syllable at \k0: the word would light with the one before.
    text = karaoke([("a", 0), ("b", 1_000), ("c", 1_000), ("d", 2_000)], 0)
    assert text == "{\\k100}a {\\k100}b c {\\k1}d"
    assert "\\k0}" not in text
    assert _reveals(text, 0) == [("a", 0), ("b", 1_000), ("c", 1_000), ("d", 2_000)]


def test_karaoke_offsets_are_absolute_rounded() -> None:
    # c_i = cs(reveal_i) - cs(event_start): every absolute reveal is within 5 ms.
    text = karaoke([("a", 1_005), ("b", 1_340)], 1_005)
    assert text == "{\\k33}a {\\k1}b"
    event_cs = int(ass_time(1_005)[-2:]) * 10 + 1_000
    assert event_cs == 1_010
    rebuilt = _reveals(text, event_cs)
    assert all(abs(got - want) <= 5 for (_, got), want in zip(rebuilt, (1_005, 1_340), strict=True))


def test_caption_events_empty_for_verse_or_no_words() -> None:
    assert caption_events(_scene([]), Language.ENGLISH) == []
    verse = SceneTiming(
        scene_id="v1",
        kind=SceneKind.VERSE,
        start_ms=0,
        end_ms=1_000,
        speech_start_ms=0,
        speech_end_ms=1_000,
        words=[],
        timing_source=TimingSource.AYAH,
        ayat=[AyahTiming(surah=31, ayah=33, start_ms=0, end_ms=1_000)],
    )
    assert caption_events(verse, Language.ARABIC) == []


def test_arabic_caption_pages_use_the_one_line_limit() -> None:
    assert caption_max_chars(Language.ENGLISH) == CAPTION_MAX_CHARS
    assert caption_max_chars(Language.ARABIC) == CAPTION_MAX_CHARS_AR == 28
    # Tashkeel (combining marks) takes no width, so it does not count toward the limit.
    fatha = "\u064e"
    voweled = [_w((SEEN3 + fatha) * 1, i * 100) for i in range(7)]
    assert [len(p) for p in caption_pages(voweled, CAPTION_MAX_CHARS_AR)] == [6, 1]
    long_words = [_w(SEEN3 * 2, i * 100) for i in range(5)]  # 6 + 1 + 6 + 1 + 6 + 1 + 6 = 27
    events = _scene_ar(long_words)
    assert len(events) == 2


# name: (Arabic Fontsize, English Fontsize, bold, alignment, (L, R, V)).
# English sizes are the brief's; Arabic sizes give Amiri the same em (39.5 px for Caption).
_EXPECTED: dict[str, tuple[int, int, bool, int, tuple[int, int, int]]] = {
    "Caption": (109, 60, True, 8, (60, 192, 1260)),
    "Verse": (139, 139, False, 5, (84, 216, 0)),
    "VerseRef": (73, 40, False, 2, (84, 216, 696)),
    "TopLeft": (65, 36, True, 7, (60, 60, 172)),
    "TopRight": (65, 36, False, 9, (60, 60, 172)),
    "Headline": (84, 46, True, 8, (60, 60, 240)),
    "Card": (160, 88, True, 5, (84, 216, 0)),
}


def _by_name(styles: list[AssStyle]) -> dict[str, AssStyle]:
    return {s.name: s for s in styles}


def test_arabic_styles_auto_direction_english_ltr() -> None:
    fonts = default_fonts()
    ar = _by_name(build_styles(Language.ARABIC, "#111111", fonts))
    en = _by_name(build_styles(Language.ENGLISH, "#111111", fonts))
    assert set(ar) == set(en) == set(_EXPECTED)

    assert all(s.encoding == -1 for s in ar.values())
    assert {n: s.encoding for n, s in en.items()} == {n: -1 if n == "Verse" else 1 for n in en}

    amiri = fonts.get("amiri-bold").family
    noto = fonts.get("notosans-bold").family
    quran = fonts.get("amiri-quran").family
    assert ar["Verse"].family == en["Verse"].family == quran
    assert {s.family for n, s in ar.items() if n != "Verse"} == {amiri}
    assert {s.family for n, s in en.items() if n != "Verse"} == {noto}

    for styles, col in ((ar, 0), (en, 1)):
        for name, row in _EXPECTED.items():
            s = styles[name]
            bold, align, (ml, mr, mv) = row[2], row[3], row[4]
            assert s.size == row[col], name
            assert (s.bold, s.alignment, s.margin_l, s.margin_r, s.margin_v) == (
                bold,
                align,
                ml,
                mr,
                mv,
            ), name
            assert s.primary == "#111111"
            assert s.primary_alpha == 0
        cap = styles["Caption"]
        assert (cap.secondary, cap.secondary_alpha) == ("#111111", 0x99)
        assert (cap.outline, cap.outline_px) == ("#FFFFFF", 3)
        verse = styles["Verse"]
        assert (verse.secondary_alpha, verse.outline_px) == (0xFF, 0)


def _doc(language: Language = Language.ENGLISH) -> AssDocument:
    styles = build_styles(language, "#111111", default_fonts())
    return AssDocument(styles=styles, events=[])


def test_never_kf() -> None:
    doc = _doc()
    words = [_w(f"w{i}", i * 150, i * 150 + 100) for i in range(20)]
    doc.events = caption_events(_scene(words), Language.ENGLISH)
    text = doc.to_text()
    assert "\\k" in text
    assert "\\kf" not in text
    assert "\\K" not in text
    assert "\\ko" not in text


def test_document_sections_and_playres(tmp_path: Path) -> None:
    doc = _doc(Language.ARABIC)
    doc.events = [
        AssEvent(2_000, 3_000, "Caption", "late"),
        AssEvent(0, 1_000, "TopLeft", "label", layer=1),
        AssEvent(0, 1_000, "Caption", SEEN3),
    ]
    text = doc.to_text()
    lines = text.splitlines()
    assert lines[0] == "[Script Info]"
    for line in (
        "ScriptType: v4.00+",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: None",
    ):
        assert line in lines
    assert lines.index("[Script Info]") < lines.index("[V4+ Styles]") < lines.index("[Events]")
    style_fmt = lines[lines.index("[V4+ Styles]") + 1]
    assert style_fmt == (
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
    )
    assert (
        "Style: Caption,Amiri,109,&H00111111,&H99111111,&H00FFFFFF,&HFF000000,"
        "-1,0,0,0,100,100,0,0,1,3,0,8,60,192,1260,-1"
    ) in lines
    assert (
        "Style: Verse,Amiri Quran,139,&H00111111,&HFF111111,&H00FFFFFF,&HFF000000,"
        "0,0,0,0,100,100,0,0,1,0,0,5,84,216,0,-1"
    ) in lines
    dialogues = [ln for ln in lines if ln.startswith("Dialogue:")]
    assert dialogues == [
        f"Dialogue: 0,0:00:00.00,0:00:01.00,Caption,,0,0,0,,{SEEN3}",
        "Dialogue: 1,0:00:00.00,0:00:01.00,TopLeft,,0,0,0,,label",
        "Dialogue: 0,0:00:02.00,0:00:03.00,Caption,,0,0,0,,late",
    ]
    events_fmt = lines[lines.index("[Events]") + 1]
    assert events_fmt == (
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
    )

    out = tmp_path / "captions.ass"
    out.write_text("old", encoding="utf-8")
    doc.write(out)
    raw = out.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert raw.decode("utf-8") == text
    assert b"\r\n" not in raw
    assert sorted(p.name for p in tmp_path.iterdir()) == ["captions.ass"]


def test_write_failure_leaves_no_temp_file(tmp_path: Path) -> None:
    target = tmp_path / "is-a-dir"
    target.mkdir()
    with pytest.raises(RenderError, match="cannot write"):
        _doc().write(target)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["is-a-dir"]
    with pytest.raises(RenderError, match="cannot write"):
        _doc().write(tmp_path / "missing" / "captions.ass")


def test_arabic_caption_pages_weight_latin_like_truncate() -> None:
    latin = [_w("WWWW", i * 100) for i in range(5)] + [_w("WWW", 500)]  # 28 characters
    assert [len(p) for p in caption_pages(latin, CAPTION_MAX_CHARS_AR)] == [6]
    # In an Arabic style each Latin letter counts 5/4 (spaces 1): five words are
    # 20 * 5 + 4 * 4 = 116 > 112 quarter units, so a page holds four.
    assert [len(p) for p in caption_pages(latin, CAPTION_MAX_CHARS_AR, Language.ARABIC)] == [4, 2]
    arabic = [_w(SEEN3 + SEEN3[0], i * 100) for i in range(5)] + [_w(SEEN3, 500)]
    assert [len(p) for p in caption_pages(arabic, CAPTION_MAX_CHARS_AR, Language.ARABIC)] == [6]


def test_ass_round_ms_is_what_ass_time_writes() -> None:
    assert [ass_round_ms(ms) for ms in (0, 4, 5, 1004, 1005, 1994, 1995)] == [
        0,
        0,
        10,
        1000,
        1010,
        1990,
        2000,
    ]
    assert ass_time(1005) == "0:00:01.01"


def test_parse_events_roundtrip_with_commas_and_tags() -> None:
    events = [
        AssEvent(3004, 5005, "Caption", "{\\k10}a, b,{\\k5} c"),
        AssEvent(0, 3_600_000 + 61_234, "Verse", "{\\an5\\pos(474,815)}" + SEEN3 + ", " + SEEN3),
        AssEvent(0, 9, "VerseRef", "31:33", layer=2),
        AssEvent(0, 1000, "TopLeft", "x,,y", layer=1),
    ]
    doc = AssDocument(
        styles=build_styles(Language.ARABIC, "#000000", default_fonts()), events=events
    )
    want = [
        AssEvent(ass_round_ms(e.start_ms), ass_round_ms(e.end_ms), e.style, e.text, e.layer)
        for e in sorted(events, key=lambda e: (e.start_ms, e.layer))
    ]
    assert parse_events(doc.to_text()) == want


def test_parse_events_ignores_lines_outside_events() -> None:
    text = "[Script Info]\nDialogue: nonsense\n[Events]\nFormat: x\nComment: 0,bad\n\n"
    text += "Dialogue: 0,0:00:01.00,0:00:02.50,Caption,,0,0,0,,hi\n[Fonts]\nDialogue: junk\n"
    assert parse_events(text) == [AssEvent(1000, 2500, "Caption", "hi")]


@pytest.mark.parametrize(
    "line",
    [
        "Dialogue: 0,0:00:01.00,0:00:02.00,Caption",
        "Dialogue: x,0:00:01.00,0:00:02.00,Caption,,0,0,0,,hi",
        "Dialogue: 0,0:00:01.0,0:00:02.00,Caption,,0,0,0,,hi",
        "Dialogue: 0,0:00:01.00,0:0:02.00,Caption,,0,0,0,,hi",
        "Dialogue: 0,0:00:01.00,0:00:61.00,Caption,,0,0,0,,hi",
    ],
)
def test_parse_events_rejects_malformed_line(line: str) -> None:
    with pytest.raises(ValueError, match="line 3") as info:
        parse_events(f"[Events]\nFormat: x\n{line}\n")
    assert line in str(info.value)
