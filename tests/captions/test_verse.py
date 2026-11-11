"""Verse reveal: corpus text only, word-exact karaoke, pages and the reference label."""

import itertools
import re
from pathlib import Path

import pytest
from conftest import FIXTURES

from narratoflow.audio.recitation import display_words
from narratoflow.audio.timeline import AyahTiming, SceneTiming, TimelineWord, TimingSource
from narratoflow.captions.ass import AssEvent
from narratoflow.captions.verse import (
    VERSE_PAGE_MAX_WORDS,
    VersePage,
    check_verse_text,
    shown_text,
    verse_events,
    verse_pages,
    verse_ref_label,
)
from narratoflow.errors import RenderError
from narratoflow.models import Language, Narration, Scene, SceneKind, VerseRef
from narratoflow.quran.corpus import QuranCorpus
from narratoflow.render.assets import FontPack, default_fonts
from narratoflow.render.ffmpeg import RenderTools
from narratoflow.render.layout import STAGE

from .stills import assert_in_zone, bbox, ink, still

Q = FIXTURES / "quran"


@pytest.fixture(scope="module")
def corpus() -> QuranCorpus:
    return QuranCorpus.from_files(Q / "quran-simple-clean.txt", Q / "quran-uthmani.txt")


def verse_scene(surah: int = 31, start: int = 33, end: int = 33) -> Scene:
    return Scene(
        id="v1",
        narration=Narration(source_text=""),
        kind=SceneKind.VERSE,
        verse=VerseRef(
            surah=surah, ayah_start=start, ayah_end=end, certain=True, uthmani_text="pasted"
        ),
    )


def segments_timing(
    corpus: QuranCorpus,
    keys: list[tuple[int, int]],
    *,
    lead_ms: int = 0,
    step_ms: int = 300,
    texts: dict[tuple[int, int], list[str]] | None = None,
) -> SceneTiming:
    """SEGMENTS timing: words `step_ms` apart, the first `lead_ms` after the ayah start."""
    ayat: list[AyahTiming] = []
    t = 0
    for s, a in keys:
        words = (texts or {}).get((s, a)) or display_words(corpus.ayah(s, a).uthmani)
        start = t
        starts = [start + lead_ms + i * step_ms for i in range(len(words))]
        timed = [
            TimelineWord(text=w, start_ms=b, end_ms=b + step_ms - 20)
            for w, b in zip(words, starts, strict=True)
        ]
        end = timed[-1].end_ms + 100
        ayat.append(AyahTiming(surah=s, ayah=a, start_ms=start, end_ms=end, words=timed))
        t = end
    return SceneTiming(
        scene_id="v1",
        kind=SceneKind.VERSE,
        start_ms=0,
        end_ms=t + 400,
        speech_start_ms=0,
        speech_end_ms=t,
        words=[],
        timing_source=TimingSource.SEGMENTS,
        ayat=ayat,
    )


def ayah_timing(spans: list[tuple[int, int, int, int]], end_ms: int) -> SceneTiming:
    return SceneTiming(
        scene_id="v1",
        kind=SceneKind.VERSE,
        start_ms=0,
        end_ms=end_ms,
        speech_start_ms=0,
        speech_end_ms=end_ms,
        words=[],
        timing_source=TimingSource.AYAH,
        ayat=[AyahTiming(surah=s, ayah=a, start_ms=b, end_ms=e) for s, a, b, e in spans],
    )


def test_shown_text_equals_corpus_for_31_33(corpus: QuranCorpus) -> None:
    timing = segments_timing(corpus, [(31, 33)])
    events = verse_events(verse_scene(), timing, corpus, Language.ARABIC)
    uthmani = corpus.ayah(31, 33).uthmani
    assert shown_text(events) == " ".join(uthmani.split())
    # Never the pasted text.
    assert "pasted" not in shown_text(events)


def test_stale_word_text_is_fail(corpus: QuranCorpus) -> None:
    words = display_words(corpus.ayah(31, 33).uthmani)
    stale = [*words[:-1], "\u0633\u0633\u0633"]
    timing = segments_timing(corpus, [(31, 33)], texts={(31, 33): stale})
    with pytest.raises(RenderError, match=r"^FAIL: verse 31:33 "):
        check_verse_text(verse_scene(), timing, corpus, default_fonts())


def test_ayah_range_mismatch_is_fail(corpus: QuranCorpus) -> None:
    timing = segments_timing(corpus, [(31, 33)])
    with pytest.raises(RenderError, match=r"^FAIL: verse 19:4"):
        check_verse_text(verse_scene(19, 4, 4), timing, corpus, default_fonts())
    two = segments_timing(corpus, [(19, 4), (19, 5)])
    with pytest.raises(RenderError, match=r"^FAIL: verse 19:4-6"):
        check_verse_text(verse_scene(19, 4, 6), two, corpus, default_fonts())


def test_missing_verse_glyph_is_fail(corpus: QuranCorpus, monkeypatch: pytest.MonkeyPatch) -> None:
    timing = segments_timing(corpus, [(31, 33)])
    calls: list[str] = []

    def fake(self: FontPack, font_id: str, text: str) -> list[str]:
        calls.append(font_id)
        return ["U+06DF"]

    monkeypatch.setattr(FontPack, "missing_glyphs", fake)
    with pytest.raises(RenderError, match=r"^FAIL: verse 31:33 .*U\+06DF"):
        check_verse_text(verse_scene(), timing, corpus, default_fonts())
    assert calls == ["amiri-quran"]


def test_valid_verse_passes(corpus: QuranCorpus) -> None:
    timing = segments_timing(corpus, [(19, 4), (19, 5)])
    check_verse_text(verse_scene(19, 4, 5), timing, corpus, default_fonts())
    ayah = ayah_timing([(19, 4, 0, 1_000)], 1_200)
    check_verse_text(verse_scene(19, 4, 4), ayah, corpus, default_fonts())


def test_scene_without_verse_ref_is_fail(corpus: QuranCorpus) -> None:
    scene = verse_scene().model_copy(update={"verse": None})
    with pytest.raises(RenderError, match=r"^FAIL: verse "):
        check_verse_text(scene, segments_timing(corpus, [(31, 33)]), corpus, default_fonts())
    with pytest.raises(RenderError, match=r"^FAIL: verse "):
        verse_events(scene, segments_timing(corpus, [(31, 33)]), corpus, Language.ARABIC)


def test_ayah_timing_reveals_page_at_once_and_long_ayah_paged(corpus: QuranCorpus) -> None:
    timing = ayah_timing([(19, 4, 0, 2_000), (19, 5, 2_000, 9_000)], 9_500)
    one = verse_pages(timing, corpus, verse_scene(19, 4, 5).verse)  # type: ignore[arg-type]
    first = [p for p in one if p.ayah == 4]
    assert len(display_words(corpus.ayah(19, 4).uthmani)) > VERSE_PAGE_MAX_WORDS
    assert len(first) == 2
    for p in one:
        assert {r for _, r in p.words} == {p.start_ms}
    assert first[0].start_ms == 0
    assert first[1].start_ms > 0
    second = [p for p in one if p.ayah == 5]
    assert second[0].start_ms == 2_000
    assert one[-1].end_ms == 9_500
    for a, b in itertools.pairwise(one):
        assert a.end_ms == b.start_ms
    # Page starts follow character weight over the ayah span.
    chars = [len(" ".join(t for t, _ in p.words)) for p in first]
    expected = round(2_000 * chars[0] / sum(chars))
    assert abs(first[1].start_ms - expected) <= 1


def test_one_page_ayah_appears_whole_at_ayah_start(corpus: QuranCorpus) -> None:
    timing = ayah_timing([(19, 2, 500, 3_000)], 3_000)
    timing = timing.model_copy(update={"speech_start_ms": 0})
    pages = verse_pages(timing, corpus, verse_scene(19, 2, 2).verse)  # type: ignore[arg-type]
    assert len(pages) == 1
    assert pages[0].start_ms == 500
    assert all(r == 500 for _, r in pages[0].words)


def test_pages_cover_all_words_in_order(corpus: QuranCorpus) -> None:
    keys = [(19, 4), (19, 5), (31, 33)]
    for timing in (
        segments_timing(corpus, [(19, 4), (19, 5)]),
        ayah_timing([(19, 4, 0, 3_000), (19, 5, 3_000, 6_000)], 6_000),
    ):
        verse = VerseRef(surah=19, ayah_start=4, ayah_end=5, certain=True, uthmani_text="")
        pages = verse_pages(timing, corpus, verse)
        for s, a in keys[:2]:
            got = [t for p in pages if (p.surah, p.ayah) == (s, a) for t, _ in p.words]
            assert got == display_words(corpus.ayah(s, a).uthmani)
        assert all(1 <= len(p.words) <= VERSE_PAGE_MAX_WORDS for p in pages)
        reveals = [r for p in pages for _, r in p.words]
        assert reveals == sorted(reveals)


def test_segments_first_page_starts_at_ayah_start_later_pages_at_first_word(
    corpus: QuranCorpus,
) -> None:
    timing = segments_timing(corpus, [(31, 33)], lead_ms=450)
    pages = verse_pages(timing, corpus, verse_scene().verse)  # type: ignore[arg-type]
    ayah = timing.ayat[0]
    assert len(pages) == 3
    assert pages[0].start_ms == ayah.start_ms == 0
    assert pages[0].words[0][1] == 450
    for p in pages[1:]:
        assert p.start_ms == p.words[0][1]
    assert [r for p in pages for _, r in p.words] == [w.start_ms for w in ayah.words]
    assert pages[-1].end_ms == timing.end_ms


def _reveals(event: AssEvent, texts: list[str]) -> list[int]:
    """Each word's reveal time rebuilt from the event's ``\\k`` durations."""
    body = re.sub(r"^\{[^}]*\\pos[^}]*\}", "", event.text)
    parts = re.split(r"\{\\k(\d+)\}", body)
    assert parts[0] == ""
    t = (event.start_ms + 5) // 10 * 10  # libass starts the event at its centisecond
    pending = list(texts)
    out: list[int] = []
    for k in range(1, len(parts), 2):
        seg = parts[k + 1].strip()
        while seg:
            word = pending.pop(0)
            assert seg.startswith(word), (seg, word)
            out.append(t)
            seg = seg[len(word) :].strip()
        t += int(parts[k]) * 10
    assert not pending
    return out


def test_verse_words_never_revealed_early(corpus: QuranCorpus) -> None:
    timing = segments_timing(corpus, [(31, 33)], lead_ms=455, step_ms=333)
    events = verse_events(verse_scene(), timing, corpus, Language.ARABIC)
    pages = [e for e in events if e.style == "Verse"]
    assert pages[0].start_ms == 0
    assert pages[0].text.startswith("{\\an5\\pos(474,815)\\fad(150,0)}{\\k46}")
    words = timing.ayat[0].words
    built = verse_pages(timing, corpus, verse_scene().verse)  # type: ignore[arg-type]
    assert [p.start_ms for p in built] == [e.start_ms for e in pages]
    got: list[int] = []
    for e, page in zip(pages, built, strict=True):
        got += _reveals(e, [t for t, _ in page.words])
    assert len(got) == len(words)
    for g, w in zip(got, words, strict=True):
        assert abs(g - w.start_ms) <= 5, (g, w.start_ms)


def test_verse_ref_label_digits() -> None:
    one = VerseRef(surah=31, ayah_start=33, ayah_end=33, certain=True, uthmani_text="")
    rng = VerseRef(surah=31, ayah_start=33, ayah_end=35, certain=True, uthmani_text="")
    assert verse_ref_label(one, Language.ENGLISH) == "31:33"
    assert verse_ref_label(rng, Language.ENGLISH) == "31:33-35"
    assert verse_ref_label(rng, Language.ARABIC) == ("\u0663\u0661:\u0663\u0663-\u0663\u0665")


def test_verse_ref_event_spans_scene(corpus: QuranCorpus) -> None:
    timing = segments_timing(corpus, [(31, 33)])
    events = verse_events(verse_scene(), timing, corpus, Language.ENGLISH)
    refs = [e for e in events if e.style == "VerseRef"]
    assert refs == [AssEvent(timing.start_ms, timing.end_ms, "VerseRef", "31:33")]


def test_zero_length_pages_are_skipped(corpus: QuranCorpus) -> None:
    timing = ayah_timing([(19, 4, 0, 0), (19, 5, 0, 2_000)], 2_000)
    events = verse_events(verse_scene(19, 4, 5), timing, corpus, Language.ARABIC)
    verse = [e for e in events if e.style == "Verse"]
    assert all(e.end_ms > e.start_ms for e in verse)


def test_segments_word_count_mismatch_is_fail(corpus: QuranCorpus) -> None:
    words = display_words(corpus.ayah(31, 33).uthmani)
    timing = segments_timing(corpus, [(31, 33)], texts={(31, 33): words[:-1]})
    with pytest.raises(RenderError, match=r"^FAIL: verse 31:33 "):
        verse_pages(timing, corpus, verse_scene().verse)  # type: ignore[arg-type]
    with pytest.raises(RenderError, match=r"^FAIL: verse 31:33 "):
        check_verse_text(verse_scene(), timing, corpus, default_fonts())


def test_verse_page_is_frozen() -> None:
    page = VersePage(surah=1, ayah=1, words=[("x", 0)], start_ms=0, end_ms=1)
    with pytest.raises(AttributeError):
        page.start_ms = 2  # type: ignore[misc]


def test_verse_page_ink_inside_stage(
    corpus: QuranCorpus, render_tools: RenderTools, tmp_path: Path
) -> None:
    # Every page of 31:33 (the 10-word first page is long, with full tashkeel) and of
    # 19:4-5, fully revealed, plus the reference line, in both channel languages.
    scenes = [
        (verse_scene(), segments_timing(corpus, [(31, 33)])),
        (verse_scene(19, 4, 5), segments_timing(corpus, [(19, 4), (19, 5)])),
    ]
    for language in Language:
        for n, (scene, timing) in enumerate(scenes):
            events = verse_events(scene, timing, corpus, language)
            ref = next(e for e in events if e.style == "VerseRef")
            work = tmp_path / f"{language.value}{n}"
            label = bbox(ink(still(render_tools, work / "ref", language, [ref], 500)))
            assert_in_zone(label, STAGE, f"{language.value} verse ref")
            for i, page in enumerate(e for e in events if e.style == "Verse"):
                frame = still(render_tools, work / str(i), language, [page], page.end_ms - 50)
                box = bbox(ink(frame))
                assert_in_zone(box, STAGE, f"{language.value} {scene.id} page {i}")
                assert label.y >= box.bottom, (label, box)


def test_verse_range_label_bidi_order(render_tools: RenderTools, tmp_path: Path) -> None:
    rng = VerseRef(surah=31, ayah_start=33, ayah_end=35, certain=True, uthmani_text="")
    text = verse_ref_label(rng, Language.ARABIC)
    head, tail = text.split("-")
    assert (head, tail) == ("\u0663\u0661:\u0663\u0663", "\u0663\u0665")
    # Inline alpha hides one part; libass resolves bidi over the whole line, so the
    # visible part keeps the position it has in the full label.
    hide, show = "{\\1a&HFF&}", "{\\1a&H00&}"
    shots = {
        "head": f"{head}{hide}-{tail}",  # only 31:33 inked
        "tail": f"{hide}{head}-{show}{tail}",  # only 35 inked
    }
    boxes = {
        name: bbox(
            ink(
                still(
                    render_tools,
                    tmp_path / name,
                    Language.ARABIC,
                    [AssEvent(0, 1_000, "VerseRef", body)],
                    500,
                )
            )
        )
        for name, body in shots.items()
    }
    assert boxes["tail"].right < boxes["head"].x, boxes
