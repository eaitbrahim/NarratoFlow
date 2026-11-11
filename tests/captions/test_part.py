"""The part ASS: every event source composed, glyph warnings and verse failures."""

from decimal import Decimal

import pytest
from conftest import FIXTURES

from narratoflow.audio.recitation import display_words
from narratoflow.audio.timeline import (
    AyahTiming,
    PartTimeline,
    SceneTiming,
    TimelineWord,
    TimingSource,
)
from narratoflow.captions.part import build_part_ass
from narratoflow.captions.verse import shown_text
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
    Visual,
    VisualSource,
)
from narratoflow.quran.corpus import QuranCorpus
from narratoflow.render.assets import default_fonts

Q = FIXTURES / "quran"
INK = "#111111"
SEEN = "\u0633\u0644\u0627\u0645"


@pytest.fixture(scope="module")
def corpus() -> QuranCorpus:
    return QuranCorpus.from_files(Q / "quran-simple-clean.txt", Q / "quran-uthmani.txt")


def _narration_slot(sid: str, start: int, texts: list[str]) -> SceneTiming:
    words = [
        TimelineWord(text=t, start_ms=start + i * 200, end_ms=start + i * 200 + 150)
        for i, t in enumerate(texts)
    ]
    return SceneTiming(
        scene_id=sid,
        kind=SceneKind.NARRATION,
        start_ms=start,
        end_ms=start + 2_000,
        speech_start_ms=start,
        speech_end_ms=words[-1].end_ms,
        words=words,
        timing_source=TimingSource.PROVIDER,
    )


def _verse_slot(corpus: QuranCorpus, start: int, texts: list[str] | None = None) -> SceneTiming:
    texts = texts or display_words(corpus.ayah(19, 2).uthmani)
    words = [
        TimelineWord(text=t, start_ms=start + i * 300, end_ms=start + i * 300 + 250)
        for i, t in enumerate(texts)
    ]
    end = words[-1].end_ms
    return SceneTiming(
        scene_id="v1",
        kind=SceneKind.VERSE,
        start_ms=start,
        end_ms=end + 200,
        speech_start_ms=start,
        speech_end_ms=end,
        words=[],
        timing_source=TimingSource.SEGMENTS,
        ayat=[AyahTiming(surah=19, ayah=2, start_ms=start, end_ms=end, words=words)],
    )


def _setup(
    corpus: QuranCorpus,
    language: Language,
    caption: list[str],
    verse_texts: list[str] | None = None,
) -> tuple[Series, Part, PartTimeline]:
    scenes = [
        Scene(id="n1", narration=Narration(source_text=" ".join(caption)), title="Topic"),
        Scene(
            id="c1",
            narration=Narration(source_text="card words here"),
            visual=Visual(source=VisualSource.TEXT_CARD),
            card_text="On the card",
        ),
        Scene(
            id="v1",
            narration=Narration(source_text=""),
            kind=SceneKind.VERSE,
            verse=VerseRef(surah=19, ayah_start=2, ayah_end=2, certain=True, uthmani_text="x"),
        ),
    ]
    part = Part(index=1, count=2, hook_text="Hook", teaser_text="Next one", scenes=scenes)
    series = Series(
        title="Series",
        channel_slug="chan",
        language=language,
        source=ScriptSource(author="a", work="w", rights=RightsBasis.OWN),
        parts=[part],
    )
    slots = [
        _narration_slot("n1", 0, caption),
        _narration_slot("c1", 2_000, ["card", "words", "here"]),
        _verse_slot(corpus, 4_000, verse_texts),
    ]
    timeline = PartTimeline(
        series_title="Series",
        channel_slug="chan",
        part_index=1,
        part_count=2,
        audio_file="part-01.flac",
        duration_ms=slots[-1].end_ms,
        integrated_lufs=None,
        true_peak_dbtp=None,
        scenes=slots,
        sfx=[],
        voice=None,
        recitations=[],
        assets=[],
        spent_usd=Decimal(0),
        cost_estimated=False,
        warnings=[],
    )
    return series, part, timeline


def test_build_part_ass_composes_every_source(corpus: QuranCorpus) -> None:
    series, part, timeline = _setup(corpus, Language.ENGLISH, ["hello", "world"])
    doc, warnings = build_part_ass(
        series, part, timeline, corpus=corpus, fonts=default_fonts(), ink=INK
    )
    assert warnings == []
    styles = sorted({e.style for e in doc.events})
    assert styles == [
        "Caption",
        "Card",
        "Headline",
        "TopLeft",
        "TopRight",
        "Verse",
        "VerseRef",
    ]
    assert [s.name for s in doc.styles][:2] == ["Caption", "Verse"]
    assert shown_text(doc.events) == " ".join(corpus.ayah(19, 2).uthmani.split())
    card = next(e for e in doc.events if e.style == "Card")
    assert card.text.endswith("On the card")
    assert (card.start_ms, card.end_ms) == (2_000, 4_000)
    assert "Dialogue: 0,0:00:00.00" in doc.to_text()


def test_build_part_ass_warns_on_missing_caption_glyphs(corpus: QuranCorpus) -> None:
    series, part, timeline = _setup(corpus, Language.ENGLISH, ["hello", SEEN])
    part = part.model_copy(update={"hook_text": "Hook " + SEEN})
    _, warnings = build_part_ass(
        series, part, timeline, corpus=corpus, fonts=default_fonts(), ink=INK
    )
    assert warnings == [
        "n1: caption glyphs missing from notosans-bold: U+0627 U+0633 U+0644 U+0645",
        "overlays: caption glyphs missing from notosans-bold: U+0627 U+0633 U+0644 U+0645",
    ]


def test_build_part_ass_raises_on_stale_verse(corpus: QuranCorpus) -> None:
    stale = [*display_words(corpus.ayah(19, 2).uthmani)[:-1], SEEN]
    series, part, timeline = _setup(corpus, Language.ARABIC, ["a"], verse_texts=stale)
    with pytest.raises(RenderError, match=r"^FAIL: verse 19:2 "):
        build_part_ass(series, part, timeline, corpus=corpus, fonts=default_fonts(), ink=INK)


def test_build_part_ass_needs_corpus_for_verses(corpus: QuranCorpus) -> None:
    series, part, timeline = _setup(corpus, Language.ARABIC, ["a"])
    with pytest.raises(RenderError, match=r"^FAIL: verse scene v1 "):
        build_part_ass(series, part, timeline, corpus=None, fonts=default_fonts(), ink=INK)


def test_build_part_ass_rejects_unknown_scene(corpus: QuranCorpus) -> None:
    series, part, timeline = _setup(corpus, Language.ENGLISH, ["a"])
    part = part.model_copy(update={"scenes": part.scenes[1:]})
    with pytest.raises(RenderError, match="n1"):
        build_part_ass(series, part, timeline, corpus=corpus, fonts=default_fonts(), ink=INK)
