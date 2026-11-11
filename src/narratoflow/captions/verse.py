"""Verse reveal: the corpus Uthmani text, paged and lit word by word on its recitation.

The on-screen text always comes from ``corpus.ayah(s, a).uthmani`` cut by
`display_words`; never the pasted text, never the timeline's word texts. With
``SEGMENTS`` timing each word lights at its own ``start_ms``; with ``AYAH`` timing a
page appears whole at its share of the ayah span.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from narratoflow.audio.recitation import display_words
from narratoflow.audio.timeline import AyahTiming, SceneTiming, TimingSource
from narratoflow.audio.words import split_ms
from narratoflow.captions.ass import AssEvent, karaoke
from narratoflow.errors import RenderError
from narratoflow.models import Language, Scene, VerseRef
from narratoflow.quran.corpus import QuranCorpus
from narratoflow.render.assets import FontPack

VERSE_PAGE_MAX_WORDS = 10
VERSE_POS = (474, 815)  # the stage centre at 1080x1920
VERSE_FADE_MS = 150
VERSE_FONT = "amiri-quran"

_ARABIC_DIGITS = str.maketrans("0123456789", "".join(chr(0x0660 + d) for d in range(10)))
_TAGS = re.compile(r"\{[^}]*\}")


@dataclass(frozen=True)
class VersePage:
    surah: int
    ayah: int
    words: list[tuple[str, int]]  # (display word, reveal ms)
    start_ms: int
    end_ms: int


def localize_digits(text: str, language: Language) -> str:
    """Arabic-Indic digits (U+0660..U+0669) for an Arabic channel; unchanged otherwise."""
    return text.translate(_ARABIC_DIGITS) if language is Language.ARABIC else text


def verse_ref_label(verse: VerseRef, language: Language) -> str:
    """``"<s>:<a>"`` or ``"<s>:<a>-<b>"``, with the channel's digits."""
    label = f"{verse.surah}:{verse.ayah_start}"
    if verse.ayah_end != verse.ayah_start:
        label += f"-{verse.ayah_end}"
    return localize_digits(label, language)


def _verse_of(scene: Scene) -> VerseRef:
    if scene.verse is None:
        raise RenderError(f"FAIL: verse scene {scene.id} has no verse reference")
    return scene.verse


def _check_ayat(timing: SceneTiming, verse: VerseRef) -> None:
    want = [(verse.surah, a) for a in range(verse.ayah_start, verse.ayah_end + 1)]
    got = [(a.surah, a.ayah) for a in timing.ayat]
    if got != want:
        shown = ", ".join(f"{s}:{a}" for s, a in got) or "none"
        raise RenderError(
            f"FAIL: verse {verse_ref_label(verse, Language.ENGLISH)}: the timeline has "
            f"ayat {shown}; rebuild audio: narratoflow audio build"
        )


def _segment_words(ayah: AyahTiming, texts: list[str]) -> list[tuple[str, int]]:
    timed = [w.text for w in ayah.words]
    if timed != texts:
        raise RenderError(
            f"FAIL: verse {ayah.surah}:{ayah.ayah} word texts differ from the corpus "
            f"({len(timed)} timed, {len(texts)} in the corpus); "
            "rebuild audio: narratoflow audio build"
        )
    return [(t, w.start_ms) for t, w in zip(texts, ayah.words, strict=True)]


def check_verse_text(
    scene: Scene, timing: SceneTiming, corpus: QuranCorpus, fonts: FontPack
) -> None:
    """Raise ``FAIL: verse ...`` unless the timeline matches the corpus and the font."""
    verse = _verse_of(scene)
    _check_ayat(timing, verse)
    for ayah in timing.ayat:
        uthmani = corpus.ayah(ayah.surah, ayah.ayah).uthmani
        if timing.timing_source is TimingSource.SEGMENTS:
            _segment_words(ayah, display_words(uthmani))
        missing = fonts.missing_glyphs(VERSE_FONT, uthmani)
        if missing:
            raise RenderError(
                f"FAIL: verse {ayah.surah}:{ayah.ayah} has glyphs missing from "
                f"{VERSE_FONT}: {' '.join(missing)}"
            )


def _chunks(words: list[str]) -> list[list[str]]:
    n = VERSE_PAGE_MAX_WORDS
    return [words[i : i + n] for i in range(0, len(words), n)]


def verse_pages(timing: SceneTiming, corpus: QuranCorpus, verse: VerseRef) -> list[VersePage]:
    """Consecutive pages of at most `VERSE_PAGE_MAX_WORDS` corpus words per ayah."""
    _check_ayat(timing, verse)
    raw: list[tuple[int, int, list[tuple[str, int]], int]] = []
    for ayah in timing.ayat:
        texts = display_words(corpus.ayah(ayah.surah, ayah.ayah).uthmani)
        if timing.timing_source is TimingSource.SEGMENTS:
            timed = _segment_words(ayah, texts)
            n = VERSE_PAGE_MAX_WORDS
            for i in range(0, len(timed), n):
                page = timed[i : i + n]
                start = ayah.start_ms if i == 0 else page[0][1]
                raw.append((ayah.surah, ayah.ayah, page, start))
        else:
            chunks = _chunks(texts)
            spans = split_ms([float(len(" ".join(c))) for c in chunks], ayah.start_ms, ayah.end_ms)
            for chunk, (start, _) in zip(chunks, spans, strict=True):
                raw.append((ayah.surah, ayah.ayah, [(t, start) for t in chunk], start))
    pages: list[VersePage] = []
    for i, (surah, ayah_no, words, start) in enumerate(raw):
        end = raw[i + 1][3] if i + 1 < len(raw) else timing.end_ms
        pages.append(VersePage(surah, ayah_no, words, start, end))
    return pages


def verse_events(
    scene: Scene, timing: SceneTiming, corpus: QuranCorpus, language: Language
) -> list[AssEvent]:
    """One ``Verse`` event per page (zero-length pages dropped) and one ``VerseRef``."""
    verse = _verse_of(scene)
    x, y = VERSE_POS
    head = f"{{\\an5\\pos({x},{y})\\fad({VERSE_FADE_MS},0)}}"
    events = [
        AssEvent(p.start_ms, p.end_ms, "Verse", head + karaoke(p.words, p.start_ms))
        for p in verse_pages(timing, corpus, verse)
        if p.end_ms > p.start_ms
    ]
    events.append(
        AssEvent(timing.start_ms, timing.end_ms, "VerseRef", verse_ref_label(verse, language))
    )
    return events


def shown_text(events: Sequence[AssEvent]) -> str:
    """The verse text on screen: tags stripped, pages joined, whitespace collapsed."""
    return " ".join(" ".join(_TAGS.sub(" ", e.text) for e in events if e.style == "Verse").split())
