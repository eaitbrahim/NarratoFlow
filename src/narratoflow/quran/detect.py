"""Corpus-based detection of quoted Qur'anic verses in free text.

Quotes are found by normalized 4-gram lookup into the corpus, then extended
greedily word by word, crossing into the next ayah of the same surah only when
the ayat are consecutive. A run that pokes fewer than four words into an ayah at
either edge is trimmed back to the ayah boundary, so the author's own words next
to a quote are not swallowed. An edge is kept when the author marked it with an
opening or closing quote or bracket, or when it is a whole ayah of two or more
words. The longest non-overlapping runs win, and each becomes a `VerseHit` whose
`VerseRef` carries the verified Uthmani text of every ayah the run touches.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import ClassVar
from weakref import WeakKeyDictionary

from narratoflow.models import VerseRef
from narratoflow.quran.corpus import QuranCorpus
from narratoflow.quran.normalize import is_punctuation, match_normalize

_GRAM = 4
_MIN_WHOLE = 2  # a whole edge ayah this long survives trimming
# Arabic, Arabic Supplement, Arabic Extended-A and the presentation-form blocks.
_ARABIC = re.compile("[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufefc]")
_TOKEN = re.compile(r"\S+")
_OPEN_CATEGORIES = ("Ps", "Pi")
_CLOSE_CATEGORIES = ("Pe", "Pf")


@dataclass(frozen=True)
class VerseHit:
    start: int  # offset of the first quoted word in the original text
    end: int  # offset just past the last quoted word
    ref: VerseRef


@dataclass(frozen=True)
class _Word:
    norm: str
    start: int
    end: int


@dataclass(frozen=True)
class _Slot:
    word: str
    ayah: int
    pos: int  # word position inside its ayah
    size: int  # word count of its ayah


@dataclass(frozen=True)
class _Segment:
    """A run of consecutive ayat of one surah, as one word stream."""

    surah: int
    slots: tuple[_Slot, ...]


@dataclass(frozen=True)
class _Run:
    first: int  # index of the first text word
    length: int
    seg: int  # segment index
    slot: int  # index of the first matched slot in the segment


class VerseIndex:
    """Normalized word streams of consecutive ayat plus a 4-gram index into them.

    Build it with `VerseIndex.for_corpus`, which caches one index per corpus
    instance in a `WeakKeyDictionary`. The index holds no reference back to the
    corpus, so the cache entry dies with the corpus.
    """

    _cache: ClassVar[WeakKeyDictionary[QuranCorpus, "VerseIndex"]] = WeakKeyDictionary()

    def __init__(self, corpus: QuranCorpus) -> None:
        groups: list[tuple[int, list[_Slot]]] = []
        prev: tuple[int, int] | None = None
        for ayah in sorted(corpus, key=lambda a: (a.surah, a.ayah)):
            if prev != (ayah.surah, ayah.ayah - 1):
                groups.append((ayah.surah, []))
            size = len(ayah.norm_words)
            groups[-1][1].extend(
                _Slot(w, ayah.ayah, n, size) for n, w in enumerate(ayah.norm_words)
            )
            prev = (ayah.surah, ayah.ayah)
        self.segments = [_Segment(surah, tuple(slots)) for surah, slots in groups]
        self.grams: dict[tuple[str, ...], list[tuple[int, int]]] = {}
        for seg, segment in enumerate(self.segments):
            slots = segment.slots
            for p in range(len(slots) - _GRAM + 1):
                key = tuple(s.word for s in slots[p : p + _GRAM])
                self.grams.setdefault(key, []).append((seg, p))

    @classmethod
    def for_corpus(cls, corpus: QuranCorpus) -> "VerseIndex":
        index = cls._cache.get(corpus)
        if index is None:
            index = cls._cache[corpus] = cls(corpus)
        return index


# End-of-ayah sign, rub el hizb and sajdah sign: Qur'anic markers, not word parts.
_MARKERS = frozenset("\u06dd\u06de\u06e9")


def _is_separator(ch: str) -> bool:
    # Punctuation, digits (including Arabic-Indic ayah numbers) and Qur'anic markers
    # never belong to a word.
    return is_punctuation(ch) or ch in _MARKERS or unicodedata.category(ch) == "Nd"


def _tokenize(text: str) -> list[_Word]:
    """Normalized words with spans; separators split tokens and are trimmed from spans."""
    words: list[_Word] = []

    def emit(a: int, b: int) -> None:
        for w in match_normalize(text[a:b]).split():
            words.append(_Word(w, a, b))

    for m in _TOKEN.finditer(text):
        piece: int | None = None
        for offset in range(m.start(), m.end()):
            if _is_separator(text[offset]):
                if piece is not None:
                    emit(piece, offset)
                    piece = None
            elif piece is None:
                piece = offset
        if piece is not None:
            emit(piece, m.end())
    return words


def _neighbour(text: str, at: int, step: int) -> str:
    """The first non-space character from `at` in direction `step`, or ""."""
    while 0 <= at < len(text) and text[at].isspace():
        at += step
    return text[at] if 0 <= at < len(text) else ""


def _opens(text: str, start: int) -> bool:
    ch = _neighbour(text, start - 1, -1)
    return ch == '"' or (ch != "" and unicodedata.category(ch) in _OPEN_CATEGORIES)


def _closes(text: str, end: int) -> bool:
    ch = _neighbour(text, end, 1)
    return ch == '"' or (ch != "" and unicodedata.category(ch) in _CLOSE_CATEGORIES)


def _whole(complete: bool, size: int) -> bool:
    # Single-word ayat (e.g. muqatta'at) are too likely to be coincidences to keep.
    return complete and size >= _MIN_WHOLE


def _trim(text: str, words: list[_Word], slots: tuple[_Slot, ...], run: _Run) -> _Run:
    """Drop an undelimited edge piece of fewer than `_GRAM` words in another ayah.

    A piece that is a complete ayah of at least `_MIN_WHOLE` words is kept.
    """
    first, p, k = run.first, run.slot, run.length
    head = slots[p]
    if head.ayah != slots[p + k - 1].ayah and not _opens(text, words[first].start):
        piece = head.size - head.pos
        if piece < _GRAM and not _whole(head.pos == 0, head.size):
            first, p, k = first + piece, p + piece, k - piece
    tail = slots[p + k - 1]
    if tail.ayah != slots[p].ayah and not _closes(text, words[first + k - 1].end):
        piece = tail.pos + 1
        if piece < _GRAM and not _whole(tail.pos == tail.size - 1, tail.size):
            k -= piece
    return _Run(first, k, run.seg, p)


def _runs(text: str, words: list[_Word], index: VerseIndex, min_words: int) -> list[_Run]:
    norms = [w.norm for w in words]
    runs: list[_Run] = []
    for i in range(len(norms) - _GRAM + 1):
        for seg, p in index.grams.get(tuple(norms[i : i + _GRAM]), ()):
            slots = index.segments[seg].slots
            if i > 0 and p > 0 and norms[i - 1] == slots[p - 1].word:
                continue  # contained in the run that starts one word earlier
            k = _GRAM
            while i + k < len(norms) and p + k < len(slots) and norms[i + k] == slots[p + k].word:
                k += 1
            run = _trim(text, words, slots, _Run(i, k, seg, p))
            if run.length >= min_words:
                runs.append(run)
    return runs


def detect_verses(
    text: str, corpus: QuranCorpus, *, min_words: int = 4, certain_words: int = 6
) -> list[VerseHit]:
    """Find quoted verses in `text`, sorted by start offset.

    A run is certain when it covers whole ayat exactly or is at least
    `certain_words` long, unless the same words also match at another corpus
    location: then the best location (full-ayah first, then earliest) is kept
    and the hit is uncertain.
    """
    if min_words < _GRAM:
        raise ValueError(f"min_words must be >= {_GRAM} (the index n-gram size)")
    if not _ARABIC.search(text):
        return []
    index = VerseIndex.for_corpus(corpus)
    words = _tokenize(text)

    def ends(r: _Run) -> tuple[_Slot, _Slot]:
        slots = index.segments[r.seg].slots
        return slots[r.slot], slots[r.slot + r.length - 1]

    def full(r: _Run) -> bool:
        head, tail = ends(r)
        return head.pos == 0 and tail.pos == tail.size - 1

    def rank(r: _Run) -> tuple[int, bool, int, int, int, int]:
        head, _ = ends(r)
        return (-r.length, not full(r), index.segments[r.seg].surah, head.ayah, head.pos, r.first)

    runs = _runs(text, words, index, min_words)
    # Runs over the same text words match the same normalized sequence by construction.
    locations: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for r in runs:
        locations.setdefault((r.first, r.length), set()).add((r.seg, r.slot))

    taken = [False] * len(words)
    hits: list[VerseHit] = []
    for run in sorted(runs, key=rank):
        covered = range(run.first, run.first + run.length)
        if any(taken[i] for i in covered):
            continue
        for i in covered:
            taken[i] = True
        surah = index.segments[run.seg].surah
        head, tail = ends(run)
        ambiguous = len(locations[(run.first, run.length)]) > 1
        ref = VerseRef(
            surah=surah,
            ayah_start=head.ayah,
            ayah_end=tail.ayah,
            certain=not ambiguous and (full(run) or run.length >= certain_words),
            uthmani_text=corpus.text_range(surah, head.ayah, tail.ayah),
        )
        hits.append(VerseHit(words[run.first].start, words[covered[-1]].end, ref))
    return sorted(hits, key=lambda h: h.start)
