"""Corpus-based detection of quoted Qur'anic verses in free text.

Quotes are found by normalized 4-gram lookup into the corpus, then extended
greedily word by word (crossing into the next ayah of the same surah). The
longest non-overlapping runs win; each becomes a `VerseHit` whose `VerseRef`
carries the verified Uthmani text of every ayah the run touches.
"""

import re
from dataclasses import dataclass
from typing import ClassVar
from weakref import WeakKeyDictionary

from narratoflow.models import VerseRef
from narratoflow.quran.corpus import QuranCorpus
from narratoflow.quran.normalize import is_punctuation, match_normalize

_GRAM = 4
# Arabic, Arabic Supplement, Arabic Extended-A and the presentation-form blocks.
_ARABIC = re.compile("[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufefc]")
_TOKEN = re.compile(r"\S+")


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
    last: bool  # final word of its ayah


@dataclass(frozen=True)
class _Run:
    first: int  # index of the first text word
    length: int
    surah: int
    slot: int  # index of the first matched slot in the surah stream


class VerseIndex:
    """Per-surah normalized word streams plus a 4-gram index into them.

    Build it with `VerseIndex.for_corpus`, which caches one index per corpus
    instance in a `WeakKeyDictionary`. The index holds no reference back to the
    corpus, so the cache entry dies with the corpus.
    """

    _cache: ClassVar[WeakKeyDictionary[QuranCorpus, "VerseIndex"]] = WeakKeyDictionary()

    def __init__(self, corpus: QuranCorpus) -> None:
        self.streams: dict[int, list[_Slot]] = {}
        self.grams: dict[tuple[str, ...], list[tuple[int, int]]] = {}
        for ayah in sorted(corpus, key=lambda a: (a.surah, a.ayah)):
            stream = self.streams.setdefault(ayah.surah, [])
            words = ayah.norm_words
            stream.extend(_Slot(w, ayah.ayah, n, n == len(words) - 1) for n, w in enumerate(words))
        for surah, stream in self.streams.items():
            for p in range(len(stream) - _GRAM + 1):
                key = tuple(s.word for s in stream[p : p + _GRAM])
                self.grams.setdefault(key, []).append((surah, p))

    @classmethod
    def for_corpus(cls, corpus: QuranCorpus) -> "VerseIndex":
        index = cls._cache.get(corpus)
        if index is None:
            index = cls._cache[corpus] = cls(corpus)
        return index


def _tokenize(text: str) -> list[_Word]:
    """Normalized words with spans; punctuation splits tokens and is trimmed from spans."""
    words: list[_Word] = []

    def emit(a: int, b: int) -> None:
        for w in match_normalize(text[a:b]).split():
            words.append(_Word(w, a, b))

    for m in _TOKEN.finditer(text):
        piece: int | None = None
        for offset in range(m.start(), m.end()):
            if is_punctuation(text[offset]):
                if piece is not None:
                    emit(piece, offset)
                    piece = None
            elif piece is None:
                piece = offset
        if piece is not None:
            emit(piece, m.end())
    return words


def _runs(words: list[_Word], index: VerseIndex, min_words: int) -> list[_Run]:
    norms = [w.norm for w in words]
    runs: list[_Run] = []
    for i in range(len(norms) - _GRAM + 1):
        for surah, p in index.grams.get(tuple(norms[i : i + _GRAM]), ()):
            stream = index.streams[surah]
            if i > 0 and p > 0 and norms[i - 1] == stream[p - 1].word:
                continue  # contained in the run that starts one word earlier
            k = _GRAM
            while i + k < len(norms) and p + k < len(stream) and norms[i + k] == stream[p + k].word:
                k += 1
            if k >= min_words:
                runs.append(_Run(i, k, surah, p))
    return runs


def detect_verses(
    text: str, corpus: QuranCorpus, *, min_words: int = 4, certain_words: int = 6
) -> list[VerseHit]:
    """Find quoted verses in `text`, sorted by start offset."""
    if min_words < _GRAM:
        raise ValueError(f"min_words must be >= {_GRAM} (the index n-gram size)")
    if not _ARABIC.search(text):
        return []
    index = VerseIndex.for_corpus(corpus)
    words = _tokenize(text)

    def rank(r: _Run) -> tuple[int, int, int, int, int]:
        slot = index.streams[r.surah][r.slot]
        return (-r.length, r.surah, slot.ayah, slot.pos, r.first)

    taken = [False] * len(words)
    hits: list[VerseHit] = []
    for run in sorted(_runs(words, index, min_words), key=rank):
        covered = range(run.first, run.first + run.length)
        if any(taken[i] for i in covered):
            continue
        for i in covered:
            taken[i] = True
        stream = index.streams[run.surah]
        head, tail = stream[run.slot], stream[run.slot + run.length - 1]
        full = head.pos == 0 and tail.last
        ref = VerseRef(
            surah=run.surah,
            ayah_start=head.ayah,
            ayah_end=tail.ayah,
            certain=full or run.length >= certain_words,
            uthmani_text=corpus.text_range(run.surah, head.ayah, tail.ayah),
        )
        hits.append(VerseHit(words[run.first].start, words[covered[-1]].end, ref))
    return sorted(hits, key=lambda h: h.start)
