"""Verbatim scene segmentation.

Detected Qur'anic verse spans are cut out first and become protected VERSE scenes.
Only the stretches between them are sent to the LLM, one call per stretch, and the
LLM may only choose where to cut: every returned span is located in the original
stretch (whitespace-insensitively, in order), and each scene's `source_text` is the
original slice, never the LLM's copy. A span that cannot be located, or spans that do
not cover the whole stretch, are retried once with the first differing offset; a
second failure raises `SegmentationError`.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from narratoflow.errors import SegmentationError
from narratoflow.llm.base import LLMClient
from narratoflow.models import Narration, Scene, SceneKind, VerseRef
from narratoflow.planning.prompts import SEGMENT_SYSTEM
from narratoflow.quran.detect import VerseHit

UNMATCHED_BRACKETS_FLAG = "unmatched_verse_brackets"
# Ornate parentheses used around Qur'anic quotes.
_VERSE_BRACKETS = frozenset("\ufd3e\ufd3f")  # either direction flags a narration
_WS = re.compile(r"\s+")
_OPENING_BRACKET = "\ufd3f"
_CLOSING_BRACKET = "\ufd3e"
# Clause and sentence punctuation a scene may end on: . ! ? , ; : and their Arabic
# forms (question mark, comma, semicolon) plus the ellipsis.
_CUT_PUNCTUATION = frozenset(".!?,;:\u061f\u060c\u061b\u2026")
_CLOSING_CATEGORIES = ("Pe", "Pf")


class SegmentSpan(BaseModel):
    text: str
    title: str


class SegmentOut(BaseModel):
    scenes: list[SegmentSpan]


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


@dataclass
class _Piece:
    start: int
    end: int
    title: str
    verse: VerseRef | None = None


def _ends_clause(ch: str) -> bool:
    return ch in _CUT_PUNCTUATION or ch in "\"'" or unicodedata.category(ch) in _CLOSING_CATEGORIES


def _is_cut(stretch: str, at: int) -> bool:
    """A scene may start at `at`: after whitespace, or right after closing punctuation.

    An apostrophe is a clause end only after clause punctuation (`stop.'`); between
    letters it is a contraction (`don't`), and a closing quote before whitespace is
    already a whitespace cut.
    """
    before = stretch[at - 1]
    if before.isspace():
        return True
    if before == "'" and not (at >= 2 and _ends_clause(stretch[at - 2]) and stretch[at - 2] != "'"):
        return False
    return _ends_clause(before) and not _ends_clause(stretch[at])


def _locate(stretch: str, spans: Sequence[SegmentSpan]) -> list[tuple[int, str]] | int:
    """Start offsets and titles of the spans in `stretch`, or the first differing offset."""
    pos = [i for i, ch in enumerate(stretch) if not ch.isspace()]
    k = 0
    starts: list[tuple[int, str]] = []
    for span in spans:
        chars = "".join(span.text.split())
        if not chars:
            continue
        for j, ch in enumerate(chars):
            if k + j >= len(pos):
                return len(stretch)
            if stretch[pos[k + j]] != ch:
                return pos[k + j]
        if starts and not _is_cut(stretch, pos[k]):
            return pos[k]
        starts.append((pos[k], span.title))
        k += len(chars)
    if k < len(pos):
        return pos[k]
    return starts


def _user_prompt(stretch: str) -> str:
    return f"Split this text into scenes. Return every span verbatim.\n\nTEXT:\n{stretch}"


def _retry_note(stretch: str, offset: int) -> str:
    return (
        "\n\nYour previous spans did not reproduce the text exactly. They first differ "
        f"from the text at character offset {offset} (counting from 0), at: "
        f"{stretch[offset : offset + 30]!r}. Return the original characters unchanged; "
        "only choose where to cut."
    )


def _ask(llm: LLMClient, stretch: str, user: str) -> list[tuple[int, str]] | int:
    out = llm.complete_structured(system=SEGMENT_SYSTEM, user=user, output=SegmentOut)
    return _locate(stretch, out.value.scenes)


def _segment_stretch(stretch: str, base: int, llm: LLMClient) -> list[_Piece]:
    user = _user_prompt(stretch)
    located = _ask(llm, stretch, user)
    if isinstance(located, int):
        located = _ask(llm, stretch, user + _retry_note(stretch, located))
        if isinstance(located, int):
            raise SegmentationError(
                f"segmentation changed the script text at offset {located} of the stretch "
                f"starting at text offset {base} (text offset {base + located})"
            )
    pieces: list[_Piece] = []
    for n, (_, title) in enumerate(located):
        start = 0 if n == 0 else located[n][0]
        end = located[n + 1][0] if n + 1 < len(located) else len(stretch)
        pieces.append(_Piece(base + start, base + end, title))
    return pieces


def _has_text(stretch: str) -> bool:
    return any(ch.isalnum() for ch in stretch)


def _verse_title(ref: VerseRef) -> str:
    if ref.ayah_end == ref.ayah_start:
        return f"{ref.surah}:{ref.ayah_start}"
    return f"{ref.surah}:{ref.ayah_start}-{ref.ayah_end}"


def _flush_brackets(text: str, verses: Sequence[VerseHit]) -> set[int]:
    """Offsets of an opening bracket right before a verse and a closing one right after."""
    flush: set[int] = set()
    for hit in verses:
        if hit.start > 0 and text[hit.start - 1] == _OPENING_BRACKET:
            flush.add(hit.start - 1)
        if hit.end < len(text) and text[hit.end] == _CLOSING_BRACKET:
            flush.add(hit.end)
    return flush


def _merge_letterless(text: str, pieces: list[_Piece]) -> list[_Piece]:
    """Fold each narration piece without letters or digits into the previous piece,
    or into the next one when it comes first. Pieces are contiguous, so the join of
    their slices is unchanged."""
    merged: list[_Piece] = []
    lead: int | None = None
    for piece in pieces:
        if piece.verse is None and not _has_text(text[piece.start : piece.end]):
            if merged:
                merged[-1].end = piece.end
            elif lead is None:
                lead = piece.start
            continue
        if lead is not None:
            piece.start, lead = lead, None
        merged.append(piece)
    return merged


def segment_script(text: str, verses: Sequence[VerseHit], llm: LLMClient) -> list[Scene]:
    """Split `text` into scenes without changing a character.

    `verses` must be sorted and non-overlapping. A stretch between verses that holds
    no letters or digits (brackets, punctuation, whitespace) is not sent to the LLM; it
    joins the previous scene, or the next one when it comes first, so that the scenes'
    `source_text` values concatenate back to `text`. The same holds for a letterless
    span the LLM cuts off: it is merged into its neighbour (a verse scene included), so
    no scene without letters or digits reaches idea assignment.
    """
    pieces: list[_Piece] = []
    cursor = 0
    bounds = [(h.start, h.end, h) for h in verses] + [(len(text), len(text), None)]
    for start, end, hit in bounds:
        if start < cursor or end < start or end > len(text):
            raise SegmentationError(f"verse hits overlap or are out of range at offset {start}")
        stretch = text[cursor:start]
        if _has_text(stretch):
            pieces.extend(_segment_stretch(stretch, cursor, llm))
        elif stretch and pieces:
            pieces[-1].end = start  # a filler joins the previous scene
        if hit is not None:
            # with no scene before it, a leading filler joins this one
            lead = start if pieces else cursor
            pieces.append(_Piece(lead, end, _verse_title(hit.ref), hit.ref))
        cursor = end
    if not pieces:
        raise SegmentationError("the script has no text to segment")
    pieces = _merge_letterless(text, pieces)

    flush = _flush_brackets(text, verses)
    scenes: list[Scene] = []
    for n, piece in enumerate(pieces, start=1):
        source = text[piece.start : piece.end]
        if piece.verse is not None:
            scenes.append(
                Scene(
                    id=f"s{n:03d}",
                    narration=Narration(source_text=source, protected=True),
                    kind=SceneKind.VERSE,
                    verse=piece.verse,
                    title=piece.title,
                )
            )
            continue
        unmatched = any(
            text[i] in _VERSE_BRACKETS and i not in flush for i in range(piece.start, piece.end)
        )
        scenes.append(
            Scene(
                id=f"s{n:03d}",
                narration=Narration(source_text=source),
                title=piece.title,
                flags=[UNMATCHED_BRACKETS_FLAG] if unmatched else [],
            )
        )
    return scenes
