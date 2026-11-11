"""Word timings: map provider units onto the author's exact words, or estimate them.

Narration words are the whitespace tokens of the author's text. Provider units
(ElevenLabs characters, edge word boundaries, aligner words) are matched to them
through a shared normalization that ignores harakat, case, punctuation, symbols
and whitespace, so no word is ever rewritten.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from narratoflow.models import WordTiming
from narratoflow.text.tashkeel.verify import strip_diacritics


@dataclass(frozen=True)
class TimedSpan:
    """One provider unit: a character, a word boundary or an aligner word."""

    text: str
    start_ms: int
    end_ms: int


def tokens(text: str) -> list[str]:
    """The narration words of `text`: its whitespace tokens."""
    return text.split()


def _dropped(ch: str) -> bool:
    return ch.isspace() or unicodedata.category(ch)[0] in "PS"


def norm_for_timing(s: str) -> str:
    """NFC, strip diacritics, casefold, then drop whitespace and P*/S* characters."""
    folded = strip_diacritics(unicodedata.normalize("NFC", s)).casefold()
    return "".join(ch for ch in folded if not _dropped(ch))


def _ranges(pieces: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    parts: list[str] = []
    ranges: list[tuple[int, int]] = []
    pos = 0
    for piece in pieces:
        norm = norm_for_timing(piece)
        parts.append(norm)
        ranges.append((pos, pos + len(norm)))
        pos += len(norm)
    return "".join(parts), ranges


def timings_from_spans(words: list[str], spans: Sequence[TimedSpan]) -> list[WordTiming] | None:
    """Time each of `words` from provider `spans`; None when their texts disagree."""
    w_text, w_ranges = _ranges(words)
    s_text, s_ranges = _ranges([s.text for s in spans])
    if w_text != s_text:
        return None
    timed = [(r, s) for r, s in zip(s_ranges, spans, strict=True) if r[0] < r[1]]
    out: list[WordTiming] = []
    prev_start = 0
    prev_end = 0
    j = 0  # first non-empty span that may still overlap the current word
    for word, (a, b) in zip(words, w_ranges, strict=True):
        if a == b:
            start = end = prev_end
        else:
            while timed[j][0][1] <= a:
                j += 1
            k = j
            while k + 1 < len(timed) and timed[k + 1][0][0] < b:
                k += 1
            start, end = timed[j][1].start_ms, timed[k][1].end_ms
        start = max(start, prev_start)
        end = max(end, start)
        out.append(WordTiming(text=word, start_ms=start, end_ms=end))
        prev_start, prev_end = start, end
    return out


def split_ms(weights: Sequence[float], start_ms: int, end_ms: int) -> list[tuple[int, int]]:
    """Split [start_ms, end_ms] in proportion to `weights`, contiguous, ending at end_ms.

    Boundaries come from rounding the cumulative weight, so rounding error never
    accumulates. All-zero weights split evenly.
    """
    if end_ms < start_ms:
        raise ValueError(f"end_ms {end_ms} is before start_ms {start_ms}")
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("weights must be finite and >= 0")
    if not weights:
        return []
    total = math.fsum(weights)
    if total == 0:
        weights = [1.0] * len(weights)
        total = float(len(weights))
    span = end_ms - start_ms
    out: list[tuple[int, int]] = []
    cum = 0.0
    prev = start_ms
    for i, w in enumerate(weights):
        cum += w
        if i == len(weights) - 1:
            edge = end_ms
        else:
            edge = start_ms + min(span, math.floor(span * cum / total + 0.5))
        edge = max(edge, prev)
        out.append((prev, edge))
        prev = edge
    return out


def estimate_timings(words: list[str], onset_ms: int, offset_ms: int) -> list[WordTiming]:
    """Spread `words` over [onset_ms, offset_ms] by normalized length (at least 1)."""
    weights = [float(max(1, len(norm_for_timing(w)))) for w in words]
    return [
        WordTiming(text=w, start_ms=s, end_ms=e)
        for w, (s, e) in zip(words, split_ms(weights, onset_ms, offset_ms), strict=True)
    ]


def shift(words: list[WordTiming], delta_ms: int, *, limit_ms: int) -> list[WordTiming]:
    """Move `words` by `delta_ms`, clamping each time to [0, limit_ms]."""

    def clamp(v: int) -> int:
        return min(max(v + delta_ms, 0), limit_ms)

    return [
        WordTiming(text=w.text, start_ms=clamp(w.start_ms), end_ms=clamp(w.end_ms)) for w in words
    ]
