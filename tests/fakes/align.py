"""A fake forced aligner."""

from __future__ import annotations

from fakes.voice import TIMED_LEAD_MS, word_spans
from narratoflow.audio.io import Samples
from narratoflow.audio.words import TimedSpan
from narratoflow.models import Language


class FakeAligner:
    """An aligner that returns word spans shifted `shift_ms` later than a timed voice's."""

    def __init__(self, error: Exception | None = None, *, shift_ms: int = 10) -> None:
        self.error = error
        self.shift_ms = shift_ms
        self.calls: list[tuple[int, int, str, Language]] = []

    def align(
        self, samples: Samples, sample_rate: int, text: str, language: Language
    ) -> list[TimedSpan]:
        self.calls.append((len(samples), sample_rate, text, language))
        if self.error is not None:
            raise self.error
        return list(word_spans(text, lead_ms=TIMED_LEAD_MS + self.shift_ms))
