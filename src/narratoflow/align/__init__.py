"""Forced alignment for engines without native word timings."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from narratoflow.audio.io import Samples
from narratoflow.audio.words import TimedSpan
from narratoflow.models import Language


@runtime_checkable
class Aligner(Protocol):
    """Times the words of `text` in `samples` (mono float32 at `sample_rate`).

    Returns one span per word of the text the aligner saw; `timings_from_spans`
    maps them back onto the author's tokens. Raises `AlignerUnavailableError` when
    the aligner is not installed and `ProviderError` for any other failure.
    """

    def align(
        self, samples: Samples, sample_rate: int, text: str, language: Language
    ) -> list[TimedSpan]: ...


__all__ = ["Aligner"]
