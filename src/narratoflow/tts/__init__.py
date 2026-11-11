"""Speech synthesis: the provider protocol and its request/response types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from narratoflow.audio.words import TimedSpan
from narratoflow.models import Language


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    voice_id: str
    language: Language


@dataclass(frozen=True)
class Speech:
    """Synthesized audio. Empty `spans` means no timings; `billed_chars` None means no count."""

    audio: bytes
    spans: tuple[TimedSpan, ...]
    provider: str
    model: str
    billed_chars: int | None


class SpeechProvider(Protocol):
    name: str
    model: str

    def synthesize(self, req: SpeechRequest) -> Speech: ...


@dataclass(frozen=True)
class VoiceChoice:
    provider: SpeechProvider
    voice_id: str


__all__ = ["Speech", "SpeechProvider", "SpeechRequest", "VoiceChoice"]
