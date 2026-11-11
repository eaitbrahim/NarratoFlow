"""edge-tts engine (extra: tts-edge; free, no key). Requests WordBoundary events explicitly.

API verified 2026-11-11 against the edge-tts 7.3.1 source (``communicate.py``):
- ``Communicate(text, voice, *, rate="+0%", volume="+0%", pitch="+0Hz",
  boundary="SentenceBoundary", ...)``; ``boundary="WordBoundary"`` asks for word events;
- ``stream()`` is an async generator of ``{"type": "audio", "data": bytes}`` (mp3) and
  ``{"type": "WordBoundary", "offset", "duration", "text"}`` with times in 100 ns units.
``stream_sync()`` is not used: in 7.3.1 it runs ``stream()`` in a worker thread whose
exception is never read, so a failed request blocks the caller forever. The adapter runs
``stream()`` with ``asyncio.run`` in a worker thread (as the library's ``save_sync``
does), so every library error reaches the caller and becomes a ``ProviderError``.
The service is Microsoft's; tests only ever use a fake module.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from narratoflow.audio.words import TimedSpan
from narratoflow.errors import ExtraMissingError, ProviderError
from narratoflow.models import Language
from narratoflow.tts import Speech, SpeechRequest

DEFAULT_EDGE_VOICES: dict[Language, str] = {
    Language.ARABIC: "ar-SA-HamedNeural",
    Language.ENGLISH: "en-US-AndrewNeural",
}
EXTRA_HINT = "install the 'tts-edge' extra: uv sync --extra tts-edge"
_TICKS_PER_MS = 10_000  # edge-tts times are in 100 ns units


class EdgeProvider:
    name: str = "edge"
    model: str = "edge-tts"

    def __init__(self, rate: str = "+0%", pitch: str = "+0Hz") -> None:
        self.rate = rate
        self.pitch = pitch

    def synthesize(self, req: SpeechRequest) -> Speech:
        try:
            import edge_tts
        except ImportError as err:
            raise ExtraMissingError(EXTRA_HINT) from err

        async def collect() -> list[Any]:
            communicate = edge_tts.Communicate(
                req.text, req.voice_id, rate=self.rate, pitch=self.pitch, boundary="WordBoundary"
            )
            return [chunk async for chunk in communicate.stream()]

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                chunks = pool.submit(asyncio.run, collect()).result()
        except Exception as err:
            raise ProviderError(f"edge-tts failed: {type(err).__name__}") from err
        audio, spans = _parse(chunks)
        return Speech(
            audio=audio, spans=spans, provider=self.name, model=self.model, billed_chars=None
        )


def _ticks(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProviderError("edge-tts returned a malformed word boundary")
    return value


def _parse(chunks: list[Any]) -> tuple[bytes, tuple[TimedSpan, ...]]:
    audio = bytearray()
    spans: list[TimedSpan] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ProviderError("edge-tts returned a malformed chunk")
        kind = chunk.get("type")
        if kind == "audio":
            data = chunk.get("data")
            if not isinstance(data, bytes):
                raise ProviderError("edge-tts returned a malformed audio chunk")
            audio += data
        elif kind == "WordBoundary":
            offset, duration = _ticks(chunk.get("offset")), _ticks(chunk.get("duration"))
            text = chunk.get("text")
            if not isinstance(text, str):
                raise ProviderError("edge-tts returned a malformed word boundary")
            spans.append(
                TimedSpan(
                    text=text,
                    start_ms=offset // _TICKS_PER_MS,
                    end_ms=(offset + duration) // _TICKS_PER_MS,
                )
            )
    if not audio:
        raise ProviderError("edge-tts returned no audio")
    return bytes(audio), tuple(spans)
