"""EdgeProvider against a fake `edge_tts` module (never the Microsoft service).

The fake copies the 7.3.1 API: `Communicate(text, voice, *, rate, pitch, boundary)`
and an async `stream()` yielding `{"type": "audio", "data"}` and
`{"type": "WordBoundary", "offset", "duration", "text"}` chunks (100 ns units).
"""

import sys
import types
from collections.abc import AsyncIterator
from typing import Any

import pytest

from narratoflow.audio.words import TimedSpan
from narratoflow.errors import ChannelConfigError, ExtraMissingError, ProviderError
from narratoflow.models import Language
from narratoflow.tts import SpeechRequest
from narratoflow.tts.edge import DEFAULT_EDGE_VOICES, EdgeProvider

REQ = SpeechRequest(text="hello world", voice_id="en-US-AndrewNeural", language=Language.ENGLISH)


def fake_edge(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[dict[str, Any]],
    *,
    fail: Exception | None = None,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    class Communicate:
        def __init__(self, text: str, voice: str, **kwargs: Any) -> None:
            calls.append({"text": text, "voice": voice, **kwargs})

        async def stream(self) -> AsyncIterator[dict[str, Any]]:
            for chunk in chunks:
                yield chunk
            if fail is not None:
                raise fail

    module = types.ModuleType("edge_tts")
    module.Communicate = Communicate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "edge_tts", module)
    return calls


def test_edge_word_boundaries_to_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_edge(
        monkeypatch,
        [
            {"type": "audio", "data": b"ID3a"},
            {"type": "WordBoundary", "offset": 1_000_000, "duration": 4_000_000, "text": "hello"},
            {"type": "audio", "data": b"bc"},
            {"type": "WordBoundary", "offset": 5_500_000, "duration": 3_000_000, "text": "world"},
        ],
    )
    speech = EdgeProvider(rate="+10%", pitch="-2Hz").synthesize(REQ)
    assert speech.audio == b"ID3abc"
    assert speech.spans == (TimedSpan("hello", 100, 500), TimedSpan("world", 550, 850))
    assert (speech.provider, speech.model, speech.billed_chars) == ("edge", "edge-tts", None)
    assert calls == [
        {
            "text": "hello world",
            "voice": "en-US-AndrewNeural",
            "rate": "+10%",
            "pitch": "-2Hz",
            "boundary": "WordBoundary",
        }
    ]


def test_edge_defaults_and_voice_table() -> None:
    edge = EdgeProvider()
    assert (edge.name, edge.model) == ("edge", "edge-tts")
    assert DEFAULT_EDGE_VOICES == {
        Language.ARABIC: "ar-SA-HamedNeural",
        Language.ENGLISH: "en-US-AndrewNeural",
    }


def test_edge_missing_extra_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "edge_tts", None)
    with pytest.raises(ExtraMissingError) as info:
        EdgeProvider().synthesize(REQ)
    assert isinstance(info.value, ChannelConfigError)
    assert str(info.value) == "install the 'tts-edge' extra: uv sync --extra tts-edge"


def test_edge_library_error_maps_to_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class NoAudioReceived(Exception):
        pass

    fake_edge(
        monkeypatch,
        [{"type": "audio", "data": b"x"}],
        fail=NoAudioReceived("secret body text"),
    )
    with pytest.raises(ProviderError) as info:
        EdgeProvider().synthesize(REQ)
    assert str(info.value) == "edge-tts failed: NoAudioReceived"


def test_edge_no_audio_is_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_edge(monkeypatch, [{"type": "SentenceBoundary", "offset": 0, "duration": 1}])
    with pytest.raises(ProviderError, match="no audio"):
        EdgeProvider().synthesize(REQ)


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "WordBoundary", "offset": -5, "duration": 1, "text": "a"},
        {"type": "WordBoundary", "offset": True, "duration": 1, "text": "a"},
        {"type": "WordBoundary", "offset": 0, "duration": 1, "text": None},
        {"type": "audio", "data": "not bytes"},
        "not a dict",
    ],
)
def test_edge_malformed_chunk_is_provider_error(monkeypatch: pytest.MonkeyPatch, bad: Any) -> None:
    fake_edge(monkeypatch, [{"type": "audio", "data": b"x"}, bad])
    with pytest.raises(ProviderError, match="edge-tts"):
        EdgeProvider().synthesize(REQ)
