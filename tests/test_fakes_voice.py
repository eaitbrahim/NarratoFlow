"""The shared FakeVoice: its plain and timed profiles match the fakes it replaced."""

from __future__ import annotations

import pytest
from fakes.voice import FakeVoice
from fixtures.audio import speechlike, to_wav_bytes

from narratoflow.errors import ProviderError
from narratoflow.models import Language
from narratoflow.tts import SpeechRequest


def _req(text: str = "a b c") -> SpeechRequest:
    return SpeechRequest(text, "v", Language.ENGLISH)


def test_plain_profile_matches_old_parts_fake() -> None:
    speech = FakeVoice().synthesize(_req())

    assert speech.audio == to_wav_bytes(speechlike(3, lead_ms=150))
    assert speech.spans == ()
    assert speech.billed_chars == 5
    assert (speech.provider, speech.model) == ("paid", "m1")


def test_timed_profile_matches_old_narrate_fake() -> None:
    speech = FakeVoice(spans=True, lead_ms=200).synthesize(_req())

    assert speech.audio == to_wav_bytes(speechlike(3, 300, 80, lead_ms=200))
    assert [(s.text, s.start_ms, s.end_ms) for s in speech.spans] == [
        ("a", 200, 500),
        ("b", 580, 880),
        ("c", 960, 1260),
    ]


def test_fail_on_empty_always_fails() -> None:
    voice = FakeVoice(fail_on="")

    with pytest.raises(ProviderError, match="HTTP 500"):
        voice.synthesize(_req("anything"))
    assert [r.text for r in voice.requests] == ["anything"]


def test_fail_on_substring() -> None:
    voice = FakeVoice(fail_on="zeta")

    voice.synthesize(_req("alpha"))
    with pytest.raises(ProviderError):
        voice.synthesize(_req("zeta now"))
    assert [r.text for r in voice.requests] == ["alpha", "zeta now"]
