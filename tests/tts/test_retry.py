import logging

import pytest

from narratoflow.errors import BilledProviderError, ProviderError, TransientProviderError
from narratoflow.models import Language
from narratoflow.tts import Speech, SpeechRequest
from narratoflow.tts.retry import RETRY_ATTEMPTS, RETRY_DELAYS_S, RetryingSpeechProvider

REQ = SpeechRequest(text="secret script text", voice_id="v", language=Language.ENGLISH)
SPEECH = Speech(audio=b"a", spans=(), provider="fake", model="m1", billed_chars=None)


class Scripted:
    name = "fake"
    model = "m1"

    def __init__(self, outcomes: list[Exception | Speech]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def synthesize(self, req: SpeechRequest) -> Speech:
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def wrap(inner: Scripted) -> tuple[RetryingSpeechProvider, list[float]]:
    sleeps: list[float] = []
    return RetryingSpeechProvider(inner, sleep=sleeps.append), sleeps


def test_constants() -> None:
    assert RETRY_ATTEMPTS == 3
    assert RETRY_DELAYS_S == (1.0, 4.0)


def test_succeeds_after_transient() -> None:
    inner = Scripted([TransientProviderError("down"), SPEECH])
    provider, sleeps = wrap(inner)
    assert provider.synthesize(REQ) is SPEECH
    assert inner.calls == 2
    assert sleeps == [1.0]


def test_gives_up_after_attempts() -> None:
    inner = Scripted([TransientProviderError("down")])
    provider, sleeps = wrap(inner)
    with pytest.raises(TransientProviderError, match="down"):
        provider.synthesize(REQ)
    assert inner.calls == 3
    assert sleeps == [1.0, 4.0]


def test_plain_provider_error_not_retried() -> None:
    inner = Scripted([ProviderError("bad")])
    provider, sleeps = wrap(inner)
    with pytest.raises(ProviderError):
        provider.synthesize(REQ)
    assert inner.calls == 1
    assert sleeps == []


def test_billed_error_not_retried() -> None:
    inner = Scripted([BilledProviderError("unusable")])
    provider, sleeps = wrap(inner)
    with pytest.raises(BilledProviderError):
        provider.synthesize(REQ)
    assert inner.calls == 1
    assert sleeps == []


def test_transient_is_not_billed() -> None:
    assert not issubclass(TransientProviderError, BilledProviderError)


def test_retry_logged_without_text(caplog: pytest.LogCaptureFixture) -> None:
    provider, _ = wrap(Scripted([TransientProviderError("down"), SPEECH]))
    with caplog.at_level(logging.WARNING):
        provider.synthesize(REQ)
    records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "fake" in message and "1" in message
    assert REQ.text not in caplog.text


def test_name_and_model_mirror_inner() -> None:
    provider, _ = wrap(Scripted([SPEECH]))
    assert (provider.name, provider.model) == ("fake", "m1")
