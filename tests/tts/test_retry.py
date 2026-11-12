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
        self.closed = False
        self.calls = 0

    def close(self) -> None:
        self.closed = True

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
    inner = Scripted(
        [
            TransientProviderError("first"),
            TransientProviderError("second"),
            TransientProviderError("third"),
        ]
    )
    provider, sleeps = wrap(inner)
    with pytest.raises(TransientProviderError, match="third"):
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
    inner = Scripted([TransientProviderError("a"), TransientProviderError("b"), SPEECH])
    provider, _ = wrap(inner)
    with caplog.at_level(logging.WARNING):
        provider.synthesize(REQ)
    records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.args for r in records] == [("fake", 1, 3, 1.0), ("fake", 2, 3, 4.0)]
    assert [r.getMessage() for r in records] == [
        "voice fake failed transiently (attempt 1 of 3); retrying in 1 s",
        "voice fake failed transiently (attempt 2 of 3); retrying in 4 s",
    ]
    assert REQ.text not in caplog.text


def test_close_closes_inner() -> None:
    inner = Scripted([SPEECH])
    provider, _ = wrap(inner)
    provider.close()
    assert inner.closed


def test_close_without_inner_close_is_a_noop() -> None:
    class Bare:
        name = "bare"
        model = "m"

        def synthesize(self, req: SpeechRequest) -> Speech:
            return SPEECH

    RetryingSpeechProvider(Bare()).close()


def test_name_and_model_mirror_inner() -> None:
    provider, _ = wrap(Scripted([SPEECH]))
    assert (provider.name, provider.model) == ("fake", "m1")
