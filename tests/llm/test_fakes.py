import pytest
from fakes.llm import FakeLLM
from pydantic import BaseModel

from narratoflow.errors import ProviderError
from narratoflow.llm.base import LLMClient, Usage


class Answer(BaseModel):
    text: str


class Other(BaseModel):
    n: int


def test_fake_returns_queued_values_and_records_calls() -> None:
    fake = FakeLLM([Answer(text="a"), ProviderError("boom")])
    client: LLMClient = fake

    result = client.complete_structured(system="s", user="u", output=Answer)

    assert result.value == Answer(text="a")
    assert result.usage == Usage(input_tokens=100, output_tokens=50)
    assert fake.calls == [("s", "u", Answer)]
    with pytest.raises(ProviderError, match="boom"):
        client.complete_structured(system="s2", user="u2", output=Answer)
    with pytest.raises(AssertionError, match="no queued responses"):
        client.complete_structured(system="s3", user="u3", output=Answer)


def test_fake_rejects_wrong_queued_type() -> None:
    fake = FakeLLM([Other(n=1)])
    with pytest.raises(AssertionError, match="caller asked for Answer"):
        fake.complete_structured(system="s", user="u", output=Answer)
