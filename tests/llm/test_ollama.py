import json

import httpx
import pytest
from pydantic import BaseModel

from narratoflow.errors import ProviderError
from narratoflow.llm.base import LLMClient, Usage
from narratoflow.llm.ollama import OllamaClient


class Idea(BaseModel):
    key: str
    weight: int


def _client(handler: httpx.MockTransport) -> OllamaClient:
    http = httpx.Client(transport=handler, base_url="http://ollama.test")
    return OllamaClient(base_url="http://ollama.test", model="qwen2.5", http=http)


def _reply(content: str, status: int = 200) -> httpx.Response:
    body = {
        "model": "qwen2.5",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": 12,
        "eval_count": 7,
    }
    return httpx.Response(status, json=body)


def test_valid_reply_is_parsed_and_schema_sent() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _reply('{"key": "object|lamp", "weight": 3}')

    client: LLMClient = _client(httpx.MockTransport(handler))
    result = client.complete_structured(system="sys", user="hello", output=Idea)

    assert result.value == Idea(key="object|lamp", weight=3)
    assert result.usage == Usage(input_tokens=12, output_tokens=7)
    assert result.model == "qwen2.5"
    assert client.name == "ollama"

    (request,) = seen
    assert request.method == "POST"
    assert request.url.path == "/api/chat"
    body = json.loads(request.content)
    assert body["model"] == "qwen2.5"
    assert body["stream"] is False
    assert body["format"] == Idea.model_json_schema()
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]


def test_invalid_json_raises_provider_error() -> None:
    client = _client(httpx.MockTransport(lambda r: _reply("not json {")))
    with pytest.raises(ProviderError, match="did not match"):
        client.complete_structured(system="s", user="u", output=Idea)


def test_schema_mismatch_raises_provider_error() -> None:
    client = _client(httpx.MockTransport(lambda r: _reply('{"key": "x"}')))
    with pytest.raises(ProviderError, match="did not match"):
        client.complete_structured(system="s", user="u", output=Idea)


def test_non_2xx_raises_provider_error() -> None:
    client = _client(
        httpx.MockTransport(lambda r: httpx.Response(404, json={"error": "model not found"}))
    )
    with pytest.raises(ProviderError, match="HTTP 404"):
        client.complete_structured(system="s", user="u", output=Idea)


def test_non_json_body_raises_provider_error() -> None:
    client = _client(httpx.MockTransport(lambda r: httpx.Response(200, text="<html>")))
    with pytest.raises(ProviderError, match="not valid JSON"):
        client.complete_structured(system="s", user="u", output=Idea)


def test_missing_message_content_raises_provider_error() -> None:
    client = _client(httpx.MockTransport(lambda r: httpx.Response(200, json={"done": True})))
    with pytest.raises(ProviderError, match="no message content"):
        client.complete_structured(system="s", user="u", output=Idea)


def test_connection_error_raises_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(ProviderError, match="Cannot reach Ollama") as info:
        client.complete_structured(system="s", user="u", output=Idea)
    assert isinstance(info.value.__cause__, httpx.ConnectError)


def test_timeout_raises_provider_error_naming_the_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    http = httpx.Client(transport=httpx.MockTransport(handler), timeout=45.0)
    client = OllamaClient(base_url="http://ollama.test", model="qwen2.5", http=http)
    with pytest.raises(ProviderError, match="Ollama timed out after 45s") as info:
        client.complete_structured(system="s", user="u", output=Idea)
    assert isinstance(info.value.__cause__, httpx.ReadTimeout)


@pytest.mark.parametrize(
    ("prompt_count", "eval_count", "expected"),
    [
        ("12", 7.0, Usage(input_tokens=12, output_tokens=7)),
        ("lots", None, Usage(input_tokens=0, output_tokens=0)),
        (True, {"n": 3}, Usage(input_tokens=0, output_tokens=0)),
    ],
)
def test_non_numeric_usage_counts_fall_back_to_zero(
    prompt_count: object, eval_count: object, expected: Usage
) -> None:
    body = {
        "model": "qwen2.5",
        "message": {"role": "assistant", "content": '{"key": "k", "weight": 1}'},
        "prompt_eval_count": prompt_count,
        "eval_count": eval_count,
    }
    client = _client(httpx.MockTransport(lambda r: httpx.Response(200, json=body)))
    result = client.complete_structured(system="s", user="u", output=Idea)
    assert result.usage == expected


def test_default_http_client_has_timeout() -> None:
    client = OllamaClient(base_url="http://localhost:11434", model="qwen2.5")
    assert client.http.timeout.read == 120.0


def test_base_url_trailing_slash_is_tolerated() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return _reply('{"key": "k", "weight": 1}')

    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = OllamaClient(base_url="http://ollama.test/", model="qwen2.5", http=http)
    client.complete_structured(system="s", user="u", output=Idea)
    assert seen == ["http://ollama.test/api/chat"]


def test_images_are_sent_base64_on_the_user_message() -> None:
    import base64

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _reply('{"key": "k", "weight": 1}')

    client = _client(httpx.MockTransport(handler))
    client.complete_structured(system="s", user="u", output=Idea, images=[b"\x89PNGa", b"bb"])

    messages = json.loads(seen[0].content)["messages"]
    assert messages[1]["images"] == [
        base64.b64encode(b"\x89PNGa").decode(),
        base64.b64encode(b"bb").decode(),
    ]
    assert "images" not in messages[0]


def test_no_images_leaves_messages_unchanged() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _reply('{"key": "k", "weight": 1}')

    _client(httpx.MockTransport(handler)).complete_structured(system="s", user="u", output=Idea)
    assert all("images" not in m for m in json.loads(seen[0].content)["messages"])
