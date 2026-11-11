import json
from collections.abc import Callable
from typing import Any

import anthropic
import httpx2
import pytest
from pydantic import BaseModel, SecretStr

from narratoflow.errors import ProviderError, ProviderRefusalError
from narratoflow.llm.anthropic_client import AnthropicClient
from narratoflow.llm.base import LLMClient, Usage

FAKE_KEY = "sk-ant-test-DO-NOT-LEAK-7f3c9a"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

Handler = Callable[[httpx2.Request], httpx2.Response]


class Scene(BaseModel):
    title: str
    seconds: int


def _message(
    content: list[dict[str, Any]],
    *,
    stop_reason: str = "end_turn",
    stop_details: dict[str, Any] | None = None,
    model: str = "claude-opus-5-5",
) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "stop_details": stop_details,
        "usage": {"input_tokens": 321, "output_tokens": 45},
    }


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


SCENE_JSON = '{"title": "The well", "seconds": 7}'


def _make(
    handler: Handler,
    *,
    effort: Any = "high",
    refusal_fallback: bool = False,
    max_tokens: int = 16000,
) -> AnthropicClient:
    sdk = anthropic.Anthropic(
        api_key=FAKE_KEY,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    return AnthropicClient(
        api_key=SecretStr(FAKE_KEY),
        model="claude-opus-5-5",
        effort=effort,
        max_tokens=max_tokens,
        refusal_fallback=refusal_fallback,
        client=sdk,
    )


def _recording(response: dict[str, Any], seen: list[httpx2.Request], status: int = 200) -> Handler:
    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(status, json=response)

    return handler


def test_success_returns_parsed_value_usage_and_model() -> None:
    seen: list[httpx2.Request] = []
    client: LLMClient = _make(_recording(_message([_text(SCENE_JSON)]), seen))

    result = client.complete_structured(system="be brief", user="split this", output=Scene)

    assert result.value == Scene(title="The well", seconds=7)
    assert result.usage == Usage(input_tokens=321, output_tokens=45)
    assert result.model == "claude-opus-5-5"
    assert client.name == "anthropic"


def test_request_shape_carries_effort_system_and_schema() -> None:
    seen: list[httpx2.Request] = []
    client = _make(_recording(_message([_text(SCENE_JSON)]), seen), effort="high", max_tokens=9000)

    client.complete_structured(system="be brief", user="split this", output=Scene)

    (request,) = seen
    assert request.method == "POST"
    assert request.url.path == "/v1/messages"
    body = json.loads(request.content)
    assert body["model"] == "claude-opus-5-5"
    assert body["max_tokens"] == 9000
    assert body["system"] == "be brief"
    assert body["messages"] == [{"role": "user", "content": "split this"}]
    assert body["output_config"]["effort"] == "high"
    fmt = body["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert set(fmt["schema"]["properties"]) == {"title", "seconds"}
    assert "fallbacks" not in body
    assert "temperature" not in body
    assert FALLBACK_BETA not in request.headers.get("anthropic-beta", "")


def test_refusal_fallback_sends_beta_and_default_fallbacks() -> None:
    seen: list[httpx2.Request] = []
    served_by = _message([_text(SCENE_JSON)], model="claude-opus-4-8")
    client = _make(_recording(served_by, seen), effort="medium", refusal_fallback=True)

    result = client.complete_structured(system="s", user="u", output=Scene)

    (request,) = seen
    body = json.loads(request.content)
    assert body["fallbacks"] == "default"
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert FALLBACK_BETA in request.headers["anthropic-beta"].split(",")
    assert result.value == Scene(title="The well", seconds=7)
    assert result.model == "claude-opus-4-8"


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("content", [[], [_text('{"title": "partial')], [_text("I can't help")]])
def test_refusal_raises_refusal_error_with_category(
    fallback: bool, content: list[dict[str, Any]]
) -> None:
    details = {"type": "refusal", "category": "cyber", "explanation": "declined"}
    refusal = _message(content, stop_reason="refusal", stop_details=details)
    client = _make(_recording(refusal, []), refusal_fallback=fallback)

    with pytest.raises(ProviderRefusalError) as info:
        client.complete_structured(system="s", user="u", output=Scene)
    assert info.value.category == "cyber"


def test_refusal_without_details_has_no_category() -> None:
    refusal = _message([], stop_reason="refusal", stop_details=None)
    client = _make(_recording(refusal, []))

    with pytest.raises(ProviderRefusalError) as info:
        client.complete_structured(system="s", user="u", output=Scene)
    assert info.value.category is None


def test_missing_parsed_output_raises_provider_error() -> None:
    client = _make(_recording(_message([]), []))
    with pytest.raises(ProviderError, match="no structured output"):
        client.complete_structured(system="s", user="u", output=Scene)


@pytest.mark.parametrize("text", ['{"title": "The w', SCENE_JSON])
def test_truncated_output_names_the_max_tokens_limit(text: str) -> None:
    truncated = _message([_text(text)], stop_reason="max_tokens")
    client = _make(_recording(truncated, []), max_tokens=9000)
    with pytest.raises(ProviderError, match=r"truncated at max_tokens=9000") as info:
        client.complete_structured(system="s", user="u", output=Scene)
    assert "llm.max_tokens" in str(info.value)
    assert not isinstance(info.value, ProviderRefusalError)


def test_schema_mismatch_raises_provider_error() -> None:
    client = _make(_recording(_message([_text('{"title": "x"}')]), []))
    with pytest.raises(ProviderError, match="did not match Scene"):
        client.complete_structured(system="s", user="u", output=Scene)


def _error_body(kind: str, message: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": message}}


def test_401_maps_to_provider_error_without_leaking_key() -> None:
    body = _error_body("authentication_error", f"invalid x-api-key {FAKE_KEY}")
    client = _make(_recording(body, [], status=401))

    with pytest.raises(ProviderError) as info:
        client.complete_structured(system="s", user="u", output=Scene)

    err = info.value
    assert "authentication failed" in str(err).lower()
    assert FAKE_KEY not in str(err)
    assert FAKE_KEY not in repr(err)
    assert isinstance(err.__cause__, anthropic.AuthenticationError)


@pytest.mark.parametrize(
    ("status", "kind", "expected", "cause"),
    [
        (403, "permission_error", "permission", anthropic.PermissionDeniedError),
        (404, "not_found_error", "not found", anthropic.NotFoundError),
        (429, "rate_limit_error", "rate limit", anthropic.RateLimitError),
        (400, "invalid_request_error", "rejected the request", anthropic.BadRequestError),
        (500, "api_error", "server error", anthropic.InternalServerError),
        (529, "overloaded_error", "server error", anthropic.InternalServerError),
        (413, "request_too_large", "HTTP 413", anthropic.APIStatusError),
    ],
)
def test_status_errors_map_to_provider_error(
    status: int, kind: str, expected: str, cause: type[Exception]
) -> None:
    body = _error_body(kind, f"detail mentioning {FAKE_KEY}")
    client = _make(_recording(body, [], status=status))

    with pytest.raises(ProviderError) as info:
        client.complete_structured(system="s", user="u", output=Scene)

    assert expected.lower() in str(info.value).lower()
    assert FAKE_KEY not in str(info.value)
    assert isinstance(info.value.__cause__, cause)
    assert not isinstance(info.value, ProviderRefusalError)


def test_bad_request_includes_redacted_api_detail() -> None:
    body = _error_body("invalid_request_error", "schema too complex")
    client = _make(_recording(body, [], status=400))

    with pytest.raises(ProviderError, match="schema too complex"):
        client.complete_structured(system="s", user="u", output=Scene)


@pytest.mark.parametrize(
    ("exc", "expected", "cause"),
    [
        (httpx2.ConnectError("refused"), "cannot reach", anthropic.APIConnectionError),
        (httpx2.ReadTimeout("slow"), "timed out", anthropic.APITimeoutError),
    ],
)
def test_transport_errors_map_to_provider_error(
    exc: Exception, expected: str, cause: type[Exception]
) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise exc

    client = _make(handler)
    with pytest.raises(ProviderError) as info:
        client.complete_structured(system="s", user="u", output=Scene)
    assert expected in str(info.value).lower()
    assert isinstance(info.value.__cause__, cause)


def test_default_sdk_client_uses_secret_key() -> None:
    client = AnthropicClient(
        api_key=SecretStr(FAKE_KEY),
        model="claude-opus-5-5",
        effort="medium",
        max_tokens=16000,
        refusal_fallback=True,
    )
    assert client.sdk.api_key == FAKE_KEY
    assert FAKE_KEY not in repr(client)


def test_malformed_body_raises_provider_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=["not", "a", "message"])

    client = _make(handler)
    with pytest.raises(ProviderError, match="malformed"):
        client.complete_structured(system="s", user="u", output=Scene)
