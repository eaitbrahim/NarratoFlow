import json
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from narratoflow.channels.config import ChannelConfig, EmbedConfig, load_channel
from narratoflow.costs.prices import PriceTable, default_prices
from narratoflow.costs.tracker import SpendTracker, estimate_tokens
from narratoflow.embed import (
    Embedder,
    EmbedResult,
    OllamaEmbedder,
    TrackedEmbedder,
    VoyageEmbedder,
    make_embedder,
)
from narratoflow.errors import (
    BudgetExceededError,
    ChannelConfigError,
    MissingApiKeyError,
    ProviderError,
)
from narratoflow.settings import Settings

FAKE_KEY = "pa-SUPERSECRET-0xDEADBEEF"
Handler = Callable[[httpx.Request], httpx.Response]


def _voyage(handler: Handler, model: str = "voyage-multilingual-2") -> VoyageEmbedder:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return VoyageEmbedder(SecretStr(FAKE_KEY), model, http)


def _voyage_reply(request: httpx.Request, *, reverse: bool = False) -> httpx.Response:
    texts = json.loads(request.content)["input"]
    data = [
        {"object": "embedding", "embedding": [float(len(t)), 1.0], "index": i}
        for i, t in enumerate(texts)
    ]
    if reverse:
        data.reverse()
    return httpx.Response(200, json={"data": data, "usage": {"total_tokens": 7 * len(texts)}})


def test_voyage_request_shape() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _voyage_reply(request)

    embedder: Embedder = _voyage(handler)
    assert embedder.embed(["ab", "abcd"]) == [[2.0, 1.0], [4.0, 1.0]]
    assert embedder.name == "voyage"
    assert embedder.model == "voyage-multilingual-2"
    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == "https://api.voyageai.com/v1/embeddings"
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert json.loads(request.content) == {
        "input": ["ab", "abcd"],
        "model": "voyage-multilingual-2",
    }


def test_voyage_batches_at_128_and_keeps_order() -> None:
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sizes.append(len(json.loads(request.content)["input"]))
        return _voyage_reply(request, reverse=True)

    texts = ["x" * (n % 50 + 1) for n in range(300)]
    embedder = _voyage(handler)
    result = embedder.embed_with_usage(texts)
    assert sizes == [128, 128, 44]
    assert result.vectors == [[float(len(t)), 1.0] for t in texts]
    assert result.total_tokens == 7 * 300


def test_voyage_empty_input_makes_no_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no HTTP call expected")

    assert _voyage(handler).embed([]) == []


def test_voyage_401_does_not_leak_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": f"bad key {FAKE_KEY}"})

    with pytest.raises(ProviderError, match="HTTP 401") as info:
        _voyage(handler).embed(["a"])
    assert FAKE_KEY not in str(info.value)
    assert "DEADBEEF" not in str(info.value)


def test_voyage_timeout_and_connection_errors() -> None:
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(f"slow {FAKE_KEY}", request=request)

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused {FAKE_KEY}", request=request)

    with pytest.raises(ProviderError, match="timed out") as info:
        _voyage(slow).embed(["a"])
    assert FAKE_KEY not in str(info.value)
    with pytest.raises(ProviderError, match="Cannot reach Voyage") as info:
        _voyage(down).embed(["a"])
    assert FAKE_KEY not in str(info.value)
    assert isinstance(info.value.__cause__, httpx.ConnectError)


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        json.dumps([1, 2]),
        json.dumps({"data": "x"}),
        json.dumps({"data": [{"embedding": [1.0], "index": 0}, {"embedding": [1.0], "index": 0}]}),
        json.dumps({"data": [{"embedding": "x", "index": 0}, {"embedding": [1.0], "index": 1}]}),
        json.dumps({"data": [{"embedding": [True], "index": 0}, {"embedding": [1], "index": 1}]}),
        json.dumps({"data": [{"embedding": [1.0]}, {"embedding": [1.0], "index": 1}]}),
        json.dumps({"data": [{"embedding": [1.0], "index": 0}]}),
        json.dumps({"data": ["x", "y"]}),
        json.dumps({"data": [{"embedding": [1.0], "index": 0}, {"embedding": [1.0], "index": 5}]}),
    ],
)
def test_voyage_malformed_body(body: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    with pytest.raises(ProviderError, match="Voyage") as info:
        _voyage(handler).embed(["a", "b"])
    assert FAKE_KEY not in str(info.value)


def test_voyage_usage_missing_or_bad_is_none() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"data": [{"embedding": [1.0], "index": 0}], "usage": {"total_tokens": "x"}}
        )

    assert _voyage(handler).embed_with_usage(["a"]).total_tokens is None


def test_voyage_default_client_timeout_60s() -> None:
    embedder = VoyageEmbedder(SecretStr(FAKE_KEY), "m")
    assert embedder.http.timeout.read == 60.0
    assert FAKE_KEY not in repr(embedder.__dict__)


def _ollama(handler: Handler) -> OllamaEmbedder:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaEmbedder("http://ollama.test/", "nomic-embed-text", http)


def test_ollama_request_shape() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"embeddings": [[1.0, 2.0], [3, 4]]})

    embedder: Embedder = _ollama(handler)
    assert embedder.embed(["a", "b"]) == [[1.0, 2.0], [3.0, 4.0]]
    assert embedder.name == "ollama"
    (request,) = seen
    assert str(request.url) == "http://ollama.test/api/embed"
    assert json.loads(request.content) == {"model": "nomic-embed-text", "input": ["a", "b"]}
    assert _ollama(handler).embed_with_usage(["a", "b"]).total_tokens is None
    assert _ollama(handler).embed([]) == []


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (httpx.Response(404, json={"error": "model not found"}), "HTTP 404"),
        (httpx.Response(200, text="<html>"), "not valid JSON"),
        (httpx.Response(200, json={"embeddings": [[1.0]]}), "vectors"),
        (httpx.Response(200, json={"nope": 1}), "vectors"),
        (httpx.Response(200, json={"embeddings": [[1.0], ["x"]]}), "vectors"),
    ],
)
def test_ollama_errors(response: httpx.Response, match: str) -> None:
    with pytest.raises(ProviderError, match=match):
        _ollama(lambda r: response).embed(["a", "b"])


def test_ollama_transport_errors() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError, match="Cannot reach Ollama"):
        _ollama(down).embed(["a"])
    with pytest.raises(ProviderError, match="timed out"):
        _ollama(slow).embed(["a"])


class _Inner:
    name = "voyage"
    model = "voyage-multilingual-2"

    def __init__(self, total_tokens: int | None, fail: bool = False) -> None:
        self.total_tokens = total_tokens
        self.fail = fail
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_with_usage(texts).vectors

    def embed_with_usage(self, texts: Sequence[str]) -> EmbedResult:
        self.calls += 1
        if self.fail:
            raise ProviderError("boom")
        return EmbedResult([[1.0] for _ in texts], self.total_tokens)


_PRICES = PriceTable({"embed": {"voyage/voyage-multilingual-2": (Decimal("0.12"),)}})


def _price(tokens: int) -> Decimal:
    return tokens * Decimal("0.12") / Decimal(1_000_000)


def test_tracked_charges_real_usage_when_reported() -> None:
    tracker = SpendTracker(None)
    tracked = TrackedEmbedder(_Inner(500), tracker, "voyage", "voyage-multilingual-2", _PRICES)
    assert tracked.embed(["abcd", "abcdefgh"]) == [[1.0], [1.0]]
    assert tracked.name == "voyage"
    assert tracked.model == "voyage-multilingual-2"
    assert tracker.spent == _price(500)
    assert "embed voyage/voyage-multilingual-2" in tracker.entries[0].what


def test_tracked_charges_estimate_without_usage() -> None:
    tracker = SpendTracker(None)
    texts = ["abcd", "abcdefgh"]
    tracked = TrackedEmbedder(_Inner(None), tracker, "voyage", "voyage-multilingual-2", _PRICES)
    tracked.embed(texts)
    assert tracker.spent == _price(sum(estimate_tokens(t) for t in texts))


def test_tracked_checks_budget_before_call() -> None:
    inner = _Inner(1)
    tracker = SpendTracker(Decimal("0.000001"))
    tracked = TrackedEmbedder(inner, tracker, "voyage", "voyage-multilingual-2", _PRICES)
    with pytest.raises(BudgetExceededError):
        tracked.embed(["a" * 400])
    assert inner.calls == 0


def test_tracked_failed_call_charges_nothing() -> None:
    tracker = SpendTracker(None)
    tracked = TrackedEmbedder(
        _Inner(5, fail=True), tracker, "voyage", "voyage-multilingual-2", _PRICES
    )
    with pytest.raises(ProviderError):
        tracked.embed(["a"])
    assert tracker.spent == 0


def test_tracked_empty_input_skips_call() -> None:
    inner = _Inner(5)
    tracked = TrackedEmbedder(inner, SpendTracker(None), "voyage", "m", _PRICES)
    assert tracked.embed([]) == []
    assert inner.calls == 0


def _channel(
    channel_file: Path, provider: str, model: str = "voyage-multilingual-2"
) -> ChannelConfig:
    base = load_channel(channel_file)
    return base.model_copy(update={"embed": EmbedConfig(provider=provider, model=model)})


def _make(channel: ChannelConfig, settings: Settings) -> Embedder:
    return make_embedder(channel, settings, SpendTracker(None), default_prices(settings))


def test_make_embedder_voyage_requires_key(channel_file: Path) -> None:
    for key in (None, SecretStr("")):
        with pytest.raises(MissingApiKeyError, match="NARRATOFLOW_EMBED_API_KEY"):
            _make(_channel(channel_file, "voyage"), Settings(embed_api_key=key))


def test_make_embedder_voyage_case_insensitive(channel_file: Path) -> None:
    embedder = _make(_channel(channel_file, " Voyage "), Settings(embed_api_key=SecretStr("k")))
    assert isinstance(embedder, TrackedEmbedder)
    assert isinstance(embedder.inner, VoyageEmbedder)
    assert embedder.provider == "voyage"


def test_make_embedder_ollama_uses_settings_url(channel_file: Path) -> None:
    embedder = _make(
        _channel(channel_file, "ollama", "nomic-embed-text"), Settings(ollama_url="http://h:1/")
    )
    assert isinstance(embedder, TrackedEmbedder)
    assert isinstance(embedder.inner, OllamaEmbedder)
    assert embedder.inner.url == "http://h:1/api/embed"


def test_make_embedder_unknown_provider(channel_file: Path) -> None:
    with pytest.raises(ChannelConfigError, match="nope"):
        _make(_channel(channel_file, "nope"), Settings())


def test_embed_config_defaults() -> None:
    cfg = EmbedConfig()
    assert (cfg.provider, cfg.model) == ("voyage", "voyage-multilingual-2")
