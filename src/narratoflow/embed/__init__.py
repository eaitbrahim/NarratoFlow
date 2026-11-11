"""Text embedders (Voyage REST, local Ollama) and the budget-checked wrapper.

Texts are never logged: they are the author's script and the channel's catalog.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

import httpx
from pydantic import SecretStr

from narratoflow.channels.config import ChannelConfig
from narratoflow.costs.prices import PriceTable
from narratoflow.costs.tracker import SpendTracker, estimate_tokens
from narratoflow.errors import ChannelConfigError, MissingApiKeyError, ProviderError
from narratoflow.settings import Settings

logger = logging.getLogger(__name__)

VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
VOYAGE_BATCH = 128
VOYAGE_TIMEOUT_S = 60.0
OLLAMA_TIMEOUT_S = 120.0
KEY_ENV_VAR = "NARRATOFLOW_EMBED_API_KEY"
_MILLION = Decimal(1_000_000)


class Embedder(Protocol):
    name: str
    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class EmbedResult:
    vectors: list[list[float]]
    total_tokens: int | None


class UsageEmbedder(Embedder, Protocol):
    """An embedder that also reports billed tokens, when the provider returns them."""

    def embed_with_usage(self, texts: Sequence[str]) -> EmbedResult: ...


def _vector(raw: object) -> list[float] | None:
    if not isinstance(raw, list) or not raw:
        return None
    if not all(isinstance(x, int | float) and not isinstance(x, bool) for x in raw):
        return None
    return [float(x) for x in raw]


def _vectors(raw: object, expected: int) -> list[list[float]] | None:
    if not isinstance(raw, list) or len(raw) != expected:
        return None
    vectors = [_vector(item) for item in raw]
    if any(v is None for v in vectors):
        return None
    return [v for v in vectors if v is not None]


class VoyageEmbedder:
    """Paid provider: POST /v1/embeddings with bearer auth, batches of 128."""

    name: str = "voyage"

    def __init__(self, api_key: SecretStr, model: str, http: httpx.Client | None = None) -> None:
        self._api_key = api_key
        self.model = model
        self.http = http if http is not None else httpx.Client(timeout=VOYAGE_TIMEOUT_S)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_with_usage(texts).vectors

    def embed_with_usage(self, texts: Sequence[str]) -> EmbedResult:
        vectors: list[list[float]] = []
        total: int | None = 0
        for start in range(0, len(texts), VOYAGE_BATCH):
            batch_vectors, batch_tokens = self._batch(list(texts[start : start + VOYAGE_BATCH]))
            vectors.extend(batch_vectors)
            total = None if total is None or batch_tokens is None else total + batch_tokens
        return EmbedResult(vectors, total)

    def _scrub(self, text: str) -> str:
        secret = self._api_key.get_secret_value()
        return text.replace(secret, "***") if secret else text

    def _batch(self, texts: list[str]) -> tuple[list[list[float]], int | None]:
        headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
        try:
            response = self.http.post(
                VOYAGE_URL, json={"input": texts, "model": self.model}, headers=headers
            )
        except httpx.TimeoutException as e:
            raise ProviderError(f"Voyage embeddings timed out ({type(e).__name__})") from e
        except httpx.HTTPError as e:
            raise ProviderError(
                f"Cannot reach Voyage at {VOYAGE_URL}: {self._scrub(str(e))}"
            ) from e
        if not response.is_success:
            raise ProviderError(
                f"Voyage returned HTTP {response.status_code} for model {self.model!r}: "
                f"{self._scrub(response.text[:200])}"
            )
        try:
            body: Any = response.json()
        except ValueError as e:
            raise ProviderError("Voyage response body is not valid JSON") from e
        vectors = self._parse(body, len(texts))
        if vectors is None:
            raise ProviderError(f"Voyage response does not hold {len(texts)} embeddings")
        usage = body.get("usage")
        tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        valid = isinstance(tokens, int) and not isinstance(tokens, bool)
        return vectors, tokens if valid else None

    @staticmethod
    def _parse(body: Any, expected: int) -> list[list[float]] | None:
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list) or len(data) != expected:
            return None
        by_index: dict[int, object] = {}
        for item in data:
            index = item.get("index") if isinstance(item, dict) else None
            if not isinstance(index, int) or isinstance(index, bool) or index in by_index:
                return None
            by_index[index] = item.get("embedding")
        if sorted(by_index) != list(range(expected)):
            return None
        return _vectors([by_index[i] for i in range(expected)], expected)


class OllamaEmbedder:
    """Free local provider: POST /api/embed, reading `embeddings`."""

    name: str = "ollama"

    def __init__(self, base_url: str, model: str, http: httpx.Client | None = None) -> None:
        self.url = f"{base_url.rstrip('/')}/api/embed"
        self.model = model
        self.http = http if http is not None else httpx.Client(timeout=OLLAMA_TIMEOUT_S)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_with_usage(texts).vectors

    def embed_with_usage(self, texts: Sequence[str]) -> EmbedResult:
        if not texts:
            return EmbedResult([], None)
        try:
            response = self.http.post(self.url, json={"model": self.model, "input": list(texts)})
        except httpx.TimeoutException as e:
            raise ProviderError(f"Ollama embeddings timed out ({type(e).__name__})") from e
        except httpx.HTTPError as e:
            raise ProviderError(f"Cannot reach Ollama at {self.url}: {e}") from e
        if not response.is_success:
            raise ProviderError(
                f"Ollama returned HTTP {response.status_code} for model {self.model!r}: "
                f"{response.text[:200]}"
            )
        try:
            body: Any = response.json()
        except ValueError as e:
            raise ProviderError("Ollama response body is not valid JSON") from e
        raw = body.get("embeddings") if isinstance(body, dict) else None
        vectors = _vectors(raw, len(texts))
        if vectors is None:
            raise ProviderError(f"Ollama response does not hold {len(texts)} vectors")
        return EmbedResult(vectors, None)


class TrackedEmbedder:
    """Wraps an embedder: checks the budget before each call, charges after."""

    def __init__(
        self,
        inner: UsageEmbedder,
        tracker: SpendTracker,
        provider: str,
        model: str,
        prices: PriceTable,
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = model
        self.tracker = tracker
        self.provider = provider
        self.prices = prices

    def _cost(self, tokens: int) -> Decimal:
        return tokens * self.prices.embed(self.provider, self.model) / _MILLION

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        what = f"embed {self.provider}/{self.model}"
        estimate = self._cost(max(1, sum(estimate_tokens(t) for t in texts)))
        self.tracker.ensure_room(estimate, what)
        try:
            result = self.inner.embed_with_usage(texts)
        except ProviderError as err:
            # A failed call is charged nothing: no usage was returned to bill.
            logger.warning("embedding call failed (%s/%s): %s", self.provider, self.model, err)
            raise
        if result.total_tokens is not None:
            self.tracker.charge(self._cost(result.total_tokens), what)
        else:
            self.tracker.charge(estimate, what)
        return result.vectors


def make_embedder(
    channel: ChannelConfig, settings: Settings, tracker: SpendTracker, prices: PriceTable
) -> TrackedEmbedder:
    cfg = channel.embed
    provider = cfg.provider.strip().lower()
    inner: UsageEmbedder
    if provider == "voyage":
        key = settings.embed_api_key
        if key is None or not key.get_secret_value():
            raise MissingApiKeyError(f"Missing API key environment variable(s): {KEY_ENV_VAR}")
        inner = VoyageEmbedder(api_key=key, model=cfg.model)
    elif provider == "ollama":
        inner = OllamaEmbedder(base_url=settings.ollama_url, model=cfg.model)
    else:
        raise ChannelConfigError(f"unsupported embedding provider {cfg.provider!r}")
    return TrackedEmbedder(inner, tracker, provider, cfg.model, prices)


__all__ = [
    "EmbedResult",
    "Embedder",
    "OllamaEmbedder",
    "TrackedEmbedder",
    "UsageEmbedder",
    "VoyageEmbedder",
    "make_embedder",
]
