"""Build the configured LLM client, wrapped so every call is budget-checked."""

from __future__ import annotations

from narratoflow.channels.config import ChannelConfig
from narratoflow.costs.prices import PriceTable
from narratoflow.costs.tracker import SpendTracker, TrackedLLM
from narratoflow.errors import ChannelConfigError, MissingApiKeyError
from narratoflow.llm.anthropic_client import KEY_ENV_VAR, AnthropicClient
from narratoflow.llm.base import LLMClient
from narratoflow.llm.ollama import OllamaClient
from narratoflow.settings import Settings


def make_llm(
    channel: ChannelConfig, settings: Settings, tracker: SpendTracker, prices: PriceTable
) -> LLMClient:
    cfg = channel.llm
    provider = cfg.provider.strip().lower()
    inner: LLMClient
    if provider == "anthropic":
        key = settings.llm_api_key
        if key is None or not key.get_secret_value():
            raise MissingApiKeyError(f"Missing API key environment variable(s): {KEY_ENV_VAR}")
        inner = AnthropicClient(
            api_key=key,
            model=cfg.model,
            effort=cfg.effort,
            max_tokens=cfg.max_tokens,
            refusal_fallback=cfg.refusal_fallback,
        )
    elif provider == "ollama":
        inner = OllamaClient(base_url=settings.ollama_url, model=cfg.model)
    else:
        raise ChannelConfigError(f"unsupported LLM provider {cfg.provider!r}")
    return TrackedLLM(inner, tracker, provider, cfg.model, prices)
