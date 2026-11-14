"""Fail-fast provider API key check."""

from __future__ import annotations

from enum import StrEnum

from narratoflow.channels.config import ChannelConfig
from narratoflow.errors import MissingApiKeyError
from narratoflow.settings import Settings


class ProviderKind(StrEnum):
    IMAGE = "image"
    VOICE = "voice"
    LLM = "llm"
    EMBED = "embed"


FREE_PROVIDERS: frozenset[str] = frozenset(
    {"ollama", "local-diffusion", "lineart-diffusion", "edge", "piper", "mishkal"}
)


def required_key_env_vars(channel: ChannelConfig) -> dict[ProviderKind, str]:
    providers = {
        ProviderKind.IMAGE: channel.style.image_provider,
        ProviderKind.VOICE: channel.voice.provider,
        ProviderKind.LLM: channel.llm.provider,
        ProviderKind.EMBED: channel.embed.provider,
    }
    return {
        kind: f"NARRATOFLOW_{kind.value.upper()}_API_KEY"
        for kind, provider in providers.items()
        if provider.strip().lower() not in FREE_PROVIDERS
    }


def check_provider_keys(channel: ChannelConfig, settings: Settings) -> None:
    configured = {
        ProviderKind.IMAGE: settings.image_api_key,
        ProviderKind.VOICE: settings.voice_api_key,
        ProviderKind.LLM: settings.llm_api_key,
        ProviderKind.EMBED: settings.embed_api_key,
    }
    missing = []
    for kind, var in required_key_env_vars(channel).items():
        secret = configured[kind]
        if secret is None or not secret.get_secret_value():
            missing.append(var)
    if missing:
        raise MissingApiKeyError(f"Missing API key environment variable(s): {', '.join(missing)}")
