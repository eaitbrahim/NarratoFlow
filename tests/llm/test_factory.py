from pathlib import Path

import pytest
from pydantic import SecretStr

from narratoflow.channels.config import ChannelConfig, load_channel
from narratoflow.costs.prices import default_prices
from narratoflow.costs.tracker import SpendTracker, TrackedLLM
from narratoflow.errors import ChannelConfigError, MissingApiKeyError
from narratoflow.llm.anthropic_client import AnthropicClient
from narratoflow.llm.factory import make_llm
from narratoflow.llm.ollama import OllamaClient
from narratoflow.settings import Settings


def _channel(channel_file: Path, provider: str, model: str = "m") -> ChannelConfig:
    base = load_channel(channel_file)
    return base.model_copy(
        update={"llm": base.llm.model_copy(update={"provider": provider, "model": model})}
    )


def _make(channel: ChannelConfig, settings: Settings) -> TrackedLLM:
    prices = default_prices(settings)
    llm = make_llm(channel, settings, SpendTracker(None), prices)
    assert isinstance(llm, TrackedLLM)
    return llm


def test_make_llm_unknown_provider(channel_file: Path) -> None:
    with pytest.raises(ChannelConfigError):
        make_llm(
            _channel(channel_file, "nope"),
            Settings(),
            SpendTracker(None),
            default_prices(Settings()),
        )


def test_make_llm_anthropic_without_key(channel_file: Path) -> None:
    with pytest.raises(MissingApiKeyError):
        _make(_channel(channel_file, "anthropic"), Settings(llm_api_key=None))
    with pytest.raises(MissingApiKeyError):
        _make(_channel(channel_file, "anthropic"), Settings(llm_api_key=SecretStr("")))


def test_make_llm_anthropic_case_insensitive(channel_file: Path) -> None:
    llm = _make(
        _channel(channel_file, "Anthropic", "claude-opus-5-5"),
        Settings(llm_api_key=SecretStr("k")),
    )
    assert isinstance(llm.inner, AnthropicClient)


def test_make_llm_ollama_uses_settings_url(channel_file: Path) -> None:
    llm = _make(_channel(channel_file, "ollama"), Settings(ollama_url="http://h:1/"))
    assert isinstance(llm.inner, OllamaClient)
    assert llm.inner.url == "http://h:1/api/chat"


def test_ollama_url_default() -> None:
    assert Settings().ollama_url == "http://localhost:11434"
