from pathlib import Path

import pytest
from pydantic import SecretStr

from narratoflow.channels.config import ChannelConfig, VisionConfig, load_channel, vision_model
from narratoflow.costs.prices import default_prices
from narratoflow.costs.tracker import SpendTracker, TrackedLLM
from narratoflow.errors import ChannelConfigError, MissingApiKeyError
from narratoflow.llm.anthropic_client import AnthropicClient
from narratoflow.llm.factory import build_llm_client, make_llm, vision_llm_config
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


def test_vision_model_defaults_to_channel_llm(channel_file: Path) -> None:
    base = _channel(channel_file, " Ollama ", "qwen2.5")
    assert vision_model(base) == ("ollama", "qwen2.5")
    split = base.model_copy(
        update={"vision": VisionConfig(provider="Anthropic", model="claude-haiku-5-5")}
    )
    assert vision_model(split) == ("anthropic", "claude-haiku-5-5")
    only_model = base.model_copy(update={"vision": VisionConfig(model="llava")})
    assert vision_model(only_model) == ("ollama", "llava")


def test_vision_llm_config_keeps_llm_settings(channel_file: Path) -> None:
    base = _channel(channel_file, "ollama", "qwen2.5")
    base = base.model_copy(update={"llm": base.llm.model_copy(update={"max_tokens": 77})})
    split = base.model_copy(
        update={"vision": VisionConfig(provider="anthropic", model="claude-haiku-5-5")}
    )
    cfg = vision_llm_config(split)
    assert (cfg.provider, cfg.model, cfg.max_tokens) == ("anthropic", "claude-haiku-5-5", 77)
    client = build_llm_client(cfg, Settings(llm_api_key=SecretStr("k")))
    assert isinstance(client, AnthropicClient)
    with pytest.raises(MissingApiKeyError):
        build_llm_client(cfg, Settings(llm_api_key=None))
