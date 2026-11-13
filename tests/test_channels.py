from pathlib import Path

import pytest

from narratoflow.channels.config import (
    EmbedConfig,
    LlmConfig,
    load_channel,
    style_fingerprint,
)
from narratoflow.channels.keys import ProviderKind, check_provider_keys, required_key_env_vars
from narratoflow.errors import ChannelConfigError, MissingApiKeyError
from narratoflow.models import AspectRatio
from narratoflow.settings import get_settings


def test_load_resolves_reference_relative(channel_file: Path) -> None:
    ch = load_channel(channel_file)
    assert ch.style.reference_sketch == (channel_file.parent / "ref.png").resolve()
    assert ch.outputs == [AspectRatio.PORTRAIT]
    assert ch.background_music is False
    assert ch.riwaya == "hafs"


def test_fingerprint_changes_with_model_and_reference(channel_file: Path) -> None:
    ch = load_channel(channel_file)
    fp = style_fingerprint(ch.style)
    assert fp == style_fingerprint(ch.style)
    other = ch.style.model_copy(update={"image_model": "other-model"})
    assert style_fingerprint(other) != fp
    ch.style.reference_sketch.write_bytes(b"different")
    assert style_fingerprint(ch.style) != fp


def test_missing_reference_names_path(channel_file: Path) -> None:
    ch = load_channel(channel_file)
    ch.style.reference_sketch.unlink()
    with pytest.raises(ChannelConfigError, match=r"ref\.png"):
        style_fingerprint(ch.style)


def test_invalid_toml_wrapped(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("slug = [")
    with pytest.raises(ChannelConfigError, match=r"bad\.toml"):
        load_channel(bad)


def test_missing_file_wrapped(tmp_path: Path) -> None:
    with pytest.raises(ChannelConfigError, match=r"nope\.toml"):
        load_channel(tmp_path / "nope.toml")


def test_part_bounds_validated(channel_file: Path) -> None:
    channel_file.write_text(
        "part_min_s = 90\npart_max_s = 60\n" + channel_file.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(ChannelConfigError, match=r"history\.toml"):
        load_channel(channel_file)


def test_only_paid_kinds_need_keys(channel_file: Path) -> None:
    ch = load_channel(channel_file)
    assert required_key_env_vars(ch) == {ProviderKind.IMAGE: "NARRATOFLOW_IMAGE_API_KEY"}


def test_missing_key_fails_before_work(channel_file: Path) -> None:
    with pytest.raises(MissingApiKeyError, match="NARRATOFLOW_IMAGE_API_KEY"):
        check_provider_keys(load_channel(channel_file), get_settings())


def test_all_missing_keys_listed_together(channel_file: Path) -> None:
    text = (
        channel_file.read_text()
        .replace('"edge"', '"elevenlabs"')
        .replace('"ollama"', '"voyage"', 1)
        .replace('"ollama"', '"openai"')
    )
    channel_file.write_text(text, encoding="utf-8")
    with pytest.raises(MissingApiKeyError) as exc:
        check_provider_keys(load_channel(channel_file), get_settings())
    for var in ("IMAGE", "VOICE", "LLM", "EMBED"):
        assert f"NARRATOFLOW_{var}_API_KEY" in str(exc.value)


def test_present_key_passes(channel_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NARRATOFLOW_IMAGE_API_KEY", "x")
    check_provider_keys(load_channel(channel_file), get_settings())


def test_non_utf8_toml_wrapped(tmp_path: Path) -> None:
    bad = tmp_path / "c.toml"
    bad.write_bytes('name = "\u0642\u0646\u0627\u0629"\n'.encode("cp1256"))
    with pytest.raises(ChannelConfigError, match=r"c\.toml"):
        load_channel(bad)


def test_provider_names_case_insensitive(channel_file: Path) -> None:
    channel = load_channel(channel_file)
    channel.llm.provider = " Ollama "
    channel.voice.provider = "EDGE"
    assert ProviderKind.LLM not in required_key_env_vars(channel)
    assert ProviderKind.VOICE not in required_key_env_vars(channel)


def test_llm_config_defaults() -> None:
    llm = LlmConfig(provider="anthropic", model="claude-opus-5-5")
    assert llm.effort == "medium"
    assert llm.max_tokens == 16000
    assert llm.refusal_fallback is True


def test_llm_config_from_toml(channel_file: Path) -> None:
    text = channel_file.read_text(encoding="utf-8")
    text += 'effort = "xhigh"\nmax_tokens = 32000\nrefusal_fallback = false\n'
    channel_file.write_text(text, encoding="utf-8")
    llm = load_channel(channel_file).llm
    assert (llm.effort, llm.max_tokens, llm.refusal_fallback) == ("xhigh", 32000, False)


@pytest.mark.parametrize("field", ['effort = "extreme"', "max_tokens = 0"])
def test_llm_config_rejects_bad_values(channel_file: Path, field: str) -> None:
    channel_file.write_text(
        channel_file.read_text(encoding="utf-8") + field + "\n", encoding="utf-8"
    )
    with pytest.raises(ChannelConfigError):
        load_channel(channel_file)


def test_default_embed_provider_is_paid(
    channel_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    channel = load_channel(channel_file).model_copy(update={"embed": EmbedConfig()})
    assert required_key_env_vars(channel)[ProviderKind.EMBED] == "NARRATOFLOW_EMBED_API_KEY"
    monkeypatch.setenv("NARRATOFLOW_IMAGE_API_KEY", "x")
    with pytest.raises(MissingApiKeyError, match="NARRATOFLOW_EMBED_API_KEY"):
        check_provider_keys(channel, get_settings())
    monkeypatch.setenv("NARRATOFLOW_EMBED_API_KEY", "e")
    get_settings.cache_clear()
    check_provider_keys(channel, get_settings())


def test_fixture_embed_is_free_ollama(channel_file: Path) -> None:
    channel = load_channel(channel_file)
    assert channel.embed == EmbedConfig(provider="ollama", model="nomic-embed-text")
    assert ProviderKind.EMBED not in required_key_env_vars(channel)


def test_voice_fallback_defaults_and_reciter(channel_file: Path) -> None:
    ch = load_channel(channel_file)
    assert ch.voice.fallback == ["edge"]
    assert ch.voice.edge_voice is None
    assert ch.voice.piper_model is None
    assert ch.reciter is None


def test_voice_provider_lowercased(channel_file: Path) -> None:
    channel_file.write_text(
        channel_file.read_text(encoding="utf-8").replace('"edge"', '" EDGE "'), encoding="utf-8"
    )
    assert load_channel(channel_file).voice.provider == "edge"


def test_piper_model_resolved_relative(channel_file: Path) -> None:
    channel_file.write_text(
        channel_file.read_text(encoding="utf-8").replace(
            'voice_id = "ar-SA-HamedNeural"',
            'voice_id = "ar-SA-HamedNeural"\nfallback = ["piper"]\npiper_model = "voices/m.onnx"',
        ),
        encoding="utf-8",
    )
    ch = load_channel(channel_file)
    assert ch.voice.piper_model == (channel_file.parent / "voices" / "m.onnx").resolve()
    assert ch.voice.fallback == ["piper"]


@pytest.mark.parametrize(
    ("provider", "extra"),
    [("edge", 'fallback = ["piper"]'), ("Piper", "fallback = []"), ("piper", "")],
)
def test_piper_without_model_rejected_by_config(
    channel_file: Path, provider: str, extra: str
) -> None:
    text = (
        channel_file.read_text(encoding="utf-8")
        .replace('provider = "edge"', f'provider = "{provider}"')
        .replace('voice_id = "ar-SA-HamedNeural"', f'voice_id = "x"\n{extra}')
    )
    channel_file.write_text(text, encoding="utf-8")
    with pytest.raises(ChannelConfigError, match=r"piper needs voice\.piper_model"):
        load_channel(channel_file)


def _piper_channel(channel_file: Path, extra: str) -> Path:
    channel_file.write_text(
        channel_file.read_text(encoding="utf-8").replace(
            'voice_id = "ar-SA-HamedNeural"',
            'voice_id = "ar-SA-HamedNeural"\nfallback = ["piper"]\n'
            f'piper_model = "voices/m.onnx"\n{extra}',
        ),
        encoding="utf-8",
    )
    return channel_file


def test_piper_licence_needs_source(channel_file: Path) -> None:
    _piper_channel(channel_file, 'piper_model_license = "MIT"')
    with pytest.raises(ChannelConfigError, match=r"piper_model_license needs.*piper_model_source"):
        load_channel(channel_file)


def test_piper_licence_loads(channel_file: Path) -> None:
    _piper_channel(
        channel_file,
        'piper_model_license = "CC-BY-4.0"\n'
        'piper_model_source = "https://example.org/voice"\n'
        'piper_model_attribution = "Voice by A. Reader"',
    )
    voice = load_channel(channel_file).voice
    assert voice.piper_model_license == "CC-BY-4.0"
    assert voice.piper_model_source == "https://example.org/voice"
    assert voice.piper_model_attribution == "Voice by A. Reader"


def test_piper_licence_defaults_undeclared(channel_file: Path) -> None:
    voice = load_channel(_piper_channel(channel_file, "")).voice
    assert (voice.piper_model_license, voice.piper_model_source) == (None, None)
    assert voice.piper_model_attribution == ""


def test_unknown_fallback_rejected(channel_file: Path) -> None:
    channel_file.write_text(
        channel_file.read_text(encoding="utf-8").replace(
            'voice_id = "ar-SA-HamedNeural"', 'voice_id = "x"\nfallback = ["elevenlabs"]'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ChannelConfigError, match="fallback"):
        load_channel(channel_file)


def test_typo_in_spending_limit_is_an_error(channel_file: Path) -> None:
    channel_file.write_text("spending_limt_usd = 2\n" + channel_file.read_text())
    with pytest.raises(ChannelConfigError) as exc:
        load_channel(channel_file)
    assert "spending_limt_usd" in str(exc.value)
    assert "Extra inputs are not permitted" in str(exc.value)


@pytest.mark.parametrize(
    ("table", "line"),
    [
        ("style", "line_weigth_px = 3"),
        ("voice", 'fallbak = ["edge"]'),
        ("llm", 'efort = "low"'),
        ("embed", 'modle = "x"'),
        ("vision", 'provder = "none"'),
    ],
)
def test_unknown_nested_key_is_an_error(channel_file: Path, table: str, line: str) -> None:
    text = channel_file.read_text()
    header = f"[{table}]"
    if header in text:
        text = text.replace(header, f"{header}\n{line}", 1)
    else:
        text += f"\n{header}\n{line}\n"
    channel_file.write_text(text)
    with pytest.raises(ChannelConfigError) as exc:
        load_channel(channel_file)
    key = line.split(" ")[0]
    assert f"{table}.{key}" in str(exc.value)


def test_unknown_key_holding_a_secret_is_named_but_not_echoed(channel_file: Path) -> None:
    channel_file.write_text(
        channel_file.read_text().replace("[llm]", '[llm]\napi_key = "sk-ant-SECRET999"', 1)
    )
    with pytest.raises(ChannelConfigError) as exc:
        load_channel(channel_file)
    assert "llm.api_key" in str(exc.value)
    assert "sk-ant-SECRET999" not in str(exc.value)
    assert "sk-ant-SECRET999" not in str(exc.value.__cause__)


def test_fixture_channel_still_loads(channel_file: Path) -> None:
    assert load_channel(channel_file).riwaya == "hafs"


def test_pronunciation_table_loads_and_validates(channel_file: Path) -> None:
    base = channel_file.read_text(encoding="utf-8")
    assert load_channel(channel_file).pronunciation == {}
    channel_file.write_text(base + '\n[pronunciation]\n"NASA" = "nasa"\n', encoding="utf-8")
    assert load_channel(channel_file).pronunciation == {"NASA": "nasa"}
    channel_file.write_text(
        base + '\n[pronunciation]\n"two words" = "SECRETPHRASE"\n"ok" = "OTHERVALUE"\n',
        encoding="utf-8",
    )
    with pytest.raises(ChannelConfigError, match="two words") as exc:
        load_channel(channel_file)
    assert "SECRETPHRASE" not in str(exc.value)
    assert "OTHERVALUE" not in str(exc.value)


def test_end_card_config(channel_file: Path) -> None:
    base = channel_file.read_text(encoding="utf-8")
    assert load_channel(channel_file).end_card is None
    channel_file.write_text('end_card = ""\n' + base, encoding="utf-8")
    assert load_channel(channel_file).end_card == ""
