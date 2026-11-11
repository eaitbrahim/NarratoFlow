"""PiperProvider against a fake `piper` module (the 1.8.0 API:
`PiperVoice.load(model_path)` and `voice.synthesize_wav(text, wave_writer)`)."""

import io
import sys
import types
import wave
from pathlib import Path
from typing import Any

import pytest

from narratoflow.errors import ChannelConfigError, ExtraMissingError, ProviderError
from narratoflow.models import Language
from narratoflow.tts import SpeechRequest
from narratoflow.tts.piper import PiperProvider

REQ = SpeechRequest(text="hello", voice_id="piper", language=Language.ENGLISH)


def fake_piper(monkeypatch: pytest.MonkeyPatch, *, fail: Exception | None = None) -> list[str]:
    loads: list[str] = []

    class PiperVoice:
        @staticmethod
        def load(model_path: Any) -> "PiperVoice":
            loads.append(str(model_path))
            return PiperVoice()

        def synthesize_wav(self, text: str, wav_file: wave.Wave_write) -> None:
            if fail is not None:
                raise fail
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b"\x00\x00" * 100)

    module = types.ModuleType("piper")
    module.PiperVoice = PiperVoice  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "piper", module)
    return loads


@pytest.fixture
def model(tmp_path: Path) -> Path:
    path = tmp_path / "en_US-lessac-medium.onnx"
    path.write_bytes(b"onnx")
    return path


def test_piper_returns_wav_without_spans(monkeypatch: pytest.MonkeyPatch, model: Path) -> None:
    loads = fake_piper(monkeypatch)
    piper = PiperProvider(model)
    assert (piper.name, piper.model) == ("piper", "en_US-lessac-medium")
    speech = piper.synthesize(REQ)
    piper.synthesize(REQ)
    assert loads == [str(model)]  # the loaded voice is cached
    assert speech.spans == ()
    assert (speech.provider, speech.model, speech.billed_chars) == (
        "piper",
        "en_US-lessac-medium",
        None,
    )
    with wave.open(io.BytesIO(speech.audio), "rb") as wav:
        assert (wav.getnchannels(), wav.getframerate(), wav.getnframes()) == (1, 22050, 100)


def test_piper_missing_model_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_piper(monkeypatch)
    with pytest.raises(ChannelConfigError, match="piper model") as info:
        PiperProvider(tmp_path / "none.onnx").synthesize(REQ)
    assert not isinstance(info.value, ExtraMissingError)


def test_piper_missing_extra(monkeypatch: pytest.MonkeyPatch, model: Path) -> None:
    monkeypatch.setitem(sys.modules, "piper", None)
    with pytest.raises(ExtraMissingError) as info:
        PiperProvider(model).synthesize(REQ)
    assert str(info.value) == "install the 'tts-local' extra: uv sync --extra tts-local"


def test_piper_synthesis_error_maps_to_provider_error(
    monkeypatch: pytest.MonkeyPatch, model: Path
) -> None:
    fake_piper(monkeypatch, fail=RuntimeError("onnx boom"))
    with pytest.raises(ProviderError) as info:
        PiperProvider(model).synthesize(REQ)
    assert str(info.value) == "piper failed: RuntimeError"


def test_piper_load_error_is_config_error(monkeypatch: pytest.MonkeyPatch, model: Path) -> None:
    fake_piper(monkeypatch)

    def broken(model_path: Any) -> None:
        raise ValueError("bad json")

    monkeypatch.setattr(sys.modules["piper"].PiperVoice, "load", staticmethod(broken))
    with pytest.raises(ChannelConfigError, match=r"cannot load piper model .*ValueError"):
        PiperProvider(model).synthesize(REQ)
