"""build_voice_chain, TrackedSpeechProvider and estimate_speech_cost."""

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from narratoflow.channels.config import ChannelConfig
from narratoflow.costs.prices import PriceTable
from narratoflow.costs.tracker import SpendTracker, estimate_speech_cost, require_prices
from narratoflow.errors import (
    BilledProviderError,
    BudgetExceededError,
    ChannelConfigError,
    MissingApiKeyError,
    ProviderError,
)
from narratoflow.models import Language
from narratoflow.settings import Settings
from narratoflow.tts import Speech, SpeechRequest
from narratoflow.tts.edge import EdgeProvider
from narratoflow.tts.elevenlabs import ElevenLabsProvider
from narratoflow.tts.factory import TrackedSpeechProvider, build_voice_chain
from narratoflow.tts.piper import PiperProvider
from narratoflow.tts.retry import RetryingSpeechProvider

VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
PRICES = PriceTable({"voice": {"elevenlabs/eleven_multilingual_v2": (Decimal("0.30"),)}})


def channel(language: str = "ar", **voice: Any) -> ChannelConfig:
    return ChannelConfig.model_validate(
        {
            "name": "History",
            "slug": "history",
            "language": language,
            "style": {
                "reference_sketch": "ref.png",
                "instructions": "ink",
                "image_provider": "openai",
                "image_model": "gpt-image-1",
            },
            "voice": voice,
            "llm": {"provider": "ollama", "model": "qwen2.5"},
        }
    )


def piper_model(path: Path) -> Path:
    path.write_bytes(b"onnx")
    Path(f"{path}.json").write_text("{}", encoding="utf-8")
    return path


def keyed() -> Settings:
    return Settings(voice_api_key=SecretStr("xi-key"))


def eleven(**over: Any) -> dict[str, Any]:
    return {
        "provider": "elevenlabs",
        "model": "eleven_multilingual_v2",
        "voice_id": VOICE_ID,
        **over,
    }


def test_chain_paid_then_edge_default_voice_for_language() -> None:
    chain = build_voice_chain(channel("ar", **eleven()), keyed())
    assert len(chain) == 2
    paid, free = chain
    assert isinstance(paid.provider, RetryingSpeechProvider)
    assert isinstance(paid.provider.inner, ElevenLabsProvider)
    assert (paid.provider.model, paid.voice_id) == ("eleven_multilingual_v2", VOICE_ID)
    assert isinstance(free.provider, RetryingSpeechProvider)
    assert isinstance(free.provider.inner, EdgeProvider)
    assert free.voice_id == "ar-SA-HamedNeural"
    english = build_voice_chain(channel("en", **eleven()), keyed())
    assert english[1].voice_id == "en-US-AndrewNeural"
    custom = build_voice_chain(channel("en", **eleven(edge_voice="en-GB-RyanNeural")), keyed())
    assert custom[1].voice_id == "en-GB-RyanNeural"


def test_chain_with_piper_fallback(tmp_path: Path) -> None:
    model = piper_model(tmp_path / "ar_JO-kareem-medium.onnx")
    chain = build_voice_chain(
        channel(**eleven(fallback=["piper", "edge", "piper"], piper_model=str(model))), keyed()
    )
    assert [c.provider.name for c in chain] == ["elevenlabs", "piper", "edge"]
    assert isinstance(chain[1].provider, PiperProvider)
    assert (chain[1].provider.model, chain[1].voice_id) == ("ar_JO-kareem-medium", "piper")


def test_piper_carries_declared_licence(tmp_path: Path) -> None:
    model = piper_model(tmp_path / "m.onnx")
    chain = build_voice_chain(
        channel(
            **eleven(
                fallback=["piper"],
                piper_model=str(model),
                piper_model_license="CC-BY-4.0",
                piper_model_source="https://example.org/voice",
                piper_model_attribution="Voice by A. Reader",
            )
        ),
        keyed(),
    )
    piper = chain[1].provider
    assert isinstance(piper, PiperProvider)
    assert piper.model_license == "CC-BY-4.0"
    assert piper.model_source == "https://example.org/voice"
    assert piper.model_attribution == "Voice by A. Reader"


def test_piper_undeclared_licence_is_none(tmp_path: Path) -> None:
    model = piper_model(tmp_path / "m.onnx")
    chain = build_voice_chain(
        channel(**eleven(fallback=["piper"], piper_model=str(model))), keyed()
    )
    piper = chain[1].provider
    assert isinstance(piper, PiperProvider)
    assert (piper.model_license, piper.model_source, piper.model_attribution) == (None, None, "")


def test_chain_skips_primary_in_fallback(tmp_path: Path) -> None:
    chain = build_voice_chain(
        channel(provider="edge", voice_id="ar-SA-ZariyahNeural", fallback=["edge"]), Settings()
    )
    assert len(chain) == 1
    assert isinstance(chain[0].provider, RetryingSpeechProvider)
    assert isinstance(chain[0].provider.inner, EdgeProvider)
    assert chain[0].voice_id == "ar-SA-ZariyahNeural"
    model = piper_model(tmp_path / "m.onnx")
    piper = build_voice_chain(
        channel(provider="piper", voice_id="x", piper_model=str(model), fallback=["piper", "edge"]),
        Settings(),
    )
    assert [c.provider.name for c in piper] == ["piper", "edge"]
    assert piper[0].voice_id == "piper"
    assert build_voice_chain(channel(**eleven(fallback=[])), keyed())[0].voice_id == VOICE_ID


def test_chain_missing_voice_key() -> None:
    with pytest.raises(MissingApiKeyError, match="NARRATOFLOW_VOICE_API_KEY"):
        build_voice_chain(channel(**eleven()), Settings())
    with pytest.raises(MissingApiKeyError, match="NARRATOFLOW_VOICE_API_KEY"):
        build_voice_chain(channel(**eleven()), Settings(voice_api_key=SecretStr("")))


def test_chain_unknown_provider() -> None:
    with pytest.raises(ChannelConfigError, match="azure"):
        build_voice_chain(channel(provider="azure", voice_id="v"), keyed())


def test_chain_provider_case_insensitive() -> None:
    chain = build_voice_chain(channel(**eleven(provider=" ElevenLabs ")), keyed())
    assert isinstance(chain[0].provider, RetryingSpeechProvider)
    assert isinstance(chain[0].provider.inner, ElevenLabsProvider)
    edge = build_voice_chain(channel(provider="EDGE", voice_id="ar-SA-HamedNeural"), Settings())
    assert isinstance(edge[0].provider, RetryingSpeechProvider)
    assert isinstance(edge[0].provider.inner, EdgeProvider)


def test_chain_bad_elevenlabs_voice_id_and_missing_piper_model_fail_up_front(
    tmp_path: Path,
) -> None:
    for bad in ("short", VOICE_ID + "/../x", VOICE_ID + "\n"):
        with pytest.raises(ChannelConfigError, match="voice id"):
            build_voice_chain(channel(**eleven(voice_id=bad)), keyed())
    missing = str(tmp_path / "gone.onnx")
    with pytest.raises(ChannelConfigError, match=r"gone\.onnx"):
        build_voice_chain(channel(**eleven(fallback=["piper"], piper_model=missing)), keyed())
    with pytest.raises(ChannelConfigError, match=r"gone\.onnx"):
        build_voice_chain(channel(provider="piper", voice_id="x", piper_model=missing), Settings())


class FakeSpeech:
    name = "elevenlabs"
    model = "eleven_multilingual_v2"

    def __init__(
        self, billed: int | None = None, fail: bool = False, billed_fail: bool = False
    ) -> None:
        self.calls = 0
        self.billed = billed
        self.fail = fail
        self.billed_fail = billed_fail

    def synthesize(self, req: SpeechRequest) -> Speech:
        self.calls += 1
        if self.billed_fail:
            raise BilledProviderError("ElevenLabs audio payload is not valid base64")
        if self.fail:
            raise ProviderError("ElevenLabs returned HTTP 500")
        return Speech(b"mp3", (), self.name, self.model, self.billed)


REQ = SpeechRequest(text="x" * 2000, voice_id=VOICE_ID, language=Language.ARABIC)


def test_tracked_blocks_before_call() -> None:
    fake = FakeSpeech()
    tracked = TrackedSpeechProvider(fake, SpendTracker(Decimal("0.59")), PRICES)
    with pytest.raises(BudgetExceededError):
        tracked.synthesize(REQ)  # 2000 chars x 0.30 / 1000 = 0.60
    assert fake.calls == 0
    assert (tracked.name, tracked.model) == ("elevenlabs", "eleven_multilingual_v2")


def test_tracked_charges_billed_chars_else_len_and_marks_estimated() -> None:
    tracker = SpendTracker(Decimal(10))
    billed = TrackedSpeechProvider(FakeSpeech(billed=1000), tracker, PRICES)
    billed.synthesize(REQ)
    assert tracker.spent == Decimal("0.30")
    assert billed.cost_estimated is False
    estimated = TrackedSpeechProvider(FakeSpeech(billed=None), tracker, PRICES)
    estimated.synthesize(REQ)
    assert tracker.spent == Decimal("0.90")
    assert estimated.cost_estimated is True
    assert [e.what for e in tracker.entries] == ["voice elevenlabs/eleven_multilingual_v2"] * 2


def test_tracked_failure_charges_nothing() -> None:
    tracker = SpendTracker(Decimal(10))
    tracked = TrackedSpeechProvider(FakeSpeech(fail=True), tracker, PRICES)
    with pytest.raises(ProviderError):
        tracked.synthesize(REQ)
    assert tracker.spent == 0
    assert tracker.entries == []
    assert tracked.cost_estimated is False


def test_free_voice_costs_zero() -> None:
    empty = PriceTable({})
    assert estimate_speech_cost(empty, "edge", "edge-tts", 5000) == 0
    assert estimate_speech_cost(empty, "Piper", "ar_JO-kareem-medium", 5000) == 0
    assert estimate_speech_cost(PRICES, "elevenlabs", "eleven_multilingual_v2", 1500) == Decimal(
        "0.45"
    )
    free = FakeSpeech()
    free.name, free.model = "edge", "edge-tts"
    tracker = SpendTracker(Decimal("0.0001"))
    tracked = TrackedSpeechProvider(free, tracker, empty)
    tracked.synthesize(REQ)
    assert free.calls == 1
    assert tracker.spent == 0


def test_free_fallbacks_need_no_prices() -> None:
    table = PriceTable(
        {
            "llm": {"ollama/qwen2.5": (Decimal(0), Decimal(0))},
            "image": {"openai/gpt-image-1": (Decimal("0.04"),)},
        }
    )
    ch = channel(provider="edge", voice_id="ar-SA-HamedNeural", fallback=["edge"])
    no_vision = ch.model_copy(update={"vision": ch.vision.model_copy(update={"provider": "none"})})
    require_prices(no_vision, table)  # no voice entry needed for edge or its fallbacks


def test_tracked_billed_failure_charges_estimate_and_reraises() -> None:
    tracker = SpendTracker(Decimal(10))
    tracked = TrackedSpeechProvider(FakeSpeech(billed_fail=True), tracker, PRICES)
    with pytest.raises(BilledProviderError):
        tracked.synthesize(REQ)
    assert tracker.spent == Decimal("0.60")
    assert [e.what for e in tracker.entries] == ["voice elevenlabs/eleven_multilingual_v2"]
    assert tracked.cost_estimated is True


def test_tracked_warns_when_charge_exceeds_estimate(caplog: pytest.LogCaptureFixture) -> None:
    tracker = SpendTracker(Decimal(10))
    tracked = TrackedSpeechProvider(FakeSpeech(billed=3000), tracker, PRICES)
    with caplog.at_level("WARNING"):
        tracked.synthesize(REQ)
    assert tracker.spent == Decimal("0.90")
    assert "exceeds the pre-call estimate" in caplog.text
    assert "xxxx" not in caplog.text
    assert "xi-key" not in caplog.text


def test_tracked_no_warning_within_estimate(caplog: pytest.LogCaptureFixture) -> None:
    tracked = TrackedSpeechProvider(FakeSpeech(billed=1000), SpendTracker(None), PRICES)
    with caplog.at_level("WARNING"):
        tracked.synthesize(REQ)
    assert caplog.text == ""


@pytest.mark.parametrize(
    "name",
    [
        "ar-SA-HamedNeural",
        "en-US-AriaNeural",
        "en-US-AndrewMultilingualNeural",
        "zh-CN-liaoning-XiaobeiNeural",
        "iu-Latn-CA-SiqiniqNeural",
        "fil-PH-AngeloNeural",
    ],
)
def test_edge_voice_name_accepted(name: str) -> None:
    assert (
        build_voice_chain(channel(provider="edge", voice_id=name), Settings())[0].voice_id == name
    )
    chain = build_voice_chain(channel(**eleven(edge_voice=name)), keyed())
    assert chain[1].voice_id == name


@pytest.mark.parametrize(
    "name",
    ["", "HamedNeural", "ar-SA-Hamed", "ar-sa-HamedNeural", "ar-SA-HamedNeural\n", "../x"],
)
def test_edge_voice_name_rejected_up_front(name: str) -> None:
    with pytest.raises(ChannelConfigError, match="edge voice"):
        build_voice_chain(channel(provider="edge", voice_id=name), Settings())
    with pytest.raises(ChannelConfigError, match="edge voice"):
        build_voice_chain(channel(**eleven(edge_voice=name)), keyed())


def test_piper_model_config_required_up_front(tmp_path: Path) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"onnx")
    with pytest.raises(ChannelConfigError, match=r"m\.onnx\.json"):
        build_voice_chain(channel(**eleven(fallback=["piper"], piper_model=str(model))), keyed())


def test_built_elevenlabs_voice_closes_its_http_client() -> None:
    choice = build_voice_chain(channel("en", **eleven(fallback=[])), keyed())[0]
    assert isinstance(choice.provider, RetryingSpeechProvider)
    inner = choice.provider.inner
    assert isinstance(inner, ElevenLabsProvider)
    close = getattr(choice.provider, "close", None)  # the cleanup audio/build.py registers
    assert callable(close)
    assert not inner.http.is_closed
    close()
    assert inner.http.is_closed
