"""Build the channel's voice chain and wrap paid voices so every call is budget-checked."""

from __future__ import annotations

import logging
from pathlib import Path

from narratoflow.channels.config import ChannelConfig
from narratoflow.costs.prices import PriceTable
from narratoflow.costs.tracker import SpendTracker, estimate_speech_cost
from narratoflow.errors import BilledProviderError, ChannelConfigError, MissingApiKeyError
from narratoflow.settings import Settings
from narratoflow.tts import Speech, SpeechProvider, SpeechRequest, VoiceChoice
from narratoflow.tts.edge import DEFAULT_EDGE_VOICES, EDGE_VOICE_RE, EdgeProvider
from narratoflow.tts.elevenlabs import ELEVENLABS_VOICE_ID_RE, ElevenLabsProvider
from narratoflow.tts.piper import PiperProvider

logger = logging.getLogger(__name__)
KEY_ENV_VAR = "NARRATOFLOW_VOICE_API_KEY"
PIPER_VOICE_ID = "piper"


class TrackedSpeechProvider:
    """Wraps a provider: checks the budget before each call, charges once after success.

    The charge uses the provider's `billed_chars`, else `len(text)`; in the latter
    case `cost_estimated` becomes True (ElevenLabs reports no billed count).
    A `BilledProviderError` (the provider billed an unusable response) charges the
    pre-call estimate, marked estimated, and is re-raised; any other failure charges
    nothing and is re-raised.
    """

    def __init__(self, inner: SpeechProvider, tracker: SpendTracker, prices: PriceTable) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.tracker = tracker
        self.prices = prices
        self.cost_estimated = False

    def synthesize(self, req: SpeechRequest) -> Speech:
        what = f"voice {self.name}/{self.model}"
        estimate = estimate_speech_cost(self.prices, self.name, self.model, len(req.text))
        self.tracker.ensure_room(estimate, what)
        try:
            speech = self.inner.synthesize(req)
        except BilledProviderError:
            self.cost_estimated = True
            self.tracker.charge(estimate, what)
            raise
        chars = speech.billed_chars
        if not chars:
            chars = len(req.text)
            self.cost_estimated = True
        cost = estimate_speech_cost(self.prices, self.name, self.model, chars)
        if cost > estimate:
            logger.warning("%s charge %s exceeds the pre-call estimate %s", what, cost, estimate)
        self.tracker.charge(cost, what)
        return speech


def _piper(channel: ChannelConfig) -> VoiceChoice:
    model = channel.voice.piper_model
    if model is None or not model.is_file():
        raise ChannelConfigError(f"piper model not found: {model}")
    config = Path(f"{model}.json")
    if not config.is_file():
        raise ChannelConfigError(f"piper model config not found: {config}")
    return VoiceChoice(PiperProvider(model), PIPER_VOICE_ID)


def _edge(voice_id: str) -> VoiceChoice:
    if not EDGE_VOICE_RE.fullmatch(voice_id):
        raise ChannelConfigError(
            f"edge voice {voice_id!r} is not a voice short name like ar-SA-HamedNeural"
        )
    return VoiceChoice(EdgeProvider(), voice_id)


def _primary(channel: ChannelConfig, settings: Settings) -> VoiceChoice:
    voice = channel.voice
    provider = voice.provider  # VoiceConfig strips and lowercases it
    if provider == "elevenlabs":
        key = settings.voice_api_key
        if key is None or not key.get_secret_value():
            raise MissingApiKeyError(f"Missing API key environment variable(s): {KEY_ENV_VAR}")
        if not ELEVENLABS_VOICE_ID_RE.fullmatch(voice.voice_id):
            raise ChannelConfigError(
                "ElevenLabs voice id must be 8-64 letters and digits (check [voice] voice_id)"
            )
        return VoiceChoice(ElevenLabsProvider(key, voice.model), voice.voice_id)
    if provider == "edge":
        return _edge(voice.voice_id)
    if provider == "piper":
        return _piper(channel)
    raise ChannelConfigError(f"unsupported voice provider {voice.provider!r}")


def build_voice_chain(channel: ChannelConfig, settings: Settings) -> list[VoiceChoice]:
    """The primary voice, then each `voice.fallback` entry not already in the chain.

    Validates up front (before any spend): the voice key, the ElevenLabs voice id,
    every edge voice name and every Piper model file with its `<model>.json` config.
    Imports no optional extra. The providers are untracked; callers wrap them in
    `TrackedSpeechProvider` before any call.
    """
    chain = [_primary(channel, settings)]
    seen = {chain[0].provider.name}
    for name in channel.voice.fallback:
        if name in seen:
            continue
        seen.add(name)
        if name == "edge":
            edge_voice = channel.voice.edge_voice
            voice_id = DEFAULT_EDGE_VOICES[channel.language] if edge_voice is None else edge_voice
            chain.append(_edge(voice_id))
        else:
            chain.append(_piper(channel))
    return chain
