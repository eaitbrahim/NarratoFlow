"""Build the channel's voice chain and wrap paid voices so every call is budget-checked."""

from __future__ import annotations

import logging

from narratoflow.channels.config import ChannelConfig
from narratoflow.costs.prices import PriceTable
from narratoflow.costs.tracker import SpendTracker, estimate_speech_cost
from narratoflow.errors import ChannelConfigError, MissingApiKeyError, ProviderError
from narratoflow.settings import Settings
from narratoflow.tts import Speech, SpeechProvider, SpeechRequest, VoiceChoice
from narratoflow.tts.edge import DEFAULT_EDGE_VOICES, EdgeProvider
from narratoflow.tts.elevenlabs import ELEVENLABS_VOICE_ID_RE, ElevenLabsProvider
from narratoflow.tts.piper import PiperProvider

logger = logging.getLogger(__name__)
KEY_ENV_VAR = "NARRATOFLOW_VOICE_API_KEY"
PIPER_VOICE_ID = "piper"


class TrackedSpeechProvider:
    """Wraps a provider: checks the budget before each call, charges once after success.

    The charge uses the provider's `billed_chars`, else `len(text)`; in the latter
    case `cost_estimated` becomes True (ElevenLabs reports no billed count).
    A failed call charges nothing and is re-raised.
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
        except ProviderError as err:
            logger.warning("voice call failed (%s/%s): %s", self.name, self.model, err)
            raise
        chars = speech.billed_chars
        if not chars:
            chars = len(req.text)
            self.cost_estimated = True
        self.tracker.charge(estimate_speech_cost(self.prices, self.name, self.model, chars), what)
        return speech


def _piper(channel: ChannelConfig) -> VoiceChoice:
    model = channel.voice.piper_model
    if model is None or not model.is_file():
        raise ChannelConfigError(f"piper model not found: {model}")
    return VoiceChoice(PiperProvider(model), PIPER_VOICE_ID)


def _primary(channel: ChannelConfig, settings: Settings) -> VoiceChoice:
    voice = channel.voice
    provider = voice.provider.strip().lower()
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
        return VoiceChoice(EdgeProvider(), voice.voice_id)
    if provider == "piper":
        return _piper(channel)
    raise ChannelConfigError(f"unsupported voice provider {voice.provider!r}")


def build_voice_chain(channel: ChannelConfig, settings: Settings) -> list[VoiceChoice]:
    """The primary voice, then each `voice.fallback` entry not already in the chain.

    Validates up front (before any spend): the voice key, the ElevenLabs voice id
    and every Piper model file. Imports no optional extra. The providers are
    untracked; callers wrap them in `TrackedSpeechProvider` before any call.
    """
    chain = [_primary(channel, settings)]
    seen = {chain[0].provider.name}
    for name in channel.voice.fallback:
        if name in seen:
            continue
        seen.add(name)
        if name == "edge":
            voice_id = channel.voice.edge_voice or DEFAULT_EDGE_VOICES[channel.language]
            chain.append(VoiceChoice(EdgeProvider(), voice_id))
        else:
            chain.append(_piper(channel))
    return chain
