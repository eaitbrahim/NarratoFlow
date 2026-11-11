"""Piper offline engine (extra: tts-local; free, no key).

API verified 2026-11-11 against the piper-tts 1.8.0 source (``piper/voice.py``):
``PiperVoice.load(model_path, config_path=None, ...)`` (the config defaults to
``<model>.json``) and ``voice.synthesize_wav(text, wav_file: wave.Wave_write, ...)``,
which sets the WAV format itself. Piper gives no word timings, so ``spans`` is empty.
"""

from __future__ import annotations

import io
import wave
from pathlib import Path
from typing import Any

from narratoflow.errors import ChannelConfigError, ExtraMissingError, ProviderError
from narratoflow.tts import Speech, SpeechRequest

EXTRA_HINT = "install the 'tts-local' extra: uv sync --extra tts-local"


class PiperProvider:
    name: str = "piper"

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model = model_path.stem
        self._voice: Any = None

    def _load(self) -> Any:
        if self._voice is not None:
            return self._voice
        if not self.model_path.is_file():
            raise ChannelConfigError(f"piper model not found: {self.model_path}")
        try:
            from piper import PiperVoice
        except ImportError as err:
            raise ExtraMissingError(EXTRA_HINT) from err
        try:
            self._voice = PiperVoice.load(str(self.model_path))
        except Exception as err:
            raise ChannelConfigError(
                f"cannot load piper model {self.model_path.name} ({type(err).__name__})"
            ) from err
        return self._voice

    def synthesize(self, req: SpeechRequest) -> Speech:
        voice = self._load()
        buffer = io.BytesIO()
        try:
            with wave.open(buffer, "wb") as wav:
                voice.synthesize_wav(req.text, wav)
        except Exception as err:
            # Closing a writer whose format was never set raises wave.Error over
            # the library's exception; report the library's.
            cause = err.__context__ if isinstance(err, wave.Error) else None
            failure = cause if isinstance(cause, Exception) else err
            raise ProviderError(f"piper failed: {type(failure).__name__}") from failure
        return Speech(
            audio=buffer.getvalue(),
            spans=(),
            provider=self.name,
            model=self.model,
            billed_chars=None,
        )
