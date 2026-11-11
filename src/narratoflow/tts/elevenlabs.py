"""ElevenLabs text-to-speech adapter using the with-timestamps endpoint.

API verified 2026-11-11 against
https://elevenlabs.io/docs/api-reference/text-to-speech/convert-with-timestamps
- POST /v1/text-to-speech/{voice_id}/with-timestamps, query ``output_format``
  (default mp3_44100_128), JSON body ``text`` (required) and ``model_id``.
- 200 response: ``audio_base64``, ``alignment`` (timing for the original text) and
  ``normalized_alignment``; each holds ``characters``,
  ``character_start_times_seconds`` and ``character_end_times_seconds``.
- The docs document no character-count response header, so ``billed_chars`` is
  always None and the caller estimates the charge from ``len(text)``.
Spans use ``alignment`` (the author's characters), never ``normalized_alignment``.
"""

from __future__ import annotations

import base64
import binascii
import math
import re
from typing import Any

import httpx
from pydantic import SecretStr

from narratoflow.audio.words import TimedSpan
from narratoflow.errors import (
    BilledProviderError,
    ChannelConfigError,
    ProviderAuthError,
    ProviderError,
)
from narratoflow.tts import Speech, SpeechRequest

BASE_URL = "https://api.elevenlabs.io/v1"
OUTPUT_FORMAT = "mp3_44100_128"
TIMEOUT_S = 120.0
MAX_SECONDS = 3600
ELEVENLABS_VOICE_ID_RE = re.compile(r"^[A-Za-z0-9]{8,64}$")


class ElevenLabsProvider:
    name: str = "elevenlabs"

    def __init__(self, api_key: SecretStr, model: str, http: httpx.Client | None = None) -> None:
        self._api_key = api_key
        self.model = model
        self._owns_http = http is None
        self.http = http if http is not None else httpx.Client(timeout=TIMEOUT_S)

    def __enter__(self) -> ElevenLabsProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client this provider created; an injected one is the caller's."""
        if self._owns_http:
            self.http.close()

    def _scrub(self, text: str) -> str:
        secret = self._api_key.get_secret_value()
        return text.replace(secret, "***") if secret else text

    def _http_error(self, response: httpx.Response) -> str:
        status = response.status_code
        if status in (401, 403):
            return f"ElevenLabs rejected the API key (HTTP {status})"
        if status == 429:
            return f"ElevenLabs rate limit (HTTP {status})"
        detail_status = ""
        try:
            body: Any = response.json()
        except ValueError:
            body = None
        detail = body.get("detail") if isinstance(body, dict) else None
        value = detail.get("status") if isinstance(detail, dict) else None
        if isinstance(value, str) and value:
            detail_status = f": {self._scrub(value)[:80]}"
        return f"ElevenLabs returned HTTP {status} for model {self.model!r}{detail_status}"

    def synthesize(self, req: SpeechRequest) -> Speech:
        if not ELEVENLABS_VOICE_ID_RE.fullmatch(req.voice_id):
            raise ChannelConfigError(
                "ElevenLabs voice id must be 8-64 letters and digits (check voices in the channel)"
            )
        url = f"{BASE_URL}/text-to-speech/{req.voice_id}/with-timestamps"
        try:
            response = self.http.post(
                url,
                params={"output_format": OUTPUT_FORMAT},
                headers={"xi-api-key": self._api_key.get_secret_value()},
                json={"text": req.text, "model_id": self.model},
                timeout=TIMEOUT_S,
            )
        except httpx.TimeoutException as e:
            raise ProviderError(f"ElevenLabs request timed out ({type(e).__name__})") from e
        except httpx.HTTPError as e:
            raise ProviderError(f"Cannot reach ElevenLabs: {self._scrub(str(e))}") from e
        if not response.is_success:
            if response.status_code in (401, 403, 404):
                # A rejected key or an unknown voice is a configuration error, not an
                # outage: it must stop the build, never fall back to a free voice.
                raise ProviderAuthError(self._http_error(response))
            raise ProviderError(self._http_error(response))
        try:
            return self._speech(response)
        except ProviderError as e:
            # A 2xx response is billed by ElevenLabs even when it is unusable.
            raise BilledProviderError(str(e)) from e

    def _speech(self, response: httpx.Response) -> Speech:
        try:
            body: Any = response.json()
        except ValueError as e:
            raise ProviderError("ElevenLabs response body is not valid JSON") from e
        if not isinstance(body, dict):
            raise ProviderError("ElevenLabs response is not a JSON object")
        return Speech(
            audio=_audio(body.get("audio_base64")),
            spans=_spans(body.get("alignment")),
            provider=self.name,
            model=self.model,
            billed_chars=None,
        )


def _audio(encoded: object) -> bytes:
    if not isinstance(encoded, str) or not encoded:
        raise ProviderError("ElevenLabs response has no audio_base64")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ProviderError("ElevenLabs audio payload is not valid base64") from e
    if not raw:
        raise ProviderError("ElevenLabs audio payload is empty")
    return raw


def _ms(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ProviderError("ElevenLabs alignment has a non-numeric time")
    if not math.isfinite(value) or abs(value) > MAX_SECONDS * 10:
        raise ProviderError("ElevenLabs alignment times are out of range")
    return round(value * 1000)


def _checked_ms(start: object, end: object) -> tuple[int, int]:
    s, e = _ms(start), _ms(end)
    if not 0 <= s <= e <= MAX_SECONDS * 1000:
        raise ProviderError("ElevenLabs alignment times are out of range")
    return s, e


def _spans(alignment: object) -> tuple[TimedSpan, ...]:
    if alignment is None:
        return ()
    if not isinstance(alignment, dict):
        raise ProviderError("ElevenLabs alignment is malformed")
    chars = alignment.get("characters")
    starts = alignment.get("character_start_times_seconds")
    ends = alignment.get("character_end_times_seconds")
    if not (isinstance(chars, list) and isinstance(starts, list) and isinstance(ends, list)):
        raise ProviderError("ElevenLabs alignment is missing its character arrays")
    if not len(chars) == len(starts) == len(ends):
        raise ProviderError("ElevenLabs alignment arrays have unequal lengths")
    if not all(isinstance(c, str) for c in chars):
        raise ProviderError("ElevenLabs alignment characters are not strings")
    return tuple(
        TimedSpan(text=c, start_ms=ms[0], end_ms=ms[1])
        for c, ms in zip(
            chars, (_checked_ms(s, e) for s, e in zip(starts, ends, strict=True)), strict=True
        )
    )
