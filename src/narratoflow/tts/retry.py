"""Retry transient voice failures with backoff, inside one voice."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from narratoflow.errors import TransientProviderError
from narratoflow.tts import Speech, SpeechProvider, SpeechRequest

logger = logging.getLogger(__name__)
RETRY_ATTEMPTS = 3
RETRY_DELAYS_S = (1.0, 4.0)


class RetryingSpeechProvider:
    """Retries `TransientProviderError` only, up to `RETRY_ATTEMPTS` calls.

    Any other error (including `BilledProviderError`, whose request may already be
    billed) propagates at once. The last transient error is re-raised, so the caller's
    voice fallback still applies.
    """

    def __init__(
        self, inner: SpeechProvider, *, sleep: Callable[[float], None] = time.sleep
    ) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self._sleep = sleep

    def synthesize(self, req: SpeechRequest) -> Speech:
        attempt = 1
        while True:
            try:
                return self.inner.synthesize(req)
            except TransientProviderError:
                if attempt >= RETRY_ATTEMPTS:
                    raise
                delay = RETRY_DELAYS_S[attempt - 1]
                logger.warning(
                    "voice %s failed transiently (attempt %d of %d); retrying in %g s",
                    self.name,
                    attempt,
                    RETRY_ATTEMPTS,
                    delay,
                )
                self._sleep(delay)
                attempt += 1
