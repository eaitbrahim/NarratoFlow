"""A fake TTS voice: audio made of speech-like bursts, optional word spans, scripted failures."""

from __future__ import annotations

from fixtures.audio import speechlike, to_wav_bytes

from narratoflow.audio.words import TimedSpan
from narratoflow.errors import ProviderError
from narratoflow.tts import Speech, SpeechRequest

WORD_MS = 300
GAP_MS = 80
PLAIN_LEAD_MS = 150
TIMED_LEAD_MS = 200


def word_spans(
    text: str,
    lead_ms: int = TIMED_LEAD_MS,
    word_ms: int = WORD_MS,
    gap_ms: int = GAP_MS,
) -> tuple[TimedSpan, ...]:
    """One span per word of `text`, `word_ms` long, `gap_ms` apart, starting at `lead_ms`."""
    out = []
    t = lead_ms
    for w in text.split():
        out.append(TimedSpan(w, t, t + word_ms))
        t += word_ms + gap_ms
    return tuple(out)


class FakeVoice:
    """A `VoiceProvider`: the plain profile has no spans; `spans=True` adds word spans.

    It fails for every request whose text contains `fail_on` ("" matches all).
    """

    def __init__(
        self,
        name: str = "paid",
        model: str = "m1",
        *,
        spans: bool = False,
        fail_on: str | None = None,
        error: Exception | None = None,
        billed: bool = True,
        word_ms: int = WORD_MS,
        gap_ms: int = GAP_MS,
        lead_ms: int = PLAIN_LEAD_MS,
    ) -> None:
        self.name = name
        self.model = model
        self.spans = spans
        self.fail_on = fail_on
        self.error = error if error is not None else ProviderError("HTTP 500")
        self.billed = billed
        self.word_ms = word_ms
        self.gap_ms = gap_ms
        self.lead_ms = lead_ms
        self.requests: list[SpeechRequest] = []

    def synthesize(self, req: SpeechRequest) -> Speech:
        self.requests.append(req)
        if self.fail_on is not None and self.fail_on in req.text:
            raise self.error
        n = len(req.text.split())
        audio = to_wav_bytes(speechlike(n, self.word_ms, self.gap_ms, lead_ms=self.lead_ms))
        spans = word_spans(req.text, self.lead_ms, self.word_ms, self.gap_ms) if self.spans else ()
        return Speech(
            audio=audio,
            spans=spans,
            provider=self.name,
            model=self.model,
            billed_chars=len(req.text) if self.billed else None,
        )
