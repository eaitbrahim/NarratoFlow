import base64
import json
from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from narratoflow.errors import BilledProviderError, ChannelConfigError, ProviderError
from narratoflow.models import Language
from narratoflow.tts import SpeechRequest
from narratoflow.tts.elevenlabs import ELEVENLABS_VOICE_ID_RE, ElevenLabsProvider

KEY = "xi-SUPERSECRET-0xDEADBEEF"
VOICE = "21m00Tcm4TlvDq8ikWAM"
Handler = Callable[[httpx.Request], httpx.Response]
AUDIO = b"ID3fake-mp3"


def body(**over: object) -> dict[str, object]:
    out: dict[str, object] = {
        "audio_base64": base64.b64encode(AUDIO).decode(),
        "alignment": {
            "characters": ["a", "b"],
            "character_start_times_seconds": [0.0, 0.1234],
            "character_end_times_seconds": [0.1234, 0.2],
        },
        "normalized_alignment": None,
    }
    out.update(over)
    return out


def provider(handler: Handler) -> ElevenLabsProvider:
    return ElevenLabsProvider(
        SecretStr(KEY),
        "eleven_multilingual_v2",
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def req(voice: str = VOICE) -> SpeechRequest:
    return SpeechRequest(text="ab", voice_id=voice, language=Language.ENGLISH)


def ok(**over: object) -> Handler:
    return lambda r: httpx.Response(200, json=body(**over))


def test_request_shape_path_header_body_and_output_format() -> None:
    seen: list[httpx.Request] = []

    def handler(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        return httpx.Response(200, json=body())

    speech = provider(handler).synthesize(req())
    r = seen[0]
    assert r.method == "POST"
    assert r.url.path == f"/v1/text-to-speech/{VOICE}/with-timestamps"
    assert r.url.host == "api.elevenlabs.io"
    assert dict(r.url.params) == {"output_format": "mp3_44100_128"}
    assert r.headers["xi-api-key"] == KEY
    assert json.loads(r.content) == {"text": "ab", "model_id": "eleven_multilingual_v2"}
    assert r.extensions["timeout"]["read"] == 120.0
    assert speech.audio == AUDIO
    assert (speech.provider, speech.model) == ("elevenlabs", "eleven_multilingual_v2")


def test_character_alignment_becomes_spans_in_ms() -> None:
    spans = provider(ok()).synthesize(req()).spans
    assert [(s.text, s.start_ms, s.end_ms) for s in spans] == [("a", 0, 123), ("b", 123, 200)]


def test_uses_alignment_not_normalized_alignment() -> None:
    norm = {
        "characters": ["x"],
        "character_start_times_seconds": [5.0],
        "character_end_times_seconds": [6.0],
    }
    spans = provider(ok(normalized_alignment=norm)).synthesize(req()).spans
    assert [s.text for s in spans] == ["a", "b"]


def test_unequal_alignment_arrays_raise() -> None:
    bad = {
        "characters": ["a", "b"],
        "character_start_times_seconds": [0.0],
        "character_end_times_seconds": [0.1, 0.2],
    }
    with pytest.raises(ProviderError, match="alignment"):
        provider(ok(alignment=bad)).synthesize(req())


def test_null_alignment_gives_empty_spans() -> None:
    assert provider(ok(alignment=None)).synthesize(req()).spans == ()


@pytest.mark.parametrize("audio", [None, "", "!!!notbase64", 5])
def test_bad_audio_raises(audio: object) -> None:
    with pytest.raises(ProviderError):
        provider(ok(audio_base64=audio)).synthesize(req())


def test_billed_chars_is_none_because_no_documented_header() -> None:
    h: Handler = lambda r: httpx.Response(200, json=body(), headers={"x-character-count": "2"})  # noqa: E731
    assert provider(h).synthesize(req()).billed_chars is None
    assert provider(ok()).synthesize(req()).billed_chars is None


def test_401_does_not_echo_key() -> None:
    h: Handler = lambda r: httpx.Response(401, json={"detail": {"status": KEY}})  # noqa: E731
    with pytest.raises(ProviderError) as ei:
        provider(h).synthesize(req())
    assert str(ei.value) == "ElevenLabs rejected the API key (HTTP 401)"
    assert KEY not in str(ei.value)


def test_429_message() -> None:
    h: Handler = lambda r: httpx.Response(429, text="slow down")  # noqa: E731
    with pytest.raises(ProviderError) as ei:
        provider(h).synthesize(req())
    assert str(ei.value) == "ElevenLabs rate limit (HTTP 429)"


def test_other_error_carries_status_and_scrubbed_detail_status() -> None:
    h: Handler = lambda r: httpx.Response(  # noqa: E731
        422, json={"detail": {"status": f"bad_{KEY}", "message": "secret text ab"}}
    )
    with pytest.raises(ProviderError) as ei:
        provider(h).synthesize(req())
    msg = str(ei.value)
    assert "HTTP 422" in msg and "bad_***" in msg
    assert KEY not in msg and "secret text" not in msg


def test_timeout_maps_to_provider_error() -> None:
    def h(r: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=r)

    with pytest.raises(ProviderError, match="timed out"):
        provider(h).synthesize(req())


def test_connect_error_maps_to_provider_error() -> None:
    def h(r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"down {KEY}", request=r)

    with pytest.raises(ProviderError) as ei:
        provider(h).synthesize(req())
    assert KEY not in str(ei.value)


@pytest.mark.parametrize(
    "voice", ["", "short", "../../x/y/z/abc", "a" * 65, "abc def ghi", VOICE + "/"]
)
def test_bad_voice_id_rejected_before_request(voice: str) -> None:
    calls: list[int] = []

    def h(r: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, json=body())

    with pytest.raises(ChannelConfigError):
        provider(h).synthesize(req(voice))
    assert calls == []


def test_voice_id_regex_accepts_valid() -> None:
    assert ELEVENLABS_VOICE_ID_RE.fullmatch(VOICE)


@pytest.mark.parametrize(("start", "end"), [(-0.1, 0.2), (0.5, 0.2), (0.0, 1e300), (0.0, 3600.5)])
def test_out_of_range_times_raise(start: float, end: float) -> None:
    bad = {
        "characters": ["a"],
        "character_start_times_seconds": [start],
        "character_end_times_seconds": [end],
    }
    with pytest.raises(ProviderError, match="alignment"):
        provider(ok(alignment=bad)).synthesize(req())


def test_503_message_has_no_body_text() -> None:
    h: Handler = lambda r: httpx.Response(503, text="upstream says ab")  # noqa: E731
    with pytest.raises(ProviderError) as ei:
        provider(h).synthesize(req())
    assert "HTTP 503" in str(ei.value) and "upstream" not in str(ei.value)


def test_non_json_200_body_raises_without_body_text() -> None:
    h: Handler = lambda r: httpx.Response(200, text="<html>secret ab</html>")  # noqa: E731
    with pytest.raises(ProviderError) as ei:
        provider(h).synthesize(req())
    assert "secret" not in str(ei.value)


@pytest.mark.parametrize(
    "handler",
    [
        lambda r: httpx.Response(200, content=b"not json"),
        lambda r: httpx.Response(200, json=[1, 2]),
        lambda r: httpx.Response(200, json=body(audio_base64="!!!")),
        lambda r: httpx.Response(200, json=body(audio_base64=None)),
        lambda r: httpx.Response(200, json=body(alignment={"characters": ["a"]})),
        lambda r: httpx.Response(
            200,
            json=body(
                alignment={
                    "characters": ["a"],
                    "character_start_times_seconds": [-1.0],
                    "character_end_times_seconds": [0.1],
                }
            ),
        ),
    ],
)
def test_failures_after_2xx_are_billed(handler: Handler) -> None:
    with pytest.raises(BilledProviderError):
        provider(handler).synthesize(req())


@pytest.mark.parametrize("status", [401, 429, 500, 503])
def test_http_errors_are_not_billed(status: int) -> None:
    with pytest.raises(ProviderError) as info:
        provider(lambda r: httpx.Response(status, json={})).synthesize(req())
    assert not isinstance(info.value, BilledProviderError)


def test_transport_errors_are_not_billed() -> None:
    def boom(r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(ProviderError) as info:
        provider(boom).synthesize(req())
    assert not isinstance(info.value, BilledProviderError)
