import base64
import io
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from PIL import Image
from pydantic import SecretStr

import narratoflow.images as images_module
import narratoflow.images.openai_images as openai_module
import narratoflow.images.prompt as prompt_module
from narratoflow.channels.config import StyleConfig
from narratoflow.errors import BilledProviderError, ProviderError
from narratoflow.images import GeneratedImage, ImageProvider, ImageRequest
from narratoflow.images.openai_images import OpenAIImageProvider
from narratoflow.images.prompt import NEGATIVE_TERMS, build_prompt

FAKE_KEY = "sk-SUPERSECRET-0xDEADBEEF"
Handler = Callable[[httpx.Request], httpx.Response]


def png_bytes(fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buf, format=fmt)
    return buf.getvalue()


def ok_body(raw: bytes | None = None, **extra: object) -> dict[str, object]:
    data = base64.b64encode(raw if raw is not None else png_bytes()).decode()
    return {"data": [{"b64_json": data, **extra}]}


def provider(handler: Handler) -> OpenAIImageProvider:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAIImageProvider(SecretStr(FAKE_KEY), "gpt-image-1", http=http)


def style() -> StyleConfig:
    return StyleConfig(
        reference_sketch=Path("ref.png"),
        instructions="Hand-drawn whiteboard marker style.",
        image_provider="openai",
        image_model="gpt-image-1",
        line_weight_px=3.0,
        ink="#111111",
    )


def test_prompt_contains_every_negative_term_and_style() -> None:
    prompt = build_prompt(style(), "scale|justice", "a balance weighing a heart")
    assert "Hand-drawn whiteboard marker style." in prompt
    assert "Subject: a balance weighing a heart" in prompt
    for term in NEGATIVE_TERMS:
        assert term in prompt
    sentence = next(s for s in prompt.split(". ") if s.startswith("No "))
    assert "people, faces, human figures, text, letters, logos" in sentence
    assert all(term in sentence for term in NEGATIVE_TERMS)


def test_prompt_empty_description_uses_key() -> None:
    prompt = build_prompt(style(), "scale|justice", "  ")
    assert "Subject: scale, justice" in prompt


def test_prompt_is_deterministic() -> None:
    a = build_prompt(style(), "scale|justice", "x")
    assert a == build_prompt(style(), "scale|justice", "x")


def test_generation_request_shape_without_reference() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ok_body(revised_prompt="rp"))

    result = provider(handler).generate(ImageRequest(prompt="draw", reference_png=None))
    req = seen[0]
    assert req.method == "POST"
    assert str(req.url) == "https://api.openai.com/v1/images/generations"
    assert req.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert json.loads(req.content) == {
        "model": "gpt-image-1",
        "prompt": "draw",
        "size": "1024x1024",
        "n": 1,
    }
    assert isinstance(result, GeneratedImage)
    assert result.provider == "openai"
    assert result.model == "gpt-image-1"
    assert result.revised_prompt == "rp"
    assert Image.open(io.BytesIO(result.png)).format == "PNG"


def test_edit_request_is_multipart_with_reference() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ok_body())

    ref = png_bytes()
    result = provider(handler).generate(ImageRequest(prompt="draw", reference_png=ref))
    req = seen[0]
    assert str(req.url) == "https://api.openai.com/v1/images/edits"
    assert req.headers["content-type"].startswith("multipart/form-data")
    body = req.content
    assert b'name="image[]"' in body
    assert b'filename="reference.png"' in body
    assert b"Content-Type: image/png" in body
    assert ref in body
    for field in (b'name="prompt"', b'name="model"', b'name="size"'):
        assert field in body
    assert result.revised_prompt is None


def test_jpeg_is_reencoded_to_png() -> None:
    p = provider(lambda r: httpx.Response(200, json=ok_body(png_bytes("JPEG"))))
    out = p.generate(ImageRequest(prompt="x", reference_png=None))
    assert out.png.startswith(b"\x89PNG")


def test_bad_b64_raises_provider_error() -> None:
    body = {"data": [{"b64_json": "!!!not base64!!!"}]}
    with pytest.raises(ProviderError, match="base64"):
        provider(lambda r: httpx.Response(200, json=body)).generate(
            ImageRequest(prompt="x", reference_png=None)
        )


def test_non_image_payload_raises() -> None:
    p = provider(lambda r: httpx.Response(200, json=ok_body(b"not an image")))
    with pytest.raises(ProviderError, match="image"):
        p.generate(ImageRequest(prompt="x", reference_png=None))


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"<html>"),
        httpx.Response(200, json=[1]),
        httpx.Response(200, json={"data": []}),
        httpx.Response(200, json={"data": [{}]}),
        httpx.Response(200, json={"data": [{"b64_json": 5}]}),
    ],
)
def test_malformed_bodies_raise(response: httpx.Response) -> None:
    with pytest.raises(ProviderError):
        provider(lambda r: response).generate(ImageRequest(prompt="x", reference_png=None))


def test_401_does_not_echo_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad key {FAKE_KEY}")

    with pytest.raises(ProviderError) as info:
        provider(handler).generate(ImageRequest(prompt="x", reference_png=None))
    assert "401" in str(info.value)
    assert FAKE_KEY not in str(info.value)


def test_error_bodies_never_echoed() -> None:
    masked = "sk-proj-****abcd"
    body = {"error": {"message": f"key {masked} prompt SECRETPROMPT", "code": "bad_x"}}
    with pytest.raises(ProviderError) as info:
        provider(lambda r: httpx.Response(400, json=body)).generate(
            ImageRequest(prompt="SECRETPROMPT", reference_png=None)
        )
    msg = str(info.value)
    assert "400" in msg
    assert "bad_x" in msg
    assert masked not in msg
    assert "SECRETPROMPT" not in msg
    with pytest.raises(ProviderError) as info2:
        provider(lambda r: httpx.Response(401, text=f"{masked} SECRETPROMPT")).generate(
            ImageRequest(prompt="x", reference_png=None)
        )
    assert str(info2.value) == "OpenAI rejected the API key (HTTP 401)"
    with pytest.raises(ProviderError) as info3:
        provider(lambda r: httpx.Response(500, text="oops SECRETPROMPT")).generate(
            ImageRequest(prompt="x", reference_png=None)
        )
    assert "SECRETPROMPT" not in str(info3.value)
    assert "500" in str(info3.value)


def test_decompression_bomb_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    p = provider(lambda r: httpx.Response(200, json=ok_body()))
    with pytest.raises(ProviderError, match="readable image"):
        p.generate(ImageRequest(prompt="x", reference_png=None))


def test_prompt_strips_trailing_periods() -> None:
    prompt = build_prompt(style(), "scale|justice", "a balance... ")
    assert "Subject: a balance. Black" in prompt


def test_timeout_maps_to_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ProviderError, match="timed out"):
        provider(handler).generate(ImageRequest(prompt="x", reference_png=None))


def test_connect_error_scrubs_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"boom {FAKE_KEY}", request=request)

    with pytest.raises(ProviderError) as info:
        provider(handler).generate(ImageRequest(prompt="x", reference_png=None))
    assert FAKE_KEY not in str(info.value)


def test_default_client_has_180s_timeout() -> None:
    p = OpenAIImageProvider(SecretStr(FAKE_KEY), "gpt-image-1")
    assert p.http.timeout.read == 180.0


def test_provider_satisfies_protocol() -> None:
    p: ImageProvider = OpenAIImageProvider(SecretStr(FAKE_KEY), "gpt-image-1")
    assert p.name == "openai"


def test_sources_are_ascii() -> None:
    for module in (images_module, openai_module, prompt_module):
        assert Path(module.__file__ or "").read_bytes().isascii()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"<html>"),
        httpx.Response(200, json=[1]),
        httpx.Response(200, json={"data": []}),
        httpx.Response(200, json={"data": [{"b64_json": "!!!"}]}),
        httpx.Response(200, json=ok_body(b"not an image")),
    ],
)
def test_failures_after_2xx_are_billed(response: httpx.Response) -> None:
    with pytest.raises(BilledProviderError):
        provider(lambda r: response).generate(ImageRequest(prompt="x", reference_png=None))


@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_errors_are_not_billed(status: int) -> None:
    with pytest.raises(ProviderError) as info:
        provider(lambda r: httpx.Response(status, json={})).generate(
            ImageRequest(prompt="x", reference_png=None)
        )
    assert not isinstance(info.value, BilledProviderError)
