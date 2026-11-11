"""OpenAI gpt-image-1 adapter (generations and edits).

API verified 2026-11-10 against
https://developers.openai.com/api/reference/resources/images/methods/edit
(platform.openai.com returned HTTP 403 to the fetch tool):
- POST /v1/images/generations takes JSON; POST /v1/images/edits takes multipart.
- The multipart input image field is ``image[]`` (the plain ``image`` field is
  not documented).
- GPT image models always return ``b64_json``; ``response_format`` is not
  allowed, so it is not sent.
- Sizes for GPT image models: 1024x1024, 1536x1024, 1024x1536, auto. Only
  1024x1024 is requested here.
The request ``seed`` is not supported by this API and is ignored.
"""

from __future__ import annotations

import base64
import binascii
import io
from typing import Any

import httpx
from PIL import Image, UnidentifiedImageError
from pydantic import SecretStr

from narratoflow.errors import ProviderError
from narratoflow.images import GeneratedImage, ImageRequest

BASE_URL = "https://api.openai.com/v1"
TIMEOUT_S = 180.0


class OpenAIImageProvider:
    name: str = "openai"

    def __init__(self, api_key: SecretStr, model: str, http: httpx.Client | None = None) -> None:
        self._api_key = api_key
        self.model = model
        self.http = http if http is not None else httpx.Client(timeout=TIMEOUT_S)

    def _scrub(self, text: str) -> str:
        secret = self._api_key.get_secret_value()
        return text.replace(secret, "***") if secret else text

    def generate(self, req: ImageRequest) -> GeneratedImage:
        headers = {"Authorization": f"Bearer {self._api_key.get_secret_value()}"}
        fields = {"model": self.model, "prompt": req.prompt, "size": req.size}
        try:
            if req.reference_png is not None:
                response = self.http.post(
                    f"{BASE_URL}/images/edits",
                    headers=headers,
                    data=fields,
                    files={"image[]": ("reference.png", req.reference_png, "image/png")},
                )
            else:
                response = self.http.post(
                    f"{BASE_URL}/images/generations", headers=headers, json={**fields, "n": 1}
                )
        except httpx.TimeoutException as e:
            raise ProviderError(f"OpenAI image request timed out ({type(e).__name__})") from e
        except httpx.HTTPError as e:
            raise ProviderError(f"Cannot reach OpenAI images: {self._scrub(str(e))}") from e
        if not response.is_success:
            raise ProviderError(
                f"OpenAI returned HTTP {response.status_code} for model {self.model!r}: "
                f"{self._scrub(response.text[:200])}"
            )
        try:
            body: Any = response.json()
        except ValueError as e:
            raise ProviderError("OpenAI response body is not valid JSON") from e
        data = body.get("data") if isinstance(body, dict) else None
        item = data[0] if isinstance(data, list) and data else None
        encoded = item.get("b64_json") if isinstance(item, dict) else None
        if not isinstance(encoded, str):
            raise ProviderError("OpenAI response has no data[0].b64_json")
        revised = item.get("revised_prompt") if isinstance(item, dict) else None
        return GeneratedImage(
            png=_to_png(encoded),
            provider=self.name,
            model=self.model,
            revised_prompt=revised if isinstance(revised, str) else None,
        )


def _to_png(encoded: str) -> bytes:
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ProviderError("OpenAI image payload is not valid base64") from e
    try:
        with Image.open(io.BytesIO(raw)) as img:
            img.load()
            if img.format == "PNG":
                return raw
            out = io.BytesIO()
            img.save(out, format="PNG")
            return out.getvalue()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ProviderError("OpenAI payload is not a readable image") from e
