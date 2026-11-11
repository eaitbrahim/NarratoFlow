"""Ollama client using structured JSON output (format=schema)."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from narratoflow.errors import ProviderError
from narratoflow.llm.base import StructuredResult, Usage

DEFAULT_TIMEOUT_S = 120.0


class OllamaClient:
    """Free local provider: POST /api/chat with a JSON schema in `format`."""

    name: str = "ollama"

    def __init__(self, base_url: str, model: str, http: httpx.Client | None = None) -> None:
        self.url = f"{base_url.rstrip('/')}/api/chat"
        self.model = model
        self.http = http if http is not None else httpx.Client(timeout=DEFAULT_TIMEOUT_S)

    def complete_structured[T: BaseModel](
        self, *, system: str, user: str, output: type[T]
    ) -> StructuredResult[T]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": output.model_json_schema(),
            "stream": False,
        }
        try:
            response = self.http.post(self.url, json=payload)
        except httpx.HTTPError as e:
            raise ProviderError(f"Cannot reach Ollama at {self.url}: {e}") from e
        if not response.is_success:
            raise ProviderError(
                f"Ollama returned HTTP {response.status_code} for model {self.model!r}: "
                f"{response.text[:200]}"
            )
        try:
            body: Any = response.json()
        except ValueError as e:
            raise ProviderError("Ollama response body is not valid JSON") from e
        message = body.get("message") if isinstance(body, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ProviderError("Ollama response has no message content")
        try:
            value = output.model_validate_json(content)
        except ValidationError as e:
            raise ProviderError(
                f"Ollama output did not match {output.__name__}: {e.error_count()} error(s)"
            ) from e
        usage = Usage(
            input_tokens=int(body.get("prompt_eval_count") or 0),
            output_tokens=int(body.get("eval_count") or 0),
        )
        return StructuredResult(
            value=value, usage=usage, model=str(body.get("model") or self.model)
        )
