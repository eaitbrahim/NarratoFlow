"""Anthropic client: structured output via `messages.parse` (paid default).

Requests go through `client.beta.messages.with_raw_response.parse`. The raw
wrapper lets the client read `stop_reason` from the body *before* the SDK
validates the text against the schema: a refusal can carry partial, non-JSON
text, and `parse` would otherwise raise a schema error instead of a refusal.
The non-beta `messages.with_raw_response` has no `parse` in the installed SDK.
Without `betas`/`fallbacks` the beta surface sends the same request as
`client.messages.parse`.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any, Literal

import anthropic
from anthropic import APIResponse, Omit, omit
from anthropic.types.anthropic_beta_param import AnthropicBetaParam
from anthropic.types.beta.parsed_beta_message import ParsedBetaMessage
from pydantic import BaseModel, SecretStr, ValidationError

from narratoflow.errors import BilledProviderError, ProviderError, ProviderRefusalError
from narratoflow.llm.base import Effort, StructuredResult, Usage

FALLBACK_BETA: AnthropicBetaParam = "server-side-fallback-2026-07-01"
KEY_ENV_VAR = "NARRATOFLOW_LLM_API_KEY"


class _StopDetails(BaseModel):
    category: str | None = None


class _Envelope(BaseModel):
    """The fields read from the raw body before the SDK validates the content."""

    stop_reason: str | None = None
    stop_details: _StopDetails | None = None


class AnthropicClient:
    """Paid default provider.

    With `refusal_fallback`, requests go through the beta surface with the
    `server-side-fallback-2026-07-01` beta and `fallbacks="default"`, so a policy
    decline is retried server-side on the model's default fallback. On models
    without fallback support the API runs the requested model alone.
    """

    name: str = "anthropic"

    def __init__(
        self,
        api_key: SecretStr,
        model: str,
        effort: Effort,
        max_tokens: int,
        refusal_fallback: bool,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self.effort: Effort = effort
        self.max_tokens = max_tokens
        self.refusal_fallback = refusal_fallback
        self.sdk = (
            client
            if client is not None
            else anthropic.Anthropic(api_key=api_key.get_secret_value())
        )
        self._secret = api_key

    def __repr__(self) -> str:
        return f"AnthropicClient(model={self.model!r}, effort={self.effort!r})"

    def complete_structured[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        output: type[T],
        images: Sequence[bytes] = (),
    ) -> StructuredResult[T]:
        raw = self._send(system=system, user=user, output=output, images=images)
        # From here the API answered 2xx and bills the call: an unusable response is a
        # BilledProviderError so the tracker charges it. A refusal is kept as it was.
        try:
            envelope = _Envelope.model_validate(raw.json())
        except ValueError as e:
            raise BilledProviderError("Anthropic returned a malformed response body") from e
        if envelope.stop_reason == "refusal":
            details = envelope.stop_details
            raise ProviderRefusalError(details.category if details else None)
        if envelope.stop_reason == "max_tokens":
            raise BilledProviderError(
                f"Anthropic output truncated at max_tokens={self.max_tokens}; "
                "raise llm.max_tokens in the channel config"
            )
        try:
            message = raw.parse()
        except ValidationError as e:
            raise BilledProviderError(
                f"Anthropic output did not match {output.__name__} "
                f"(stop_reason: {envelope.stop_reason})"
            ) from e
        value = message.parsed_output
        if value is None:
            raise BilledProviderError(
                f"Anthropic returned no structured output (stop_reason: {envelope.stop_reason})"
            )
        usage = Usage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
        return StructuredResult(value=value, usage=usage, model=message.model)

    def _send[T: BaseModel](
        self, *, system: str, user: str, output: type[T], images: Sequence[bytes]
    ) -> APIResponse[ParsedBetaMessage[T]]:
        content: Any = user
        if images:
            content = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(png).decode("ascii"),
                    },
                }
                for png in images
            ]
            content.append({"type": "text", "text": user})
        betas: list[AnthropicBetaParam] | Omit = omit
        fallbacks: Literal["default"] | Omit = omit
        if self.refusal_fallback:
            betas, fallbacks = [FALLBACK_BETA], "default"
        try:
            return self.sdk.beta.messages.with_raw_response.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_format=output,
                output_config={"effort": self.effort},
                betas=betas,
                fallbacks=fallbacks,
            )
        except anthropic.AuthenticationError as e:
            raise ProviderError(
                f"Anthropic authentication failed (HTTP 401); check {KEY_ENV_VAR}"
            ) from e
        except anthropic.PermissionDeniedError as e:
            raise ProviderError(
                f"Anthropic denied permission for model {self.model!r} (HTTP 403)"
            ) from e
        except anthropic.NotFoundError as e:
            raise ProviderError(f"Anthropic model {self.model!r} not found (HTTP 404)") from e
        except anthropic.RateLimitError as e:
            raise ProviderError("Anthropic rate limit reached (HTTP 429); retry later") from e
        except anthropic.BadRequestError as e:
            raise ProviderError(
                f"Anthropic rejected the request (HTTP 400): {self._redact(e.message)}"
            ) from e
        except anthropic.InternalServerError as e:
            raise ProviderError(
                f"Anthropic server error (HTTP {e.status_code}); retry later"
            ) from e
        except anthropic.APIStatusError as e:
            raise ProviderError(f"Anthropic request failed (HTTP {e.status_code})") from e
        except anthropic.APITimeoutError as e:
            raise ProviderError("Anthropic request timed out") from e
        except anthropic.APIConnectionError as e:
            raise ProviderError("Cannot reach the Anthropic API (connection error)") from e

    def _redact(self, text: str) -> str:
        secret = self._secret.get_secret_value()
        return text.replace(secret, "[redacted]") if secret else text
