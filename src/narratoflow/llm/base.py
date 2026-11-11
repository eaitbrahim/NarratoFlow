"""Provider-neutral structured-output LLM interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class StructuredResult[T: BaseModel]:
    value: T
    usage: Usage
    model: str


class LLMClient(Protocol):
    """A client that returns a validated instance of `output` for a prompt."""

    name: str

    def complete_structured[T: BaseModel](
        self, *, system: str, user: str, output: type[T]
    ) -> StructuredResult[T]: ...
