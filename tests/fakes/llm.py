"""A fake for the structured-output LLM client protocol."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from narratoflow.llm.base import StructuredResult, Usage

FAKE_USAGE = Usage(input_tokens=100, output_tokens=50)


class FakeLLM:
    """An `LLMClient` that returns queued values and records every call."""

    name: str = "fake"

    def __init__(self, responses: list[BaseModel | Exception], model: str = "fake-model") -> None:
        self.responses = list(responses)
        self.model = model
        self.calls: list[tuple[str, str, type[BaseModel]]] = []
        self.images: list[tuple[bytes, ...]] = []

    def complete_structured[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        output: type[T],
        images: Sequence[bytes] = (),
    ) -> StructuredResult[T]:
        self.calls.append((system, user, output))
        self.images.append(tuple(images))
        if not self.responses:
            raise AssertionError("FakeLLM has no queued responses left")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if not isinstance(item, output):
            raise AssertionError(
                f"FakeLLM queued {type(item).__name__}, but caller asked for {output.__name__}"
            )
        return StructuredResult(value=item, usage=FAKE_USAGE, model=self.model)
