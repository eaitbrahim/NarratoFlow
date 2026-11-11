"""Deterministic test doubles for the planning pipeline."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

from narratoflow.costs.tracker import estimate_tokens
from narratoflow.llm.base import StructuredResult, Usage
from narratoflow.planning.ideas import IdeaOut
from narratoflow.planning.segment import SegmentOut, SegmentSpan
from narratoflow.text.tashkeel.engines import Vocalized

FATHA = "\u064e"
SCRIPTED_IDEA = "scroll|generic"
# A sentence ends at . ! ? or the Arabic question mark (U+061F) followed by whitespace.
_SENTENCE_END = re.compile(r"(?<=[.!?\u061f])\s+")
_TEXT_MARK = "TEXT:\n"


def _is_arabic_letter(ch: str) -> bool:
    return "\u0621" <= ch <= "\u064a"


def add_fatha(text: str) -> str:
    """`text` with U+064E (fatha) inserted after every Arabic letter."""
    return "".join(ch + FATHA if _is_arabic_letter(ch) else ch for ch in text)


class ScriptedLLM:
    """An `LLMClient` whose answers are computed from the user prompt.

    SegmentOut: the stretch split at sentence ends (or every `split_words` words);
    IdeaOut: `scroll|generic`; Vocalized: the text with a fatha after each Arabic letter.
    Usage is the `estimate_tokens` of the prompt and of the JSON answer.
    """

    name: str = "scripted"

    def __init__(self, model: str = "scripted-model", split_words: int | None = None) -> None:
        self.model = model
        self.split_words = split_words
        self.calls: list[tuple[str, str, type[BaseModel]]] = []
        self._groups = 0

    def _segments(self, user: str) -> SegmentOut:
        stretch = user.split(_TEXT_MARK, 1)[1]
        self._groups += 1
        title = f"Section {self._groups}"
        if self.split_words is None:
            parts = [p.strip() for p in _SENTENCE_END.split(stretch) if p.strip()]
        else:  # fixed-size scenes, as SEGMENT_SYSTEM's 5-10 s scenes would give
            words = stretch.split()
            n = self.split_words
            parts = [" ".join(words[i : i + n]) for i in range(0, len(words), n)]
        return SegmentOut(scenes=[SegmentSpan(text=p, title=title) for p in parts])

    def complete_structured[T: BaseModel](
        self,
        *,
        system: str,
        user: str,
        output: type[T],
        images: Sequence[bytes] = (),
    ) -> StructuredResult[T]:
        self.calls.append((system, user, output))
        value: BaseModel
        if output is SegmentOut:
            value = self._segments(user)
        elif output is IdeaOut:
            value = IdeaOut(
                idea_key=SCRIPTED_IDEA,
                description="a rolled paper scroll",
                depicts_person=False,
                depicts_logo=False,
            )
        elif output is Vocalized:
            value = Vocalized(text=add_fatha(user))
        else:
            raise AssertionError(f"ScriptedLLM cannot answer {output.__name__}")
        # usage as a real model would bill it: the prompt in, the JSON answer out
        usage = Usage(
            input_tokens=estimate_tokens(system + user),
            output_tokens=estimate_tokens(value.model_dump_json()),
        )
        return StructuredResult(
            value=output.model_validate(value.model_dump()), usage=usage, model=self.model
        )


class ConstantEmbedder:
    """An `Embedder` that maps every text to the same unit vector and records calls."""

    name = "constant"

    def __init__(self, model: str = "constant-embed") -> None:
        self.model = model
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]
