"""Pluggable Arabic diacritizers guarded by the skeleton check."""

from __future__ import annotations

import logging
import unicodedata
from typing import Any

from pydantic import BaseModel

from narratoflow.errors import ChannelConfigError, ProviderError
from narratoflow.interfaces import Diacritizer
from narratoflow.llm.base import LLMClient
from narratoflow.models import Scene, SceneKind
from narratoflow.planning.prompts import TASHKEEL_SYSTEM
from narratoflow.text.tashkeel.verify import (
    SkeletonMismatchError,
    assert_skeleton_preserved,
    is_diacritic,
)

__all__ = ["Diacritizer", "LlmDiacritizer", "MishkalDiacritizer", "Vocalized", "vocalize_scene"]

logger = logging.getLogger(__name__)

REJECTED_FLAG = "tashkeel_rejected"


class Vocalized(BaseModel):
    text: str


class LlmDiacritizer:
    name = "llm"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def diacritize(self, text: str) -> str:
        result = self._llm.complete_structured(system=TASHKEEL_SYSTEM, user=text, output=Vocalized)
        return result.value.text


def _is_neutral(ch: str) -> bool:
    """Whitespace, punctuation, symbol or control char: ignored when matching base letters."""
    return ch.isspace() or unicodedata.category(ch)[0] in "PSC"


def _transfer_marks(original: str, diacritized: str) -> str:
    """Project the marks of `diacritized` onto `original`, keeping its spacing.

    Base characters (not diacritic, whitespace, punctuation or symbol) must
    match in order, otherwise SkeletonMismatchError. Marks already in
    `original` win; its punctuation and spacing are kept as is.
    """
    marks: list[str] = []
    bases: list[str] = []
    for ch in diacritized:
        if _is_neutral(ch):
            continue
        if is_diacritic(ch):
            if marks:
                marks[-1] += ch
        else:
            bases.append(ch)
            marks.append("")
    expected = [ch for ch in original if not _is_neutral(ch) and not is_diacritic(ch)]
    if bases != expected:
        raise SkeletonMismatchError("Mishkal output changed the base letters")
    out: list[str] = []
    n = 0
    for i, ch in enumerate(original):
        out.append(ch)
        if _is_neutral(ch) or is_diacritic(ch):
            continue
        nxt = original[i + 1] if i + 1 < len(original) else ""
        if not (nxt and is_diacritic(nxt)):
            out.append(marks[n])
        n += 1
    return "".join(out)


class MishkalDiacritizer:
    name = "mishkal"

    def __init__(self) -> None:
        try:
            from mishkal.tashkeel import TashkeelClass  # type: ignore[import-not-found,unused-ignore]
        except ImportError as e:
            raise ChannelConfigError("install the 'arabic' extra: uv sync --extra arabic") from e
        self._engine: Any = TashkeelClass()

    def diacritize(self, text: str) -> str:
        return _transfer_marks(text, str(self._engine.tashkeel(text)))


def vocalize_scene(scene: Scene, d: Diacritizer) -> Scene:
    if scene.kind is SceneKind.VERSE or scene.narration.protected:
        return scene
    source = scene.narration.source_text
    try:
        vocalized = d.diacritize(source)
        assert_skeleton_preserved(source, vocalized)
    except (SkeletonMismatchError, ProviderError):
        logger.warning("tashkeel rejected for scene %s", scene.id)
        flags = scene.flags if REJECTED_FLAG in scene.flags else [*scene.flags, REJECTED_FLAG]
        narration = scene.narration.model_copy(update={"tts_text": source, "caption_text": source})
        return scene.model_copy(update={"narration": narration, "flags": flags})
    narration = scene.narration.model_copy(update={"tts_text": vocalized, "caption_text": source})
    return scene.model_copy(update={"narration": narration})
