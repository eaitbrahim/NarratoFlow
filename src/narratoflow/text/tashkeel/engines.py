"""Pluggable Arabic diacritizers guarded by the skeleton check."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from narratoflow.errors import ChannelConfigError, ProviderError
from narratoflow.interfaces import Diacritizer
from narratoflow.llm.base import LLMClient
from narratoflow.models import Scene, SceneKind
from narratoflow.planning.prompts import TASHKEEL_SYSTEM
from narratoflow.text.tashkeel.verify import SkeletonMismatchError, assert_skeleton_preserved

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


class MishkalDiacritizer:
    name = "mishkal"

    def __init__(self) -> None:
        try:
            from mishkal.tashkeel import TashkeelClass  # type: ignore[import-not-found,unused-ignore]
        except ImportError as e:
            raise ChannelConfigError("install the 'arabic' extra: uv sync --extra arabic") from e
        self._engine: Any = TashkeelClass()

    def diacritize(self, text: str) -> str:
        out = str(self._engine.tashkeel(text)).strip()
        lead = text[: len(text) - len(text.lstrip())]
        trail = text[len(text.rstrip()) :]
        return lead + out + trail


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
