"""Calibrated vision check for a generated sketch (spec 6.3)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from narratoflow.channels.config import StyleConfig
from narratoflow.library.schemas import SketchStatus
from narratoflow.llm.base import LLMClient
from narratoflow.planning.prompts import VISION_SYSTEM

NOT_CHECKED = "vision: not checked"


class VisionVerdict(BaseModel):
    human_figure: bool
    face: bool
    logo_or_text: bool
    style_outlier: bool
    mirror_safe: bool
    reason: str = Field(max_length=200)


def _user_text(style: StyleConfig, idea_key: str, with_reference: bool) -> str:
    text = (
        f"Idea key: {idea_key}\n"
        f"Style instructions: {style.instructions}\n"
        "The first image is the candidate sketch to check."
    )
    if with_reference:
        text += (
            " The second image is the channel's reference sketch, the style anchor "
            "for the style_outlier judgment; do not check the reference itself."
        )
    return text


def check_sketch(
    png: bytes,
    style: StyleConfig,
    idea_key: str,
    llm: LLMClient | None,
    *,
    reference_png: bytes | None = None,
) -> tuple[SketchStatus, str | None, bool, VisionVerdict | None]:
    """Return `(status, status_reason, mirror_safe, verdict)` for a candidate sketch.

    Without an LLM the sketch is UNCHECKED (never a silent pass) and not mirror safe.
    A human figure, face, logo or text FAILs; a style outlier alone is a WARN that
    still passes. LLM errors propagate to the caller.
    """
    if llm is None:
        return SketchStatus.UNCHECKED, NOT_CHECKED, False, None
    images = [png] if reference_png is None else [png, reference_png]
    result = llm.complete_structured(
        system=VISION_SYSTEM,
        user=_user_text(style, idea_key, reference_png is not None),
        output=VisionVerdict,
        images=images,
    )
    verdict = result.value
    failed = [name for name in ("human_figure", "face", "logo_or_text") if getattr(verdict, name)]
    if failed:
        reason = f"vision: {', '.join(failed)}; {verdict.reason}"
        return SketchStatus.FAILED, reason, verdict.mirror_safe, verdict
    if verdict.style_outlier:
        reason = f"warn: style outlier; {verdict.reason}"
        return SketchStatus.PASSED, reason, verdict.mirror_safe, verdict
    return SketchStatus.PASSED, None, verdict.mirror_safe, verdict
