"""Deterministic prompt for one whiteboard sketch."""

from __future__ import annotations

from narratoflow.channels.config import StyleConfig
from narratoflow.library.keys import parse_idea_key

NEGATIVE_TERMS: tuple[str, ...] = (
    "shading",
    "fills",
    "grayscale",
    "textures",
    "double lines",
    "gradients",
    "hatching",
    "3d",
)
_FORBIDDEN_CONTENT = "people, faces, human figures, text, letters, logos"


def build_prompt(style: StyleConfig, idea_key: str, description: str) -> str:
    """Style instructions, subject, ink look and one explicit 'No ...' sentence."""
    text = description.strip().rstrip(".").rstrip()
    if not text:
        key = parse_idea_key(idea_key)
        text = f"{key.object}, {key.context}"
    banned = ", ".join((*NEGATIVE_TERMS, _FORBIDDEN_CONTENT))
    return " ".join(
        (
            style.instructions,
            f"Subject: {text}.",
            "Black ink line drawing on a plain white background, single clean outlines.",
            f"No {banned}.",
        )
    )
