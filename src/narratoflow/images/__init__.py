"""Image generation: the provider protocol and its request/response types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class ImageRequest:
    prompt: str
    reference_png: bytes | None
    size: Literal["1024x1024"] = "1024x1024"
    seed: int | None = None


@dataclass(frozen=True)
class GeneratedImage:
    png: bytes
    provider: str
    model: str
    revised_prompt: str | None


class ImageProvider(Protocol):
    name: str
    model: str

    def generate(self, req: ImageRequest) -> GeneratedImage: ...


__all__ = ["GeneratedImage", "ImageProvider", "ImageRequest"]
