"""A fake image generator."""

from __future__ import annotations

from fixtures import images as img

from narratoflow.images import GeneratedImage, ImageRequest


class FakeImages:
    name = "openai"
    model = "gpt-image-1"

    def __init__(self, pngs: list[bytes] | None = None) -> None:
        self.pngs = list(pngs or [])
        self.requests: list[ImageRequest] = []

    def generate(self, req: ImageRequest) -> GeneratedImage:
        self.requests.append(req)
        png = self.pngs.pop(0) if self.pngs else img.line_drawing_scroll()
        return GeneratedImage(png, "openai", "gpt-image-1", None)
