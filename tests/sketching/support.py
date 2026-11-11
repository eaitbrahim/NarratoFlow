"""Shared fakes and builders for the sketching tests."""

from __future__ import annotations

from decimal import Decimal

from fixtures import images as img
from llm.fakes import FakeLLM
from pydantic import BaseModel

from narratoflow.channels.config import ChannelConfig, VisionConfig
from narratoflow.costs.prices import PriceTable
from narratoflow.images import GeneratedImage, ImageRequest
from narratoflow.library.repository import SketchLibrary
from narratoflow.models import (
    Language,
    Narration,
    Part,
    RightsBasis,
    Scene,
    ScriptSource,
    Series,
)
from narratoflow.settings import Settings
from narratoflow.sketching import generate as generate_module
from narratoflow.sketching.generate import generate_missing
from narratoflow.sketching.vision import VisionVerdict

PRICES = PriceTable._from_text(
    """
[llm."anthropic/claude-opus-5-5"]
input_per_mtok = "0"
output_per_mtok = "0"
[image."openai/gpt-image-1"]
per_image = "0.04"
[vision."anthropic/claude-opus-5-5"]
per_image_check = "0.01"
""",
    "test",
)
PAID_VISION = VisionConfig(provider="anthropic", model="claude-opus-5-5")
PER_SKETCH = Decimal("0.04") + 2 * Decimal("0.01")  # image + candidate and reference checks


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


def verdict(**over: object) -> VisionVerdict:
    base: dict[str, object] = {
        "human_figure": False,
        "face": False,
        "logo_or_text": False,
        "style_outlier": False,
        "mirror_safe": True,
        "reason": "clean",
    }
    return VisionVerdict.model_validate({**base, **over})


def clean(n: int) -> list[BaseModel | Exception]:
    return [verdict() for _ in range(n)]


def scene(i: int, key: str = "", **over: object) -> Scene:
    data: dict[str, object] = {
        "id": f"s{i:03d}",
        "narration": Narration(source_text=f"text {i}"),
        "idea_key": key or f"thing-{i}|roman",
        "idea_description": f"thing number {i}",
        "match": "new",
    }
    return Scene.model_validate({**data, **over})


def series(*parts: list[Scene], approved: bool = True) -> Series:
    return Series(
        title="Scenes",
        channel_slug="history",
        language=Language.ARABIC,
        source=ScriptSource(author="a", work="Scenes", rights=RightsBasis.OWN),
        parts=[
            Part(index=i + 1, count=len(parts), hook_text="h", scenes=scenes)
            for i, scenes in enumerate(parts)
        ],
        approved=approved,
    )


def run(
    s: Series,
    channel: ChannelConfig,
    settings: Settings,
    library: SketchLibrary,
    images: FakeImages,
    vision: FakeLLM | None,
) -> tuple[Series, generate_module.GenerationResult]:
    return generate_missing(
        s, channel, settings, images=images, vision=vision, library=library, prices=PRICES
    )


def scenes_of(s: Series) -> list[Scene]:
    return [sc for p in s.parts for sc in p.scenes]
