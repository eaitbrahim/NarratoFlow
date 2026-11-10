"""Storyboard intermediate representation (IR).

Every pipeline stage reads and writes a `Project`; the review UI edits it too.
Bump `SCHEMA_VERSION` on breaking changes and add a migration.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = 1


class Language(StrEnum):
    ARABIC = "ar"
    ENGLISH = "en"


class AspectRatio(StrEnum):
    LANDSCAPE = "16:9"
    PORTRAIT = "9:16"

    @property
    def resolution(self) -> tuple[int, int]:
        return (1920, 1080) if self is AspectRatio.LANDSCAPE else (1080, 1920)


class VisualSource(StrEnum):
    ICON = "icon"
    DIFFUSION = "diffusion"
    IMAGE_TO_SKETCH = "image_to_sketch"
    TEXT_CARD = "text_card"


class WordTiming(BaseModel):
    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> WordTiming:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must be >= start_ms")
        return self


class Narration(BaseModel):
    source_text: str
    """Text as authored (no automatic changes)."""
    tts_text: str | None = None
    """Fully vocalized text sent to TTS."""
    caption_text: str | None = None
    """Lightly/un-diacritized text shown in captions."""
    protected: bool = False
    """Verified text (e.g. Qur'an) that must never be auto-diacritized."""
    audio_path: Path | None = None
    words: list[WordTiming] = Field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        return self.words[-1].end_ms if self.words else 0


class Box(BaseModel):
    """Normalized [0, 1] placement on the canvas, independent of resolution."""

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)


class Visual(BaseModel):
    source: VisualSource = VisualSource.ICON
    prompt: str = ""
    """Concept / prompt used to find or generate the visual."""
    asset_path: Path | None = None
    """Raster or SVG produced by the visual stage."""
    strokes_path: Path | None = None
    """Ordered centerline strokes (JSON) used by the reveal animation."""
    seed: int | None = None


class Scene(BaseModel):
    id: str
    narration: Narration
    visual: Visual = Field(default_factory=Visual)
    layouts: dict[AspectRatio, Box] = Field(default_factory=dict)
    min_duration_ms: int = 3000
    draw_fraction: float = Field(default=0.65, gt=0, le=1)
    """Share of the scene spent drawing; the remainder is a hold."""


class Voice(BaseModel):
    engine: str = "edge"
    voice_id: str = "ar-SA-HamedNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"


class Project(BaseModel):
    schema_version: int = SCHEMA_VERSION
    title: str
    language: Language
    voice: Voice = Field(default_factory=Voice)
    outputs: list[AspectRatio] = Field(default_factory=lambda: [AspectRatio.LANDSCAPE])
    scenes: list[Scene] = Field(default_factory=list)
    ai_disclosure: bool = True

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Project:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))
