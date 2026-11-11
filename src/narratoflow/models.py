"""Storyboard intermediate representation (IR).

`Series` (schema v2) is the planning root; `Project` (v1) remains for `validate`.
Every pipeline stage reads and writes a `Project`; the review UI edits it too.
Bump `SCHEMA_VERSION` on breaking changes and add a migration.
"""

from __future__ import annotations

import json
import os
import tempfile
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from narratoflow.errors import PlanningError

SCHEMA_VERSION = 2
PROJECT_SCHEMA_VERSION = 1


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


class SceneKind(StrEnum):
    NARRATION = "narration"
    VERSE = "verse"


class RightsBasis(StrEnum):
    OWN = "own"
    PUBLIC_DOMAIN = "public_domain"
    PERMISSION = "permission"
    THIRD_PARTY_IN_COPYRIGHT = "third_party_in_copyright"


class ScriptSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    author: str
    work: str
    edition: str = ""
    rights: RightsBasis
    permission_note: str = ""
    ai_transcribed: bool = False


class VerseRef(BaseModel):
    surah: int = Field(ge=1, le=114)
    ayah_start: int = Field(ge=1)
    ayah_end: int = Field(ge=1)
    certain: bool
    uthmani_text: str

    @model_validator(mode="after")
    def _range(self) -> VerseRef:
        if self.ayah_end < self.ayah_start:
            raise ValueError("ayah_end must be >= ayah_start")
        return self


class Scene(BaseModel):
    id: str
    narration: Narration
    title: str = ""
    """Short on-screen title for the scene's paragraph group (verse scenes: "2:255")."""
    visual: Visual = Field(default_factory=Visual)
    layouts: dict[AspectRatio, Box] = Field(default_factory=dict)
    min_duration_ms: int = 3000
    draw_fraction: float = Field(default=0.65, gt=0, le=1)
    """Share of the scene spent drawing; the remainder is a hold."""
    kind: SceneKind = SceneKind.NARRATION
    verse: VerseRef | None = None
    idea_key: str | None = None
    idea_description: str = ""
    match: Literal["reused", "bundled", "new"] | None = None
    est_duration_s: float = 0
    flags: list[str] = Field(default_factory=list)


class Voice(BaseModel):
    engine: str = "edge"
    voice_id: str = "ar-SA-HamedNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"


class Project(BaseModel):
    schema_version: int = PROJECT_SCHEMA_VERSION
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


class Part(BaseModel):
    index: int = Field(ge=1)
    count: int = Field(ge=1)
    hook_text: str
    recap_text: str = ""
    teaser_text: str = ""
    scenes: list[Scene]
    flags: list[str] = Field(default_factory=list)
    est_cost_usd: Decimal = Decimal(0)

    @property
    def est_duration_s(self) -> float:
        return sum(s.est_duration_s for s in self.scenes)


class Series(BaseModel):
    schema_version: int = SCHEMA_VERSION
    title: str
    channel_slug: str
    language: Language
    source: ScriptSource
    parts: list[Part]
    approved: bool = False
    est_planning_cost_usd: Decimal = Decimal(0)
    ai_disclosure: bool = True

    def save(self, path: Path) -> None:
        """Write atomically: a temp file in the same directory, then `os.replace`."""
        tmp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as tmp:
                tmp_name = tmp.name
                tmp.write(self.model_dump_json(indent=2))
            os.replace(tmp_name, path)
        except OSError as exc:
            if tmp_name is not None:
                Path(tmp_name).unlink(missing_ok=True)
            raise PlanningError(f"cannot write series file {path}: {exc}") from exc

    @classmethod
    def load(cls, path: Path) -> Series:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise PlanningError(f"cannot read series file {path}: {exc}") from exc
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PlanningError(f"series file {path} is not valid JSON: {exc}") from exc
        found = raw.get("schema_version") if isinstance(raw, dict) else None
        if found != SCHEMA_VERSION:
            raise PlanningError(
                f"unsupported series schema_version {found}; expected {SCHEMA_VERSION}"
            )
        try:
            return cls.model_validate_json(text)
        except ValidationError as exc:
            raise PlanningError(f"series file {path} is invalid:\n{exc}") from exc
