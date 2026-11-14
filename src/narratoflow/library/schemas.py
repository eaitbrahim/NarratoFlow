"""Pydantic models for sketch library records."""

from datetime import datetime
from enum import StrEnum

from pydantic import AwareDatetime, BaseModel


class SketchOrigin(StrEnum):
    GENERATED = "generated"
    BUNDLED = "bundled"
    COPIED = "copied"
    UPLOADED = "uploaded"


class SketchStatus(StrEnum):
    PASSED = "passed"
    UNCHECKED = "unchecked"
    FAILED = "failed"


class Provenance(BaseModel):
    source: str
    license: str
    provider: str | None = None
    model: str | None = None
    created_at: AwareDatetime
    usage_terms: str | None = None
    copied_from_sketch_id: int | None = None
    attribution: str = ""


class SketchRecord(BaseModel):
    id: int
    style_version_id: int
    idea_key: str
    idea_description: str
    blob_hash: str
    suffix: str
    origin: SketchOrigin
    status: SketchStatus
    status_reason: str | None
    mirror_safe: bool
    provenance: Provenance
    created_at: datetime
    strokes_hash: str | None = None
