"""``part-NN.render.json``: what render drew, with which assets, for Plan 6 (publish).

Every model rejects unknown keys and non-finite floats; `RenderReport.load` validates
strictly (no int/bool coercion), like `PartTimeline.load`.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from narratoflow.audio.timeline import TimingSource
from narratoflow.errors import RenderError
from narratoflow.models import SceneKind
from narratoflow.render.variation import Variation

REPORT_SCHEMA_VERSION: Literal[1] = 1
_MODE = 0o644
_STRICT = ConfigDict(extra="forbid", allow_inf_nan=False)


class SceneRender(BaseModel):
    model_config = _STRICT

    scene_id: str
    kind: SceneKind
    visual: Literal["sketch", "text_card", "verse"]
    sketch_id: int | None
    frames: int
    variation: Variation | None


class VerseRender(BaseModel):
    model_config = _STRICT

    scene_id: str
    ayat: list[str]  # "31:33"
    shown_sha256: str  # sha256 of `captions.verse.shown_text` (UTF-8)
    font_id: str
    timing_source: TimingSource


class RenderAsset(BaseModel):
    model_config = _STRICT

    kind: Literal["font", "hand"]
    asset_id: str
    source: str
    license: str
    sha256: str
    attribution: str = ""


def _bare(name: str) -> bool:
    return bool(name) and name not in (".", "..") and not any(c in name for c in "/\\\0")


class RenderReport(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1] = REPORT_SCHEMA_VERSION
    series_title: str
    channel_slug: str
    part_index: int
    part_count: int
    video_file: str
    timeline_file: str
    audio_file: str
    captions_file: str
    width: int
    height: int
    fps: Literal[30] = 30
    frames: int
    duration_ms: int
    video_codec: Literal["h264"] = "h264"
    crf: int
    ffmpeg_version: str
    integrated_lufs: float | None
    true_peak_dbtp: float | None
    scenes: list[SceneRender]
    verses: list[VerseRender]
    assets: list[RenderAsset]
    warnings: list[str]
    inputs_key: str | None = None  # `render.build` fingerprint; None = written before it

    @model_validator(mode="after")
    def _files(self) -> RenderReport:
        for field in ("video_file", "timeline_file", "audio_file", "captions_file"):
            name = getattr(self, field)
            if not _bare(name):
                raise ValueError(f"{field} must be a bare file name, got {name!r}")
        return self

    def to_json(self) -> str:
        """Serialize, refusing any non-finite float."""
        return json.dumps(
            self.model_dump(mode="json"), indent=2, ensure_ascii=False, allow_nan=False
        )

    def save(self, path: Path) -> None:
        """Write atomically: a temp file in the same directory, then `os.replace`."""
        text = self.to_json()
        tmp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as tmp:
                tmp_name = tmp.name
                tmp.write(text)
                tmp.flush()
                os.fchmod(tmp.fileno(), _MODE)
                os.fsync(tmp.fileno())
            os.replace(tmp_name, path)
        except OSError as exc:
            if tmp_name is not None:
                Path(tmp_name).unlink(missing_ok=True)
            raise RenderError(f"cannot write render report {path}: {exc}") from exc

    @classmethod
    def load(cls, path: Path) -> RenderReport:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise RenderError(f"cannot read render report {path}: {exc}") from exc
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RenderError(f"render report {path} is not valid JSON: {exc}") from exc
        except RecursionError as exc:
            raise RenderError(f"render report {path} is nested too deeply") from exc
        found = raw.get("schema_version") if isinstance(raw, dict) else None
        if type(found) is not int or found != REPORT_SCHEMA_VERSION:
            raise RenderError(
                f"unsupported render report schema_version {found!r}; "
                f"expected {REPORT_SCHEMA_VERSION}"
            )
        try:
            return cls.model_validate_json(text, strict=True)
        except ValidationError as exc:
            raise RenderError(f"render report {path} is invalid:\n{exc}") from exc
