"""Check framework shared by every publishing check."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from narratoflow.audio.timeline import PartTimeline
from narratoflow.channels.config import ChannelConfig
from narratoflow.library.schemas import SketchRecord
from narratoflow.models import Part, Series
from narratoflow.render.report import RenderReport


class Severity(StrEnum):
    PASS = "PASS"  # noqa: S105
    WARN = "WARN"
    FAIL = "FAIL"


class CheckId(StrEnum):
    """Check identifiers, in report order."""

    RULES_FILE = "rules_file"
    RENDER_CURRENT = "render_current"
    ASSET_PROVENANCE = "asset_provenance"
    SCRIPT_SOURCE = "script_source"
    SKETCH_LIKENESS = "sketch_likeness"
    SKETCH_STYLE = "sketch_style"
    VERSE_TEXT = "verse_text"
    AI_DISCLOSURE = "ai_disclosure"
    CONTENT_RULES = "content_rules"
    TECHNICAL = "technical"


_RANK = {Severity.PASS: 0, Severity.WARN: 1, Severity.FAIL: 2}


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    check: CheckId
    severity: Severity
    reason: str = Field(min_length=1)
    scene_id: str | None = None
    platform: str | None = None


def worst(results: Iterable[CheckResult]) -> Severity:
    """The most severe result; an empty input is PASS."""
    return max((r.severity for r in results), key=_RANK.__getitem__, default=Severity.PASS)


def summarize(check: CheckId, findings: list[CheckResult], ok: str) -> list[CheckResult]:
    """The findings, or a single PASS with the `ok` reason when there are none."""
    if findings:
        return findings
    return [CheckResult(check=check, severity=Severity.PASS, reason=ok)]


@dataclass(frozen=True)
class PartInputs:
    series: Series
    part: Part
    channel: ChannelConfig
    render: RenderReport
    timeline: PartTimeline
    sketches: Mapping[int, SketchRecord | None]
    render_dir: Path
