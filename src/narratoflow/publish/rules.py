"""The dated platform rules file: loading, validation and the age check."""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from narratoflow.errors import PublishError
from narratoflow.publish.checks import CheckId, CheckResult, Severity
from narratoflow.settings import Settings

RULES_MAX_AGE_DAYS = 180

_STRICT = ConfigDict(extra="forbid", allow_inf_nan=False)


class LicenceRule(BaseModel):
    model_config = _STRICT

    verdict: Literal["pass", "warn"]
    note: str = ""
    attribution: bool = False


class Licences(BaseModel):
    model_config = _STRICT

    assets: dict[str, LicenceRule]
    generated: dict[str, LicenceRule]
    voice: dict[str, LicenceRule]


class LoudnessRules(BaseModel):
    model_config = _STRICT

    target_lufs: float
    tolerance_lu: float = Field(gt=0)
    warn_true_peak_dbtp: float
    fail_true_peak_dbtp: float
    verify: str

    @model_validator(mode="after")
    def _peaks_ordered(self) -> LoudnessRules:
        if self.warn_true_peak_dbtp > self.fail_true_peak_dbtp:
            raise ValueError("warn_true_peak_dbtp must not exceed fail_true_peak_dbtp")
        return self


class PlatformRules(BaseModel):
    model_config = _STRICT

    name: str
    max_duration_s: int = Field(gt=0)
    width: int
    height: int
    fps: int
    video_codecs: list[str]
    pix_fmts: list[str]
    audio_codecs: list[str]
    audio_rate: int
    ai_label: str
    sources: list[Annotated[str, Field(pattern=r"^https://")]]
    verify: str


class ContentCategory(BaseModel):
    model_config = _STRICT

    description: str
    max_severity: Literal["warn", "fail"]
    policies: list[str]


class RulesFile(BaseModel):
    model_config = _STRICT

    version: str
    checked: date
    loudness: LoudnessRules
    platforms: dict[Annotated[str, Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")], PlatformRules] = (
        Field(min_length=1)
    )
    licences: Licences
    content: dict[str, ContentCategory]

    @model_validator(mode="after")
    def _warn_only_categories(self) -> RulesFile:
        for name in ("religious_discussion", "other"):
            category = self.content.get(name)
            if category is None:
                raise ValueError(f"content.{name} is required")
            if category.max_severity != "warn":
                raise ValueError(f"content.{name} must have max_severity = 'warn'")
        return self


@dataclass(frozen=True)
class Rules:
    file: RulesFile
    sha256: str
    origin: str  # "packaged" or the absolute path of the override file


def _parse(data: bytes, label: str, origin: str) -> Rules:
    try:
        raw = tomllib.loads(data.decode("utf-8"))
        file = RulesFile.model_validate(raw)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, ValidationError) as e:
        raise PublishError(f"platform rules {label} are invalid: {e}") from e
    return Rules(file=file, sha256=hashlib.sha256(data).hexdigest(), origin=origin)


def load_rules(path: Path) -> Rules:
    try:
        data = path.read_bytes()
    except OSError as e:
        raise PublishError(f"platform rules {path} cannot be read: {e}") from e
    return _parse(data, str(path), str(path.resolve()))


def default_rules(settings: Settings) -> Rules:
    override = settings.data_dir / "platform_rules.toml"
    if override.is_file():
        return load_rules(override)
    packaged = resources.files("narratoflow").joinpath("data", "platform_rules.toml")
    return _parse(packaged.read_bytes(), "packaged platform_rules.toml", "packaged")


def rules_checks(rules: Rules, today: date) -> list[CheckResult]:
    version, checked = rules.file.version, rules.file.checked
    if checked > today:
        return [
            CheckResult(
                check=CheckId.RULES_FILE,
                severity=Severity.WARN,
                reason=(
                    f"platform rules {version} have a checked date {checked.isoformat()} "
                    "in the future; fix data/platform_rules.toml"
                ),
            )
        ]
    if (today - checked).days > RULES_MAX_AGE_DAYS:
        return [
            CheckResult(
                check=CheckId.RULES_FILE,
                severity=Severity.WARN,
                reason=(
                    f"platform rules {version} were checked {checked.isoformat()}, over "
                    f"{RULES_MAX_AGE_DAYS} days ago; review data/platform_rules.toml"
                ),
            )
        ]
    return [
        CheckResult(
            check=CheckId.RULES_FILE,
            severity=Severity.PASS,
            reason=f"platform rules {version} (checked {checked.isoformat()})",
        )
    ]
