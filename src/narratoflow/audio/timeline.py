"""The part timeline: the audio contract Plan 5 (Render) consumes.

Every ms field is an int >= 0 on the part clock (0 = the part's first sample).
Every float is finite, and the file is written with JSON that rejects NaN.
"""

from __future__ import annotations

import json
import os
import tempfile
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from narratoflow.errors import AudioError
from narratoflow.models import SceneKind, WordTiming

TIMELINE_SCHEMA_VERSION: Literal[1] = 1

Ms = Annotated[int, Field(ge=0)]

_STRICT = ConfigDict(extra="forbid", allow_inf_nan=False)


class TimingSource(StrEnum):
    PROVIDER = "provider"
    ALIGNED = "aligned"
    ESTIMATED = "estimated"
    SEGMENTS = "segments"
    AYAH = "ayah"  # no segments: reveal each whole ayah at its start


def _check_words(words: list[WordTiming], lo: int, hi: int, where: str) -> None:
    prev = lo
    for w in words:
        if w.start_ms < lo or w.end_ms > hi:
            raise ValueError(f"{where}: word {w.text!r} lies outside [{lo}, {hi}]")
        if w.start_ms < prev:
            raise ValueError(f"{where}: word {w.text!r} starts before the previous word")
        prev = w.start_ms


class AyahTiming(BaseModel):
    model_config = _STRICT

    surah: int = Field(ge=1)
    ayah: int = Field(ge=1)
    start_ms: Ms
    end_ms: Ms
    words: list[WordTiming] = Field(default_factory=list)  # Uthmani display words; [] when AYAH

    @model_validator(mode="after")
    def _inside(self) -> AyahTiming:
        where = f"ayah {self.surah}:{self.ayah}"
        if self.end_ms < self.start_ms:
            raise ValueError(f"{where}: end_ms is before start_ms")
        _check_words(self.words, self.start_ms, self.end_ms, where)
        return self


class SceneTiming(BaseModel):
    model_config = _STRICT

    scene_id: str
    kind: SceneKind
    start_ms: Ms
    end_ms: Ms
    speech_start_ms: Ms
    speech_end_ms: Ms
    words: list[WordTiming]  # NARRATION: caption tokens; VERSE: []
    timing_source: TimingSource
    ayat: list[AyahTiming] = Field(default_factory=list)
    draw_start_ms: Ms | None = None  # None: no strokes (verse, text card)
    draw_end_ms: Ms | None = None

    @model_validator(mode="after")
    def _consistent(self) -> SceneTiming:
        where = f"scene {self.scene_id}"
        if not (self.start_ms <= self.speech_start_ms <= self.speech_end_ms <= self.end_ms):
            raise ValueError(
                f"{where}: need start <= speech_start <= speech_end <= end, got "
                f"{self.start_ms}, {self.speech_start_ms}, {self.speech_end_ms}, {self.end_ms}"
            )
        _check_words(self.words, self.speech_start_ms, self.speech_end_ms, f"{where} speech")
        if (self.draw_start_ms is None) != (self.draw_end_ms is None):
            raise ValueError(f"{where}: draw_start_ms and draw_end_ms must both be set or None")
        if (
            self.draw_start_ms is not None
            and self.draw_end_ms is not None
            and not (self.start_ms <= self.draw_start_ms <= self.draw_end_ms <= self.end_ms)
        ):
            raise ValueError(f"{where}: draw times lie outside the scene slot")
        if self.kind is SceneKind.VERSE:
            if self.words:
                raise ValueError(f"{where}: a verse scene carries words only in its ayat")
        elif self.ayat:
            raise ValueError(f"{where}: ayat are allowed only on verse scenes")
        for a in self.ayat:
            if a.start_ms < self.start_ms or a.end_ms > self.end_ms:
                raise ValueError(f"{where}: ayah {a.surah}:{a.ayah} lies outside the scene")
        return self


class SfxHit(BaseModel):
    model_config = _STRICT

    scene_id: str
    stroke_index: int = Field(ge=0)
    at_ms: Ms
    duration_ms: int = Field(gt=0)
    asset_id: str
    gain_db: float


class VoiceUsed(BaseModel):
    model_config = _STRICT

    provider: str
    model: str
    voice_id: str
    fallback_from: str | None = None
    fallback_reasons: list[str] = Field(default_factory=list)


class RecitationUsed(BaseModel):
    model_config = _STRICT

    reciter_id: str
    name: str
    riwaya: str
    license: str
    license_url: str
    source: str
    segments_source: str | None
    ayat: list[str]  # "31:33"


class AssetUse(BaseModel):
    model_config = _STRICT

    kind: Literal["sfx"]
    asset_id: str
    source: str
    license: str
    attribution: str = ""


class PartTimeline(BaseModel):
    model_config = _STRICT

    schema_version: Literal[1] = TIMELINE_SCHEMA_VERSION
    series_title: str
    channel_slug: str
    part_index: int = Field(ge=1)
    part_count: int = Field(ge=1)
    audio_file: str  # bare file name next to the timeline, e.g. "part-01.flac"
    sample_rate: Literal[48000] = 48000
    duration_ms: Ms
    integrated_lufs: float | None
    true_peak_dbtp: float
    scenes: list[SceneTiming] = Field(min_length=1)
    sfx: list[SfxHit]
    voice: VoiceUsed | None  # None: verse-only part
    recitations: list[RecitationUsed]
    assets: list[AssetUse]
    spent_usd: Decimal = Field(ge=0, allow_inf_nan=False)  # cumulative part spend
    cost_estimated: bool
    warnings: list[str]

    @model_validator(mode="after")
    def _consistent(self) -> PartTimeline:
        name = self.audio_file
        if not name or name in (".", "..") or "/" in name or "\\" in name or "\0" in name:
            raise ValueError(f"audio_file must be a bare file name, got {name!r}")
        if self.scenes[0].start_ms != 0:
            raise ValueError("the first scene must start at 0")
        for a, b in zip(self.scenes, self.scenes[1:], strict=False):
            if a.end_ms != b.start_ms:
                raise ValueError(
                    f"scenes must be contiguous: {a.scene_id} ends at {a.end_ms}, "
                    f"{b.scene_id} starts at {b.start_ms}"
                )
        if self.scenes[-1].end_ms != self.duration_ms:
            raise ValueError(
                f"the last scene ends at {self.scenes[-1].end_ms}, "
                f"not duration_ms {self.duration_ms}"
            )
        prev = 0
        verses = [s for s in self.scenes if s.kind is SceneKind.VERSE]
        for hit in self.sfx:
            if hit.at_ms < prev:
                raise ValueError("sfx must be sorted by at_ms")
            prev = hit.at_ms
            hit_end = hit.at_ms + hit.duration_ms
            for v in verses:
                if hit.at_ms < v.end_ms and hit_end > v.start_ms:
                    raise ValueError(
                        f"sfx {hit.asset_id} at {hit.at_ms} overlaps verse scene {v.scene_id}"
                    )
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
            os.replace(tmp_name, path)
        except OSError as exc:
            if tmp_name is not None:
                Path(tmp_name).unlink(missing_ok=True)
            raise AudioError(f"cannot write timeline {path}: {exc}") from exc

    @classmethod
    def load(cls, path: Path) -> PartTimeline:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise AudioError(f"cannot read timeline {path}: {exc}") from exc
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AudioError(f"timeline {path} is not valid JSON: {exc}") from exc
        found = raw.get("schema_version") if isinstance(raw, dict) else None
        if type(found) is not int or found != TIMELINE_SCHEMA_VERSION:
            raise AudioError(
                f"unsupported timeline schema_version {found!r}; expected {TIMELINE_SCHEMA_VERSION}"
            )
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            raise AudioError(f"timeline {path} is invalid:\n{exc}") from exc
