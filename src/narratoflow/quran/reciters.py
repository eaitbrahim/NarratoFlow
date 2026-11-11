"""The licensed reciter allow-list and reciter selection for verse scenes.

Only reciters listed in ``assets/reciters/index.toml`` are ever used. Each entry
records the licence that permits using its recordings in published videos, so an
unlisted reciter, a riwaya mismatch or an empty list is a FAIL (`ReciterError`).
"""

from __future__ import annotations

import logging
import tomllib
from collections.abc import Iterator, Sequence
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, model_validator

from narratoflow.channels.config import ChannelConfig
from narratoflow.errors import ChannelConfigError, ReciterError

logger = logging.getLogger(__name__)

AUDIO_URL_KEY = "{surah:03d}{ayah:03d}"
QURANCOM_SEGMENTS = "quran.com"
_ID_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"


def _https(value: str) -> bool:
    parts = urlsplit(value)
    return parts.scheme == "https" and bool(parts.netloc)


class Reciter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=_ID_PATTERN)
    name: str
    riwaya: str
    kind: Literal["qurancom", "files"]
    recitation_id: StrictInt | None = Field(default=None, ge=1)
    audio_url: str | None = None
    license: str
    license_url: str
    source: str
    segments_source: str | None = None
    checked: date

    @model_validator(mode="after")
    def _consistent(self) -> Reciter:
        for field in ("name", "riwaya", "license", "license_url", "source"):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} is required")
        for field in ("license_url", "source"):
            if not _https(getattr(self, field)):
                raise ValueError(f"{field} must be an https:// URL")
        if self.kind == "qurancom":
            self._check_qurancom()
        else:
            self._check_files()
        return self

    def _check_qurancom(self) -> None:
        if self.recitation_id is None:
            raise ValueError("recitation_id is required for kind qurancom")
        if self.segments_source != QURANCOM_SEGMENTS:
            raise ValueError(f"segments_source must be {QURANCOM_SEGMENTS!r} for kind qurancom")
        if self.audio_url is not None:
            raise ValueError("audio_url is only for kind files")

    def _check_files(self) -> None:
        if self.audio_url is None:
            raise ValueError("audio_url is required for kind files")
        if self.recitation_id is not None:
            raise ValueError("recitation_id is only for kind qurancom")
        if self.segments_source is not None:
            raise ValueError("segments_source must be omitted for kind files (no segments)")
        if AUDIO_URL_KEY not in self.audio_url:
            raise ValueError(f"audio_url must contain {AUDIO_URL_KEY}")
        try:
            sample = self.audio_url.format(surah=1, ayah=1)
        except (KeyError, IndexError, ValueError) as exc:
            raise ValueError(f"audio_url is not a valid template: {exc!r}") from exc
        if not _https(self.audio_url) or not _https(sample):
            raise ValueError("audio_url must be an https:// URL")

    def file_url(self, surah: int, ayah: int) -> str:
        if self.audio_url is None:
            raise ReciterError(f"{self.name} has no audio_url template")
        return self.audio_url.format(surah=surah, ayah=ayah)


class ReciterIndex:
    def __init__(self, reciters: Sequence[Reciter]) -> None:
        self._reciters = list(reciters)
        self._by_id = {r.id: r for r in self._reciters}

    @classmethod
    def from_index(cls, path: Path) -> ReciterIndex:
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError) as exc:
            raise ChannelConfigError(f"reciter index {path} is broken: {exc!r}") from exc
        raw = data.get("reciters", [])
        if not isinstance(raw, list):
            raise ChannelConfigError(
                f"reciter index {path}: reciters must be an array of tables ([[reciters]])"
            )
        reciters: list[Reciter] = []
        seen: set[str] = set()
        for number, entry in enumerate(raw, start=1):
            reciter = _build(entry, number)
            if reciter.id in seen:
                raise ChannelConfigError(
                    f"reciter index entry {reciter.id!r}: duplicate id {reciter.id!r}"
                )
            seen.add(reciter.id)
            reciters.append(reciter)
        return cls(reciters)

    def __iter__(self) -> Iterator[Reciter]:
        return iter(self._reciters)

    def get(self, reciter_id: str) -> Reciter | None:
        return self._by_id.get(reciter_id)

    def for_riwaya(self, riwaya: str) -> list[Reciter]:
        return [r for r in self._reciters if r.riwaya == riwaya]


def _label(entry: object, number: int) -> str:
    value = entry.get("id") if isinstance(entry, dict) else None
    return repr(value) if isinstance(value, str) and value else f"entry {number}"


def _build(entry: object, number: int) -> Reciter:
    label = _label(entry, number)
    if not isinstance(entry, dict):
        raise ChannelConfigError(f"reciter index {label}: entry must be a table")
    try:
        return Reciter.model_validate(entry)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'entry'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ChannelConfigError(f"reciter index entry {label}: {problems}") from exc


def _packaged_index() -> Path:
    return Path(str(resources.files("narratoflow") / "assets" / "reciters" / "index.toml"))


def _dev_index() -> Path:
    return Path(__file__).parents[3] / "assets" / "reciters" / "index.toml"


def default_reciters() -> ReciterIndex:
    """The repository's reciter allow-list, or an empty one when it is not present."""
    for index in (_packaged_index(), _dev_index()):
        if index.is_file():
            return ReciterIndex.from_index(index)
    logger.warning("reciter index not found; no reciter is allow-listed")
    return ReciterIndex([])


RECITER_HINT = (
    "add a licensed reciter to assets/reciters/index.toml (see README) "
    "or build verse-free parts with --part N"
)


def select_reciter(
    channel: ChannelConfig, index: ReciterIndex, *, needed_by: str | None = None
) -> Reciter:
    """The channel's reciter, or the first allow-listed one of its riwaya.

    A refusal is prefixed with `needed_by` (e.g. "scene s001 (31:33)") and ends
    with what to do next.
    """
    prefix = f"{needed_by}: " if needed_by else ""

    def refuse(reason: str) -> ReciterError:
        return ReciterError(f"{prefix}{reason}; {RECITER_HINT}")

    if channel.reciter is not None:
        chosen = index.get(channel.reciter)
        if chosen is None:
            raise refuse(f"reciter {channel.reciter} is not on the allow-list")
        if chosen.riwaya != channel.riwaya:
            raise refuse(
                f"reciter {chosen.id} recites riwaya {chosen.riwaya}, "
                f"but the channel uses riwaya {channel.riwaya}"
            )
        return chosen
    matches = index.for_riwaya(channel.riwaya)
    if not matches:
        raise refuse(
            f"no allow-listed reciter for riwaya {channel.riwaya}; "
            "unlicensed recitation is not allowed"
        )
    return matches[0]
