"""The bundled marker SFX pack and deterministic placement on strokes."""

from __future__ import annotations

import logging
import tomllib
from collections.abc import Mapping
from importlib import resources
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from narratoflow.audio import io as aio
from narratoflow.audio.timeline import SfxHit
from narratoflow.errors import ChannelConfigError
from narratoflow.visuals.bundled import confined_file
from narratoflow.visuals.schedule import StrokeEvent

logger = logging.getLogger(__name__)

MIN_SFX_STROKE_MS = 120
SFX_MIN_SPACING_MS = 250
SFX_GAIN_DB = -12.0
SFX_FADE_MS = 15

_FIELDS = frozenset({"file", "license", "source", "attribution"})
_LICENSES = frozenset({"CC0-1.0", "MIT", "ISC", "Apache-2.0", "CC-BY-4.0"})


class SfxAsset(BaseModel):
    id: str
    file: Path
    license: str
    source: str
    attribution: str = ""


def _text(where: str, entry: dict[str, object], field: str, *, required: bool = True) -> str:
    value = entry.get(field)
    if value is None and not required:
        return ""
    if not isinstance(value, str) or not value.strip():
        raise ChannelConfigError(f"{where}: {field} is required")
    return value


def _build(sid: str, entry: object, index_path: Path) -> SfxAsset:
    where = f"sfx index entry {sid!r}"
    if not isinstance(entry, dict):
        raise ChannelConfigError(f"{where}: entry must be a table")
    unknown = sorted(set(entry) - _FIELDS)
    if unknown:
        raise ChannelConfigError(f"{where}: unknown fields {unknown}")
    license_ = _text(where, entry, "license")
    source = _text(where, entry, "source")
    attribution = _text(where, entry, "attribution", required=False)
    if license_ not in _LICENSES:
        raise ChannelConfigError(f"{where}: license {license_!r} is not allowed")
    if license_ == "CC-BY-4.0" and not attribution.strip():
        raise ChannelConfigError(f"{where}: CC-BY-4.0 requires an attribution")
    path = confined_file(where, Path(_text(where, entry, "file")), index_path, ".flac")
    return SfxAsset(id=sid, file=path, license=license_, source=source, attribution=attribution)


class SfxPack:
    def __init__(self, assets: Mapping[str, SfxAsset]) -> None:
        self._assets = dict(assets)
        self._cache: dict[str, aio.Samples] = {}

    @classmethod
    def from_index(cls, path: Path) -> SfxPack:
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            table = data.get("sfx", {})
            return cls({sid: _build(sid, entry, path) for sid, entry in table.items()})
        except (tomllib.TOMLDecodeError, TypeError, AttributeError, OSError, ValueError) as exc:
            raise ChannelConfigError(f"sfx index {path} is broken: {exc!r}") from exc

    def ids(self) -> list[str]:
        return sorted(self._assets)

    def asset(self, sid: str) -> SfxAsset:
        """The asset entry (file, licence, source) of `sid`."""
        return self._assets[sid]

    def samples(self, sid: str) -> aio.Samples:
        cached = self._cache.get(sid)
        if cached is None:
            cached = aio.decode(self._assets[sid].file.read_bytes())
            self._cache[sid] = cached
        return cached


def _packaged_index() -> Path:
    return Path(str(resources.files("narratoflow") / "assets" / "sfx" / "index.toml"))


def _dev_index() -> Path:
    return Path(__file__).parents[3] / "assets" / "sfx" / "index.toml"


def default_sfx() -> SfxPack:
    """The repository's SFX pack, or an empty pack (with a warning) when absent."""
    for index in (_packaged_index(), _dev_index()):
        if index.is_file():
            return SfxPack.from_index(index)
    logger.warning("SFX index not found; marker SFX disabled")
    return SfxPack({})


def place_sfx(
    events: list[StrokeEvent],
    *,
    scene_id: str,
    seed: int,
    pack: SfxPack,
    blocked: list[tuple[int, int]],
) -> list[SfxHit]:
    """One SFX at the start of each long-enough, spaced-out, unblocked stroke."""
    ids = pack.ids()
    if not ids:
        return []
    rng = np.random.default_rng(seed)
    hits: list[SfxHit] = []
    last_at: int | None = None
    for event in events:
        pick = ids[int(rng.integers(len(ids)))]  # one draw per stroke, hit or not
        duration = event.end_ms - event.start_ms
        if duration < MIN_SFX_STROKE_MS:
            continue
        if last_at is not None and event.start_ms - last_at < SFX_MIN_SPACING_MS:
            continue
        if any(event.start_ms < b and event.end_ms > a for a, b in blocked):
            continue
        sample_ms = len(pack.samples(pick)) * 1000 // aio.SAMPLE_RATE
        hits.append(
            SfxHit(
                scene_id=scene_id,
                stroke_index=event.index,
                at_ms=event.start_ms,
                duration_ms=max(1, min(sample_ms, duration)),
                asset_id=pick,
                gain_db=SFX_GAIN_DB,
            )
        )
        last_at = event.start_ms
    return hits
