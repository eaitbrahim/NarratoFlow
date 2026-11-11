"""The bundled OFL fonts and the generated hand sprite, both verified against their indexes."""

from __future__ import annotations

import hashlib
import shutil
import tomllib
import unicodedata
from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from importlib import resources
from pathlib import Path

import cv2
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image

from narratoflow.audio.sfx import ASSET_LICENSES
from narratoflow.errors import ChannelConfigError
from narratoflow.visuals.bundled import confined_file

_FONT_FIELDS = frozenset({"file", "family", "license", "license_file", "source", "sha256"})
_HAND_FIELDS = frozenset({"file", "license", "source", "attribution", "tip_x", "tip_y", "sha256"})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(where: str, entry: dict[str, object], field: str, *, required: bool = True) -> str:
    value = entry.get(field)
    if value is None and not required:
        return ""
    if not isinstance(value, str) or not value.strip():
        raise ChannelConfigError(f"{where}: {field} is required")
    return value


def _int(where: str, entry: dict[str, object], field: str) -> int:
    value = entry.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ChannelConfigError(f"{where}: {field} must be an integer")
    return value


@dataclass(frozen=True)
class FontAsset:
    id: str
    file: Path
    family: str
    license: str
    license_file: Path
    source: str
    sha256: str


@cache
def _cmap(file: Path, sha256: str) -> frozenset[int]:
    font = TTFont(file)
    try:
        return frozenset(font.getBestCmap() or {})
    finally:
        font.close()


@cache
def _win_metrics(file: Path, sha256: str) -> tuple[int, int]:
    """(unitsPerEm, usWinAscent + usWinDescent) of a font file."""
    font = TTFont(file)
    try:
        os2 = font["OS/2"]
        return int(font["head"].unitsPerEm), int(os2.usWinAscent) + int(os2.usWinDescent)
    finally:
        font.close()


def _font_family(file: Path) -> str:
    font = TTFont(file)
    try:
        names = font["name"]
        return str(names.getDebugName(16) or names.getDebugName(1) or "")
    finally:
        font.close()


def _build_font(fid: str, entry: object, index_path: Path) -> FontAsset:
    where = f"font index entry {fid!r}"
    if not isinstance(entry, dict):
        raise ChannelConfigError(f"{where}: entry must be a table")
    unknown = sorted(set(entry) - _FONT_FIELDS)
    if unknown:
        raise ChannelConfigError(f"{where}: unknown fields {unknown}")
    family = _text(where, entry, "family")
    license_ = _text(where, entry, "license")
    source = _text(where, entry, "source")
    sha = _text(where, entry, "sha256")
    if license_ != "OFL-1.1":
        raise ChannelConfigError(f"{where}: license {license_!r} is not allowed (OFL-1.1 only)")
    file = confined_file(where, Path(_text(where, entry, "file")), index_path, ".ttf")
    license_file = confined_file(
        where, Path(_text(where, entry, "license_file")), index_path, ".txt"
    )
    if _sha256(file) != sha:
        raise ChannelConfigError(f"{where}: sha256 of {file.name} does not match the index")
    try:
        actual = _font_family(file)
    except Exception as exc:
        raise ChannelConfigError(f"{where}: cannot read font {file.name}: {exc!r}") from exc
    if actual != family:
        raise ChannelConfigError(f"{where}: family {family!r} != font name table {actual!r}")
    return FontAsset(fid, file, family, license_, license_file, source, sha)


class FontPack:
    def __init__(self, fonts: dict[str, FontAsset]) -> None:
        self._fonts = dict(fonts)

    @classmethod
    def from_index(cls, path: Path) -> FontPack:
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            table = data.get("fonts", {})
            return cls({fid: _build_font(fid, entry, path) for fid, entry in table.items()})
        except (tomllib.TOMLDecodeError, TypeError, AttributeError, OSError, ValueError) as exc:
            raise ChannelConfigError(f"font index {path} is broken: {exc!r}") from exc

    def get(self, font_id: str) -> FontAsset:
        return self._fonts[font_id]

    def ids(self) -> list[str]:
        return sorted(self._fonts)

    def copy_to(self, directory: Path) -> None:
        """Copy the font files (not the licences) into `directory`."""
        directory.mkdir(parents=True, exist_ok=True)
        for asset in self._fonts.values():
            shutil.copyfile(asset.file, directory / asset.file.name)

    def fontsize_per_em(self, font_id: str) -> Fraction:
        """ASS Fontsize per em pixel: (usWinAscent + usWinDescent) / unitsPerEm.

        libass sizes a font like VSFilter: Fontsize is the win-metric line height, so a
        font with tall win metrics (Amiri) renders a smaller em at the same Fontsize.
        """
        asset = self._fonts[font_id]
        upem, win = _win_metrics(asset.file, asset.sha256)
        return Fraction(win, upem)

    def fontsize_for_em(self, font_id: str, em_px: Fraction) -> int:
        """The ASS Fontsize that renders `font_id` with an em of `em_px` pixels (half up)."""
        exact = em_px * self.fontsize_per_em(font_id)
        return int((exact * 2 + 1) // 2)

    def missing_glyphs(self, font_id: str, text: str) -> list[str]:
        """Sorted unique "U+XXXX" of the codepoints of `text` the font's cmap lacks."""
        asset = self._fonts[font_id]
        cmap = _cmap(asset.file, asset.sha256)
        missing = {
            ord(ch)
            for ch in text
            if not ch.isspace() and unicodedata.category(ch) != "Cf" and ord(ch) not in cmap
        }
        return [f"U+{cp:04X}" for cp in sorted(missing)]


@cache
def _scaled_sprite(
    file: Path, sha256: str, tip: tuple[int, int], scale: Fraction
) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(file) as im:
        rgba = np.asarray(im.convert("RGBA"))
    h, w = rgba.shape[:2]
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    out = (
        rgba.copy()
        if (nw, nh) == (w, h)
        else cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA)
    )
    out.setflags(write=False)
    scaled = (min(nw - 1, round(tip[0] * scale)), min(nh - 1, round(tip[1] * scale)))
    return out, scaled


@dataclass(frozen=True)
class HandSprite:
    id: str
    file: Path
    license: str
    source: str
    attribution: str
    tip: tuple[int, int]
    sha256: str

    @classmethod
    def from_index(cls, path: Path) -> HandSprite:
        where = "hand index entry 'right'"
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            entry = data.get("hands", {}).get("right")
            if not isinstance(entry, dict):
                raise ChannelConfigError(f"{where}: missing [hands.right] table")
            unknown = sorted(set(entry) - _HAND_FIELDS)
            if unknown:
                raise ChannelConfigError(f"{where}: unknown fields {unknown}")
            license_ = _text(where, entry, "license")
            source = _text(where, entry, "source")
            attribution = _text(where, entry, "attribution", required=False)
            sha = _text(where, entry, "sha256")
            tip = (_int(where, entry, "tip_x"), _int(where, entry, "tip_y"))
            if license_ not in ASSET_LICENSES:
                raise ChannelConfigError(f"{where}: license {license_!r} is not allowed")
            if license_ == "CC-BY-4.0" and not attribution.strip():
                raise ChannelConfigError(f"{where}: CC-BY-4.0 requires an attribution")
            file = confined_file(where, Path(_text(where, entry, "file")), path, ".png")
            if _sha256(file) != sha:
                raise ChannelConfigError(f"{where}: sha256 of {file.name} does not match")
            with Image.open(file) as im:
                if im.mode != "RGBA":
                    raise ChannelConfigError(f"{where}: PNG must be RGBA, got {im.mode}")
                w, h = im.size
                if not (0 <= tip[0] < w and 0 <= tip[1] < h):
                    raise ChannelConfigError(f"{where}: tip {tip} is outside the {w}x{h} image")
                alpha = im.getpixel(tip)[3]  # type: ignore[index]
            if alpha <= 0:
                raise ChannelConfigError(f"{where}: tip pixel {tip} is transparent")
            return cls("right", file, license_, source, attribution, tip, sha)
        except (tomllib.TOMLDecodeError, TypeError, AttributeError, OSError, ValueError) as exc:
            raise ChannelConfigError(f"hand index {path} is broken: {exc!r}") from exc

    def image(self, scale: Fraction) -> tuple[np.ndarray, tuple[int, int]]:
        """The RGBA sprite resized by `scale` (INTER_AREA) and its scaled tip; read-only."""
        return _scaled_sprite(self.file, self.sha256, self.tip, scale)


def _index(kind: str) -> Path:
    packaged = Path(str(resources.files("narratoflow") / "assets" / kind / "index.toml"))
    dev = Path(__file__).parents[3] / "assets" / kind / "index.toml"
    for index in (packaged, dev):
        if index.is_file():
            return index
    raise ChannelConfigError(f"bundled {kind} not found (looked for {packaged} and {dev})")


def default_fonts() -> FontPack:
    return FontPack.from_index(_index("fonts"))


def default_hand() -> HandSprite:
    return HandSprite.from_index(_index("hands"))
