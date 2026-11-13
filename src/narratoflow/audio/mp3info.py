"""Read the first MP3 frame's Xing/Info/VBRI header, including the LAME delay fields.

The encoder delay and padding tell how many samples of priming a decoder should trim.
Diagnostic only: nothing under src/ imports this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from narratoflow.errors import AudioError

DECODER_DELAY_SAMPLES = 529
_RATES = {
    "1": (44100, 48000, 32000),
    "2": (22050, 24000, 16000),
    "2.5": (11025, 12000, 8000),
}
_VERSIONS: dict[int, Literal["1", "2", "2.5"]] = {3: "1", 2: "2", 0: "2.5"}
_LAYER_III = 1
_MONO = 3
_VBRI_OFFSET = 36
_VBRI_LENGTH = 18
_LAME_EXT_LENGTH = 24
_SYNC = re.compile(rb"\xff[\xe0-\xff]")


@dataclass(frozen=True)
class Mp3Info:
    mpeg_version: Literal["1", "2", "2.5"]
    sample_rate: int
    channels: int
    samples_per_frame: int
    tag: Literal["Xing", "Info", "VBRI"] | None
    encoder: str | None
    encoder_delay: int | None
    encoder_padding: int | None
    frames: int | None


@dataclass(frozen=True)
class _Tag:
    tag: Literal["Xing", "Info", "VBRI"] | None = None
    encoder: str | None = None
    delay: int | None = None
    padding: int | None = None
    frames: int | None = None


def _take(data: bytes, pos: int, size: int, what: str) -> bytes:
    chunk = data[pos : pos + size]
    if len(chunk) < size:
        raise AudioError(f"mp3 header: {what} is truncated")
    return chunk


def _skip_id3(data: bytes) -> int:
    if data[:3] != b"ID3":
        return 0
    header = _take(data, 0, 10, "ID3 header")
    size = 0
    for byte in header[6:10]:
        size = (size << 7) | (byte & 0x7F)
    return 10 + size + (10 if header[5] & 0x10 else 0)


def _frame(data: bytes) -> tuple[int, Literal["1", "2", "2.5"], int, bool]:
    """Position of the first frame, its MPEG version, sample rate and mono flag."""
    found = _SYNC.search(data, _skip_id3(data))
    if found is None:
        raise AudioError("mp3 header: no frame sync found")
    pos = found.start()
    header = _take(data, pos, 4, "frame header")
    version = _VERSIONS.get((header[1] >> 3) & 3)
    if version is None:
        raise AudioError("mp3 header: reserved MPEG version")
    if (header[1] >> 1) & 3 != _LAYER_III:
        raise AudioError("mp3 header: not Layer III")
    rate_index = (header[2] >> 2) & 3
    if rate_index == 3:
        raise AudioError("mp3 header: reserved sample rate")
    return pos, version, _RATES[version][rate_index], (header[3] >> 6) == _MONO


def _uint(data: bytes, pos: int, size: int, what: str) -> int:
    return int.from_bytes(_take(data, pos, size, what), "big")


def _xing(data: bytes, pos: int, name: Literal["Xing", "Info"]) -> _Tag:
    pos += 4
    flags = _uint(data, pos, 4, f"{name} flags")
    pos += 4
    frames = None
    if flags & 1:
        frames = _uint(data, pos, 4, f"{name} frames")
        pos += 4
    for bit, size, field in ((2, 4, "bytes"), (4, 100, "TOC"), (8, 4, "quality")):
        if flags & bit:
            _take(data, pos, size, f"{name} {field}")
            pos += size
    return _lame(data, pos, name, frames)


def _lame(data: bytes, pos: int, name: Literal["Xing", "Info"], frames: int | None) -> _Tag:
    if data[pos : pos + 4] not in (b"LAME", b"Lavc"):
        return _Tag(name, frames=frames)
    ext = _take(data, pos, _LAME_EXT_LENGTH, "LAME extension")
    encoder = ext[:9].decode("ascii", errors="replace").strip()
    delay = (ext[21] << 4) | (ext[22] >> 4)
    padding = ((ext[22] & 0xF) << 8) | ext[23]
    return _Tag(name, encoder, delay, padding, frames)


def _vbri(data: bytes, pos: int) -> _Tag:
    body = _take(data, pos, _VBRI_LENGTH, "VBRI header")
    return _Tag(
        "VBRI", delay=int.from_bytes(body[6:8], "big"), frames=int.from_bytes(body[14:18], "big")
    )


def _tag(data: bytes, pos: int, version: str, mono: bool) -> _Tag:
    side_info = (17 if mono else 32) if version == "1" else (9 if mono else 17)
    at = pos + 4 + side_info
    name = data[at : at + 4]
    if name == b"Xing":
        return _xing(data, at, "Xing")
    if name == b"Info":
        return _xing(data, at, "Info")
    if data[pos + _VBRI_OFFSET : pos + _VBRI_OFFSET + 4] == b"VBRI":
        return _vbri(data, pos + _VBRI_OFFSET)
    return _Tag()


def read_mp3_info(data: bytes) -> Mp3Info:
    """Header facts of the first MP3 frame in `data`; raises AudioError when malformed."""
    pos, version, rate, mono = _frame(data)
    tag = _tag(data, pos, version, mono)
    return Mp3Info(
        mpeg_version=version,
        sample_rate=rate,
        channels=1 if mono else 2,
        samples_per_frame=1152 if version == "1" else 576,
        tag=tag.tag,
        encoder=tag.encoder,
        encoder_delay=tag.delay,
        encoder_padding=tag.padding,
        frames=tag.frames,
    )
