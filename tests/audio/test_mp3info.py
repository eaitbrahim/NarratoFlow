"""MP3 header reader and the decoder-priming characterization against real ffmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from narratoflow.audio.io import decode
from narratoflow.audio.mp3info import DECODER_DELAY_SAMPLES, Mp3Info, read_mp3_info
from narratoflow.audio.onsets import first_onset_ms
from narratoflow.errors import AudioError

Version = Literal["1", "2", "2.5"]
Tag = Literal["Xing", "Info", "VBRI"]
FRAMES = 100
_VERSION_BITS = {"1": 3, "2": 2, "2.5": 0}
_SIDE_INFO = {("1", True): 17, ("1", False): 32, ("2", True): 9, ("2", False): 17}


def _frame_header(version: Version, rate_index: int, mono: bool, layer_bits: int = 1) -> bytes:
    byte1 = 0xE0 | (_VERSION_BITS[version] << 3) | (layer_bits << 1) | 1
    return bytes([0xFF, byte1, 0x90 | (rate_index << 2), 0xC0 if mono else 0x00])


def _lame_ext(lame: tuple[int, int]) -> bytes:
    delay, padding = lame
    ext = bytearray(b"Lavc63.1." + bytes(15))
    ext[21] = delay >> 4
    ext[22] = ((delay & 0xF) << 4) | (padding >> 8)
    ext[23] = padding & 0xFF
    return bytes(ext)


def _xing(flags: int, lame: tuple[int, int] | None) -> bytes:
    fields = FRAMES.to_bytes(4, "big") if flags & 1 else b""
    fields += (1234).to_bytes(4, "big") if flags & 2 else b""
    fields += bytes(100) if flags & 4 else b""
    fields += (77).to_bytes(4, "big") if flags & 8 else b""
    return flags.to_bytes(4, "big") + fields + (_lame_ext(lame) if lame else b"")


def _mp3(
    *,
    id3: bool = False,
    version: Version = "2",
    rate_index: int = 1,
    mono: bool = True,
    tag: Tag | None = None,
    flags: int = 15,
    lame: tuple[int, int] | None = None,
    vbri_delay: int | None = None,
) -> bytes:
    head = b""
    if id3:
        head = b"ID3\x04\x00\x00" + bytes([0, 0, 0, 0x7F]) + bytes(127)
    body = bytearray(_frame_header(version, rate_index, mono) + bytes(200))
    if tag in ("Xing", "Info"):
        at = 4 + _SIDE_INFO[(version if version == "1" else "2", mono)]
        payload = tag.encode() + _xing(flags, lame)
        body[at : at + len(payload)] = payload
        body = body[: at + len(payload)]
    elif tag == "VBRI":
        payload = b"VBRI" + (1).to_bytes(2, "big") + (vbri_delay or 0).to_bytes(2, "big")
        payload += bytes(2) + (999).to_bytes(4, "big") + FRAMES.to_bytes(4, "big")
        body[36 : 36 + len(payload)] = payload
    return head + bytes(body)


def test_info_tag_with_lame_delay() -> None:
    info = read_mp3_info(_mp3(tag="Info", lame=(576, 960)))
    assert info == Mp3Info("2", 24000, 1, 576, "Info", "Lavc63.1.", 576, 960, FRAMES)


def test_xing_without_lame_extension() -> None:
    info = read_mp3_info(_mp3(tag="Xing", flags=15, lame=None))
    assert (info.tag, info.frames) == ("Xing", FRAMES)
    assert (info.encoder, info.encoder_delay, info.encoder_padding) == (None, None, None)


def test_xing_with_only_flags_has_no_frames() -> None:
    info = read_mp3_info(_mp3(tag="Xing", flags=0))
    assert (info.tag, info.frames) == ("Xing", None)


@pytest.mark.parametrize(
    ("version", "rate_index", "mono", "rate", "channels", "spf"),
    [
        ("1", 0, False, 44100, 2, 1152),
        ("2", 0, False, 22050, 2, 576),
        ("2.5", 2, True, 8000, 1, 576),
    ],
)
def test_layout_per_version(
    version: Version, rate_index: int, mono: bool, rate: int, channels: int, spf: int
) -> None:
    info = read_mp3_info(
        _mp3(version=version, rate_index=rate_index, mono=mono, tag="Info", lame=(576, 960))
    )
    assert (info.sample_rate, info.channels, info.samples_per_frame) == (rate, channels, spf)
    assert (info.mpeg_version, info.encoder_delay) == (version, 576)


def test_vbri_delay() -> None:
    info = read_mp3_info(_mp3(tag="VBRI", vbri_delay=1105))
    assert (info.tag, info.encoder_delay, info.frames) == ("VBRI", 1105, FRAMES)
    assert info.encoder is None


def test_no_tag() -> None:
    info = read_mp3_info(_mp3())
    assert (info.tag, info.encoder_delay, info.frames) == (None, None, None)


def test_header_only_buffer_has_no_tag() -> None:
    assert read_mp3_info(_frame_header("2", 1, True)).tag is None


def test_id3_is_skipped() -> None:
    plain = read_mp3_info(_mp3(tag="Info", lame=(576, 960)))
    assert read_mp3_info(_mp3(id3=True, tag="Info", lame=(576, 960))) == plain


def test_id3_footer_is_skipped() -> None:
    data = bytearray(_mp3(id3=True, tag="Info", lame=(576, 960)))
    data[5] = 0x10
    # A footer whose bytes look like a frame sync: a reader that scans it finds no Info tag.
    data[10 + 127 : 10 + 127] = bytes([0xFF, 0xFB, 0x90, 0xC0]) + bytes(6)
    assert read_mp3_info(bytes(data)).tag == "Info"


@pytest.mark.parametrize(
    ("data", "problem"),
    [
        (b"", "no frame sync"),
        (b"ID3\x04\x00\x00", "ID3 header is truncated"),
        (b"ID3\x04\x00\x00\x00\x00\x00\x00", "no frame sync"),
        (bytes(1024), "no frame sync"),
        (b"\xff\xfb", "frame header is truncated"),
        (_frame_header("2", 1, True, layer_bits=2) + bytes(50), "not Layer III"),
        (b"\xff\xea\x90\xc0" + bytes(50), "reserved MPEG version"),
        (_frame_header("2", 3, True) + bytes(50), "reserved sample rate"),
        (_mp3(tag="Info", flags=15, lame=None)[: 4 + 9 + 8], "Info frames"),
        (_mp3(tag="Info", flags=15, lame=None)[: 4 + 9 + 6], "Info flags"),
        (_mp3(tag="Info", flags=15, lame=None)[: 4 + 9 + 12 + 50], "Info TOC"),
        (_mp3(tag="Info", lame=(576, 960))[:-5], "LAME extension"),
        (_mp3(tag="VBRI")[: 36 + 10], "VBRI"),
    ],
)
def test_malformed_inputs_raise_audio_error(data: bytes, problem: str) -> None:
    with pytest.raises(AudioError, match=problem):
        read_mp3_info(data)


def _click_train() -> bytes:
    rate = 24000
    clicks = np.zeros(2 * rate, dtype=np.float32)
    t = np.arange(rate * 30 // 1000) / rate
    burst = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    for ms in (200, 700, 1200):
        start = rate * ms // 1000
        clicks[start : start + len(burst)] = burst
    return clicks.tobytes()


def _encode(ffmpeg: str, tmp_path: Path, write_xing: int) -> bytes:
    out = tmp_path / f"click{write_xing}.mp3"
    args = ["-f", "f32le", "-ar", "24000", "-ac", "1", "-i", "pipe:0"]
    args += ["-c:a", "libmp3lame", "-b:a", "48k", "-write_xing", str(write_xing), str(out)]
    cmd = [ffmpeg, "-y", "-loglevel", "error", *args]
    subprocess.run(cmd, input=_click_train(), check=True)  # noqa: S603
    return out.read_bytes()


def _onset(data: bytes) -> float:
    onset = first_onset_ms(decode(data), 48000)
    assert onset is not None
    return onset


def test_ffmpeg_honours_info_tag_delay(ffmpeg_bin: str, tmp_path: Path) -> None:
    data = _encode(ffmpeg_bin, tmp_path, 1)
    info = read_mp3_info(data)
    assert info.tag == "Info"
    assert info.encoder is not None
    assert info.encoder.startswith(("Lavc", "LAME"))
    assert info.encoder_delay == 576
    assert info.encoder_padding is not None
    assert 0 < info.encoder_padding <= 1152
    assert abs(_onset(data) - 200) <= 3


def test_ffmpeg_without_tag_keeps_priming(ffmpeg_bin: str, tmp_path: Path) -> None:
    data = _encode(ffmpeg_bin, tmp_path, 0)
    assert read_mp3_info(data).tag is None
    expected = (576 + DECODER_DELAY_SAMPLES) * 1000 / 24000
    assert abs(_onset(data) - 200 - expected) <= 3


def test_id3_body_and_footer_are_never_searched_for_a_frame() -> None:
    decoy = _frame_header("1", 0, False)  # MPEG 1 stereo 44.1 kHz; the real frame is MPEG 2
    real = _mp3(tag="Info", lame=(576, 960))
    header = bytearray(b"ID3\x04\x00\x10" + bytes([0, 0, 0, 40]))
    body = bytearray(40)
    body[10:14] = decoy
    footer = bytearray(10)
    footer[2:6] = decoy
    info = read_mp3_info(bytes(header + body + footer + real))
    assert (info.mpeg_version, info.sample_rate, info.tag) == ("2", 24000, "Info")


# The first 260 bytes (ID3v2 tag, then the Info frame with its LAME extension) of a file
# written by ffmpeg 9.0.2 with libmp3lame: lavfi sine 440 Hz, 24000 Hz mono, 1.234 s,
# 48 kbit/s, -write_xing 1. The encoder reports delay 576 and padding 912 (0x240 and
# 0x390), which the synthetic builder above does not produce, so the bit masks are
# checked against bytes this file did not write.
_REAL_LAVC_INFO_FRAME = bytes.fromhex(
    "49443304000000000022545353450000000e0000034c61766636332e312e3130"
    "320000000000000000000000fff384c0000000000000000000496e666f000000"
    "0f0000003600001f20000a0f0f141418181d1d2121262b2b2f2f343439393d3d"
    "4242464b4b5050545459595e5e6262676b6b7070757579797e7e8387878c8c90"
    "9095959a9a9e9ea3a8a8acacb1b1b5b5bababfbfc3c8c8cdcdd1d1d6d6dadadf"
    "dfe4e8e8ededf2f2f6f6fbfbff000000004c61766336332e312e000000000000"
    "0000000000002403900000000000001f20afc334010000000000000000000000"
    "000000000000000000000000fff364c4001c708e68175918003a926e172cb965"
    "b32cb982"
)


def test_real_lame_header_bytes_give_the_known_delay_and_padding() -> None:
    info = read_mp3_info(_REAL_LAVC_INFO_FRAME)
    assert info.tag == "Info"
    assert info.encoder == "Lavc63.1."
    assert (info.encoder_delay, info.encoder_padding) == (576, 912)
    assert (info.sample_rate, info.channels, info.frames) == (24000, 1, 54)
