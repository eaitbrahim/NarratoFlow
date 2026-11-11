"""Audio fixtures generated in code; no audio files live in git."""

from __future__ import annotations

import io
import shutil
import subprocess
import wave

import numpy as np
from numpy.typing import NDArray

SR = 48_000


def _n(ms: int) -> int:
    return SR * ms // 1000


def _amp(dbfs: float) -> float:
    return float(10 ** (dbfs / 20))


def tone(ms: int, hz: float = 440, dbfs: float = -20, lead_ms: int = 0) -> NDArray[np.float32]:
    """Sine tone; ``dbfs`` is the RMS level (peak is 3 dB higher)."""
    t = np.arange(_n(ms)) / SR
    body = (np.sin(2 * np.pi * hz * t) * _amp(dbfs) * np.sqrt(2)).astype(np.float32)
    return np.concatenate([np.zeros(_n(lead_ms), dtype=np.float32), body])


def speechlike(
    words: int,
    word_ms: int = 300,
    gap_ms: int = 80,
    lead_ms: int = 0,
    dbfs: float = -20,
    seed: int = 0,
) -> NDArray[np.float32]:
    """Gated band-limited noise: bursts of 'words' separated by gaps."""
    rng = np.random.default_rng(seed)
    parts: list[NDArray[np.float32]] = [np.zeros(_n(lead_ms), dtype=np.float32)]
    for _ in range(words):
        noise = rng.standard_normal(_n(word_ms))
        spec = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(noise.size, 1 / SR)
        spec[(freqs < 200) | (freqs > 3400)] = 0
        band = np.fft.irfft(spec, noise.size)
        band = band / (np.sqrt(np.mean(band**2)) + 1e-12) * _amp(dbfs)
        env = np.minimum(
            1.0, np.minimum(np.arange(band.size), band.size - np.arange(band.size)) / 240
        )
        parts.append((band * env).astype(np.float32))
        parts.append(np.zeros(_n(gap_ms), dtype=np.float32))
    return np.concatenate(parts)


def to_wav_bytes(samples: NDArray[np.float32]) -> bytes:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def to_mp3_bytes(samples: NDArray[np.float32]) -> bytes:
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    proc = subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(SR),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            "libmp3lame",
            "-f",
            "mp3",
            "pipe:1",
        ],
        input=samples.astype("<f4").tobytes(),
        capture_output=True,
        check=True,
        timeout=120,
    )
    return proc.stdout
