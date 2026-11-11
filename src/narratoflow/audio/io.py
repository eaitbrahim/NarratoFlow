"""Audio I/O and loudness through ffmpeg. Internal audio: mono float32 at 48 kHz."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from narratoflow.errors import AudioError

SAMPLE_RATE = 48_000
FFMPEG_TIMEOUT_S = 120
MIN_GATED_MS = 400

Samples = NDArray[np.float32]


def require_ffmpeg() -> Path:
    found = shutil.which("ffmpeg")
    if found is None:
        raise AudioError("ffmpeg not found; install it (brew install ffmpeg / apt install ffmpeg)")
    return Path(found)


def _run(args: list[str], data: bytes) -> tuple[bytes, str]:
    cmd = [str(require_ffmpeg()), "-hide_banner", "-nostdin", *args]
    try:
        proc = subprocess.run(  # noqa: S603
            cmd, input=data, capture_output=True, check=False, timeout=FFMPEG_TIMEOUT_S
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioError(f"ffmpeg timed out after {FFMPEG_TIMEOUT_S}s") from exc
    stderr = proc.stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(line[:200] for line in stderr.strip().splitlines()[-5:])
        raise AudioError(f"ffmpeg failed (exit {proc.returncode}): {tail}")
    return proc.stdout, stderr


def ms_to_samples(ms: int) -> int:
    return SAMPLE_RATE * ms // 1000


def silence(ms: int) -> Samples:
    return np.zeros(ms_to_samples(ms), dtype=np.float32)


def _raw(samples: Samples) -> bytes:
    return np.ascontiguousarray(samples, dtype="<f4").tobytes()


def _from_raw(data: bytes) -> Samples:
    usable = len(data) - len(data) % 4
    return np.frombuffer(data[:usable], dtype="<f4").astype(np.float32)


def decode(data: bytes) -> Samples:
    out, _ = _run(
        [
            "-v",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "pipe:1",
        ],
        data,
    )
    samples = _from_raw(out)
    if samples.size == 0:
        raise AudioError("audio decodes to no samples")
    return samples


def resample(samples: Samples, rate: int) -> Samples:
    """Mono float32 at `SAMPLE_RATE` resampled to `rate` Hz through ffmpeg."""
    if rate <= 0:
        raise AudioError(f"invalid sample rate: {rate}")
    out, _ = _run(
        [
            "-v",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(rate),
            "pipe:1",
        ],
        _raw(samples),
    )
    return _from_raw(out)


def encode_flac(samples: Samples, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        os.close(fd)
    except OSError as exc:
        raise AudioError(f"cannot write {path}: {exc}") from exc
    tmp = Path(tmp_name)
    try:
        out, _ = _run(
            [
                "-v",
                "error",
                "-f",
                "f32le",
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                "-i",
                "pipe:0",
                "-sample_fmt",
                "s16",
                "-ar",
                str(SAMPLE_RATE),
                "-f",
                "flac",
                "pipe:1",
            ],
            _raw(samples),
        )
        tmp.write_bytes(out)
        tmp.chmod(0o644)
        os.replace(tmp, path)
    except OSError as exc:
        raise AudioError(f"cannot write {path}: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)


@dataclass(frozen=True)
class Loudness:
    integrated_lufs: float | None
    true_peak_dbtp: float | None
    lra: float
    threshold: float
    offset: float


def _num(value: object) -> float:
    return float(str(value))


def _loudnorm_args(params: str) -> list[str]:
    return [
        "-v",
        "info",
        "-f",
        "f32le",
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-af",
        params,
    ]


def measure(samples: Samples) -> Loudness:
    _, stderr = _run(
        [*_loudnorm_args("loudnorm=I=-14:TP=-1:LRA=11:print_format=json"), "-f", "null", "-"],
        _raw(samples),
    )
    too_short = len(samples) < ms_to_samples(MIN_GATED_MS)
    return parse_loudnorm(stderr, gated=not too_short)


def parse_loudnorm(stderr: str, *, gated: bool) -> Loudness:
    """The last loudnorm JSON block in ffmpeg's `stderr` as a `Loudness`.

    `gated` False (a clip too short for gating) drops the integrated value; non-finite
    values become None (integrated, true peak) or 0.0 (the rest)."""
    blocks = re.findall(r"\{[^{}]*\}", stderr, flags=re.DOTALL)
    if not blocks:
        raise AudioError("ffmpeg loudnorm produced no measurement")
    try:
        data = json.loads(blocks[-1])
        integrated = _num(data["input_i"])
        peak = _num(data["input_tp"])
        lra = _num(data["input_lra"])
        thresh = _num(data["input_thresh"])
        offset = _num(data["target_offset"])
    except (ValueError, KeyError, TypeError) as exc:
        raise AudioError("ffmpeg loudnorm measurement could not be parsed") from exc
    gated = gated and math.isfinite(integrated)
    return Loudness(
        integrated if gated else None,
        peak if math.isfinite(peak) else None,
        _finite(lra),
        _finite(thresh),
        _finite(offset),
    )


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def master(
    samples: Samples,
    *,
    target_lufs: float = -14.0,
    true_peak_dbtp: float = -1.0,
    lra: float = 11.0,
) -> tuple[Samples, Loudness]:
    first = measure(samples)
    if first.integrated_lufs is None or first.true_peak_dbtp is None:
        return samples, first
    params = (
        f"loudnorm=I={target_lufs}:TP={true_peak_dbtp}:LRA={lra}"
        f":measured_I={first.integrated_lufs}:measured_TP={first.true_peak_dbtp}"
        f":measured_LRA={first.lra}:measured_thresh={first.threshold}"
        f":offset={first.offset}:linear=true"
    )
    out, _ = _run(
        [*_loudnorm_args(params), "-f", "f32le", "-ar", str(SAMPLE_RATE), "pipe:1"],
        _raw(samples),
    )
    mastered = _from_raw(out)
    return mastered, measure(mastered)


def trim_silence(
    samples: Samples,
    *,
    threshold_dbfs: float = -45.0,
    keep_ms: int = 20,
    keep_lead_ms: int | None = None,
) -> tuple[Samples, int]:
    win = ms_to_samples(10)
    if len(samples) == 0:
        return samples[:0], 0
    count = math.ceil(len(samples) / win)
    padded = np.zeros(count * win, dtype=np.float64)
    padded[: len(samples)] = samples
    sums = np.sum(padded.reshape(count, win) ** 2, axis=1)
    sizes = np.full(count, win, dtype=np.float64)
    sizes[-1] = len(samples) - (count - 1) * win
    rms = np.sqrt(sums / sizes)
    loud = np.flatnonzero(rms > 10 ** (threshold_dbfs / 20))
    if loud.size == 0:
        return samples[:0], 0
    lead_keep = ms_to_samples(keep_ms if keep_lead_ms is None else keep_lead_ms)
    start = max(0, int(loud[0]) * win - lead_keep)
    end = min(len(samples), (int(loud[-1]) + 1) * win + ms_to_samples(keep_ms))
    return samples[start:end].copy(), start * 1000 // SAMPLE_RATE


def fade(samples: Samples, ms: int = 10) -> Samples:
    out = samples.copy()
    n = min(ms_to_samples(ms), len(out) // 2)
    if n == 0:
        return out
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def apply_gain(samples: Samples, db: float) -> Samples:
    return (samples * np.float32(10 ** (db / 20))).astype(np.float32)
