"""Duck the marker SFX under narration with a frame-level, slew-limited gain."""

from __future__ import annotations

import numpy as np

from narratoflow.audio import io as aio
from narratoflow.audio.io import Samples

DUCK_DB = 8.0
DUCK_THRESHOLD_DBFS = -40.0
DUCK_FRAME_MS = 10
DUCK_ATTACK_MS = 30
DUCK_RELEASE_MS = 250


def _frame_targets(voice: Samples, frame: int) -> np.ndarray:
    """Per-frame target gain in dB: -DUCK_DB where the voice is above the threshold."""
    count = -(-len(voice) // frame)
    padded = np.zeros(count * frame, dtype=np.float64)
    padded[: len(voice)] = voice
    rms = np.sqrt(np.mean(padded.reshape(count, frame) ** 2, axis=1))
    floor = 10 ** (DUCK_THRESHOLD_DBFS / 20)
    return np.where(rms > floor, -DUCK_DB, 0.0)


def duck_gain_db(voice: Samples) -> np.ndarray:
    """Per-sample duck gain in dB, in [-DUCK_DB, 0], for a voice bus."""
    frame = aio.ms_to_samples(DUCK_FRAME_MS)
    if len(voice) == 0:
        return np.zeros(0, dtype=np.float64)
    targets = _frame_targets(voice, frame)
    fall = DUCK_DB * DUCK_FRAME_MS / DUCK_ATTACK_MS
    rise = DUCK_DB * DUCK_FRAME_MS / DUCK_RELEASE_MS
    gains = np.empty_like(targets)
    current = 0.0
    for i, target in enumerate(targets):  # frames, not samples
        current = max(target, current - fall) if target < current else min(target, current + rise)
        gains[i] = current
    centres = (np.arange(len(gains)) + 0.5) * frame
    return np.asarray(np.interp(np.arange(len(voice)), centres, gains))


def duck(sfx: Samples, voice: Samples) -> Samples:
    """`sfx` lowered by up to DUCK_DB wherever `voice` is speaking."""
    if len(sfx) != len(voice):
        raise ValueError(f"sfx has {len(sfx)} samples but voice has {len(voice)}")
    gain = duck_gain_db(voice)
    if not np.any(gain < 0.0):
        return sfx
    return (sfx * (10 ** (gain / 20))).astype(np.float32)
