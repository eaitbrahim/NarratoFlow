"""Energy onsets and offsets of voiced regions, as ground truth for word timing.

Energy onsets are reliable at the sentence start, after pauses and at the final
word's end. They are not reliable inside connected speech, where words run together
without a dip in energy. Resolution is one window (5 ms by default).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from narratoflow.audio.io import Samples

WINDOW_MS = 5
THRESHOLD_DBFS = -45.0  # same as mixer.ONSET_THRESHOLD_DBFS
MIN_VOICED_MS = 20
MIN_GAP_MS = 60


@dataclass(frozen=True)
class VoicedRegion:
    start_ms: float
    end_ms: float


def voiced_regions(
    samples: Samples,
    sample_rate: int,
    *,
    threshold_dbfs: float = THRESHOLD_DBFS,
    window_ms: int = WINDOW_MS,
    min_voiced_ms: int = MIN_VOICED_MS,
    min_gap_ms: int = MIN_GAP_MS,
) -> list[VoicedRegion]:
    """Voiced regions of `samples`: RMS windows above the threshold, gaps shorter than
    `min_gap_ms` merged first, then regions shorter than `min_voiced_ms` dropped."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    win = sample_rate * window_ms // 1000
    if win < 1:
        raise ValueError(f"window_ms {window_ms} is under one sample at {sample_rate} Hz")
    if len(samples) == 0:
        return []
    count = math.ceil(len(samples) / win)
    padded = np.zeros(count * win, dtype=np.float64)
    padded[: len(samples)] = samples
    sums = np.sum(padded.reshape(count, win) ** 2, axis=1)
    sizes = np.full(count, win, dtype=np.float64)
    sizes[-1] = len(samples) - (count - 1) * win
    voiced = np.sqrt(sums / sizes) > 10 ** (threshold_dbfs / 20)
    # Run boundaries as window indices: [start, end) pairs.
    edges = np.flatnonzero(np.diff(np.concatenate([[False], voiced, [False]]).astype(np.int8)))
    runs = [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]
    merged: list[tuple[int, int]] = []
    for start, end in runs:
        if merged and (start - merged[-1][1]) * window_ms < min_gap_ms:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    clip_ms = len(samples) * 1000 / sample_rate
    return [
        VoicedRegion(float(start * window_ms), min(float(end * window_ms), clip_ms))
        for start, end in merged
        if (end - start) * window_ms >= min_voiced_ms
    ]


def first_onset_ms(
    samples: Samples,
    sample_rate: int,
    *,
    threshold_dbfs: float = THRESHOLD_DBFS,
    window_ms: int = WINDOW_MS,
    min_voiced_ms: int = MIN_VOICED_MS,
    min_gap_ms: int = MIN_GAP_MS,
) -> float | None:
    """Start of the first voiced region, or None when nothing is voiced."""
    regions = voiced_regions(
        samples,
        sample_rate,
        threshold_dbfs=threshold_dbfs,
        window_ms=window_ms,
        min_voiced_ms=min_voiced_ms,
        min_gap_ms=min_gap_ms,
    )
    return regions[0].start_ms if regions else None


def last_offset_ms(
    samples: Samples,
    sample_rate: int,
    *,
    threshold_dbfs: float = THRESHOLD_DBFS,
    window_ms: int = WINDOW_MS,
    min_voiced_ms: int = MIN_VOICED_MS,
    min_gap_ms: int = MIN_GAP_MS,
) -> float | None:
    """End of the last voiced region, or None when nothing is voiced."""
    regions = voiced_regions(
        samples,
        sample_rate,
        threshold_dbfs=threshold_dbfs,
        window_ms=window_ms,
        min_voiced_ms=min_voiced_ms,
        min_gap_ms=min_gap_ms,
    )
    return regions[-1].end_ms if regions else None
