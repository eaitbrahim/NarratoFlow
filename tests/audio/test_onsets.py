"""Energy onset and offset detector."""

from __future__ import annotations

import math

import numpy as np
import pytest
from fixtures.audio import SR, tone

from narratoflow.audio.io import Samples
from narratoflow.audio.onsets import first_onset_ms, last_offset_ms, voiced_regions


def _zeros(ms: int) -> Samples:
    return np.zeros(SR * ms // 1000, dtype=np.float32)


def _bursts() -> Samples:
    return np.concatenate(
        [_zeros(200), tone(300), _zeros(100), tone(300), _zeros(400), tone(200), _zeros(300)]
    )


def _close(actual: float, expected: float) -> None:
    assert abs(actual - expected) <= 5


def test_regions_match_bursts() -> None:
    regions = voiced_regions(_bursts(), SR)
    assert len(regions) == 3
    for region, (start, end) in zip(regions, [(200, 500), (600, 900), (1300, 1500)], strict=True):
        _close(region.start_ms, start)
        _close(region.end_ms, end)


def test_min_gap_merges_short_pauses() -> None:
    regions = voiced_regions(_bursts(), SR, min_gap_ms=150)
    assert len(regions) == 2
    _close(regions[0].start_ms, 200)
    _close(regions[0].end_ms, 900)
    _close(regions[1].start_ms, 1300)
    _close(regions[1].end_ms, 1500)


def test_first_and_last() -> None:
    first = first_onset_ms(_bursts(), SR)
    last = last_offset_ms(_bursts(), SR)
    assert first is not None
    assert last is not None
    _close(first, 200)
    _close(last, 1500)


def test_short_click_dropped() -> None:
    click = np.concatenate([_zeros(100), tone(5), _zeros(100)])
    assert voiced_regions(click, SR) == []
    assert first_onset_ms(click, SR) is None


def test_silent_and_empty() -> None:
    for samples in (_zeros(300), np.zeros(0, dtype=np.float32)):
        assert voiced_regions(samples, SR) == []
        assert first_onset_ms(samples, SR) is None
        assert last_offset_ms(samples, SR) is None


def test_voiced_from_sample_zero() -> None:
    assert first_onset_ms(tone(100), SR) == 0.0


def test_end_is_capped_at_clip_length() -> None:
    # 103 ms is not a multiple of the 5 ms window: the last window is partial.
    regions = voiced_regions(tone(103), SR)
    assert regions[-1].end_ms == pytest.approx(103.0)


def test_quiet_tone_below_threshold() -> None:
    assert voiced_regions(tone(300, dbfs=-60), SR) == []


def test_bad_arguments() -> None:
    with pytest.raises(ValueError, match="sample_rate"):
        voiced_regions(tone(100), 0)
    with pytest.raises(ValueError, match="window_ms"):
        voiced_regions(tone(100), 100, window_ms=1)


_WIN = SR * 5 // 1000


def _windows(levels: list[float]) -> Samples:
    """Constant-level 5 ms windows, so each window's RMS is exactly its level."""
    return np.concatenate([np.full(_WIN, level, dtype=np.float32) for level in levels])


def _spans(
    samples: Samples, *, min_gap_ms: int = 60, min_voiced_ms: int = 20, dbfs: float = -45.0
) -> list[tuple[float, float]]:
    regions = voiced_regions(
        samples, SR, threshold_dbfs=dbfs, min_gap_ms=min_gap_ms, min_voiced_ms=min_voiced_ms
    )
    return [(r.start_ms, r.end_ms) for r in regions]


def test_window_exactly_at_threshold_is_not_voiced() -> None:
    half = 20 * math.log10(0.5)
    assert _spans(_windows([0.0, 0.5, 0.5, 0.5, 0.5, 0.0]), dbfs=half) == []
    assert _spans(_windows([0.0, 0.51, 0.51, 0.51, 0.51, 0.0]), dbfs=half) == [(5.0, 25.0)]


def test_run_exactly_min_voiced_is_kept() -> None:
    assert _spans(_windows([0.0] + [0.5] * 4 + [0.0])) == [(5.0, 25.0)]
    assert _spans(_windows([0.0] + [0.5] * 3 + [0.0])) == []


def test_gap_exactly_min_gap_is_not_merged() -> None:
    gap = [0.0] * 12  # 60 ms
    assert _spans(_windows([0.5] * 4 + gap + [0.5] * 4)) == [(0.0, 20.0), (80.0, 100.0)]
    assert _spans(_windows([0.5] * 4 + [0.0] * 11 + [0.5] * 4)) == [(0.0, 95.0)]


def test_merge_extends_the_latest_region_only() -> None:
    far = [0.0] * 40
    near = [0.0] * 2
    levels = [0.5] * 4 + far + [0.5] * 4 + near + [0.5] * 4
    assert _spans(_windows(levels)) == [(0.0, 20.0), (220.0, 270.0)]


def test_partial_last_window_uses_its_real_size() -> None:
    # 10 loud samples at -40 dBFS in a 240-sample window: only the real size clears -45.
    tail = np.full(10, 0.01, dtype=np.float32)
    samples = np.concatenate([np.zeros(2 * _WIN, dtype=np.float32), tail])
    regions = voiced_regions(samples, SR, min_voiced_ms=1)
    assert len(regions) == 1
    assert regions[0].start_ms == 10.0
    assert regions[0].end_ms == pytest.approx(len(samples) * 1000 / SR)


def test_window_of_exactly_one_sample_is_allowed() -> None:
    samples = np.concatenate([np.zeros(10), np.full(30, 0.5), np.zeros(10)]).astype(np.float32)
    regions = voiced_regions(samples, 1000, window_ms=1, min_voiced_ms=20, min_gap_ms=5)
    assert [(r.start_ms, r.end_ms) for r in regions] == [(10.0, 40.0)]


def test_regions_are_exact_at_rates_where_window_is_not_whole_samples() -> None:
    for rate in (44100, 22050):
        t = np.arange(rate * 300 // 1000) / rate
        burst = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        clip = np.concatenate(
            [np.zeros(rate * 10, dtype=np.float32), burst, np.zeros(rate, dtype=np.float32)]
        )
        regions = voiced_regions(clip, rate)
        assert len(regions) == 1
        _close(regions[0].start_ms, 10000)
        _close(regions[0].end_ms, 10300)
