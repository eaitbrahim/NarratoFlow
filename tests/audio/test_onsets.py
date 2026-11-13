"""Energy onset and offset detector."""

from __future__ import annotations

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
