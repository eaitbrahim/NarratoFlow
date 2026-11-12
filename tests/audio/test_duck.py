"""duck: SFX gain follows the narration with a fast attack and a gradual release."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from narratoflow.audio import io as aio
from narratoflow.audio.duck import DUCK_DB, duck, duck_gain_db
from narratoflow.audio.io import Samples

SR = aio.SAMPLE_RATE


def _at(ms: int) -> int:
    return SR * ms // 1000


def _tone(ms: int, dbfs: float = -20.0) -> Samples:
    t = np.arange(_at(ms)) / SR
    amp = 10 ** (dbfs / 20) * np.sqrt(2)
    out: Samples = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return out


def _noise(n: int) -> Samples:
    return (np.random.default_rng(1).standard_normal(n) * 0.1).astype(np.float32)


def _rms(x: Samples) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def test_sfx_under_steady_voice_reduced() -> None:
    voice = _tone(1000)
    sfx = _noise(len(voice))
    out = duck(sfx, voice)
    ratio_db = 20 * np.log10(_rms(out[_at(100) :]) / _rms(sfx[_at(100) :]))
    assert abs(ratio_db + DUCK_DB) <= 0.5
    assert out.dtype == sfx.dtype


def test_sfx_in_silence_unchanged() -> None:
    sfx = _noise(SR)
    out = duck(sfx, np.zeros(SR, dtype=np.float32))
    assert np.array_equal(out, sfx)


def test_gain_bounded() -> None:
    voice = (np.random.default_rng(2).standard_normal(SR) * 0.05).astype(np.float32)
    voice[_at(300) : _at(600)] = 0.0
    gain = duck_gain_db(voice)
    assert gain.min() >= -DUCK_DB - 1e-9
    assert gain.max() <= 0.0


def test_release_is_gradual() -> None:
    voice = np.concatenate([_tone(500), np.zeros(_at(1000), dtype=np.float32)])
    gain = duck_gain_db(voice)
    assert -DUCK_DB < gain[_at(600)] < 0.0
    assert abs(gain[_at(500 + 250 + 20)]) <= 0.1


def test_attack_is_fast() -> None:
    voice = np.concatenate([np.zeros(_at(200), dtype=np.float32), _tone(800)])
    gain = duck_gain_db(voice)
    assert abs(gain[_at(200 + 30 + 20)] + DUCK_DB) <= 0.1


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="samples"):
        duck(np.zeros(10, dtype=np.float32), np.zeros(11, dtype=np.float32))


def test_float64_sfx_keeps_dtype_on_both_paths() -> None:
    voice = _tone(500)
    sfx: Any = _noise(len(voice)).astype(np.float64)
    assert duck(sfx, voice).dtype == np.float64
    assert duck(sfx, np.zeros(len(voice), dtype=np.float32)).dtype == np.float64
