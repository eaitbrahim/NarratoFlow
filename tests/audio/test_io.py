from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from fixtures.audio import speechlike, to_mp3_bytes, to_wav_bytes, tone

from narratoflow.audio import io as aio
from narratoflow.errors import AudioError


def test_decode_wav_and_mp3_roundtrip_length(ffmpeg_bin: str) -> None:
    src = tone(2000)
    wav = aio.decode(to_wav_bytes(src))
    assert wav.dtype == np.float32
    assert abs(len(wav) - len(src)) <= aio.ms_to_samples(30)
    mp3 = aio.decode(to_mp3_bytes(src))
    assert abs(len(mp3) - len(src)) <= aio.ms_to_samples(60)


def test_decode_garbage_raises_audio_error(ffmpeg_bin: str) -> None:
    with pytest.raises(AudioError) as exc:
        aio.decode(b"SECRETGARBAGE" * 10)
    assert "SECRETGARBAGE" not in str(exc.value)


def test_measure_is_relative_exact(ffmpeg_bin: str) -> None:
    x = tone(3000, hz=1000)
    a = aio.measure(x).integrated_lufs
    b = aio.measure(aio.apply_gain(x, 6)).integrated_lufs
    assert a is not None
    assert b is not None
    assert abs((b - a) - 6) <= 0.2


def test_master_hits_target(ffmpeg_bin: str) -> None:
    out, loud = aio.master(speechlike(20))
    assert loud.integrated_lufs is not None
    assert abs(loud.integrated_lufs + 14) <= 1.0
    assert loud.true_peak_dbtp <= -1.0
    assert out.dtype == np.float32


def test_master_short_part_no_clipping(ffmpeg_bin: str) -> None:
    out, _ = aio.master(tone(1500, dbfs=0))
    assert float(np.abs(out).max()) <= 10 ** (-1 / 20) + 1e-3


def test_master_silence_returned_unchanged(ffmpeg_bin: str) -> None:
    sil = aio.silence(1000)
    out, loud = aio.master(sil)
    assert loud.integrated_lufs is None
    assert np.array_equal(out, sil)


def test_trim_silence_reports_lead() -> None:
    x = speechlike(3, lead_ms=500)
    out, lead = aio.trim_silence(x, keep_ms=20)
    assert 475 <= lead <= 485
    assert len(out) < len(x)


def test_trim_all_silent() -> None:
    out, lead = aio.trim_silence(aio.silence(500))
    assert len(out) == 0
    assert lead == 0


def test_fade_gain_helpers() -> None:
    x = np.ones(aio.ms_to_samples(100), dtype=np.float32)
    f = aio.fade(x, 10)
    assert f[0] == 0
    assert f[-1] == 0
    assert f[len(f) // 2] == 1
    assert np.allclose(aio.apply_gain(x, -6.0206), 0.5, atol=1e-3)


def test_encode_flac_atomic(ffmpeg_bin: str, tmp_path: Path) -> None:
    path = tmp_path / "out.flac"
    aio.encode_flac(tone(500), path)
    assert path.exists()
    assert [p.name for p in tmp_path.iterdir()] == ["out.flac"]
    back = aio.decode(path.read_bytes())
    assert abs(len(back) - aio.ms_to_samples(500)) <= aio.ms_to_samples(30)


def test_missing_ffmpeg_raises_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(AudioError, match="ffmpeg not found"):
        aio.require_ffmpeg()
