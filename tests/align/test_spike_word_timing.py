import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from fixtures import audio as fx

from narratoflow.audio.words import TimedSpan
from narratoflow.models import WordTiming

ROOT = Path(__file__).parents[2]


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


spike = _script("spike_word_timing")


def _timings(starts: list[int], ends: list[int] | None = None) -> list[WordTiming]:
    ends = ends or [s + 200 for s in starts]
    return [
        WordTiming(text=f"w{i}", start_ms=s, end_ms=e)
        for i, (s, e) in enumerate(zip(starts, ends, strict=True))
    ]


def test_drift_stats_values() -> None:
    native = _timings([1000, 2040, 3100], [1200, 2300, 3500])
    reference = _timings([1000, 2000, 3000], [1200, 2200, 3300])
    stats = spike.drift_stats(native, reference)
    assert stats.words == 3
    assert stats.median_start_ms == 40
    assert stats.max_start_ms == 100
    assert stats.p95_start_ms == pytest.approx(94.0)
    assert stats.median_end_ms == 100
    assert stats.max_end_ms == 200


def test_drift_stats_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        spike.drift_stats(_timings([0, 10]), _timings([0]))
    other = [WordTiming(text="x", start_ms=0, end_ms=1)]
    with pytest.raises(ValueError, match="text"):
        spike.drift_stats(_timings([0]), other)
    with pytest.raises(ValueError, match="no words"):
        spike.drift_stats([], [])


def test_sentences_are_diacritized_arabic() -> None:
    sentences = spike.SENTENCES
    assert 10 <= len(sentences) <= 12
    harakat = {chr(c) for c in range(0x064B, 0x0653)}
    assert all(harakat & set(s) for s in sentences)
    assert max(len(s.split()) for s in sentences) > 25
    joined = "".join(sentences)
    for mark in tuple(chr(c) for c in (0xAB, 0xBB, 0x61F, 0x60C, 0x61B)):
        assert mark in joined
    for clitic in tuple(
        chr(c) + chr(0x64E if c in (0x648, 0x641, 0x643) else 0x650)
        for c in (0x648, 0x641, 0x628, 0x644, 0x643)
    ):
        assert any(w.startswith(clitic) for s in sentences for w in s.split())


def test_signed_deltas_keep_sign() -> None:
    native = _timings([1100, 2000], [1300, 2500])
    reference = _timings([1000, 2050], [1200, 2100])
    assert spike.signed_deltas(native, reference) == ([100, -50], [100, 400])


def test_signed_deltas_reject_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        spike.signed_deltas(_timings([0, 1]), _timings([0]))
    with pytest.raises(ValueError, match="text"):
        spike.signed_deltas(_timings([0]), [WordTiming(text="x", start_ms=0, end_ms=1)])
    with pytest.raises(ValueError, match="no words"):
        spike.signed_deltas([], [])


def test_offset_stats() -> None:
    stats = spike.offset_stats([100, 110, 120, 130])
    assert stats.mean_ms == pytest.approx(115.0)
    assert stats.median_ms == pytest.approx(115.0)
    assert stats.std_ms == pytest.approx(11.180339887, rel=1e-6)
    assert stats.negatives == 0
    assert spike.offset_stats([-5, 5, 10]).negatives == 1


def test_offset_stats_rejects_empty() -> None:
    with pytest.raises(ValueError, match="no words"):
        spike.offset_stats([])


def test_residual_after_offset() -> None:
    median, p95 = spike.residual([100, 110, 120, 130, 90], 110.0)
    assert median == pytest.approx(10.0)
    assert p95 == pytest.approx(20.0)
    with pytest.raises(ValueError, match="no words"):
        spike.residual([], 0.0)


def test_worst_end_gap() -> None:
    assert spike.worst_end_gap([10, -900, 40]) == (1, -900)
    with pytest.raises(ValueError, match="no words"):
        spike.worst_end_gap([])


SR = fx.SR
SENTENCE_ONE = "aa bb cc"
SENTENCE_TWO = "dd ee"


def _capture(index: int, sentence: str, audio: bytes, ctc: bool = False) -> object:
    words = sentence.split()
    edge = [TimedSpan(w, 190 + 700 * i, 450 + 700 * i) for i, w in enumerate(words)]
    ctc_spans = [TimedSpan(w, 200 + 700 * i, 500 + 700 * i) for i, w in enumerate(words)]
    return spike.Capture(index, sentence, audio, words, edge, ctc_spans if ctc else None)


def _bursts(words: int) -> np.ndarray:
    # Bursts at 200, 900, 1600 ms: a 400 ms gap after each 300 ms burst.
    return fx.speechlike(words, word_ms=300, gap_ms=400, lead_ms=200)


def _write_capture(directory: Path, ffmpeg_bin: str, ctc: bool = False) -> None:
    assert ffmpeg_bin
    sentences = [SENTENCE_ONE, SENTENCE_TWO]
    for n, (sentence, count) in enumerate(zip(sentences, [3, 2], strict=True), 1):
        audio = fx.to_mp3_bytes(_bursts(count))
        spike.save_capture(directory, _capture(n, sentence, audio, ctc))
    spike.write_manifest(
        directory, voice="v", edge_tts_version="1.0", captured_at="t", sentences=sentences
    )


def _no_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    real = importlib.util.find_spec

    def fake(name: str, package: str | None = None) -> object:
        if name in {"edge_tts", "ctc_forced_aligner"}:
            return None
        return real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


@pytest.mark.parametrize("ctc", [False, True])
def test_capture_round_trip(tmp_path: Path, ffmpeg_bin: str, ctc: bool) -> None:
    _write_capture(tmp_path, ffmpeg_bin, ctc)
    loaded = spike.load_captures(tmp_path)
    assert len(loaded) == 2
    audio = (tmp_path / "01.mp3").read_bytes()
    assert loaded[0] == _capture(1, SENTENCE_ONE, audio, ctc)
    assert (tmp_path / "01.ctc.json").exists() is ctc
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["version"] == 1
    assert manifest["edge_tts_version"] == "1.0"


@pytest.mark.parametrize("victim", ["01.edge.json", "01.words.json", "manifest.json"])
def test_load_captures_names_the_bad_file(tmp_path: Path, ffmpeg_bin: str, victim: str) -> None:
    _write_capture(tmp_path, ffmpeg_bin)
    (tmp_path / victim).unlink()
    with pytest.raises(ValueError, match=victim):
        spike.load_captures(tmp_path)
    (tmp_path / victim).write_text("{not json")
    with pytest.raises(ValueError, match=victim):
        spike.load_captures(tmp_path)


def test_load_captures_rejects_words_that_differ_from_the_sentence(
    tmp_path: Path, ffmpeg_bin: str
) -> None:
    _write_capture(tmp_path, ffmpeg_bin)
    (tmp_path / "01.words.json").write_text(json.dumps(["aa", "zz", "cc"]))
    with pytest.raises(ValueError, match=r"01\.words\.json"):
        spike.load_captures(tmp_path)


def test_wave_truth_on_bursts() -> None:
    samples = np.concatenate(
        [
            fx.speechlike(1, word_ms=300, gap_ms=80, lead_ms=200),
            fx.speechlike(1, word_ms=300, gap_ms=400),
            fx.speechlike(1, word_ms=300, gap_ms=0),
        ]
    )
    truth = spike.wave_truth(samples, SR)
    assert truth.first_onset_ms == pytest.approx(200, abs=10)
    assert truth.last_offset_ms == pytest.approx(1580, abs=10)
    assert len(truth.pause_onsets_ms) == 1
    assert truth.pause_onsets_ms[0] == pytest.approx(1280, abs=10)
    assert spike.wave_truth(np.zeros(SR, dtype=np.float32), SR) is None


def test_wave_deltas_signed() -> None:
    truth = spike.WaveTruth(first_onset_ms=200.0, last_offset_ms=1540.0, pause_onsets_ms=[1000.0])
    timings = _timings([90, 300, 600, 960], [200, 400, 700, 2400])
    got = spike.wave_deltas(timings, truth)
    assert got.first_start_ms == -110
    assert got.last_end_ms == 860
    assert got.pause_starts_ms == [(3, -40)]
    far = spike.WaveTruth(200.0, 1540.0, [1400.0])
    assert spike.wave_deltas(timings, far).pause_starts_ms == []
    at_word_zero = spike.WaveTruth(200.0, 1540.0, [90.0])
    assert spike.wave_deltas(timings, at_word_zero, window_ms=100).pause_starts_ms == []


def test_labels_round_trip_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"

    def write(version: int = 1, sentence: int = 1, onset: int = 812) -> None:
        body = {
            "version": version,
            "labels": [{"sentence": sentence, "word": 3, "onset_ms": onset}],
        }
        path.write_text(json.dumps(body))

    write()
    assert spike.load_labels(path) == {1: [(3, 812)]}
    for bad in ({"version": 2}, {"sentence": 99}, {"onset": -1}):
        write(**bad)
        with pytest.raises(ValueError, match="labels"):
            spike.load_labels(path)
    timings = _timings([0, 100, 200, 1000])
    assert spike.label_deltas(timings, [(3, 940)]) == [60]
    with pytest.raises(ValueError, match="word"):
        spike.label_deltas(timings, [(4, 940)])


def test_from_runs_offline(
    tmp_path: Path,
    ffmpeg_bin: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_capture(tmp_path, ffmpeg_bin)
    _no_extras(monkeypatch)
    assert spike.main(["--from", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "MP3 header" in out
    assert "edge vs waveform" in out
    assert "CTC: not available" in out
    assert "ctc first" not in out
    assert "edge first" in out
    assert "estimate minus waveform" in out


def test_from_uses_saved_ctc_spans(
    tmp_path: Path,
    ffmpeg_bin: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_capture(tmp_path, ffmpeg_bin, ctc=True)
    _no_extras(monkeypatch)
    assert spike.main(["--from", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "ctc first" in out
    assert "CTC: not available" not in out


def test_from_scores_labels(
    tmp_path: Path,
    ffmpeg_bin: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_capture(tmp_path, ffmpeg_bin)
    _no_extras(monkeypatch)
    labels = [
        {"sentence": 1, "word": 1, "onset_ms": 880},
        {"sentence": 2, "word": 1, "onset_ms": 920},
    ]
    (tmp_path / "labels.json").write_text(json.dumps({"version": 1, "labels": labels}))
    assert spike.main(["--from", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "labelled words: 2" in out
    assert "signed median" in out


def test_save_without_edge_extra_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_extras(monkeypatch)
    assert spike.main(["--save", str(tmp_path / "cap")]) == 2
    assert "uv sync --extra tts-edge" in capsys.readouterr().err
    assert not (tmp_path / "cap").exists()


def test_save_and_from_together_is_usage_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        spike.main(["--save", str(tmp_path / "a"), "--from", str(tmp_path / "b")])
    assert exc.value.code == 2


def test_edge_tts_version_falls_back_to_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib import metadata

    def missing(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing)
    assert spike.edge_tts_version() == "unknown"


def test_offset_stats_accepts_float() -> None:
    assert spike.offset_stats([0.5, 1.5]).median_ms == pytest.approx(1.0)
