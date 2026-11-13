import dataclasses
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from fixtures import audio as fx

from narratoflow.audio.words import TimedSpan, tokens
from narratoflow.models import Language, WordTiming
from narratoflow.tts import Speech, SpeechRequest

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


def _word_spans(text: str, shift: int = 0) -> list[TimedSpan]:
    return [
        TimedSpan(w, 190 + shift + 700 * i, 450 + shift + 700 * i)
        for i, w in enumerate(tokens(text))
    ]


BROKEN_ALIGNER = False


class _FakeAligner:
    def align(
        self, samples: object, sample_rate: int, text: str, language: Language
    ) -> list[TimedSpan]:
        return [TimedSpan("zz", 0, 10)] if BROKEN_ALIGNER else _word_spans(text, shift=10)


def _fake_services(
    monkeypatch: pytest.MonkeyPatch, mp3: bytes, *, ctc: bool = True, broken: bool = False
) -> None:
    """Pretend both extras are installed, with no network and no model."""
    real = importlib.util.find_spec
    present = {"edge_tts"} | ({"ctc_forced_aligner"} if ctc else set())

    def fake_find(name: str, package: str | None = None) -> object:
        if name in {"edge_tts", "ctc_forced_aligner"}:
            return object() if name in present else None
        return real(name, package)

    class FakeEdge:
        def synthesize(self, req: SpeechRequest) -> Speech:
            spans = [TimedSpan("zz", 0, 10)] if broken else _word_spans(req.text)
            return Speech(mp3, tuple(spans), "edge", "fake", None)

    edge = types.ModuleType("narratoflow.tts.edge")
    monkeypatch.setattr(edge, "EdgeProvider", FakeEdge, raising=False)
    align = types.ModuleType("narratoflow.align.ctc")
    monkeypatch.setattr(align, "CtcAligner", _FakeAligner, raising=False)
    monkeypatch.setattr(sys.modules[__name__], "BROKEN_ALIGNER", broken)
    monkeypatch.setattr(importlib.util, "find_spec", fake_find)
    monkeypatch.setitem(sys.modules, "narratoflow.tts.edge", edge)
    monkeypatch.setitem(sys.modules, "narratoflow.align.ctc", align)


@pytest.fixture
def mp3(ffmpeg_bin: str) -> bytes:
    assert ffmpeg_bin
    return fx.to_mp3_bytes(_bursts(3))


@pytest.mark.parametrize("ctc", [True, False])
def test_save_captures_every_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mp3: bytes, ctc: bool
) -> None:
    _fake_services(monkeypatch, mp3, ctc=ctc)
    monkeypatch.setattr(spike.metadata, "version", lambda name: "9.9")
    target = tmp_path / "cap"
    assert spike.main(["--save", str(target)]) == 0
    captures = spike.load_captures(target)
    assert [c.sentence for c in captures] == list(spike.SENTENCES)
    assert all(c.audio == mp3 for c in captures)
    assert all((c.ctc is not None) is ctc for c in captures)
    manifest = json.loads((target / "manifest.json").read_text())
    assert manifest["voice"] == spike.VOICE
    assert manifest["edge_tts_version"] == "9.9"
    assert manifest["captured_at"]


def test_legacy_run_prints_the_report(
    monkeypatch: pytest.MonkeyPatch, mp3: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_services(monkeypatch, mp3)
    assert spike.main([]) == 0
    out = capsys.readouterr().out
    assert "| # | words | spans |" in out
    assert "Absolute totals over" in out
    assert "Native mapping failures: none" in out
    assert "Pooled signed end median" in out


def test_legacy_run_lists_mapping_failures(
    monkeypatch: pytest.MonkeyPatch, mp3: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_services(monkeypatch, mp3, broken=True)
    assert spike.main([]) == 0
    out = capsys.readouterr().out
    assert "Native mapping failures:\n- sentence 1: at 0:" in out
    assert "CTC mapping failures:\n- sentence 1: at 0:" in out
    assert "Absolute totals" not in out


def test_legacy_run_without_extras_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_extras(monkeypatch)
    assert spike.main([]) == 2
    assert spike.EXTRAS_HINT in capsys.readouterr().err


@pytest.mark.parametrize("body", ["[1]", '[["a", 1, "b"]]'])
def test_malformed_spans_name_the_file(tmp_path: Path, ffmpeg_bin: str, body: str) -> None:
    _write_capture(tmp_path, ffmpeg_bin)
    (tmp_path / "01.edge.json").write_text(body)
    with pytest.raises(ValueError, match=r"01\.edge\.json"):
        spike.load_captures(tmp_path)


def test_missing_mp3_names_the_file(tmp_path: Path, ffmpeg_bin: str) -> None:
    _write_capture(tmp_path, ffmpeg_bin)
    (tmp_path / "01.mp3").unlink()
    with pytest.raises(ValueError, match=r"01\.mp3"):
        spike.load_captures(tmp_path)


@pytest.mark.parametrize("body", ["[]", '{"sentences": [1]}', '{"sentences": "x"}'])
def test_manifest_must_list_sentences(tmp_path: Path, ffmpeg_bin: str, body: str) -> None:
    _write_capture(tmp_path, ffmpeg_bin)
    (tmp_path / "manifest.json").write_text(body)
    with pytest.raises(ValueError, match=r"manifest\.json"):
        spike.load_captures(tmp_path)


@pytest.mark.parametrize(
    "entry", [5, {"sentence": "1", "word": 0, "onset_ms": 1}, {"sentence": 1, "word": 0}]
)
def test_label_entries_must_be_integer_objects(tmp_path: Path, entry: object) -> None:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"version": 1, "labels": [entry]}))
    with pytest.raises(ValueError, match="labels"):
        spike.load_labels(path)


def test_from_runs_live_ctc_for_missing_spans(
    tmp_path: Path,
    ffmpeg_bin: str,
    monkeypatch: pytest.MonkeyPatch,
    mp3: bytes,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_capture(tmp_path, ffmpeg_bin, ctc=True)
    (tmp_path / "02.ctc.json").unlink()
    _fake_services(monkeypatch, mp3)
    assert spike.main(["--from", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "ctc first" in out
    assert "CTC: not available" not in out


def test_from_marks_unmappable_and_unmatched_cells(
    tmp_path: Path,
    ffmpeg_bin: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_capture(tmp_path, ffmpeg_bin)
    _no_extras(monkeypatch)
    far = [["aa", 5000, 5260], ["bb", 5700, 5960], ["cc", 6400, 6660]]
    (tmp_path / "01.edge.json").write_text(json.dumps(far))
    (tmp_path / "02.edge.json").write_text(json.dumps([["zz", 0, 10]]))
    label = {"sentence": 1, "word": 1, "onset_ms": 5000}
    (tmp_path / "labels.json").write_text(json.dumps({"version": 1, "labels": [label]}))
    assert spike.main(["--from", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "| 1 | +4780 | +4735 | - |" in out
    assert "| 2 | n/a | n/a | - |" in out


def test_from_skips_estimators_without_labelled_words(
    tmp_path: Path,
    ffmpeg_bin: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_capture(tmp_path, ffmpeg_bin)
    _no_extras(monkeypatch)
    (tmp_path / "labels.json").write_text(json.dumps({"version": 1, "labels": []}))
    assert spike.main(["--from", str(tmp_path)]) == 0
    assert "labelled words" not in capsys.readouterr().out


def test_from_reports_a_bad_directory(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert spike.main(["--from", str(tmp_path / "missing")]) == 1
    assert "error: manifest.json" in capsys.readouterr().err


def test_wave_deltas_accepts_a_word_exactly_at_the_window_edge() -> None:
    timings = _timings([0, 700, 2000])
    truth = spike.WaveTruth(0.0, 1500.0, [1000.0])
    assert spike.wave_deltas(timings, truth, window_ms=300).pause_starts_ms == [(1, -300.0)]
    assert spike.wave_deltas(timings, truth, window_ms=299).pause_starts_ms == []


def test_label_entry_bounds(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"

    def write(sentence: int, word: int, onset: int) -> None:
        entry = {"sentence": sentence, "word": word, "onset_ms": onset}
        path.write_text(json.dumps({"version": 1, "labels": [entry]}))

    write(sentence=1, word=0, onset=0)
    assert spike.load_labels(path, known=1) == {1: [(0, 0)]}
    write(sentence=2, word=0, onset=0)
    assert spike.load_labels(path, known=2) == {2: [(0, 0)]}
    write(sentence=0, word=0, onset=0)
    with pytest.raises(ValueError, match="unknown sentence 0"):
        spike.load_labels(path, known=2)
    write(sentence=1, word=-1, onset=0)
    with pytest.raises(ValueError, match="negative"):
        spike.load_labels(path, known=2)
    write(sentence=1, word=0, onset=-1)
    with pytest.raises(ValueError, match="negative"):
        spike.load_labels(path, known=2)


def test_pooled_selects_the_kind_and_skips_missing_estimators() -> None:
    first = spike.WaveDeltas(10.0, 20.0, [(1, 30.0), (2, 31.0)])
    second = spike.WaveDeltas(11.0, 21.0, [(1, 32.0)])
    scored = [spike._Scored(1, {}, {"edge": first}), spike._Scored(2, {}, {"edge": second})]
    scored.append(spike._Scored(3, {}, {}))
    assert spike._pooled(scored, "edge", "first") == [10.0, 11.0]
    assert spike._pooled(scored, "edge", "last") == [20.0, 21.0]
    assert spike._pooled(scored, "edge", "pause") == [30.0, 31.0, 32.0]


def test_capture_is_immutable() -> None:
    capture = spike.Capture(1, "aa", b"", ["aa"], [], None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        capture.index = 2


def test_json_files_are_indented_and_rewritable(tmp_path: Path) -> None:
    capture = spike.Capture(1, "aa", b"x", ["aa"], [TimedSpan("aa", 1, 2)], None)
    for _ in range(2):
        spike.save_capture(tmp_path, capture)
        spike.write_manifest(tmp_path, voice="v", edge_tts_version="1", captured_at="t")
    assert (tmp_path / "01.words.json").read_text() == '[\n "aa"\n]'
    assert (tmp_path / "01.edge.json").read_text() == json.dumps([["aa", 1, 2]], indent=1)


def test_parser_description_is_the_docstring_title() -> None:
    assert spike._parser().description == (spike.__doc__ or "").splitlines()[0]


def test_every_estimator_is_scored_on_the_same_post_pause_word(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The pause ends at 1000 ms. CTC puts bb at 1010; edge runs ~110 ms early, so bb
    # is at 890 and cc at 1080, which is nearer to 1000 on its own. Both must score bb.
    zeros = np.zeros
    samples = np.concatenate(
        [zeros(SR * 200 // 1000, dtype=np.float32), fx.tone(300), zeros(SR // 2), fx.tone(400)]
    ).astype(np.float32)
    monkeypatch.setattr(spike, "decode", lambda _audio: samples)
    edge = [TimedSpan("aa", 100, 400), TimedSpan("bb", 890, 1070), TimedSpan("cc", 1080, 1300)]
    ctc = [TimedSpan("aa", 200, 500), TimedSpan("bb", 1010, 1190), TimedSpan("cc", 1200, 1400)]
    capture = spike.Capture(1, "aa bb cc", b"", ["aa", "bb", "cc"], edge, ctc)
    scored = spike._score(capture, ctc)
    assert scored.deltas["edge"].pause_starts_ms == [(1, pytest.approx(-110, abs=10))]
    assert scored.deltas["ctc"].pause_starts_ms == [(1, pytest.approx(10, abs=10))]


def test_match_pause_words_picks_one_word_for_all_estimators() -> None:
    late = _timings([0, 1010, 1200])
    early = _timings([0, 890, 1080])
    got = spike.match_pause_words({"edge": early, "ctc": late}, [1000.0], window_ms=300)
    assert got == [1]
    assert spike.match_pause_words({"edge": early, "ctc": late}, [5000.0], window_ms=300) == [None]


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("field", ["sentence", "word", "onset_ms"])
def test_label_entries_reject_json_booleans(tmp_path: Path, field: str, flag: bool) -> None:
    entry = {"sentence": 1, "word": 1, "onset_ms": 1, field: flag}
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"version": 1, "labels": [entry]}))
    with pytest.raises(ValueError, match="integers"):
        spike.load_labels(path)


@pytest.mark.parametrize("labels", [5, "x", {"sentence": 1}, None])
def test_labels_must_be_a_list(tmp_path: Path, labels: object) -> None:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"version": 1, "labels": labels}))
    with pytest.raises(ValueError, match="labels"):
        spike.load_labels(path)


@pytest.mark.parametrize("version", ["missing", 2, "1", True])
def test_manifest_version_must_match(tmp_path: Path, ffmpeg_bin: str, version: object) -> None:
    _write_capture(tmp_path, ffmpeg_bin)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    if version == "missing":
        del manifest["version"]
    else:
        manifest["version"] = version
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r"manifest\.json.*version"):
        spike.load_captures(tmp_path)
