from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from narratoflow.audio.timeline import PartTimeline, TimingSource
from narratoflow.errors import AudioError


def base() -> dict[str, Any]:
    return {
        "series_title": "Test",
        "channel_slug": "demo",
        "part_index": 1,
        "part_count": 2,
        "audio_file": "part-01.flac",
        "duration_ms": 9000,
        "integrated_lufs": -14.1,
        "true_peak_dbtp": -1.2,
        "scenes": [
            {
                "scene_id": "s1",
                "kind": "narration",
                "start_ms": 0,
                "end_ms": 4000,
                "speech_start_ms": 50,
                "speech_end_ms": 3800,
                "words": [
                    {"text": "Hello,", "start_ms": 50, "end_ms": 900},
                    {"text": "world.", "start_ms": 1000, "end_ms": 3800},
                ],
                "timing_source": "provider",
                "draw_start_ms": 0,
                "draw_end_ms": 3500,
            },
            {
                "scene_id": "s2",
                "kind": "verse",
                "start_ms": 4000,
                "end_ms": 9000,
                "speech_start_ms": 4400,
                "speech_end_ms": 8600,
                "words": [],
                "timing_source": "segments",
                "ayat": [
                    {
                        "surah": 31,
                        "ayah": 33,
                        "start_ms": 4400,
                        "end_ms": 8600,
                        "words": [{"text": "\u064a\u064e\u0627", "start_ms": 4400, "end_ms": 5000}],
                    }
                ],
            },
        ],
        "sfx": [
            {
                "scene_id": "s1",
                "stroke_index": 0,
                "at_ms": 100,
                "duration_ms": 300,
                "asset_id": "pen-1",
                "gain_db": -6.0,
            }
        ],
        "voice": {"provider": "elevenlabs", "model": "m", "voice_id": "v"},
        "recitations": [
            {
                "reciter_id": "r1",
                "name": "Reciter",
                "riwaya": "hafs",
                "license": "CC",
                "license_url": "https://example.invalid/l",
                "source": "https://example.invalid/s",
                "segments_source": None,
                "ayat": ["31:33"],
            }
        ],
        "assets": [
            {"kind": "sfx", "asset_id": "pen-1", "source": "own", "license": "CC0"},
        ],
        "spent_usd": "0.42",
        "cost_estimated": False,
        "warnings": [],
    }


def make(data: dict[str, Any]) -> PartTimeline:
    return PartTimeline.model_validate(data)


def test_timeline_roundtrip(tmp_path: Path) -> None:
    tl = make(base())
    assert tl.schema_version == 1
    assert tl.sample_rate == 48000
    assert tl.scenes[1].timing_source is TimingSource.SEGMENTS
    assert tl.spent_usd == Decimal("0.42")
    path = tmp_path / "part-01.timeline.json"
    tl.save(path)
    assert PartTimeline.load(path) == tl
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".")]


def test_timeline_rejects_gap_between_scenes() -> None:
    data = base()
    data["scenes"][1]["start_ms"] = 4100
    with pytest.raises(ValueError, match="contiguous"):
        make(data)


def test_timeline_rejects_first_scene_not_at_zero() -> None:
    data = base()
    data["scenes"][0]["start_ms"] = 10
    data["scenes"][0]["draw_start_ms"] = 10
    data["scenes"][0]["speech_start_ms"] = 50
    with pytest.raises(ValueError, match="start at 0"):
        make(data)


def test_timeline_rejects_last_scene_not_at_duration() -> None:
    data = base()
    data["duration_ms"] = 9500
    with pytest.raises(ValueError, match="duration_ms"):
        make(data)


def test_timeline_rejects_sfx_inside_verse_slot() -> None:
    data = base()
    data["sfx"][0]["at_ms"] = 3800  # 3800 + 300 runs into the verse slot at 4000
    with pytest.raises(ValueError, match="verse"):
        make(data)


def test_timeline_rejects_unsorted_sfx() -> None:
    data = base()
    data["sfx"].insert(0, dict(data["sfx"][0], at_ms=500, stroke_index=1))
    with pytest.raises(ValueError, match="sorted"):
        make(data)


def test_timeline_rejects_word_outside_speech() -> None:
    data = base()
    data["scenes"][0]["words"][1]["end_ms"] = 3900
    with pytest.raises(ValueError, match="speech"):
        make(data)


def test_timeline_rejects_decreasing_words() -> None:
    data = base()
    data["scenes"][0]["words"][1]["start_ms"] = 40
    data["scenes"][0]["words"][1]["end_ms"] = 60
    with pytest.raises(ValueError, match="speech"):
        make(data)
    data = base()
    data["scenes"][0]["words"][1]["start_ms"] = 20
    data["scenes"][0]["words"][1]["end_ms"] = 3800
    data["scenes"][0]["speech_start_ms"] = 10
    data["scenes"][0]["words"][0]["start_ms"] = 30
    with pytest.raises(ValueError, match="previous word"):
        make(data)


def test_timeline_rejects_bad_speech_span() -> None:
    data = base()
    data["scenes"][0]["speech_end_ms"] = 4100
    with pytest.raises(ValueError, match="speech"):
        make(data)


def test_timeline_rejects_draw_outside_slot() -> None:
    data = base()
    data["scenes"][0]["draw_end_ms"] = 4500
    with pytest.raises(ValueError, match="draw"):
        make(data)
    data = base()
    data["scenes"][0]["draw_end_ms"] = None
    with pytest.raises(ValueError, match="draw"):
        make(data)


def test_timeline_rejects_ayat_on_narration_and_outside_scene() -> None:
    data = base()
    data["scenes"][0]["ayat"] = data["scenes"][1]["ayat"]
    with pytest.raises(ValueError, match="ayat"):
        make(data)
    data = base()
    data["scenes"][1]["ayat"][0]["end_ms"] = 9100
    with pytest.raises(ValueError, match="ayah"):
        make(data)
    data = base()
    data["scenes"][1]["ayat"][0]["start_ms"] = 8700
    with pytest.raises(ValueError, match="before start_ms"):
        make(data)
    data = base()
    data["scenes"][1]["ayat"][0]["words"][0]["end_ms"] = 8700
    with pytest.raises(ValueError, match="ayah"):
        make(data)


def test_timeline_rejects_words_on_verse_scene() -> None:
    data = base()
    data["scenes"][1]["words"] = [{"text": "x", "start_ms": 4400, "end_ms": 4500}]
    with pytest.raises(ValueError, match="verse"):
        make(data)


@pytest.mark.parametrize("name", ["dir/part-01.flac", "dir\\part-01.flac", "", "..", "."])
def test_timeline_rejects_path_in_audio_file(name: str) -> None:
    data = base()
    data["audio_file"] = name
    with pytest.raises(ValueError, match="audio_file"):
        make(data)


def test_timeline_rejects_extra_and_negative_and_non_finite() -> None:
    data = base()
    data["extra"] = 1
    with pytest.raises(ValidationError):
        make(data)
    data = base()
    data["scenes"][0]["words"][0]["start_ms"] = -1
    with pytest.raises(ValidationError):
        make(data)
    for key, value in (
        ("integrated_lufs", float("nan")),
        ("true_peak_dbtp", float("inf")),
        ("spent_usd", "NaN"),
    ):
        data = base()
        data[key] = value
        with pytest.raises(ValidationError):
            make(data)
    data = base()
    data["sfx"][0]["gain_db"] = float("-inf")
    with pytest.raises(ValidationError):
        make(data)
    data = base()
    data["sfx"][0]["duration_ms"] = 0
    with pytest.raises(ValidationError):
        make(data)


def test_timeline_allows_silent_lufs_and_no_voice() -> None:
    data = base()
    data["integrated_lufs"] = None
    data["voice"] = None
    assert make(data).voice is None


def test_saved_json_has_no_nan(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    make(base()).save(path)
    raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject)
    assert raw["schema_version"] == 1


def _reject(token: str) -> None:
    raise AssertionError(f"non-finite JSON constant {token}")


def test_load_unknown_version_raises_audio_error(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    data = base()
    for version in (2, True, "1", None):
        data["schema_version"] = version
        path.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(AudioError, match="schema_version"):
            PartTimeline.load(path)


def test_load_unreadable_or_invalid_raises_audio_error(tmp_path: Path) -> None:
    with pytest.raises(AudioError):
        PartTimeline.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(AudioError):
        PartTimeline.load(bad)
    invalid = tmp_path / "invalid.json"
    data = base()
    data["schema_version"] = 1
    data["scenes"] = []
    invalid.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(AudioError):
        PartTimeline.load(invalid)
    nan = tmp_path / "nan.json"
    data = base()
    data["schema_version"] = 1
    data["integrated_lufs"] = float("nan")
    nan.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(AudioError):
        PartTimeline.load(nan)


def test_save_failure_raises_audio_error(tmp_path: Path) -> None:
    with pytest.raises(AudioError):
        make(base()).save(tmp_path / "no-such-dir" / "t.json")
