import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

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
    for mark in ("«", "»", "؟", "،", "؛"):
        assert mark in joined
    for clitic in ("وَ", "فَ", "بِ", "لِ", "كَ"):
        assert any(w.startswith(clitic) for s in sentences for w in s.split())
