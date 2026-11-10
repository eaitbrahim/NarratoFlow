from pathlib import Path

import pytest
from pydantic import ValidationError

from narratoflow.models import (
    AspectRatio,
    Language,
    Narration,
    Project,
    Scene,
    WordTiming,
)
from narratoflow.pipeline.cache import cache_key


def test_roundtrip(tmp_path: Path) -> None:
    project = Project(
        title="Demo",
        language=Language.ARABIC,
        outputs=[AspectRatio.LANDSCAPE, AspectRatio.PORTRAIT],
        scenes=[
            Scene(
                id="s1",
                narration=Narration(
                    source_text="العلم نور",
                    words=[WordTiming(text="العلم", start_ms=0, end_ms=400)],
                ),
            )
        ],
    )
    path = tmp_path / "storyboard.json"
    project.save(path)
    loaded = Project.load(path)
    assert loaded == project
    assert loaded.scenes[0].narration.duration_ms == 400


def test_word_timing_order_enforced() -> None:
    with pytest.raises(ValidationError):
        WordTiming(text="x", start_ms=500, end_ms=100)


def test_resolution() -> None:
    assert AspectRatio.PORTRAIT.resolution == (1080, 1920)


def test_cache_key_is_stable_and_sensitive() -> None:
    a = cache_key("tts", "1", {"text": "نور", "voice": "ar-SA-HamedNeural"})
    b = cache_key("tts", "1", {"voice": "ar-SA-HamedNeural", "text": "نور"})
    c = cache_key("tts", "2", {"text": "نور", "voice": "ar-SA-HamedNeural"})
    assert a == b
    assert a != c
