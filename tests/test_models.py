import json
import os
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from narratoflow.errors import (
    BudgetExceededError,
    NarratoFlowError,
    PlanningError,
    ProviderError,
    ProviderRefusalError,
    ScriptBlockedError,
    SegmentationError,
    VerseCorpusError,
)
from narratoflow.models import (
    SCHEMA_VERSION,
    AspectRatio,
    Language,
    Narration,
    Part,
    Project,
    RightsBasis,
    Scene,
    SceneKind,
    ScriptSource,
    Series,
    VerseRef,
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


def _series() -> Series:
    return Series(
        title="t",
        channel_slug="history",
        language=Language.ARABIC,
        source=ScriptSource(author="a", work="w", rights=RightsBasis.OWN),
        parts=[
            Part(
                index=1,
                count=1,
                hook_text="h",
                est_cost_usd=Decimal("1.25"),
                scenes=[
                    Scene(
                        id="p1s1",
                        narration=Narration(source_text="x"),
                        est_duration_s=4.5,
                    )
                ],
            )
        ],
        est_planning_cost_usd=Decimal("0.07"),
    )


def test_series_roundtrip(tmp_path: Path) -> None:
    s = _series()
    s.save(tmp_path / "series.json")
    assert Series.load(tmp_path / "series.json") == s
    assert s.parts[0].est_duration_s == 4.5
    assert s.schema_version == SCHEMA_VERSION == 2


def test_scene_v2_defaults() -> None:
    sc = Scene(id="a", narration=Narration(source_text="x"))
    assert sc.kind is SceneKind.NARRATION
    assert sc.verse is None and sc.idea_key is None and sc.match is None
    assert sc.flags == [] and sc.est_duration_s == 0
    assert sc.sketch_id is None
    dumped = sc.model_copy(update={"sketch_id": 7}).model_dump_json()
    assert Scene.model_validate_json(dumped).sketch_id == 7


def test_verse_ref_range_validated() -> None:
    with pytest.raises(ValidationError):
        VerseRef(surah=19, ayah_start=13, ayah_end=12, certain=True, uthmani_text="x")
    with pytest.raises(ValidationError):
        VerseRef(surah=115, ayah_start=1, ayah_end=1, certain=True, uthmani_text="x")
    ok = VerseRef(surah=19, ayah_start=12, ayah_end=12, certain=True, uthmani_text="x")
    assert ok.ayah_end == 12


def test_series_load_rejects_other_schema(tmp_path: Path) -> None:
    path = tmp_path / "series.json"
    path.write_text(
        _series().model_copy(update={"schema_version": 1}).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(PlanningError):
        Series.load(path)


def test_project_still_loads_v1(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text('{"schema_version": 1, "title": "T", "language": "ar"}', encoding="utf-8")
    assert Project.load(path).schema_version == 1


def test_budget_error_message_has_values() -> None:
    e = BudgetExceededError(Decimal("1.20"), Decimal("1.00"))
    assert "$1.20" in str(e) and "$1.00" in str(e)
    fine = BudgetExceededError(Decimal("0.0123456"), Decimal("0.01"))
    assert "$0.0123" in str(fine) and "$0.0123456" not in str(fine)
    assert e.estimate_usd == Decimal("1.20")


def test_error_hierarchy() -> None:
    for cls in (
        ScriptBlockedError,
        ProviderError,
        SegmentationError,
        PlanningError,
        VerseCorpusError,
    ):
        assert issubclass(cls, NarratoFlowError)
    assert issubclass(ProviderRefusalError, ProviderError)
    assert ProviderRefusalError("cyber").category == "cyber"
    assert ProviderRefusalError(None).category is None


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        '{"schema_version": 2, "title": "t"}',
        b"\xff\xfe\x00bad",
    ],
    ids=["bad-json", "invalid-series", "not-utf8"],
)
def test_series_load_wraps_bad_files(tmp_path: Path, content: str | bytes) -> None:
    path = tmp_path / "series.json"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(PlanningError, match="series"):
        Series.load(path)


def test_series_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PlanningError, match="cannot read series"):
        Series.load(tmp_path / "missing.json")


def test_series_save_unwritable(tmp_path: Path) -> None:
    with pytest.raises(PlanningError, match="cannot write series"):
        _series().save(tmp_path / "no-such-dir" / "series.json")


def test_series_save_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "series.json"
    original = _series()
    original.save(path)
    before = path.read_text(encoding="utf-8")

    def boom(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(PlanningError, match="cannot write series"):
        original.model_copy(update={"title": "Other"}).save(path)
    assert path.read_text(encoding="utf-8") == before
    assert [p.name for p in tmp_path.iterdir()] == ["series.json"]


def test_old_series_without_spend_loads(tmp_path: Path) -> None:
    data = json.loads(_series().model_dump_json())
    for part in data["parts"]:
        del part["spend"]
    path = tmp_path / "series.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert [p.spend for p in Series.load(path).parts] == [{}] * len(data["parts"])
