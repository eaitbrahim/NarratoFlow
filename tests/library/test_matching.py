import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from narratoflow.errors import InvalidIdeaKeyError
from narratoflow.library.keys import parse_idea_key
from narratoflow.library.matching import MatchKind, choose_variant, match_idea
from narratoflow.library.repository import SketchLibrary
from narratoflow.library.schemas import Provenance, SketchOrigin, SketchRecord, SketchStatus
from narratoflow.visuals.bundled import BundledIcons

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _prov() -> Provenance:
    return Provenance(source="test", license="CC0", created_at=NOW)


def make_record(i: int) -> SketchRecord:
    return SketchRecord(
        id=i,
        style_version_id=1,
        idea_key="eye|generic",
        idea_description="",
        blob_hash=f"{i:064x}",
        suffix=".png",
        origin=SketchOrigin.GENERATED,
        status=SketchStatus.PASSED,
        status_reason=None,
        mirror_safe=True,
        provenance=_prov(),
        created_at=NOW,
    )


def _write_index(tmp_path: Path, keys: list[str]) -> Path:
    (tmp_path / "icon.svg").write_text("<svg/>")
    body = "[icons]\n" if not keys else ""
    for key in keys:
        body += (
            f'[icons."{key}"]\nfile = "icon.svg"\nlicense = "MIT"\nsource = "https://example.com"\n'
        )
    index = tmp_path / "index.toml"
    index.write_text(body)
    return index


@pytest.fixture
def bundled_eye(tmp_path: Path) -> BundledIcons:
    return BundledIcons.from_index(_write_index(tmp_path, ["eye|generic"]))


@pytest.fixture
def bundled_empty(tmp_path: Path) -> BundledIcons:
    return BundledIcons.from_index(_write_index(tmp_path, []))


def _store(library: SketchLibrary, key: str, status: SketchStatus = SketchStatus.PASSED) -> None:
    v = library.ensure_style_version("history", "fp1")
    rec = library.add_sketch(v, key, b"X", ".png", SketchOrigin.GENERATED, status, True, _prov())
    if status is SketchStatus.FAILED:
        library.quarantine(rec.id, "bad")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("eye|generic", "eye|generic"),
        ("  Old Scroll|Roman ", "old-scroll|roman"),
        ("papyrus|ancient_egypt", "papyrus|ancient-egypt"),
        ("scroll | roman", "scroll|roman"),
        (" Old Scroll  |  Ancient Egypt ", "old-scroll|ancient-egypt"),
    ],
)
def test_parse_normalizes(raw: str, expected: str) -> None:
    assert str(parse_idea_key(raw)) == expected


@pytest.mark.parametrize("raw", ["قلم|generic", "eye", "a|b|c", "|generic", "eye|"])
def test_parse_rejects(raw: str) -> None:
    with pytest.raises(InvalidIdeaKeyError, match=re.escape(repr(raw))):
        parse_idea_key(raw)


def test_is_generic() -> None:
    assert parse_idea_key("eye|generic").is_generic
    assert not parse_idea_key("scroll|roman").is_generic


def test_library_beats_bundled(library: SketchLibrary, bundled_eye: BundledIcons) -> None:
    _store(library, "eye|generic")
    v = library.ensure_style_version("history", "fp1")
    result = match_idea(library, v, parse_idea_key("eye|generic"), bundled_eye)
    assert result.kind is MatchKind.REUSED
    assert result.candidates
    assert result.icon is None


def test_bundled_only_for_generic(library: SketchLibrary, tmp_path: Path) -> None:
    bundled = BundledIcons.from_index(_write_index(tmp_path, ["scroll|generic"]))
    v = library.ensure_style_version("history", "fp1")
    result = match_idea(library, v, parse_idea_key("scroll|roman"), bundled)
    assert result.kind is MatchKind.NEW
    assert result.icon is None


def test_bundled_for_generic_when_library_empty(
    library: SketchLibrary, bundled_eye: BundledIcons
) -> None:
    v = library.ensure_style_version("history", "fp1")
    result = match_idea(library, v, parse_idea_key("eye|generic"), bundled_eye)
    assert result.kind is MatchKind.BUNDLED
    assert result.icon is not None
    assert result.icon.license == "MIT"


def test_generic_without_icon_is_new(library: SketchLibrary, bundled_empty: BundledIcons) -> None:
    v = library.ensure_style_version("history", "fp1")
    result = match_idea(library, v, parse_idea_key("eye|generic"), bundled_empty)
    assert result.kind is MatchKind.NEW


def test_bundled_index_rejects_non_generic(tmp_path: Path) -> None:
    with pytest.raises(InvalidIdeaKeyError):
        BundledIcons.from_index(_write_index(tmp_path, ["scroll|roman"]))


def test_failed_sketch_does_not_match(library: SketchLibrary, bundled_empty: BundledIcons) -> None:
    _store(library, "eye|generic", SketchStatus.FAILED)
    v = library.ensure_style_version("history", "fp1")
    result = match_idea(library, v, parse_idea_key("eye|generic"), bundled_empty)
    assert result.kind is MatchKind.NEW


def test_choose_variant_deterministic_and_varies() -> None:
    cands = [make_record(i) for i in range(5)]
    assert choose_variant(cands, "vid1", "s3") == choose_variant(cands, "vid1", "s3")
    picks = {choose_variant(cands, f"vid{n}", "s3").id for n in range(50)}
    assert len(picks) > 1


def test_choose_variant_empty_raises() -> None:
    with pytest.raises(ValueError):
        choose_variant([], "v", "s")
