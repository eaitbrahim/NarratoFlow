from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from narratoflow.errors import InvalidIdeaKeyError, LibraryError
from narratoflow.library.blobs import BlobStore
from narratoflow.library.keys import parse_idea_key
from narratoflow.library.orm import Sketch
from narratoflow.library.repository import SketchLibrary, default_library
from narratoflow.library.schemas import Provenance, SketchOrigin, SketchRecord, SketchStatus
from narratoflow.settings import Settings
from narratoflow.visuals.strokes import StrokeSet


def prov() -> Provenance:
    return Provenance(source="test", license="CC0", created_at=datetime(2026, 1, 1, tzinfo=UTC))


def _add(
    library: SketchLibrary,
    v: int,
    key: str,
    data: bytes,
    status: SketchStatus = SketchStatus.PASSED,
    description: str = "",
) -> SketchRecord:
    return library.add_sketch(
        v, key, data, ".png", SketchOrigin.GENERATED, status, True, prov(), description=description
    )


def test_style_version_is_idempotent_and_per_channel(library: SketchLibrary) -> None:
    a = library.ensure_style_version("history", "fp1")
    assert library.ensure_style_version("history", "fp1") == a
    assert library.ensure_style_version("history", "fp2") != a
    assert library.ensure_style_version("tech", "fp1") != a


def test_reuse_excludes_failed(library: SketchLibrary) -> None:
    v = library.ensure_style_version("history", "fp1")
    ok = _add(library, v, "eye|generic", b"A")
    bad = _add(library, v, "eye|generic", b"B")
    library.quarantine(bad.id, "face detected")
    assert [r.id for r in library.find_reusable(v, "eye|generic")] == [ok.id]


def test_reuse_includes_unchecked_ordered_by_id(library: SketchLibrary) -> None:
    v = library.ensure_style_version("history", "fp1")
    a = _add(library, v, "k|generic", b"A", SketchStatus.UNCHECKED)
    b = _add(library, v, "k|generic", b"B")
    assert [r.id for r in library.find_reusable(v, "k|generic")] == [a.id, b.id]
    assert a.provenance == prov()
    assert a.created_at.tzinfo is not None


def test_other_style_version_not_reused(library: SketchLibrary) -> None:
    v1 = library.ensure_style_version("history", "fp1")
    v2 = library.ensure_style_version("history", "fp2")
    _add(library, v1, "eye|generic", b"A")
    assert library.find_reusable(v2, "eye|generic") == []


def test_duplicate_add_dedupes(library: SketchLibrary, tmp_path: Path) -> None:
    v = library.ensure_style_version("history", "fp1")
    a = _add(library, v, "eye|generic", b"A")
    b = _add(library, v, "eye|generic", b"A")
    assert a.id == b.id
    assert len(list((tmp_path / "blobs").rglob("*.png"))) == 1


def test_blob_store_path_and_missing(tmp_path: Path) -> None:
    store = BlobStore(tmp_path / "blobs")
    h = store.put(b"A", ".png")
    assert store.path(h, ".png").read_bytes() == b"A"
    assert store.put(b"A", ".png") == h
    with pytest.raises(LibraryError):
        store.path("0" * 64, ".png")


def test_copy_keeps_provenance_and_blob(library: SketchLibrary) -> None:
    v1 = library.ensure_style_version("history", "fp1")
    v2 = library.ensure_style_version("tech", "fpX")
    src = _add(library, v1, "eye|generic", b"A")
    dst = library.copy_to(src.id, v2)
    assert dst.origin is SketchOrigin.COPIED and dst.blob_hash == src.blob_hash
    assert dst.provenance.copied_from_sketch_id == src.id
    assert dst.status is src.status
    assert dst.style_version_id == v2


def test_copy_failed_rejected(library: SketchLibrary) -> None:
    v1 = library.ensure_style_version("history", "fp1")
    v2 = library.ensure_style_version("tech", "fpX")
    src = _add(library, v1, "k|generic", b"A")
    library.quarantine(src.id, "bad")
    with pytest.raises(LibraryError):
        library.copy_to(src.id, v2)


def test_copy_unknown_id(library: SketchLibrary) -> None:
    v = library.ensure_style_version("history", "fp1")
    with pytest.raises(LibraryError):
        library.copy_to(999, v)


def test_quarantine_unknown_id(library: SketchLibrary) -> None:
    with pytest.raises(LibraryError):
        library.quarantine(999, "nope")


def test_idea_keys_lists_only_reusable(library: SketchLibrary) -> None:
    v = library.ensure_style_version("history", "fp1")
    _add(library, v, "zeta|generic", b"A")
    _add(library, v, "alpha|generic", b"B")
    _add(library, v, "alpha|generic", b"C")
    dead = _add(library, v, "dead|generic", b"D")
    library.quarantine(dead.id, "bad")
    assert library.idea_keys(v) == ["alpha|generic", "zeta|generic"]


def test_idea_catalog_returns_latest_description(library: SketchLibrary) -> None:
    v = library.ensure_style_version("history", "fp1")
    _add(library, v, "k|generic", b"A", description="a")
    _add(library, v, "k|generic", b"B", description="b")
    _add(library, v, "k|generic", b"C", description="")
    _add(library, v, "other|generic", b"D")
    assert library.idea_catalog(v) == [("k|generic", "b"), ("other|generic", "")]


def test_copy_twice_dedupes(library: SketchLibrary) -> None:
    v1 = library.ensure_style_version("history", "fp1")
    v2 = library.ensure_style_version("tech", "fpX")
    src = _add(library, v1, "k|generic", b"A")
    assert library.copy_to(src.id, v2).id == library.copy_to(src.id, v2).id


def test_default_library_uses_data_dir(tmp_path: Path) -> None:
    lib = default_library(Settings(data_dir=tmp_path / "data"))
    v = lib.ensure_style_version("history", "fp1")
    _add(lib, v, "k|generic", b"A")
    assert (tmp_path / "data" / "library.db").is_file()
    assert len(list((tmp_path / "data" / "blobs").rglob("*.png"))) == 1


def test_readd_quarantined_stays_failed(library: SketchLibrary) -> None:
    v = library.ensure_style_version("history", "fp1")
    a = _add(library, v, "k|generic", b"A")
    library.quarantine(a.id, "bad")
    again = _add(library, v, "k|generic", b"A")
    assert again.id == a.id and again.status is SketchStatus.FAILED
    assert library.find_reusable(v, "k|generic") == []


def test_copy_onto_existing_failed_returns_failed(library: SketchLibrary) -> None:
    v1 = library.ensure_style_version("history", "fp1")
    v2 = library.ensure_style_version("tech", "fpX")
    src = _add(library, v1, "k|generic", b"A")
    dst = library.copy_to(src.id, v2)
    library.quarantine(dst.id, "bad")
    again = library.copy_to(src.id, v2)
    assert again.id == dst.id and again.status is SketchStatus.FAILED


def _blind_once(monkeypatch: pytest.MonkeyPatch, times: int) -> None:
    real = SketchLibrary._find_existing
    calls = {"n": 0}

    def fake(s: Session, v: int, k: str, h: str) -> Sketch | None:
        calls["n"] += 1
        return None if calls["n"] <= times else real(s, v, k, h)

    monkeypatch.setattr(SketchLibrary, "_find_existing", staticmethod(fake))


def test_add_race_returns_existing(library: SketchLibrary, monkeypatch: pytest.MonkeyPatch) -> None:
    v = library.ensure_style_version("history", "fp1")
    a = _add(library, v, "k|generic", b"A")
    _blind_once(monkeypatch, 1)
    assert _add(library, v, "k|generic", b"A").id == a.id


def test_add_race_unresolvable_raises(
    library: SketchLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    v = library.ensure_style_version("history", "fp1")
    _add(library, v, "k|generic", b"A")
    _blind_once(monkeypatch, 99)
    with pytest.raises(LibraryError):
        _add(library, v, "k|generic", b"A")


def test_copy_race_returns_existing(
    library: SketchLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    v1 = library.ensure_style_version("history", "fp1")
    v2 = library.ensure_style_version("tech", "fpX")
    src = _add(library, v1, "k|generic", b"A")
    dst = library.copy_to(src.id, v2)
    _blind_once(monkeypatch, 1)
    assert library.copy_to(src.id, v2).id == dst.id


def test_copy_race_unresolvable_raises(
    library: SketchLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    v1 = library.ensure_style_version("history", "fp1")
    v2 = library.ensure_style_version("tech", "fpX")
    src = _add(library, v1, "k|generic", b"A")
    library.copy_to(src.id, v2)
    _blind_once(monkeypatch, 99)
    with pytest.raises(LibraryError):
        library.copy_to(src.id, v2)


def _boom(n: int) -> object:
    real = SketchLibrary._ensure_style_version
    calls = {"n": 0}

    def fake(self: SketchLibrary, slug: str, fp: str) -> int:
        calls["n"] += 1
        if calls["n"] <= n:
            raise IntegrityError("stmt", {}, Exception("dup"))
        return real(self, slug, fp)

    return fake


def test_ensure_style_version_retries_then_succeeds(
    library: SketchLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(SketchLibrary, "_ensure_style_version", _boom(1))
    assert library.ensure_style_version("history", "fp1") > 0


def test_ensure_style_version_bounded(
    library: SketchLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(SketchLibrary, "_ensure_style_version", _boom(99))
    with pytest.raises(LibraryError):
        library.ensure_style_version("history", "fp1")


def test_add_sketch_normalizes_idea_key(library: SketchLibrary) -> None:
    v = library.ensure_style_version("history", "fp1")
    rec = _add(library, v, " Old Scroll | Roman ", b"A")
    assert rec.idea_key == "old-scroll|roman"
    assert [r.id for r in library.find_reusable(v, "old scroll|roman")] == [rec.id]
    assert [r.id for r in library.find_reusable(v, parse_idea_key("old-scroll|roman"))] == [rec.id]
    assert library.idea_keys(v) == ["old-scroll|roman"]


@pytest.mark.parametrize("bad", ["alpha", "a|b|c", "\u0642\u0644\u0645|generic", "|generic"])
def test_invalid_idea_key_rejected(library: SketchLibrary, bad: str, tmp_path: Path) -> None:
    v = library.ensure_style_version("history", "fp1")
    with pytest.raises(InvalidIdeaKeyError):
        _add(library, v, bad, b"A")
    with pytest.raises(InvalidIdeaKeyError):
        library.find_reusable(v, bad)
    assert not (tmp_path / "blobs").exists() or not list((tmp_path / "blobs").rglob("*.png"))


def test_provenance_requires_aware_datetime() -> None:
    with pytest.raises(ValidationError):
        Provenance(source="test", license="CC0", created_at=datetime(2026, 1, 1))


def test_record_fields_match_orm_columns() -> None:
    assert set(SketchRecord.model_fields) == set(Sketch.__table__.columns.keys())


def test_add_integrity_error_is_chained(
    library: SketchLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    v = library.ensure_style_version("history", "fp1")
    _add(library, v, "k|generic", b"A")
    _blind_once(monkeypatch, 99)
    with pytest.raises(LibraryError) as exc:
        _add(library, v, "k|generic", b"A")
    assert isinstance(exc.value.__cause__, IntegrityError)


def test_copy_integrity_error_is_chained(
    library: SketchLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    v1 = library.ensure_style_version("history", "fp1")
    v2 = library.ensure_style_version("tech", "fpX")
    src = _add(library, v1, "k|generic", b"A")
    library.copy_to(src.id, v2)
    _blind_once(monkeypatch, 99)
    with pytest.raises(LibraryError) as exc:
        library.copy_to(src.id, v2)
    assert isinstance(exc.value.__cause__, IntegrityError)


def _strokes() -> StrokeSet:
    from narratoflow.visuals.strokes import Stroke

    return StrokeSet(width=8, height=4, strokes=[Stroke(points=[(0.0, 0.0), (1.0, 0.5)])])


def test_strokes_roundtrip_as_content_addressed_blob(
    library: SketchLibrary, tmp_path: Path
) -> None:
    v = library.ensure_style_version("history", "fp1")
    a = _add(library, v, "k|generic", b"A")
    assert library.get_strokes(a.id) is None
    library.put_strokes(a.id, _strokes())
    assert library.get_strokes(a.id) == _strokes()
    [blob] = list((tmp_path / "blobs").rglob("*.strokes.json"))
    assert blob.read_bytes() == _strokes().to_json()


def test_put_strokes_unknown_sketch_raises(library: SketchLibrary) -> None:
    with pytest.raises(LibraryError, match="unknown sketch id"):
        library.put_strokes(999, _strokes())
    with pytest.raises(LibraryError, match="unknown sketch id"):
        library.get_strokes(999)


def test_sketch_bytes_reads_the_stored_blob(library: SketchLibrary) -> None:
    v = library.ensure_style_version("history", "fp1")
    a = _add(library, v, "k|generic", b"PNGDATA")
    assert library.sketch_bytes(a) == b"PNGDATA"


def test_default_library_caches_engine_per_data_dir(tmp_path: Path) -> None:
    from narratoflow.library import repository

    one = default_library(Settings(data_dir=tmp_path / "a"))
    two = default_library(Settings(data_dir=tmp_path / "a" / ".." / "a"))
    other = default_library(Settings(data_dir=tmp_path / "b"))
    assert one._engine is two._engine
    assert other._engine is not one._engine
    repository.clear_library_cache()
    assert default_library(Settings(data_dir=tmp_path / "a"))._engine is not one._engine
    repository.clear_library_cache()
