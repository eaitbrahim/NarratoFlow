from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from narratoflow.errors import LibraryError
from narratoflow.library.blobs import BlobStore
from narratoflow.library.db import open_library
from narratoflow.library.orm import Sketch
from narratoflow.library.repository import SketchLibrary, default_library
from narratoflow.library.schemas import Provenance, SketchOrigin, SketchRecord, SketchStatus
from narratoflow.settings import Settings


def prov() -> Provenance:
    return Provenance(source="test", license="CC0", created_at=datetime(2026, 1, 1, tzinfo=UTC))


@pytest.fixture
def library(tmp_path: Path) -> Iterator[SketchLibrary]:
    engine = open_library(tmp_path / "library.db")
    try:
        yield SketchLibrary(engine, BlobStore(tmp_path / "blobs"))
    finally:
        engine.dispose()


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
    a = _add(library, v, "k", b"A", SketchStatus.UNCHECKED)
    b = _add(library, v, "k", b"B")
    assert [r.id for r in library.find_reusable(v, "k")] == [a.id, b.id]
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
    src = _add(library, v1, "k", b"A")
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
    _add(library, v, "zeta", b"A")
    _add(library, v, "alpha", b"B")
    _add(library, v, "alpha", b"C")
    dead = _add(library, v, "dead", b"D")
    library.quarantine(dead.id, "bad")
    assert library.idea_keys(v) == ["alpha", "zeta"]


def test_idea_catalog_returns_latest_description(library: SketchLibrary) -> None:
    v = library.ensure_style_version("history", "fp1")
    _add(library, v, "k", b"A", description="a")
    _add(library, v, "k", b"B", description="b")
    _add(library, v, "k", b"C", description="")
    _add(library, v, "other", b"D")
    assert library.idea_catalog(v) == [("k", "b"), ("other", "")]


def test_copy_twice_dedupes(library: SketchLibrary) -> None:
    v1 = library.ensure_style_version("history", "fp1")
    v2 = library.ensure_style_version("tech", "fpX")
    src = _add(library, v1, "k", b"A")
    assert library.copy_to(src.id, v2).id == library.copy_to(src.id, v2).id


def test_default_library_uses_data_dir(tmp_path: Path) -> None:
    lib = default_library(Settings(data_dir=tmp_path / "data"))
    v = lib.ensure_style_version("history", "fp1")
    _add(lib, v, "k", b"A")
    assert (tmp_path / "data" / "library.db").is_file()
    assert len(list((tmp_path / "data" / "blobs").rglob("*.png"))) == 1


def test_readd_quarantined_stays_failed(library: SketchLibrary) -> None:
    v = library.ensure_style_version("history", "fp1")
    a = _add(library, v, "k", b"A")
    library.quarantine(a.id, "bad")
    again = _add(library, v, "k", b"A")
    assert again.id == a.id and again.status is SketchStatus.FAILED
    assert library.find_reusable(v, "k") == []


def test_copy_onto_existing_failed_returns_failed(library: SketchLibrary) -> None:
    v1 = library.ensure_style_version("history", "fp1")
    v2 = library.ensure_style_version("tech", "fpX")
    src = _add(library, v1, "k", b"A")
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
    a = _add(library, v, "k", b"A")
    _blind_once(monkeypatch, 1)
    assert _add(library, v, "k", b"A").id == a.id


def test_add_race_unresolvable_raises(
    library: SketchLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    v = library.ensure_style_version("history", "fp1")
    _add(library, v, "k", b"A")
    _blind_once(monkeypatch, 99)
    with pytest.raises(LibraryError):
        _add(library, v, "k", b"A")


def test_copy_race_returns_existing(
    library: SketchLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    v1 = library.ensure_style_version("history", "fp1")
    v2 = library.ensure_style_version("tech", "fpX")
    src = _add(library, v1, "k", b"A")
    dst = library.copy_to(src.id, v2)
    _blind_once(monkeypatch, 1)
    assert library.copy_to(src.id, v2).id == dst.id


def test_copy_race_unresolvable_raises(
    library: SketchLibrary, monkeypatch: pytest.MonkeyPatch
) -> None:
    v1 = library.ensure_style_version("history", "fp1")
    v2 = library.ensure_style_version("tech", "fpX")
    src = _add(library, v1, "k", b"A")
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
