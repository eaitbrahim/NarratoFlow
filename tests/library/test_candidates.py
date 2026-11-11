from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pytest

from narratoflow.errors import PlanningError
from narratoflow.library.candidates import candidate_keys
from narratoflow.library.repository import SketchLibrary
from narratoflow.library.schemas import Provenance, SketchOrigin, SketchStatus

# Each word maps to a fixed axis, so a text's vector is its bag of known words.
_AXES = {"lamp": 0, "light": 0, "cup": 1, "drink": 1, "sword": 2, "battle": 2, "zero": 3}


class FakeEmbedder:
    """Deterministic bag-of-words embedder that records every call."""

    name = "fake"

    def __init__(self, model: str = "fake-embed") -> None:
        self.model = model
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        out = []
        for text in texts:
            vec = [0.0] * 4
            for word in text.replace("|", " ").replace(":", " ").split():
                if word in _AXES and word != "zero":
                    vec[_AXES[word]] += 1.0
            out.append(vec)
        return out


def _prov() -> Provenance:
    return Provenance(source="test", license="CC0", created_at=datetime(2026, 1, 1, tzinfo=UTC))


def _seed(library: SketchLibrary, items: Sequence[tuple[str, str]]) -> int:
    v = library.ensure_style_version("history", "fp1")
    for n, (key, description) in enumerate(items):
        library.add_sketch(
            v,
            key,
            f"blob{n}".encode(),
            ".png",
            SketchOrigin.GENERATED,
            SketchStatus.PASSED,
            True,
            _prov(),
            description=description,
        )
    return v


_THREE = [
    ("object|cup", "a cup to drink from"),
    ("object|lamp", "a lamp giving light"),
    ("object|sword", "a sword for battle"),
]


def test_empty_scenes_returns_empty(library: SketchLibrary) -> None:
    v = _seed(library, _THREE)
    embedder = FakeEmbedder()
    assert candidate_keys(library, v, [], embedder) == []
    assert embedder.calls == []


def test_empty_catalog_returns_separate_empty_lists(library: SketchLibrary) -> None:
    v = library.ensure_style_version("history", "fp1")
    result = candidate_keys(library, v, ["a", "b"], None)
    assert result == [[], []]
    result[0].append(("x|generic", ""))
    assert result[1] == []


def test_small_catalog_returns_full_catalog_without_embedding(library: SketchLibrary) -> None:
    v = _seed(library, _THREE)
    embedder = FakeEmbedder()
    result = candidate_keys(library, v, ["lamp", "battle", "drink"], embedder)
    assert result == [_THREE, _THREE, _THREE]
    assert embedder.calls == []
    result[0].pop()
    assert result[1] == _THREE


def test_small_catalog_needs_no_embedder(library: SketchLibrary) -> None:
    v = _seed(library, _THREE)
    assert candidate_keys(library, v, ["x"], None) == [_THREE]


def test_large_catalog_without_embedder_raises(library: SketchLibrary) -> None:
    v = _seed(library, _THREE)
    with pytest.raises(PlanningError, match="embed"):
        candidate_keys(library, v, ["lamp"], None, full_limit=2)


def test_large_catalog_ranks_by_cosine(library: SketchLibrary) -> None:
    v = _seed(library, _THREE)
    embedder = FakeEmbedder()
    result = candidate_keys(
        library, v, ["the lamp light", "a battle", "drink"], embedder, full_limit=2, top_k=1
    )
    assert result == [[_THREE[1]], [_THREE[2]], [_THREE[0]]]


def test_top_k_orders_by_score_and_breaks_ties_by_catalog_order(
    library: SketchLibrary,
) -> None:
    v = _seed(library, _THREE)
    embedder = FakeEmbedder()
    # "sword lamp" is equally close to lamp and sword; cup scores 0.
    (ranked,) = candidate_keys(library, v, ["sword lamp"], embedder, full_limit=2, top_k=3)
    assert ranked == [_THREE[1], _THREE[2], _THREE[0]]
    # A zero scene vector ties everything: catalog order, no NaN.
    (tied,) = candidate_keys(library, v, ["zero"], embedder, full_limit=2, top_k=2)
    assert tied == [_THREE[0], _THREE[1]]


def test_catalog_embeddings_are_cached(library: SketchLibrary) -> None:
    v = _seed(library, _THREE)
    first = FakeEmbedder()
    candidate_keys(library, v, ["lamp"], first, full_limit=2)
    assert [len(c) for c in first.calls] == [3, 1]

    second = FakeEmbedder()
    result = candidate_keys(library, v, ["lamp"], second, full_limit=2, top_k=1)
    assert second.calls == [["lamp"]]
    assert result == [[_THREE[1]]]


def test_only_missing_catalog_rows_are_embedded(library: SketchLibrary) -> None:
    v = _seed(library, _THREE)
    candidate_keys(library, v, ["lamp"], FakeEmbedder(), full_limit=2)
    _seed(library, [("object|mug", "drink cup")])
    embedder = FakeEmbedder()
    candidate_keys(library, v, ["lamp"], embedder, full_limit=2)
    assert len(embedder.calls) == 2
    (catalog_call, scene_call) = embedder.calls
    assert len(catalog_call) == 1
    assert "object|mug" in catalog_call[0]
    assert scene_call == ["lamp"]


def test_cache_is_per_model(library: SketchLibrary) -> None:
    v = _seed(library, _THREE)
    candidate_keys(library, v, ["lamp"], FakeEmbedder("m1"), full_limit=2)
    other = FakeEmbedder("m2")
    candidate_keys(library, v, ["lamp"], other, full_limit=2)
    assert [len(c) for c in other.calls] == [3, 1]


def test_wrong_vector_count_raises(library: SketchLibrary) -> None:
    v = _seed(library, _THREE)

    class Short(FakeEmbedder):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            return super().embed(texts)[:-1]

    with pytest.raises(PlanningError, match="vectors"):
        candidate_keys(library, v, ["lamp"], Short(), full_limit=2)


def test_store_and_read_cached_embeddings(library: SketchLibrary) -> None:
    v = _seed(library, _THREE)
    rows = [("object|cup", "a cup", [0.5, -1.25]), ("object|lamp", "", [1.0, 0.0])]
    library.store_embeddings(v, "m", rows)
    library.store_embeddings(v, "m", rows)  # idempotent: duplicates are ignored
    library.store_embeddings(v, "m", [])
    cached = library.cached_embeddings(v, "m")
    assert cached == {("object|cup", "a cup"): [0.5, -1.25], ("object|lamp", ""): [1.0, 0.0]}
    assert library.cached_embeddings(v, "other") == {}


def test_vectors_stored_as_float32_little_endian(library: SketchLibrary) -> None:
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    from narratoflow.library.orm import IdeaEmbedding

    v = _seed(library, _THREE)
    library.store_embeddings(v, "m", [("object|cup", "d", [1.5, 2.0])])
    with Session(library._engine) as s:
        blob = s.scalars(select(IdeaEmbedding.vector)).one()
    assert blob == np.asarray([1.5, 2.0], dtype="<f4").tobytes()


@pytest.mark.parametrize("bad", ["ragged", "empty", "dimension"])
def test_inconsistent_vectors_raise(library: SketchLibrary, bad: str) -> None:
    v = _seed(library, _THREE)

    class Broken(FakeEmbedder):
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            vectors = super().embed(texts)
            is_scene = texts == ["lamp"]
            if bad == "ragged" and not is_scene:
                vectors[0] = [1.0]
            if bad == "dimension" and is_scene:
                vectors = [[1.0, 0.0]]
            if bad == "empty" and is_scene:
                return [[]]
            return vectors

    with pytest.raises(PlanningError, match="dimension"):
        candidate_keys(library, v, ["lamp"], Broken(), full_limit=2)
