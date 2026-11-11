"""Pick the idea keys the planner may choose from for each scene.

Small catalogs are shown whole. Large ones are narrowed to the top-k entries per
scene by cosine similarity of embeddings; catalog vectors are cached in SQLite.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from narratoflow.errors import PlanningError
from narratoflow.library.repository import SketchLibrary

if TYPE_CHECKING:
    from narratoflow.embed import Embedder

Entry = tuple[str, str]


def catalog_text(idea_key: str, description: str) -> str:
    """The text embedded for one catalog entry."""
    return f"{idea_key}: {description}" if description else idea_key


def _matrix(vectors: Sequence[Sequence[float]], what: str) -> npt.NDArray[np.float64]:
    try:
        matrix = np.asarray(vectors, dtype=np.float64)
    except ValueError as e:
        raise PlanningError(f"{what} vectors have inconsistent dimensions") from e
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Zero vectors stay zero, so they score 0 against everything instead of NaN.
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _embed(embedder: Embedder, texts: Sequence[str], what: str) -> list[list[float]]:
    vectors = embedder.embed(texts)
    if len(vectors) != len(texts):
        raise PlanningError(
            f"embedder returned {len(vectors)} {what} vectors for {len(texts)} texts"
        )
    return vectors


def _catalog_vectors(
    library: SketchLibrary, style_version_id: int, catalog: list[Entry], embedder: Embedder
) -> list[list[float]]:
    model_id = f"{embedder.name}/{embedder.model}"
    cached = library.cached_embeddings(style_version_id, model_id)
    missing = [entry for entry in catalog if entry not in cached]
    if missing:
        fresh = _embed(embedder, [catalog_text(k, d) for k, d in missing], "catalog")
        rows = [(k, d, v) for (k, d), v in zip(missing, fresh, strict=True)]
        library.store_embeddings(style_version_id, model_id, rows)
        cached.update({(k, d): v for k, d, v in rows})
    return [cached[entry] for entry in catalog]


def candidate_keys(
    library: SketchLibrary,
    style_version_id: int,
    scenes: Sequence[str],
    embedder: Embedder | None,
    *,
    full_limit: int = 300,
    top_k: int = 5,
) -> list[list[Entry]]:
    """Per scene, the (idea_key, description) entries the planner may pick from.

    With `full_limit` entries or fewer every scene gets the whole catalog and no
    embedding call is made. Above it, each scene gets its `top_k` nearest entries
    by cosine similarity; ties keep catalog order.
    """
    if not scenes:
        return []
    catalog = library.idea_catalog(style_version_id)
    if len(catalog) <= full_limit:
        return [list(catalog) for _ in scenes]
    if embedder is None:
        raise PlanningError(
            f"catalog has {len(catalog)} ideas (> {full_limit}); an embedder is required "
            "to narrow it per scene"
        )
    catalog_matrix = _matrix(
        _catalog_vectors(library, style_version_id, catalog, embedder), "catalog"
    )
    scene_matrix = _matrix(_embed(embedder, scenes, "scene"), "scene")
    if scene_matrix.shape[1] != catalog_matrix.shape[1]:
        raise PlanningError(
            f"scene vectors ({scene_matrix.shape[1]}) and catalog vectors "
            f"({catalog_matrix.shape[1]}) differ in dimension"
        )
    scores = scene_matrix @ catalog_matrix.T
    result: list[list[Entry]] = []
    for row in scores:
        order = np.argsort(-row, kind="stable")[:top_k]
        result.append([catalog[int(i)] for i in order])
    return result
