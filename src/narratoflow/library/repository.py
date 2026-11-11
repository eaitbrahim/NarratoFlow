"""Repository for storing and reusing sketches per channel and style version."""

from collections.abc import Sequence

import numpy as np
from sqlalchemy import Engine, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from narratoflow.errors import LibraryError
from narratoflow.library.blobs import BlobStore
from narratoflow.library.db import open_library
from narratoflow.library.keys import IdeaKey, parse_idea_key
from narratoflow.library.orm import Channel, IdeaEmbedding, Sketch, StyleVersion, utc_now
from narratoflow.library.schemas import Provenance, SketchOrigin, SketchRecord, SketchStatus
from narratoflow.settings import Settings

_REUSABLE = (SketchStatus.PASSED, SketchStatus.UNCHECKED)
_VECTOR_DTYPE = "<f4"


def _canonical_key(idea_key: str | IdeaKey) -> str:
    """Validate and normalize an idea key; raises InvalidIdeaKeyError."""
    return str(parse_idea_key(str(idea_key)))


def _record(row: Sketch) -> SketchRecord:
    return SketchRecord.model_validate(row, from_attributes=True)


class SketchLibrary:
    def __init__(self, engine: Engine, blobs: BlobStore) -> None:
        self._engine = engine
        self._blobs = blobs

    def ensure_style_version(self, channel_slug: str, fingerprint: str) -> int:
        last_error: IntegrityError | None = None
        for _ in range(3):
            try:
                return self._ensure_style_version(channel_slug, fingerprint)
            except IntegrityError as e:
                # Another writer created the same row concurrently; retry the lookup.
                last_error = e
        raise LibraryError(
            f"could not ensure style version for {channel_slug!r}/{fingerprint!r}"
        ) from last_error

    def _ensure_style_version(self, channel_slug: str, fingerprint: str) -> int:
        with Session(self._engine) as s:
            channel = s.scalars(select(Channel).where(Channel.slug == channel_slug)).first()
            if channel is None:
                channel = Channel(slug=channel_slug)
                s.add(channel)
                s.flush()
            version = s.scalars(
                select(StyleVersion).where(
                    StyleVersion.channel_id == channel.id,
                    StyleVersion.fingerprint == fingerprint,
                )
            ).first()
            if version is None:
                version = StyleVersion(channel_id=channel.id, fingerprint=fingerprint)
                s.add(version)
                s.flush()
            version_id = version.id
            s.commit()
            return version_id

    def add_sketch(
        self,
        style_version_id: int,
        idea_key: str | IdeaKey,
        data: bytes,
        suffix: str,
        origin: SketchOrigin,
        status: SketchStatus,
        mirror_safe: bool,
        provenance: Provenance,
        status_reason: str | None = None,
        *,
        description: str = "",
    ) -> SketchRecord:
        """Store a sketch, deduplicated on (style version, idea key, blob hash).

        Returns the existing record if present; its status is authoritative and may be FAILED.
        Re-adding identical content never un-quarantines it. The idea key is normalized
        with ``parse_idea_key``; an invalid key raises InvalidIdeaKeyError.
        """
        key = _canonical_key(idea_key)
        blob_hash = self._blobs.put(data, suffix)
        with Session(self._engine) as s:
            existing = self._find_existing(s, style_version_id, key, blob_hash)
            if existing is not None:
                return _record(existing)
            new = Sketch(
                style_version_id=style_version_id,
                idea_key=key,
                idea_description=description,
                blob_hash=blob_hash,
                suffix=suffix,
                origin=origin,
                status=status,
                status_reason=status_reason,
                mirror_safe=mirror_safe,
                provenance=provenance.model_dump(mode="json"),
            )
            s.add(new)
            try:
                s.commit()
            except IntegrityError as e:
                s.rollback()
                existing = self._find_existing(s, style_version_id, key, blob_hash)
                if existing is None:
                    raise LibraryError("could not add sketch: integrity violation") from e
                return _record(existing)
            return _record(new)

    @staticmethod
    def _find_existing(
        s: Session, style_version_id: int, idea_key: str, blob_hash: str
    ) -> Sketch | None:
        return s.scalars(
            select(Sketch).where(
                Sketch.style_version_id == style_version_id,
                Sketch.idea_key == idea_key,
                Sketch.blob_hash == blob_hash,
            )
        ).first()

    def find_reusable(self, style_version_id: int, idea_key: str | IdeaKey) -> list[SketchRecord]:
        key = _canonical_key(idea_key)
        with Session(self._engine) as s:
            rows = s.scalars(
                select(Sketch)
                .where(
                    Sketch.style_version_id == style_version_id,
                    Sketch.idea_key == key,
                    Sketch.status.in_(_REUSABLE),
                )
                .order_by(Sketch.id)
            ).all()
            return [_record(r) for r in rows]

    def idea_keys(self, style_version_id: int) -> list[str]:
        with Session(self._engine) as s:
            return list(
                s.scalars(
                    select(Sketch.idea_key)
                    .where(
                        Sketch.style_version_id == style_version_id,
                        Sketch.status.in_(_REUSABLE),
                    )
                    .distinct()
                    .order_by(Sketch.idea_key)
                )
            )

    def idea_catalog(self, style_version_id: int) -> list[tuple[str, str]]:
        with Session(self._engine) as s:
            rows = s.execute(
                select(Sketch.idea_key, Sketch.idea_description)
                .where(
                    Sketch.style_version_id == style_version_id,
                    Sketch.status.in_(_REUSABLE),
                )
                .order_by(Sketch.id)
            ).all()
        catalog: dict[str, str] = {}
        for key, description in rows:
            if description or key not in catalog:
                catalog[key] = description
        return sorted(catalog.items())

    def cached_embeddings(
        self, style_version_id: int, embed_model: str
    ) -> dict[tuple[str, str], list[float]]:
        """Cached catalog vectors keyed by (idea_key, description)."""
        with Session(self._engine) as s:
            rows = s.execute(
                select(IdeaEmbedding.idea_key, IdeaEmbedding.description, IdeaEmbedding.vector)
                .where(
                    IdeaEmbedding.style_version_id == style_version_id,
                    IdeaEmbedding.embed_model == embed_model,
                )
                .order_by(IdeaEmbedding.id)
            ).all()
        return {
            (key, description): np.frombuffer(blob, dtype=_VECTOR_DTYPE).astype(float).tolist()
            for key, description, blob in rows
        }

    def store_embeddings(
        self,
        style_version_id: int,
        embed_model: str,
        rows: Sequence[tuple[str, str, list[float]]],
    ) -> None:
        """Cache catalog vectors; rows already cached are left as they are."""
        if not rows:
            return
        now = utc_now()
        values = [
            {
                "style_version_id": style_version_id,
                "idea_key": key,
                "description": description,
                "embed_model": embed_model,
                "vector": np.asarray(vector, dtype=_VECTOR_DTYPE).tobytes(),
                "created_at": now,
            }
            for key, description, vector in rows
        ]
        statement = sqlite_insert(IdeaEmbedding).on_conflict_do_nothing()
        with Session(self._engine) as s:
            s.execute(statement, values)
            s.commit()

    def quarantine(self, sketch_id: int, reason: str) -> None:
        with Session(self._engine) as s:
            row = s.get(Sketch, sketch_id)
            if row is None:
                raise LibraryError(f"unknown sketch id: {sketch_id}")
            row.status = SketchStatus.FAILED
            row.status_reason = reason
            s.commit()

    def copy_to(self, sketch_id: int, target_style_version_id: int) -> SketchRecord:
        """Copy a non-failed sketch into another style version.

        Returns the existing record if present; its status is authoritative and may be FAILED.
        """
        with Session(self._engine) as s:
            src = s.get(Sketch, sketch_id)
            if src is None:
                raise LibraryError(f"unknown sketch id: {sketch_id}")
            if src.status is SketchStatus.FAILED:
                raise LibraryError(f"cannot copy failed sketch {sketch_id}")
            provenance = Provenance.model_validate(src.provenance).model_copy(
                update={"copied_from_sketch_id": src.id}
            )
            existing = self._find_existing(s, target_style_version_id, src.idea_key, src.blob_hash)
            if existing is not None:
                return _record(existing)
            new = Sketch(
                style_version_id=target_style_version_id,
                idea_key=src.idea_key,
                idea_description=src.idea_description,
                blob_hash=src.blob_hash,
                suffix=src.suffix,
                origin=SketchOrigin.COPIED,
                status=src.status,
                status_reason=src.status_reason,
                mirror_safe=src.mirror_safe,
                provenance=provenance.model_dump(mode="json"),
            )
            s.add(new)
            try:
                s.commit()
            except IntegrityError as e:
                s.rollback()
                existing = self._find_existing(
                    s, target_style_version_id, src.idea_key, src.blob_hash
                )
                if existing is None:
                    raise LibraryError("could not copy sketch: integrity violation") from e
                return _record(existing)
            return _record(new)


def default_library(settings: Settings) -> SketchLibrary:
    engine = open_library(settings.data_dir / "library.db")
    return SketchLibrary(engine, BlobStore(settings.data_dir / "blobs"))
