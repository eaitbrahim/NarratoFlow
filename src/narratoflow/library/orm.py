"""SQLAlchemy ORM tables for the sketch library."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    LargeBinary,
    String,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from narratoflow.library.schemas import SketchOrigin, SketchStatus


def utc_now() -> datetime:
    return datetime.now(UTC)


def _enum_values(enum_cls: type[SketchOrigin] | type[SketchStatus]) -> list[str]:
    return [member.value for member in enum_cls]


def enum_check_sql(column: str, enum_cls: type[SketchOrigin] | type[SketchStatus]) -> str:
    """SQL for a CHECK that `column` holds one of the enum's values."""
    values = ",".join(f"'{value}'" for value in _enum_values(enum_cls))
    return f"{column} IN ({values})"


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC datetimes; SQLite drops tzinfo, so restore it on read."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime not allowed; use timezone-aware UTC")
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class Base(DeclarativeBase):
    pass


class Channel(Base):
    __tablename__ = "channel"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


class StyleVersion(Base):
    __tablename__ = "style_version"
    __table_args__ = (UniqueConstraint("channel_id", "fingerprint"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    channel_id: Mapped[int] = mapped_column(
        ForeignKey("channel.id", ondelete="CASCADE"), index=True, nullable=False
    )
    fingerprint: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)


class Sketch(Base):
    __tablename__ = "sketch"
    __table_args__ = (
        UniqueConstraint("style_version_id", "idea_key", "blob_hash"),
        CheckConstraint(enum_check_sql("status", SketchStatus), name="ck_sketch_status"),
        CheckConstraint(enum_check_sql("origin", SketchOrigin), name="ck_sketch_origin"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    style_version_id: Mapped[int] = mapped_column(
        ForeignKey("style_version.id"), index=True, nullable=False
    )
    idea_key: Mapped[str] = mapped_column(String, index=True, nullable=False)
    idea_description: Mapped[str] = mapped_column(
        String, nullable=False, default="", server_default=""
    )
    blob_hash: Mapped[str] = mapped_column(String, nullable=False)
    suffix: Mapped[str] = mapped_column(String, nullable=False)
    origin: Mapped[SketchOrigin] = mapped_column(
        Enum(
            SketchOrigin, values_callable=_enum_values, native_enum=False, create_constraint=False
        ),
        nullable=False,
    )
    status: Mapped[SketchStatus] = mapped_column(
        Enum(
            SketchStatus, values_callable=_enum_values, native_enum=False, create_constraint=False
        ),
        nullable=False,
    )
    status_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    mirror_safe: Mapped[bool] = mapped_column(nullable=False)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    strokes_hash: Mapped[str | None] = mapped_column(String, nullable=True)
    """Blob hash of the stroke set (suffix `.strokes.json`), once extracted."""


class IdeaEmbedding(Base):
    """Cached embedding of one catalog entry; `vector` is float32 little-endian."""

    __tablename__ = "idea_embedding"
    __table_args__ = (
        UniqueConstraint("style_version_id", "idea_key", "embed_model", "description"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    style_version_id: Mapped[int] = mapped_column(
        ForeignKey("style_version.id"), index=True, nullable=False
    )
    idea_key: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(String, nullable=False)
    embed_model: Mapped[str] = mapped_column(String, nullable=False)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
