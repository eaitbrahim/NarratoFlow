"""SQLAlchemy ORM tables for the sketch library."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, String, TypeDecorator, UniqueConstraint
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from narratoflow.library.schemas import SketchOrigin, SketchStatus


def _now() -> datetime:
    return datetime.now(UTC)


def _enum_values(enum_cls: type[SketchOrigin] | type[SketchStatus]) -> list[str]:
    return [member.value for member in enum_cls]


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
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=_now)


class StyleVersion(Base):
    __tablename__ = "style_version"
    __table_args__ = (UniqueConstraint("channel_id", "fingerprint"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    channel_id: Mapped[int] = mapped_column(
        ForeignKey("channel.id", ondelete="CASCADE"), index=True, nullable=False
    )
    fingerprint: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=_now)


class Sketch(Base):
    __tablename__ = "sketch"
    __table_args__ = (UniqueConstraint("style_version_id", "idea_key", "blob_hash"),)

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
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=_now)
