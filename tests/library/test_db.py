from datetime import datetime, timedelta
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import inspect
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import Session

from narratoflow.library.db import open_library
from narratoflow.library.orm import Base, Channel


def test_open_creates_schema_and_wal(tmp_path: Path) -> None:
    engine = open_library(tmp_path / "lib" / "library.db")
    with engine.connect() as c:
        assert c.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"
        assert c.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
    assert {"channel", "style_version", "sketch"} <= set(inspect(engine).get_table_names())
    engine.dispose()


def test_open_is_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "library.db"
    open_library(p).dispose()
    engine = open_library(p)
    with engine.connect() as c:
        rows = c.exec_driver_sql("SELECT version_num FROM alembic_version").fetchall()
    assert len(rows) == 1
    engine.dispose()


def test_migrations_match_models(tmp_path: Path) -> None:
    engine = open_library(tmp_path / "library.db")
    with engine.connect() as c:
        diff = compare_metadata(MigrationContext.configure(c), Base.metadata)
    assert diff == []
    engine.dispose()


def test_timestamps_roundtrip_as_aware_utc(tmp_path: Path) -> None:
    engine = open_library(tmp_path / "library.db")
    with Session(engine) as s:
        s.add(Channel(slug="history"))
        s.commit()
    with Session(engine) as s:
        created = s.query(Channel).one().created_at
    assert created.tzinfo is not None
    assert created.utcoffset() == timedelta(0)
    with Session(engine) as s, pytest.raises(StatementError, match="naive"):
        s.add(Channel(slug="x", created_at=datetime(2026, 1, 1)))
        s.commit()
    engine.dispose()
