from pathlib import Path

from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import inspect

from narratoflow.library.db import open_library
from narratoflow.library.orm import Base


def test_open_creates_schema_and_wal(tmp_path: Path) -> None:
    engine = open_library(tmp_path / "lib" / "library.db")
    with engine.connect() as c:
        assert c.exec_driver_sql("PRAGMA journal_mode").scalar() == "wal"
        assert c.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1
    assert {"channel", "style_version", "sketch"} <= set(inspect(engine).get_table_names())


def test_open_is_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "library.db"
    open_library(p).dispose()
    engine = open_library(p)
    with engine.connect() as c:
        rows = c.exec_driver_sql("SELECT version_num FROM alembic_version").fetchall()
    assert len(rows) == 1


def test_migrations_match_models(tmp_path: Path) -> None:
    engine = open_library(tmp_path / "library.db")
    with engine.connect() as c:
        diff = compare_metadata(MigrationContext.configure(c), Base.metadata)
    assert diff == []
