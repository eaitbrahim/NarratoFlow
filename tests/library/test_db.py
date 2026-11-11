import multiprocessing
import sqlite3
from datetime import datetime, timedelta
from multiprocessing.synchronize import Barrier
from pathlib import Path
from typing import Any

import pytest
from alembic import command as alembic_command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import inspect
from sqlalchemy.exc import StatementError
from sqlalchemy.orm import Session

from narratoflow.errors import LibraryError
from narratoflow.library import db
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


_PROCESSES = 4
_ROUNDS = 5


def _open_concurrently(paths: list[str], barrier: Barrier, results: Any) -> None:
    """Child process: open each fresh DB at the same moment as the other children."""
    for path in paths:
        barrier.wait(timeout=60)
        try:
            open_library(Path(path)).dispose()
        except Exception as e:  # reported to the parent, which fails the test
            results.put(f"{path}: {type(e).__name__}: {e}")
        else:
            results.put("ok")


def test_concurrent_first_open_is_serialized(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    barrier = ctx.Barrier(_PROCESSES)
    results = ctx.Queue()
    paths = [str(tmp_path / f"lib{n}" / "library.db") for n in range(_ROUNDS)]
    procs = [
        ctx.Process(target=_open_concurrently, args=(paths, barrier, results))
        for _ in range(_PROCESSES)
    ]
    for p in procs:
        p.start()
    outcomes = [results.get(timeout=60) for _ in range(_PROCESSES * _ROUNDS)]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    assert [o for o in outcomes if o != "ok"] == []
    for path in paths:
        engine = open_library(Path(path))
        with engine.connect() as c:
            versions = c.exec_driver_sql("SELECT version_num FROM alembic_version").fetchall()
        assert len(versions) == 1
        assert {"channel", "style_version", "sketch"} <= set(inspect(engine).get_table_names())
        engine.dispose()


def test_open_failure_wrapped_in_library_error(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(LibraryError, match="cannot open library") as exc:
        open_library(blocker / "library.db")
    assert exc.value.__cause__ is not None


def test_open_non_database_wrapped_in_library_error(tmp_path: Path) -> None:
    bogus = tmp_path / "library.db"
    bogus.write_bytes(b"not a sqlite database" * 100)
    with pytest.raises(LibraryError, match="cannot open library") as exc:
        open_library(bogus)
    assert exc.value.__cause__ is not None


def test_failed_migration_rolls_back_ddl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_upgrade = alembic_command.upgrade

    def upgrade_then_crash(config: Any, revision: str) -> None:
        real_upgrade(config, revision)
        raise RuntimeError("crash after DDL")

    monkeypatch.setattr(alembic_command, "upgrade", upgrade_then_crash)
    path = tmp_path / "library.db"
    with pytest.raises(RuntimeError, match="crash after DDL"):
        open_library(path)
    with sqlite3.connect(path) as conn:
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    conn.close()
    assert tables == []


class _FlakyCursor:
    def __init__(self, failures: int, error: sqlite3.OperationalError) -> None:
        self.failures = failures
        self.error = error
        self.calls = 0

    def execute(self, sql: str) -> None:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error


def _error(code: int) -> sqlite3.OperationalError:
    error = sqlite3.OperationalError("simulated")
    error.sqlite_errorcode = code
    return error


def test_enable_wal_retries_while_busy() -> None:
    cursor = _FlakyCursor(2, _error(sqlite3.SQLITE_BUSY))
    db._enable_wal(cursor)
    assert cursor.calls == 3


def test_enable_wal_reraises_other_errors() -> None:
    cursor = _FlakyCursor(1, _error(sqlite3.SQLITE_IOERR))
    with pytest.raises(sqlite3.OperationalError):
        db._enable_wal(cursor)
    assert cursor.calls == 1


def test_enable_wal_gives_up_after_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db, "_BUSY_TIMEOUT_S", 0.0)
    cursor = _FlakyCursor(99, _error(sqlite3.SQLITE_BUSY))
    with pytest.raises(sqlite3.OperationalError):
        db._enable_wal(cursor)
    assert cursor.calls == 1


@pytest.mark.parametrize("offline", [False, True])
def test_env_without_connection_explains_open_library(offline: bool) -> None:
    from importlib import resources

    from alembic.config import Config

    config = Config()
    location = resources.files("narratoflow.library") / "migrations"
    with resources.as_file(location) as path:
        config.set_main_option("script_location", str(path))
        with pytest.raises(LibraryError, match="open_library"):
            alembic_command.upgrade(config, "head", sql=offline)
