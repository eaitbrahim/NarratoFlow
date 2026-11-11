"""Open the library database and bring its schema up to date."""

import sqlite3
import time
from importlib import resources
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import Connection, Engine, create_engine, event
from sqlalchemy.exc import DatabaseError

from narratoflow.errors import LibraryError

_BUSY_TIMEOUT_S = 30.0
_RETRY_MAX_DELAY_S = 0.1


def _is_busy(error: sqlite3.OperationalError) -> bool:
    code = getattr(error, "sqlite_errorcode", None)
    return code is not None and code & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)


def _enable_wal(cursor: Any) -> None:
    """Switch to WAL, retrying while another connection holds a lock.

    The switch needs an exclusive lock and SQLite reports SQLITE_BUSY at once (the busy
    timeout is not applied), so concurrent first opens of a new library must retry.
    """
    deadline = time.monotonic() + _BUSY_TIMEOUT_S
    delay = 0.005
    while True:
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as e:
            remaining = deadline - time.monotonic()
            if not _is_busy(e) or remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, _RETRY_MAX_DELAY_S)


def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    try:
        _enable_wal(cursor)
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def _begin_immediate(connection: Connection) -> None:
    connection.exec_driver_sql("BEGIN IMMEDIATE")


def _migrate(db_path: Path) -> None:
    """Run Alembic to head inside one BEGIN IMMEDIATE transaction.

    pysqlite's own transaction handling is disabled (isolation_level=None) so that the
    DDL runs inside our explicit transaction: the upgrade is atomic, and SQLite's write
    lock (with the busy timeout) serializes concurrent first opens of a new library.
    """
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"timeout": _BUSY_TIMEOUT_S, "isolation_level": None},
    )
    event.listen(engine, "connect", _set_pragmas)
    event.listen(engine, "begin", _begin_immediate)
    config = Config()
    script_location = resources.files("narratoflow.library") / "migrations"
    try:
        with resources.as_file(script_location) as location:
            config.set_main_option("script_location", str(location))
            with engine.begin() as connection:
                config.attributes["connection"] = connection
                command.upgrade(config, "head")
    finally:
        engine.dispose()


def open_library(db_path: Path) -> Engine:
    """Create (if needed) and migrate the library DB, returning a ready engine."""
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _migrate(db_path)
    except (OSError, DatabaseError) as e:
        raise LibraryError(f"cannot open library at {db_path}: {e}") from e
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": _BUSY_TIMEOUT_S})
    event.listen(engine, "connect", _set_pragmas)
    return engine
