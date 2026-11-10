"""Open the library database and bring its schema up to date."""

from importlib import resources
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event


def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def open_library(db_path: Path) -> Engine:
    """Create (if needed) and migrate the library DB, returning a ready engine."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}", connect_args={"timeout": 30})
    event.listen(engine, "connect", _set_pragmas)
    config = Config()
    script_location = resources.files("narratoflow.library") / "migrations"
    with resources.as_file(script_location) as location:
        config.set_main_option("script_location", str(location))
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
    return engine
