"""Alembic environment; the connection is supplied by ``open_library``."""

from alembic import context
from sqlalchemy import Connection

import narratoflow.library.orm  # noqa: F401  (registers tables on the metadata)
from narratoflow.errors import LibraryError
from narratoflow.library.orm import Base

target_metadata = Base.metadata


def run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


connection = context.config.attributes.get("connection")
if connection is None:
    raise LibraryError(
        "library migrations run only through narratoflow.library.db.open_library(), which "
        "supplies the connection; the alembic CLI and offline (--sql) mode are not supported"
    )
run_migrations(connection)
