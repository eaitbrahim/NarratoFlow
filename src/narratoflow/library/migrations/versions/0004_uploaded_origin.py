"""allow origin 'uploaded' in the sketch.origin CHECK

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-13

The CHECK values are frozen here on purpose (not read from the enums), like 0002.
Batch mode rebuilds `sketch`, which is safe because nothing references it (see 0002).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ORIGIN_CHECK = "origin IN ('generated','bundled','copied','uploaded')"
_OLD_ORIGIN_CHECK = "origin IN ('generated','bundled','copied')"


def _rebuild_origin_check(check: str) -> None:
    with op.batch_alter_table("sketch") as batch:
        batch.drop_constraint("ck_sketch_origin", type_="check")
        batch.create_check_constraint("ck_sketch_origin", check)


def upgrade() -> None:
    _rebuild_origin_check(_ORIGIN_CHECK)


def downgrade() -> None:
    uploaded = op.get_bind().execute(
        sa.text("SELECT COUNT(*) FROM sketch WHERE origin = 'uploaded'")
    )
    count = int(uploaded.scalar_one())
    if count:
        raise RuntimeError(f"cannot downgrade: {count} uploaded sketches exist")
    _rebuild_origin_check(_OLD_ORIGIN_CHECK)
