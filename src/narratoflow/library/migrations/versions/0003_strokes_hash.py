"""nullable sketch.strokes_hash pointing at the stored stroke set blob

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("sketch", sa.Column("strokes_hash", sa.String(), nullable=True))


def downgrade() -> None:
    # Batch mode rebuilds `sketch` (safe: nothing references it, see 0002) and keeps
    # its rows, foreign key, unique constraint, CHECKs and indexes.
    with op.batch_alter_table("sketch") as batch:
        batch.drop_column("strokes_hash")
