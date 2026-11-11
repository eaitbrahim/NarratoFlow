"""idea embedding cache and CHECK constraints on sketch.status/origin

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-10

The CHECK values are frozen here on purpose (not read from the enums), so this
revision keeps meaning the same thing if the enums grow later.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_STATUS_CHECK = "status IN ('passed','unchecked','failed')"
_ORIGIN_CHECK = "origin IN ('generated','bundled','copied')"


def upgrade() -> None:
    op.create_table(
        "idea_embedding",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("style_version_id", sa.Integer(), nullable=False),
        sa.Column("idea_key", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("embed_model", sa.String(), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["style_version_id"], ["style_version.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("style_version_id", "idea_key", "embed_model", "description"),
    )
    op.create_index("ix_idea_embedding_style_version_id", "idea_embedding", ["style_version_id"])
    # SQLite cannot ALTER in a CHECK: batch mode rebuilds the table, copying rows,
    # the foreign key, the unique dedupe constraint and the indexes.
    with op.batch_alter_table("sketch") as batch:
        batch.create_check_constraint("ck_sketch_status", _STATUS_CHECK)
        batch.create_check_constraint("ck_sketch_origin", _ORIGIN_CHECK)


def downgrade() -> None:
    with op.batch_alter_table("sketch") as batch:
        batch.drop_constraint("ck_sketch_origin", type_="check")
        batch.drop_constraint("ck_sketch_status", type_="check")
    op.drop_index("ix_idea_embedding_style_version_id", table_name="idea_embedding")
    op.drop_table("idea_embedding")
