"""initial sketch library schema

Revision ID: 0001
Revises:
Create Date: 2026-11-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "channel",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("slug", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug"),
    )
    op.create_table(
        "style_version",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("channel_id", sa.Integer(), nullable=False),
        sa.Column("fingerprint", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["channel_id"], ["channel.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("channel_id", "fingerprint"),
    )
    op.create_index("ix_style_version_channel_id", "style_version", ["channel_id"])
    op.create_table(
        "sketch",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("style_version_id", sa.Integer(), nullable=False),
        sa.Column("idea_key", sa.String(), nullable=False),
        sa.Column("idea_description", sa.String(), server_default="", nullable=False),
        sa.Column("blob_hash", sa.String(), nullable=False),
        sa.Column("suffix", sa.String(), nullable=False),
        sa.Column(
            "origin",
            sa.Enum(
                "generated",
                "bundled",
                "copied",
                name="sketchorigin",
                native_enum=False,
                create_constraint=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Enum(
                "passed",
                "unchecked",
                "failed",
                name="sketchstatus",
                native_enum=False,
                create_constraint=False,
            ),
            nullable=False,
        ),
        sa.Column("status_reason", sa.String(), nullable=True),
        sa.Column("mirror_safe", sa.Boolean(), nullable=False),
        sa.Column("provenance", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["style_version_id"], ["style_version.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("style_version_id", "idea_key", "blob_hash"),
    )
    op.create_index("ix_sketch_style_version_id", "sketch", ["style_version_id"])
    op.create_index("ix_sketch_idea_key", "sketch", ["idea_key"])


def downgrade() -> None:
    op.drop_table("sketch")
    op.drop_table("style_version")
    op.drop_table("channel")
