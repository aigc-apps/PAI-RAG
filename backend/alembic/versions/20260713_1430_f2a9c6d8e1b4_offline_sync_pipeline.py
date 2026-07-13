"""offline sync pipeline state

Revision ID: f2a9c6d8e1b4
Revises: 9a4e7b2c1d30
Create Date: 2026-07-13 14:30:00.000000+00:00
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


revision: str = "f2a9c6d8e1b4"
down_revision: Union[str, None] = "9a4e7b2c1d30"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("background_jobs", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "progress",
                sa.JSON(),
                nullable=False,
                server_default=sa.text("'{}'"),
            )
        )
        batch_op.add_column(
            sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.create_index(
            batch_op.f("ix_background_jobs_lease_expires_at"),
            ["lease_expires_at"],
            unique=False,
        )

    with op.batch_alter_table("knowledge_data_sources", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "active_job_id",
                sqlmodel.sql.sqltypes.AutoString(length=64),
                nullable=True,
            )
        )
        batch_op.create_index(
            batch_op.f("ix_knowledge_data_sources_active_job_id"),
            ["active_job_id"],
            unique=False,
        )

    with op.batch_alter_table("knowledge_documents", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "search_index_status",
                sqlmodel.sql.sqltypes.AutoString(length=32),
                nullable=False,
                server_default="indexed",
            )
        )
        batch_op.add_column(
            sa.Column("search_index_error", sa.Text(), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "search_index_attempts",
                sa.Integer(),
                nullable=False,
                server_default="0",
            )
        )
        batch_op.create_index(
            batch_op.f("ix_knowledge_documents_search_index_status"),
            ["search_index_status"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("knowledge_documents", schema=None) as batch_op:
        batch_op.drop_index(
            batch_op.f("ix_knowledge_documents_search_index_status")
        )
        batch_op.drop_column("search_index_attempts")
        batch_op.drop_column("search_index_error")
        batch_op.drop_column("search_index_status")

    with op.batch_alter_table("knowledge_data_sources", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_knowledge_data_sources_active_job_id"))
        batch_op.drop_column("active_job_id")

    with op.batch_alter_table("background_jobs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_background_jobs_lease_expires_at"))
        batch_op.drop_column("cancel_requested_at")
        batch_op.drop_column("lease_expires_at")
        batch_op.drop_column("heartbeat_at")
        batch_op.drop_column("progress")
