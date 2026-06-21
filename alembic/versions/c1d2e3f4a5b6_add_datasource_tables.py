"""add datasource tables

Adds the DataSource layer tables:
- pai_datasource           (config + aggregate sync state, one-to-many under a KB)
- pai_datasource_document  (per-document manifest / file list)
- pai_datasource_sync_run  (sync history)

Idempotent: skips tables that already exist (dev create_all may have made them).

Revision ID: c1d2e3f4a5b6
Revises: a1b2c3d4e5f6
Create Date: 2026-06-21 12:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c1d2e3f4a5b6"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = set(sa.inspect(bind).get_table_names())

    if "pai_datasource" not in existing:
        op.create_table(
            "pai_datasource",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("tenant_id", sa.String(length=64), nullable=True),
            sa.Column("kb_id", sa.String(length=64), nullable=True),
            sa.Column("name", sa.String(length=100), nullable=True),
            sa.Column("datasource_key", sa.String(length=64), nullable=True),
            sa.Column("source_type", sa.String(length=32), nullable=True),
            sa.Column("source_config", sa.JSON(), nullable=True),
            sa.Column("sync_schedule", sa.String(length=128), nullable=True),
            sa.Column("next_sync_at", sa.DateTime(), nullable=True),
            sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="idle"),
            sa.Column("last_sync_at", sa.DateTime(), nullable=True),
            sa.Column("last_sync_finished_at", sa.DateTime(), nullable=True),
            sa.Column("last_sync_duration_ms", sa.Integer(), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("doc_count", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("last_sync_report", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["kb_id"], ["pai_knowledgebase.id"], ondelete="CASCADE"),
            sa.UniqueConstraint("kb_id", "datasource_key", "tenant_id", name="unique_datasource_key"),
        )

    if "pai_datasource_document" not in existing:
        op.create_table(
            "pai_datasource_document",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("tenant_id", sa.String(length=64), nullable=True),
            sa.Column("datasource_id", sa.String(length=64), nullable=True),
            sa.Column("kb_id", sa.String(length=64), nullable=True),
            sa.Column("doc_id", sa.String(length=512), nullable=True),
            sa.Column("file_id", sa.String(length=64), nullable=True),
            sa.Column("path", sa.Text(), nullable=True),
            sa.Column("source_url", sa.Text(), nullable=True),
            sa.Column("fetch_url", sa.Text(), nullable=True),
            sa.Column("title", sa.Text(), nullable=True),
            sa.Column("section", sa.String(length=255), nullable=True),
            sa.Column("product", sa.String(length=255), nullable=True),
            sa.Column("summary", sa.Text(), nullable=True),
            sa.Column("lang", sa.String(length=16), nullable=True),
            sa.Column("content_hash", sa.String(length=64), nullable=True),
            sa.Column("byte_size", sa.Integer(), nullable=True),
            sa.Column("source_meta", sa.JSON(), nullable=True),
            sa.Column("doc_status", sa.String(length=32), nullable=False, server_default="discovered"),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("first_seen_at", sa.DateTime(), nullable=True),
            sa.Column("last_fetched_at", sa.DateTime(), nullable=True),
            sa.Column("last_changed_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["datasource_id"], ["pai_datasource.id"], ondelete="CASCADE"),
            sa.UniqueConstraint("datasource_id", "doc_id", name="unique_datasource_doc"),
        )
        op.create_index(
            "idx_datasource_doc_status", "pai_datasource_document",
            ["datasource_id", "doc_status"],
        )

    if "pai_datasource_sync_run" not in existing:
        op.create_table(
            "pai_datasource_sync_run",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("tenant_id", sa.String(length=64), nullable=True),
            sa.Column("datasource_id", sa.String(length=64), nullable=True),
            sa.Column("kb_id", sa.String(length=64), nullable=True),
            sa.Column("trigger", sa.String(length=32), nullable=False, server_default="manual"),
            sa.Column("triggered_by", sa.String(length=64), nullable=True),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="running"),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("finished_at", sa.DateTime(), nullable=True),
            sa.Column("duration_ms", sa.Integer(), nullable=True),
            sa.Column("n_discovered", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("n_added", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("n_updated", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("n_deleted", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("n_unchanged", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("n_failed", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("report", sa.JSON(), nullable=True),
            sa.Column("error", sa.Text(), nullable=True),
            sa.ForeignKeyConstraint(["datasource_id"], ["pai_datasource.id"], ondelete="CASCADE"),
        )
        op.create_index(
            "idx_datasource_sync_run", "pai_datasource_sync_run",
            ["datasource_id", "started_at"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    existing = set(sa.inspect(bind).get_table_names())
    if "pai_datasource_sync_run" in existing:
        op.drop_index("idx_datasource_sync_run", table_name="pai_datasource_sync_run")
        op.drop_table("pai_datasource_sync_run")
    if "pai_datasource_document" in existing:
        op.drop_index("idx_datasource_doc_status", table_name="pai_datasource_document")
        op.drop_table("pai_datasource_document")
    if "pai_datasource" in existing:
        op.drop_table("pai_datasource")
