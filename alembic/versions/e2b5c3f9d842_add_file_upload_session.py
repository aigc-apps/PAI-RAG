"""add file upload session table

Part of Phase 2 — enables resumable / chunked uploads to /v1/files.

Revision ID: e2b5c3f9d842
Revises: d1a4f2b7e301
Create Date: 2026-04-17 01:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e2b5c3f9d842'
down_revision: Union[str, Sequence[str], None] = 'd1a4f2b7e301'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    if 'pai_file_upload_session' not in existing:
        op.create_table(
            'pai_file_upload_session',
            sa.Column('id', sa.String(length=64), primary_key=True),
            sa.Column('tenant_id', sa.String(length=64), nullable=False),
            sa.Column('file_name', sa.String(length=255), nullable=True),
            sa.Column('purpose', sa.String(length=32), nullable=False, server_default='chat_attachment'),
            sa.Column('expires_in_seconds', sa.Integer(), nullable=True),
            sa.Column('status', sa.String(length=32), nullable=False, server_default='active'),
            sa.Column('expires_at', sa.DateTime(), nullable=True),
            sa.Column('parts', sa.JSON(), nullable=True),
            sa.Column('file_id', sa.String(length=64), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), nullable=False),
        )
        op.create_index(
            'ix_pai_file_upload_session_tenant_id',
            'pai_file_upload_session',
            ['tenant_id'],
        )
        op.create_index(
            'ix_file_upload_session_tenant_status',
            'pai_file_upload_session',
            ['tenant_id', 'status'],
        )
        op.create_index(
            'ix_file_upload_session_expires_at',
            'pai_file_upload_session',
            ['expires_at'],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())
    if 'pai_file_upload_session' in existing:
        op.drop_index(
            'ix_file_upload_session_expires_at',
            table_name='pai_file_upload_session',
        )
        op.drop_index(
            'ix_file_upload_session_tenant_status',
            table_name='pai_file_upload_session',
        )
        op.drop_index(
            'ix_pai_file_upload_session_tenant_id',
            table_name='pai_file_upload_session',
        )
        op.drop_table('pai_file_upload_session')
