"""add file chunk table

Backs in-file retrieval (GET /v1/files/{id}/chunks?query=...). Populated by
``process_file_resource_task`` after text extraction; cascade-deleted when
the parent FileEntity is hard-deleted.

Revision ID: f4c9a7213eb5
Revises: e2b5c3f9d842
Create Date: 2026-04-17 02:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f4c9a7213eb5'
down_revision: Union[str, Sequence[str], None] = 'e2b5c3f9d842'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if 'pai_file_chunk' in set(inspector.get_table_names()):
        return

    op.create_table(
        'pai_file_chunk',
        sa.Column('id', sa.String(length=64), primary_key=True),
        sa.Column('tenant_id', sa.String(length=64), nullable=False),
        sa.Column('file_id', sa.String(length=64), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('content', sa.Text(), nullable=True),
        sa.Column('start_offset', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('end_offset', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('token_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('chunk_metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
    )
    op.create_index('ix_pai_file_chunk_tenant_id', 'pai_file_chunk', ['tenant_id'])
    op.create_index('ix_file_chunk_tenant_file', 'pai_file_chunk', ['tenant_id', 'file_id'])
    op.create_index(
        'ix_file_chunk_tenant_file_index', 'pai_file_chunk',
        ['tenant_id', 'file_id', 'chunk_index'],
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if 'pai_file_chunk' not in set(inspector.get_table_names()):
        return
    op.drop_index('ix_file_chunk_tenant_file_index', table_name='pai_file_chunk')
    op.drop_index('ix_file_chunk_tenant_file', table_name='pai_file_chunk')
    op.drop_index('ix_pai_file_chunk_tenant_id', table_name='pai_file_chunk')
    op.drop_table('pai_file_chunk')
