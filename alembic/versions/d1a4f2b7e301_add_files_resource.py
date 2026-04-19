"""add files resource (pai_file, pai_file_text_content)

Part of Phase 1 of the /v1/files refactor — stands up a new, independent
File resource decoupled from the knowledgebase-file table.

Revision ID: d1a4f2b7e301
Revises: c8e1f3a2b4d7
Create Date: 2026-04-17 00:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd1a4f2b7e301'
down_revision: Union[str, Sequence[str], None] = 'c8e1f3a2b4d7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    if 'pai_file' not in existing:
        op.create_table(
            'pai_file',
            sa.Column('id', sa.String(length=64), primary_key=True),
            sa.Column('tenant_id', sa.String(length=64), nullable=False),
            sa.Column('purpose', sa.String(length=32), nullable=False, server_default='chat_attachment'),
            sa.Column('alias_id', sa.String(length=64), nullable=True),
            sa.Column('file_name', sa.String(length=255), nullable=True),
            sa.Column('file_extension', sa.String(length=32), nullable=True),
            sa.Column('file_size', sa.BigInteger(), nullable=False, server_default='0'),
            sa.Column('file_md5', sa.String(length=64), nullable=True),
            sa.Column('file_path', sa.Text(), nullable=True),
            sa.Column('mime_type', sa.String(length=128), nullable=True),
            sa.Column('status', sa.String(length=32), nullable=False, server_default='pending'),
            sa.Column('failed_reason', sa.Text(), nullable=True),
            sa.Column('ref_count', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('expires_at', sa.DateTime(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), nullable=False),
            sa.Column('file_metadata', sa.JSON(), nullable=True),
            sa.UniqueConstraint('tenant_id', 'file_md5', 'purpose', name='uq_file_tenant_md5_purpose'),
            sa.UniqueConstraint('tenant_id', 'alias_id', name='uq_file_tenant_alias'),
        )
        op.create_index('ix_pai_file_tenant_id', 'pai_file', ['tenant_id'])
        op.create_index('ix_file_tenant_purpose_created', 'pai_file', ['tenant_id', 'purpose', 'created_at'])
        op.create_index('ix_file_expires_at', 'pai_file', ['expires_at'])
        op.create_index('ix_pai_file_ref_count', 'pai_file', ['ref_count'])

    if 'pai_file_text_content' not in existing:
        op.create_table(
            'pai_file_text_content',
            sa.Column('file_id', sa.String(length=64), primary_key=True),
            sa.Column('tenant_id', sa.String(length=64), nullable=False),
            sa.Column('content', sa.Text(), nullable=True),
            sa.Column('content_length', sa.Integer(), nullable=False, server_default='0'),
            sa.Column('extractor_version', sa.String(length=32), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=False),
            sa.Column('updated_at', sa.DateTime(), nullable=False),
        )
        op.create_index('ix_pai_file_text_content_tenant_id', 'pai_file_text_content', ['tenant_id'])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())

    if 'pai_file_text_content' in existing:
        op.drop_index('ix_pai_file_text_content_tenant_id', table_name='pai_file_text_content')
        op.drop_table('pai_file_text_content')

    if 'pai_file' in existing:
        op.drop_index('ix_pai_file_ref_count', table_name='pai_file')
        op.drop_index('ix_file_expires_at', table_name='pai_file')
        op.drop_index('ix_file_tenant_purpose_created', table_name='pai_file')
        op.drop_index('ix_pai_file_tenant_id', table_name='pai_file')
        op.drop_table('pai_file')
