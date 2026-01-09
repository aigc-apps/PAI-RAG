"""add chunk_config, FAQ tables and FAQ config columns

Revision ID: 4fc61e385531
Revises: 1432eea7c5b9
Create Date: 2025-12-25 17:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from db.op.safe_add import safe_add_column


# revision identifiers, used by Alembic.
revision: str = '4fc61e385531'
down_revision: Union[str, Sequence[str], None] = '1432eea7c5b9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. Add chunk_config column to pai_knowledgebase_file table
    safe_add_column('pai_knowledgebase_file', sa.Column('chunk_config', sa.JSON(), nullable=True))
    
    # 2. Create pai_chatbot_faq_config table (FAQ configuration)
    op.create_table(
        'pai_chatbot_faq_config',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('chatbot_id', sa.String(), nullable=True),
        sa.Column('tenant_id', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_pai_chatbot_faq_config_chatbot_id', 'pai_chatbot_faq_config', ['chatbot_id'], unique=False)
    op.create_index('ix_pai_chatbot_faq_config_tenant_id', 'pai_chatbot_faq_config', ['tenant_id'], unique=False)
    
    # 3. Create pai_chatbot_faq table (FAQ items)
    op.create_table(
        'pai_chatbot_faq',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('question', sa.Text(), nullable=True),
        sa.Column('answer', sa.Text(), nullable=True),
        sa.Column('chatbot_id', sa.String(), nullable=True),
        sa.Column('file_id', sa.String(), nullable=True),
        sa.Column('active', sa.Boolean(), nullable=True, server_default=sa.text('true')),
        sa.Column('tenant_id', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['chatbot_id'], ['pai_chatbot_model.app_id'], ondelete='CASCADE')
    )
    op.create_index('ix_pai_chatbot_faq_chatbot_id', 'pai_chatbot_faq', ['chatbot_id'], unique=False)
    op.create_index('ix_pai_chatbot_faq_tenant_id', 'pai_chatbot_faq', ['tenant_id'], unique=False)
    
    # 4. Add FAQ configuration columns to pai_chatbot_faq_config table
    safe_add_column(
        'pai_chatbot_faq_config',
        sa.Column('score_threshold', sa.Float(), nullable=True)
    )
    safe_add_column(
        'pai_chatbot_faq_config',
        sa.Column('embedding_model', sa.String(), nullable=True)
    )
    safe_add_column(
        'pai_chatbot_faq_config',
        sa.Column('enable_question_in_retrieval', sa.Boolean(), nullable=True)
    )
    safe_add_column(
        'pai_chatbot_faq_config',
        sa.Column('enable_question_in_response', sa.Boolean(), nullable=True)
    )
    safe_add_column(
        'pai_chatbot_faq_config',
        sa.Column('enable_answer_in_retrieval', sa.Boolean(), nullable=True)
    )
    safe_add_column(
        'pai_chatbot_faq_config',
        sa.Column('enable_answer_in_response', sa.Boolean(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Remove FAQ configuration columns from pai_chatbot_faq_config table
    op.drop_column('pai_chatbot_faq_config', 'enable_answer_in_response')
    op.drop_column('pai_chatbot_faq_config', 'enable_answer_in_retrieval')
    op.drop_column('pai_chatbot_faq_config', 'enable_question_in_response')
    op.drop_column('pai_chatbot_faq_config', 'enable_question_in_retrieval')
    op.drop_column('pai_chatbot_faq_config', 'embedding_model')
    op.drop_column('pai_chatbot_faq_config', 'score_threshold')
    
    # Drop pai_chatbot_faq table
    op.drop_index('ix_pai_chatbot_faq_tenant_id', table_name='pai_chatbot_faq')
    op.drop_index('ix_pai_chatbot_faq_chatbot_id', table_name='pai_chatbot_faq')
    op.drop_table('pai_chatbot_faq')
    
    # Drop pai_chatbot_faq_config table
    op.drop_index('ix_pai_chatbot_faq_config_tenant_id', table_name='pai_chatbot_faq_config')
    op.drop_index('ix_pai_chatbot_faq_config_chatbot_id', table_name='pai_chatbot_faq_config')
    op.drop_table('pai_chatbot_faq_config')
    
    # Remove chunk_config column from pai_knowledgebase_file table
    op.drop_column('pai_knowledgebase_file', 'chunk_config')

