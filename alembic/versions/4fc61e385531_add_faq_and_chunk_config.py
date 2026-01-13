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
down_revision: Union[str, Sequence[str], None] = '1f0950a076a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 1. Add chunk_config column to pai_knowledgebase_file table
    safe_add_column('pai_knowledgebase_file', sa.Column('chunk_config', sa.JSON(), nullable=True))
    safe_add_column('pai_chatbot_model', sa.Column('enable_faq', sa.Boolean(), nullable=True, default=False))
    safe_add_column('pai_chatbot_model', sa.Column('faq_config', sa.JSON(), nullable=True))
    

def downgrade() -> None:
    """Downgrade schema."""
    # Remove chunk_config column from pai_knowledgebase_file table
    op.drop_column('pai_knowledgebase_file', 'chunk_config')
    op.drop_column('pai_chatbot_model', 'faq_config')
    op.drop_column('pai_chatbot_model', 'enable_faq')

