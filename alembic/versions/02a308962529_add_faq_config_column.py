"""add faq_config column to pai_chatbot_faq_config

Revision ID: 02a308962529
Revises: 93f02e32f851
Create Date: 2025-12-25 17:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from db.op.safe_add import safe_add_column


# revision identifiers, used by Alembic.
revision: str = '02a308962529'
down_revision: Union[str, Sequence[str], None] = '93f02e32f851'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add FAQ configuration columns to pai_chatbot_faq_config table
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
        sa.Column('question_in_retrieval', sa.Boolean(), nullable=True)
    )
    safe_add_column(
        'pai_chatbot_faq_config',
        sa.Column('question_in_response', sa.Boolean(), nullable=True)
    )
    safe_add_column(
        'pai_chatbot_faq_config',
        sa.Column('answer_in_retrieval', sa.Boolean(), nullable=True)
    )
    safe_add_column(
        'pai_chatbot_faq_config',
        sa.Column('answer_in_response', sa.Boolean(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('pai_chatbot_faq_config', 'answer_in_response')
    op.drop_column('pai_chatbot_faq_config', 'answer_in_retrieval')
    op.drop_column('pai_chatbot_faq_config', 'question_in_response')
    op.drop_column('pai_chatbot_faq_config', 'question_in_retrieval')
    op.drop_column('pai_chatbot_faq_config', 'embedding_model')
    op.drop_column('pai_chatbot_faq_config', 'score_threshold')

