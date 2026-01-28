"""add parallel_count to run_config and llm_judge_prompt to evaluator_config

Revision ID: fd546739d769
Revises: 1f0950a076a7
Create Date: 2026-01-26 15:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from db.op.safe_add import safe_add_column

# revision identifiers, used by Alembic.
revision: str = 'fd546739d769'
down_revision: Union[str, Sequence[str], None] = '4fc61e385531'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add parallel_count column to pai_run_config table
    safe_add_column('pai_run_config', sa.Column('parallel_count', sa.Integer(), nullable=True))
    
    # Add llm_judge_prompt column to pai_evaluator_config table
    safe_add_column('pai_evaluator_config', sa.Column('llm_judge_prompt', sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('pai_run_config', 'parallel_count')
    op.drop_column('pai_evaluator_config', 'llm_judge_prompt')

