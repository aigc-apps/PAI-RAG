"""add evaluator prompt_template

Revision ID: d9f2e4a6b8c1
Revises: c8e1f3a2b4d7
Create Date: 2026-04-17 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
from db.op.safe_add import safe_add_column
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd9f2e4a6b8c1'
down_revision: Union[str, Sequence[str], None] = 'c8e1f3a2b4d7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    safe_add_column(
        'pai_evaluator_config',
        sa.Column('prompt_template', sa.Text(), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('pai_evaluator_config', 'prompt_template')
