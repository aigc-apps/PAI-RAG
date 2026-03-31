"""add message token_usage

Revision ID: a3f5c7d89e12
Revises: 4fc61e385531
Create Date: 2026-03-16 16:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
from db.op.safe_add import safe_add_column
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3f5c7d89e12'
down_revision: Union[str, Sequence[str], None] = '4fc61e385531'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add token_usage column to pai_message table
    safe_add_column('pai_message', sa.Column('token_usage', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('pai_message', 'token_usage')
