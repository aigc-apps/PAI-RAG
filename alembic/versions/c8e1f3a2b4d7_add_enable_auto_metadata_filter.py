"""add enable_auto_metadata_filter

Revision ID: c8e1f3a2b4d7
Revises: a3f5c7d89e12
Create Date: 2026-04-15 15:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
from db.op.safe_add import safe_add_column
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c8e1f3a2b4d7'
down_revision: Union[str, Sequence[str], None] = 'a3f5c7d89e12'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    safe_add_column(
        'pai_chatbot_model',
        sa.Column('enable_auto_metadata_filter', sa.Boolean(), nullable=True, default=False),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('pai_chatbot_model', 'enable_auto_metadata_filter')
