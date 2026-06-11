"""add max_tokens to llm model

Revision ID: a1b2c3d4e5f6
Revises: 7c6d9a1b2e34
Create Date: 2026-06-10 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from db.op.safe_add import safe_add_column


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "7c6d9a1b2e34"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    safe_add_column(
        "pai_llm_model",
        sa.Column("max_tokens", sa.Integer(), nullable=True, server_default="8000"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("pai_llm_model", "max_tokens")
