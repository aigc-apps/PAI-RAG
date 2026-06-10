"""add vision_model_id to chatapp

Revision ID: 7c6d9a1b2e34
Revises: 2a8f1c4d6e90
Create Date: 2026-06-10 14:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from db.op.safe_add import safe_add_column


# revision identifiers, used by Alembic.
revision: str = "7c6d9a1b2e34"
down_revision: Union[str, Sequence[str], None] = "2a8f1c4d6e90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    safe_add_column(
        "pai_chatbot_model",
        sa.Column("vision_model_id", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("pai_chatbot_model", "vision_model_id")
