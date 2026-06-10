"""add is_multimodal column to embedding & reranker model

Revision ID: 2a8f1c4d6e90
Revises: f4c9a7213eb5
Create Date: 2026-05-29 16:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from db.op.safe_add import safe_add_column


# revision identifiers, used by Alembic.
revision: str = '2a8f1c4d6e90'
down_revision: Union[str, Sequence[str], None] = 'f4c9a7213eb5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    safe_add_column(
        "pai_embedding_model",
        sa.Column("is_multimodal", sa.Boolean(), nullable=True, server_default=sa.false()),
    )
    safe_add_column(
        "pai_reranker_model",
        sa.Column("is_multimodal", sa.Boolean(), nullable=True, server_default=sa.false()),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("pai_reranker_model", "is_multimodal")
    op.drop_column("pai_embedding_model", "is_multimodal")
