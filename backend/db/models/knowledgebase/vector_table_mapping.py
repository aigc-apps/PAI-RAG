"""Vector Table Mapping Model - Maps (tenant_id, kb_id) to vector table name."""

from datetime import datetime, timezone
import hashlib
import uuid
from pydantic import field_serializer
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, DateTime, UniqueConstraint
from common.system_constants import DEFAULT_TENANT_ID


class VectorTableMappingEntity(SQLModel, table=True):
    """
    Mapping table for (tenant_id, kb_id) -> table_name.
    Used to store the vector table name for each knowledgebase.
    """
    __tablename__ = "pai_vector_table_mapping"
    __table_args__ = (
        UniqueConstraint("tenant_id", "kb_id", name="unique_tenant_kb"),
    )

    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        primary_key=True,
        max_length=64
    )
    tenant_id: str = Field(max_length=64, index=True)
    kb_id: str = Field(max_length=64, index=True)
    table_name: str = Field(max_length=128)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )

    @field_serializer("created_at", "updated_at")
    def serialize_dt(self, dt: datetime, _info):
        # If the datetime is naive, assume it's UTC and add 'Z'
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        # If it's already aware, convert to ISO format
        return dt.isoformat()

def generate_vector_table_name(tenant_id: str, kb_id: str) -> str:
    """
    Generate a vector table name for a given tenant_id and kb_id.

    Args:
        tenant_id: The tenant ID
        kb_id: The knowledgebase ID

    Returns:
        For DEFAULT_TENANT_ID: returns kb_id directly
        For other tenants: returns "kb" + first 16 chars of hash(tenant_id + kb_id)
    """
    if tenant_id == DEFAULT_TENANT_ID:
        return kb_id

    # Generate hash for non-default tenants
    hash_input = f"{tenant_id}_{kb_id}"
    hash_value = hashlib.sha256(hash_input.encode()).hexdigest()
    return f"kb{hash_value[:16]}"
