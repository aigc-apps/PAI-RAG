from datetime import datetime, timezone
import re
import uuid
from pydantic import field_validator
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime, Text, UniqueConstraint
from common.knowledgebase.constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SIMILARITY_TOP_K,
    DEFAULT_RERANK_SIMILARITY_TOP_K,
    DEFAULT_VECTOR_WEIGHT,
)
from common.knowledgebase.types import VectorIndexRetrievalType
from typing import Optional
from common.system_constants import DEFAULT_TENANT_ID
from pydantic import field_serializer
from pairag.file.nodeparsers.file_parser import ChunkConfig



class RetrievalConfig(SQLModel):
    retrieval_mode: VectorIndexRetrievalType = Field(
        default=VectorIndexRetrievalType.hybrid
    )
    top_k: int = Field(default=DEFAULT_SIMILARITY_TOP_K)
    similarity_threshold: float = Field(default=0)
    vector_weight: float = Field(default=DEFAULT_VECTOR_WEIGHT)
    enable_rerank: bool = Field(default=False)
    rerank_model: str = Field(default="")
    rerank_provider_name: str = Field(default="openai_like")
    rerank_top_k: Optional[int] = Field(default=DEFAULT_RERANK_SIMILARITY_TOP_K)


class KnowledgebaseCreate(SQLModel):
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)
    name: str = Field(default=None)
    description: str = Field(default=None, sa_column=Column(Text))
    embedding_model: str = Field(default=None)
    embedding_provider_name: str = Field(default="openai_like")
    chunk_config: ChunkConfig | None = Field(default=None)
    retrieval_config: RetrievalConfig | None = Field(default=None)

    @field_validator("chunk_config")
    def validate_chunk_config(cls, v):
        if not v:
            return ChunkConfig()
        if v.chunk_size <= v.chunk_overlap:
            raise ValueError(f"Chunk size `{v.chunk_size}` must be greater than chunk overlap `{v.chunk_overlap}`.")
        return v

    @field_validator("name")
    def validate_name(cls, v):
        if not v:
            raise ValueError("Knowledge base name cannot be empty.")
        if len(v) > 100:
            raise ValueError("Knowledge base name cannot exceed 100 characters.")
        if not re.fullmatch(r"[\w-]+", v):
            raise ValueError("Knowledge base name can only contain letters, numbers, hyphens, and underscores.")
        return v

# table entity
class KbEntity(SQLModel, table=True):
    __tablename__ = "pai_knowledgebase"
    __table_args__ = (UniqueConstraint("tenant_id", "name", name="unique_kb_name"),)

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True, max_length=64)
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID, max_length=64)
    name: str = Field(default=None)
    description: str = Field(default=None, sa_column=Column(Text))

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )

    embedding_model: str = Field(default=DEFAULT_EMBEDDING_MODEL, max_length=255)
    embedding_provider_name: str = Field(default="openai_like")

    chunk_config: dict = Field(
        default=lambda: ChunkConfig(), sa_column=Column("chunk_config", JSON)
    )
    retrieval_config: dict = Field(
        default=lambda: RetrievalConfig(), sa_column=Column("retrieval_config", JSON)
    )


    @field_serializer("created_at", "updated_at")
    def serialize_dt(self, dt: datetime, _info):
        # If the datetime is naive, assume it's UTC and add 'Z'
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        # If it's already aware, convert to ISO format
        return dt.isoformat()
