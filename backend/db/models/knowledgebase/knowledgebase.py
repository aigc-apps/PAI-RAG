from datetime import datetime, timezone
import re
import uuid
from pydantic import ConfigDict, field_validator
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime, Text, UniqueConstraint
from common.knowledgebase.constants import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SENTENCE_SEPARATOR,
    DEFAULT_PARSER_TYPE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_SIMILARITY_TOP_K,
    DEFAULT_RERANK_SIMILARITY_TOP_K,
)
from common.knowledgebase.types import VectorIndexRetrievalType
from typing import Optional
from common.system_constants import DEFAULT_TENANT_ID


class ChunkConfig(SQLModel):
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE)
    chunk_overlap: int = Field(default=DEFAULT_CHUNK_OVERLAP)
    parser_type: str = Field(default=DEFAULT_PARSER_TYPE)
    separator: str = Field(default=DEFAULT_SENTENCE_SEPARATOR)
    image_caption_model: Optional[str] = Field(default=None)
    image_caption_provider_name: str = Field(default="openai_like")


class RetrievalConfig(SQLModel):
    retrieval_mode: VectorIndexRetrievalType = Field(
        default=VectorIndexRetrievalType.hybrid
    )
    top_k: int = Field(default=DEFAULT_SIMILARITY_TOP_K)
    similarity_threshold: float = Field(default=DEFAULT_SIMILARITY_THRESHOLD)
    vector_weight: float = Field(default=0.5)
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

    @field_validator("name")
    def validate_name(cls, v):
        if not v:
            raise ValueError("知识库名称不能为空。")
        if len(v) > 100:
            raise ValueError("知识库名称不能超过 100 个字符。")
        if not re.fullmatch(r"[\w-]+", v):
            raise ValueError("知识库名称只能包含字母、数字和下划线。")
        return v

    # Pydantic V2 配置
    model_config = ConfigDict(json_encoders={
        datetime: lambda v: v.isoformat()
    })
