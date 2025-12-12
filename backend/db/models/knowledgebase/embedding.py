from typing import Optional
import uuid
from pydantic import model_validator
from sqlmodel import Field, SQLModel
from common.knowledgebase.constants import DEFAULT_EMBEDDING_MODEL
from enum import Enum
from sqlalchemy import UniqueConstraint
from common.system_constants import DEFAULT_TENANT_ID

# 支持openai_like和local两种模式
class EmbeddingType(str, Enum):
    OPENAI_LIKE = "openai_like"
    LOCAL = "local"


class EmbeddingModel(SQLModel):
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID, max_length=64)
    model_name: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    dimension: int | None = Field(default=None)
    endpoint: str | None = Field(default=None)
    type: EmbeddingType = Field(default=EmbeddingType.LOCAL)
    embed_batch_size: int = Field(default=10)
    model_id: str = Field(default=None, max_length=64)
    is_ready: Optional[bool] = Field(default=False) # 是否已经加载完成，用于本地模型下载
    is_default: Optional[bool] = Field(default=False)
    provider_name: Optional[str] = Field(default=None)


class EmbeddingModelCreate(SQLModel):
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID, max_length=64)
    api_key: str | None = Field(default=None)  # required for openai_like type
    model_name: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    dimension: int | None = Field(default=None)
    endpoint: str | None = Field(default=None)
    type: EmbeddingType = Field(default=EmbeddingType.LOCAL)
    embed_batch_size: int = Field(default=10)
    model_id: str = Field(default=None, max_length=64)
    is_ready: Optional[bool] = False
    is_default: Optional[bool] = False
    provider_name: Optional[str] = Field(default=None)

    @model_validator(mode='after')
    def set_is_ready(self) -> 'EmbeddingModelCreate':
        if self.embed_batch_size <= 0:
            self.embed_batch_size = 10

        if self.dimension is not None and self.dimension < 0:
            self.dimension = None

        return self


class EmbeddingModelRead(EmbeddingModel):
    id: str = Field(default=None)


class EmbeddingModelEntity(EmbeddingModel, table=True):
    __tablename__ = "pai_embedding_model"
    __table_args__ = (UniqueConstraint("tenant_id", "provider_name", "model_id", name="unique_embedding_model"),)

    id: str = Field(default_factory=lambda x: uuid.uuid4().hex, primary_key=True)
    encrypted_api_key: str | None = Field(default=None)
