import uuid
from pydantic import model_validator
from sqlmodel import Field, SQLModel, Column, Boolean
from common.knowledgebase.constants import DEFAULT_EMBEDDING_MODEL
from enum import Enum


# 支持openai_like和local两种模式
class EmbeddingType(str, Enum):
    OPENAI_LIKE = "openai_like"
    LOCAL = "local"


class EmbeddingModel(SQLModel):
    model_name: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    dimension: int | None = Field(default=None)
    endpoint: str | None = Field(default=None)
    type: EmbeddingType = Field(default=EmbeddingType.LOCAL)
    embed_batch_size: int = Field(default=10)
    model_id: str = Field(default=None, unique=True)
    is_ready: bool = Field(
        sa_column=Column(Boolean, default=False),
    ) # 是否已经加载完成，用于本地模型下载
    is_default: bool = Field(
        sa_column=Column(Boolean, default=False),
    )

    @model_validator(mode='after')
    def set_is_ready(self) -> 'EmbeddingModel':
        if self.embed_batch_size <= 0:
            self.embed_batch_size = 10
        if self.is_ready is None:
            self.is_ready = self.type == EmbeddingType.OPENAI_LIKE
        return self


class EmbeddingModelCreate(EmbeddingModel):
    api_key: str | None = Field(default=None)  # required for openai_like type


class EmbeddingModelRead(EmbeddingModel):
    id: str = Field(default=None)


class EmbeddingModelEntity(EmbeddingModel, table=True):
    __tablename__ = "pai_embedding_model"

    id: str = Field(default_factory=lambda x: uuid.uuid4().hex, primary_key=True)
    encrypted_api_key: str | None = Field(default=None)
