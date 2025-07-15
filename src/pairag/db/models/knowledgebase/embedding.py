import uuid
from sqlmodel import Field, SQLModel
from pairag.common.knowledgebase.constants import DEFAULT_EMBEDDING_MODEL
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


class EmbeddingModelCreate(EmbeddingModel):
    api_key: str | None = Field(default=None)  # required for openai_like type


class EmbeddingModelRead(EmbeddingModel):
    id: str = Field(default=None)


class EmbeddingModelEntity(EmbeddingModel, table=True):
    __tablename__ = "pai_embedding_model"

    id: str = Field(default_factory=lambda x: uuid.uuid4().hex, primary_key=True)
    encrypted_api_key: str | None = Field(default=None)
