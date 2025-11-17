import uuid
from sqlmodel import Field, SQLModel
from typing import Optional
from enum import Enum


# 支持openai_like和dashscope两种模式
class RerankerType(str, Enum):
    OPENAI_LIKE = "openai_like"
    DASHSCOPE = "dashscope"

class RerankerModel(SQLModel):
    model_name: str = Field(default=None)
    base_url: str = Field(default=None)
    model_id: str = Field(default=None, unique=True)
    type: Optional[str] = Field(default=RerankerType.OPENAI_LIKE)


class RerankerModelCreate(RerankerModel):
    api_key: str | None = Field(default=None)  # required for openai_like type


class RerankerModelRead(RerankerModel):
    id: str = Field(default=None)


class RerankerModelEntity(RerankerModel, table=True):
    __tablename__ = "pai_reranker_model"

    id: str = Field(default_factory=lambda x: uuid.uuid4().hex, primary_key=True)
    encrypted_api_key: str | None = Field(default=None)
