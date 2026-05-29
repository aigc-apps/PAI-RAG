import uuid
from sqlmodel import Field, SQLModel
from typing import Optional
from enum import Enum
from common.system_constants import DEFAULT_TENANT_ID
from sqlalchemy import UniqueConstraint

# 支持openai_like、dashscope与multimodal_dashscope三种模式
class RerankerType(str, Enum):
    OPENAI_LIKE = "openai_like"
    DASHSCOPE = "dashscope"
    MULTIMODAL_DASHSCOPE = "multimodal_dashscope"


class RerankerModel(SQLModel):
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID, max_length=64)
    model_name: str = Field(default=None)
    base_url: str = Field(default=None)
    model_id: str = Field(default=None)
    type: Optional[str] = Field(default=RerankerType.OPENAI_LIKE)
    is_multimodal: Optional[bool] = Field(default=False) # 是否为多模态 rerank，true 时会把节点 images_info 拼入 documents
    provider_name: Optional[str] = Field(default=None)


class RerankerModelCreate(RerankerModel):
    api_key: str | None = Field(default=None)  # required for openai_like type


class RerankerModelRead(RerankerModel):
    id: str = Field(default=None)


class RerankerModelEntity(RerankerModel, table=True):
    __tablename__ = "pai_reranker_model"
    __table_args__ = (UniqueConstraint("tenant_id", "provider_name", "model_id", name="unique_reranker_model"),)

    id: str = Field(default_factory=lambda x: uuid.uuid4().hex, primary_key=True, max_length=64)
    encrypted_api_key: str | None = Field(default=None)
