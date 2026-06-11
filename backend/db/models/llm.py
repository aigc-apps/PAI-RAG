import uuid
from sqlmodel import Field, SQLModel
from common.system_constants import DEFAULT_TENANT_ID
from typing import Optional
from sqlalchemy import UniqueConstraint

class LlmModel(SQLModel):
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID, max_length=64)
    base_url: str = Field(default=None)

    model: str = Field(default=None) # deprecated, use model_name instead

    model_name: Optional[str] = Field(default=None)
    context_window: int = Field(default=110000)
    temperature: float = Field(default=0.1)
    model_id: str = Field(default=None, max_length=64)
    enabled: bool = Field(default=True)
    vision_support: bool = Field(default=False)
    max_tokens: int = Field(default=8000)
    enable_thinking: bool = Field(default=False, description="Whether the LLM supports thinking mode.")
    provider_name: Optional[str] = Field(default=None)
    source: str = Field(default=None)

class LlmModelCreate(LlmModel):
    api_key: str = Field(default=None)


class LlmModelRead(LlmModel):
    id: str = Field(default=None)


# table entity
class LlmModelEntity(LlmModelRead, table=True):
    __tablename__ = "pai_llm_model"
    __table_args__ = (UniqueConstraint("tenant_id", "provider_name", "model_id", name="unique_llm_model"),)

    id: str = Field(default_factory=lambda x: str(uuid.uuid4().hex), primary_key=True, max_length=64)
    encrypted_api_key: str = Field(default=None)
