import uuid
from sqlmodel import Field, SQLModel


class LlmModel(SQLModel):
    base_url: str = Field(default=None)
    model: str = Field(default=None)
    context_window: int = Field(default=8000)
    temperature: float = Field(default=0.1)
    model_id: str = Field(default=None, unique=True)
    enabled: bool = Field(default=True)
    vision_support: bool = Field(default=False)


class LlmModelCreate(LlmModel):
    api_key: str = Field(default=None)


class LlmModelRead(LlmModel):
    id: str = Field(default=None)
    source: str = Field(default=None)


# table entity
class LlmModelEntity(LlmModelRead, table=True):
    __tablename__ = "pai_llm_model"

    id: str = Field(default_factory=lambda x: str(uuid.uuid4().hex), primary_key=True)
    encrypted_api_key: str = Field(default=None)
