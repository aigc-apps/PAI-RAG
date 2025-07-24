import uuid
from sqlmodel import Field, SQLModel

class RerankerModel(SQLModel):
    model_name: str = Field(default=None)
    base_url: str = Field(default=None)
    model_id: str = Field(default=None, unique=True)


class RerankerModelCreate(RerankerModel):
    api_key: str | None = Field(default=None)  # required for openai_like type


class RerankerModelRead(RerankerModel):
    id: str = Field(default=None)


class RerankerModelEntity(RerankerModel, table=True):
    __tablename__ = "pai_reranker_model"

    id: str = Field(default_factory=lambda x: uuid.uuid4().hex, primary_key=True)
    encrypted_api_key: str | None = Field(default=None)
