import uuid
from sqlmodel import JSON, Column, Field, SQLModel

class VectorDbConfig(SQLModel, table=True):
    __tablename__ = "pai_vectordb_config"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)

    type: str = Field(default="local")
    config: dict = Field(default_factory=dict, sa_column=Column(JSON))
