import uuid
from sqlmodel import JSON, Column, Field, SQLModel
from common.system_constants import DEFAULT_TENANT_ID
from typing import Optional


class VectorDbConfig(SQLModel, table=True):
    __tablename__ = "pai_vectordb_config"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)

    type: str = Field(default="local")
    config: dict = Field(default_factory=dict, sa_column=Column(JSON))
