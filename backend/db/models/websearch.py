import uuid
from sqlmodel import Field, SQLModel
from common.system_constants import DEFAULT_TENANT_ID
from typing import Optional

class WebSearchConfig(SQLModel):
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)
    type: str = Field(default=None) # tavily, aliyun
    search_count: int = Field(default=10)
    endpoint: str | None = Field(default=None)


class WebSearchConfigRead(WebSearchConfig):
    id: Optional[str] = Field(default=None)
    is_aliyun_empty: bool = Field(default=False)
    is_tavily_empty: bool = Field(default=False)


class WebSearchConfigCreate(WebSearchConfig):
    access_key_id: str = Field(default=None)
    access_key_secret: str = Field(default=None)
    tavily_api_key: str = Field(default=None)


class WebSearchConfigEntity(WebSearchConfig, table=True):
    __tablename__ = "pai_websearch_config"

    id: str = Field(default_factory=lambda x: str(uuid.uuid4().hex), primary_key=True)
    encrypted_access_key_id: Optional[str] = Field(default=None)
    encrypted_access_key_secret: Optional[str] = Field(default=None)
    encrypted_tavily_api_key: Optional[str] = Field(default=None)
