import uuid
from sqlmodel import Field, SQLModel
from common.system_constants import DEFAULT_TENANT_ID
from typing import Optional


class McpServer(SQLModel):
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)
    name: str = Field(default=None)
    url: str = Field(default=None)
    type: str = Field(default="sse")
    enabled: bool = Field(default=True)
    need_token: bool = Field(default=False)


class McpServerCreate(McpServer):
    auth_token: Optional[str] = Field(default=None)


class McpServerRead(McpServer):
    id: str = Field(default=None)


# table名为pai_mcp_server
class McpServerEntity(McpServer, table=True):
    __tablename__ = "pai_mcp_server"

    id: str = Field(default_factory=lambda x: str(uuid.uuid4().hex), primary_key=True)
    encrypted_auth_token: Optional[str] = Field(default=None)
