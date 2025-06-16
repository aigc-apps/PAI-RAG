import uuid
from sqlmodel import Field, SQLModel


# table名为paimcpserver
class McpServer(SQLModel):
    name: str = Field(default=None)
    url: str = Field(default=None)
    type: str = Field(default="sse")
    enabled: bool = Field(default=True)


class McpServerCreate(McpServer):
    auth_token: str | None = Field(default=None)


class McpServerRead(McpServer):
    id: str = Field(default=None)


class McpServerEntity(McpServer, table=True):
    __tablename__ = "pai_mcp_server"

    id: str = Field(default_factory=lambda x: str(uuid.uuid4()), primary_key=True)
    encrypted_auth_token: str | None = Field(default=None)
