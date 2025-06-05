import uuid
from sqlmodel import Field, SQLModel


class LlmModel(SQLModel):
    base_url: str = Field(default=None)
    model: str = Field(default=None)
    context_window: int = Field(default=8000)
    temperature: int = Field(default=0.1)


class LlmModelCreate(LlmModel):
    api_key: str


class LlmModelRead(LlmModel):
    id: str = Field(default=None)


# table entity
class LlmModelEntity(LlmModel, table=True):
    __tablename__ = "pai_llm_model"

    id: str = Field(default_factory=uuid.uuid4, primary_key=True)
    encrypted_api_key: str = Field(default=None)


# table名为paimcpserver
class McpServer(SQLModel, table=True):
    name: str = Field(default=None)
    url: str = Field(default=None)
    type: str = Field(default="sse")
    active: bool = Field(default=True)


class McpServerCreate(McpServer):
    auth_token: str


class McpServerRead(McpServer):
    id: str = Field(default=None)


class McpServerEntity(McpServer, table=True):
    __tablename__ = "pai_mcp_server"

    id: str = Field(default_factory=uuid.uuid4, primary_key=True)
    encrypted_auth_token: str = Field(default=None)


class TraceConfig(SQLModel, table=True):
    endpoint: str = Field(default=None)
    token: str = Field(default=None)
    service_name: str = Field(default=None)
    active: bool = Field(default=False)


class TraceConfigEntity(TraceConfig, table=True):
    __tablename__ = "pai_trace_config"
