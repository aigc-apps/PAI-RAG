import uuid
from sqlmodel import Field, SQLModel


class CodeSandboxConfig(SQLModel):
    type: str = Field(default='aliyun-fc')
    aliyun_id: str = Field(default=None)
    interpreter_id: str = Field(default=None)
    enabled: bool = Field(default=False)
    timeout_default: int = Field(default=50)

     # aliyun-fc



class CodeSandboxConfigCreate(CodeSandboxConfig):
    aliyun_id: str = Field(default=None)
    interpreter_id: str = Field(default=None)
    enabled: bool = Field(default=False)


class CodeSandboxConfigRead(CodeSandboxConfigCreate):
    id: str = Field(default=None)


class CodeSandboxConfigEntity(CodeSandboxConfig, table=True):
    __tablename__ = "pai_code_sandbox_config"

    id: str = Field(default_factory=lambda x: str(uuid.uuid4().hex), primary_key=True)
    aliyun_id: str | None = Field(default=None)
    interpreter_id: str | None = Field(default=None)
    enabled: bool | None = Field(default=False)
    timeout_default: int | None = Field(default=50)
