from datetime import datetime, timezone
import uuid
from sqlmodel import Column, DateTime, Field, SQLModel


class ChatDbConfig(SQLModel):
    dialect: str # postgresql/mysql
    db_name: str = None
    username: str = None
    port: int = None
    host: str = None

    model_id: str = None


class ChatDbCreate(ChatDbConfig):
    password: str = None


class ChatDbConfigEntity(ChatDbConfig, table=True):
    __tablename__ = "pai_chatdb_config"

    id: str = Field(default_factory=lambda x: str(uuid.uuid4().hex), primary_key=True)
    encrypted_password: str = Field(default=None)

    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
