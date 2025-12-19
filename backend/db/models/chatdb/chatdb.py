from datetime import datetime, timezone
import uuid
from pydantic import field_serializer
from sqlmodel import Column, DateTime, Field, SQLModel
from common.system_constants import DEFAULT_TENANT_ID
from typing import Optional

class ChatDbConfig(SQLModel):
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)
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

    @field_serializer("created_at", "updated_at")
    def serialize_dt(self, dt: datetime, _info):
        # If the datetime is naive, assume it's UTC and add 'Z'
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        # If it's already aware, convert to ISO format
        return dt.isoformat()
