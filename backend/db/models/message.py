import uuid
from pydantic import field_serializer
from sqlmodel import Field, SQLModel
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, JSON
from typing import List, Optional
from common.system_constants import DEFAULT_TENANT_ID


class MessageCreate(SQLModel):
    local_id: Optional[str] = Field(default=None)
    thread_id: str = Field(default=None)
    role: str = Field(default=None)
    content: List[dict] = Field(default=[], sa_column=Column("content", JSON))
    attachments: List[dict] = Field(default=[], sa_column=Column("attachments", JSON))
    token_usage: Optional[dict] = Field(default=None, sa_column=Column("token_usage", JSON))
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )


class MessageRead(MessageCreate):
    id: str = Field(default=None, primary_key=True)
    token_usage: Optional[dict] = Field(default=None)


class MessageEntity(SQLModel, table=True):
    __tablename__ = "pai_message"

    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex), primary_key=True)
    thread_id: str = Field(
        default=None, foreign_key="pai_thread.id", ondelete="CASCADE", nullable=False
    )
    local_id: Optional[str] = Field(default=None) # 记录当前message的local_id信息，避免重新生成时的重复
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)

    role: str = Field(default=None)  # e.g., "user", "assistant", "system"
    content: List[dict] = Field(default=[], sa_column=Column("content", JSON))
    attachments: List[dict] = Field(default=[], sa_column=Column("attachments", JSON))
    token_usage: Optional[dict] = Field(default=None, sa_column=Column("token_usage", JSON))

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )

    @field_serializer("created_at")
    def serialize_dt(self, dt: datetime, _info):
        # If the datetime is naive, assume it's UTC and add 'Z'
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        # If it's already aware, convert to ISO format
        return dt.isoformat()
