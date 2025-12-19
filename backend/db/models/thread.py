import uuid
from pydantic import field_serializer
from sqlmodel import Field, SQLModel
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Text
from common.system_constants import DEFAULT_TENANT_ID
from typing import Optional


class ThreadCreate(SQLModel):
    user_id: str = Field(default="PAI-RAG Assistant")
    title: str = Field(default=None, sa_column=Column(Text))

class ThreadRead(ThreadCreate):
    id: str = Field(default=None, primary_key=True)
    archived: bool = Field(default=False)


class ThreadEntity(SQLModel, table=True):
    __tablename__ = "pai_thread"

    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex), primary_key=True)
    user_id: str = Field(default="PAI-RAG Assistant", nullable=False)
    title: str = Field(default=None, sa_column=Column(Text))
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
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
