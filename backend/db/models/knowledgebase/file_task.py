from datetime import datetime, timezone
from typing import Optional
import uuid
from pydantic import field_serializer
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, DateTime, UniqueConstraint, Text
from common.knowledgebase.types import FileStatus
from common.system_constants import DEFAULT_TENANT_ID


class KbFileTaskEntity(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("kb_id", "file_id", "file_part", "tenant_id", name="unique_kb_file_task"),)
    __tablename__ = "pai_knowledgebase_file_task"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True, max_length=64)
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID, max_length=64)
    kb_id: str = Field(default=None, foreign_key="pai_knowledgebase.id", ondelete="CASCADE", max_length=64)
    file_id: str = Field(default=None, foreign_key="pai_knowledgebase_file.id", ondelete="CASCADE", max_length=64)
    file_part: int = Field(default=0) # file part index, if file is split into multiple parts
    file_path: str = Field(default=None, sa_column=Column(Text), max_length=255)
    file_version: Optional[int] = Field(default=0)

    status: str = Field(default=FileStatus.pending)
    failed_reason: str | None = Field(default=None, sa_column=Column(Text))

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None), sa_column=Column(DateTime)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None), sa_column=Column(DateTime)
    )


    @field_serializer("created_at", "updated_at")
    def serialize_dt(self, dt: datetime, _info):
        # If the datetime is naive, assume it's UTC and add 'Z'
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        # If it's already aware, convert to ISO format
        return dt.isoformat()
