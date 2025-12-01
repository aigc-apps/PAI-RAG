from datetime import datetime, timezone
from typing import Optional
import uuid
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, DateTime, UniqueConstraint, Text
from common.knowledgebase.types import FileStatus


class KbFileTaskEntity(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("kb_id", "file_id", "file_part", name="unique_kb_file_task"),)
    __tablename__ = "pai_knowledgebase_file_task"

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True)
    kb_id: str = Field(default=None, foreign_key="pai_knowledgebase.id", ondelete="CASCADE")
    file_id: str = Field(default=None, foreign_key="pai_knowledgebase_file.id", ondelete="CASCADE")
    file_part: int = Field(default=0) # file part index, if file is split into multiple parts
    file_path: str = Field(default=None, sa_column=Column(Text))
    file_version: Optional[int] = Field(default=0)

    status: str = Field(default=FileStatus.pending)
    failed_reason: str | None = Field(default=None, sa_column=Column(Text))

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None), sa_column=Column(DateTime)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None), sa_column=Column(DateTime)
    )
