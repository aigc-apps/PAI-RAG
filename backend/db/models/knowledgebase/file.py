from datetime import datetime, timezone
from typing import List, Optional
import uuid
from pydantic import BaseModel
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime, UniqueConstraint
from common.knowledgebase.types import FileStatus


class MetadataEntry(BaseModel):
    metadata_id: str
    name: str
    value: str | int | float


class MetadataEntryData(BaseModel):
    entries: List[MetadataEntry]


class KbFileEntity(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("kb_id", "message_id", "file_name", name="unique_kb_file"),)

    __tablename__ = "pai_knowledgebase_file"
    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex), primary_key=True)
    kb_id: str = Field(default=None, foreign_key="pai_knowledgebase.id", ondelete="CASCADE")
    # for attachment files, we need to store the message_id, file_content and file_content_length
    message_id: str = Field(default=None)
    file_content: str = Field(default=None)
    file_content_length: int = Field(default=0)

    file_source: Optional[str] = Field(default=None)
    file_name: str = Field(default=None)
    file_path: str = Field(default=None)
    file_extension: str = Field(default=None)
    file_size: int = Field(default=None)
    file_md5: str = Field(default=None)

    file_version: Optional[int] = Field(default=0)

    status: str = Field(default=FileStatus.pending)
    failed_reason: str | None = Field(default=None)
    active: bool = Field(default=True)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None), sa_column=Column(DateTime)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None), sa_column=Column(DateTime)
    )

    file_metadata: dict = Field(default={}, sa_column=Column("file_metadata", JSON))
