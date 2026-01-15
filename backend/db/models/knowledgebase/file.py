from datetime import datetime, timezone
from typing import List, Optional
import uuid
from pydantic import BaseModel, field_serializer
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime, UniqueConstraint, Text, String
from common.knowledgebase.types import FileStatus
from common.system_constants import DEFAULT_TENANT_ID

class MetadataEntry(BaseModel):
    name: str
    value: str | int | float


class MetadataEntryData(BaseModel):
    entries: List[MetadataEntry]


class KbFileEntity(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("kb_id", "message_id", "file_name", "tenant_id", name="unique_kb_file"),)

    __tablename__ = "pai_knowledgebase_file"
    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex), primary_key=True, max_length=64)
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID, sa_column=Column(String(64)), max_length=64)
    kb_id: str = Field(default=None, foreign_key="pai_knowledgebase.id", ondelete="CASCADE", max_length=64)
    # for attachment files, we need to store the message_id, file_content and file_content_length
    message_id: str = Field(default=None, sa_column=Column(String(64)), max_length=64)
    file_content: str = Field(default=None, sa_column=Column(Text))
    file_content_length: int = Field(default=0)

    file_source: Optional[str] = Field(default=None)
    file_name: str = Field(default=None, max_length=255)
    file_path: str = Field(default=None, sa_column=Column(Text))
    file_extension: str = Field(default=None)
    file_size: int = Field(default=None)
    file_md5: str = Field(default=None)

    file_version: Optional[int] = Field(default=0)

    status: str = Field(default=FileStatus.pending)
    failed_reason: str | None = Field(default=None, sa_column=Column(Text))
    active: bool = Field(default=True)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None), sa_column=Column(DateTime)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None), sa_column=Column(DateTime)
    )

    file_metadata: dict = Field(default={}, sa_column=Column("file_metadata", JSON))

    chunk_config: dict | None = Field(
        default=None, sa_column=Column("chunk_config", JSON)
    )

    @field_serializer("created_at", "updated_at")
    def serialize_dt(self, dt: datetime, _info):
        # If the datetime is naive, assume it's UTC and add 'Z'
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        # If it's already aware, convert to ISO format
        return dt.isoformat()

    def get_file_content(self) -> str:
        content = f"📄 文件“{self.file_name}” 的内容如下：\n\n {self.file_content}"
        if self.file_content_length > 1000:
            content += "\n\n[The file content is too long, has been truncated]"

        return content
