from datetime import datetime, timezone
import uuid
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime
from pairag.common.knowledgebase.types import FileStatus

class AttachmentFileEntity(SQLModel, table=True):
    __tablename__ = "pai_attachment_file"
    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex), primary_key=True)
    kb_id: str = Field(default=None)
    message_id: str = Field(default=None)
    file_content: str = Field(default=None)
    file_content_length: int = Field(default=0)

    file_name: str = Field(default=None)
    file_path: str = Field(default=None)
    file_extension: str = Field(default=None)
    file_size: int = Field(default=None)
    file_md5: str = Field(default=None)
    frontend_file_id: str = Field(default=None)

    status: str = Field(default=FileStatus.pending)
    failed_reason: str | None = Field(default=None)
    active: bool = Field(default=True)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    update_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )

    file_metadata: dict = Field(default={}, sa_column=Column("file_metadata", JSON))
