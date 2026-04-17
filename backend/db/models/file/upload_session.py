"""Upload session for resumable / chunked file uploads to /v1/files."""
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
import uuid

from sqlmodel import Field, SQLModel
from sqlalchemy import Column, DateTime, Index, JSON, String

from common.system_constants import DEFAULT_TENANT_ID


class UploadSessionStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


def _gen_upload_id() -> str:
    return f"upl-{uuid.uuid4().hex}"


class FileUploadSessionEntity(SQLModel, table=True):
    __tablename__ = "pai_file_upload_session"
    __table_args__ = (
        Index("ix_file_upload_session_tenant_status", "tenant_id", "status"),
        Index("ix_file_upload_session_expires_at", "expires_at"),
    )

    id: str = Field(default_factory=_gen_upload_id, primary_key=True, max_length=64)
    tenant_id: str = Field(
        default=DEFAULT_TENANT_ID, sa_column=Column(String(64), index=True, nullable=False)
    )

    file_name: str = Field(default=None, max_length=255)
    purpose: str = Field(default="chat_attachment", max_length=32)
    expires_in_seconds: Optional[int] = Field(default=None)

    status: str = Field(default=UploadSessionStatus.ACTIVE.value, max_length=32)
    expires_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))

    # List of {part: int, size: int, path: str, md5: str}. JSON keeps the
    # implementation simple — chunk counts in the tens, not thousands.
    parts: List[dict] = Field(default=[], sa_column=Column("parts", JSON))

    # Once complete, the created FileEntity id is stored here for idempotency.
    file_id: Optional[str] = Field(default=None, max_length=64)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
