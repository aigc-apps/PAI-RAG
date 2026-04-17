"""Independent File resource — decoupled from knowledgebase.

This is Phase 1 of the /v1/files refactor. The goal is to provide a clean
OpenAI/Anthropic-style Files API that replaces the legacy
/v1/config/attachments flow (which abuses a default_attachments KB).
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid

from pydantic import field_serializer
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, DateTime, Index, JSON, String, Text, UniqueConstraint

from common.knowledgebase.types import FileStatus
from common.system_constants import DEFAULT_TENANT_ID


class FilePurpose(str, Enum):
    CHAT_ATTACHMENT = "chat_attachment"
    KB_INGESTION = "kb_ingestion"
    VISION = "vision"
    AVATAR = "avatar"


def _gen_file_id() -> str:
    return f"file-{uuid.uuid4().hex}"


class FileEntity(SQLModel, table=True):
    __tablename__ = "pai_file"
    __table_args__ = (
        # Dedup: same bytes + same purpose within a tenant reuses the row.
        # Different purpose is allowed to coexist (different lifecycle/ACL).
        UniqueConstraint("tenant_id", "file_md5", "purpose", name="uq_file_tenant_md5_purpose"),
        # Legacy compat: Phase 4 adapter keys the old client-supplied file_id here.
        UniqueConstraint("tenant_id", "alias_id", name="uq_file_tenant_alias"),
        Index("ix_file_tenant_purpose_created", "tenant_id", "purpose", "created_at"),
        Index("ix_file_expires_at", "expires_at"),
    )

    id: str = Field(default_factory=_gen_file_id, primary_key=True, max_length=64)
    tenant_id: str = Field(default=DEFAULT_TENANT_ID, sa_column=Column(String(64), index=True, nullable=False))

    purpose: str = Field(default=FilePurpose.CHAT_ATTACHMENT.value, max_length=32)
    alias_id: Optional[str] = Field(default=None, max_length=64)

    file_name: str = Field(default=None, max_length=255)
    file_extension: Optional[str] = Field(default=None, max_length=32)
    file_size: int = Field(default=0)
    file_md5: Optional[str] = Field(default=None, max_length=64)
    file_path: Optional[str] = Field(default=None, sa_column=Column(Text))
    mime_type: Optional[str] = Field(default=None, max_length=128)

    status: str = Field(default=FileStatus.pending.value, max_length=32)
    failed_reason: Optional[str] = Field(default=None, sa_column=Column(Text))

    ref_count: int = Field(default=0, index=True)
    expires_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime, nullable=True))

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )

    file_metadata: dict = Field(default={}, sa_column=Column("file_metadata", JSON))

    @field_serializer("created_at", "updated_at", "expires_at")
    def _serialize_dt(self, dt: Optional[datetime], _info):
        if dt is None:
            return None
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        return dt.isoformat()


class FileTextContentEntity(SQLModel, table=True):
    """Extracted text for attachments / indexable files.

    Kept in a separate table so `SELECT * FROM pai_file` stays cheap for list views.
    """
    __tablename__ = "pai_file_text_content"

    file_id: str = Field(primary_key=True, max_length=64)
    tenant_id: str = Field(default=DEFAULT_TENANT_ID, sa_column=Column(String(64), index=True, nullable=False))
    content: str = Field(default="", sa_column=Column(Text))
    content_length: int = Field(default=0)
    extractor_version: Optional[str] = Field(default=None, max_length=32)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
