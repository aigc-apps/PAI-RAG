"""File chunks — the unit of retrieval for large file attachments.

Chunks are produced during ``process_file_resource_task`` from the extracted
text. They back ``GET /v1/files/{id}/chunks?query=...`` and the agent's
``search_file_chunks`` tool. Cascading delete happens in
``FileResourceService.hard_delete``.
"""
from datetime import datetime, timezone
import uuid

from sqlmodel import Field, SQLModel
from sqlalchemy import Column, DateTime, Index, JSON, String, Text

from common.system_constants import DEFAULT_TENANT_ID


def _gen_chunk_id() -> str:
    return f"fchk-{uuid.uuid4().hex}"


class FileChunkEntity(SQLModel, table=True):
    __tablename__ = "pai_file_chunk"
    __table_args__ = (
        Index("ix_file_chunk_tenant_file", "tenant_id", "file_id"),
        Index("ix_file_chunk_tenant_file_index", "tenant_id", "file_id", "chunk_index"),
    )

    id: str = Field(default_factory=_gen_chunk_id, primary_key=True, max_length=64)
    tenant_id: str = Field(
        default=DEFAULT_TENANT_ID, sa_column=Column(String(64), index=True, nullable=False)
    )
    file_id: str = Field(default=None, max_length=64)

    chunk_index: int = Field(default=0)
    content: str = Field(default="", sa_column=Column(Text))
    # Character offsets into the *extracted* text (pai_file_text_content.content)
    # — useful for highlighting, not strictly required for retrieval.
    start_offset: int = Field(default=0)
    end_offset: int = Field(default=0)
    token_count: int = Field(default=0)

    chunk_metadata: dict = Field(default={}, sa_column=Column("chunk_metadata", JSON))

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
