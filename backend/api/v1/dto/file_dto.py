"""Pydantic DTOs for the /v1/files API."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class FileRead(BaseModel):
    id: str
    tenant_id: str
    purpose: str
    file_name: Optional[str] = None
    file_extension: Optional[str] = None
    file_size: int = 0
    file_md5: Optional[str] = None
    mime_type: Optional[str] = None
    status: str
    failed_reason: Optional[str] = None
    ref_count: int = 0
    expires_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    file_metadata: Optional[dict] = None


class FileTextRead(BaseModel):
    file_id: str
    content: str
    offset: int = 0
    limit: int
    total_length: int
    has_more: bool
    # True if the extractor itself hit its size cap — the stored blob is a
    # prefix of the file's actual extracted text. Client can't paginate past
    # total_length; re-extraction is the escape hatch.
    truncated_at_extract: bool = False
    extractor_version: Optional[str] = None


class FileUrlRead(BaseModel):
    file_id: str
    url: str


class FileChunkHit(BaseModel):
    chunk_id: str
    chunk_index: int
    content: str
    start_offset: int
    end_offset: int
    score: float


class FileChunkSearchResult(BaseModel):
    file_id: str
    query: str
    total_chunks: int
    hits: list[FileChunkHit]
