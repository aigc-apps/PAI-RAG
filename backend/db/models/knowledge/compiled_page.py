"""Layer 2: Agent Compiled Page entity.

Compiled pages are exclusively owned and written by the Knowledge Compiler Agent.
Humans can READ (review) these pages but NEVER edit them directly.
If content is wrong, humans fix the source (wiki or raw doc) and agent recompiles.
"""

from datetime import datetime, timezone
from typing import Optional, List
import uuid

from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime, Text
from pydantic import field_serializer

from common.system_constants import DEFAULT_TENANT_ID


class CompiledPageSource(SQLModel):
    """A single source reference for provenance tracking."""

    source_type: str = Field(description="human_wiki | raw_doc")
    source_id: str = Field(description="ID of the source (wiki page ID or raw doc ID)")
    source_version: Optional[int] = Field(
        default=None, description="Version at compilation time (for wiki pages)"
    )


class CompiledPageCreate(SQLModel):
    """DTO for the agent to create a compiled page."""

    title: str = Field(max_length=255)
    page_type: str = Field(
        default="topic_summary",
        max_length=64,
        description="Enum: topic_summary | entity | claim_index | relationship",
    )
    content: str = Field(default="", sa_column=Column(Text))
    compiled_from: List[dict] = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="Provenance: [{source_type, source_id, source_version}]",
    )
    compiler_model: str = Field(default="", max_length=128)
    confidence: float = Field(default=0.0)
    tags: List[str] = Field(default_factory=list, sa_column=Column(JSON))


class CompiledPageEntity(SQLModel, table=True):
    """Layer 2: Agent-compiled knowledge page.

    Ownership: KNOWLEDGE COMPILER AGENT (exclusive write access).
    Human access: READ-ONLY (review via UI, never direct edit).
    Contract:
      - Every page has full provenance (compiled_from).
      - Every claim in content should cite its source.
      - `stale` flag auto-set when any input source is updated.
      - Humans do NOT edit these — they fix the source and agent recompiles.
    """

    __tablename__ = "knowledge_compiled_page"

    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex, primary_key=True, max_length=64
    )
    tenant_id: str = Field(default=DEFAULT_TENANT_ID, max_length=64)

    title: str = Field(max_length=255)
    page_type: str = Field(default="topic_summary", max_length=64)
    content: str = Field(default="", sa_column=Column(Text))

    # Provenance: which sources produced this page
    compiled_from: list = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="[{source_type, source_id, source_version}]",
    )

    # Compilation metadata
    compiled_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    compiler_model: str = Field(default="", max_length=128)
    confidence: float = Field(default=0.0)

    # Staleness tracking
    stale: bool = Field(default=False)
    stale_reason: Optional[str] = Field(default=None, max_length=512)

    # Search metadata
    tags: list = Field(default_factory=list, sa_column=Column(JSON))

    # Lifecycle
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )

    @field_serializer("created_at", "updated_at", "compiled_at")
    def serialize_dt(self, dt: datetime, _info):
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        return dt.isoformat()
