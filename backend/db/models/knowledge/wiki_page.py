"""Layer 0: Human Wiki Page entity.

Human Wiki is the source of truth — exclusively owned and edited by humans.
The agent can READ these pages but NEVER modify them directly.
"""

from datetime import datetime, timezone
from typing import Optional, List
import uuid

from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime, Text, UniqueConstraint
from pydantic import field_serializer

from common.system_constants import DEFAULT_TENANT_ID


class WikiPageCreate(SQLModel):
    """DTO for creating a new wiki page."""

    title: str = Field(max_length=255)
    category: str = Field(
        default="domain",
        max_length=64,
        description="Enum: business-rules | domain | process | faq | architecture",
    )
    content: str = Field(default="", sa_column=Column(Text))
    tags: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    source_doc_ids: List[str] = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="Optional: raw doc IDs this wiki page relates to",
    )


class WikiPageUpdate(SQLModel):
    """DTO for updating an existing wiki page."""

    title: Optional[str] = None
    category: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    source_doc_ids: Optional[List[str]] = None


class WikiPageEntity(SQLModel, table=True):
    """Layer 0: Human-authored wiki page.

    Ownership: HUMANS ONLY (exclusive write access).
    Agent access: READ-ONLY.
    Contract:
      - Agent NEVER modifies these records.
      - Agent CAN create Suggestion records referencing these pages.
      - Versioned via `version` field (increment on every edit).
      - Soft-delete via `archived` flag.
    """

    __tablename__ = "knowledge_wiki_page"
    __table_args__ = (
        UniqueConstraint("tenant_id", "title", name="uq_wiki_page_tenant_title"),
    )

    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex, primary_key=True, max_length=64
    )
    tenant_id: str = Field(default=DEFAULT_TENANT_ID, max_length=64)

    title: str = Field(max_length=255)
    category: str = Field(default="domain", max_length=64)
    content: str = Field(default="", sa_column=Column(Text))
    tags: list = Field(default_factory=list, sa_column=Column(JSON))
    source_doc_ids: list = Field(default_factory=list, sa_column=Column(JSON))

    # Ownership tracking
    last_edited_by: str = Field(default="system", max_length=128)
    version: int = Field(default=1)

    # Lifecycle
    archived: bool = Field(default=False)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )

    @field_serializer("created_at", "updated_at")
    def serialize_dt(self, dt: datetime, _info):
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        return dt.isoformat()
