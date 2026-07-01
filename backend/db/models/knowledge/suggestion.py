"""Suggestions Queue: Agent → Human feedback channel.

The ONLY upward flow in the knowledge layer. Agent proposes, human disposes.
Agent can suggest edits to wiki pages, but NEVER directly modifies them.

Trigger conditions:
  - Contradiction detected between wiki and raw docs
  - New information in raw docs not reflected in wiki
  - Coverage gap (user queries about topics wiki doesn't cover)
  - Quality issue (wiki page is ambiguous, outdated language, etc.)
"""

from datetime import datetime, timezone
from typing import Optional, List
import uuid

from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime, Text, Index
from pydantic import field_serializer

from common.system_constants import DEFAULT_TENANT_ID


class SuggestionCreate(SQLModel):
    """DTO for agent to create a new suggestion."""

    target_wiki_page_id: str = Field(
        max_length=64, description="Wiki page this suggestion targets"
    )
    issue_type: str = Field(
        max_length=64,
        description="Enum: contradiction | coverage_gap | quality | new_info",
    )
    description: str = Field(
        default="", sa_column=Column(Text), description="What the agent detected"
    )
    evidence: List[dict] = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="[{source_type, source_id, quote, location}]",
    )
    suggested_action: str = Field(
        default="",
        sa_column=Column(Text),
        description="What the agent suggests the human should do",
    )
    confidence: float = Field(default=0.0)


class SuggestionEntity(SQLModel, table=True):
    """Suggestions queue: agent-to-human feedback.

    Ownership: AGENT writes suggestions. HUMANS read + resolve.
    Contract:
      - Agent CANNOT auto-approve its own suggestions.
      - Agent CANNOT modify wiki content directly.
      - Agent CANNOT delete wiki pages.
      - Agent CANNOT override human rejections.
      - Status flow: pending → accepted | rejected | deferred
    """

    __tablename__ = "knowledge_suggestion"
    __table_args__ = (
        Index("ix_suggestion_target_page", "target_wiki_page_id"),
        Index("ix_suggestion_status", "status"),
        Index("ix_suggestion_tenant_status", "tenant_id", "status"),
    )

    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex, primary_key=True, max_length=64
    )
    tenant_id: str = Field(default=DEFAULT_TENANT_ID, max_length=64)

    # Target: which wiki page does this suggestion reference
    target_wiki_page_id: str = Field(max_length=64)

    # Classification
    issue_type: str = Field(
        default="contradiction",
        max_length=64,
        description="contradiction | coverage_gap | quality | new_info",
    )

    # Content
    description: str = Field(default="", sa_column=Column(Text))
    evidence: list = Field(
        default_factory=list,
        sa_column=Column(JSON),
        description="[{source_type, source_id, quote, location}]",
    )
    suggested_action: str = Field(default="", sa_column=Column(Text))
    confidence: float = Field(default=0.0)

    # Resolution workflow
    status: str = Field(
        default="pending",
        max_length=32,
        description="pending | accepted | rejected | deferred",
    )
    resolved_by: Optional[str] = Field(default=None, max_length=128)
    resolved_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    resolution_note: Optional[str] = Field(default=None, max_length=512)

    # Lifecycle
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        sa_column=Column(DateTime),
    )

    @field_serializer("created_at", "resolved_at")
    def serialize_dt(self, dt: Optional[datetime], _info):
        if dt is None:
            return None
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        return dt.isoformat()
