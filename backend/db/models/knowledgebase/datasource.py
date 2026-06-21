from datetime import datetime, timezone
import re
import uuid
from typing import Optional

from pydantic import field_serializer, field_validator
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON, DateTime, Text, Index, UniqueConstraint, String

from common.knowledgebase.types import (
    DataSourceType,
    DataSourceStatus,
    DataSourceDocStatus,
    SyncRunStatus,
    SyncTrigger,
)
from common.system_constants import DEFAULT_TENANT_ID


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# API payloads
# ---------------------------------------------------------------------------
class DataSourceCreate(SQLModel):
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID)
    name: str = Field(default=None)
    datasource_key: str = Field(default=None)
    source_type: DataSourceType = Field(default=None)
    source_config: dict = Field(default_factory=dict)
    sync_schedule: Optional[str] = Field(default=None)
    enabled: bool = Field(default=True)

    @field_validator("name")
    def validate_name(cls, v):
        if not v:
            raise ValueError("Data source name cannot be empty.")
        if len(v) > 100:
            raise ValueError("Data source name cannot exceed 100 characters.")
        return v

    @field_validator("datasource_key")
    def validate_key(cls, v):
        if not v:
            raise ValueError("datasource_key cannot be empty.")
        if len(v) > 64:
            raise ValueError("datasource_key cannot exceed 64 characters.")
        if not re.fullmatch(r"[\w-]+", v):
            raise ValueError(
                "datasource_key can only contain letters, numbers, hyphens, and underscores."
            )
        return v


class DataSourceUpdate(SQLModel):
    name: Optional[str] = Field(default=None)
    source_config: Optional[dict] = Field(default=None)
    sync_schedule: Optional[str] = Field(default=None)
    enabled: Optional[bool] = Field(default=None)


# ---------------------------------------------------------------------------
# 1.1 DataSourceEntity — config + aggregate state
# ---------------------------------------------------------------------------
class DataSourceEntity(SQLModel, table=True):
    __tablename__ = "pai_datasource"
    __table_args__ = (
        UniqueConstraint("kb_id", "datasource_key", "tenant_id", name="unique_datasource_key"),
    )

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True, max_length=64)
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID, max_length=64)
    kb_id: str = Field(default=None, foreign_key="pai_knowledgebase.id", ondelete="CASCADE", max_length=64)

    name: str = Field(default=None, max_length=100)
    # slug, used as doc_id prefix ("{datasource_key}/{path}"); unique within a KB
    datasource_key: str = Field(default=None, max_length=64)
    source_type: str = Field(default=None, max_length=32)
    source_config: dict = Field(default_factory=dict, sa_column=Column("source_config", JSON))

    # scheduling
    sync_schedule: Optional[str] = Field(default=None, max_length=128)  # cron / interval; None = manual only
    next_sync_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    enabled: bool = Field(default=True)

    # aggregate sync state (two-phase, see plan §1.4)
    status: str = Field(default=DataSourceStatus.idle)
    last_sync_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    last_sync_finished_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    last_sync_duration_ms: Optional[int] = Field(default=None)
    last_error: Optional[str] = Field(default=None, sa_column=Column(Text))

    # counts
    doc_count: int = Field(default=0)
    last_sync_report: dict = Field(default_factory=dict, sa_column=Column("last_sync_report", JSON))

    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime))
    updated_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime))

    @field_serializer("created_at", "updated_at", "next_sync_at", "last_sync_at", "last_sync_finished_at")
    def serialize_dt(self, dt: Optional[datetime], _info):
        if dt is None:
            return None
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        return dt.isoformat()


# ---------------------------------------------------------------------------
# 1.2 DataSourceDocumentEntity — per-document manifest (the "file list")
# ---------------------------------------------------------------------------
class DataSourceDocumentEntity(SQLModel, table=True):
    __tablename__ = "pai_datasource_document"
    __table_args__ = (
        UniqueConstraint("datasource_id", "doc_id", name="unique_datasource_doc"),
        Index("idx_datasource_doc_status", "datasource_id", "doc_status"),
    )

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True, max_length=64)
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID, max_length=64)
    datasource_id: str = Field(default=None, foreign_key="pai_datasource.id", ondelete="CASCADE", max_length=64)
    kb_id: str = Field(default=None, max_length=64)

    # addressing — bounded VARCHAR so the (datasource_id, doc_id) unique index is
    # valid on MySQL (TEXT columns can't be uniquely indexed without a prefix length)
    doc_id: str = Field(default=None, sa_column=Column(String(512)))  # "{datasource_key}/{path}", unique within source
    # FK to the ingested KB file; backfilled after upsert (nullable until ingested)
    file_id: Optional[str] = Field(default=None, max_length=64)
    path: str = Field(default=None, sa_column=Column(Text))
    source_url: Optional[str] = Field(default=None, sa_column=Column(Text))  # human page, for citation
    fetch_url: Optional[str] = Field(default=None, sa_column=Column(Text))  # actual resource fetched (e.g. .md)

    # metadata
    title: Optional[str] = Field(default=None, sa_column=Column(Text))
    section: Optional[str] = Field(default=None, max_length=255)
    product: Optional[str] = Field(default=None, max_length=255)
    summary: Optional[str] = Field(default=None, sa_column=Column(Text))
    lang: Optional[str] = Field(default=None, max_length=16)
    content_hash: Optional[str] = Field(default=None, max_length=64)
    byte_size: Optional[int] = Field(default=None)
    # adapter-specific extras (llms.txt native summary, sphinx toctree caption, ...); manifest-only
    source_meta: dict = Field(default_factory=dict, sa_column=Column("source_meta", JSON))

    # per-document sync state (parse status derived on read via file_id -> KbFileEntity.status)
    doc_status: str = Field(default=DataSourceDocStatus.discovered)
    last_error: Optional[str] = Field(default=None, sa_column=Column(Text))

    # timeline
    first_seen_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime))
    last_fetched_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    last_changed_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))

    created_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime))
    updated_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime))

    @field_serializer("created_at", "updated_at", "first_seen_at", "last_fetched_at", "last_changed_at")
    def serialize_dt(self, dt: Optional[datetime], _info):
        if dt is None:
            return None
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        return dt.isoformat()


# ---------------------------------------------------------------------------
# 1.3 DataSourceSyncRunEntity — sync history (one row per sync)
# ---------------------------------------------------------------------------
class DataSourceSyncRunEntity(SQLModel, table=True):
    __tablename__ = "pai_datasource_sync_run"
    __table_args__ = (
        Index("idx_datasource_sync_run", "datasource_id", "started_at"),
    )

    id: str = Field(default_factory=lambda: uuid.uuid4().hex, primary_key=True, max_length=64)
    tenant_id: Optional[str] = Field(default=DEFAULT_TENANT_ID, max_length=64)
    datasource_id: str = Field(default=None, foreign_key="pai_datasource.id", ondelete="CASCADE", max_length=64)
    kb_id: str = Field(default=None, max_length=64)

    trigger: str = Field(default=SyncTrigger.manual)
    triggered_by: Optional[str] = Field(default=None, max_length=64)  # user_id
    status: str = Field(default=SyncRunStatus.running)

    started_at: datetime = Field(default_factory=_utcnow, sa_column=Column(DateTime))
    finished_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))
    duration_ms: Optional[int] = Field(default=None)

    # change-set counters
    n_discovered: int = Field(default=0)
    n_added: int = Field(default=0)
    n_updated: int = Field(default=0)
    n_deleted: int = Field(default=0)
    n_unchanged: int = Field(default=0)
    n_failed: int = Field(default=0)

    report: dict = Field(default_factory=dict, sa_column=Column("report", JSON))
    error: Optional[str] = Field(default=None, sa_column=Column(Text))

    @field_serializer("started_at", "finished_at")
    def serialize_dt(self, dt: Optional[datetime], _info):
        if dt is None:
            return None
        if dt.tzinfo is None:
            return f"{dt.isoformat()}Z"
        return dt.isoformat()
