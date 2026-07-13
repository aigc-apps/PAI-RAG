from __future__ import annotations
from datetime import datetime, timezone
from typing import Optional
from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON, BigInteger, DateTime, Text


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_field(**kwargs):
    return Field(sa_type=DateTime(timezone=True), **kwargs)


class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"
    id: str = Field(primary_key=True, max_length=64)
    user_id: Optional[str] = Field(default=None, index=True, max_length=64)
    created_at: datetime = _utc_field(default_factory=_now)
    updated_at: datetime = _utc_field(default_factory=_now)
    title: Optional[str] = Field(default=None, max_length=200)
    last_response_id: Optional[str] = Field(default=None, max_length=64)
    summary: Optional[str] = Field(default=None, sa_column=Column("summary", Text))
    summarized_seq: int = Field(default=-1)
    meta: dict = Field(default_factory=dict, sa_column=Column("metadata", JSON))


class ConversationItem(SQLModel, table=True):
    __tablename__ = "conversation_items"
    id: str = Field(primary_key=True, max_length=64)
    conversation_id: str = Field(index=True, max_length=64)
    seq: int = Field(sa_column=Column(BigInteger))
    type: str = Field(max_length=32)        # message|reasoning|function_call|function_call_output
    role: Optional[str] = Field(default=None, max_length=16)
    content: dict = Field(default_factory=dict, sa_column=Column(JSON))
    response_id: Optional[str] = Field(default=None, index=True, max_length=64)
    user_id: Optional[str] = Field(default=None, index=True, max_length=64)
    created_at: datetime = _utc_field(default_factory=_now)


class UserRow(SQLModel, table=True):
    __tablename__ = "users"
    id: str = Field(primary_key=True, max_length=64)
    display_name: Optional[str] = Field(default=None, max_length=200)
    # Auth identity. `id` (uuid) stays the ownership key for conversations /
    # memories / aliyun bindings; `email` is only the login handle. `status` is
    # invited (link issued, no password yet) -> active -> disabled. Nullable
    # because pre-auth rows and freshly-invited users have no password/email yet.
    email: Optional[str] = Field(default=None, index=True, unique=True, max_length=320)
    password_hash: Optional[str] = Field(default=None, max_length=512)
    role: str = Field(default="user", max_length=16)          # admin | user
    status: str = Field(default="active", max_length=16)      # invited | active | disabled
    invite_token_hash: Optional[str] = Field(default=None, max_length=128)
    invite_expires_at: Optional[datetime] = _utc_field(default=None)
    created_at: datetime = _utc_field(default_factory=_now)
    meta: dict = Field(default_factory=dict, sa_column=Column("metadata", JSON))


class ResponseRow(SQLModel, table=True):
    __tablename__ = "responses"
    id: str = Field(primary_key=True, max_length=64)
    conversation_id: Optional[str] = Field(default=None, index=True, max_length=64)
    previous_response_id: Optional[str] = Field(default=None, index=True, max_length=64)
    model: str = Field(max_length=128)
    status: str = Field(max_length=32)
    usage: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    error: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = _utc_field(default_factory=_now)
    meta: dict = Field(default_factory=dict, sa_column=Column("metadata", JSON))


class MemoryRow(SQLModel, table=True):
    __tablename__ = "memory_items"
    id: str = Field(primary_key=True, max_length=64)
    user_id: str = Field(index=True, max_length=64)
    text: str = Field(sa_column=Column("text", Text))
    kind: str = Field(default="fact", max_length=32)
    source_response_id: Optional[str] = Field(default=None, max_length=64)
    status: str = Field(default="active", max_length=16)
    created_at: datetime = _utc_field(default_factory=_now)
    updated_at: datetime = _utc_field(default_factory=_now)


class KnowledgeBaseRow(SQLModel, table=True):
    __tablename__ = "knowledge_bases"
    id: str = Field(primary_key=True, max_length=64)
    name: str = Field(max_length=200)
    description: str = Field(default="", sa_column=Column(Text))
    owner_user_id: str = Field(index=True, max_length=64)
    visibility: str = Field(default="private", max_length=32)
    status: str = Field(default="empty", index=True, max_length=32)
    default_parser_config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    default_retrieval_config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    embedding_config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    vector_store_config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    keyword_index_config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    rerank_config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    active_index_version_id: Optional[str] = Field(default=None, max_length=64)
    document_count: int = Field(default=0)
    chunk_count: int = Field(default=0)
    created_at: datetime = _utc_field(default_factory=_now)
    updated_at: datetime = _utc_field(default_factory=_now)
    deleted_at: Optional[datetime] = _utc_field(default=None, index=True)


class KnowledgeDocumentRow(SQLModel, table=True):
    __tablename__ = "knowledge_documents"
    id: str = Field(primary_key=True, max_length=64)
    kb_id: str = Field(index=True, max_length=64)
    source_id: Optional[str] = Field(default=None, index=True, max_length=64)
    uri: str = Field(sa_column=Column(Text))
    source_type: str = Field(default="text", index=True, max_length=32)
    title: str = Field(default="", max_length=500)
    description: str = Field(default="", sa_column=Column(Text))
    mime_type: str = Field(default="text/plain", max_length=128)
    size_bytes: int = Field(default=0, sa_column=Column(BigInteger))
    content_hash: str = Field(default="", index=True, max_length=128)
    etag: Optional[str] = Field(default=None, max_length=256)
    last_modified: Optional[datetime] = _utc_field(default=None)
    language: Optional[str] = Field(default=None, max_length=32)
    tags: list = Field(default_factory=list, sa_column=Column(JSON))
    category: Optional[str] = Field(default=None, index=True, max_length=128)
    visibility: str = Field(default="inherit", max_length=32)
    custom_metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))
    system_metadata: dict = Field(default_factory=dict, sa_column=Column(JSON))
    status: str = Field(default="indexed", index=True, max_length=32)
    # SQL content and search-engine visibility are separate durability stages.
    # Manual/local imports default to indexed; the production sync writer sets
    # pending before dispatching the corresponding Elasticsearch batch.
    search_index_status: str = Field(default="indexed", index=True, max_length=32)
    search_index_error: Optional[str] = Field(default=None, sa_column=Column(Text))
    search_index_attempts: int = Field(default=0)
    chunk_count: int = Field(default=0)
    error_code: Optional[str] = Field(default=None, max_length=128)
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text))
    indexed_at: Optional[datetime] = _utc_field(default=None, index=True)
    created_by: str = Field(max_length=64)
    updated_by: Optional[str] = Field(default=None, max_length=64)
    created_at: datetime = _utc_field(default_factory=_now)
    updated_at: datetime = _utc_field(default_factory=_now)
    deleted_at: Optional[datetime] = _utc_field(default=None, index=True)


class KnowledgeChunkRow(SQLModel, table=True):
    __tablename__ = "knowledge_chunks"
    id: str = Field(primary_key=True, max_length=64)
    kb_id: str = Field(index=True, max_length=64)
    document_id: str = Field(index=True, max_length=64)
    chunk_index: int = Field(default=0)
    text: str = Field(sa_column=Column(Text))
    text_hash: str = Field(default="", max_length=128)
    heading_path: list = Field(default_factory=list, sa_column=Column(JSON))
    char_start: Optional[int] = Field(default=None)
    char_end: Optional[int] = Field(default=None)
    token_count: int = Field(default=0)
    chunk_metadata: dict = Field(default_factory=dict, sa_column=Column("metadata", JSON))
    status: str = Field(default="active", index=True, max_length=32)
    embedding: list = Field(default_factory=list, sa_column=Column(JSON))
    embedding_ref: Optional[str] = Field(default=None, max_length=128)
    indexed_at: Optional[datetime] = _utc_field(default=None, index=True)
    disabled_by: Optional[str] = Field(default=None, max_length=64)
    disabled_reason: Optional[str] = Field(default=None, sa_column=Column(Text))
    created_at: datetime = _utc_field(default_factory=_now)
    updated_at: datetime = _utc_field(default_factory=_now)
    deleted_at: Optional[datetime] = _utc_field(default=None, index=True)


class KnowledgeDocumentContentRow(SQLModel, table=True):
    """The exact ingested text of a document, stored once (1:1 with a document).

    Kept in its own table — not a column on ``knowledge_documents`` — so the hot
    ``list_documents`` scan (which selects every document column) never drags the
    full body along; ``knowledge_read`` loads it only when it reads a file.
    Lets full-document reads return the original verbatim instead of re-stitching
    overlapping chunks."""
    __tablename__ = "knowledge_document_contents"
    document_id: str = Field(primary_key=True, max_length=64)
    kb_id: str = Field(index=True, max_length=64)
    text: str = Field(sa_column=Column(Text))
    content_hash: str = Field(default="", max_length=128)
    char_len: int = Field(default=0)
    # Over-long documents are stored capped at MAX_STORED_CONTENT_CHARS; this flags
    # that the persisted ``text`` is a prefix and deeper text lives only in chunks.
    truncated: bool = Field(default=False)
    updated_at: datetime = _utc_field(default_factory=_now)


class KnowledgeIngestionJobRow(SQLModel, table=True):
    __tablename__ = "knowledge_ingestion_jobs"
    id: str = Field(primary_key=True, max_length=64)
    kb_id: str = Field(index=True, max_length=64)
    source_id: Optional[str] = Field(default=None, index=True, max_length=64)
    document_id: Optional[str] = Field(default=None, index=True, max_length=64)
    type: str = Field(default="import", max_length=32)
    trigger_type: str = Field(default="manual", max_length=32)
    triggered_by: Optional[str] = Field(default=None, max_length=64)
    status: str = Field(default="completed", index=True, max_length=32)
    total_count: int = Field(default=0)
    succeeded_count: int = Field(default=0)
    failed_count: int = Field(default=0)
    skipped_count: int = Field(default=0)
    started_at: Optional[datetime] = _utc_field(default=None)
    finished_at: Optional[datetime] = _utc_field(default=None)
    error_summary: Optional[str] = Field(default=None, sa_column=Column(Text))
    created_at: datetime = _utc_field(default_factory=_now)
    updated_at: datetime = _utc_field(default_factory=_now)


class BackgroundJobRow(SQLModel, table=True):
    """Generic durable work queue — the execution substrate for deferred work.

    Distinct from ``knowledge_ingestion_jobs`` (which is a per-document audit log
    of ingestion history): this is the claimable queue a worker pool drains.
    ``kind`` dispatches to a handler; ``payload`` carries its args. Designed to be
    reused by future producers (a cron scheduler sets ``run_after``; a HITL
    approval flips a ``waiting`` job back to ``queued``) — see app/jobs.py.
    """

    __tablename__ = "background_jobs"
    id: str = Field(primary_key=True, max_length=64)
    kind: str = Field(index=True, max_length=64)         # kb_ingest | kb_sync | ...
    # queued | running | waiting | succeeded | failed.
    # `waiting` is non-terminal and NOT claimable — reserved for a run that is
    # parked awaiting a human (HITL); an external event returns it to `queued`.
    status: str = Field(default="queued", index=True, max_length=32)
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON))
    progress: dict = Field(default_factory=dict, sa_column=Column(JSON))
    result: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    error: Optional[str] = Field(default=None, sa_column=Column(Text))
    attempts: int = Field(default=0)
    max_attempts: int = Field(default=3)
    # Earliest time the job may be claimed. NULL = immediately. A future scheduler
    # sets this for delayed / one-shot runs; retry/backoff also uses it.
    run_after: Optional[datetime] = _utc_field(default=None, index=True)
    priority: int = Field(default=0)                     # lower runs first
    worker_id: Optional[str] = Field(default=None, max_length=64)   # lease holder
    claimed_at: Optional[datetime] = _utc_field(default=None)
    heartbeat_at: Optional[datetime] = _utc_field(default=None)
    lease_expires_at: Optional[datetime] = _utc_field(default=None, index=True)
    cancel_requested_at: Optional[datetime] = _utc_field(default=None)
    kb_id: Optional[str] = Field(default=None, index=True, max_length=64)
    created_by: Optional[str] = Field(default=None, index=True, max_length=64)
    created_at: datetime = _utc_field(default_factory=_now)
    updated_at: datetime = _utc_field(default_factory=_now)
    started_at: Optional[datetime] = _utc_field(default=None)
    finished_at: Optional[datetime] = _utc_field(default=None)


class AppConfigDocumentRow(SQLModel, table=True):
    """Singleton authored product configuration.

    Runtime status is intentionally not stored here; it is recomputed from this
    authored document and deployment environment on read.
    """

    __tablename__ = "app_config_documents"
    id: str = Field(primary_key=True, max_length=64)
    schema_version: int = Field(default=1)
    document_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    revision: int = Field(default=1, index=True)
    checksum: str = Field(default="", max_length=128)
    updated_by: Optional[str] = Field(default=None, max_length=64)
    updated_at: datetime = _utc_field(default_factory=_now)


class AppConfigRevisionRow(SQLModel, table=True):
    """Append-only revision history for admin config changes."""

    __tablename__ = "app_config_revisions"
    id: Optional[int] = Field(default=None, primary_key=True)
    config_id: str = Field(index=True, max_length=64)
    revision: int = Field(index=True)
    schema_version: int = Field(default=1)
    document_json: dict = Field(default_factory=dict, sa_column=Column(JSON))
    checksum: str = Field(default="", max_length=128)
    updated_by: Optional[str] = Field(default=None, max_length=64)
    updated_at: datetime = _utc_field(default_factory=_now)


class KnowledgeDataSourceRow(SQLModel, table=True):
    """A configured data source feeding a knowledge base (e.g. an Aliyun docs
    ``llms.txt`` manifest). Aggregate sync state is stored inline — the backend
    does not keep a per-document manifest table; incremental diff is computed from
    the ingested ``knowledge_documents`` themselves (source_id + uri + content_hash).
    """

    __tablename__ = "knowledge_data_sources"
    id: str = Field(primary_key=True, max_length=64)
    kb_id: str = Field(index=True, max_length=64)
    name: str = Field(max_length=200)
    source_key: str = Field(default="", index=True, max_length=128)
    source_type: str = Field(default="llms_txt", index=True, max_length=32)
    source_config: dict = Field(default_factory=dict, sa_column=Column(JSON))
    enabled: bool = Field(default=True)
    # stored but not yet acted on (no scheduler in MVP)
    sync_schedule: Optional[str] = Field(default=None, max_length=128)
    status: str = Field(default="idle", index=True, max_length=32)
    active_job_id: Optional[str] = Field(default=None, index=True, max_length=64)
    last_sync_at: Optional[datetime] = _utc_field(default=None)
    last_sync_finished_at: Optional[datetime] = _utc_field(default=None)
    last_error: Optional[str] = Field(default=None, sa_column=Column(Text))
    doc_count: int = Field(default=0)
    last_sync_report: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_by: str = Field(max_length=64)
    created_at: datetime = _utc_field(default_factory=_now)
    updated_at: datetime = _utc_field(default_factory=_now)
    deleted_at: Optional[datetime] = _utc_field(default=None, index=True)


class KnowledgeIndexVersionRow(SQLModel, table=True):
    __tablename__ = "knowledge_index_versions"
    id: str = Field(primary_key=True, max_length=64)
    kb_id: str = Field(index=True, max_length=64)
    version: int = Field(default=1)
    status: str = Field(default="active", index=True, max_length=32)
    embedding_provider_id: str = Field(default="local_hash", max_length=128)
    embedding_model: str = Field(default="local-hash-v1", max_length=128)
    embedding_dimension: int = Field(default=64)
    vector_store_provider_id: str = Field(default="local_sql", max_length=128)
    vector_index_name: str = Field(default="", max_length=256)
    vector_namespace: str = Field(default="", max_length=256)
    keyword_index_name: Optional[str] = Field(default=None, max_length=256)
    document_count: int = Field(default=0)
    chunk_count: int = Field(default=0)
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text))
    created_by: str = Field(default="", max_length=64)
    created_at: datetime = _utc_field(default_factory=_now)
    activated_at: Optional[datetime] = _utc_field(default=None)
    retired_at: Optional[datetime] = _utc_field(default=None)
