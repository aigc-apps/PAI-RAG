from __future__ import annotations

import asyncio
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.auth import require_user
from app.deps import AppState, get_state
from app.knowledge import KnowledgeService
from app.store.base import User

router = APIRouter()

# Keep strong references to fire-and-forget sync tasks so they are not
# garbage-collected mid-run (asyncio only holds a weak reference).
_sync_tasks: set[asyncio.Task] = set()


def get_knowledge_service(state: AppState = Depends(get_state)) -> KnowledgeService:
    svc = getattr(state, "knowledge", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="knowledge service is not configured")
    return svc


def _dump(row) -> dict:
    return row.model_dump(mode="json")


def _dump_chunk(row) -> dict:
    # Drop the embedding vector — it's large (dims-length float array) and never
    # used by the UI; shipping it on every chunk list is pure waste.
    data = row.model_dump(mode="json")
    data.pop("embedding", None)
    return data


def _not_found(exc: Exception) -> HTTPException:
    msg = str(exc) or "knowledge resource not found"
    status = 404 if "not found" in msg else 403
    return HTTPException(status_code=status, detail=msg)


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    visibility: str = "private"
    default_parser_config: dict[str, Any] = Field(default_factory=dict)
    default_retrieval_config: dict[str, Any] = Field(default_factory=dict)
    embedding_config: dict[str, Any] = Field(default_factory=dict)
    vector_store_config: dict[str, Any] = Field(default_factory=dict)
    keyword_index_config: dict[str, Any] = Field(default_factory=dict)
    rerank_config: dict[str, Any] = Field(default_factory=dict)


class KnowledgeBasePatch(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    visibility: Optional[str] = None
    default_parser_config: Optional[dict[str, Any]] = None
    default_retrieval_config: Optional[dict[str, Any]] = None
    embedding_config: Optional[dict[str, Any]] = None
    vector_store_config: Optional[dict[str, Any]] = None
    keyword_index_config: Optional[dict[str, Any]] = None
    rerank_config: Optional[dict[str, Any]] = None


class ImportTextDocumentPayload(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1)
    uri: Optional[str] = None
    source_type: str = "text"
    mime_type: str = "text/plain"
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    category: Optional[str] = None
    custom_metadata: dict[str, Any] = Field(default_factory=dict)


class SearchPayload(BaseModel):
    kb_ids: list[str]
    query: str
    top_k: int = 6
    offset: int = 0
    score_threshold: float = 0.0
    mode: str = "hybrid"
    filters: dict[str, Any] = Field(default_factory=dict)


class KeywordPayload(BaseModel):
    kb_ids: list[str]
    pattern: str
    case_sensitive: bool = False
    filters: dict[str, Any] = Field(default_factory=dict)
    limit: int = 20
    context_lines: int = 0


class CatalogPayload(BaseModel):
    kb_ids: list[str]
    query: str = ""
    filters: dict[str, Any] = Field(default_factory=dict)
    limit: int = 20


class MetadataPayload(BaseModel):
    kb_ids: list[str]
    filters: dict[str, Any] = Field(default_factory=dict)
    limit: int = 50


class FetchPayload(BaseModel):
    kb_id: str
    ref: dict[str, str]
    mode: str = "full_doc"
    max_chars: int = 6000
    offset: int = 0


class ChunkStatusPayload(BaseModel):
    reason: str = ""


class DataSourceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    source_type: str = "llms_txt"
    source_config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    sync_schedule: Optional[str] = None


class DataSourcePatch(BaseModel):
    name: Optional[str] = None
    source_config: Optional[dict[str, Any]] = None
    enabled: Optional[bool] = None
    sync_schedule: Optional[str] = None


@router.get("/v1/knowledge-bases")
async def list_knowledge_bases(
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    rows = await svc.list_kbs(user=user)
    return {"data": [_dump(r) for r in rows]}


@router.post("/v1/knowledge-bases")
async def create_knowledge_base(
    payload: KnowledgeBaseCreate,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    row = await svc.create_kb(user=user, **payload.model_dump())
    return _dump(row)


@router.get("/v1/knowledge-bases/{kb_id}")
async def get_knowledge_base(
    kb_id: str,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    try:
        row = await svc.get_kb(kb_id, user=user)
    except PermissionError as exc:
        raise _not_found(exc) from exc
    return _dump(row)


@router.patch("/v1/knowledge-bases/{kb_id}")
async def update_knowledge_base(
    kb_id: str,
    payload: KnowledgeBasePatch,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    try:
        row = await svc.update_kb(
            kb_id,
            user=user,
            patch=payload.model_dump(exclude_unset=True),
        )
    except PermissionError as exc:
        raise _not_found(exc) from exc
    return _dump(row)


@router.delete("/v1/knowledge-bases/{kb_id}")
async def delete_knowledge_base(
    kb_id: str,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    try:
        await svc.delete_kb(kb_id, user=user)
    except PermissionError as exc:
        raise _not_found(exc) from exc
    return {"ok": True}


@router.post("/v1/knowledge-bases/{kb_id}/documents/import")
async def import_text_document(
    kb_id: str,
    payload: ImportTextDocumentPayload,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    try:
        doc, job = await svc.import_text_document(kb_id, user=user, **payload.model_dump())
    except PermissionError as exc:
        raise _not_found(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"document": _dump(doc), "job": _dump(job)}


@router.get("/v1/knowledge-bases/{kb_id}/documents")
async def list_documents(
    kb_id: str,
    status: Optional[str] = None,
    source_type: Optional[str] = None,
    category: Optional[str] = None,
    tag: Optional[str] = None,
    query: str = "",
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    try:
        rows, total = await svc.list_documents(
            kb_id,
            user=user,
            filters={
                "status": status,
                "source_type": source_type,
                "category": category,
                "tag": tag,
                "query": query,
            },
            limit=limit,
            offset=offset,
        )
    except PermissionError as exc:
        raise _not_found(exc) from exc
    return {"data": [_dump(r) for r in rows], "total": total, "offset": offset, "limit": limit,
            "has_more": offset + len(rows) < total}


@router.get("/v1/knowledge-bases/{kb_id}/chunks")
async def list_chunks(
    kb_id: str,
    document_id: Optional[str] = None,
    status: Optional[str] = Query(default=None),
    query: str = "",
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    try:
        rows, total = await svc.list_chunks(
            kb_id,
            user=user,
            document_id=document_id,
            status=status,
            query=query,
            limit=limit,
            offset=offset,
        )
    except PermissionError as exc:
        raise _not_found(exc) from exc
    return {"data": [_dump_chunk(r) for r in rows], "total": total, "offset": offset, "limit": limit,
            "has_more": offset + len(rows) < total}


@router.post("/v1/knowledge-bases/{kb_id}/chunks/{chunk_id}/disable")
async def disable_chunk(
    kb_id: str,
    chunk_id: str,
    payload: ChunkStatusPayload,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    try:
        row = await svc.set_chunk_status(
            kb_id,
            chunk_id,
            user=user,
            status="disabled",
            reason=payload.reason,
        )
    except PermissionError as exc:
        raise _not_found(exc) from exc
    return _dump(row)


@router.post("/v1/knowledge-bases/{kb_id}/chunks/{chunk_id}/enable")
async def enable_chunk(
    kb_id: str,
    chunk_id: str,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    try:
        row = await svc.set_chunk_status(kb_id, chunk_id, user=user, status="active")
    except PermissionError as exc:
        raise _not_found(exc) from exc
    return _dump(row)


@router.post("/v1/knowledge/query/search")
async def semantic_search(
    payload: SearchPayload,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    hits, total = await svc.search(user=user, **payload.model_dump())
    return {
        "data": [hit.__dict__ for hit in hits],
        "total": total,
        "offset": payload.offset,
        "limit": payload.top_k,
        "has_more": payload.offset + len(hits) < total,
    }


@router.post("/v1/knowledge/query/keyword")
async def keyword_match(
    payload: KeywordPayload,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    query = payload.pattern if payload.case_sensitive else payload.pattern.lower()
    hits, total = await svc.search(
        user=user,
        kb_ids=payload.kb_ids,
        query=payload.pattern,
        top_k=payload.limit,
        score_threshold=0.0,
        mode="keyword",
        filters=payload.filters,
    )
    data = []
    for hit in hits:
        text = hit.text if payload.case_sensitive else hit.text.lower()
        if query in text:
            data.append(hit.__dict__)
    return {"data": data[: payload.limit], "total": total}


@router.get("/v1/knowledge/engine")
async def search_engine_status(
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    return await svc.engine_status()


@router.post("/v1/knowledge-bases/{kb_id}/reindex")
async def reindex_kb(
    kb_id: str,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    try:
        return await svc.reindex_kb(kb_id, user=user)
    except PermissionError as exc:
        raise _not_found(exc) from exc


@router.post("/v1/knowledge/query/catalog")
async def catalog_search(
    payload: CatalogPayload,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    rows = await svc.catalog_search(user=user, **payload.model_dump())
    return {"data": [_dump(r) for r in rows]}


@router.post("/v1/knowledge/query/metadata")
async def metadata_search(
    payload: MetadataPayload,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    rows = await svc.catalog_search(
        user=user,
        kb_ids=payload.kb_ids,
        query="",
        filters=payload.filters,
        limit=payload.limit,
    )
    return {"data": [_dump(r) for r in rows]}


@router.post("/v1/knowledge/query/fetch")
async def fetch_context(
    payload: FetchPayload,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    try:
        return await svc.fetch_document_or_chunk(
            user=user,
            kb_id=payload.kb_id,
            document_id=payload.ref.get("document_id"),
            chunk_id=payload.ref.get("chunk_id"),
            mode=payload.mode,
            max_chars=payload.max_chars,
            offset=payload.offset,
        )
    except (LookupError, PermissionError) as exc:
        raise _not_found(exc) from exc


# --------------------------------------------------------------------------- #
# Data sources
# --------------------------------------------------------------------------- #


@router.get("/v1/knowledge-bases/{kb_id}/datasources")
async def list_data_sources(
    kb_id: str,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    try:
        rows = await svc.list_data_sources(kb_id, user=user)
    except PermissionError as exc:
        raise _not_found(exc) from exc
    return {"data": [_dump(r) for r in rows]}


@router.post("/v1/knowledge-bases/{kb_id}/datasources")
async def create_data_source(
    kb_id: str,
    payload: DataSourceCreate,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    try:
        row = await svc.create_data_source(kb_id, user=user, **payload.model_dump())
    except PermissionError as exc:
        raise _not_found(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _dump(row)


@router.get("/v1/knowledge-bases/{kb_id}/datasources/{ds_id}")
async def get_data_source(
    kb_id: str,
    ds_id: str,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    try:
        row = await svc.get_data_source(kb_id, ds_id, user=user)
    except PermissionError as exc:
        raise _not_found(exc) from exc
    return _dump(row)


@router.patch("/v1/knowledge-bases/{kb_id}/datasources/{ds_id}")
async def update_data_source(
    kb_id: str,
    ds_id: str,
    payload: DataSourcePatch,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    try:
        row = await svc.update_data_source(
            kb_id, ds_id, user=user, patch=payload.model_dump(exclude_unset=True)
        )
    except PermissionError as exc:
        raise _not_found(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _dump(row)


@router.delete("/v1/knowledge-bases/{kb_id}/datasources/{ds_id}")
async def delete_data_source(
    kb_id: str,
    ds_id: str,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    try:
        await svc.delete_data_source(kb_id, ds_id, user=user)
    except PermissionError as exc:
        raise _not_found(exc) from exc
    return {"ok": True}


@router.post("/v1/knowledge-bases/{kb_id}/datasources/{ds_id}/sync", status_code=202)
async def sync_data_source(
    kb_id: str,
    ds_id: str,
    user: User = Depends(require_user),
    svc: KnowledgeService = Depends(get_knowledge_service),
):
    # Pre-flight synchronously so permission / not-found / disabled errors reach
    # the caller; the actual sync runs fire-and-forget and is polled via GET.
    try:
        await svc.get_kb(kb_id, user=user, require_manage=True)
        row = await svc.get_data_source(kb_id, ds_id, user=user)
    except PermissionError as exc:
        raise _not_found(exc) from exc
    if not row.enabled:
        raise HTTPException(status_code=400, detail="data source is disabled")
    if row.status == "syncing":
        raise HTTPException(status_code=409, detail="a sync is already in progress")

    task = asyncio.create_task(svc.sync_data_source(kb_id, ds_id, user=user))
    _sync_tasks.add(task)
    task.add_done_callback(_sync_tasks.discard)
    return {"ok": True, "status": "syncing", "data_source": _dump(row)}
