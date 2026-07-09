from __future__ import annotations

import asyncio
import hashlib
import math
import re
from datetime import datetime, timezone
from typing import Optional

from loguru import logger
from sqlalchemy import String, cast, delete, func
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.datasource.registry import get_adapter, supported_source_types
from app.datasource.schema import SourceDocument
from app.retrieval_models import build_embedder
from app.search_engine import LocalSearchEngine, SearchHit
from app.models import (
    KnowledgeBaseRow,
    KnowledgeChunkRow,
    KnowledgeDataSourceRow,
    KnowledgeDocumentRow,
    KnowledgeIndexVersionRow,
    KnowledgeIngestionJobRow,
)
from app.store.base import User, _uuid


DEFAULT_EMBEDDING_CONFIG = {
    "provider_id": "local_hash",
    "model": "local-hash-v1",
    "dimension": 64,
    "normalize": True,
}
DEFAULT_VECTOR_STORE_CONFIG = {
    "provider_id": "local_sql",
    "index_name": "",
    "namespace": "",
    "metric": "cosine",
    "dimension": 64,
}
DEFAULT_KEYWORD_INDEX_CONFIG = {
    "provider_id": "local_sql_like",
    "enabled": True,
}
DEFAULT_RERANK_CONFIG = {"enabled": False}
DEFAULT_PARSER_CONFIG = {"chunk_size": 1000, "chunk_overlap": 150}
DEFAULT_RETRIEVAL_CONFIG = {
    "mode": "hybrid",
    "top_k": 6,
    "score_threshold": 0.0,
    "force_citation": True,
}


_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Max concurrent network fetches during a data source sync. Fetching is done off
# the event loop (asyncio.to_thread); ingestion is serialized to avoid concurrent
# writes to the (possibly sqlite) engine.
SYNC_FETCH_CONCURRENCY = 6


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _slugify(text: str) -> str:
    slug = _SLUG_RE.sub("-", (text or "").strip().lower()).strip("-")
    return slug[:64]


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def embed_text(text: str, *, dimension: int = 64) -> list[float]:
    """Small deterministic embedding for local MVP.

    This is not intended to be semantically strong. It gives the MVP a real
    vector-search path and a stable adapter boundary before external embedding
    providers are wired.
    """
    vec = [0.0] * dimension
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm:
        vec = [round(v / norm, 8) for v in vec]
    return vec


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    return float(sum(a[i] * b[i] for i in range(n)))


def keyword_score(query: str, text: str) -> float:
    q = _tokens(query)
    if not q:
        return 0.0
    text_l = (text or "").lower()
    hits = sum(1 for token in q if token in text_l)
    return hits / len(q)


def chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int, int]]:
    text = (text or "").strip()
    if not text:
        return []
    chunk_size = max(100, int(chunk_size or 1000))
    chunk_overlap = max(0, min(int(chunk_overlap or 0), chunk_size // 2))
    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        body = text[start:end].strip()
        if body:
            chunks.append((body, start, end))
        if end >= len(text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def _merge(defaults: dict, override: Optional[dict]) -> dict:
    data = dict(defaults)
    data.update(override or {})
    return data


class KnowledgeService:
    def __init__(self, engine, search_engine=None, fallback_to_local: bool = True, router=None):
        self._engine = engine
        # The local SQL scan is always available as a degradation target. The
        # primary engine (ES when configured) serves queries and receives the
        # offline index hooks; on an ES transport error in ``auto`` mode we fall
        # back to local so retrieval degrades instead of failing.
        self._local = LocalSearchEngine(engine)
        self._search = search_engine or self._local
        self._fallback_to_local = fallback_to_local
        # ProviderRouter — resolves a KB's embedder (ingest + query) and reranker.
        # Optional: without it every KB embeds with the local hash (old behaviour),
        # keeping offline/test paths network-free.
        self._router = router

    def _resolve_default_embedding(self) -> dict:
        """The embedding_config a new KB gets when the caller doesn't specify one.

        If a healthy embedding model is catalogued (its key resolves) — DashScope
        native or any openai-compatible provider — freeze the KB onto it;
        otherwise fall back to the local hash embedder. Recorded at creation and
        never changed afterward (see ``update_kb``)."""
        router = self._router
        if router is not None:
            model_id = router.default_model_id_of_type("embedding")
            if model_id:
                try:
                    cfg = router.get_config(model_id)
                    key = cfg.resolve_key()
                    if key or not cfg.api_key_env:
                        return {
                            "provider_id": cfg.provider,
                            "model": model_id,
                            "dimension": int(cfg.dimension or 1024),
                            "normalize": True,
                        }
                except Exception as ex:  # pragma: no cover - defensive
                    logger.warning(f"[knowledge] default embedding resolution failed: {ex!r}")
        return dict(DEFAULT_EMBEDDING_CONFIG)

    async def create_kb(
        self,
        *,
        user: User,
        name: str,
        description: str = "",
        visibility: str = "private",
        default_parser_config: Optional[dict] = None,
        default_retrieval_config: Optional[dict] = None,
        embedding_config: Optional[dict] = None,
        vector_store_config: Optional[dict] = None,
        keyword_index_config: Optional[dict] = None,
        rerank_config: Optional[dict] = None,
    ) -> KnowledgeBaseRow:
        kb_id = _uuid("kb")
        # embedding is frozen at creation. An explicit config wins; otherwise pick
        # the catalogued DashScope embedder when healthy, else the local hash.
        if embedding_config is not None:
            embedding = _merge(DEFAULT_EMBEDDING_CONFIG, embedding_config)
        else:
            embedding = self._resolve_default_embedding()
        vector = _merge(DEFAULT_VECTOR_STORE_CONFIG, vector_store_config)
        # Keep the vector store's declared dimension in step with the embedder
        # (the ES mapping reads embedding_config, but keep both coherent).
        if not (vector_store_config and "dimension" in vector_store_config):
            vector["dimension"] = int(embedding.get("dimension") or 64)
        vector.setdefault("namespace", kb_id)
        vector.setdefault("index_name", f"{kb_id}_vectors_v1")
        parser = _merge(DEFAULT_PARSER_CONFIG, default_parser_config)
        retrieval = _merge(DEFAULT_RETRIEVAL_CONFIG, default_retrieval_config)
        async with AsyncSession(self._engine) as s:
            kb = KnowledgeBaseRow(
                id=kb_id,
                name=name,
                description=description or "",
                owner_user_id=user.id,
                visibility=visibility,
                default_parser_config=parser,
                default_retrieval_config=retrieval,
                embedding_config=embedding,
                vector_store_config=vector,
                keyword_index_config=_merge(DEFAULT_KEYWORD_INDEX_CONFIG, keyword_index_config),
                rerank_config=_merge(DEFAULT_RERANK_CONFIG, rerank_config),
            )
            idx = KnowledgeIndexVersionRow(
                id=_uuid("idx"),
                kb_id=kb_id,
                version=1,
                status="active",
                embedding_provider_id=str(embedding.get("provider_id") or "local_hash"),
                embedding_model=str(embedding.get("model") or "local-hash-v1"),
                embedding_dimension=int(embedding.get("dimension") or 64),
                vector_store_provider_id=str(vector.get("provider_id") or "local_sql"),
                vector_index_name=str(vector.get("index_name") or f"{kb_id}_vectors_v1"),
                vector_namespace=str(vector.get("namespace") or kb_id),
                keyword_index_name=f"{kb_id}_keyword_v1",
                created_by=user.id,
                activated_at=now_utc(),
            )
            kb.active_index_version_id = idx.id
            s.add(kb)
            s.add(idx)
            await s.commit()
            await s.refresh(kb)
            return kb

    async def list_kbs(self, *, user: User) -> list[KnowledgeBaseRow]:
        async with AsyncSession(self._engine) as s:
            stmt = select(KnowledgeBaseRow).where(KnowledgeBaseRow.deleted_at.is_(None))
            if user.role != "admin":
                stmt = stmt.where(
                    (KnowledgeBaseRow.owner_user_id == user.id)
                    | (KnowledgeBaseRow.visibility.in_(["workspace", "public"]))
                )
            rows = (await s.exec(stmt.order_by(KnowledgeBaseRow.updated_at.desc()))).all()
            return list(rows)

    async def get_kb(self, kb_id: str, *, user: User, require_manage: bool = False) -> KnowledgeBaseRow:
        async with AsyncSession(self._engine) as s:
            kb = await s.get(KnowledgeBaseRow, kb_id)
        if kb is None or kb.deleted_at is not None:
            raise PermissionError("knowledge base not found")
        if require_manage and not self.can_manage(kb, user):
            raise PermissionError("knowledge base edit permission required")
        if not require_manage and not self.can_query(kb, user):
            raise PermissionError("knowledge base query permission required")
        return kb

    def can_manage(self, kb: KnowledgeBaseRow, user: User) -> bool:
        return user.role == "admin" or kb.owner_user_id == user.id

    def can_query(self, kb: KnowledgeBaseRow, user: User) -> bool:
        return self.can_manage(kb, user) or kb.visibility in {"workspace", "public"}

    async def update_kb(self, kb_id: str, *, user: User, patch: dict) -> KnowledgeBaseRow:
        async with AsyncSession(self._engine) as s:
            kb = await s.get(KnowledgeBaseRow, kb_id)
            if kb is None or kb.deleted_at is not None or not self.can_manage(kb, user):
                raise PermissionError("knowledge base edit permission required")
            # embedding is frozen at creation — changing it would strand every
            # already-ingested chunk vector (no reindex path). Reject explicitly.
            if patch.get("embedding_config") is not None:
                raise ValueError(
                    "embedding_config is immutable after knowledge base creation"
                )
            for key in ("name", "description", "visibility"):
                if key in patch and patch[key] is not None:
                    setattr(kb, key, patch[key])
            for key in (
                "default_parser_config",
                "default_retrieval_config",
                "vector_store_config",
                "keyword_index_config",
                "rerank_config",
            ):
                if key in patch and patch[key] is not None:
                    current = dict(getattr(kb, key) or {})
                    current.update(patch[key])
                    setattr(kb, key, current)
                    flag_modified(kb, key)
            kb.updated_at = now_utc()
            s.add(kb)
            await s.commit()
            await s.refresh(kb)
            return kb

    async def delete_kb(self, kb_id: str, *, user: User) -> None:
        async with AsyncSession(self._engine) as s:
            kb = await s.get(KnowledgeBaseRow, kb_id)
            if kb is None or kb.deleted_at is not None or not self.can_manage(kb, user):
                raise PermissionError("knowledge base edit permission required")
            kb.deleted_at = now_utc()
            kb.status = "disabled"
            kb.updated_at = now_utc()
            s.add(kb)
            await s.commit()
        if self._search is not self._local:
            try:
                await self._search.delete_kb(kb_id)
            except Exception as ex:
                logger.warning(f"[search] delete_kb failed for {kb_id}: {ex!r}")

    async def create_pending_document(
        self,
        kb_id: str,
        *,
        user: User,
        title: str,
        uri: Optional[str] = None,
        source_type: str = "file",
        source_id: Optional[str] = None,
        mime_type: str = "text/markdown",
        tags: Optional[list[str]] = None,
        category: Optional[str] = None,
    ) -> KnowledgeDocumentRow:
        """Upsert a document row in ``status="processing"`` before its content is
        ingested — so an enqueued upload/import shows up in the UI immediately.

        Deduped by ``(kb_id, uri)`` the same way ``import_text_document`` is, so the
        background worker's later ``import_text_document`` reuses this exact row
        (finds it as ``existing``) and flips it to ``indexed``.
        """
        async with AsyncSession(self._engine) as s:
            kb = await s.get(KnowledgeBaseRow, kb_id)
            if kb is None or kb.deleted_at is not None or not self.can_manage(kb, user):
                raise PermissionError("knowledge base edit permission required")
            stable_uri = uri or f"pending://{_uuid('doc')}"
            existing = (
                await s.exec(
                    select(KnowledgeDocumentRow).where(
                        KnowledgeDocumentRow.kb_id == kb_id,
                        KnowledgeDocumentRow.uri == stable_uri,
                        KnowledgeDocumentRow.deleted_at.is_(None),
                    )
                )
            ).first()
            doc = existing or KnowledgeDocumentRow(
                id=_uuid("doc"),
                kb_id=kb_id,
                uri=stable_uri,
                source_type=source_type,
                title=title or stable_uri,
                created_by=user.id,
            )
            if source_id is not None:
                doc.source_id = source_id
            doc.source_type = source_type
            doc.title = title or doc.title or stable_uri
            doc.mime_type = mime_type or "text/markdown"
            doc.tags = tags or []
            doc.category = category
            doc.status = "processing"
            doc.updated_by = user.id
            doc.updated_at = now_utc()
            s.add(doc)
            await s.commit()
            await s.refresh(doc)
            return doc

    async def import_text_document(
        self,
        kb_id: str,
        *,
        user: User,
        title: str,
        content: str,
        uri: Optional[str] = None,
        source_type: str = "text",
        source_id: Optional[str] = None,
        mime_type: str = "text/plain",
        description: str = "",
        tags: Optional[list[str]] = None,
        category: Optional[str] = None,
        custom_metadata: Optional[dict] = None,
        trigger_type: str = "manual",
    ) -> tuple[KnowledgeDocumentRow, KnowledgeIngestionJobRow]:
        async with AsyncSession(self._engine) as s:
            kb = await s.get(KnowledgeBaseRow, kb_id)
            if kb is None or kb.deleted_at is not None or not self.can_manage(kb, user):
                raise PermissionError("knowledge base edit permission required")
            content = content or ""
            if not content.strip():
                raise ValueError("content is required")
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            stable_uri = uri or f"text://{digest[:16]}"
            existing = (
                await s.exec(
                    select(KnowledgeDocumentRow).where(
                        KnowledgeDocumentRow.kb_id == kb_id,
                        KnowledgeDocumentRow.uri == stable_uri,
                        KnowledgeDocumentRow.deleted_at.is_(None),
                    )
                )
            ).first()
            doc_id = existing.id if existing else _uuid("doc")
            await s.exec(delete(KnowledgeChunkRow).where(KnowledgeChunkRow.document_id == doc_id))
            parser = _merge(DEFAULT_PARSER_CONFIG, kb.default_parser_config)
            chunks = chunk_text(
                content,
                chunk_size=int(parser.get("chunk_size") or 1000),
                chunk_overlap=int(parser.get("chunk_overlap") or 150),
            )
            # Embed every chunk body with the KB's frozen embedder (batched by the
            # embedder itself — DashScope caps at 10/req). Same embedder is used at
            # query time, so ingest and query vectors are always comparable.
            embedder = build_embedder(kb.embedding_config, self._router)
            chunk_vectors = await embedder.embed(
                [body for body, _s, _e in chunks], text_type="document"
            ) if chunks else []
            indexed_at = now_utc()
            if existing is None:
                doc = KnowledgeDocumentRow(
                    id=doc_id,
                    kb_id=kb_id,
                    uri=stable_uri,
                    source_type=source_type,
                    title=title or stable_uri,
                    created_by=user.id,
                )
            else:
                doc = existing
            if source_id is not None:
                doc.source_id = source_id
            doc.source_type = source_type
            doc.title = title or doc.title or stable_uri
            doc.description = description or ""
            doc.mime_type = mime_type or "text/plain"
            doc.size_bytes = len(content.encode("utf-8"))
            doc.content_hash = digest
            doc.tags = tags or []
            doc.category = category
            doc.custom_metadata = custom_metadata or {}
            doc.system_metadata = {"content_preview": content[:500]}
            doc.status = "indexed"
            doc.chunk_count = len(chunks)
            doc.indexed_at = indexed_at
            doc.updated_by = user.id
            doc.updated_at = indexed_at
            s.add(doc)
            es_chunks: list[dict] = []
            for idx, (body, start, end) in enumerate(chunks):
                chunk_id = _uuid("chk")
                chunk_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
                embedding = chunk_vectors[idx] if idx < len(chunk_vectors) else []
                s.add(
                    KnowledgeChunkRow(
                        id=chunk_id,
                        kb_id=kb_id,
                        document_id=doc_id,
                        chunk_index=idx,
                        text=body,
                        text_hash=chunk_hash,
                        char_start=start,
                        char_end=end,
                        token_count=len(_tokens(body)),
                        chunk_metadata={
                            "title": doc.title,
                            "source_uri": stable_uri,
                            "source_type": source_type,
                            "tags": tags or [],
                            "category": category,
                        },
                        embedding=embedding,
                        embedding_ref=f"{kb.active_index_version_id}:{chunk_id}",
                        indexed_at=indexed_at,
                    )
                )
                es_chunks.append({"chunk_id": chunk_id, "chunk_index": idx, "text": body, "embedding": embedding})
            job = KnowledgeIngestionJobRow(
                id=_uuid("job"),
                kb_id=kb_id,
                source_id=source_id,
                document_id=doc_id,
                type="import",
                trigger_type=trigger_type,
                triggered_by=user.id,
                status="completed",
                total_count=1,
                succeeded_count=1,
                started_at=indexed_at,
                finished_at=indexed_at,
            )
            s.add(job)
            await self._refresh_counts(s, kb)
            await s.commit()
            await s.refresh(doc)
            await s.refresh(job)
        # Mirror into the search engine (ES) best-effort — SQL is the source of
        # truth, so an ES outage must not fail ingestion. No-op for local.
        await self._index_chunks_best_effort(kb, doc, es_chunks)
        return doc, job

    async def _index_chunks_best_effort(self, kb, doc, es_chunks: list[dict]) -> None:
        if self._search is self._local:
            return
        try:
            await self._search.index_chunks(kb, doc, es_chunks)
        except Exception as ex:
            logger.warning(f"[search] index_chunks failed for doc {getattr(doc, 'id', '?')}: {ex!r}")

    async def _delete_from_engine_best_effort(self, kb_id: str, document_ids: list[str]) -> None:
        if self._search is self._local or not document_ids:
            return
        for doc_id in document_ids:
            try:
                await self._search.delete_document(kb_id, doc_id)
            except Exception as ex:
                logger.warning(f"[search] delete_document failed for {doc_id}: {ex!r}")

    async def _refresh_counts(self, s: AsyncSession, kb: KnowledgeBaseRow) -> None:
        doc_count = (
            await s.exec(
                select(func.count()).select_from(KnowledgeDocumentRow).where(
                    KnowledgeDocumentRow.kb_id == kb.id,
                    KnowledgeDocumentRow.deleted_at.is_(None),
                    KnowledgeDocumentRow.status != "deleted",
                )
            )
        ).one()
        chunk_count = (
            await s.exec(
                select(func.count()).select_from(KnowledgeChunkRow).where(
                    KnowledgeChunkRow.kb_id == kb.id,
                    KnowledgeChunkRow.deleted_at.is_(None),
                    KnowledgeChunkRow.status == "active",
                )
            )
        ).one()
        kb.document_count = int(doc_count)
        kb.chunk_count = int(chunk_count)
        kb.status = "ready" if kb.document_count > 0 else "empty"
        kb.updated_at = now_utc()
        s.add(kb)

    async def list_documents(
        self,
        kb_id: str,
        *,
        user: User,
        filters: Optional[dict] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> tuple[list[KnowledgeDocumentRow], int]:
        """Return ``(rows, total)``. All filters are pushed into SQL so ``total``
        reflects the full filtered match count (independent of limit/offset)."""
        await self.get_kb(kb_id, user=user)
        filters = filters or {}
        async with AsyncSession(self._engine) as s:
            conds = [
                KnowledgeDocumentRow.kb_id == kb_id,
                KnowledgeDocumentRow.deleted_at.is_(None),
            ]
            if filters.get("status"):
                conds.append(KnowledgeDocumentRow.status == filters["status"])
            if filters.get("source_type"):
                conds.append(KnowledgeDocumentRow.source_type == filters["source_type"])
            if filters.get("category"):
                conds.append(KnowledgeDocumentRow.category == filters["category"])
            query = (filters.get("query") or "").strip()
            if query:
                like = f"%{query}%"
                conds.append(
                    func.lower(KnowledgeDocumentRow.title).like(like.lower())
                    | func.lower(func.coalesce(KnowledgeDocumentRow.uri, "")).like(like.lower())
                )
            tag = filters.get("tag")
            if tag:
                # tags is a JSON array column; match the serialized element portably.
                conds.append(cast(KnowledgeDocumentRow.tags, String).like(f'%"{tag}"%'))
            total = int((await s.exec(select(func.count()).select_from(KnowledgeDocumentRow).where(*conds))).one())
            stmt = select(KnowledgeDocumentRow).where(*conds).order_by(KnowledgeDocumentRow.updated_at.desc())
            if limit is not None:
                stmt = stmt.offset(max(0, offset)).limit(max(1, limit))
            rows = (await s.exec(stmt)).all()
            return list(rows), total

    async def list_chunks(
        self,
        kb_id: str,
        *,
        user: User,
        document_id: Optional[str] = None,
        status: Optional[str] = None,
        query: str = "",
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> tuple[list[KnowledgeChunkRow], int]:
        """Return ``(rows, total)`` with all filters pushed into SQL."""
        await self.get_kb(kb_id, user=user)
        async with AsyncSession(self._engine) as s:
            conds = [
                KnowledgeChunkRow.kb_id == kb_id,
                KnowledgeChunkRow.deleted_at.is_(None),
            ]
            if document_id:
                conds.append(KnowledgeChunkRow.document_id == document_id)
            if status:
                conds.append(KnowledgeChunkRow.status == status)
            if query.strip():
                conds.append(func.lower(KnowledgeChunkRow.text).like(f"%{query.strip().lower()}%"))
            total = int((await s.exec(select(func.count()).select_from(KnowledgeChunkRow).where(*conds))).one())
            stmt = (
                select(KnowledgeChunkRow)
                .where(*conds)
                .order_by(KnowledgeChunkRow.document_id, KnowledgeChunkRow.chunk_index)
            )
            if limit is not None:
                stmt = stmt.offset(max(0, offset)).limit(max(1, limit))
            rows = (await s.exec(stmt)).all()
            return list(rows), total

    async def search(
        self,
        *,
        user: User,
        kb_ids: list[str],
        query: str,
        top_k: int = 6,
        offset: int = 0,
        score_threshold: float = 0.0,
        mode: str = "hybrid",
        filters: Optional[dict] = None,
    ) -> tuple[list[SearchHit], int]:
        """Permission-scope the requested KBs, then delegate retrieval to the
        active engine. Returns ``(hits, total)`` where ``total`` is the full match
        count (for pagination), independent of ``offset``/``top_k``."""
        if not query.strip():
            return [], 0
        filters = filters or {}
        allowed_kbs: list[str] = []
        dimension = 64
        first_kb: Optional[KnowledgeBaseRow] = None
        for kb_id in kb_ids:
            try:
                kb = await self.get_kb(kb_id, user=user)
                if not allowed_kbs:
                    first_kb = kb
                    dimension = int((kb.embedding_config or {}).get("dimension") or 64)
                allowed_kbs.append(kb_id)
            except PermissionError:
                continue
        if not allowed_kbs:
            return [], 0
        limit = max(1, min(int(top_k or 6), 50))
        offset = max(0, int(offset or 0))
        # Cross-KB search over heterogeneous embedders isn't supported: the query
        # is embedded once, with the FIRST allowed KB's embedder, and that vector
        # is compared against every KB's chunks. Same-embedder KBs are the norm.
        query_vector = None
        if mode in ("vector", "hybrid") and first_kb is not None:
            try:
                embedder = build_embedder(first_kb.embedding_config, self._router)
                vecs = await embedder.embed([query], text_type="query")
                query_vector = vecs[0] if vecs else None
            except Exception as ex:
                logger.warning(f"[search] query embedding failed ({ex!r}); engine will fall back")
                query_vector = None
        # Rerank is a query-time step (toggle via rerank_config.enabled). When on,
        # over-fetch a candidate pool, rerank it, and trim to the requested page.
        rerank_cfg = (first_kb.rerank_config if first_kb else None) or {}
        rerank_on = bool(rerank_cfg.get("enabled")) and self._router is not None
        fetch_limit = max(limit * 4, 50) if rerank_on else limit
        kwargs = dict(
            kb_ids=allowed_kbs, query=query, mode=mode, offset=offset, limit=fetch_limit,
            score_threshold=score_threshold, dimension=dimension, filters=filters,
            query_vector=query_vector,
        )
        engine = self._search
        if engine is self._local:
            hits, total = await self._local.search(**kwargs)
        else:
            try:
                hits, total = await engine.search(**kwargs)
            except Exception as ex:
                if not self._fallback_to_local:
                    raise
                logger.warning(f"[search] primary engine '{getattr(engine, 'name', '?')}' failed ({ex!r}); falling back to local")
                hits, total = await self._local.search(**kwargs)
        if rerank_on and hits:
            hits = await self._rerank_hits(query, hits, limit, rerank_cfg)
        return hits[:limit], total

    async def _rerank_hits(self, query, hits, limit, rerank_cfg) -> list[SearchHit]:
        """Reorder a candidate window with a DashScope reranker, best-effort:
        any failure returns the original order (trimmed). Overwrites each hit's
        ``score`` with the reranker's relevance score."""
        model_id = rerank_cfg.get("model") or self._router.default_model_id_of_type("rerank")
        if not model_id:
            return hits[:limit]
        top_n = int(rerank_cfg.get("top_n") or limit)
        try:
            reranker = self._router.get_reranker(model_id)
            ranked = await reranker.rerank(query, [h.text for h in hits], top_n=top_n)
        except Exception as ex:
            logger.warning(f"[rerank] '{model_id}' failed ({ex!r}); keeping original order")
            return hits[:limit]
        if not ranked:
            return hits[:limit]
        out: list[SearchHit] = []
        for idx, score in ranked:
            if 0 <= idx < len(hits):
                hit = hits[idx]
                hit.score = round(float(score), 6)
                out.append(hit)
        return out[:limit]

    async def reindex_kb(self, kb_id: str, *, user: User) -> dict:
        """Admin: (re)build the search-engine index for a KB from its current
        active chunks — for backfilling existing KBs or after switching engines.
        No-op (count only) when the local engine is active, since SQL is its index."""
        kb = await self.get_kb(kb_id, user=user, require_manage=True)
        async with AsyncSession(self._engine) as s:
            rows = (
                await s.exec(
                    select(KnowledgeChunkRow, KnowledgeDocumentRow)
                    .join(KnowledgeDocumentRow, KnowledgeDocumentRow.id == KnowledgeChunkRow.document_id)
                    .where(
                        KnowledgeChunkRow.kb_id == kb_id,
                        KnowledgeChunkRow.status == "active",
                        KnowledgeChunkRow.deleted_at.is_(None),
                        KnowledgeDocumentRow.status == "indexed",
                        KnowledgeDocumentRow.deleted_at.is_(None),
                    )
                )
            ).all()
        if self._search is self._local:
            return {"engine": "local", "indexed": 0, "chunks": len(rows), "reindexed": False}
        # Group chunks by document so index_chunks replaces each doc's set.
        by_doc: dict[str, tuple[KnowledgeDocumentRow, list[dict]]] = {}
        for chunk, doc in rows:
            entry = by_doc.setdefault(doc.id, (doc, []))
            entry[1].append({
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
                "embedding": list(chunk.embedding or []),
            })
        await self._search.ensure_index(kb)
        indexed = 0
        for doc, chunks in by_doc.values():
            await self._search.index_chunks(kb, doc, chunks)
            indexed += len(chunks)
        return {"engine": self._search.name, "indexed": indexed, "documents": len(by_doc), "reindexed": True}

    async def engine_status(self) -> dict:
        """Which retrieval engine is active + whether it's reachable (frontend badge)."""
        engine = self._search
        name = getattr(engine, "name", "local")
        configured = engine is not self._local
        healthy = True
        if configured:
            try:
                healthy = await engine.healthy()
            except Exception:
                healthy = False
        return {"engine": name, "configured": configured, "healthy": healthy}

    async def catalog_search(
        self,
        *,
        user: User,
        kb_ids: list[str],
        query: str = "",
        filters: Optional[dict] = None,
        limit: int = 20,
    ) -> list[KnowledgeDocumentRow]:
        docs: list[KnowledgeDocumentRow] = []
        for kb_id in kb_ids:
            rows, _ = await self.list_documents(kb_id, user=user, filters=filters)
            docs.extend(rows)
        if query:
            q = query.lower()
            docs = [
                d for d in docs
                if q in (d.title or "").lower()
                or q in (d.uri or "").lower()
                or q in " ".join(d.tags or []).lower()
                or q in (d.category or "").lower()
            ]
        return docs[: max(1, min(int(limit or 20), 100))]

    async def fetch_document_or_chunk(
        self,
        *,
        user: User,
        kb_id: str,
        document_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
        mode: str = "full_doc",
        max_chars: int = 6000,
        offset: int = 0,
    ) -> dict:
        await self.get_kb(kb_id, user=user)
        max_chars = max(1, min(int(max_chars or 6000), 50000))
        offset = max(0, int(offset or 0))
        async with AsyncSession(self._engine) as s:
            if chunk_id:
                chunk = await s.get(KnowledgeChunkRow, chunk_id)
                if chunk is None or chunk.kb_id != kb_id or chunk.deleted_at is not None:
                    raise LookupError("chunk not found")
                doc = await s.get(KnowledgeDocumentRow, chunk.document_id)
                if mode == "chunk_neighbors":
                    rows = (
                        await s.exec(
                            select(KnowledgeChunkRow).where(
                                KnowledgeChunkRow.document_id == chunk.document_id,
                                KnowledgeChunkRow.chunk_index >= chunk.chunk_index - 1,
                                KnowledgeChunkRow.chunk_index <= chunk.chunk_index + 1,
                                KnowledgeChunkRow.status == "active",
                            ).order_by(KnowledgeChunkRow.chunk_index)
                        )
                    ).all()
                    text = "\n\n".join(c.text for c in rows)
                else:
                    text = chunk.text
                return {
                    "kb_id": kb_id,
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.id,
                    "title": doc.title if doc else "",
                    "text": text[offset: offset + max_chars],
                }
            if not document_id:
                raise LookupError("document_id or chunk_id is required")
            doc = await s.get(KnowledgeDocumentRow, document_id)
            if doc is None or doc.kb_id != kb_id or doc.deleted_at is not None:
                raise LookupError("document not found")
            rows = (
                await s.exec(
                    select(KnowledgeChunkRow).where(
                        KnowledgeChunkRow.document_id == document_id,
                        KnowledgeChunkRow.status == "active",
                        KnowledgeChunkRow.deleted_at.is_(None),
                    ).order_by(KnowledgeChunkRow.chunk_index)
                )
            ).all()
            text = "\n\n".join(c.text for c in rows)
            return {
                "kb_id": kb_id,
                "document_id": doc.id,
                "title": doc.title,
                "source_uri": doc.uri,
                "text": text[offset: offset + max_chars],
            }

    async def set_chunk_status(
        self,
        kb_id: str,
        chunk_id: str,
        *,
        user: User,
        status: str,
        reason: str = "",
    ) -> KnowledgeChunkRow:
        async with AsyncSession(self._engine) as s:
            kb = await s.get(KnowledgeBaseRow, kb_id)
            chunk = await s.get(KnowledgeChunkRow, chunk_id)
            if kb is None or chunk is None or chunk.kb_id != kb_id or not self.can_manage(kb, user):
                raise PermissionError("knowledge base edit permission required")
            chunk.status = status
            chunk.disabled_by = user.id if status == "disabled" else None
            chunk.disabled_reason = reason if status == "disabled" else None
            chunk.updated_at = now_utc()
            s.add(chunk)
            await self._refresh_counts(s, kb)
            await s.commit()
            await s.refresh(chunk)
            return chunk

    # ------------------------------------------------------------------ #
    # Data sources
    # ------------------------------------------------------------------ #

    def _validate_source_config(self, source_type: str, source_key: str, source_config: dict) -> None:
        """Fail fast on an unsupported type or an adapter that can't be built."""
        if source_type not in supported_source_types():
            raise ValueError(
                f"unsupported source_type '{source_type}'; "
                f"supported: {', '.join(supported_source_types())}"
            )
        # validate_config surfaces missing required fields (e.g. product/llms_url)
        adapter = get_adapter(source_type, source_key or source_type, source_config or {})
        adapter.validate_config()

    async def create_data_source(
        self,
        kb_id: str,
        *,
        user: User,
        name: str,
        source_type: str,
        source_config: dict,
        enabled: bool = True,
        sync_schedule: Optional[str] = None,
    ) -> KnowledgeDataSourceRow:
        if not (name or "").strip():
            raise ValueError("name is required")
        source_key = _slugify(name) or _slugify(source_type) or "source"
        self._validate_source_config(source_type, source_key, source_config or {})
        async with AsyncSession(self._engine) as s:
            kb = await s.get(KnowledgeBaseRow, kb_id)
            if kb is None or kb.deleted_at is not None or not self.can_manage(kb, user):
                raise PermissionError("knowledge base edit permission required")
            ds = KnowledgeDataSourceRow(
                id=_uuid("ds"),
                kb_id=kb_id,
                name=name,
                source_key=source_key,
                source_type=source_type,
                source_config=source_config or {},
                enabled=enabled,
                sync_schedule=sync_schedule,
                created_by=user.id,
            )
            s.add(ds)
            await s.commit()
            await s.refresh(ds)
            return ds

    async def list_data_sources(self, kb_id: str, *, user: User) -> list[KnowledgeDataSourceRow]:
        await self.get_kb(kb_id, user=user)
        async with AsyncSession(self._engine) as s:
            rows = (
                await s.exec(
                    select(KnowledgeDataSourceRow)
                    .where(
                        KnowledgeDataSourceRow.kb_id == kb_id,
                        KnowledgeDataSourceRow.deleted_at.is_(None),
                    )
                    .order_by(KnowledgeDataSourceRow.created_at.desc())
                )
            ).all()
            return list(rows)

    async def get_data_source(self, kb_id: str, ds_id: str, *, user: User) -> KnowledgeDataSourceRow:
        await self.get_kb(kb_id, user=user)
        async with AsyncSession(self._engine) as s:
            ds = await s.get(KnowledgeDataSourceRow, ds_id)
            if ds is None or ds.deleted_at is not None or ds.kb_id != kb_id:
                raise PermissionError("data source not found")
            return ds

    async def update_data_source(
        self, kb_id: str, ds_id: str, *, user: User, patch: dict
    ) -> KnowledgeDataSourceRow:
        async with AsyncSession(self._engine) as s:
            kb = await s.get(KnowledgeBaseRow, kb_id)
            if kb is None or kb.deleted_at is not None or not self.can_manage(kb, user):
                raise PermissionError("knowledge base edit permission required")
            ds = await s.get(KnowledgeDataSourceRow, ds_id)
            if ds is None or ds.deleted_at is not None or ds.kb_id != kb_id:
                raise PermissionError("data source not found")
            for key in ("name", "enabled", "sync_schedule"):
                if key in patch and patch[key] is not None:
                    setattr(ds, key, patch[key])
            if patch.get("source_config") is not None:
                ds.source_config = dict(patch["source_config"])
                flag_modified(ds, "source_config")
            # re-validate the (possibly changed) config before persisting
            self._validate_source_config(ds.source_type, ds.source_key, ds.source_config or {})
            ds.updated_at = now_utc()
            s.add(ds)
            await s.commit()
            await s.refresh(ds)
            return ds

    async def delete_data_source(self, kb_id: str, ds_id: str, *, user: User) -> None:
        async with AsyncSession(self._engine) as s:
            kb = await s.get(KnowledgeBaseRow, kb_id)
            if kb is None or kb.deleted_at is not None or not self.can_manage(kb, user):
                raise PermissionError("knowledge base edit permission required")
            ds = await s.get(KnowledgeDataSourceRow, ds_id)
            if ds is None or ds.deleted_at is not None or ds.kb_id != kb_id:
                raise PermissionError("data source not found")
            ds.deleted_at = now_utc()
            ds.enabled = False
            ds.updated_at = now_utc()
            s.add(ds)
            await s.commit()

    def _synced_doc_metadata(self, ds_id: str, source_key: str, sd: SourceDocument) -> dict:
        return {
            "datasource_id": ds_id,
            "source_key": source_key,
            "source_doc_id": sd.doc_id,
            "source_url": sd.source_url,
            "fetch_url": sd.fetch_url,
            "source_site": sd.source_site,
            "product": sd.product,
            "section": sd.section,
            "lang": sd.lang,
            "summary": sd.summary,
            "content_hash": sd.content_hash,
            "fetched_from": sd.fetched_from,
            "fetched_at": sd.fetched_at,
        }

    async def sync_data_source(self, kb_id: str, ds_id: str, *, user: User) -> KnowledgeDataSourceRow:
        """Pull the data source and reconcile it into the KB (add/update/delete).

        Runs as a normal coroutine; routes launch it fire-and-forget via
        ``asyncio.create_task`` and clients poll the row's ``status``. Network
        fetches are offloaded to threads and bounded in concurrency; ingestion is
        serialized (one DB writer) to stay safe on sqlite.
        """
        # -- phase 0: claim the source (status=syncing) --------------------
        async with AsyncSession(self._engine) as s:
            kb = await s.get(KnowledgeBaseRow, kb_id)
            if kb is None or kb.deleted_at is not None or not self.can_manage(kb, user):
                raise PermissionError("knowledge base edit permission required")
            ds = await s.get(KnowledgeDataSourceRow, ds_id)
            if ds is None or ds.deleted_at is not None or ds.kb_id != kb_id:
                raise PermissionError("data source not found")
            if not ds.enabled:
                raise ValueError("data source is disabled")
            # capture before commit — commit expires the instance's attributes
            source_type = ds.source_type
            source_key = ds.source_key
            source_config = dict(ds.source_config or {})
            ds.status = "syncing"
            ds.last_sync_at = now_utc()
            ds.last_error = None
            ds.updated_at = now_utc()
            s.add(ds)
            await s.commit()

        report = {
            "discovered": 0, "added": 0, "updated": 0,
            "unchanged": 0, "deleted": 0, "failed": 0, "errors": [],
        }
        try:
            adapter = get_adapter(source_type, source_key, source_config)
            discovered = await asyncio.to_thread(adapter.discover)
            report["discovered"] = len(discovered)

            # fetch + normalize concurrently (network-bound, off the event loop)
            sem = asyncio.Semaphore(SYNC_FETCH_CONCURRENCY)

            async def _fetch(d):
                async with sem:
                    try:
                        body = await asyncio.to_thread(adapter.fetch, d)
                        return d, adapter.emit(d, body), None
                    except Exception as exc:  # noqa: BLE001 — record and continue
                        return d, None, str(exc)

            fetched = await asyncio.gather(*[_fetch(d) for d in discovered]) if discovered else []

            # snapshot existing docs owned by this data source (for diff)
            async with AsyncSession(self._engine) as s:
                existing_rows = (
                    await s.exec(
                        select(KnowledgeDocumentRow).where(
                            KnowledgeDocumentRow.kb_id == kb_id,
                            KnowledgeDocumentRow.source_id == ds_id,
                            KnowledgeDocumentRow.deleted_at.is_(None),
                        )
                    )
                ).all()
            existing_by_uri = {r.uri: r for r in existing_rows}

            seen_uris: set[str] = set()
            for d, sd, err in fetched:
                if err is not None or sd is None:
                    report["failed"] += 1
                    report["errors"].append({"path": d.path, "error": err})
                    continue
                uri = sd.source_url or sd.doc_id
                seen_uris.add(uri)
                prev = existing_by_uri.get(uri)
                if prev is not None and prev.content_hash == sd.content_hash and prev.status == "indexed":
                    report["unchanged"] += 1
                    continue
                try:
                    await self.import_text_document(
                        kb_id,
                        user=user,
                        title=sd.title,
                        content=sd.content,
                        uri=uri,
                        source_type=source_type,
                        source_id=ds_id,
                        mime_type="text/markdown",
                        description=sd.summary or "",
                        tags=[sd.section] if sd.section else [],
                        category=sd.product,
                        custom_metadata=self._synced_doc_metadata(ds_id, source_key, sd),
                        trigger_type="datasource_sync",
                    )
                    if prev is None:
                        report["added"] += 1
                    else:
                        report["updated"] += 1
                except Exception as exc:  # noqa: BLE001
                    report["failed"] += 1
                    report["errors"].append({"path": d.path, "error": str(exc)})

            # deletions — skip when discovery is known-incomplete so a transient
            # crawl failure can't be mistaken for source-side removal.
            if not getattr(adapter, "discovery_partial", False):
                stale_ids = [r.id for u, r in existing_by_uri.items() if u not in seen_uris]
                if stale_ids:
                    await self._soft_delete_documents(kb_id, stale_ids, user=user)
                    report["deleted"] = len(stale_ids)

            processed = report["added"] + report["updated"] + report["unchanged"]
            if report["failed"] and processed == 0:
                status = "failed"
            elif report["failed"]:
                status = "partial"
            else:
                status = "succeeded"
            # cap stored error detail so a mass failure can't bloat the row
            report["errors"] = report["errors"][:20]
            await self._finalize_sync(ds_id, status=status, report=report, error=None)
        except Exception as exc:  # noqa: BLE001 — surface as failed status, don't crash the task
            logger.exception(f"[datasource] sync failed for {ds_id}: {exc}")
            report["errors"] = report["errors"][:20]
            await self._finalize_sync(ds_id, status="failed", report=report, error=str(exc))

        return await self.get_data_source(kb_id, ds_id, user=user)

    async def _soft_delete_documents(self, kb_id: str, doc_ids: list[str], *, user: User) -> None:
        if not doc_ids:
            return
        removed: list[str] = []
        async with AsyncSession(self._engine) as s:
            kb = await s.get(KnowledgeBaseRow, kb_id)
            if kb is None:
                return
            now = now_utc()
            for doc_id in doc_ids:
                doc = await s.get(KnowledgeDocumentRow, doc_id)
                if doc is None or doc.kb_id != kb_id or doc.deleted_at is not None:
                    continue
                await s.exec(delete(KnowledgeChunkRow).where(KnowledgeChunkRow.document_id == doc_id))
                doc.status = "deleted"
                doc.deleted_at = now
                doc.chunk_count = 0
                doc.updated_by = user.id
                doc.updated_at = now
                s.add(doc)
                removed.append(doc_id)
            await self._refresh_counts(s, kb)
            await s.commit()
        await self._delete_from_engine_best_effort(kb_id, removed)

    async def _finalize_sync(self, ds_id: str, *, status: str, report: dict, error: Optional[str]) -> None:
        async with AsyncSession(self._engine) as s:
            ds = await s.get(KnowledgeDataSourceRow, ds_id)
            if ds is None:
                return
            doc_count = (
                await s.exec(
                    select(func.count()).select_from(KnowledgeDocumentRow).where(
                        KnowledgeDocumentRow.kb_id == ds.kb_id,
                        KnowledgeDocumentRow.source_id == ds_id,
                        KnowledgeDocumentRow.deleted_at.is_(None),
                        KnowledgeDocumentRow.status != "deleted",
                    )
                )
            ).one()
            ds.doc_count = int(doc_count)
            ds.status = status
            ds.last_error = error
            ds.last_sync_report = report
            flag_modified(ds, "last_sync_report")
            ds.last_sync_finished_at = now_utc()
            ds.updated_at = now_utc()
            s.add(ds)
            await s.commit()
