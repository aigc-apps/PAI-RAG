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
from app.sync_pipeline import (
    EmbeddedDocument,
    FetchedDocument,
    PersistedBatch,
    PipelineLimits,
    PreparedDocument,
    SyncPipeline,
)
from app.models import (
    BackgroundJobRow,
    KnowledgeBaseRow,
    KnowledgeChunkRow,
    KnowledgeDataSourceRow,
    KnowledgeDocumentContentRow,
    KnowledgeDocumentRow,
    KnowledgeIndexVersionRow,
    KnowledgeIngestionJobRow,
)
from app.store.base import User, _uuid, with_id_retry


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
    "top_k": 10,
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


def _chunk_id(
    index_version_id: Optional[str],
    document_id: str,
    chunk_index: int,
    text_hash: str,
) -> str:
    material = f"{index_version_id or 'none'}\0{document_id}\0{chunk_index}\0{text_hash}"
    return "chk_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


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


# Cap the verbatim copy we persist per document. ~200k chars ≈ 80 pages of prose
# or a whole CJK chapter — covers virtually every real document. Over-long docs
# store this prefix (flagged ``truncated``); their deeper text still lives in
# chunks. Tunable; can be promoted to a setting later.
MAX_STORED_CONTENT_CHARS = 200_000


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


# ATX markdown heading: 1–6 leading '#', then the title. Used to split markdown by
# section so each chunk carries its heading path (ancestor titles).
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*\S)[ \t]*$")


def _looks_like_markdown(text: str, mime_type: str) -> bool:
    if mime_type and "markdown" in mime_type.lower():
        return True
    # No explicit markdown mime, but an ATX heading present → treat as structured.
    return bool(re.search(r"(?m)^#{1,6}[ \t]+\S", text or ""))


def _window_span(
    base: str, start: int, end: int, *, chunk_size: int, chunk_overlap: int, heading_path: list
) -> list[dict]:
    """Sliding-window ``base[start:end]`` into chunk dicts, keeping offsets into the
    original ``base`` (so char_start/char_end stay aligned with the stored content).
    A span that fits in one window yields a single chunk."""
    chunk_size = max(100, int(chunk_size or 1000))
    chunk_overlap = max(0, min(int(chunk_overlap or 0), chunk_size // 2))
    out: list[dict] = []
    pos = start
    while pos < end:
        stop = min(end, pos + chunk_size)
        seg = base[pos:stop]
        body = seg.strip()
        if body:
            lead = len(seg) - len(seg.lstrip())
            cs = pos + lead
            out.append(
                {
                    "text": body,
                    "char_start": cs,
                    "char_end": cs + len(body),
                    "heading_path": list(heading_path),
                }
            )
        if stop >= end:
            break
        pos = max(pos + 1, stop - chunk_overlap)
    return out


def split_document(
    text: str, *, mime_type: str = "text/plain", chunk_size: int, chunk_overlap: int
) -> list[dict]:
    """Split a document into chunk dicts ``{text, char_start, char_end, heading_path}``.

    Markdown (by mime type or a detected ATX heading) is split at heading
    boundaries so each chunk records its ancestor-heading path; over-long sections
    are still windowed to ``chunk_size``. Plain text falls back to a fixed-size
    sliding window with an empty heading_path. Offsets index the *stripped* text —
    the same base persisted by ``import_text_document`` — so locate/view align."""
    base = (text or "").strip()
    if not base:
        return []
    if not _looks_like_markdown(base, mime_type):
        return _window_span(
            base, 0, len(base), chunk_size=chunk_size, chunk_overlap=chunk_overlap, heading_path=[]
        )
    # Markdown: carve contiguous sections at heading boundaries, tracking a heading
    # stack so each section knows its full ancestor path.
    sections: list[tuple[int, int, list]] = []
    stack: list[tuple[int, str]] = []
    cur_start = 0
    cur_path: list = []
    pos = 0
    for line in base.splitlines(keepends=True):
        m = _HEADING_RE.match(line)
        if m:
            if pos > cur_start:
                sections.append((cur_start, pos, list(cur_path)))
            level = len(m.group(1))
            title = m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur_path = [t for _lvl, t in stack]
            cur_start = pos
        pos += len(line)
    if pos > cur_start:
        sections.append((cur_start, pos, list(cur_path)))
    out: list[dict] = []
    for s, e, path in sections:
        out.extend(
            _window_span(
                base, s, e, chunk_size=chunk_size, chunk_overlap=chunk_overlap, heading_path=path
            )
        )
    return out


def _embed_input(title: str, heading_path: list, body: str) -> str:
    """Context-enriched text to *embed* (not to store): document title + heading
    breadcrumb prepended to the chunk body. Empty parts are dropped so plain-text
    chunks (no headings) just get the title, and an untitled doc gets the body."""
    context = "\n".join(
        p for p in (title or "", " > ".join(heading_path or [])) if p
    )
    return f"{context}\n\n{body}" if context else body


def _merge(defaults: dict, override: Optional[dict]) -> dict:
    data = dict(defaults)
    data.update(override or {})
    return data


class KnowledgeService:
    def __init__(
        self,
        engine,
        search_engine=None,
        fallback_to_local: bool = True,
        router=None,
        pipeline_limits: Optional[PipelineLimits] = None,
    ):
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
        self._pipeline_limits = pipeline_limits or PipelineLimits(
            fetch_concurrency=SYNC_FETCH_CONCURRENCY
        )
        self._sync_enqueue_lock = asyncio.Lock()

    async def enqueue_data_source_sync(self, queue, kb_id: str, ds_id: str, *, user: User):
        """Atomically reserve the data source and create its durable queue row."""
        job = queue.build_job(
            kind="kb_sync",
            kb_id=kb_id,
            created_by=user.id,
            payload={
                "kb_id": kb_id,
                "ds_id": ds_id,
                "user_id": user.id,
                "user_email": user.email,
                "user_role": user.role,
            },
        )
        async with self._sync_enqueue_lock:
            async with AsyncSession(self._engine, expire_on_commit=False) as s:
                kb = await s.get(KnowledgeBaseRow, kb_id)
                if kb is None or kb.deleted_at is not None or not self.can_manage(kb, user):
                    raise PermissionError("knowledge base edit permission required")
                stmt = select(KnowledgeDataSourceRow).where(
                    KnowledgeDataSourceRow.id == ds_id,
                    KnowledgeDataSourceRow.kb_id == kb_id,
                    KnowledgeDataSourceRow.deleted_at.is_(None),
                )
                if self._engine.dialect.name == "postgresql":
                    stmt = stmt.with_for_update()
                ds = (await s.exec(stmt)).first()
                if ds is None:
                    raise PermissionError("data source not found")
                if not ds.enabled:
                    raise ValueError("data source is disabled")
                if ds.active_job_id:
                    raise RuntimeError("a sync is already in progress")
                ds.active_job_id = job.id
                ds.status = "syncing"
                ds.last_sync_at = now_utc()
                ds.last_error = None
                ds.updated_at = now_utc()
                s.add(ds)
                s.add(job)
                await s.commit()
        queue.notify()
        return job.id, ds

    def set_search_engine(self, search_engine, *, fallback_to_local: bool = True) -> None:
        """Swap the primary search engine at runtime (keeps the local fallback).

        ``self._local`` is left intact so degradation still works."""
        self._search = search_engine or self._local
        self._fallback_to_local = fallback_to_local

    def rebuild_search_engine(self, vectordb) -> None:
        """Rebuild the primary engine from the global ``knowledgebase.vectordb``
        config section and swap it in. Called on config reload so a saved
        vector-store change takes effect without a process restart."""
        from app.search_engine import build_search_engine

        self.set_search_engine(
            build_search_engine(None, self._engine, vectordb=vectordb),
            fallback_to_local=(vectordb.engine != "local"),
        )

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

    def _resolve_default_vector_store(self) -> dict:
        """The vector_store_config a new KB gets when the caller doesn't pin a
        provider. Derives ``provider_id`` from the active global search engine
        (``knowledgebase.vectordb``) so a KB created while Elasticsearch is
        configured records ``elasticsearch`` instead of the local SQL default —
        the value the UI surfaces as the KB's vector engine. Recorded at creation
        and never rewritten afterwards."""
        base = dict(DEFAULT_VECTOR_STORE_CONFIG)
        engine = self._search
        if engine is not None and engine is not self._local:
            name = getattr(engine, "name", None)
            if name:
                base["provider_id"] = str(name)
        return base

    def embedding_config_for(self, model_id: str) -> dict:
        """Resolve a caller-chosen embedding model id into the frozen
        embedding_config a KB stores. Validates the id is catalogued and is an
        embedding model; raises ValueError otherwise (surfaced as HTTP 400)."""
        if self._router is None:
            raise ValueError("no model catalog configured; cannot select an embedding model")
        try:
            cfg = self._router.get_config(model_id)
        except KeyError:
            raise ValueError(f"unknown embedding model '{model_id}'")
        if cfg.type != "embedding":
            raise ValueError(f"model '{model_id}' is a {cfg.type} model, not an embedding model")
        return {
            "provider_id": cfg.provider,
            "model": model_id,
            "dimension": int(cfg.dimension or 1024),
            "normalize": True,
        }

    def rerank_config_for(self, model_id: str, *, top_n: int = 5) -> dict:
        """Resolve a caller-chosen rerank model id into a rerank_config. Validates
        the id is catalogued and is a rerank model; raises ValueError otherwise."""
        if self._router is None:
            raise ValueError("no model catalog configured; cannot select a rerank model")
        try:
            cfg = self._router.get_config(model_id)
        except KeyError:
            raise ValueError(f"unknown rerank model '{model_id}'")
        if cfg.type != "rerank":
            raise ValueError(f"model '{model_id}' is a {cfg.type} model, not a rerank model")
        return {
            "enabled": True,
            "provider_id": cfg.provider,
            "model": model_id,
            "top_n": int(top_n),
        }

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
        # embedding is frozen at creation. An explicit config wins; otherwise pick
        # the catalogued DashScope embedder when healthy, else the local hash.
        if embedding_config is not None:
            embedding = _merge(DEFAULT_EMBEDDING_CONFIG, embedding_config)
        else:
            embedding = self._resolve_default_embedding()
        parser = _merge(DEFAULT_PARSER_CONFIG, default_parser_config)
        retrieval = _merge(DEFAULT_RETRIEVAL_CONFIG, default_retrieval_config)
        keyword = _merge(DEFAULT_KEYWORD_INDEX_CONFIG, keyword_index_config)
        rerank = _merge(DEFAULT_RERANK_CONFIG, rerank_config)

        async def work() -> KnowledgeBaseRow:
            kb_id = _uuid("kb")
            # Fresh copy per attempt so setdefault of the kb_id-derived namespace
            # tracks a regenerated id on retry.
            vector = _merge(self._resolve_default_vector_store(), vector_store_config)
            # Keep the vector store's declared dimension in step with the embedder
            # (the ES mapping reads embedding_config, but keep both coherent).
            if not (vector_store_config and "dimension" in vector_store_config):
                vector["dimension"] = int(embedding.get("dimension") or 64)
            vector.setdefault("namespace", kb_id)
            vector.setdefault("index_name", f"{kb_id}_vectors_v1")
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
                    keyword_index_config=keyword,
                    rerank_config=rerank,
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

        return await with_id_retry(work)

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
        async def work() -> KnowledgeDocumentRow:
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

        return await with_id_retry(work)

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
        content = content or ""
        if not content.strip():
            raise ValueError("content is required")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        stable_uri = uri or f"text://{digest[:16]}"
        # Resolve the KB + do the expensive chunk/embed ONCE, outside the retry:
        # only the row write (and its ids) is re-run on the ~never collision.
        async with AsyncSession(self._engine) as s0:
            kb = await s0.get(KnowledgeBaseRow, kb_id)
            if kb is None or kb.deleted_at is not None or not self.can_manage(kb, user):
                raise PermissionError("knowledge base edit permission required")
        parser = _merge(DEFAULT_PARSER_CONFIG, kb.default_parser_config)
        # Structure-aware split: markdown carries a heading_path per chunk; plain
        # text is a fixed-size window. Offsets index the stripped text (below).
        chunks = split_document(
            content,
            mime_type=mime_type or "text/plain",
            chunk_size=int(parser.get("chunk_size") or 1000),
            chunk_overlap=int(parser.get("chunk_overlap") or 150),
        )
        # Verbatim copy to persist (1:1 content row). Strip so its offsets align
        # with chunk char_start/char_end (split_document strips first too), then cap
        # — over-long docs keep the prefix and flag ``truncated``.
        stored_full = content.strip()
        stored_text = stored_full[:MAX_STORED_CONTENT_CHARS]
        stored_truncated = len(stored_full) > MAX_STORED_CONTENT_CHARS
        # Embed a context-enriched representation (document title + heading path
        # prepended to the body) while STORING the raw body in the chunk row. The
        # richer input lifts recall on both the local cosine and ES kNN paths (both
        # score the stored vector); knowledge_read/knowledge_find stay clean on raw text.
        doc_title = title or stable_uri
        embedder = build_embedder(kb.embedding_config, self._router)
        chunk_vectors = await embedder.embed(
            [_embed_input(doc_title, ch["heading_path"], ch["text"]) for ch in chunks],
            text_type="document",
        ) if chunks else []
        active_index_version_id = kb.active_index_version_id

        async def work() -> tuple[KnowledgeDocumentRow, KnowledgeIngestionJobRow, list[dict]]:
            async with AsyncSession(self._engine) as s:
                kb_row = await s.get(KnowledgeBaseRow, kb_id)
                if kb_row is None or kb_row.deleted_at is not None:
                    raise PermissionError("knowledge base edit permission required")
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
                # Persist the exact ingested text (1:1) so full-document reads
                # return it verbatim instead of re-stitching overlapping chunks.
                # Upsert: drop any prior content row for this doc, then insert.
                await s.exec(
                    delete(KnowledgeDocumentContentRow).where(
                        KnowledgeDocumentContentRow.document_id == doc_id
                    )
                )
                s.add(
                    KnowledgeDocumentContentRow(
                        document_id=doc_id,
                        kb_id=kb_id,
                        text=stored_text,
                        content_hash=digest,
                        char_len=len(stored_text),
                        truncated=stored_truncated,
                        updated_at=indexed_at,
                    )
                )
                es_chunks: list[dict] = []
                for idx, ch in enumerate(chunks):
                    body = ch["text"]
                    heading_path = ch.get("heading_path") or []
                    chunk_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
                    chunk_id = _chunk_id(
                        active_index_version_id, doc_id, idx, chunk_hash
                    )
                    embedding = chunk_vectors[idx] if idx < len(chunk_vectors) else []
                    s.add(
                        KnowledgeChunkRow(
                            id=chunk_id,
                            kb_id=kb_id,
                            document_id=doc_id,
                            chunk_index=idx,
                            text=body,
                            text_hash=chunk_hash,
                            heading_path=heading_path,
                            char_start=ch.get("char_start"),
                            char_end=ch.get("char_end"),
                            token_count=len(_tokens(body)),
                            chunk_metadata={
                                "title": doc.title,
                                "source_uri": stable_uri,
                                "source_type": source_type,
                                "tags": tags or [],
                                "category": category,
                                "heading_path": heading_path,
                            },
                            embedding=embedding,
                            embedding_ref=f"{active_index_version_id}:{chunk_id}",
                            indexed_at=indexed_at,
                        )
                    )
                    es_chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "chunk_index": idx,
                            "text": body,
                            "heading_path": heading_path,
                            "embedding": embedding,
                        }
                    )
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
                await self._refresh_counts(s, kb_row)
                await s.commit()
                await s.refresh(doc)
                await s.refresh(job)
                return doc, job, es_chunks

        doc, job, es_chunks = await with_id_retry(work)
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
        try:
            delete_batch = getattr(self._search, "delete_document_batch", None)
            if delete_batch is not None:
                await delete_batch(kb_id, document_ids, refresh=False)
            else:
                for doc_id in document_ids:
                    await self._search.delete_document(kb_id, doc_id)
            async with AsyncSession(self._engine) as s:
                rows = (
                    await s.exec(
                        select(KnowledgeDocumentRow).where(
                            KnowledgeDocumentRow.id.in_(document_ids)
                        )
                    )
                ).all()
                for row in rows:
                    row.search_index_status = "deleted"
                    row.search_index_error = None
                    s.add(row)
                await s.commit()
        except Exception as ex:
            logger.warning(
                f"[search] batch document deletion failed for {len(document_ids)} docs: {ex!r}"
            )

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

    async def resolve_search_kbs(
        self, *, user: User, kb_ids: list[str]
    ) -> list[KnowledgeBaseRow]:
        """Return the requested KBs the caller may query, preserving order."""
        rows: list[KnowledgeBaseRow] = []
        seen: set[str] = set()
        for kb_id in kb_ids:
            if kb_id in seen:
                continue
            seen.add(kb_id)
            try:
                rows.append(await self.get_kb(kb_id, user=user))
            except PermissionError:
                continue
        return rows

    @staticmethod
    def _embedding_group_key(kb: KnowledgeBaseRow) -> tuple[str, str, int]:
        cfg = kb.embedding_config or {}
        return (
            str(cfg.get("provider_id") or "local_hash"),
            str(cfg.get("model") or "local-hash-v1"),
            int(cfg.get("dimension") or 64),
        )

    async def search(
        self,
        *,
        user: User,
        kb_ids: list[str],
        query: str,
        top_k: int = 10,
        offset: int = 0,
        score_threshold: float = 0.0,
        mode: str = "hybrid",
        filters: Optional[dict] = None,
        rerank_config: Optional[dict] = None,
    ) -> tuple[list[SearchHit], int]:
        """Search permission-filtered KBs and globally rank their candidates."""
        if not query.strip():
            return [], 0
        filters = filters or {}
        allowed = await self.resolve_search_kbs(user=user, kb_ids=kb_ids)
        if not allowed:
            return [], 0

        limit = max(1, min(int(top_k or 10), 50))
        offset = max(0, int(offset or 0))
        rerank_cfg = dict(rerank_config or {})
        rerank_on = bool(rerank_cfg.get("enabled")) and self._router is not None
        # Give every knowledge base an equal opportunity to contribute to the
        # unified ranking. A single cross-index request lets a large KB crowd a
        # small, more relevant KB out of the candidate window.
        per_kb_fetch_limit = max(20, limit + offset)

        groups: dict[tuple[str, str, int], list[KnowledgeBaseRow]] = {}
        for kb in allowed:
            groups.setdefault(self._embedding_group_key(kb), []).append(kb)

        candidates: list[SearchHit] = []
        total = 0
        for (_provider, _model, dimension), group in groups.items():
            query_vector = None
            if mode in ("vector", "hybrid"):
                try:
                    embedder = build_embedder(group[0].embedding_config, self._router)
                    vecs = await embedder.embed([query], text_type="query")
                    query_vector = vecs[0] if vecs else None
                except Exception as ex:
                    logger.warning(
                        "operation=query_embedding error_type={} fallback=search_engine",
                        type(ex).__name__,
                    )
            for kb in group:
                kwargs = dict(
                    kb_ids=[kb.id],
                    query=query,
                    mode=mode,
                    offset=0,
                    limit=per_kb_fetch_limit,
                    score_threshold=score_threshold,
                    dimension=dimension,
                    filters=filters,
                    query_vector=query_vector,
                )
                engine = self._search
                if engine is self._local:
                    kb_hits, kb_total = await self._local.search(**kwargs)
                else:
                    try:
                        kb_hits, kb_total = await engine.search(**kwargs)
                    except Exception as ex:
                        if not self._fallback_to_local:
                            raise
                        logger.warning(
                            "operation=primary_search engine={} error_type={} "
                            "fallback=local",
                            getattr(engine, "name", "?"),
                            type(ex).__name__,
                        )
                        kb_hits, kb_total = await self._local.search(**kwargs)
                candidates.extend(kb_hits)
                total += kb_total

        candidates.sort(key=lambda hit: hit.score, reverse=True)
        logger.info(
            f"[search] kb_ids={[kb.id for kb in allowed]} "
            f"candidates={len(candidates)} rerank="
            f"{rerank_cfg.get('model') if rerank_on else None}"
        )
        if rerank_on and candidates:
            candidates = self._cap_chunks_per_document(candidates, max_chunks=3)
            candidates = await self._rerank_hits(
                query,
                candidates,
                limit + offset,
                rerank_cfg,
                kb_names={kb.id: kb.name for kb in allowed},
            )
        result = candidates[offset : offset + limit]
        logger.info(f"[search] final_results={len(result)} total={total}")
        return result, total

    @staticmethod
    def _cap_chunks_per_document(
        hits: list[SearchHit], *, max_chunks: int
    ) -> list[SearchHit]:
        counts: dict[tuple[str, str], int] = {}
        diversified: list[SearchHit] = []
        for hit in hits:
            document_key = hit.document_id or hit.chunk_id
            key = (hit.kb_id, document_key)
            if counts.get(key, 0) >= max_chunks:
                continue
            counts[key] = counts.get(key, 0) + 1
            diversified.append(hit)
        return diversified

    @staticmethod
    def _rerank_document(hit: SearchHit, kb_names: dict[str, str]) -> str:
        metadata = hit.metadata or {}
        heading = metadata.get("heading")
        if not heading and metadata.get("heading_path"):
            heading = " > ".join(metadata["heading_path"])
        return "\n".join(
            [
                f"Knowledge base: {kb_names.get(hit.kb_id, hit.kb_id)}",
                f"Document: {hit.title or '(untitled)'}",
                f"Heading: {heading or '(none)'}",
                "Content:",
                hit.text,
            ]
        )

    async def _rerank_hits(
        self, query, hits, limit, rerank_cfg, *, kb_names
    ) -> list[SearchHit]:
        """Reorder a candidate window with a DashScope reranker, best-effort:
        any failure returns the original order (trimmed). Overwrites each hit's
        ``score`` with the reranker's relevance score."""
        model_id = rerank_cfg.get("model") or self._router.default_model_id_of_type("rerank")
        if not model_id:
            return hits[:limit]
        top_n = int(rerank_cfg.get("top_n") or limit)
        try:
            reranker = self._router.get_reranker(model_id)
            documents = [self._rerank_document(hit, kb_names) for hit in hits]
            ranked = await reranker.rerank(query, documents, top_n=top_n)
        except Exception as ex:
            logger.warning(
                "operation=rerank model={} error_type={} fallback=original_order",
                model_id,
                type(ex).__name__,
            )
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
        """Which retrieval engine is active + whether it's reachable (frontend badge).

        ``detail`` carries a human reason for the reachability state (e.g. a
        connection-refused / auth / missing-dependency message) so the settings
        page can explain *why* an engine is unreachable, not just that it is."""
        engine = self._search
        name = getattr(engine, "name", "local")
        configured = engine is not self._local
        healthy = True
        detail = ""
        if configured:
            try:
                if hasattr(engine, "health_detail"):
                    healthy, detail = await engine.health_detail()
                else:
                    healthy = await engine.healthy()
            except Exception as ex:
                healthy = False
                detail = str(ex)
        return {"engine": name, "configured": configured, "healthy": healthy, "detail": detail}

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
        kb_id: Optional[str] = None,
        document_id: Optional[str] = None,
        chunk_id: Optional[str] = None,
        mode: str = "full_doc",
        max_chars: int = 6000,
        offset: int = 0,
    ) -> dict:
        max_chars = max(1, min(int(max_chars or 6000), 50000))
        offset = max(0, int(offset or 0))
        # kb_id may be omitted (agent tools pass only the short doc/chunk id):
        # resolve it from the target row, then permission-check like any read.
        if not kb_id:
            async with AsyncSession(self._engine) as s0:
                if chunk_id:
                    row = await s0.get(KnowledgeChunkRow, chunk_id)
                    if row is None or row.deleted_at is not None:
                        raise LookupError("chunk not found")
                elif document_id:
                    row = await s0.get(KnowledgeDocumentRow, document_id)
                    if row is None or row.deleted_at is not None:
                        raise LookupError("document not found")
                else:
                    raise LookupError("document_id or chunk_id is required")
                kb_id = row.kb_id
        await self.get_kb(kb_id, user=user)
        async with AsyncSession(self._engine) as s:
            if chunk_id:
                chunk = await s.get(KnowledgeChunkRow, chunk_id)
                if chunk is None or chunk.kb_id != kb_id or chunk.deleted_at is not None:
                    raise LookupError("chunk not found")
                doc = await s.get(KnowledgeDocumentRow, chunk.document_id)
                if mode == "locate":
                    # Open the full document at this chunk's position, with a little
                    # preceding context, so a search hit lands in situ. Windowed via
                    # SQL substr; falls back to the chunk's own text when the stored
                    # copy is missing or the position is past its (capped) end.
                    back = 400
                    start = max(0, int(chunk.char_start or 0) - back)
                    win = (
                        await s.exec(
                            select(
                                func.substr(
                                    KnowledgeDocumentContentRow.text,
                                    start + 1,
                                    max_chars,
                                )
                            ).where(
                                KnowledgeDocumentContentRow.document_id
                                == chunk.document_id
                            )
                        )
                    ).first()
                    text = win if win else chunk.text
                    return {
                        "kb_id": kb_id,
                        "document_id": chunk.document_id,
                        "chunk_id": chunk.id,
                        "title": doc.title if doc else "",
                        "char_start": chunk.char_start,
                        "text": text,
                    }
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
            # Prefer the exact ingested text when it was stored (1:1 content row);
            # fall back to stitching chunks for documents ingested before content
            # was persisted. Stitching duplicates chunk-overlap regions, so it is a
            # readable approximation, not byte-faithful.
            #
            # Read only the requested window: SQL substr slices in the database so a
            # huge document never loads whole into memory. substr is 1-indexed on
            # both SQLite and Postgres; a NULL scalar means no content row → stitch.
            meta = (
                await s.exec(
                    select(
                        KnowledgeDocumentContentRow.char_len,
                        KnowledgeDocumentContentRow.truncated,
                    ).where(KnowledgeDocumentContentRow.document_id == document_id)
                )
            ).first()
            if meta is not None:
                char_len, truncated = int(meta[0] or 0), bool(meta[1])
                window = (
                    await s.exec(
                        select(
                            func.substr(
                                KnowledgeDocumentContentRow.text, offset + 1, max_chars
                            )
                        ).where(
                            KnowledgeDocumentContentRow.document_id == document_id
                        )
                    )
                ).first()
                text = window or ""
                note = ""
                if truncated and offset + len(text) >= char_len:
                    note = (
                        f"\n\n[stored copy ends at {char_len} chars; this document "
                        "was truncated — deeper text is available only via its chunks]"
                    )
                return {
                    "kb_id": kb_id,
                    "document_id": doc.id,
                    "title": doc.title,
                    "source_uri": doc.uri,
                    "text": text + note,
                }
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

    async def grep_chunks(
        self,
        *,
        user: User,
        query: str,
        kb_ids: Optional[list[str]] = None,
        document_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Literal, case-insensitive substring search across chunk text — the
        exact-match complement to the tokenized BM25/vector ``search``. Reuses the
        ``list_chunks`` substring path (``LIKE '%query%'``) per accessible KB and
        merges. Permission-scoped identically to ``knowledge_search``: without
        ``kb_ids`` it greps every KB the caller may query.

        Returns dicts ``{kb_id, document_id, chunk_id, title, chunk_index, text}`` —
        the chunk's own ``chunk_metadata.title`` when present, else the document
        title. ``chunk_id`` lets the caller open the hit with ``knowledge_read`` locate.
        """
        q = (query or "").strip()
        if not q:
            return []
        limit = max(1, min(int(limit or 20), 50))
        targets = list(kb_ids) if kb_ids else [kb.id for kb in await self.list_kbs(user=user)]
        # Titles for the matched documents (chunk_metadata carries one, but fall
        # back to the document row so a match always names its file).
        out: list[dict] = []
        for kb_id in targets:
            if len(out) >= limit:
                break
            try:
                rows, _total = await self.list_chunks(
                    kb_id,
                    user=user,
                    document_id=document_id,
                    query=q,
                    limit=limit - len(out),
                )
            except PermissionError:
                continue  # a stale/forbidden kb id in an explicit list — skip it
            for c in rows:
                meta = c.chunk_metadata or {}
                out.append(
                    {
                        "kb_id": c.kb_id,
                        "document_id": c.document_id,
                        "chunk_id": c.id,
                        "title": meta.get("title") or "",
                        "chunk_index": c.chunk_index,
                        "text": c.text or "",
                    }
                )
                if len(out) >= limit:
                    break
        return out

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

        async def work() -> KnowledgeDataSourceRow:
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

        return await with_id_retry(work)

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

    async def data_source_payload(self, row: KnowledgeDataSourceRow) -> dict:
        data = row.model_dump(mode="json")
        if not row.active_job_id:
            data["sync_job"] = None
            data["sync_progress"] = None
            return data
        async with AsyncSession(self._engine) as s:
            job = await s.get(BackgroundJobRow, row.active_job_id)
        data["sync_job"] = (
            None
            if job is None
            else {
                "id": job.id,
                "status": job.status,
                "progress": dict(job.progress or {}),
                "cancel_requested": job.cancel_requested_at is not None,
                "error": job.error,
            }
        )
        data["sync_progress"] = dict(job.progress or {}) if job is not None else None
        return data

    async def cancel_data_source_sync(
        self, queue, kb_id: str, ds_id: str, *, user: User
    ) -> str:
        await self.get_kb(kb_id, user=user, require_manage=True)
        row = await self.get_data_source(kb_id, ds_id, user=user)
        if not row.active_job_id:
            raise RuntimeError("no sync is currently active")
        if not await queue.request_cancel(row.active_job_id):
            raise RuntimeError("sync is already terminal")
        return row.active_job_id

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

    async def _prepare_synced_document(
        self,
        kb: KnowledgeBaseRow,
        ds: KnowledgeDataSourceRow,
        source_key: str,
        adapter,
        existing_by_uri: dict[str, KnowledgeDocumentRow],
        fetched: FetchedDocument,
    ) -> PreparedDocument:
        sd = adapter.emit(fetched.source, fetched.body)
        uri = sd.source_url or sd.doc_id
        previous = existing_by_uri.get(uri)
        unchanged = bool(
            previous is not None
            and previous.content_hash == sd.content_hash
            and previous.status == "indexed"
            and previous.search_index_status == "indexed"
        )
        parser = _merge(DEFAULT_PARSER_CONFIG, kb.default_parser_config)
        chunks = [] if unchanged else split_document(
            sd.content,
            mime_type="text/markdown",
            chunk_size=int(parser.get("chunk_size") or 1000),
            chunk_overlap=int(parser.get("chunk_overlap") or 150),
        )
        stored_full = sd.content.strip()
        return PreparedDocument(
            source=sd,
            content=sd.content,
            chunks=chunks,
            unchanged=unchanged,
            metadata={
                "uri": uri,
                "title": sd.title,
                "source_type": ds.source_type,
                "source_id": ds.id,
                "description": sd.summary or "",
                "tags": [sd.section] if sd.section else [],
                "category": sd.product,
                "custom_metadata": self._synced_doc_metadata(ds.id, source_key, sd),
                "content_hash": sd.content_hash,
                "size_bytes": sd.byte_size,
                "stored_text": stored_full[:MAX_STORED_CONTENT_CHARS],
                "stored_truncated": len(stored_full) > MAX_STORED_CONTENT_CHARS,
                "was_existing": previous is not None,
            },
        )

    async def _embed_synced_documents(
        self, kb: KnowledgeBaseRow, documents: list[PreparedDocument]
    ) -> list[EmbeddedDocument]:
        positions: list[tuple[int, int]] = []
        inputs: list[str] = []
        for doc_index, document in enumerate(documents):
            title = document.metadata["title"]
            for chunk_index, chunk in enumerate(document.chunks):
                positions.append((doc_index, chunk_index))
                inputs.append(
                    _embed_input(title, chunk.get("heading_path") or [], chunk["text"])
                )
        embedder = build_embedder(kb.embedding_config, self._router)
        vectors = await embedder.embed(inputs, text_type="document") if inputs else []
        if len(vectors) != len(positions):
            raise RuntimeError(
                f"embedding vector count mismatch: {len(vectors)} != {len(positions)}"
            )
        grouped: list[list[list[float]]] = [[] for _ in documents]
        for (doc_index, _chunk_index), vector in zip(positions, vectors):
            grouped[doc_index].append(vector)
        return [
            EmbeddedDocument(document, grouped[index])
            for index, document in enumerate(documents)
        ]

    async def _persist_synced_batch(
        self,
        kb_id: str,
        user: User,
        documents: list[EmbeddedDocument],
    ) -> PersistedBatch:
        uris = [document.prepared.metadata["uri"] for document in documents]
        async with AsyncSession(self._engine, expire_on_commit=False) as s:
            kb = await s.get(KnowledgeBaseRow, kb_id)
            if kb is None or kb.deleted_at is not None:
                raise PermissionError("knowledge base edit permission required")
            existing_rows = (
                await s.exec(
                    select(KnowledgeDocumentRow).where(
                        KnowledgeDocumentRow.kb_id == kb_id,
                        KnowledgeDocumentRow.uri.in_(uris),
                        KnowledgeDocumentRow.deleted_at.is_(None),
                    )
                )
            ).all()
            existing_by_uri = {row.uri: row for row in existing_rows}
            ids_by_uri = {
                uri: existing_by_uri[uri].id if uri in existing_by_uri else _uuid("doc")
                for uri in uris
            }
            document_ids = list(ids_by_uri.values())
            await s.exec(
                delete(KnowledgeChunkRow).where(
                    KnowledgeChunkRow.document_id.in_(document_ids)
                )
            )
            await s.exec(
                delete(KnowledgeDocumentContentRow).where(
                    KnowledgeDocumentContentRow.document_id.in_(document_ids)
                )
            )
            indexed_at = now_utc()
            persisted_docs: list[KnowledgeDocumentRow] = []
            es_documents: list[tuple[KnowledgeDocumentRow, list[dict]]] = []
            for embedded in documents:
                prepared = embedded.prepared
                meta = prepared.metadata
                uri = meta["uri"]
                doc_id = ids_by_uri[uri]
                doc = existing_by_uri.get(uri) or KnowledgeDocumentRow(
                    id=doc_id,
                    kb_id=kb_id,
                    uri=uri,
                    source_type=meta["source_type"],
                    title=meta["title"],
                    created_by=user.id,
                )
                doc.source_id = meta["source_id"]
                doc.source_type = meta["source_type"]
                doc.title = meta["title"]
                doc.description = meta["description"]
                doc.mime_type = "text/markdown"
                doc.size_bytes = meta["size_bytes"]
                doc.content_hash = meta["content_hash"]
                doc.tags = meta["tags"]
                doc.category = meta["category"]
                doc.custom_metadata = meta["custom_metadata"]
                doc.system_metadata = {"content_preview": prepared.content[:500]}
                doc.status = "indexed"
                doc.search_index_status = (
                    "indexed" if self._search is self._local else "pending"
                )
                doc.search_index_error = None
                doc.search_index_attempts = 0
                doc.chunk_count = len(prepared.chunks)
                doc.indexed_at = indexed_at
                doc.updated_by = user.id
                doc.updated_at = indexed_at
                s.add(doc)
                s.add(
                    KnowledgeDocumentContentRow(
                        document_id=doc_id,
                        kb_id=kb_id,
                        text=meta["stored_text"],
                        content_hash=meta["content_hash"],
                        char_len=len(meta["stored_text"]),
                        truncated=meta["stored_truncated"],
                        updated_at=indexed_at,
                    )
                )
                es_chunks: list[dict] = []
                for chunk_index, chunk in enumerate(prepared.chunks):
                    body = chunk["text"]
                    text_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
                    chunk_id = _chunk_id(
                        kb.active_index_version_id,
                        doc_id,
                        chunk_index,
                        text_hash,
                    )
                    vector = (
                        embedded.vectors[chunk_index]
                        if chunk_index < len(embedded.vectors)
                        else []
                    )
                    heading_path = chunk.get("heading_path") or []
                    s.add(
                        KnowledgeChunkRow(
                            id=chunk_id,
                            kb_id=kb_id,
                            document_id=doc_id,
                            chunk_index=chunk_index,
                            text=body,
                            text_hash=text_hash,
                            heading_path=heading_path,
                            char_start=chunk.get("char_start"),
                            char_end=chunk.get("char_end"),
                            token_count=len(_tokens(body)),
                            chunk_metadata={
                                "title": doc.title,
                                "source_uri": uri,
                                "source_type": doc.source_type,
                                "tags": doc.tags or [],
                                "category": doc.category,
                                "heading_path": heading_path,
                            },
                            embedding=vector,
                            embedding_ref=f"{kb.active_index_version_id}:{chunk_id}",
                            indexed_at=indexed_at,
                        )
                    )
                    es_chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "chunk_index": chunk_index,
                            "text": body,
                            "heading_path": heading_path,
                            "embedding": vector,
                        }
                    )
                s.add(
                    KnowledgeIngestionJobRow(
                        id=_uuid("job"),
                        kb_id=kb_id,
                        source_id=meta["source_id"],
                        document_id=doc_id,
                        type="import",
                        trigger_type="datasource_sync",
                        triggered_by=user.id,
                        status="completed",
                        total_count=1,
                        succeeded_count=1,
                        started_at=indexed_at,
                        finished_at=indexed_at,
                    )
                )
                persisted_docs.append(doc)
                es_documents.append((doc, es_chunks))
            await self._refresh_counts(s, kb)
            await s.commit()
        logger.bind(
            phase="sql_batch_committed",
            kb_id=kb_id,
            batch_documents=len(persisted_docs),
            batch_chunks=sum(len(chunks) for _doc, chunks in es_documents),
        ).info("[datasource] durable batch committed")
        return PersistedBatch(persisted_docs, payload=es_documents)

    async def _index_synced_batch(
        self, kb: KnowledgeBaseRow, batch: PersistedBatch
    ) -> int:
        if self._search is self._local:
            return len(batch.documents)
        result = await self._search.index_document_batch(
            kb, batch.payload, refresh=False
        )
        async with AsyncSession(self._engine) as s:
            rows = (
                await s.exec(
                    select(KnowledgeDocumentRow).where(
                        KnowledgeDocumentRow.id.in_(
                            list(result.indexed_document_ids | result.failed_document_ids)
                        )
                    )
                )
            ).all()
            for row in rows:
                if row.id in result.indexed_document_ids:
                    row.search_index_status = "indexed"
                    row.search_index_error = None
                else:
                    row.search_index_status = "failed"
                    row.search_index_error = result.errors.get(row.id, "index failed")
                    row.search_index_attempts += 1
                s.add(row)
            await s.commit()
        if result.failed_document_ids:
            raise RuntimeError(
                f"Elasticsearch failed {len(result.failed_document_ids)} document(s)"
            )
        return len(result.indexed_document_ids)

    async def _recover_search_index(
        self, kb: KnowledgeBaseRow, ds_id: str
    ) -> list[dict]:
        """Replay SQL-committed search work after a crash or ES outage."""
        if self._search is self._local:
            return []
        async with AsyncSession(self._engine) as s:
            rows = (
                await s.exec(
                    select(KnowledgeDocumentRow).where(
                        KnowledgeDocumentRow.kb_id == kb.id,
                        KnowledgeDocumentRow.source_id == ds_id,
                        KnowledgeDocumentRow.search_index_status.in_(
                            ["pending", "failed", "delete_pending"]
                        ),
                    )
                )
            ).all()
            live = [row for row in rows if row.deleted_at is None]
            deleted = [row for row in rows if row.deleted_at is not None]
            chunks = (
                await s.exec(
                    select(KnowledgeChunkRow).where(
                        KnowledgeChunkRow.document_id.in_([row.id for row in live])
                    )
                )
            ).all() if live else []
        chunks_by_doc: dict[str, list[dict]] = {}
        for chunk in chunks:
            chunks_by_doc.setdefault(chunk.document_id, []).append(
                {
                    "chunk_id": chunk.id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "heading_path": chunk.heading_path,
                    "embedding": chunk.embedding,
                }
            )
        errors: list[dict] = []
        if live:
            batch = PersistedBatch(
                live,
                payload=[(row, chunks_by_doc.get(row.id, [])) for row in live],
            )
            try:
                await self._index_synced_batch(kb, batch)
            except Exception as exc:  # statuses were persisted by the helper
                errors.append({"stage": "recover_index", "error": str(exc)[:500]})
        if deleted:
            ids = [row.id for row in deleted]
            try:
                delete_batch = getattr(self._search, "delete_document_batch", None)
                if delete_batch is not None:
                    await delete_batch(kb.id, ids, refresh=False)
                else:
                    for document_id in ids:
                        await self._search.delete_document(kb.id, document_id)
                async with AsyncSession(self._engine) as s:
                    recovered = (
                        await s.exec(
                            select(KnowledgeDocumentRow).where(
                                KnowledgeDocumentRow.id.in_(ids)
                            )
                        )
                    ).all()
                    for row in recovered:
                        row.search_index_status = "deleted"
                        row.search_index_error = None
                        s.add(row)
                    await s.commit()
            except Exception as exc:
                errors.append({"stage": "recover_delete", "error": str(exc)[:500]})
        return errors

    async def sync_data_source(
        self, kb_id: str, ds_id: str, *, user: User, job_context=None
    ) -> KnowledgeDataSourceRow:
        """Pull the data source and reconcile it into the KB (add/update/delete).

        A durable queue normally invokes this coroutine. Work overlaps through
        bounded stages and checkpoints are written through ``job_context``.
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
            if job_context is not None and await job_context.cancel_requested():
                report["errors"] = []
                await self._finalize_sync(
                    ds_id,
                    status="cancelled",
                    report=report,
                    error=None,
                    active_job_id=job_context.job_id,
                )
                return await self.get_data_source(kb_id, ds_id, user=user)
            adapter = get_adapter(source_type, source_key, source_config)
            discovered = await asyncio.to_thread(adapter.discover)
            report["discovered"] = len(discovered)
            logger.bind(
                job_id=getattr(job_context, "job_id", None),
                datasource_id=ds_id,
                phase="discovered",
                documents=len(discovered),
            ).info("[datasource] discovery completed")
            async with AsyncSession(self._engine) as s:
                kb = await s.get(KnowledgeBaseRow, kb_id)
                ds = await s.get(KnowledgeDataSourceRow, ds_id)
                existing_rows = (
                    await s.exec(
                        select(KnowledgeDocumentRow).where(
                            KnowledgeDocumentRow.kb_id == kb_id,
                            KnowledgeDocumentRow.source_id == ds_id,
                            KnowledgeDocumentRow.deleted_at.is_(None),
                        )
                    )
                ).all()
            if kb is None or ds is None:
                raise PermissionError("data source not found")
            if job_context is not None:
                await job_context.checkpoint({"phase": "recovering"})
            recovery_errors = await self._recover_search_index(kb, ds_id)
            if recovery_errors or any(
                row.search_index_status in {"pending", "failed"}
                for row in existing_rows
            ):
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
            seen_uris = {
                item.source_url or adapter.make_doc_id(item.path) for item in discovered
            }
            changed_counts = {"added": 0, "updated": 0}

            async def fetch_one(item):
                body = await asyncio.to_thread(adapter.fetch, item)
                return FetchedDocument(item, body)

            async def prepare_one(fetched):
                return await self._prepare_synced_document(
                    kb, ds, source_key, adapter, existing_by_uri, fetched
                )

            async def embed_many(documents):
                return await self._embed_synced_documents(kb, documents)

            async def persist_many(documents):
                persisted = await self._persist_synced_batch(kb_id, user, documents)
                for document in documents:
                    key = "updated" if document.prepared.metadata["was_existing"] else "added"
                    changed_counts[key] += 1
                return persisted

            async def index_many(batch):
                return await self._index_synced_batch(kb, batch)

            async def checkpoint(progress):
                if job_context is not None:
                    await job_context.checkpoint(progress)

            async def cancel_requested():
                return bool(
                    job_context is not None
                    and await job_context.cancel_requested()
                )

            outcome = await SyncPipeline(
                limits=self._pipeline_limits,
                fetch=fetch_one,
                prepare=prepare_one,
                embed=embed_many,
                persist_batch=persist_many,
                index_batch=index_many,
                checkpoint=checkpoint,
                cancel_requested=cancel_requested,
            ).run(discovered)
            report["added"] = changed_counts["added"]
            report["updated"] = changed_counts["updated"]
            report["unchanged"] = outcome.progress["unchanged"]
            report["failed"] = outcome.progress["failed"] + len(recovery_errors)
            report["errors"] = recovery_errors + outcome.errors

            # deletions — skip when discovery is known-incomplete so a transient
            # crawl failure can't be mistaken for source-side removal.
            if not getattr(adapter, "discovery_partial", False):
                stale_ids = [r.id for u, r in existing_by_uri.items() if u not in seen_uris]
                if stale_ids:
                    await self._soft_delete_documents(kb_id, stale_ids, user=user)
                    report["deleted"] = len(stale_ids)
            if self._search is not self._local and outcome.status != "failed":
                await self._search.refresh_kb(kb_id)
            status = outcome.status
            if recovery_errors and status == "succeeded":
                status = "partial"
            # cap stored error detail so a mass failure can't bloat the row
            report["errors"] = report["errors"][:20]
            await self._finalize_sync(
                ds_id,
                status=status,
                report=report,
                error=None,
                active_job_id=getattr(job_context, "job_id", None),
            )
            logger.bind(
                job_id=getattr(job_context, "job_id", None),
                datasource_id=ds_id,
                phase="finalized",
                status=status,
                discovered=report["discovered"],
                indexed=outcome.progress["indexed"],
                failed=report["failed"],
            ).info("[datasource] sync finalized")
        except Exception as exc:  # noqa: BLE001 — surface as failed status, don't crash the task
            logger.exception(f"[datasource] sync failed for {ds_id}: {exc}")
            report["errors"] = report["errors"][:20]
            await self._finalize_sync(
                ds_id,
                status="failed",
                report=report,
                error=str(exc),
                active_job_id=getattr(job_context, "job_id", None),
            )

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
                doc.search_index_status = (
                    "deleted" if self._search is self._local else "delete_pending"
                )
                doc.deleted_at = now
                doc.chunk_count = 0
                doc.updated_by = user.id
                doc.updated_at = now
                s.add(doc)
                removed.append(doc_id)
            await self._refresh_counts(s, kb)
            await s.commit()
        await self._delete_from_engine_best_effort(kb_id, removed)

    async def _finalize_sync(
        self,
        ds_id: str,
        *,
        status: str,
        report: dict,
        error: Optional[str],
        active_job_id: Optional[str] = None,
    ) -> None:
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
            if active_job_id is None or ds.active_job_id == active_job_id:
                ds.active_job_id = None
            ds.last_error = error
            ds.last_sync_report = report
            flag_modified(ds, "last_sync_report")
            ds.last_sync_finished_at = now_utc()
            ds.updated_at = now_utc()
            s.add(ds)
            await s.commit()
