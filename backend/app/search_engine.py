"""Pluggable knowledge-base retrieval engines (the online query path).

Two engines behind one interface:

- ``LocalSearchEngine`` — the zero-dependency built-in: scans a KB's active
  chunks from SQL and scores them with the local hash embedding + naive keyword
  overlap. Correct but O(all chunks) and semantically weak; the dev/test default.
- ``ElasticsearchEngine`` — real BM25 full-text + ``dense_vector`` kNN hybrid via
  Elasticsearch 8.x. Indexes chunks on ingestion (offline) and serves queries
  (online). Hybrid combines a ``knn`` clause and a BM25 ``query`` in one search
  body (ES sums the scores — license-safe, unlike RRF which needs a paid tier).

``build_search_engine`` picks the primary engine from settings; ``KnowledgeService``
keeps a ``LocalSearchEngine`` as a fallback so an ES outage degrades to local
retrieval instead of failing the request (unless ES is explicitly forced).

Permission scoping is NOT done here — callers pass an already-authorized
``kb_ids`` list. Engines only see kb ids, never the ``User``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from loguru import logger
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from app.models import KnowledgeChunkRow, KnowledgeDocumentRow


@dataclass
class SearchHit:
    kb_id: str
    document_id: str
    chunk_id: str
    title: str
    source_uri: str
    source_type: str
    text: str
    score: float
    vector_score: float
    keyword_score: float
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class BatchIndexResult:
    indexed_document_ids: set[str]
    failed_document_ids: set[str]
    errors: dict[str, str]


def matches_filters(doc: KnowledgeDocumentRow, filters: dict) -> bool:
    """Metadata post-filter shared by the local engine (ES filters in-query)."""
    if filters.get("source_type") and doc.source_type != filters["source_type"]:
        return False
    if filters.get("category") and doc.category != filters["category"]:
        return False
    if filters.get("tags"):
        wanted = set(filters["tags"] if isinstance(filters["tags"], list) else [filters["tags"]])
        if not wanted.intersection(set(doc.tags or [])):
            return False
    return True


def _es_error_reason(msg: str, url: str) -> str:
    """Map a raw Elasticsearch/transport exception string to an actionable, human
    (zh) reason for the settings-page reachability badge."""
    low = msg.lower()
    if "aiohttp" in low:
        return "缺少 aiohttp 依赖：Elasticsearch 异步客户端需要它，请安装 aiohttp 后重试"
    if any(k in low for k in ("unauthorized", "authentication", "401", "403", "security_exception")):
        return "认证失败：请检查 API Key 或用户名/密码"
    if "timeout" in low or "timed out" in low:
        return f"连接 {url} 超时：请检查地址、端口与网络可达性"
    if any(k in low for k in ("refused", "cannot connect", "connection error", "name or service", "nodename", "getaddrinfo", "no route")):
        return f"无法连接到 {url}：连接被拒绝或地址不可达"
    if "certificate" in low or "ssl" in low:
        return f"TLS/证书校验失败：请检查证书或关闭 verify_certs（{url}）"
    return f"连接失败：{msg}"


class SearchEngine(Protocol):
    async def ensure_index(self, kb) -> None: ...
    async def index_chunks(self, kb, doc, chunks: list[dict]) -> None: ...
    async def delete_document(self, kb_id: str, document_id: str) -> None: ...
    async def delete_kb(self, kb_id: str) -> None: ...
    async def healthy(self) -> bool: ...
    async def health_detail(self) -> tuple[bool, str]: ...
    async def search(
        self, *, kb_ids: list[str], query: str, mode: str = "hybrid",
        offset: int = 0, limit: int = 6, score_threshold: float = 0.0,
        dimension: int = 64, filters: Optional[dict] = None,
        query_vector: Optional[list[float]] = None,
    ) -> tuple[list[SearchHit], int]: ...

    name: str


# --------------------------------------------------------------------------- #
# Local engine (SQL scan) — behavior identical to the pre-ES KnowledgeService.
# --------------------------------------------------------------------------- #
class LocalSearchEngine:
    name = "local"

    def __init__(self, engine):
        self._engine = engine

    # Offline hooks are no-ops: SQL rows ARE the local index.
    async def ensure_index(self, kb) -> None: ...
    async def index_chunks(self, kb, doc, chunks: list[dict]) -> None: ...
    async def delete_document(self, kb_id: str, document_id: str) -> None: ...
    async def delete_kb(self, kb_id: str) -> None: ...

    async def healthy(self) -> bool:
        return True

    async def health_detail(self) -> tuple[bool, str]:
        return True, "本地引擎（内置，无需外部连接）"

    async def search(
        self, *, kb_ids, query, mode="hybrid", offset=0, limit=6,
        score_threshold=0.0, dimension=64, filters=None, query_vector=None,
    ) -> tuple[list[SearchHit], int]:
        # Imported lazily to avoid an import cycle (knowledge imports this module
        # at load time for SearchHit; these helpers live in knowledge).
        from app.knowledge import cosine, embed_text, keyword_score

        filters = filters or {}
        if not kb_ids or not query.strip():
            return [], 0
        # Prefer the caller-supplied query vector (embedded with the KB's own
        # embedder, so it matches the stored chunk vectors). Only fall back to
        # the local hash embedding — at the KB's true dimension, not a hardcoded
        # 64 — when no vector was threaded in (direct/legacy callers).
        qvec = query_vector if query_vector is not None else embed_text(query, dimension=dimension)
        async with AsyncSession(self._engine) as s:
            stmt = (
                select(KnowledgeChunkRow, KnowledgeDocumentRow)
                .join(KnowledgeDocumentRow, KnowledgeDocumentRow.id == KnowledgeChunkRow.document_id)
                .where(
                    KnowledgeChunkRow.kb_id.in_(kb_ids),
                    KnowledgeChunkRow.status == "active",
                    KnowledgeChunkRow.deleted_at.is_(None),
                    KnowledgeDocumentRow.status == "indexed",
                    KnowledgeDocumentRow.deleted_at.is_(None),
                )
            )
            rows = (await s.exec(stmt)).all()
        hits: list[SearchHit] = []
        for chunk, doc in rows:
            if not matches_filters(doc, filters):
                continue
            v_score = cosine(qvec, list(chunk.embedding or []))
            # Fold the heading breadcrumb into the keyword-scored text so a section
            # title (which isn't repeated in every chunk body) still matches.
            heading = " > ".join(chunk.heading_path or [])
            k_score = keyword_score(
                query, f"{heading}\n{chunk.text}" if heading else chunk.text
            )
            if mode == "vector":
                score = v_score
            elif mode == "keyword":
                score = k_score
            else:
                score = (v_score + k_score) / 2
            if score < score_threshold:
                continue
            hits.append(
                SearchHit(
                    kb_id=chunk.kb_id,
                    document_id=doc.id,
                    chunk_id=chunk.id,
                    title=doc.title,
                    source_uri=doc.uri,
                    source_type=doc.source_type,
                    text=chunk.text,
                    score=round(score, 6),
                    vector_score=round(v_score, 6),
                    keyword_score=round(k_score, 6),
                    metadata={
                        "tags": doc.tags or [],
                        "category": doc.category,
                        "chunk_index": chunk.chunk_index,
                        **(chunk.chunk_metadata or {}),
                    },
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        total = len(hits)
        return hits[offset : offset + limit], total


# --------------------------------------------------------------------------- #
# Elasticsearch engine — BM25 + dense_vector kNN hybrid.
# --------------------------------------------------------------------------- #
# ES built-in "cjk" analyzer (bigram) gives usable Chinese tokenization without
# requiring the IK plugin — swap the analyzer if a cluster ships IK/smartcn.
_INDEX_SETTINGS = {"analysis": {"analyzer": {"kb_text": {"type": "cjk"}}}}


def _mapping(dimension: int) -> dict:
    return {
        "properties": {
            "kb_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "text": {"type": "text", "analyzer": "kb_text"},
            "title": {"type": "text", "analyzer": "kb_text", "fields": {"raw": {"type": "keyword"}}},
            "heading": {"type": "text", "analyzer": "kb_text"},
            "source_uri": {"type": "keyword"},
            "source_type": {"type": "keyword"},
            "category": {"type": "keyword"},
            "tags": {"type": "keyword"},
            "status": {"type": "keyword"},
            "embedding": {"type": "dense_vector", "dims": dimension, "index": True, "similarity": "cosine"},
        }
    }


class ElasticsearchEngine:
    name = "elasticsearch"

    def __init__(
        self,
        url: str,
        *,
        api_key: str = "",
        username: str = "",
        password: str = "",
        index_prefix: str = "kb",
        verify_certs: bool = True,
        timeout: int = 30,
        client_factory: Optional[Callable[[], Any]] = None,
        bulk_target_bytes: int = 5 * 1024 * 1024,
        bulk_max_bytes: int = 10 * 1024 * 1024,
    ):
        self._url = url
        self._api_key = api_key
        self._username = username
        self._password = password
        self._prefix = index_prefix
        self._verify_certs = verify_certs
        self._timeout = timeout
        self._client_factory = client_factory
        self._client: Any = None
        self._bulk_target_bytes = max(256, bulk_target_bytes)
        self._bulk_max_bytes = max(self._bulk_target_bytes, bulk_max_bytes)

    @classmethod
    def from_settings(cls, settings, client_factory=None) -> "ElasticsearchEngine":
        return cls(
            settings.elasticsearch_url,
            api_key=settings.elasticsearch_api_key,
            username=settings.elasticsearch_username,
            password=settings.elasticsearch_password,
            index_prefix=settings.elasticsearch_index_prefix,
            verify_certs=settings.elasticsearch_verify_certs,
            timeout=settings.elasticsearch_timeout,
            client_factory=client_factory,
            bulk_target_bytes=getattr(
                settings, "sync_es_bulk_target_bytes", 5 * 1024 * 1024
            ),
            bulk_max_bytes=getattr(
                settings, "sync_es_bulk_max_bytes", 10 * 1024 * 1024
            ),
        )

    @classmethod
    def from_vectordb_config(
        cls,
        cfg,
        client_factory=None,
        *,
        bulk_target_bytes: int = 5 * 1024 * 1024,
        bulk_max_bytes: int = 10 * 1024 * 1024,
    ) -> "ElasticsearchEngine":
        """Build from the global ``knowledgebase.vectordb`` config section
        (``app.agent_config.VectorDBConfig``). Secrets resolve inline first, then
        by env-var name (``*_env``), mirroring the search/sandbox providers."""
        return cls(
            cfg.url,
            api_key=cfg.api_key or os.environ.get(cfg.api_key_env or "", ""),
            username=cfg.username,
            password=cfg.password or os.environ.get(cfg.password_env or "", ""),
            index_prefix=cfg.index_prefix or "kb",
            verify_certs=cfg.verify_certs,
            timeout=cfg.timeout,
            client_factory=client_factory,
            bulk_target_bytes=bulk_target_bytes,
            bulk_max_bytes=bulk_max_bytes,
        )

    # -- client / index helpers -------------------------------------------- #
    def _index(self, kb_id: str) -> str:
        return f"{self._prefix}-{kb_id}".lower()

    def client(self):
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory()
            return self._client
        from elasticsearch import AsyncElasticsearch  # lazy: ES optional at boot

        kwargs: dict = {"hosts": [self._url], "verify_certs": self._verify_certs, "request_timeout": self._timeout}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        elif self._username:
            kwargs["basic_auth"] = (self._username, self._password)
        self._client = AsyncElasticsearch(**kwargs)
        return self._client

    @staticmethod
    def _dims(kb) -> int:
        try:
            return int((kb.embedding_config or {}).get("dimension") or 64)
        except Exception:
            return 64

    async def health_detail(self) -> tuple[bool, str]:
        """Live reachability probe returning ``(ok, human_reason)``.

        The reason is surfaced verbatim on the Knowledge Base settings page so a
        failed connection is actionable instead of a silent "unhealthy". We
        classify the common failure modes — the missing-``aiohttp`` transport
        dependency, connection refused / bad host, auth rejection, and timeout —
        because they need very different fixes."""
        try:
            ok = bool(await self.client().ping())
        except Exception as ex:
            logger.warning(f"[es] ping failed: {ex!r}")
            return False, _es_error_reason(str(ex), self._url)
        if ok:
            return True, f"已连接 {self._url}"
        # ping() returns False (no exception) when ES answers non-200 or the host
        # is simply unreachable at the transport layer.
        return False, f"无法连接到 {self._url}：Elasticsearch 未响应 ping"

    async def healthy(self) -> bool:
        ok, _ = await self.health_detail()
        return ok

    # -- offline (indexing) ------------------------------------------------- #
    async def ensure_index(self, kb) -> None:
        idx = self._index(kb.id)
        client = self.client()
        exists = await client.indices.exists(index=idx)
        if exists:
            return
        await client.indices.create(index=idx, mappings=_mapping(self._dims(kb)), settings=_INDEX_SETTINGS)

    async def index_chunks(self, kb, doc, chunks: list[dict]) -> None:
        """Replace this document's chunks in ES (delete-then-bulk), matching the
        SQL upsert semantics in ``import_text_document``."""
        result = await self.index_document_batch(kb, [(doc, chunks)], refresh=True)
        if result.failed_document_ids:
            logger.warning(f"[es] bulk index had errors for doc {doc.id}")

    async def index_document_batch(
        self,
        kb,
        documents: list[tuple[Any, list[dict]]],
        *,
        refresh: bool = False,
    ) -> BatchIndexResult:
        """Replace several documents with one delete and size-bounded bulks."""
        if not documents:
            return BatchIndexResult(set(), set(), {})
        await self.ensure_index(kb)
        idx = self._index(kb.id)
        client = self.client()
        document_ids = [doc.id for doc, _ in documents]
        await self.delete_document_batch(kb.id, document_ids, refresh=refresh)

        batches: list[list[dict]] = []
        current: list[dict] = []
        current_bytes = 0
        errors: dict[str, str] = {}
        chunk_documents: dict[str, str] = {}
        for doc, chunks in documents:
            for chunk in chunks:
                action = {"index": {"_index": idx, "_id": chunk["chunk_id"]}}
                source = {
                    "kb_id": kb.id,
                    "document_id": doc.id,
                    "chunk_id": chunk["chunk_id"],
                    "chunk_index": chunk.get("chunk_index", 0),
                    "text": chunk.get("text", ""),
                    "title": doc.title or "",
                    "heading": " > ".join(chunk.get("heading_path") or []),
                    "source_uri": doc.uri or "",
                    "source_type": doc.source_type or "",
                    "category": doc.category,
                    "tags": doc.tags or [],
                    "status": "active",
                    "embedding": chunk.get("embedding") or [],
                }
                pair_bytes = len(
                    json.dumps([action, source], ensure_ascii=False, separators=(",", ":")).encode()
                )
                if pair_bytes > self._bulk_max_bytes:
                    errors[doc.id] = (
                        f"chunk {chunk['chunk_id']} exceeds Elasticsearch bulk byte limit"
                    )
                    continue
                if current and current_bytes + pair_bytes > self._bulk_target_bytes:
                    batches.append(current)
                    current = []
                    current_bytes = 0
                current.extend([action, source])
                current_bytes += pair_bytes
                chunk_documents[chunk["chunk_id"]] = doc.id
        if current:
            batches.append(current)

        failed = set(errors)
        for operations in batches:
            response = await client.bulk(operations=operations, refresh=refresh)
            body = response.body if hasattr(response, "body") else response
            if not isinstance(body, dict) or not body.get("errors"):
                continue
            for item in body.get("items") or []:
                detail = next(iter(item.values()), {})
                if not detail.get("error"):
                    continue
                document_id = chunk_documents.get(str(detail.get("_id")))
                if document_id:
                    failed.add(document_id)
                    errors[document_id] = str(detail["error"])[:500]

        return BatchIndexResult(set(document_ids) - failed, failed, errors)

    async def delete_document_batch(
        self, kb_id: str, document_ids: list[str], *, refresh: bool = False
    ) -> None:
        if not document_ids:
            return
        await self.client().delete_by_query(
            index=self._index(kb_id),
            query={"terms": {"document_id": document_ids}},
            conflicts="proceed",
            ignore_unavailable=True,
            refresh=refresh,
        )

    async def refresh_kb(self, kb_id: str) -> None:
        await self.client().indices.refresh(index=self._index(kb_id))

    async def delete_document(self, kb_id: str, document_id: str) -> None:
        await self.client().delete_by_query(
            index=self._index(kb_id),
            query={"term": {"document_id": document_id}},
            conflicts="proceed",
            ignore_unavailable=True,
            refresh=True,
        )

    async def delete_kb(self, kb_id: str) -> None:
        await self.client().indices.delete(index=self._index(kb_id), ignore_unavailable=True)

    # -- online (query) ----------------------------------------------------- #
    def _filter_clauses(self, kb_ids: list[str], filters: dict) -> list[dict]:
        clauses: list[dict] = [
            {"terms": {"kb_id": kb_ids}},
            {"term": {"status": "active"}},
        ]
        if filters.get("source_type"):
            clauses.append({"term": {"source_type": filters["source_type"]}})
        if filters.get("category"):
            clauses.append({"term": {"category": filters["category"]}})
        if filters.get("tags"):
            tags = filters["tags"] if isinstance(filters["tags"], list) else [filters["tags"]]
            clauses.append({"terms": {"tags": tags}})
        return clauses

    async def search(
        self, *, kb_ids, query, mode="hybrid", offset=0, limit=6,
        score_threshold=0.0, dimension=64, filters=None, query_vector=None,
    ) -> tuple[list[SearchHit], int]:
        filters = filters or {}
        if not kb_ids or not query.strip():
            return [], 0
        clauses = self._filter_clauses(kb_ids, filters)
        body: dict = {"from": offset, "size": limit, "track_total_hits": True}
        if mode in ("keyword", "hybrid"):
            body["query"] = {
                "bool": {
                    "filter": clauses,
                    "must": [{"multi_match": {"query": query, "fields": ["text", "title^2", "heading^1.5"]}}],
                }
            }
        else:
            # vector-only still needs the filter applied to the base query
            body["query"] = {"bool": {"filter": clauses}}
        if mode in ("vector", "hybrid"):
            if query_vector is not None:
                qvec = query_vector
            else:
                from app.knowledge import embed_text

                qvec = embed_text(query, dimension=dimension)
            body["knn"] = {
                "field": "embedding",
                "query_vector": qvec,
                "k": offset + limit,
                "num_candidates": max(50, (offset + limit) * 4),
                "filter": {"bool": {"filter": clauses}},
            }
        indices = ",".join(self._index(k) for k in kb_ids)
        resp = await self.client().search(index=indices, ignore_unavailable=True, **body)
        raw = resp["hits"]["hits"]
        total = int(resp["hits"]["total"]["value"]) if isinstance(resp["hits"].get("total"), dict) else len(raw)
        hits: list[SearchHit] = []
        for h in raw:
            src = h.get("_source", {})
            score = float(h.get("_score") or 0.0)
            if score < score_threshold:
                continue
            hits.append(
                SearchHit(
                    kb_id=src.get("kb_id", ""),
                    document_id=src.get("document_id", ""),
                    chunk_id=src.get("chunk_id", h.get("_id", "")),
                    title=src.get("title", ""),
                    source_uri=src.get("source_uri", ""),
                    source_type=src.get("source_type", ""),
                    text=src.get("text", ""),
                    score=round(score, 6),
                    vector_score=0.0,
                    keyword_score=0.0,
                    metadata={
                        "chunk_index": src.get("chunk_index"),
                        "tags": src.get("tags", []),
                        "category": src.get("category"),
                    },
                )
            )
        return hits, total


def build_search_engine(settings, sql_engine, *, vectordb=None) -> SearchEngine:
    """Pick the primary engine. ``KnowledgeService`` supplies the local fallback
    separately, so this only decides the primary.

    The global ``knowledgebase.vectordb`` config section (``vectordb``) is
    authoritative when supplied: ``engine == "elasticsearch"`` with a non-empty
    ``url`` builds an ES engine, anything else stays local. When ``vectordb`` is
    ``None`` (e.g. offline/test callers), fall back to the legacy env-driven
    ``Settings.search_engine`` / ``elasticsearch_*`` path for backward compat."""
    local = LocalSearchEngine(sql_engine)
    if vectordb is not None:
        if vectordb.engine == "elasticsearch" and vectordb.url:
            logger.info(f"[search] primary engine = elasticsearch ({vectordb.url})")
            return ElasticsearchEngine.from_vectordb_config(
                vectordb,
                bulk_target_bytes=getattr(
                    settings, "sync_es_bulk_target_bytes", 5 * 1024 * 1024
                ),
                bulk_max_bytes=getattr(
                    settings, "sync_es_bulk_max_bytes", 10 * 1024 * 1024
                ),
            )
        if vectordb.engine == "elasticsearch":
            logger.warning("[search] knowledgebase.vectordb.engine=elasticsearch but url is empty; using local")
        return local
    mode = getattr(settings, "search_engine", "auto") or "auto"
    if mode == "local":
        return local
    url = getattr(settings, "elasticsearch_url", "")
    if url:
        logger.info(f"[search] primary engine = elasticsearch ({url})")
        return ElasticsearchEngine.from_settings(settings)
    if mode == "elasticsearch":
        logger.warning("[search] search_engine=elasticsearch but elasticsearch_url is empty; using local")
    return local
