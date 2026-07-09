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


class SearchEngine(Protocol):
    async def ensure_index(self, kb) -> None: ...
    async def index_chunks(self, kb, doc, chunks: list[dict]) -> None: ...
    async def delete_document(self, kb_id: str, document_id: str) -> None: ...
    async def delete_kb(self, kb_id: str) -> None: ...
    async def healthy(self) -> bool: ...
    async def search(
        self, *, kb_ids: list[str], query: str, mode: str = "hybrid",
        offset: int = 0, limit: int = 6, score_threshold: float = 0.0,
        dimension: int = 64, filters: Optional[dict] = None,
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

    async def search(
        self, *, kb_ids, query, mode="hybrid", offset=0, limit=6,
        score_threshold=0.0, dimension=64, filters=None,
    ) -> tuple[list[SearchHit], int]:
        # Imported lazily to avoid an import cycle (knowledge imports this module
        # at load time for SearchHit; these helpers live in knowledge).
        from app.knowledge import cosine, embed_text, keyword_score

        filters = filters or {}
        if not kb_ids or not query.strip():
            return [], 0
        qvec = embed_text(query, dimension=64)
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
            k_score = keyword_score(query, chunk.text)
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

    async def healthy(self) -> bool:
        try:
            return bool(await self.client().ping())
        except Exception as ex:
            logger.warning(f"[es] ping failed: {ex!r}")
            return False

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
        if not chunks:
            await self.delete_document(kb.id, doc.id)
            return
        await self.ensure_index(kb)
        idx = self._index(kb.id)
        client = self.client()
        await self.delete_document(kb.id, doc.id)
        ops: list[dict] = []
        for c in chunks:
            ops.append({"index": {"_index": idx, "_id": c["chunk_id"]}})
            ops.append({
                "kb_id": kb.id,
                "document_id": doc.id,
                "chunk_id": c["chunk_id"],
                "chunk_index": c.get("chunk_index", 0),
                "text": c.get("text", ""),
                "title": doc.title or "",
                "source_uri": doc.uri or "",
                "source_type": doc.source_type or "",
                "category": doc.category,
                "tags": doc.tags or [],
                "status": "active",
                "embedding": c.get("embedding") or [],
            })
        resp = await client.bulk(operations=ops, refresh=True)
        if isinstance(resp, dict) and resp.get("errors"):
            logger.warning(f"[es] bulk index had errors for doc {doc.id}")

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
        score_threshold=0.0, dimension=64, filters=None,
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
                    "must": [{"multi_match": {"query": query, "fields": ["text", "title^2"]}}],
                }
            }
        else:
            # vector-only still needs the filter applied to the base query
            body["query"] = {"bool": {"filter": clauses}}
        if mode in ("vector", "hybrid"):
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


def build_search_engine(settings, sql_engine) -> SearchEngine:
    """Pick the primary engine from settings. ``KnowledgeService`` supplies the
    local fallback separately, so this only decides the primary."""
    mode = getattr(settings, "search_engine", "auto") or "auto"
    local = LocalSearchEngine(sql_engine)
    if mode == "local":
        return local
    url = getattr(settings, "elasticsearch_url", "")
    if url:
        logger.info(f"[search] primary engine = elasticsearch ({url})")
        return ElasticsearchEngine.from_settings(settings)
    if mode == "elasticsearch":
        logger.warning("[search] search_engine=elasticsearch but elasticsearch_url is empty; using local")
    return local
