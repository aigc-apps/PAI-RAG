# ruff: noqa: E402
"""KnowledgeService with real embedder/reranker seams:
- ingest + query embed through the KB's frozen embedder (same source);
- rerank reorders the candidate window when rerank_config.enabled;
- create_kb records a DashScope embedding_config when the router is healthy,
  else the local_hash default;
- update_kb refuses to change a KB's (immutable) embedding_config.

All offline — fake embedder/reranker injected via the router's client map."""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from loguru import logger

import app.knowledge as knowledge_module
from app.db import create_all, make_engine
from app.knowledge import KnowledgeService
from app.providers import ModelCatalog, ModelSpec, ProviderConfig, ProviderRouter
from app.store.base import User

ADMIN = User(id="u_admin", email="a@x.io", role="admin")


class FakeEmbedder:
    def __init__(self, dimension=8):
        self.dimension = dimension
        self.calls: list = []

    async def embed(self, texts, *, text_type="document"):
        self.calls.append((text_type, list(texts)))
        # distinct-but-deterministic vector per text so cosine has signal
        return [[float(len(t) % 7 + 1)] * self.dimension for t in texts]


class FakeReranker:
    def __init__(self):
        self.calls: list = []

    async def rerank(self, query, documents, *, top_n=None):
        self.calls.append((query, list(documents), top_n))
        # reverse the candidate order to prove reranking actually reorders
        order = list(range(len(documents)))[::-1]
        return [(idx, float(len(documents) - pos)) for pos, idx in enumerate(order)][
            : (top_n or len(documents))
        ]


def _chat_router():
    cat = ModelCatalog(default_model="dashscope/chat", providers=[
        ProviderConfig(name="dashscope", base_url="https://ds/v1", api_key="k",
                       models=[ModelSpec(id="chat")]),
    ])
    return ProviderRouter(cat)


async def _svc(router=None):
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    await create_all(engine)
    return KnowledgeService(engine, router=router)


def test_ingest_and_query_share_the_kb_embedder():
    async def scenario():
        router = _chat_router()
        embedder = FakeEmbedder(dimension=8)
        router.register_llm("dashscope/emb", embedder)  # inject into warm-client map
        svc = await _svc(router)
        kb = await svc.create_kb(
            user=ADMIN, name="KB", visibility="public",
            embedding_config={"provider_id": "dashscope", "model": "dashscope/emb", "dimension": 8},
        )
        await svc.import_text_document(kb.id, user=ADMIN, title="d", content="hello world body", uri="d/1")
        hits, total = await svc.search(user=ADMIN, kb_ids=[kb.id], query="hello", mode="vector")
        return embedder, hits, total

    embedder, hits, total = asyncio.run(scenario())
    text_types = [c[0] for c in embedder.calls]
    assert "document" in text_types  # ingest
    assert "query" in text_types     # query
    assert total >= 1 and hits


def test_rerank_reorders_when_enabled():
    async def scenario():
        router = _chat_router()
        router.register_llm("dashscope/emb", FakeEmbedder(dimension=8))
        reranker = FakeReranker()
        router.register_llm("dashscope/rr", reranker)
        svc = await _svc(router)
        kb = await svc.create_kb(
            user=ADMIN, name="KB", visibility="public",
            embedding_config={"provider_id": "dashscope", "model": "dashscope/emb", "dimension": 8},
            rerank_config={"enabled": True, "model": "dashscope/rr"},
        )
        for i in range(3):
            await svc.import_text_document(kb.id, user=ADMIN, title=f"d{i}", content=f"shared word doc {i}", uri=f"d/{i}")
        hits, _ = await svc.search(
            user=ADMIN,
            kb_ids=[kb.id],
            query="shared",
            mode="keyword",
            top_k=3,
            rerank_config={
                "enabled": True,
                "model": "dashscope/rr",
                "candidate_pool_size": 20,
            },
        )
        return reranker, hits

    reranker, hits = asyncio.run(scenario())
    assert reranker.calls, "reranker must be invoked when enabled"
    query, candidate_docs, top_n = reranker.calls[-1]
    assert query == "shared" and top_n == 3 and len(candidate_docs) == 3
    # the fake reranker reverses its input; final hit order must follow that
    assert [h.text for h in hits] == list(reversed(candidate_docs))
    # rerank relevance score is written onto the hits (descending)
    assert base_scores_descending(hits)


def test_kb_rerank_config_is_ignored_without_agent_policy():
    async def scenario():
        router = _chat_router()
        reranker = FakeReranker()
        router.register_llm("dashscope/rr", reranker)
        svc = await _svc(router)
        kb = await svc.create_kb(
            user=ADMIN,
            name="KB",
            visibility="public",
            rerank_config={"enabled": True, "model": "dashscope/rr"},
        )
        await svc.import_text_document(
            kb.id,
            user=ADMIN,
            title="d",
            content="shared content",
            uri="d/1",
        )
        hits, _ = await svc.search(
            user=ADMIN,
            kb_ids=[kb.id],
            query="shared",
            mode="keyword",
        )
        return reranker, hits

    reranker, hits = asyncio.run(scenario())
    assert hits
    assert reranker.calls == []


def test_multi_kb_search_queries_each_embedding_group_then_reranks_once():
    async def scenario():
        router = _chat_router()
        emb_a = FakeEmbedder(dimension=8)
        emb_b = FakeEmbedder(dimension=12)
        reranker = FakeReranker()
        router.register_llm("dashscope/emb-a", emb_a)
        router.register_llm("dashscope/emb-b", emb_b)
        router.register_llm("dashscope/rr", reranker)
        svc = await _svc(router)
        kb_a = await svc.create_kb(
            user=ADMIN,
            name="A",
            visibility="public",
            embedding_config={
                "provider_id": "dashscope",
                "model": "dashscope/emb-a",
                "dimension": 8,
            },
        )
        kb_b = await svc.create_kb(
            user=ADMIN,
            name="B",
            visibility="public",
            embedding_config={
                "provider_id": "dashscope",
                "model": "dashscope/emb-b",
                "dimension": 12,
            },
        )
        await svc.import_text_document(
            kb_a.id, user=ADMIN, title="A doc", content="turbox alpha", uri="a"
        )
        await svc.import_text_document(
            kb_b.id, user=ADMIN, title="B doc", content="turbox beta", uri="b"
        )
        emb_a.calls.clear()
        emb_b.calls.clear()
        hits, _ = await svc.search(
            user=ADMIN,
            kb_ids=[kb_a.id, kb_b.id],
            query="turbox",
            top_k=2,
            rerank_config={
                "enabled": True,
                "model": "dashscope/rr",
                "candidate_pool_size": 20,
            },
        )
        return emb_a, emb_b, reranker, hits, kb_a, kb_b

    emb_a, emb_b, reranker, hits, kb_a, kb_b = asyncio.run(scenario())
    assert [call[0] for call in emb_a.calls] == ["query"]
    assert [call[0] for call in emb_b.calls] == ["query"]
    assert len(reranker.calls) == 1
    assert {hit.kb_id for hit in hits} == {kb_a.id, kb_b.id}
    assert base_scores_descending(hits)


def test_query_embedding_fallback_redacts_exception_message(monkeypatch):
    sentinel = "SENSITIVE_EMBED_QUERY_787e"

    class FailingEmbedder:
        async def embed(self, texts, *, text_type="document"):
            raise RuntimeError(f"embedding payload={texts[0]}")

    async def scenario():
        svc = await _svc()
        kb = await svc.create_kb(user=ADMIN, name="KB", visibility="public")
        await svc.import_text_document(
            kb.id, user=ADMIN, title="d", content="safe searchable content", uri="d/1"
        )
        monkeypatch.setattr(
            knowledge_module, "build_embedder", lambda config, router: FailingEmbedder()
        )
        messages: list[str] = []
        sink = logger.add(messages.append, format="{message}")
        try:
            hits, _ = await svc.search(
                user=ADMIN, kb_ids=[kb.id], query=sentinel, mode="hybrid"
            )
        finally:
            logger.remove(sink)
        return hits, messages

    hits, messages = asyncio.run(scenario())
    assert hits and hits[0].text == "safe searchable content"
    assert all(sentinel not in message for message in messages)
    assert any(
        "operation=query_embedding" in message and "error_type=RuntimeError" in message
        for message in messages
    )


def test_primary_search_fallback_redacts_exception_message():
    sentinel = "SENSITIVE_PRIMARY_QUERY_a119"

    class FailingPrimarySearch:
        name = "test-primary"

        async def ensure_index(self, kb):
            return None

        async def index_chunks(self, kb, doc, chunks):
            return None

        async def search(self, **kwargs):
            raise RuntimeError(f"primary payload={kwargs['query']}")

    async def scenario():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        svc = KnowledgeService(
            engine, search_engine=FailingPrimarySearch(), fallback_to_local=True
        )
        kb = await svc.create_kb(user=ADMIN, name="KB", visibility="public")
        await svc.import_text_document(
            kb.id, user=ADMIN, title="d", content=sentinel, uri="d/1"
        )
        messages: list[str] = []
        sink = logger.add(messages.append, format="{message}")
        try:
            hits, _ = await svc.search(
                user=ADMIN, kb_ids=[kb.id], query=sentinel, mode="keyword"
            )
        finally:
            logger.remove(sink)
        return hits, messages

    hits, messages = asyncio.run(scenario())
    assert hits and hits[0].text == sentinel
    assert all(sentinel not in message for message in messages)
    assert any(
        "operation=primary_search" in message
        and "engine=test-primary" in message
        and "error_type=RuntimeError" in message
        for message in messages
    )


def test_rerank_fallback_redacts_query_and_document_content():
    sentinel = "SENSITIVE_RERANK_CONTENT_229f"

    class FailingReranker:
        async def rerank(self, query, documents, *, top_n=None):
            raise RuntimeError(f"rerank query={query} document={documents[0]}")

    async def scenario():
        router = _chat_router()
        router.register_llm("dashscope/rr", FailingReranker())
        svc = await _svc(router)
        kb = await svc.create_kb(user=ADMIN, name="KB", visibility="public")
        await svc.import_text_document(
            kb.id, user=ADMIN, title="d", content=sentinel, uri="d/1"
        )
        messages: list[str] = []
        sink = logger.add(messages.append, format="{message}")
        try:
            hits, _ = await svc.search(
                user=ADMIN,
                kb_ids=[kb.id],
                query=sentinel,
                mode="keyword",
                rerank_config={"enabled": True, "model": "dashscope/rr"},
            )
        finally:
            logger.remove(sink)
        return hits, messages

    hits, messages = asyncio.run(scenario())
    assert hits and hits[0].text == sentinel  # original ordering is preserved
    assert all(sentinel not in message for message in messages)
    assert any(
        "operation=rerank" in message
        and "model=dashscope/rr" in message
        and "error_type=RuntimeError" in message
        for message in messages
    )


def base_scores_descending(hits) -> bool:
    return all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))


def test_create_kb_records_dashscope_embedding_when_router_healthy():
    async def scenario():
        cat = ModelCatalog(default_model="dashscope/chat", providers=[
            ProviderConfig(name="dashscope", base_url="https://ds/v1", api_key="k", models=[
                ModelSpec(id="chat"),
                ModelSpec(id="text-embedding-v4", type="embedding", dimension=1024, base_url="https://ds/emb"),
            ]),
        ])
        svc = await _svc(ProviderRouter(cat))
        kb = await svc.create_kb(user=ADMIN, name="KB", visibility="public")
        return kb

    kb = asyncio.run(scenario())
    assert kb.embedding_config["provider_id"] == "dashscope"
    assert kb.embedding_config["model"] == "dashscope/text-embedding-v4"
    assert kb.embedding_config["dimension"] == 1024


def test_create_kb_records_openai_compatible_embedding_provider():
    async def scenario():
        cat = ModelCatalog(default_model="vendor/chat", providers=[
            ProviderConfig(name="vendor", base_url="https://api.vendor.com/v1", api_key="k", models=[
                ModelSpec(id="chat"),
                ModelSpec(id="bge-m3", type="embedding", dimension=512),  # protocol defaults openai
            ]),
        ])
        svc = await _svc(ProviderRouter(cat))
        return await svc.create_kb(user=ADMIN, name="KB", visibility="public")

    kb = asyncio.run(scenario())
    assert kb.embedding_config["provider_id"] == "vendor"
    assert kb.embedding_config["model"] == "vendor/bge-m3"
    assert kb.embedding_config["dimension"] == 512


def test_create_kb_falls_back_to_local_without_embedding_model():
    async def scenario():
        svc = await _svc(_chat_router())  # chat-only catalog, no embedding model
        return await svc.create_kb(user=ADMIN, name="KB", visibility="public")

    kb = asyncio.run(scenario())
    assert kb.embedding_config["provider_id"] == "local_hash"
    assert kb.embedding_config["dimension"] == 64


class FakeSearchEngine:
    """Minimal stand-in for a non-local primary engine — create_kb only reads
    ``.name`` and identity (``is self._local``) off it."""

    name = "elasticsearch"


def test_create_kb_records_active_engine_as_vector_store_provider():
    """A KB created while a non-local engine is active records that engine's
    name as its vector store provider, not the local SQL default."""

    async def scenario():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        svc = KnowledgeService(engine, search_engine=FakeSearchEngine())
        return await svc.create_kb(user=ADMIN, name="KB", visibility="public")

    kb = asyncio.run(scenario())
    assert kb.vector_store_config["provider_id"] == "elasticsearch"


def test_create_kb_defaults_vector_store_to_local_sql_without_engine():
    """With only the local SQL scan active (fresh install / tests), the KB keeps
    the local_sql default."""

    async def scenario():
        svc = await _svc()  # no search_engine → LocalSearchEngine
        return await svc.create_kb(user=ADMIN, name="KB", visibility="public")

    kb = asyncio.run(scenario())
    assert kb.vector_store_config["provider_id"] == "local_sql"


def test_update_kb_rejects_embedding_config_change():
    async def scenario():
        svc = await _svc()
        kb = await svc.create_kb(user=ADMIN, name="KB", visibility="public")
        with pytest.raises(ValueError, match="immutable"):
            await svc.update_kb(kb.id, user=ADMIN, patch={"embedding_config": {"dimension": 128}})
        # rerank_config stays mutable
        updated = await svc.update_kb(kb.id, user=ADMIN, patch={"rerank_config": {"enabled": True}})
        return updated

    updated = asyncio.run(scenario())
    assert updated.rerank_config.get("enabled") is True
