import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db import create_all, make_engine  # noqa: E402
from app.deps import AppState  # noqa: E402
from app.jobs import JobQueue, register_knowledge_handlers  # noqa: E402
from app.knowledge import KnowledgeService  # noqa: E402
from app.routes.knowledge import router as knowledge_router  # noqa: E402
from app.store.memory import InMemoryStore  # noqa: E402
from tests.authutil import apply_auth  # noqa: E402


def _client(*, user_id="u_owner", role="admin"):
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    asyncio.run(create_all(engine))
    knowledge = KnowledgeService(engine)
    # Import now enqueues; the worker pool isn't started in tests — we drain
    # synchronously via _drain() so ingestion is observed deterministically.
    queue = JobQueue(engine, concurrency=1)
    register_knowledge_handlers(queue, knowledge)
    app = FastAPI()
    app.state.app_state = AppState(
        store=InMemoryStore(),
        llm=None,
        default_model="test/echo",
        knowledge=knowledge,
        jobs=queue,
    )
    app.include_router(knowledge_router)
    apply_auth(app, user_id=user_id, role=role)
    return TestClient(app)


def _drain(c: TestClient) -> int:
    """Run all queued jobs to completion in-loop; returns the count processed.
    StaticPool shares the one in-memory connection across the create_all, request,
    and drain event loops."""
    return asyncio.run(c.app.state.app_state.jobs.run_until_empty())


def _create_kb(c: TestClient):
    r = c.post(
        "/v1/knowledge-bases",
        json={
            "name": "PAI Docs",
            "description": "Product docs",
            "visibility": "private",
        },
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_list_knowledge_bases_serializes_knowledge_base_rows():
    c = _client()
    kb = _create_kb(c)

    response = c.get("/v1/knowledge-bases")

    assert response.status_code == 200, response.text
    assert response.json()["data"][0]["id"] == kb["id"]
    assert "active_job_id" not in response.json()["data"][0]


def test_knowledge_base_import_search_and_fetch():
    c = _client()
    kb = _create_kb(c)

    imported = c.post(
        f"/v1/knowledge-bases/{kb['id']}/documents/import",
        json={
            "title": "EAS Quickstart",
            "uri": "https://docs.example.com/eas/quickstart",
            "source_type": "website",
            "content": (
                "EAS lets you deploy online inference services. "
                "Create a service by preparing a model and choosing an instance type. "
                "Billing depends on resource usage."
            ),
            "tags": ["pai", "eas"],
            "category": "product-docs",
        },
    )
    assert imported.status_code == 202, imported.text
    doc = imported.json()["document"]
    assert doc["status"] == "processing"

    # ingestion runs on the queue; drain it, then the doc is indexed with chunks
    assert _drain(c) == 1
    docs = c.get(f"/v1/knowledge-bases/{kb['id']}/documents", params={"tag": "eas"})
    assert docs.status_code == 200
    indexed = docs.json()["data"][0]
    assert indexed["title"] == "EAS Quickstart"
    assert indexed["status"] == "indexed"
    assert indexed["chunk_count"] >= 1

    chunks = c.get(f"/v1/knowledge-bases/{kb['id']}/chunks")
    assert chunks.status_code == 200
    chunk_id = chunks.json()["data"][0]["id"]

    search = c.post(
        "/v1/knowledge/query/search",
        json={
            "kb_ids": [kb["id"]],
            "query": "How do I create an EAS service?",
            "mode": "hybrid",
            "top_k": 3,
            "filters": {"tags": ["eas"]},
        },
    )
    assert search.status_code == 200, search.text
    hits = search.json()["data"]
    assert hits
    assert hits[0]["document_id"] == doc["id"]
    assert "Create a service" in hits[0]["text"]

    fetched = c.post(
        "/v1/knowledge/query/fetch",
        json={
            "kb_id": kb["id"],
            "ref": {"chunk_id": chunk_id},
            "mode": "chunk",
            "max_chars": 200,
        },
    )
    assert fetched.status_code == 200
    assert "online inference" in fetched.json()["text"]


def test_private_kb_is_not_queryable_by_other_user():
    owner = _client(user_id="u_owner", role="user")
    kb = _create_kb(owner)
    r = owner.post(
        f"/v1/knowledge-bases/{kb['id']}/documents/import",
        json={"title": "Secret", "content": "private deployment note"},
    )
    assert r.status_code == 202
    assert _drain(owner) == 1

    other = TestClient(owner.app)
    apply_auth(other.app, user_id="u_other", role="user")
    denied = other.get(f"/v1/knowledge-bases/{kb['id']}")
    assert denied.status_code in (403, 404)
    search = other.post(
        "/v1/knowledge/query/search",
        json={"kb_ids": [kb["id"]], "query": "private"},
    )
    assert search.status_code == 200
    assert search.json()["data"] == []


def test_disable_chunk_removes_it_from_search():
    c = _client()
    kb = _create_kb(c)
    c.post(
        f"/v1/knowledge-bases/{kb['id']}/documents/import",
        json={"title": "Billing", "content": "billing token unique-delete-me"},
    )
    assert _drain(c) == 1
    chunks = c.get(f"/v1/knowledge-bases/{kb['id']}/chunks").json()["data"]
    chunk_id = chunks[0]["id"]
    assert c.post(
        f"/v1/knowledge-bases/{kb['id']}/chunks/{chunk_id}/disable",
        json={"reason": "bad chunk"},
    ).status_code == 200
    search = c.post(
        "/v1/knowledge/query/search",
        json={"kb_ids": [kb["id"]], "query": "unique-delete-me", "mode": "keyword"},
    )
    assert search.status_code == 200
    assert search.json()["data"] == []


# ---- Embedding/rerank model selection at KB create + update ----

from app.providers import ModelCatalog, ModelSpec, ProviderConfig, ProviderRouter  # noqa: E402


def _router():
    cat = ModelCatalog(
        default_model="dashscope/qwen3.7-plus",
        default_embedding_model="dashscope/text-embedding-v4",
        default_rerank_model="dashscope/qwen3-rerank",
        providers=[
            ProviderConfig(name="dashscope", base_url="https://ds/v1", api_key="k", models=[
                ModelSpec(id="qwen3.7-plus"),
                ModelSpec(id="text-embedding-v4", type="embedding", protocol="dashscope", dimension=1024),
                ModelSpec(id="qwen3-rerank", type="rerank", protocol="dashscope"),
            ]),
        ],
    )
    return ProviderRouter(cat)


def _client_with_router(*, user_id="u_owner", role="admin"):
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    asyncio.run(create_all(engine))
    knowledge = KnowledgeService(engine, router=_router())
    queue = JobQueue(engine, concurrency=1)
    register_knowledge_handlers(queue, knowledge)
    app = FastAPI()
    app.state.app_state = AppState(
        store=InMemoryStore(), llm=None, default_model="test/echo",
        knowledge=knowledge, jobs=queue,
    )
    app.include_router(knowledge_router)
    apply_auth(app, user_id=user_id, role=role)
    return TestClient(app)


def test_create_kb_with_embedding_model_freezes_config():
    c = _client_with_router()
    r = c.post("/v1/knowledge-bases", json={
        "name": "KB", "embedding_model": "dashscope/text-embedding-v4"})
    assert r.status_code == 200, r.text
    emb = r.json()["embedding_config"]
    assert emb["model"] == "dashscope/text-embedding-v4"
    assert emb["provider_id"] == "dashscope"
    assert emb["dimension"] == 1024


def test_create_kb_default_embedding_uses_catalog_when_router_present():
    # No embedding_model chosen -> inherits the catalog default (not local hash).
    c = _client_with_router()
    r = c.post("/v1/knowledge-bases", json={"name": "KB"})
    assert r.status_code == 200, r.text
    assert r.json()["embedding_config"]["model"] == "dashscope/text-embedding-v4"


def test_create_kb_default_embedding_is_local_hash_without_router():
    c = _client()  # no router configured
    r = c.post("/v1/knowledge-bases", json={"name": "KB"})
    assert r.status_code == 200, r.text
    assert r.json()["embedding_config"]["model"] == "local-hash-v1"


def test_create_kb_unknown_embedding_model_400():
    c = _client_with_router()
    r = c.post("/v1/knowledge-bases", json={"name": "KB", "embedding_model": "dashscope/nope"})
    assert r.status_code == 400
    assert "unknown embedding model" in r.json()["detail"]


def test_create_kb_wrong_type_embedding_model_400():
    c = _client_with_router()
    r = c.post("/v1/knowledge-bases", json={"name": "KB", "embedding_model": "dashscope/qwen3-rerank"})
    assert r.status_code == 400
    assert "not an embedding model" in r.json()["detail"]


def test_update_kb_rerank_model_and_toggle():
    c = _client_with_router()
    kb = c.post("/v1/knowledge-bases", json={"name": "KB"}).json()
    # rerank off by default
    assert kb["rerank_config"].get("enabled") in (False, None)

    # choose a rerank model
    r = c.patch(f"/v1/knowledge-bases/{kb['id']}", json={
        "rerank_model": "dashscope/qwen3-rerank", "rerank_top_n": 8})
    assert r.status_code == 200, r.text
    rc = r.json()["rerank_config"]
    assert rc["enabled"] is True
    assert rc["model"] == "dashscope/qwen3-rerank"
    assert rc["top_n"] == 8

    # disable without re-sending the model (merge keeps the model)
    r2 = c.patch(f"/v1/knowledge-bases/{kb['id']}", json={"rerank_enabled": False})
    assert r2.status_code == 200, r2.text
    rc2 = r2.json()["rerank_config"]
    assert rc2["enabled"] is False
    assert rc2["model"] == "dashscope/qwen3-rerank"


def test_update_kb_bad_rerank_model_400():
    c = _client_with_router()
    kb = c.post("/v1/knowledge-bases", json={"name": "KB"}).json()
    r = c.patch(f"/v1/knowledge-bases/{kb['id']}", json={"rerank_model": "dashscope/text-embedding-v4"})
    assert r.status_code == 400
    assert "not a rerank model" in r.json()["detail"]
