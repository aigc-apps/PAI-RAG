import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.db import create_all, make_engine
from app.deps import AppState
from app.knowledge import KnowledgeService
from app.routes.knowledge import router as knowledge_router
from app.store.memory import InMemoryStore
from tests.authutil import apply_auth


def _client(*, user_id="u_owner", role="admin"):
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    asyncio.run(create_all(engine))
    app = FastAPI()
    app.state.app_state = AppState(
        store=InMemoryStore(),
        llm=None,
        default_model="test/echo",
        knowledge=KnowledgeService(engine),
    )
    app.include_router(knowledge_router)
    apply_auth(app, user_id=user_id, role=role)
    return TestClient(app)


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
    assert imported.status_code == 200, imported.text
    doc = imported.json()["document"]
    assert doc["status"] == "indexed"
    assert doc["chunk_count"] >= 1

    docs = c.get(f"/v1/knowledge-bases/{kb['id']}/documents", params={"tag": "eas"})
    assert docs.status_code == 200
    assert docs.json()["data"][0]["title"] == "EAS Quickstart"

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
    assert r.status_code == 200

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
