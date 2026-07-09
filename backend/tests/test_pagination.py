"""Route-level pagination: documents / chunks / search return
{data,total,offset,limit,has_more}; SQL-side filters are correct; chunk
responses never ship the embedding vector."""

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
        store=InMemoryStore(), llm=None, default_model="test/echo",
        knowledge=KnowledgeService(engine),
    )
    app.include_router(knowledge_router)
    apply_auth(app, user_id=user_id, role=role)
    return TestClient(app)


def _kb(c):
    r = c.post("/v1/knowledge-bases", json={"name": "Docs", "visibility": "public"})
    assert r.status_code == 200, r.text
    return r.json()


def _import(c, kb_id, title, content, **kw):
    r = c.post(f"/v1/knowledge-bases/{kb_id}/documents/import",
               json={"title": title, "content": content, **kw})
    assert r.status_code == 200, r.text
    return r.json()["document"]


def test_documents_pagination_and_total():
    c = _client()
    kb = _kb(c)
    for i in range(7):
        _import(c, kb["id"], f"doc {i}", f"body {i}")

    r = c.get(f"/v1/knowledge-bases/{kb['id']}/documents", params={"limit": 3, "offset": 0})
    body = r.json()
    assert body["total"] == 7
    assert len(body["data"]) == 3
    assert body["has_more"] is True

    r2 = c.get(f"/v1/knowledge-bases/{kb['id']}/documents", params={"limit": 3, "offset": 6})
    body2 = r2.json()
    assert body2["total"] == 7
    assert len(body2["data"]) == 1
    assert body2["has_more"] is False


def test_documents_sql_filters():
    c = _client()
    kb = _kb(c)
    _import(c, kb["id"], "Alpha guide", "alpha body", tags=["x"], category="cat-a")
    _import(c, kb["id"], "Beta manual", "beta body", tags=["y"], category="cat-b")
    _import(c, kb["id"], "Gamma alpha notes", "more", tags=["x"], category="cat-a")

    # query pushes to SQL (title match)
    r = c.get(f"/v1/knowledge-bases/{kb['id']}/documents", params={"query": "alpha"})
    body = r.json()
    assert body["total"] == 2
    assert {d["title"] for d in body["data"]} == {"Alpha guide", "Gamma alpha notes"}

    # tag filter
    r = c.get(f"/v1/knowledge-bases/{kb['id']}/documents", params={"tag": "y"})
    assert r.json()["total"] == 1

    # category filter
    r = c.get(f"/v1/knowledge-bases/{kb['id']}/documents", params={"category": "cat-a"})
    assert r.json()["total"] == 2


def test_chunks_pagination_and_no_embedding():
    c = _client()
    kb = _kb(c)
    _import(c, kb["id"], "Doc", "段落一。段落二。段落三。段落四。段落五。" * 3)

    r = c.get(f"/v1/knowledge-bases/{kb['id']}/chunks", params={"limit": 2, "offset": 0})
    body = r.json()
    assert "total" in body and "has_more" in body
    assert len(body["data"]) <= 2
    for ch in body["data"]:
        assert "embedding" not in ch  # never ship the vector to the UI
        assert "text" in ch


def test_search_pagination_envelope():
    c = _client()
    kb = _kb(c)
    for i in range(4):
        _import(c, kb["id"], f"doc {i}", f"共享词 shared 独有 {i}")

    r = c.post("/v1/knowledge/query/search", json={
        "kb_ids": [kb["id"]], "query": "shared", "mode": "keyword", "top_k": 2, "offset": 0,
    })
    body = r.json()
    assert body["total"] == 4
    assert len(body["data"]) == 2
    assert body["offset"] == 0 and body["limit"] == 2
    assert body["has_more"] is True

    r2 = c.post("/v1/knowledge/query/search", json={
        "kb_ids": [kb["id"]], "query": "shared", "mode": "keyword", "top_k": 2, "offset": 2,
    })
    body2 = r2.json()
    assert body2["total"] == 4
    assert body2["has_more"] is False


def test_engine_status_endpoint_local_default():
    c = _client()
    r = c.get("/v1/knowledge/engine")
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "local"
    assert body["configured"] is False
    assert body["healthy"] is True


def test_reindex_local_is_noop_count():
    c = _client()
    kb = _kb(c)
    _import(c, kb["id"], "Doc", "内容内容内容")
    r = c.post(f"/v1/knowledge-bases/{kb['id']}/reindex")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["engine"] == "local"
    assert body["reindexed"] is False
