"""Upload endpoint — multipart file → extract → ingest. Offline: passthrough
(.md/.txt) needs no markitdown; the markitdown path is faked via sys.modules."""

import asyncio
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.db import create_all, make_engine
from app.deps import AppState
from app.jobs import JobQueue, register_knowledge_handlers
from app.knowledge import KnowledgeService
from app.routes.knowledge import router as knowledge_router
from app.store.memory import InMemoryStore
from tests.authutil import apply_auth


def _client(*, user_id="u_owner", role="admin"):
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    asyncio.run(create_all(engine))
    knowledge = KnowledgeService(engine)
    # Ingestion now runs on the background queue; the worker pool is not started
    # in tests — we drain synchronously via _drain() for deterministic assertions.
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
    """Run all queued jobs to completion, in-loop, and return the count.

    Uses the queue stashed on the app's state (StaticPool means create_all, the
    request loop, and this drain loop all share the one in-memory connection)."""
    queue = c.app.state.app_state.jobs
    return asyncio.run(queue.run_until_empty())


def _create_kb(c: TestClient):
    r = c.post("/v1/knowledge-bases", json={"name": "KB", "visibility": "private"})
    assert r.status_code == 200, r.text
    return r.json()


def test_upload_markdown_passthrough_ingests():
    c = _client()
    kb = _create_kb(c)
    r = c.post(
        f"/v1/knowledge-bases/{kb['id']}/documents/upload",
        files={"file": ("guide.md", b"# Guide\nsome real body text here", "text/markdown")},
        data={"tags": "docs, guide", "category": "manual"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    doc = body["document"]
    assert body["job_id"]
    # Returned immediately as a processing stub; the worker ingests on drain.
    assert doc["title"] == "guide.md"
    assert doc["source_type"] == "file"
    assert doc["status"] == "processing"
    assert "docs" in doc["tags"] and "guide" in doc["tags"]

    assert _drain(c) == 1

    # after ingestion the doc is indexed with chunks
    docs = c.get(f"/v1/knowledge-bases/{kb['id']}/documents").json()["data"]
    assert docs[0]["status"] == "indexed"
    assert docs[0]["chunk_count"] >= 1
    chunks = c.get(f"/v1/knowledge-bases/{kb['id']}/chunks")
    assert chunks.status_code == 200
    assert chunks.json()["total"] >= 1


def test_upload_uses_title_override():
    c = _client()
    kb = _create_kb(c)
    r = c.post(
        f"/v1/knowledge-bases/{kb['id']}/documents/upload",
        files={"file": ("a.txt", b"plain text content", "text/plain")},
        data={"title": "Custom Title"},
    )
    assert r.status_code == 200, r.text
    # title is set on the stub at enqueue time — no drain needed to observe it
    assert r.json()["document"]["title"] == "Custom Title"


def test_upload_unsupported_type_400():
    c = _client()
    kb = _create_kb(c)
    r = c.post(
        f"/v1/knowledge-bases/{kb['id']}/documents/upload",
        files={"file": ("data.zip", b"PK\x03\x04", "application/zip")},
    )
    assert r.status_code == 400
    assert "unsupported" in r.json()["detail"]


def test_upload_too_large_413(monkeypatch):
    c = _client()
    kb = _create_kb(c)
    # default limit is 20MB; craft a body just over it
    big = b"x" * (20 * 1024 * 1024 + 1)
    r = c.post(
        f"/v1/knowledge-bases/{kb['id']}/documents/upload",
        files={"file": ("big.txt", big, "text/plain")},
    )
    assert r.status_code == 413
    assert "limit" in r.json()["detail"]


def test_upload_pdf_via_fake_markitdown(monkeypatch):
    class _Result:
        text_content = "# From PDF\nextracted markdown body"

    class _MD:
        def convert(self, path):
            return _Result()

    mod = types.ModuleType("markitdown")
    mod.MarkItDown = _MD
    monkeypatch.setitem(sys.modules, "markitdown", mod)

    c = _client()
    kb = _create_kb(c)
    r = c.post(
        f"/v1/knowledge-bases/{kb['id']}/documents/upload",
        files={"file": ("report.pdf", b"%PDF-1.4 ...", "application/pdf")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["document"]["status"] == "processing"
    assert _drain(c) == 1
    docs = c.get(f"/v1/knowledge-bases/{kb['id']}/documents").json()["data"]
    assert docs[0]["chunk_count"] >= 1


def test_upload_support_lists_formats():
    c = _client()
    r = c.get("/v1/knowledge/upload-support")
    assert r.status_code == 200
    body = r.json()
    assert ".pdf" in body["extensions"] and ".md" in body["extensions"]
    assert body["max_mb"] == 20
