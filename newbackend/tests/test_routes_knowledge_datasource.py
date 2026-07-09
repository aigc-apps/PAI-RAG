import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.datasource.adapters.llms_txt as llms_mod
from app.datasource.adapters.llms_txt import parse_llms_txt, parse_product_title
from app.datasource.url_guard import UrlNotAllowed, validate_public_url
from app.db import create_all, make_engine
from app.deps import AppState
from app.knowledge import KnowledgeService
from app.routes.knowledge import router as knowledge_router
from app.store.base import User
from app.store.memory import InMemoryStore
from tests.authutil import apply_auth


# --------------------------------------------------------------------------- #
# A fake help.aliyun.com "world" the llms_txt adapter fetches from. Mutable so a
# test can add/change/remove docs between syncs and assert the diff.
# --------------------------------------------------------------------------- #
LLMS_URL = "https://help.aliyun.com/zh/pai/llms.txt"


def _manifest(entries: list[tuple[str, str, str]]) -> str:
    # entries: (section, title, slug) → one `- [title](.../slug.md): summary` line
    lines = ["# PAI Docs", "", "## 快速开始"]
    for section, title, slug in entries:
        lines.append(f"- [{title}](https://help.aliyun.com/zh/pai/{slug}.md): {title}摘要")
    return "\n".join(lines) + "\n"


def _make_world():
    return {
        "manifest": _manifest([
            ("快速开始", "安装", "install"),
            ("快速开始", "概述", "overview"),
            ("快速开始", "调优", "tuning"),
        ]),
        "bodies": {
            "https://help.aliyun.com/zh/pai/install.md": "# 安装\n\n安装 PAI 的步骤。\n",
            "https://help.aliyun.com/zh/pai/overview.md": "# 概述\n\nPAI 平台概述。\n",
            "https://help.aliyun.com/zh/pai/tuning.md": "# 调优\n\n性能调优指南。\n",
        },
    }


def _install_fake(monkeypatch, world: dict):
    def fake_http_get(url: str, timeout: int = 30) -> str:
        if url.rstrip("/").endswith("llms.txt"):
            return world["manifest"]
        if url in world["bodies"]:
            return world["bodies"][url]
        raise RuntimeError(f"404 for {url}")

    monkeypatch.setattr(llms_mod, "http_get", fake_http_get)
    return world


ADMIN = User(id="u_owner", email="owner@example.com", role="admin", status="active", display_name=None)


# --------------------------------------------------------------------------- #
# Unit tests: parser + SSRF guard (no network, no DB)
# --------------------------------------------------------------------------- #
def test_parse_llms_txt_groups_by_section():
    text = (
        "# PAI Docs\n"
        "## 快速开始\n"
        "- [安装](https://help.aliyun.com/zh/pai/install.md): 安装指南\n"
        "- [概述](https://help.aliyun.com/zh/pai/overview.md)\n"
        "## 进阶\n"
        "- [调优](https://help.aliyun.com/zh/pai/tuning.md): 性能调优\n"
    )
    items = parse_llms_txt(text)
    assert len(items) == 3
    assert items[0] == {
        "section": "快速开始",
        "title": "安装",
        "url": "https://help.aliyun.com/zh/pai/install.md",
        "summary": "安装指南",
    }
    assert items[1]["summary"] == ""  # no summary → empty
    assert items[2]["section"] == "进阶"
    assert parse_product_title(text) == "PAI Docs"


def test_validate_public_url_blocks_private_and_non_http():
    for bad in [
        "http://127.0.0.1/x",
        "http://10.0.0.1/x",
        "http://169.254.169.254/latest/meta-data",  # cloud metadata
        "ftp://example.com/x",
        "https:///nohost",
    ]:
        with pytest.raises(UrlNotAllowed):
            validate_public_url(bad)
    # a numeric public IP validates without touching DNS
    validate_public_url("http://8.8.8.8/")


# --------------------------------------------------------------------------- #
# Service-level sync: deterministic, single event loop
# --------------------------------------------------------------------------- #
def test_sync_ingests_updates_and_deletes(monkeypatch):
    world = _install_fake(monkeypatch, _make_world())

    async def scenario():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        svc = KnowledgeService(engine)
        kb = await svc.create_kb(user=ADMIN, name="PAI KB")
        ds = await svc.create_data_source(
            kb.id, user=ADMIN, name="Aliyun PAI 文档",
            source_type="llms_txt", source_config={"product": "pai"},
        )
        assert ds.source_type == "llms_txt"
        assert ds.status == "idle"

        # first sync: 3 added
        ds = await svc.sync_data_source(kb.id, ds.id, user=ADMIN)
        assert ds.status == "succeeded", ds.last_error
        assert ds.doc_count == 3
        assert ds.last_sync_report["added"] == 3
        assert ds.last_sync_report["deleted"] == 0

        docs, docs_total = await svc.list_documents(kb.id, user=ADMIN)
        assert len(docs) == 3
        assert docs_total == 3
        assert {d.source_type for d in docs} == {"llms_txt"}
        assert all(d.source_id == ds.id for d in docs)
        install = next(d for d in docs if d.uri == "https://help.aliyun.com/zh/pai/install")
        assert install.custom_metadata["product"] == "PAI Docs"
        assert install.custom_metadata["section"] == "快速开始"
        # chunks were produced
        chunks, _ = await svc.list_chunks(kb.id, user=ADMIN, document_id=install.id)
        assert chunks

        # second sync, nothing changed: all unchanged, no re-add
        ds = await svc.sync_data_source(kb.id, ds.id, user=ADMIN)
        assert ds.last_sync_report["unchanged"] == 3
        assert ds.last_sync_report["added"] == 0
        assert ds.last_sync_report["updated"] == 0
        assert ds.doc_count == 3

        # change one body + drop one entry → 1 updated, 1 deleted, 1 unchanged
        world["bodies"]["https://help.aliyun.com/zh/pai/install.md"] = "# 安装\n\n全新的安装步骤。\n"
        world["manifest"] = _manifest([
            ("快速开始", "安装", "install"),
            ("快速开始", "概述", "overview"),
        ])
        ds = await svc.sync_data_source(kb.id, ds.id, user=ADMIN)
        assert ds.status == "succeeded", ds.last_error
        rep = ds.last_sync_report
        assert rep["updated"] == 1
        assert rep["deleted"] == 1
        assert rep["unchanged"] == 1
        assert ds.doc_count == 2

        live, _ = await svc.list_documents(kb.id, user=ADMIN)
        uris = {d.uri for d in live}
        assert "https://help.aliyun.com/zh/pai/tuning" not in uris  # deleted
        assert len(live) == 2

    asyncio.run(scenario())


def test_sync_partial_on_fetch_error(monkeypatch):
    world = _make_world()
    # one md url is missing from bodies → its fetch raises → failed, not deleted-by-mistake
    del world["bodies"]["https://help.aliyun.com/zh/pai/tuning.md"]
    _install_fake(monkeypatch, world)

    async def scenario():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        svc = KnowledgeService(engine)
        kb = await svc.create_kb(user=ADMIN, name="PAI KB")
        ds = await svc.create_data_source(
            kb.id, user=ADMIN, name="Aliyun PAI",
            source_type="llms_txt", source_config={"product": "pai"},
        )
        ds = await svc.sync_data_source(kb.id, ds.id, user=ADMIN)
        assert ds.status == "partial"
        assert ds.last_sync_report["added"] == 2
        assert ds.last_sync_report["failed"] == 1
        assert ds.doc_count == 2

    asyncio.run(scenario())


def test_create_rejects_bad_config(monkeypatch):
    async def scenario():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        svc = KnowledgeService(engine)
        kb = await svc.create_kb(user=ADMIN, name="KB")
        # neither product nor llms_url
        with pytest.raises(ValueError):
            await svc.create_data_source(
                kb.id, user=ADMIN, name="bad", source_type="llms_txt", source_config={},
            )
        # unsupported type
        with pytest.raises(ValueError):
            await svc.create_data_source(
                kb.id, user=ADMIN, name="bad2", source_type="nope", source_config={},
            )

    asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# Route-level: CRUD + fire-and-forget sync (polled) + isolation
# --------------------------------------------------------------------------- #
def _client(*, user_id="u_owner", role="admin", db_url="sqlite+aiosqlite:///:memory:"):
    engine = make_engine(db_url)
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


def _create_kb(c):
    r = c.post("/v1/knowledge-bases", json={"name": "PAI Docs", "visibility": "private"})
    assert r.status_code == 200, r.text
    return r.json()


def test_datasource_crud_routes():
    c = _client()
    kb = _create_kb(c)
    # create
    r = c.post(
        f"/v1/knowledge-bases/{kb['id']}/datasources",
        json={"name": "Aliyun PAI", "source_type": "llms_txt", "source_config": {"product": "pai"}},
    )
    assert r.status_code == 200, r.text
    ds = r.json()
    assert ds["source_key"] == "aliyun-pai"
    assert ds["status"] == "idle"
    # list
    lst = c.get(f"/v1/knowledge-bases/{kb['id']}/datasources")
    assert lst.status_code == 200
    assert len(lst.json()["data"]) == 1
    # get
    got = c.get(f"/v1/knowledge-bases/{kb['id']}/datasources/{ds['id']}")
    assert got.status_code == 200
    # patch
    patched = c.patch(
        f"/v1/knowledge-bases/{kb['id']}/datasources/{ds['id']}",
        json={"name": "Aliyun PAI 文档", "enabled": False},
    )
    assert patched.status_code == 200
    assert patched.json()["name"] == "Aliyun PAI 文档"
    assert patched.json()["enabled"] is False
    # sync a disabled source → 400
    denied = c.post(f"/v1/knowledge-bases/{kb['id']}/datasources/{ds['id']}/sync")
    assert denied.status_code == 400
    # bad config on create → 400
    bad = c.post(
        f"/v1/knowledge-bases/{kb['id']}/datasources",
        json={"name": "bad", "source_type": "llms_txt", "source_config": {}},
    )
    assert bad.status_code == 400
    # delete
    assert c.delete(f"/v1/knowledge-bases/{kb['id']}/datasources/{ds['id']}").status_code == 200
    assert c.get(f"/v1/knowledge-bases/{kb['id']}/datasources").json()["data"] == []


def test_sync_route_accepts_and_returns_202():
    # The sync endpoint is fire-and-forget: it validates permission/state
    # synchronously and returns 202. The ingestion itself is covered end-to-end
    # by the service-level tests above (the fire-and-forget task cannot be driven
    # to completion under Starlette's synchronous TestClient portal loop, so here
    # we stub it to a no-op and assert only the route's contract).
    c = _client()
    kb = _create_kb(c)

    async def _noop_sync(*args, **kwargs):
        return None

    c.app.state.app_state.knowledge.sync_data_source = _noop_sync

    # syncing a non-existent source → 404 (checked before we spawn any task)
    missing = c.post(f"/v1/knowledge-bases/{kb['id']}/datasources/ds_missing/sync")
    assert missing.status_code == 404
    ds = c.post(
        f"/v1/knowledge-bases/{kb['id']}/datasources",
        json={"name": "Aliyun PAI", "source_type": "llms_txt", "source_config": {"product": "pai"}},
    ).json()
    started = c.post(f"/v1/knowledge-bases/{kb['id']}/datasources/{ds['id']}/sync")
    assert started.status_code == 202, started.text
    body = started.json()
    assert body["status"] == "syncing"
    assert body["data_source"]["id"] == ds["id"]
    # a disabled source cannot be synced → 400
    c.patch(f"/v1/knowledge-bases/{kb['id']}/datasources/{ds['id']}", json={"enabled": False})
    disabled = c.post(f"/v1/knowledge-bases/{kb['id']}/datasources/{ds['id']}/sync")
    assert disabled.status_code == 400


def test_datasource_private_kb_isolation():
    owner = _client(user_id="u_owner", role="user")
    kb = _create_kb(owner)
    owner.post(
        f"/v1/knowledge-bases/{kb['id']}/datasources",
        json={"name": "Aliyun PAI", "source_type": "llms_txt", "source_config": {"product": "pai"}},
    )
    other = TestClient(owner.app)
    apply_auth(other.app, user_id="u_other", role="user")
    denied = other.get(f"/v1/knowledge-bases/{kb['id']}/datasources")
    assert denied.status_code in (403, 404)
