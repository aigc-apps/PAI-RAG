import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

import app.datasource.adapters.yuque as yuque_mod
from app.datasource.adapters.yuque import YuqueAdapter
from app.datasource.registry import get_adapter, supported_source_types
from app.db import create_all, make_engine
from app.knowledge import KnowledgeService
from app.store.base import User

TOKEN_ENV = "YUQUE_TEST_TOKEN"
GROUP, BOOK = "acme", "handbook"
CFG = {"group_login": GROUP, "book_slug": BOOK, "token_env": TOKEN_ENV}

ADMIN = User(id="u_owner", email="owner@example.com", role="admin",
             status="active", display_name=None)


# --------------------------------------------------------------------------- #
# A fake Yuque book: a TOC tree (two TITLE groups) + doc bodies. Mutable so a
# test can change/remove docs between syncs and assert the diff.
# --------------------------------------------------------------------------- #
def _make_world():
    return {
        "toc": [
            {"type": "TITLE", "uuid": "g1", "parent_uuid": "", "title": "指南", "slug": ""},
            {"type": "DOC", "uuid": "d1", "parent_uuid": "g1", "title": "安装", "slug": "install"},
            {"type": "DOC", "uuid": "d2", "parent_uuid": "g1", "title": "概述", "slug": "overview"},
            {"type": "TITLE", "uuid": "g2", "parent_uuid": "", "title": "进阶", "slug": ""},
            {"type": "DOC", "uuid": "d3", "parent_uuid": "g2", "title": "调优", "slug": "tuning"},
        ],
        "bodies": {
            "install": "# 安装\n\n安装步骤。\n",
            "overview": "# 概述\n\n平台概述。\n",
            "tuning": "# 调优\n\n性能调优。\n",
        },
    }


def _install_fake(monkeypatch, world: dict):
    def fake_get(url: str, token: str, timeout: int = 30):
        assert token == "sekret"  # adapter resolved token_env → this value
        if url.endswith("/toc"):
            return world["toc"]
        if "/docs/" in url:
            slug = url.rsplit("/docs/", 1)[1]
            if slug not in world["bodies"]:
                raise RuntimeError(f"404 for {slug}")
            return {"slug": slug, "body": world["bodies"][slug]}
        raise RuntimeError(f"unexpected url {url}")

    monkeypatch.setattr(yuque_mod, "yuque_get_json", fake_get)
    monkeypatch.setenv(TOKEN_ENV, "sekret")
    return world


def _adapter(cfg: dict) -> YuqueAdapter:
    return YuqueAdapter(datasource_key="yuque:test", source_config=cfg)


# --------------------------------------------------------------------------- #
# validate_config (no network)
# --------------------------------------------------------------------------- #
def test_validate_config_requires_fields_and_env(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    for missing in ("group_login", "book_slug", "token_env"):
        cfg = dict(CFG)
        cfg.pop(missing)
        with pytest.raises(ValueError):
            _adapter(cfg).validate_config()
    # all fields present but env var not set → still rejected
    with pytest.raises(ValueError):
        _adapter(dict(CFG)).validate_config()
    # env set → passes
    monkeypatch.setenv(TOKEN_ENV, "sekret")
    _adapter(dict(CFG)).validate_config()


def test_registered_in_registry():
    assert "yuque" in supported_source_types()
    assert isinstance(get_adapter("yuque", "k", dict(CFG)), YuqueAdapter)


# --------------------------------------------------------------------------- #
# discover + subtree selection
# --------------------------------------------------------------------------- #
def test_discover_whole_book(monkeypatch):
    _install_fake(monkeypatch, _make_world())
    docs = _adapter(dict(CFG)).discover()
    assert {d.path for d in docs} == {"install", "overview", "tuning"}
    install = next(d for d in docs if d.path == "install")
    assert install.section == "指南"  # parent TITLE
    assert install.source_url == f"https://www.yuque.com/{GROUP}/{BOOK}/install"


def test_discover_path_by_title_selects_subtree(monkeypatch):
    _install_fake(monkeypatch, _make_world())
    docs = _adapter({**CFG, "path": "进阶"}).discover()
    assert {d.path for d in docs} == {"tuning"}


def test_discover_path_by_slug_selects_single_doc(monkeypatch):
    _install_fake(monkeypatch, _make_world())
    docs = _adapter({**CFG, "path": "install"}).discover()
    assert {d.path for d in docs} == {"install"}


def test_discover_roots_multi_select(monkeypatch):
    _install_fake(monkeypatch, _make_world())
    docs = _adapter({**CFG, "roots": [{"title": "指南"}]}).discover()
    assert {d.path for d in docs} == {"install", "overview"}


# --------------------------------------------------------------------------- #
# fetch + emit
# --------------------------------------------------------------------------- #
def test_fetch_and_emit(monkeypatch):
    _install_fake(monkeypatch, _make_world())
    adapter = _adapter(dict(CFG))
    doc = next(d for d in adapter.discover() if d.path == "install")
    body = adapter.fetch(doc)
    assert "安装步骤" in body
    sd = adapter.emit(doc, body)
    assert sd.doc_id == "yuque:test/install"
    assert sd.source_url == f"https://www.yuque.com/{GROUP}/{BOOK}/install"
    assert sd.fetched_from == "yuque-openapi"
    assert "安装步骤" in sd.content
    assert sd.content_hash


# --------------------------------------------------------------------------- #
# end-to-end sync via KnowledgeService (add / update / delete diff)
# --------------------------------------------------------------------------- #
def test_sync_ingests_updates_and_deletes(monkeypatch):
    world = _install_fake(monkeypatch, _make_world())

    async def scenario():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        svc = KnowledgeService(engine)
        kb = await svc.create_kb(user=ADMIN, name="Yuque KB")
        ds = await svc.create_data_source(
            kb.id, user=ADMIN, name="团队手册",
            source_type="yuque", source_config=dict(CFG),
        )
        assert ds.source_type == "yuque"

        ds = await svc.sync_data_source(kb.id, ds.id, user=ADMIN)
        assert ds.status == "succeeded", ds.last_error
        assert ds.doc_count == 3
        assert ds.last_sync_report["added"] == 3

        # nothing changed → all unchanged
        ds = await svc.sync_data_source(kb.id, ds.id, user=ADMIN)
        assert ds.last_sync_report["unchanged"] == 3

        # change one body + drop one node → 1 updated, 1 deleted, 1 unchanged
        world["bodies"]["install"] = "# 安装\n\n全新的安装步骤。\n"
        world["toc"] = [n for n in world["toc"] if n["uuid"] != "d3"]
        ds = await svc.sync_data_source(kb.id, ds.id, user=ADMIN)
        rep = ds.last_sync_report
        assert rep["updated"] == 1
        assert rep["deleted"] == 1
        assert rep["unchanged"] == 1
        assert ds.doc_count == 2

    asyncio.run(scenario())


def test_token_never_stored_in_config(monkeypatch):
    """The token itself must never live in source_config — only its env-var name."""
    _install_fake(monkeypatch, _make_world())

    async def scenario():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        svc = KnowledgeService(engine)
        kb = await svc.create_kb(user=ADMIN, name="Yuque KB")
        ds = await svc.create_data_source(
            kb.id, user=ADMIN, name="手册",
            source_type="yuque", source_config=dict(CFG),
        )
        assert "sekret" not in str(ds.source_config)
        assert ds.source_config["token_env"] == TOKEN_ENV

    asyncio.run(scenario())
