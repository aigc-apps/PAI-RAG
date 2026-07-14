# ruff: noqa: E402
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agent_config import DEFAULT_DOCUMENT, AgentProfile
from app.agent_config_store import SqlAgentConfigStore
from app.db import create_all, make_engine
from app.deps import AppState, refresh_config_if_changed
from app.providers import ModelCatalog, ProviderRouter


def test_sql_agent_config_store_bootstraps_and_versions_document():
    async def run():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        store = SqlAgentConfigStore(engine)

        loaded = await store.load()
        assert loaded.doc.agents[0].id == "main"
        assert loaded.revision == 1

        next_doc = loaded.doc.model_copy(deep=True)
        next_doc.agents.append(AgentProfile(id="ops", name="Ops"))
        saved = await store.save(next_doc, updated_by="admin")

        assert saved.revision == 2
        assert await store.current_revision() == 2
        reloaded = await store.load()
        assert [agent.id for agent in reloaded.doc.agents] == ["main", "ops"]
        assert reloaded.checksum == saved.checksum

        revisions = await store.list_revisions()
        assert [item.revision for item in revisions] == [1, 2]
        assert revisions[-1].document["agents"][-1]["id"] == "ops"
        await engine.dispose()

    asyncio.run(run())


def test_sql_agent_config_store_imports_seed_once():
    async def run():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        seed = DEFAULT_DOCUMENT.model_copy(deep=True)
        seed.default_agent = "seeded"
        seed.agents = [AgentProfile(id="seeded", name="Seeded")]
        store = SqlAgentConfigStore(engine, seed=seed)

        assert (await store.load()).doc.default_agent == "seeded"

        changed = DEFAULT_DOCUMENT.model_copy(deep=True)
        changed.default_agent = "ignored"
        changed.agents = [AgentProfile(id="ignored", name="Ignored")]
        store_with_new_seed = SqlAgentConfigStore(engine, seed=changed)
        assert (await store_with_new_seed.load()).doc.default_agent == "seeded"
        await engine.dispose()

    asyncio.run(run())


def test_sql_agent_code_config_roundtrip():
    async def run():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        store = SqlAgentConfigStore(engine)

        loaded = await store.load()
        assert loaded.doc.agents[0].code.enabled is False
        assert loaded.doc.agents[0].code.manifest == ""

        changed = loaded.doc.model_copy(deep=True)
        changed.agents[0].code.enabled = True
        changed.agents[0].code.manifest = "- repo-a"
        await store.save(changed, updated_by="admin")

        reloaded = await store.load()
        assert reloaded.doc.agents[0].code.model_dump() == {
            "enabled": True,
            "manifest": "- repo-a",
        }
        await engine.dispose()

    asyncio.run(run())


def test_refresh_config_if_changed_rebuilds_runtime_from_new_revision(monkeypatch):
    async def run():
        monkeypatch.setenv("CONFIG_RELOAD_INTERVAL_SECONDS", "0")
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        store = SqlAgentConfigStore(engine)
        first = await store.load()
        state = AppState(
            store=None,
            llm=None,
            default_model="openai/gpt-4o-mini",
            router=ProviderRouter(ModelCatalog(**first.doc.models)),
            config_store=store,
            config_revision=first.revision,
        )

        changed = first.doc.model_copy(deep=True)
        changed.agents[0].name = "Changed Elsewhere"
        await store.save(changed, updated_by="other-instance")

        await refresh_config_if_changed(state)

        assert state.config_revision == 2
        assert state.agent_config.agents[0].name == "Changed Elsewhere"
        await engine.dispose()

    asyncio.run(run())
