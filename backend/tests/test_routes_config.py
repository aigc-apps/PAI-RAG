# ruff: noqa: E402
import os
import sys
import io
import zipfile
import asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.testclient import TestClient
import yaml

from app.deps import AppState
from app.agent_config import AgentConfigDocument, apply_runtime_status
from app.providers import ModelCatalog, ModelSpec, ProviderConfig, ProviderRouter
from app.routes.config import router as config_router
from app.store.memory import InMemoryStore
from app.db import create_all, make_engine
from app.agent_config_store import SqlAgentConfigStore
from tests.authutil import apply_auth


def _client(tmp_path, monkeypatch):
    config_path = str(tmp_path / "config.yaml")
    monkeypatch.setenv("CONFIG_PATH", config_path)
    monkeypatch.setenv("MODELS_PATH", config_path)
    cat = ModelCatalog(
        default_model="local/fast",
        providers=[
            ProviderConfig(
                name="local",
                base_url="http://localhost:8000/v1",
                models=[ModelSpec(id="fast")],
            )
        ],
    )
    app = FastAPI()
    app.state.app_state = AppState(
        store=InMemoryStore(),
        llm=None,
        default_model="local/fast",
        router=ProviderRouter(cat),
    )
    app.include_router(config_router)
    return TestClient(apply_auth(app))


def _client_with_sql_config(tmp_path, monkeypatch):
    config_path = str(tmp_path / "seed.yaml")
    monkeypatch.setenv("CONFIG_PATH", config_path)
    monkeypatch.setenv("MODELS_PATH", config_path)
    engine = make_engine("sqlite+aiosqlite:///:memory:")

    async def init():
        await create_all(engine)

    asyncio.run(init())
    cat = ModelCatalog(
        default_model="local/fast",
        providers=[
            ProviderConfig(
                name="local",
                base_url="http://localhost:8000/v1",
                models=[ModelSpec(id="fast")],
            )
        ],
    )
    app = FastAPI()
    app.state.app_state = AppState(
        store=InMemoryStore(),
        llm=None,
        default_model="local/fast",
        router=ProviderRouter(cat),
        config_store=SqlAgentConfigStore(engine),
    )
    app.include_router(config_router)
    return TestClient(apply_auth(app)), engine, config_path


class _FakeChunk:
    """A stream chunk with no error_message → the endpoint reads it as success."""

    def __init__(self, delta=""):
        self.delta = delta


class _FakeLLM:
    def astream(self, messages, tools=None, **kwargs):
        async def gen():
            yield _FakeChunk("pong")
        return gen()


class _AsyncFakeLLM:
    async def astream(self, messages, tools=None, **kwargs):
        async def gen():
            yield _FakeChunk("pong")
        return gen()


def _client_and_router(tmp_path, monkeypatch):
    config_path = str(tmp_path / "config.yaml")
    monkeypatch.setenv("CONFIG_PATH", config_path)
    monkeypatch.setenv("MODELS_PATH", config_path)
    cat = ModelCatalog(
        default_model="local/fast",
        providers=[
            ProviderConfig(
                name="local",
                base_url="http://localhost:8000/v1",
                models=[ModelSpec(id="fast")],
            )
        ],
    )
    router = ProviderRouter(cat)
    app = FastAPI()
    app.state.app_state = AppState(
        store=InMemoryStore(), llm=None, default_model="local/fast", router=router
    )
    app.include_router(config_router)
    return TestClient(apply_auth(app)), router


def test_model_test_endpoint_reports_success_for_a_reachable_model(tmp_path, monkeypatch):
    c, router = _client_and_router(tmp_path, monkeypatch)
    # Inject a fake client so no real network hop is made.
    router.register_llm("local/fast", _FakeLLM())
    r = c.post("/v1/config/models/test", json={"model": "local/fast"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "local/fast" in body["output"]


def test_model_test_endpoint_supports_async_astream_contract(tmp_path, monkeypatch):
    c, router = _client_and_router(tmp_path, monkeypatch)
    # Real LeanLLM.astream is async and returns an async iterator after awaiting.
    router.register_llm("local/fast", _AsyncFakeLLM())
    r = c.post("/v1/config/models/test", json={"model": "local/fast"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "local/fast" in body["output"]


def test_model_test_endpoint_defaults_to_the_deployment_default(tmp_path, monkeypatch):
    c, router = _client_and_router(tmp_path, monkeypatch)
    router.register_llm("local/fast", _FakeLLM())
    # No model in the body → falls back to router.default_model_id ("local/fast").
    r = c.post("/v1/config/models/test", json={})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_model_test_endpoint_fails_gracefully_for_unknown_model(tmp_path, monkeypatch):
    c, _router = _client_and_router(tmp_path, monkeypatch)
    r = c.post("/v1/config/models/test", json={"model": "local/nope"})
    # Never raises — a bad ref comes back as ok:false with a reason.
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["output"]


def test_get_setup_returns_default_config(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    body = c.get("/v1/setup").json()
    assert body["setup"]["completed"] is False
    caps = {item["id"]: item for item in body["capabilities"]}
    assert caps["knowledge"]["status"] == "ready"
    assert caps["search"]["status"] in {"disabled", "missing_config"}
    assert any(p["id"] == "llm.default" for p in body["providers"])
    assert body["default_agent"] == "main"
    assert body["agents"][0]["id"] == "main"
    # A blank agent model means "inherit the deployment default" — the runtime no
    # longer back-fills it (which would freeze it on the next save). The resolved
    # default is surfaced on the llm.default provider for the UI to show as a hint.
    assert body["agents"][0]["model"] == ""
    llm = next(p for p in body["providers"] if p["id"] == "llm.default")
    assert llm["settings"]["default_model"] == "local/fast"
    assert body["models"]["default_model"] == "openai/gpt-4o-mini"
    assert "skill.web_research" not in {item["id"] for item in body["capabilities"]}


def test_put_setup_persists_completion(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    r = c.put(
        "/v1/setup",
        json={
            "completed": True,
            "mode": "local_first",
            "skipped_steps": ["search", "sandbox"],
        },
    )
    assert r.status_code == 200
    body = c.get("/v1/setup").json()
    assert body["setup"]["completed"] is True
    assert body["setup"]["mode"] == "local_first"
    assert body["setup"]["completed_at"]


def test_config_yaml_roundtrip_and_registry_reload(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    body = c.get("/v1/config.yaml")
    assert body.status_code == 200
    text = body.text
    assert "models:" in text
    assert "capabilities:" in text
    assert "agents:" in text
    config = yaml.safe_load(text)
    search = next(cap for cap in config["capabilities"] if cap["id"] == "search")
    search["enabled"] = True
    search_provider = next(provider for provider in config["providers"] if provider["id"] == "search.default")
    search_provider["settings"]["provider"] = "tavily"
    search_provider["settings"]["api_key"] = "tvly-test"
    r = c.put("/v1/config.yaml", json={"yaml": yaml.safe_dump(config, sort_keys=False)})
    assert r.status_code == 200
    doc = r.json()
    search = next(cap for cap in doc["capabilities"] if cap["id"] == "search")
    assert search["permission"] == "auto"
    assert search["status"] == "ready"


def test_config_yaml_exposes_authored_fields_only(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    config = yaml.safe_load(c.get("/v1/config.yaml").text)

    assert "status" not in config["knowledgebase"]["vectordb"]
    assert "secret_configured" not in config["knowledgebase"]["vectordb"]
    assert "error" not in config["knowledgebase"]["vectordb"]
    assert all("status" not in provider for provider in config["providers"])
    assert all("secret_configured" not in provider for provider in config["providers"])
    assert all("error" not in provider for provider in config["providers"])
    assert all("status" not in cap for cap in config["capabilities"])
    assert all("error" not in cap for cap in config["capabilities"])

    # Legacy KB provider shells are merged away before YAML is shown or saved.
    config["providers"].extend([
        {"id": "embedding.default", "type": "embedding", "name": "Old embedding"},
        {"id": "rerank.default", "type": "rerank", "name": "Old rerank"},
        {"id": "vectordb.default", "type": "vectordb", "name": "Old vector"},
    ])
    knowledge = next(cap for cap in config["capabilities"] if cap["id"] == "knowledge")
    knowledge["provider_refs"] = ["embedding.default", "rerank.default", "vectordb.default"]

    assert c.put("/v1/config.yaml", json={"yaml": yaml.safe_dump(config, sort_keys=False)}).status_code == 200
    saved = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert {p["id"] for p in saved["providers"]}.isdisjoint({
        "embedding.default", "rerank.default", "vectordb.default",
    })
    saved_knowledge = next(cap for cap in saved["capabilities"] if cap["id"] == "knowledge")
    assert saved_knowledge["provider_refs"] == []
    assert all("status" not in provider for provider in saved["providers"])
    assert all("status" not in cap for cap in saved["capabilities"])


def test_config_routes_persist_to_sql_config_store(tmp_path, monkeypatch):
    client, engine, config_path = _client_with_sql_config(tmp_path, monkeypatch)
    doc = client.get("/v1/config").json()
    assert doc["default_agent"] == "main"

    doc["agents"][0]["name"] = "SQL Agent"
    response = client.put("/v1/config", json=doc)
    assert response.status_code == 200
    assert response.json()["agents"][0]["name"] == "SQL Agent"
    assert not os.path.exists(config_path)

    async def check():
        store = SqlAgentConfigStore(engine)
        stored = await store.load()
        revisions = await store.list_revisions()
        await engine.dispose()
        return stored, revisions

    stored, revisions = asyncio.run(check())
    assert stored.doc.agents[0].name == "SQL Agent"
    assert stored.revision == 2
    assert [item.revision for item in revisions] == [1, 2]


def test_agent_code_manifest_survives_save_load(tmp_path):
    from app.agent_config import (
        load_agent_config,
        save_agent_config,
    )

    path = str(tmp_path / "config.yaml")
    doc = load_agent_config(path)  # missing file -> default doc
    doc.agents[0].code_manifest = "- repo-a — the API server"
    save_agent_config(path, doc)

    reloaded = load_agent_config(path)
    assert reloaded.agents[0].code_manifest == "- repo-a — the API server"


def test_agent_instructions_survive_save_load(tmp_path):
    from app.agent_config import load_agent_config, save_agent_config

    path = str(tmp_path / "config.yaml")
    doc = load_agent_config(path)
    doc.agents[0].instructions = "# Ada\nYou are a code archaeologist. Terse and precise."
    save_agent_config(path, doc)

    reloaded = load_agent_config(path)
    assert reloaded.agents[0].instructions == "# Ada\nYou are a code archaeologist. Terse and precise."


def test_vectordb_secret_roundtrip_masked_and_preserved(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    doc = c.get("/v1/config").json()
    # default section is local
    assert doc["knowledgebase"]["vectordb"]["engine"] == "local"
    doc["knowledgebase"]["vectordb"].update({
        "engine": "elasticsearch", "url": "http://es:9200", "api_key": "real-key",
    })
    saved = c.put("/v1/config", json=doc).json()
    vdb = saved["knowledgebase"]["vectordb"]
    assert vdb["engine"] == "elasticsearch" and vdb["url"] == "http://es:9200"
    assert vdb["api_key"] == "********"      # masked on the way out
    assert vdb["status"] == "healthy"        # real key was graded healthy
    # Re-save the masked placeholder — the real key must be preserved, not blanked.
    vdb2 = c.put("/v1/config", json=saved).json()["knowledgebase"]["vectordb"]
    assert vdb2["status"] == "healthy"
    assert vdb2["secret_configured"] is True


def test_default_instructions_survives_put_config(tmp_path, monkeypatch):
    # The "Default Persona" template (doc.default_instructions) is edited in Control
    # Room and saved via the whole-doc PUT; a regression where update_agent_config
    # forgot to copy it would silently drop the template.
    c = _client(tmp_path, monkeypatch)
    doc = c.get("/v1/config").json()
    assert doc["default_instructions"]   # ships non-blank (the built-in persona)
    doc["default_instructions"] = "# House voice\nYou are a research copilot."
    saved = c.put("/v1/config", json=doc).json()
    assert saved["default_instructions"] == "# House voice\nYou are a research copilot."
    # persists across a fresh read
    assert c.get("/v1/config").json()["default_instructions"] == "# House voice\nYou are a research copilot."


def test_runtime_status_discovers_local_skill_packages(tmp_path):
    skill_dir = tmp_path / "skills" / "review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.yaml").write_text(
        "id: review\n"
        "name: Review Skill\n"
        "description: Review documents.\n"
        "permissions:\n"
        "  tools: [knowledge_search]\n",
        encoding="utf-8",
    )
    (skill_dir / "SKILL.md").write_text("Review carefully.", encoding="utf-8")
    doc = AgentConfigDocument(**{
        "skills": {"root": str(tmp_path / "skills")}
    })
    out = apply_runtime_status(doc, type("Settings", (), {"openai_api_key": "", "default_model": "m", "search_provider": "none", "search_api_key": "", "search_endpoint": ""})(), None)
    skill = next(cap for cap in out.capabilities if cap.id == "skill.review")
    assert skill.name == "Review Skill"
    assert skill.settings["source"] == "local"
    assert skill.dependencies == ["knowledge"]


def test_runtime_status_discovers_skill_md_only_package(tmp_path):
    """A community SKILL.md-only skill is discovered as a capability, with
    dependencies derived from its frontmatter allowed-tools."""
    skill_dir = tmp_path / "skills" / "pdf-form"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: pdf-form\n"
        "description: Fill and extract PDF form fields.\n"
        "allowed-tools:\n"
        "  - knowledge_search\n"
        "  - code_interpreter\n"
        "---\n\n# PDF Form\n\nExtract fields first.\n",
        encoding="utf-8",
    )
    doc = AgentConfigDocument(**{
        "skills": {"root": str(tmp_path / "skills")}
    })
    out = apply_runtime_status(doc, type("Settings", (), {"openai_api_key": "", "default_model": "m", "search_provider": "none", "search_api_key": "", "search_endpoint": ""})(), None)
    skill = next(cap for cap in out.capabilities if cap.id == "skill.pdf-form")
    assert skill.name == "pdf-form"
    assert skill.settings["source"] == "local"
    # allowed-tools [knowledge_search, code_interpreter] -> capability deps [knowledge, sandbox]
    assert skill.dependencies == ["knowledge", "sandbox"]


def _install_demo_skill(c, tmp_path):
    config = yaml.safe_load(c.get("/v1/config.yaml").text)
    config["skills"]["root"] = str(tmp_path / "skills")
    config["skills"]["install"]["upload_root"] = str(tmp_path / "uploads")
    assert c.put("/v1/config.yaml", json={"yaml": yaml.safe_dump(config, sort_keys=False)}).status_code == 200
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            "demo/skill.yaml",
            "id: demo\n"
            "name: Demo Skill\n"
            "version: 1.0.0\n"
            "description: Demo install.\n",
        )
        zf.writestr("demo/SKILL.md", "Follow demo instructions.")
    archive.seek(0)

    upload = c.post(
        "/v1/skills/uploads",
        files={"file": ("demo.zip", archive.getvalue(), "application/zip")},
    )
    assert upload.status_code == 200
    upload_id = upload.json()["upload_id"]

    install = c.post(
        "/v1/skills/install",
        json={"source": {"type": "zip_upload", "upload_id": upload_id}},
    )
    assert install.status_code == 200
    body = install.json()
    assert body["result"]["id"] == "skill.demo"
    skill = next(cap for cap in body["config"]["capabilities"] if cap["id"] == "skill.demo")
    assert skill["name"] == "Demo Skill"
    assert skill["status"] == "ready"


def test_upload_and_install_skill_zip(tmp_path, monkeypatch):
    _install_demo_skill(_client(tmp_path, monkeypatch), tmp_path)


def test_enable_skill_for_agent_endpoint(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    _install_demo_skill(c, tmp_path)

    # Unknown skill is rejected.
    bad = c.post(
        "/v1/skills/enable",
        json={"skill_id": "skill.nope"},
    )
    assert bad.status_code == 400

    # Enable the ready skill for the default agent.
    ok = c.post(
        "/v1/skills/enable",
        json={"skill_id": "demo"},
    )
    assert ok.status_code == 200
    payload = ok.json()
    assert payload["result"]["enabled"] is True
    assert payload["result"]["changed"] is True
    agent = next(a for a in payload["config"]["agents"] if a["id"] == "main")
    assert "skill.demo" in agent["skills"]["enabled"]

    # Idempotent re-enable + disable round-trip.
    again = c.post(
        "/v1/skills/enable",
        json={"skill_id": "skill.demo"},
    )
    assert again.json()["result"]["changed"] is False

    off = c.post(
        "/v1/skills/enable",
        json={"skill_id": "skill.demo", "enabled": False},
    )
    agent = next(a for a in off.json()["config"]["agents"] if a["id"] == "main")
    assert "skill.demo" not in agent["skills"]["enabled"]


def test_config_save_preserves_knowledge_tools(tmp_path, monkeypatch):
    # Regression: _save_and_reload rebuilt the registry WITHOUT knowledge_service, so
    # every PUT /v1/config silently dropped knowledge_search / view_file / grep_file /
    # list_knowledge_bases until a process restart (the reason a running agent lost its
    # KB tools right after a sandbox config change). It must funnel through
    # reload_app_state, which rebinds the already-live KnowledgeService.
    import asyncio
    from app.routes.config import _save_and_reload
    from app.agent_config import load_agent_config
    from app.knowledge import KnowledgeService
    from app.db import make_engine, create_all

    config_path = str(tmp_path / "config.yaml")
    monkeypatch.setenv("CONFIG_PATH", config_path)
    monkeypatch.setenv("MODELS_PATH", str(tmp_path / "models.yaml"))

    async def _knowledge():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        return KnowledgeService(engine, router=None)

    knowledge = asyncio.run(_knowledge())
    cat = ModelCatalog(
        default_model="local/fast",
        providers=[ProviderConfig(name="local", base_url="http://x/v1",
                                  models=[ModelSpec(id="fast")])],
    )
    state = AppState(store=InMemoryStore(), llm=None, default_model="local/fast",
                     router=ProviderRouter(cat), knowledge=knowledge)

    doc = load_agent_config(config_path)  # file missing -> default doc (knowledge enabled)
    asyncio.run(_save_and_reload(doc, state))

    names = set(state.registry.names())
    assert "knowledge_search" in names
    assert {"view_file", "grep_file", "list_knowledge_bases"} <= names
