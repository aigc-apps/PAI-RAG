import os
import sys
import io
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.testclient import TestClient
import yaml

from app.deps import AppState
from app.agent_config import AgentConfigDocument, apply_runtime_status
from app.providers import ModelCatalog, ModelSpec, ProviderConfig, ProviderRouter
from app.routes.config import router as config_router
from app.store.memory import InMemoryStore
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
    assert body["agents"][0]["model"] == "local/fast"
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


def test_agent_persona_survives_save_load(tmp_path):
    from app.agent_config import (
        AgentPersona,
        load_agent_config,
        save_agent_config,
    )

    path = str(tmp_path / "config.yaml")
    doc = load_agent_config(path)
    doc.agents[0].persona = AgentPersona(
        role="a code archaeologist",
        personality=["terse", "precise"],
    )
    save_agent_config(path, doc)

    reloaded = load_agent_config(path)
    assert reloaded.agents[0].persona.role == "a code archaeologist"
    assert reloaded.agents[0].persona.personality == ["terse", "precise"]


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
    _save_and_reload(config_path, doc, state)

    names = set(state.registry.names())
    assert "knowledge_search" in names
    assert {"view_file", "grep_file", "list_knowledge_bases"} <= names
