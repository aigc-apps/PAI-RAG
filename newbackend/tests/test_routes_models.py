import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.store.memory import InMemoryStore
from app.deps import AppState
from app.providers import ModelConfig, ModelCatalog, ProviderRouter
from app.routes.models import router as models_router


def _client(router):
    app = FastAPI()
    app.state.app_state = AppState(store=InMemoryStore(), llm=None, default_model="fast", router=router)
    app.include_router(models_router)
    return TestClient(app)


def test_list_models_returns_catalog():
    cat = ModelCatalog(default_model="fast", models=[
        ModelConfig(id="fast", provider="openai", base_url="u", api_key="k"),
        ModelConfig(id="smart", provider="anthropic", base_url="u", api_key="k"),
    ])
    c = _client(ProviderRouter(cat))
    body = c.get("/v1/models").json()
    assert body["object"] == "list"
    ids = {m["id"] for m in body["data"]}
    assert ids == {"fast", "smart"}
    assert body["data"][0]["object"] == "model"


def test_reload_picks_up_a_rewritten_catalog(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text("default_model: a\nmodels:\n  - id: a\n    provider: x\n    base_url: u\n    api_key: k\n")
    from app.providers import load_catalog

    class _S:
        openai_base_url = "u"; openai_api_key = "k"; default_model = "a"

    router = ProviderRouter(load_catalog(str(p), _S()), path=str(p))
    c = _client(router)
    assert {m["id"] for m in c.get("/v1/models").json()["data"]} == {"a"}
    # rewrite the catalog and reload
    p.write_text("default_model: a\nmodels:\n  - id: a\n    provider: x\n    base_url: u\n    api_key: k\n  - id: b\n    provider: y\n    base_url: u\n    api_key: k\n")
    r = c.post("/v1/models/reload")
    assert r.status_code == 200
    assert {m["id"] for m in c.get("/v1/models").json()["data"]} == {"a", "b"}


def test_reload_without_router_400():
    app = FastAPI()
    app.state.app_state = AppState(store=InMemoryStore(), llm=None, default_model="m")
    app.include_router(models_router)
    c = TestClient(app)
    assert c.post("/v1/models/reload").status_code == 400
