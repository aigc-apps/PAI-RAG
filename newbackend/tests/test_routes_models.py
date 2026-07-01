import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.store.memory import InMemoryStore
from app.deps import AppState
from app.providers import ModelSpec, ProviderConfig, ModelCatalog, ProviderRouter
from app.routes.models import router as models_router


def _client(router):
    app = FastAPI()
    app.state.app_state = AppState(store=InMemoryStore(), llm=None, default_model="x/fast", router=router)
    app.include_router(models_router)
    return TestClient(app)


def test_list_models_returns_catalog():
    cat = ModelCatalog(default_model="x/fast", providers=[
        ProviderConfig(name="x", base_url="u", api_key="k", models=[ModelSpec(id="fast")]),
        ProviderConfig(name="y", base_url="u", api_key="k", models=[ModelSpec(id="smart")]),
    ])
    c = _client(ProviderRouter(cat))
    body = c.get("/v1/models").json()
    assert body["object"] == "list"
    assert body["default"] == "x/fast"
    ids = {m["id"] for m in body["data"]}
    assert ids == {"x/fast", "y/smart"}
    assert body["data"][0]["object"] == "model"


def test_reload_picks_up_a_rewritten_catalog(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "models:\n"
        "  default_model: x/a\n"
        "  providers:\n"
        "    - name: x\n"
        "      base_url: u\n"
        "      api_key: k\n"
        "      models:\n"
        "        - id: a\n"
    )
    from app.providers import load_catalog

    class _S:
        openai_base_url = "u"; openai_api_key = "k"; default_model = "a"

    router = ProviderRouter(load_catalog(str(p), _S()), path=str(p))
    c = _client(router)
    assert {m["id"] for m in c.get("/v1/models").json()["data"]} == {"x/a"}
    # rewrite the catalog: add a second provider and reload
    p.write_text(
        "models:\n"
        "  default_model: x/a\n"
        "  providers:\n"
        "    - name: x\n"
        "      base_url: u\n"
        "      api_key: k\n"
        "      models:\n"
        "        - id: a\n"
        "    - name: y\n"
        "      base_url: u\n"
        "      api_key: k\n"
        "      models:\n"
        "        - id: b\n"
    )
    r = c.post("/v1/models/reload")
    assert r.status_code == 200
    body = c.get("/v1/models").json()
    assert body["default"] == "x/a"
    assert {m["id"] for m in body["data"]} == {"x/a", "y/b"}


def test_reload_rejects_missing_default_model(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "models:\n"
        "  default_model: x/a\n"
        "  providers:\n"
        "    - name: x\n"
        "      base_url: u\n"
        "      api_key: k\n"
        "      models:\n"
        "        - id: a\n"
    )
    from app.providers import load_catalog

    class _S:
        openai_base_url = "u"; openai_api_key = "k"; default_model = "a"

    router = ProviderRouter(load_catalog(str(p), _S()), path=str(p))
    c = _client(router)
    p.write_text(
        "models:\n"
        "  default_model: x/missing\n"
        "  providers:\n"
        "    - name: x\n"
        "      base_url: u\n"
        "      api_key: k\n"
        "      models:\n"
        "        - id: a\n"
    )
    r = c.post("/v1/models/reload")
    assert r.status_code == 400
    assert "default_model 'x/missing' is not defined" in r.json()["detail"]


def test_reload_without_router_400():
    app = FastAPI()
    app.state.app_state = AppState(store=InMemoryStore(), llm=None, default_model="m")
    app.include_router(models_router)
    c = TestClient(app)
    assert c.post("/v1/models/reload").status_code == 400
