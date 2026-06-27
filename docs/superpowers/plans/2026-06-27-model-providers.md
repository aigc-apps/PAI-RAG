# Model Providers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure multiple model providers via a reloadable local `models.yaml`, route each `/v1/responses` request to the right provider/client with per-model capabilities, and expose `GET /v1/models` + `POST /v1/models/reload` (+ a frontend selector that reads the catalog).

**Architecture:** A `ProviderRouter` (new `app/providers.py`) holds a `ModelConfig` catalog parsed from `models.yaml` (or synthesized from legacy `Settings` for back-compat) and lazily builds/caches a `LeanLLM` per model. The route resolves `request.model` → config → client + per-model budget, gating tools by `supports_tools` and reasoning by `supports_reasoning`. `AppState` keeps `llm`/`default_model` as a fallback so existing tests (which build `AppState` without a router) are unchanged.

**Tech Stack:** Python 3.11, pydantic v2, PyYAML (already a dep), FastAPI + TestClient, pytest; plus Vite/React/TS for the frontend selector. Backend tests run from `backend/`: `cd backend && python -m pytest ../tests/app -q`. Frontend from `newfrontend/`.

**Reference spec:** `docs/superpowers/specs/2026-06-27-model-providers-design.md`. Branch: `personal/yfei/agent-core`.

## Global Constraints

- **Import-lean:** `app/providers.py` imports only stdlib (`os`), `yaml`, `pydantic`, `loguru`, and `app.llm.LeanLLM`. Boot/import-isolation gates stay green.
- **Backward compatible:** with no `models.yaml`, the catalog is synthesized from `Settings`; `AppState` without a `router` behaves exactly as today. All existing backend tests (115) and frontend tests (50) stay green.
- **Keys via env, resolved at build time:** the catalog references `api_key_env` (name); the only place a literal key lives is the synthesized fallback (`api_key` from `settings.openai_api_key`) and tests. Never persist a key to the YAML.
- **Capabilities are authoritative:** budget = the model's `context_window`/`max_output_tokens`; tools advertised only if `supports_tools`; `enable_thinking` = `supports_reasoning`.
- Run the full app suite at the end of each backend task: `cd backend && python -m pytest ../tests/app -q`.

---

## Key existing contracts (verified, do not re-derive)

- **`app/llm.py`** — `LeanLLM(base_url: str, api_key: str, model: str, temperature: float = 0.7, max_tokens: int = 4096, timeout: int = 120, max_retries: int = 2, enable_thinking: bool = False)`. `.model` is the attribute used by `astream`. `.client = AsyncOpenAI(base_url=, api_key=, ...)` — `.client.base_url` is readable.
- **`app/deps.py`** — `@dataclass AppState(store, llm, default_model, context_window=110000, max_output_tokens=8000, soul=<factory>, registry=<factory>)`; `make_agent()` → `Agent(llm=self.llm, budget=AgentMessageManager(self.context_window, self.max_output_tokens))`; `get_state(request)`.
- **`app/routes/responses.py`** — `create_response`: `if not request.model: request.model = state.default_model`; `ctx, conversation_id = await build_context(request, state.store, soul=state.soul, registry=state.registry)`; `agent = state.make_agent()`; `events = await agent.run(ctx)`; then stream/sync. `build_context(..., registry=None)` → empty ToolBox (no tools advertised).
- **`app/config.py`** — `Settings(BaseSettings, env_prefix="")` with `openai_base_url, openai_api_key, default_model, db_url, store_backend, agent_name, agent_role, search_provider, search_api_key, search_endpoint, skills_dir`; `get_settings()`.
- **`app/lean_main.py`** — `_build_llm(settings)`; lifespan builds `AppState(store=, llm=_build_llm(settings), default_model=settings.default_model, soul=soul, registry=registry)`; mounts `responses_router`, `chat_router`, `conversations_router`.
- **`AgentMessageManager(context_window, max_output_tokens)`** (`agent/budgeting.py`).
- **Test scaffolding** — `import sys, os[, asyncio]; sys.path.insert(0, ".../backend")`. Route tests: `FastAPI()`, `app.state.app_state = AppState(...)`, `app.include_router(...)`, `TestClient`. The shared `_EchoLLM` echoes the last user message (`startswith("echo:")` because the agent prepends `[System Time]`).
- **`newfrontend/src/components/ModelSelector.tsx`** — currently `const MODELS = ["gpt-4o-mini", "gpt-4o"]` rendered as `<option>`s; props `{ model, onChange }`. `src/api/conversations.ts` shows the `fetch`+`jsonOrThrow` pattern. `src/store/chat.ts` holds `model`/`setModel`.

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/app/providers.py` (new) | `ModelConfig`, `ModelCatalog`, `load_catalog`, `ProviderRouter` |
| `backend/app/config.py` (modify) | `Settings.models_path` |
| `backend/app/deps.py` (modify) | `AppState.router`; `make_agent` overrides |
| `backend/app/routes/responses.py` (modify) | per-request model routing + capability gating |
| `backend/app/routes/models.py` (new) | `GET /v1/models`, `POST /v1/models/reload` |
| `backend/app/lean_main.py` (modify) | build router from `models.yaml`; mount models router |
| `tests/app/test_providers.py` (new) | router + catalog unit tests |
| `tests/app/test_routes_model_routing.py` (new) | route-level routing + capability gating |
| `tests/app/test_routes_models.py` (new) | `/v1/models` + reload |
| `newfrontend/src/api/models.ts` (new) | `listModels()` |
| `newfrontend/src/components/ModelSelector.tsx` (modify) | populate from `/v1/models` |
| `newfrontend/src/api/__tests__/models.test.ts` (new) | `listModels` |
| `newfrontend/src/components/__tests__/ModelSelector.test.tsx` (new) | renders fetched models |

---

## Task 1: `ProviderRouter` + catalog

The registry: parse a catalog (YAML or synthesized), resolve keys, build/cache `LeanLLM`s, reload with warm-client preservation.

**Files:**
- Create: `backend/app/providers.py`
- Modify: `backend/app/config.py` (`models_path`)
- Test: `tests/app/test_providers.py`

**Interfaces:**
- Produces: `ModelConfig`, `ModelCatalog`, `load_catalog(path, settings) -> ModelCatalog`, `ProviderRouter` with `get_config(id)`, `get_llm(id) -> LeanLLM`, `register_llm(id, llm)`, `default_model_id`, `list_models() -> List[ModelConfig]`, `reload(catalog)`, `reload_from_disk()`.

- [ ] **Step 1: Write the failing test** — `tests/app/test_providers.py`

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
import pytest
from app.providers import ModelConfig, ModelCatalog, load_catalog, ProviderRouter


class _Settings:
    openai_base_url = "https://api.openai.com/v1"
    openai_api_key = "sk-legacy"
    default_model = "gpt-4o-mini"


def _catalog():
    return ModelCatalog(
        default_model="fast",
        models=[
            ModelConfig(id="fast", provider="openai", base_url="https://fast/v1", api_key="k1",
                        context_window=128000, max_output_tokens=8192, supports_tools=True),
            ModelConfig(id="smart", provider="anthropic", base_url="https://smart/v1", api_key="k2",
                        context_window=200000, max_output_tokens=16384, supports_reasoning=True),
        ],
    )


def test_get_config_and_unknown_raises():
    r = ProviderRouter(_catalog())
    assert r.get_config("smart").base_url == "https://smart/v1"
    assert r.default_model_id == "fast"
    with pytest.raises(KeyError):
        r.get_config("nope")


def test_get_llm_builds_client_with_model_and_reasoning():
    r = ProviderRouter(_catalog())
    fast = r.get_llm("fast")
    smart = r.get_llm("smart")
    assert fast.model == "fast" and str(fast.client.base_url).startswith("https://fast")
    assert smart.enable_thinking is True and fast.enable_thinking is False
    # cached
    assert r.get_llm("fast") is fast


def test_register_llm_overrides_client():
    r = ProviderRouter(_catalog())
    sentinel = object()
    r.register_llm("fast", sentinel)
    assert r.get_llm("fast") is sentinel


def test_list_models():
    r = ProviderRouter(_catalog())
    assert {m.id for m in r.list_models()} == {"fast", "smart"}


def test_key_resolution_direct_then_env(monkeypatch):
    monkeypatch.setenv("MYKEY", "from-env")
    cat = ModelCatalog(default_model="a", models=[
        ModelConfig(id="a", provider="x", base_url="u", api_key_env="MYKEY"),
    ])
    r = ProviderRouter(cat)
    assert r.get_llm("a").client.api_key == "from-env"


def test_model_with_missing_required_key_is_omitted(monkeypatch):
    monkeypatch.delenv("ABSENT_KEY", raising=False)
    cat = ModelCatalog(default_model="present", models=[
        ModelConfig(id="present", provider="x", base_url="u", api_key="k"),
        ModelConfig(id="absent", provider="x", base_url="u", api_key_env="ABSENT_KEY"),
    ])
    r = ProviderRouter(cat)
    assert "absent" not in {m.id for m in r.list_models()}
    assert "present" in {m.id for m in r.list_models()}


def test_reload_swaps_configs_and_preserves_unchanged_warm_client():
    r = ProviderRouter(_catalog())
    fast_before = r.get_llm("fast")   # warm
    r.get_llm("smart")
    # reload: 'fast' unchanged, 'smart' changed (new base_url), 'extra' added
    r.reload(ModelCatalog(default_model="fast", models=[
        ModelConfig(id="fast", provider="openai", base_url="https://fast/v1", api_key="k1",
                    context_window=128000, max_output_tokens=8192, supports_tools=True),
        ModelConfig(id="smart", provider="anthropic", base_url="https://smart2/v1", api_key="k2"),
        ModelConfig(id="extra", provider="x", base_url="https://extra/v1", api_key="k3"),
    ]))
    assert r.get_llm("fast") is fast_before          # unchanged -> warm client preserved
    assert str(r.get_llm("smart").client.base_url).startswith("https://smart2")  # changed -> rebuilt
    assert "extra" in {m.id for m in r.list_models()}


def test_load_catalog_from_yaml(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(
        "default_model: m1\n"
        "models:\n"
        "  - id: m1\n"
        "    provider: openai\n"
        "    base_url: https://x/v1\n"
        "    api_key_env: OPENAI_API_KEY\n"
    )
    cat = load_catalog(str(p), _Settings())
    assert cat.default_model == "m1" and cat.models[0].base_url == "https://x/v1"


def test_load_catalog_fallback_synthesizes_from_settings():
    cat = load_catalog("/no/such/models.yaml", _Settings())
    assert cat.default_model == "gpt-4o-mini"
    m = cat.models[0]
    assert m.id == "gpt-4o-mini" and m.base_url == "https://api.openai.com/v1" and m.api_key == "sk-legacy"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest ../tests/app/test_providers.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.providers'`.

- [ ] **Step 3: Implement `backend/app/providers.py`**

```python
from __future__ import annotations
import os
from typing import List, Optional
import yaml
from pydantic import BaseModel
from loguru import logger
from app.llm import LeanLLM


class ModelConfig(BaseModel):
    id: str
    provider: str = "openai"
    base_url: str
    api_key_env: Optional[str] = None
    api_key: Optional[str] = None  # direct (fallback/tests); precedence over env
    context_window: int = 128000
    max_output_tokens: int = 8000
    supports_tools: bool = True
    supports_reasoning: bool = False
    temperature: Optional[float] = None

    def resolve_key(self) -> str:
        if self.api_key is not None:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env, "")
        return ""

    def has_usable_key(self) -> bool:
        # A model is usable if it has a direct key (even ""), no key requirement
        # (api_key_env unset -> keyless/local), or its env var resolves non-empty.
        if self.api_key is not None or not self.api_key_env:
            return True
        return bool(os.environ.get(self.api_key_env, ""))


class ModelCatalog(BaseModel):
    default_model: str
    models: List[ModelConfig]


def load_catalog(path: Optional[str], settings) -> ModelCatalog:
    """Parse `models.yaml` if present; otherwise synthesize a one-model catalog from
    the legacy Settings (openai_* + default_model) so env-only deployments still work."""
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return ModelCatalog(**data)
    return ModelCatalog(
        default_model=settings.default_model,
        models=[
            ModelConfig(
                id=settings.default_model,
                provider="openai",
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key,
            )
        ],
    )


class ProviderRouter:
    """Routes a requested model id to its provider config + a cached LeanLLM client.
    Catalog is reloadable; warm clients are preserved across a reload for unchanged
    models. Single-process: one router per AppState (config is GitOps/eventually
    consistent across instances; runtime-state sync is a separate concern)."""

    def __init__(self, catalog: ModelCatalog, path: Optional[str] = None):
        self._path = path
        self._configs: dict = {}
        self._clients: dict = {}
        self._default = ""
        self._apply(catalog)

    def _apply(self, catalog: ModelCatalog) -> None:
        old_configs = self._configs
        old_clients = self._clients
        new_configs: dict = {}
        for m in catalog.models:
            if not m.has_usable_key():
                logger.warning(f"model '{m.id}': required key env '{m.api_key_env}' unset; omitting")
                continue
            new_configs[m.id] = m
        self._configs = new_configs
        self._default = (
            catalog.default_model
            if catalog.default_model in new_configs
            else (next(iter(new_configs), ""))
        )
        # Preserve warm clients only for configs that are byte-for-byte unchanged.
        self._clients = {
            mid: client
            for mid, client in old_clients.items()
            if mid in new_configs and new_configs[mid] == old_configs.get(mid)
        }

    def get_config(self, model_id: str) -> ModelConfig:
        cfg = self._configs.get(model_id)
        if cfg is None:
            raise KeyError(model_id)
        return cfg

    def get_llm(self, model_id: str):
        if model_id in self._clients:
            return self._clients[model_id]
        cfg = self.get_config(model_id)
        key = cfg.resolve_key() or "EMPTY"  # AsyncOpenAI needs a non-empty string
        llm = LeanLLM(
            base_url=cfg.base_url,
            api_key=key,
            model=cfg.id,
            max_tokens=cfg.max_output_tokens,
            enable_thinking=cfg.supports_reasoning,
            temperature=cfg.temperature if cfg.temperature is not None else 0.7,
        )
        self._clients[model_id] = llm
        return llm

    def register_llm(self, model_id: str, llm) -> None:
        """Inject a client (test seam; also a warm-override)."""
        self._clients[model_id] = llm

    @property
    def default_model_id(self) -> str:
        return self._default

    def list_models(self) -> List[ModelConfig]:
        return list(self._configs.values())

    def reload(self, catalog: ModelCatalog) -> None:
        self._apply(catalog)

    def reload_from_disk(self, settings=None) -> None:
        from app.config import get_settings
        self._apply(load_catalog(self._path, settings or get_settings()))
```

- [ ] **Step 4: Add `models_path`** in `backend/app/config.py`

Add to `Settings` (after `skills_dir`):

```python
    models_path: str = "models.yaml"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python -m pytest ../tests/app/test_providers.py -q`
Expected: PASS (9 tests).

- [ ] **Step 6: Run the full app suite**

Run: `cd backend && python -m pytest ../tests/app -q`
Expected: PASS (no regressions; `providers.py` is new and unused so far).

- [ ] **Step 7: Commit**

```bash
git add backend/app/providers.py backend/app/config.py tests/app/test_providers.py
git commit -m "feat(app): ProviderRouter + model catalog (yaml + legacy-settings fallback)"
```

---

## Task 2: Route per-request model selection

Wire the router into `AppState` and `create_response`: resolve `request.model` → config → client + per-model budget; gate tools/reasoning; 404 on unknown model. No-router path unchanged.

**Files:**
- Modify: `backend/app/deps.py`, `backend/app/routes/responses.py`
- Test: `tests/app/test_routes_model_routing.py`

**Interfaces:**
- Consumes: `ProviderRouter` (Task 1).
- Produces: `AppState.router: Optional[ProviderRouter] = None`; `make_agent(self, llm=None, context_window=None, max_output_tokens=None)`; `create_response` routes by model.

- [ ] **Step 1: Write the failing test** — `tests/app/test_routes_model_routing.py`

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.store.memory import InMemoryStore
from app.deps import AppState
from app.providers import ModelConfig, ModelCatalog, ProviderRouter
from app.routes.responses import router as responses_router
from common.llm.models import TextChunk
from openai.types.chat.chat_completion_chunk import CompletionUsage


def _echo(tag):
    class _LLM:
        async def astream(self, messages, tools=None, **kwargs):
            last = ""
            for m in messages:
                if m.get("role") == "user":
                    last = m.get("content") or ""
            usage = CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)

            async def gen():
                yield TextChunk(delta=f"{tag}:{last}", usage=None)
                yield TextChunk(delta="", usage=usage)
            return gen()
    return _LLM()


def _router():
    cat = ModelCatalog(default_model="fast", models=[
        ModelConfig(id="fast", provider="x", base_url="u", api_key="k", supports_tools=True),
        ModelConfig(id="smart", provider="y", base_url="u", api_key="k", supports_tools=False),
    ])
    r = ProviderRouter(cat)
    r.register_llm("fast", _echo("FAST"))
    r.register_llm("smart", _echo("SMART"))
    return r


def _client(router):
    app = FastAPI()
    app.state.app_state = AppState(store=InMemoryStore(), llm=None, default_model="fast", router=router)
    app.include_router(responses_router)
    return TestClient(app)


def test_request_model_routes_to_the_right_client():
    c = _client(_router())
    fast = c.post("/v1/responses", json={"input": "hi", "stream": False, "model": "fast"}).json()
    smart = c.post("/v1/responses", json={"input": "hi", "stream": False, "model": "smart"}).json()
    assert fast["output"][0]["content"][0]["text"].startswith("FAST:")
    assert smart["output"][0]["content"][0]["text"].startswith("SMART:")


def test_default_model_used_when_omitted():
    c = _client(_router())
    body = c.post("/v1/responses", json={"input": "hi", "stream": False}).json()
    assert body["output"][0]["content"][0]["text"].startswith("FAST:")


def test_unknown_model_returns_404():
    c = _client(_router())
    r = c.post("/v1/responses", json={"input": "hi", "stream": False, "model": "ghost"})
    assert r.status_code == 404


def test_supports_tools_false_advertises_no_tools():
    # The agent receives no tools when the model can't use them. We assert via the
    # build_context path: a model with supports_tools=False yields an empty ToolBox.
    import asyncio
    from app.builder import build_context
    from app.schemas import ResponsesRequest
    from agent.tools.defaults import build_default_registry

    class _S:
        search_provider = "none"

    router = _router()

    async def run():
        # mimic the route's gating decision
        cfg = router.get_config("smart")
        reg = build_default_registry(_S())
        ctx, _ = await build_context(
            ResponsesRequest(model="smart", input="hi"), InMemoryStore(),
            registry=(reg if cfg.supports_tools else None),
        )
        assert ctx.tools.tools == []
    asyncio.run(run())


def test_no_router_path_unchanged():
    app = FastAPI()
    app.state.app_state = AppState(store=InMemoryStore(), llm=_echo("ECHO"), default_model="m")
    app.include_router(responses_router)
    c = TestClient(app)
    body = c.post("/v1/responses", json={"input": "hi", "stream": False}).json()
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["text"].startswith("ECHO:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest ../tests/app/test_routes_model_routing.py -q`
Expected: FAIL — `AppState` has no `router` / model isn't routed.

- [ ] **Step 3: Add `router` + flexible `make_agent`** in `backend/app/deps.py`

Add the import:

```python
from typing import Optional
from app.providers import ProviderRouter
```

Add the field (after `registry`):

```python
    router: Optional[ProviderRouter] = None
```

Replace `make_agent` to accept overrides:

```python
    def make_agent(self, llm=None, context_window: Optional[int] = None,
                   max_output_tokens: Optional[int] = None) -> Agent:
        return Agent(
            llm=llm if llm is not None else self.llm,
            budget=AgentMessageManager(
                context_window=context_window if context_window is not None else self.context_window,
                max_output_tokens=max_output_tokens if max_output_tokens is not None else self.max_output_tokens,
            ),
        )
```

- [ ] **Step 4: Route by model** in `backend/app/routes/responses.py`

Replace the head of `create_response` (the model-default + `build_context` + `make_agent` lines) with:

```python
@router.post("/v1/responses")
async def create_response(
    request: ResponsesRequest,
    req: Request,
    state: AppState = Depends(get_state),
):
    if not request.model:
        request.model = (
            state.router.default_model_id if state.router is not None else state.default_model
        )

    cfg = None
    if state.router is not None:
        try:
            cfg = state.router.get_config(request.model)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown model: {request.model}")

    tools_ok = cfg.supports_tools if cfg is not None else True
    try:
        ctx, conversation_id = await build_context(
            request, state.store, soul=state.soul,
            registry=(state.registry if tools_ok else None),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    response_id = _rid()
    if state.router is not None and cfg is not None:
        agent = state.make_agent(
            llm=state.router.get_llm(request.model),
            context_window=cfg.context_window,
            max_output_tokens=cfg.max_output_tokens,
        )
    else:
        agent = state.make_agent()
    events = await agent.run(ctx)
```

Leave everything below `events = await agent.run(ctx)` (the background branch, the stream/sync blocks) unchanged.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python -m pytest ../tests/app/test_routes_model_routing.py -q`
Expected: PASS (5 tests).

- [ ] **Step 6: Run the full app suite**

Run: `cd backend && python -m pytest ../tests/app -q`
Expected: PASS (existing route tests use the no-router `AppState` path, unchanged).

- [ ] **Step 7: Commit**

```bash
git add backend/app/deps.py backend/app/routes/responses.py tests/app/test_routes_model_routing.py
git commit -m "feat(app): route /v1/responses by model (provider client + per-model budget + tool gating)"
```

---

## Task 3: `/v1/models` + reload, wired at boot

Expose the catalog (OpenAI-compatible) and a reload endpoint; build the router in `lean_main`.

**Files:**
- Create: `backend/app/routes/models.py`
- Modify: `backend/app/lean_main.py`
- Test: `tests/app/test_routes_models.py`

**Interfaces:**
- Consumes: `ProviderRouter`/`load_catalog` (Task 1); `AppState.router` (Task 2).
- Produces: `GET /v1/models` → `{"object":"list","data":[{id,object,created,owned_by}]}`; `POST /v1/models/reload` → re-reads disk, returns the new list (400 if no router).

- [ ] **Step 1: Write the failing test** — `tests/app/test_routes_models.py`

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python -m pytest ../tests/app/test_routes_models.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.routes.models'`.

- [ ] **Step 3: Implement `backend/app/routes/models.py`**

```python
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from app.deps import AppState, get_state

router = APIRouter()


def _to_data(state: AppState):
    if state.router is not None:
        return [
            {"id": m.id, "object": "model", "created": 0, "owned_by": m.provider}
            for m in state.router.list_models()
        ]
    return [{"id": state.default_model, "object": "model", "created": 0, "owned_by": "openai"}]


@router.get("/v1/models")
async def list_models(state: AppState = Depends(get_state)):
    return JSONResponse({"object": "list", "data": _to_data(state)})


@router.post("/v1/models/reload")
async def reload_models(state: AppState = Depends(get_state)):
    if state.router is None:
        raise HTTPException(status_code=400, detail="no model router configured")
    state.router.reload_from_disk()
    return JSONResponse({"object": "list", "reloaded": True, "data": _to_data(state)})
```

- [ ] **Step 4: Build the router at boot** in `backend/app/lean_main.py`

Add imports:

```python
from app.providers import ProviderRouter, load_catalog
from app.routes.models import router as models_router
```

In `lifespan`, after building `registry`, build the provider router and pass it to `AppState`:

```python
    catalog = load_catalog(settings.models_path, settings)
    provider_router = ProviderRouter(catalog, path=settings.models_path)
    app.state.app_state = AppState(
        store=store, llm=_build_llm(settings), default_model=settings.default_model,
        soul=soul, registry=registry, router=provider_router,
    )
```

Mount the models router after the others:

```python
app.include_router(models_router)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && python -m pytest ../tests/app/test_routes_models.py -q`
Expected: PASS (3 tests).

- [ ] **Step 6: Run the full app suite + boot/isolation gates**

Run: `cd backend && python -m pytest ../tests/app -q`
Expected: PASS (incl. `test_lean_main_boot.py` — boot now builds a router from the synthesized fallback catalog since no `models.yaml` exists in the test cwd).

- [ ] **Step 7: Commit**

```bash
git add backend/app/routes/models.py backend/app/lean_main.py tests/app/test_routes_models.py
git commit -m "feat(app): GET /v1/models + POST /v1/models/reload; build ProviderRouter at boot"
```

---

## Task 4: Frontend — model selector from `/v1/models`

Populate `ModelSelector` from the live catalog.

**Files:**
- Create: `newfrontend/src/api/models.ts`
- Modify: `newfrontend/src/components/ModelSelector.tsx`
- Test: `newfrontend/src/api/__tests__/models.test.ts`, `newfrontend/src/components/__tests__/ModelSelector.test.tsx`

**Interfaces:**
- Produces: `listModels(): Promise<string[]>` (GET `/v1/models` → `data[].id`); `ModelSelector` fetches on mount and renders the ids, always including the current `model` (and a sensible fallback on error).

- [ ] **Step 1: Write the failing tests**

`newfrontend/src/api/__tests__/models.test.ts`:

```ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import { listModels } from "../models";

describe("listModels", () => {
  beforeEach(() => vi.restoreAllMocks());
  it("returns the model ids from /v1/models", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true, status: 200,
      json: async () => ({ object: "list", data: [{ id: "fast" }, { id: "smart" }] }),
    } as Response);
    vi.stubGlobal("fetch", fetchMock);
    expect(await listModels()).toEqual(["fast", "smart"]);
    expect(String(fetchMock.mock.calls[0][0])).toContain("/v1/models");
  });
  it("throws on a non-ok response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500 } as Response));
    await expect(listModels()).rejects.toThrow();
  });
});
```

`newfrontend/src/components/__tests__/ModelSelector.test.tsx`:

```tsx
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("../../api/models", () => ({ listModels: vi.fn() }));
import { ModelSelector } from "../ModelSelector";
import * as api from "../../api/models";

beforeEach(() => vi.clearAllMocks());

describe("ModelSelector", () => {
  it("renders fetched models", async () => {
    (api.listModels as any).mockResolvedValue(["fast", "smart"]);
    render(<ModelSelector model="fast" onChange={() => {}} />);
    await waitFor(() => expect(screen.getByRole("option", { name: "smart" })).toBeInTheDocument());
    expect(screen.getByRole("option", { name: "fast" })).toBeInTheDocument();
  });

  it("falls back to the current model when the fetch fails", async () => {
    (api.listModels as any).mockRejectedValue(new Error("boom"));
    render(<ModelSelector model="gpt-4o-mini" onChange={() => {}} />);
    await waitFor(() =>
      expect(screen.getByRole("option", { name: "gpt-4o-mini" })).toBeInTheDocument()
    );
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `cd newfrontend && npm test -- "models|ModelSelector"`
Expected: FAIL — `../models` missing / selector doesn't fetch.

- [ ] **Step 3: Implement `newfrontend/src/api/models.ts`**

```ts
export async function listModels(): Promise<string[]> {
  const res = await fetch("/v1/models");
  if (!res.ok) throw new Error(`models request failed: ${res.status}`);
  const body = (await res.json()) as { data: { id: string }[] };
  return body.data.map((m) => m.id);
}
```

- [ ] **Step 4: Update `newfrontend/src/components/ModelSelector.tsx`**

```tsx
import { useEffect, useState } from "react";
import { listModels } from "../api/models";

export function ModelSelector({
  model,
  onChange,
}: {
  model: string;
  onChange: (m: string) => void;
}) {
  const [models, setModels] = useState<string[]>([model]);

  useEffect(() => {
    let cancelled = false;
    listModels()
      .then((ids) => {
        if (!cancelled && ids.length) setModels(ids);
      })
      .catch(() => {
        /* keep the fallback (current model) on error */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Always include the current model so the controlled <select> has a valid option.
  const options = models.includes(model) ? models : [model, ...models];

  return (
    <select
      aria-label="Model"
      value={model}
      onChange={(e) => onChange(e.target.value)}
      className="rounded-md border border-gray-300 px-2 py-1 text-sm"
    >
      {options.map((m) => (
        <option key={m} value={m}>
          {m}
        </option>
      ))}
    </select>
  );
}
```

- [ ] **Step 5: Run to verify pass**

Run: `cd newfrontend && npm test -- "models|ModelSelector"`
Expected: PASS (4 tests).

- [ ] **Step 6: Full suite + build**

Run: `cd newfrontend && npm test && npm run build`
Expected: PASS (the existing `App.test.tsx` mocks may need `listModels` — if `App.test.tsx` renders `ModelSelector` and now calls `listModels`, ensure it doesn't error: the unmocked `fetch` in that test will reject and the selector falls back silently, so `App.test.tsx` stays green. Confirm; if it logs an unhandled rejection, add `vi.mock("../../api/models", () => ({ listModels: vi.fn().mockResolvedValue([]) }))` to `App.test.tsx`.)

- [ ] **Step 7: Commit**

```bash
git add newfrontend/src/api/models.ts newfrontend/src/components/ModelSelector.tsx newfrontend/src/api/__tests__/models.test.ts newfrontend/src/components/__tests__/ModelSelector.test.tsx
git commit -m "feat(newfrontend): populate ModelSelector from GET /v1/models"
```

---

## Self-Review (completed against the spec)

- **Model catalog over OpenAI-compatible `LeanLLM`; `models.yaml` + legacy fallback; key-by-env-name resolution; omit models with a missing required key** → Task 1.
- **`ProviderRouter` get_config/get_llm(cache)/register_llm/list_models/default/reload (warm-client preservation)** → Task 1.
- **Per-request routing: model→client, per-model budget, `supports_tools` gating, `supports_reasoning`→enable_thinking, unknown→404; no-router path unchanged** → Task 2.
- **`GET /v1/models` (OpenAI shape) + `POST /v1/models/reload`; router built at boot from `Settings.models_path`** → Task 3.
- **Frontend `listModels()` + `ModelSelector` populated from the catalog with a graceful fallback** → Task 4.
- **Backward compatibility** (no `models.yaml` → synthesized catalog; `AppState` without `router` unchanged; existing 115 backend + 50 frontend tests green) → Global Constraints + Tasks 2/4.
- **Reasoning enabled per-model**: `get_llm` sets `enable_thinking=cfg.supports_reasoning`; budget per-model via `make_agent` overrides → Tasks 1/2.

No placeholders; signatures (`ModelConfig`, `load_catalog(path, settings)`, `ProviderRouter.get_config/get_llm/register_llm/list_models/default_model_id/reload/reload_from_disk`, `make_agent(llm=, context_window=, max_output_tokens=)`, `listModels()`) are consistent across tasks. Multi-instance config is GitOps/eventually-consistent (per spec); runtime-state sync (Redis-Streams RunManager) is explicitly out of scope.
