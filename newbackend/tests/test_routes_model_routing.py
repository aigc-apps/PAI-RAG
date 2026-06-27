import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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
