import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.store.memory import InMemoryStore
from app.deps import AppState
from app.routes.responses import router as responses_router
from app.routes.conversations import router as conversations_router
from common.llm.models import TextChunk
from openai.types.chat.chat_completion_chunk import CompletionUsage


class _EchoLLM:
    async def astream(self, messages, tools=None, **kwargs):
        last = ""
        for m in messages:
            if m.get("role") == "user":
                last = m.get("content") or ""
        usage = CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)

        async def gen():
            yield TextChunk(delta=f"echo:{last}", usage=None)
            yield TextChunk(delta="", usage=usage)
        return gen()


def _client():
    app = FastAPI()
    app.state.app_state = AppState(store=InMemoryStore(), llm=_EchoLLM(), default_model="m")
    app.include_router(responses_router)
    app.include_router(conversations_router)
    return TestClient(app)


def test_background_stream_emits_events_and_persists():
    c = _client()
    with c.stream("POST", "/v1/responses",
                  json={"input": "hi", "stream": True, "background": True, "user_id": "u1"}) as r:
        assert r.status_code == 200
        raw = "".join(chunk for chunk in r.iter_text())
    assert "response.created" in raw and "response.completed" in raw
    assert "response.output_text.delta" in raw
    # the detached run persisted the conversation (listable)
    data = c.get("/v1/conversations", params={"user_id": "u1"}).json()["data"]
    assert len(data) == 1 and data[0]["title"] == "hi"


def test_background_non_stream_returns_in_progress_immediately():
    c = _client()
    body = c.post("/v1/responses",
                  json={"input": "hello", "stream": False, "background": True, "user_id": "u1"}).json()
    assert body["status"] == "in_progress"
    assert body["object"] == "response"
    assert body["id"].startswith("resp_")
    assert body["conversation"]["id"].startswith("conv_")


def test_non_background_path_unchanged():
    c = _client()
    body = c.post("/v1/responses", json={"input": "hi", "stream": False}).json()
    assert body["status"] == "completed"
    assert body["output"][0]["content"][0]["text"].startswith("echo:")
