import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.store.memory import InMemoryStore
from app.deps import AppState
from app.routes.responses import router as responses_router
from app.routes.users import router as users_router
from common.llm.models import TextChunk
from openai.types.chat.chat_completion_chunk import CompletionUsage


class _MemoryLLM:
    """Echoes for the answer turn; returns a fixed ops JSON for the extraction prompt."""
    async def astream(self, messages, tools=None, **kwargs):
        prompt = ""
        for m in messages:
            if m.get("role") == "user":
                prompt = m.get("content") or ""
        usage = CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        is_extract = "JSON operations:" in prompt

        async def gen():
            if is_extract:
                yield TextChunk(delta='[{"op":"ADD","text":"likes hiking"}]', usage=None)
            else:
                yield TextChunk(delta="ok", usage=None)
            yield TextChunk(delta="", usage=usage)
        return gen()


def _app(memory_enabled=True):
    app = FastAPI()
    app.state.app_state = AppState(
        store=InMemoryStore(), llm=_MemoryLLM(), default_model="m",
        memory_enabled=memory_enabled,
    )
    app.include_router(responses_router)
    app.include_router(users_router)
    return app


def test_stored_turn_populates_user_memory_then_injects_it():
    with TestClient(_app()) as c:
        c.post("/v1/responses", json={"input": "I like hiking", "stream": False, "user": "u1"})
        # background extraction runs on the app loop; poll the memory API
        mems = []
        for _ in range(100):
            mems = c.get("/v1/users/u1/memories").json()["data"]
            if mems:
                break
            time.sleep(0.02)
        assert any(m["text"] == "likes hiking" for m in mems)


def test_memory_disabled_writes_nothing():
    with TestClient(_app(memory_enabled=False)) as c:
        c.post("/v1/responses", json={"input": "hi", "stream": False, "user": "u1"})
        time.sleep(0.2)
        assert c.get("/v1/users/u1/memories").json()["data"] == []


def test_delete_user_memories():
    with TestClient(_app()) as c:
        c.post("/v1/responses", json={"input": "I like hiking", "stream": False, "user": "u1"})
        for _ in range(100):
            if c.get("/v1/users/u1/memories").json()["data"]:
                break
            time.sleep(0.02)
        assert c.delete("/v1/users/u1/memories").status_code == 200
        assert c.get("/v1/users/u1/memories").json()["data"] == []
