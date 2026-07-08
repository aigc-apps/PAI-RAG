import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.store.memory import InMemoryStore
from app.deps import AppState
from app.routes.responses import router as responses_router
from app.routes.conversations import router as conversations_router
from tests.authutil import apply_auth
from common.llm.models import TextChunk
from openai.types.chat.chat_completion_chunk import CompletionUsage


class _EchoLLM:
    """Yields the user's last message text back as a single assistant chunk + usage.

    The agent core does ``await self.llm.astream(...)`` then ``async for`` over the
    result, so ``astream`` is an async fn RETURNING an async generator, not itself one.
    """

    async def astream(self, messages, tools=None, **kwargs):
        last = ""
        for m in messages:
            if m.get("role") == "user":
                last = m.get("content") or ""

        usage = CompletionUsage(
            prompt_tokens=1, completion_tokens=1, total_tokens=2
        )

        async def gen():
            yield TextChunk(delta=f"echo:{last}", usage=None)
            yield TextChunk(delta="", usage=usage)

        return gen()


def _client():
    app = FastAPI()
    app.state.app_state = AppState(store=InMemoryStore(), llm=_EchoLLM(), default_model="m")
    app.include_router(responses_router)
    app.include_router(conversations_router)
    # Authenticated as u1; identity is server-derived, not from the request body.
    return TestClient(apply_auth(app, user_id="u1", role="user"))


def test_list_conversations_scoped_to_authenticated_user():
    c = _client()
    c.post("/v1/responses", json={"input": "alpha", "stream": False})
    # Another user's conversation, seeded directly, must not appear in u1's list.
    import asyncio
    asyncio.run(c.app.state.app_state.store.ensure_conversation("conv_other", "u2", "beta"))
    data = c.get("/v1/conversations").json()["data"]
    assert len(data) == 1
    assert data[0]["title"] == "alpha"
    assert "last_response_id" in data[0] and "updated_at" in data[0]


def test_get_conversation_reconstructs_messages_and_latest_id():
    c = _client()
    first = c.post("/v1/responses", json={"input": "q1", "stream": False, "user_id": "u1"}).json()
    conv_id = first["conversation"]["id"]
    second = c.post("/v1/responses", json={
        "input": "q2", "stream": False, "user_id": "u1",
        "previous_response_id": first["id"], "conversation": conv_id,
    }).json()
    detail = c.get(f"/v1/conversations/{conv_id}").json()
    assert detail["id"] == conv_id
    assert detail["latest_response_id"] == second["id"]
    roles = [m["role"] for m in detail["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert detail["messages"][0]["text"] == "q1"
    # The agent prepends a "[System Time: ...]\n" header to the rendered user turn,
    # so the echo carries that prefix + the input.
    assert detail["messages"][1]["text"].startswith("echo:") and detail["messages"][1]["text"].endswith("q1")
    assert detail["messages"][3]["previous_response_id"] == first["id"]


def test_get_conversation_404():
    c = _client()
    assert c.get("/v1/conversations/nope").status_code == 404


def test_delete_conversation():
    c = _client()
    body = c.post("/v1/responses", json={"input": "x", "stream": False, "user_id": "u1"}).json()
    conv_id = body["conversation"]["id"]
    assert c.delete(f"/v1/conversations/{conv_id}").status_code == 200
    assert c.get(f"/v1/conversations/{conv_id}").status_code == 404
    assert c.delete(f"/v1/conversations/{conv_id}").status_code == 404
