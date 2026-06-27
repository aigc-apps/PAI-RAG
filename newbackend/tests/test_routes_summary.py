import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.store.memory import InMemoryStore
from app.deps import AppState
from app.routes.responses import router as responses_router
from common.llm.models import TextChunk
from openai.types.chat.chat_completion_chunk import CompletionUsage


class _SumLLM:
    async def astream(self, messages, tools=None, **kwargs):
        prompt = ""
        for m in messages:
            if m.get("role") == "user":
                prompt = m.get("content") or ""
        usage = CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        is_summary = "Updated summary:" in prompt

        async def gen():
            yield TextChunk(delta=("ROLLED-UP SUMMARY" if is_summary else "ok"), usage=None)
            yield TextChunk(delta="", usage=usage)
        return gen()


def _client():
    app = FastAPI()
    app.state.app_state = AppState(
        store=InMemoryStore(), llm=_SumLLM(), default_model="m",
        summary_enabled=True, summary_keep_recent=1, summary_batch=1,
    )
    app.include_router(responses_router)
    return TestClient(app)


def test_long_conversation_gets_rolling_summary():
    with TestClient(_client().app) as c:
        conv_id = None
        for i in range(4):
            body = c.post("/v1/responses", json={
                "input": f"message {i}", "stream": False,
                **({"conversation": conv_id, "previous_response_id": rid} if conv_id else {}),
            }).json()
            conv_id = body["conversation"]["id"]
            rid = body["id"]
        # background summarization runs on the app loop; poll the conversation summary
        summary = None
        for _ in range(100):
            detail = c.get(f"/v1/conversations/{conv_id}").json()
            # latest_response_id present means persisted; check the store summary via detail? use store
            conv = None
            import asyncio
            async def _read():
                return await c.app.state.app_state.store.get_conversation(conv_id)
            conv = asyncio.run(_read())
            if conv and conv.summary:
                summary = conv.summary
                break
            time.sleep(0.02)
        assert summary == "ROLLED-UP SUMMARY"


def test_summary_disabled_writes_no_summary():
    app = FastAPI()
    app.state.app_state = AppState(store=InMemoryStore(), llm=_SumLLM(), default_model="m",
                                   summary_enabled=False)
    app.include_router(responses_router)
    import asyncio
    with TestClient(app) as c:
        conv_id = None
        rid = None
        for i in range(4):
            body = c.post("/v1/responses", json={"input": f"m{i}", "stream": False,
                          **({"conversation": conv_id, "previous_response_id": rid} if conv_id else {})}).json()
            conv_id = body["conversation"]["id"]; rid = body["id"]
        time.sleep(0.2)
        conv = asyncio.run(c.app.state.app_state.store.get_conversation(conv_id))
        assert conv.summary is None
