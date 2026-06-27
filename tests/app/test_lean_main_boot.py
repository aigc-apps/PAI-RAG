# tests/app/test_lean_main_boot.py
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from fastapi.testclient import TestClient


def test_app_boots_in_memory_and_serves(monkeypatch):
    monkeypatch.setenv("STORE_BACKEND", "memory")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("DEFAULT_MODEL", "m")
    import importlib

    # The legacy heavy app is `app.main`; the lean service is `app.lean_main`.
    import app.lean_main as m

    importlib.reload(m)

    # Inject a fake LLM so we don't hit the network. It must follow the REAL agent
    # contract: `astream` is an async fn RETURNING an async generator (the agent does
    # `await llm.astream(...)`), and usage is a real CompletionUsage.
    from common.llm.models import TextChunk
    from openai.types.chat.chat_completion_chunk import CompletionUsage

    class _EchoLLM:
        async def astream(self, messages, tools=None, **kwargs):
            last = ""
            for msg in messages:
                if msg.get("role") == "user":
                    last = msg.get("content") or ""

            usage = CompletionUsage(
                prompt_tokens=1, completion_tokens=1, total_tokens=2
            )

            async def gen():
                yield TextChunk(delta=f"echo:{last}", usage=None)
                yield TextChunk(delta="", usage=usage)

            return gen()

    with TestClient(m.app) as c:
        echo = _EchoLLM()
        c.app.state.app_state.llm = echo
        # When a ProviderRouter is wired, the responses route uses router.get_llm()
        # rather than state.llm; register the fake LLM so no real API call is made.
        if c.app.state.app_state.router is not None:
            c.app.state.app_state.router.register_llm(
                c.app.state.app_state.default_model, echo
            )
        r = c.post("/v1/responses", json={"input": "hi", "stream": False})
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "response" and body["status"] == "completed"
        # The agent prepends a `[System Time: ...]` prefix to the user turn, so the
        # echoed text contains "hi" rather than equalling it exactly.
        text = body["output"][0]["content"][0]["text"]
        assert text.startswith("echo:")
        assert "hi" in text
