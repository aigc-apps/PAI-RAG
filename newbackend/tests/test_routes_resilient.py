import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
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


def _seq(line_block):
    import json
    seqs = []
    for part in line_block.splitlines():
        if part.startswith("data:"):
            payload = part[len("data:"):].strip()
            if payload:
                seqs.append(json.loads(payload).get("sequence_number"))
    return [s for s in seqs if s is not None]


def test_resume_replays_only_events_after_cursor():
    c = _client()
    # start + fully drain a background stream to populate the run buffer
    with c.stream("POST", "/v1/responses",
                  json={"input": "hi", "stream": True, "background": True, "user_id": "u1"}) as r:
        first_raw = "".join(chunk for chunk in r.iter_text())
    rid = None
    import json
    for part in first_raw.splitlines():
        if part.startswith("data:") and '"response.created"' in part:
            rid = json.loads(part[len("data:"):].strip())["response"]["id"]
            break
    assert rid is not None
    # resume from cursor N=3 -> only events with sequence_number > 3
    with c.stream("GET", f"/v1/responses/{rid}",
                  params={"stream": "true", "starting_after": 3}) as r:
        assert r.status_code == 200
        resume_raw = "".join(chunk for chunk in r.iter_text())
    seqs = _seq(resume_raw)
    assert seqs and min(seqs) > 3
    assert "response.completed" in resume_raw


def test_resume_unknown_run_returns_409():
    c = _client()
    r = c.get("/v1/responses/resp_missing", params={"stream": "true", "starting_after": 0})
    assert r.status_code == 409


def test_cancel_persists_cancelled_partial_and_lists_conversation():
    import time as _t, threading as _threading

    # an LLM that yields one chunk then blocks indefinitely
    class _SlowLLM:
        async def astream(self, messages, tools=None, **kwargs):
            import asyncio
            async def gen():
                from common.llm.models import TextChunk
                yield TextChunk(delta="partial", usage=None)
                await asyncio.sleep(9999)
            return gen()

    _state = AppState(store=InMemoryStore(), llm=_SlowLLM(), default_model="m")
    _app = FastAPI()
    _app.state.app_state = _state
    _app.include_router(responses_router)
    _app.include_router(conversations_router)

    rid_holder: dict = {}
    cancel_result: dict = {}

    # Use TestClient as a context manager so the anyio portal persists across all
    # requests.  This ensures c.post(cancel) runs in the SAME event loop as _pump,
    # making run.cancel.set() reliably wake _pump without cross-loop issues.
    with TestClient(_app) as c:
        def _cancel_when_ready():
            # Poll RunManager._runs directly (no stream-read needed to find the rid).
            for _ in range(200):
                rids = list(_state.runs._runs.keys())
                if rids:
                    rid_holder["rid"] = rids[0]
                    break
                _t.sleep(0.02)
            if "rid" not in rid_holder:
                return
            cr = c.post(f"/v1/responses/{rid_holder['rid']}/cancel")
            cancel_result["status"] = cr.status_code

        th = _threading.Thread(target=_cancel_when_ready, daemon=True)
        th.start()

        # Main thread streams until the run terminates (cancel ends the stream).
        # Starlette's TestClient buffers the full response; blocking here is fine
        # because the background thread's cancel terminates _pump → stream ends.
        with c.stream("POST", "/v1/responses",
                      json={"input": "go", "stream": True,
                            "background": True, "user_id": "u1"}) as r:
            r.read()  # drain the fully-buffered response

        th.join(timeout=5)
        assert "rid" in rid_holder, "run was never registered in RunManager"
        assert cancel_result.get("status") == 200

        rid = rid_holder["rid"]
        # the cancelled turn was persisted as a real, listable conversation
        data = c.get("/v1/conversations", params={"user_id": "u1"}).json()["data"]
        assert len(data) == 1
        detail = c.get(f"/v1/conversations/{data[0]['id']}").json()
        assert detail["latest_response_id"] == rid
        stored = c.get(f"/v1/responses/{rid}").json()
        assert stored["status"] == "cancelled"


def test_cancel_unknown_returns_404():
    c = _client()
    assert c.post("/v1/responses/resp_missing/cancel").status_code == 404
