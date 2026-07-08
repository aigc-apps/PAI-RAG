import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.deps import AppState
from app.routes.auth import router as auth_router
from app.routes.config import router as config_router
from app.routes.responses import router as responses_router
from app.routes.conversations import router as conversations_router
from app.store.memory import InMemoryStore
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


def _app(monkeypatch, *, with_llm=False):
    monkeypatch.setenv("JWT_SECRET", "rbac-secret")
    store = InMemoryStore()
    app = FastAPI()
    app.state.app_state = AppState(
        store=store, llm=(_EchoLLM() if with_llm else None), default_model="m")
    app.include_router(auth_router)
    app.include_router(config_router)
    app.include_router(responses_router)
    app.include_router(conversations_router)
    return app, store


def _bootstrap(c):
    return c.post("/v1/auth/bootstrap",
                  json={"email": "admin@example.com", "password": "password123"}).json()


def _invite_user(c, admin_tok, email="user@example.com"):
    inv = c.post("/v1/auth/invite", json={"email": email, "role": "user"},
                 headers={"Authorization": f"Bearer {admin_tok}"}).json()
    acc = c.post("/v1/auth/accept-invite",
                 json={"token": inv["invite_token"], "password": "password123"}).json()
    return acc["access_token"], acc["user"]["id"]


# --------------------------------------------------------------------------- #
# admin gate on the control plane
# --------------------------------------------------------------------------- #
def test_config_requires_admin(monkeypatch):
    app, _ = _app(monkeypatch)
    c = TestClient(app)
    admin = _bootstrap(c)
    admin_tok = admin["access_token"]
    user_tok, _ = _invite_user(c, admin_tok)

    # No token → 401
    fresh = TestClient(app)
    assert fresh.get("/v1/setup").status_code == 401
    # Regular user → 403
    assert fresh.get("/v1/setup", headers={"Authorization": f"Bearer {user_tok}"}).status_code == 403
    # Admin → allowed (200)
    assert fresh.get("/v1/setup", headers={"Authorization": f"Bearer {admin_tok}"}).status_code == 200


def test_forged_x_admin_header_is_ignored(monkeypatch):
    # The old, forgeable X-Admin gate is gone: a header alone grants nothing.
    app, _ = _app(monkeypatch)
    c = TestClient(app)
    assert c.get("/v1/setup", headers={"X-Admin": "true"}).status_code == 401


# --------------------------------------------------------------------------- #
# identity is derived from the token, never from the request body
# --------------------------------------------------------------------------- #
def test_body_user_id_cannot_impersonate(monkeypatch):
    app, store = _app(monkeypatch, with_llm=True)
    c = TestClient(app)
    admin = _bootstrap(c)
    admin_tok = admin["access_token"]
    admin_id = admin["user"]["id"]

    # Post a turn while spoofing someone else's id in the body.
    r = c.post("/v1/responses",
               json={"input": "hi", "stream": False, "user_id": "victim", "user": "victim"},
               headers={"Authorization": f"Bearer {admin_tok}"})
    assert r.status_code == 200

    # The conversation is owned by the authenticated admin, not "victim".
    data = c.get("/v1/conversations",
                 headers={"Authorization": f"Bearer {admin_tok}"}).json()["data"]
    assert len(data) == 1
    conv_id = data[0]["id"]

    import asyncio
    conv = asyncio.run(store.get_conversation(conv_id))
    assert conv.user_id == admin_id
    assert conv.user_id != "victim"


def test_responses_requires_authentication(monkeypatch):
    app, _ = _app(monkeypatch, with_llm=True)
    c = TestClient(app)
    _bootstrap(c)  # a user exists, but this request carries no token
    fresh = TestClient(app)
    assert fresh.post("/v1/responses", json={"input": "hi", "stream": False}).status_code == 401
