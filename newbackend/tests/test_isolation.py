import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.deps import AppState
from app.routes.auth import router as auth_router
from app.routes.responses import router as responses_router
from app.routes.conversations import router as conversations_router
from app.routes.users import router as users_router
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


def _client(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "isolation-secret")
    app = FastAPI()
    app.state.app_state = AppState(store=InMemoryStore(), llm=_EchoLLM(), default_model="m")
    app.include_router(auth_router)
    app.include_router(responses_router)
    app.include_router(conversations_router)
    app.include_router(users_router)
    return TestClient(app)


def _hdr(tok):
    return {"Authorization": f"Bearer {tok}"}


def _bootstrap_admin(c):
    return c.post("/v1/auth/bootstrap",
                  json={"email": "admin@example.com", "password": "password123"}
                  ).json()["access_token"]


def _make_user(c, admin_tok, email):
    inv = c.post("/v1/auth/invite", json={"email": email, "role": "user"},
                 headers=_hdr(admin_tok)).json()
    acc = c.post("/v1/auth/accept-invite",
                 json={"token": inv["invite_token"], "password": "password123"}).json()
    return acc["access_token"], acc["user"]["id"]


def _post_turn(c, tok, text="hello"):
    body = c.post("/v1/responses", json={"input": text, "stream": False},
                  headers=_hdr(tok)).json()
    return body["id"], body["conversation"]["id"]


def test_users_cannot_read_or_delete_each_others_data(monkeypatch):
    c = _client(monkeypatch)
    admin_tok = _bootstrap_admin(c)
    tok_a, id_a = _make_user(c, admin_tok, "a@example.com")
    tok_b, id_b = _make_user(c, admin_tok, "b@example.com")

    resp_a, conv_a = _post_turn(c, tok_a, "from-a")
    resp_b, conv_b = _post_turn(c, tok_b, "from-b")

    # --- conversation isolation ---
    # A's list contains only A's conversation.
    a_list = c.get("/v1/conversations", headers=_hdr(tok_a)).json()["data"]
    assert [x["id"] for x in a_list] == [conv_a]
    # A cannot read or delete B's conversation (404, not 403 — no existence leak).
    assert c.get(f"/v1/conversations/{conv_b}", headers=_hdr(tok_a)).status_code == 404
    assert c.delete(f"/v1/conversations/{conv_b}", headers=_hdr(tok_a)).status_code == 404
    # B's conversation still exists (the failed delete was a no-op).
    assert c.get(f"/v1/conversations/{conv_b}", headers=_hdr(tok_b)).status_code == 200

    # --- response isolation ---
    assert c.get(f"/v1/responses/{resp_b}", headers=_hdr(tok_a)).status_code == 404
    assert c.delete(f"/v1/responses/{resp_b}", headers=_hdr(tok_a)).status_code == 404
    assert c.post(f"/v1/responses/{resp_b}/cancel", headers=_hdr(tok_a)).status_code == 404
    # owner still sees their own response
    assert c.get(f"/v1/responses/{resp_b}", headers=_hdr(tok_b)).status_code == 200

    # --- memory isolation ---
    # A cannot address B's memory namespace.
    assert c.get(f"/v1/users/{id_b}/memories", headers=_hdr(tok_a)).status_code == 404
    assert c.delete(f"/v1/users/{id_b}/memories", headers=_hdr(tok_a)).status_code == 404
    # A can address its own.
    assert c.get(f"/v1/users/{id_a}/memories", headers=_hdr(tok_a)).status_code == 200


def test_admin_is_also_scoped_on_the_data_plane(monkeypatch):
    # Admin authority is the control plane, not other users' private chats:
    # an admin cannot read a regular user's conversation or response either.
    c = _client(monkeypatch)
    admin_tok = _bootstrap_admin(c)
    tok_a, _ = _make_user(c, admin_tok, "a@example.com")
    resp_a, conv_a = _post_turn(c, tok_a, "from-a")

    assert c.get(f"/v1/conversations/{conv_a}", headers=_hdr(admin_tok)).status_code == 404
    assert c.get(f"/v1/responses/{resp_a}", headers=_hdr(admin_tok)).status_code == 404
