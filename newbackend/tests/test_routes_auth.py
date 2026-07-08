import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.deps import AppState
from app.routes.auth import router as auth_router
from app.store.memory import InMemoryStore


def _client(monkeypatch, *, secret="routes-auth-secret"):
    monkeypatch.setenv("JWT_SECRET", secret)
    store = InMemoryStore()
    app = FastAPI()
    app.state.app_state = AppState(store=store, llm=None, default_model="x")
    app.include_router(auth_router)
    return TestClient(app), store


def _bootstrap_admin(c, email="admin@example.com", password="password123"):
    return c.post("/v1/auth/bootstrap", json={"email": email, "password": password})


# --------------------------------------------------------------------------- #
# bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_needed_toggles(monkeypatch):
    c, _ = _client(monkeypatch)
    assert c.get("/v1/auth/bootstrap").json()["needed"] is True
    _bootstrap_admin(c)
    assert c.get("/v1/auth/bootstrap").json()["needed"] is False


def test_bootstrap_creates_admin_and_issues_token(monkeypatch):
    c, _ = _client(monkeypatch)
    r = _bootstrap_admin(c)
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer" and body["access_token"]
    assert body["user"]["role"] == "admin" and body["user"]["status"] == "active"
    # httpOnly cookie was set on the client
    assert "access_token" in r.cookies


def test_second_bootstrap_conflicts(monkeypatch):
    c, _ = _client(monkeypatch)
    _bootstrap_admin(c)
    r = c.post("/v1/auth/bootstrap",
               json={"email": "other@example.com", "password": "password123"})
    assert r.status_code == 409


def test_bootstrap_rejects_short_password_and_bad_email(monkeypatch):
    c, _ = _client(monkeypatch)
    assert c.post("/v1/auth/bootstrap",
                  json={"email": "admin@example.com", "password": "short"}).status_code == 400
    assert c.post("/v1/auth/bootstrap",
                  json={"email": "notanemail", "password": "password123"}).status_code == 400


def test_bootstrap_503_when_secret_unconfigured(monkeypatch):
    c, _ = _client(monkeypatch, secret="")
    r = _bootstrap_admin(c)
    assert r.status_code == 503


# --------------------------------------------------------------------------- #
# login / me / logout
# --------------------------------------------------------------------------- #
def test_login_ok_wrong_and_case_insensitive_email(monkeypatch):
    c, _ = _client(monkeypatch)
    _bootstrap_admin(c, email="admin@example.com", password="password123")
    ok = c.post("/v1/auth/login", json={"email": "ADMIN@example.com", "password": "password123"})
    assert ok.status_code == 200 and ok.json()["user"]["role"] == "admin"
    bad = c.post("/v1/auth/login", json={"email": "admin@example.com", "password": "wrong"})
    assert bad.status_code == 401
    missing = c.post("/v1/auth/login", json={"email": "ghost@example.com", "password": "password123"})
    assert missing.status_code == 401


def test_login_disabled_user_rejected(monkeypatch):
    c, store = _client(monkeypatch)
    admin = _bootstrap_admin(c).json()["user"]
    tok = _client_login_token(c)
    # invite + accept a regular user
    inv = c.post("/v1/auth/invite", json={"email": "u@example.com", "role": "user"},
                 headers={"Authorization": f"Bearer {tok}"}).json()
    c.post("/v1/auth/accept-invite",
           json={"token": inv["invite_token"], "password": "password123"})
    uid = inv["user"]["id"]
    # admin disables the user
    r = c.post(f"/v1/auth/users/{uid}/status", json={"status": "disabled"},
               headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    # the disabled user can no longer log in
    denied = c.post("/v1/auth/login", json={"email": "u@example.com", "password": "password123"})
    assert denied.status_code == 401


def _client_login_token(c, email="admin@example.com", password="password123"):
    return c.post("/v1/auth/login", json={"email": email, "password": password}).json()["access_token"]


def test_me_requires_token(monkeypatch):
    c, _ = _client(monkeypatch)
    _bootstrap_admin(c)
    tok = _client_login_token(c)
    # bearer token path
    fresh = TestClient(c.app)  # no cookies carried over
    assert fresh.get("/v1/auth/me").status_code == 401
    r = fresh.get("/v1/auth/me", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200 and r.json()["user"]["email"] == "admin@example.com"


def test_logout_clears_cookie(monkeypatch):
    c, _ = _client(monkeypatch)
    _bootstrap_admin(c)
    assert "access_token" in c.cookies
    r = c.post("/v1/auth/logout")
    assert r.status_code == 200
    # the login cookie is cleared; /me via the now-empty cookie jar is unauthorized
    assert c.get("/v1/auth/me").status_code == 401


# --------------------------------------------------------------------------- #
# invite / accept-invite
# --------------------------------------------------------------------------- #
def test_invite_requires_admin(monkeypatch):
    c, _ = _client(monkeypatch)
    _bootstrap_admin(c)
    tok = _client_login_token(c)
    # invite a regular user and accept it
    inv = c.post("/v1/auth/invite", json={"email": "u@example.com", "role": "user"},
                 headers={"Authorization": f"Bearer {tok}"}).json()
    user_tok = c.post("/v1/auth/accept-invite",
                      json={"token": inv["invite_token"], "password": "password123"}
                      ).json()["access_token"]
    # a fresh client with only the user's bearer token cannot invite
    fresh = TestClient(c.app)
    denied = fresh.post("/v1/auth/invite", json={"email": "x@example.com", "role": "user"},
                        headers={"Authorization": f"Bearer {user_tok}"})
    assert denied.status_code == 403
    # and no token at all is 401
    assert fresh.post("/v1/auth/invite", json={"email": "x@example.com"}).status_code == 401


def test_invite_returns_copyable_link_and_accept_activates(monkeypatch):
    c, _ = _client(monkeypatch)
    _bootstrap_admin(c)
    tok = _client_login_token(c)
    inv = c.post("/v1/auth/invite", json={"email": "u@example.com", "role": "user"},
                 headers={"Authorization": f"Bearer {tok}"})
    assert inv.status_code == 200
    body = inv.json()
    assert body["invite_token"] and body["invite_path"] == f"/?invite={body['invite_token']}"
    assert body["invite_url"].endswith(body["invite_path"])
    assert body["user"]["status"] == "invited"
    # accept sets the password, activates, and logs in
    acc = c.post("/v1/auth/accept-invite",
                 json={"token": body["invite_token"], "password": "password123"})
    assert acc.status_code == 200 and acc.json()["user"]["status"] == "active"
    # the new user can now log in normally
    assert c.post("/v1/auth/login",
                  json={"email": "u@example.com", "password": "password123"}).status_code == 200


def test_invite_url_uses_public_base_url_when_set(monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://pai.example.com/")
    c, _ = _client(monkeypatch)
    _bootstrap_admin(c)
    tok = _client_login_token(c)
    inv = c.post("/v1/auth/invite", json={"email": "u@example.com", "role": "user"},
                 headers={"Authorization": f"Bearer {tok}"})
    body = inv.json()
    # Link points at the configured public host, not the request host.
    assert body["invite_url"] == f"https://pai.example.com{body['invite_path']}"
    assert "testserver" not in body["invite_url"]


def test_accept_invite_rejects_bad_token(monkeypatch):
    c, _ = _client(monkeypatch)
    _bootstrap_admin(c)
    r = c.post("/v1/auth/accept-invite",
               json={"token": "not-a-real-token", "password": "password123"})
    assert r.status_code == 400


def test_invite_duplicate_email_conflicts(monkeypatch):
    c, _ = _client(monkeypatch)
    _bootstrap_admin(c, email="admin@example.com")
    tok = _client_login_token(c)
    # inviting the existing admin's email conflicts
    r = c.post("/v1/auth/invite", json={"email": "admin@example.com", "role": "user"},
               headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 409


def test_change_password_flow(monkeypatch):
    c, _ = _client(monkeypatch)
    _bootstrap_admin(c, password="password123")
    tok = _client_login_token(c)
    hdr = {"Authorization": f"Bearer {tok}"}
    # wrong current password is rejected
    assert c.post("/v1/auth/change-password",
                  json={"old_password": "wrong", "new_password": "newpassword1"},
                  headers=hdr).status_code == 400
    ok = c.post("/v1/auth/change-password",
                json={"old_password": "password123", "new_password": "newpassword1"},
                headers=hdr)
    assert ok.status_code == 200
    # the new password now logs in; the old one does not
    assert c.post("/v1/auth/login",
                  json={"email": "admin@example.com", "password": "newpassword1"}).status_code == 200
    assert c.post("/v1/auth/login",
                  json={"email": "admin@example.com", "password": "password123"}).status_code == 401


def test_admin_cannot_disable_self(monkeypatch):
    c, _ = _client(monkeypatch)
    admin = _bootstrap_admin(c).json()["user"]
    tok = _client_login_token(c)
    r = c.post(f"/v1/auth/users/{admin['id']}/status", json={"status": "disabled"},
               headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 400
