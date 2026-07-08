import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.tools.artifacts import sign_artifact_token
from app.deps import AppState
from app.routes.files import router as files_router
from app.store.memory import InMemoryStore
from tests.authutil import apply_auth

SECRET = "route-test-secret"


def _client(tmp_path, monkeypatch, *, max_bytes=None):
    nas_root = tmp_path / "nas"
    (nas_root / "users" / "u1").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FILES_URL_SECRET", SECRET)
    monkeypatch.setenv("FILES_NAS_LOCAL_ROOT", str(nas_root))
    if max_bytes is not None:
        monkeypatch.setenv("FILES_MAX_BYTES", str(max_bytes))
    app = FastAPI()
    app.state.app_state = AppState(store=InMemoryStore(), llm=None, default_model="x")
    app.include_router(files_router)
    # Authenticated as u1; the file token must belong to this user.
    return TestClient(apply_auth(app, user_id="u1", role="user")), nas_root


def _write(nas_root, rel, content: bytes):
    p = nas_root / "users" / "u1" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def _tok(rel, *, kind="markdown", mime="text/markdown", size=0, user="u1"):
    return sign_artifact_token(
        user_id=user, conversation_id="c1", rel=rel, mime=mime, kind=kind,
        size=size, secret=SECRET,
    )


def test_serves_markdown_inline(tmp_path, monkeypatch):
    c, nas = _client(tmp_path, monkeypatch)
    _write(nas, "report.md", b"# hello\n")
    r = c.get(f"/v1/files/{_tok('report.md')}", params={"user_id": "u1"})
    assert r.status_code == 200
    assert r.text == "# hello\n"
    assert "inline" in r.headers["content-disposition"]
    assert "report.md" in r.headers["content-disposition"]


def test_file_kind_is_attachment(tmp_path, monkeypatch):
    c, nas = _client(tmp_path, monkeypatch)
    _write(nas, "data.bin", b"\x00\x01\x02")
    tok = _tok("data.bin", kind="file", mime="application/octet-stream")
    r = c.get(f"/v1/files/{tok}", params={"user_id": "u1"})
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]


def test_bad_signature_403(tmp_path, monkeypatch):
    c, nas = _client(tmp_path, monkeypatch)
    _write(nas, "report.md", b"x")
    r = c.get(f"/v1/files/{_tok('report.md')}x", params={"user_id": "u1"})
    assert r.status_code == 403


def test_wrong_user_403(tmp_path, monkeypatch):
    # Token minted for u2 but the caller is authenticated as u1 → rejected.
    c, nas = _client(tmp_path, monkeypatch)
    _write(nas, "report.md", b"x")
    r = c.get(f"/v1/files/{_tok('report.md', user='u2')}")
    assert r.status_code == 403


def test_path_traversal_403(tmp_path, monkeypatch):
    c, nas = _client(tmp_path, monkeypatch)
    (nas / "evil.txt").write_bytes(b"secret")
    tok = _tok("../../evil.txt", kind="text", mime="text/plain")
    r = c.get(f"/v1/files/{tok}", params={"user_id": "u1"})
    assert r.status_code == 403


def test_missing_file_404(tmp_path, monkeypatch):
    c, _nas = _client(tmp_path, monkeypatch)
    r = c.get(f"/v1/files/{_tok('nope.md')}", params={"user_id": "u1"})
    assert r.status_code == 404


def test_oversize_413(tmp_path, monkeypatch):
    c, nas = _client(tmp_path, monkeypatch, max_bytes=4)
    _write(nas, "big.txt", b"way too many bytes")
    tok = _tok("big.txt", kind="text", mime="text/plain", size=18)
    r = c.get(f"/v1/files/{tok}", params={"user_id": "u1"})
    assert r.status_code == 413


def test_no_secret_503(tmp_path, monkeypatch):
    c, nas = _client(tmp_path, monkeypatch)
    _write(nas, "report.md", b"x")
    tok = _tok("report.md")
    monkeypatch.setenv("FILES_URL_SECRET", "")
    r = c.get(f"/v1/files/{tok}", params={"user_id": "u1"})
    assert r.status_code == 503
