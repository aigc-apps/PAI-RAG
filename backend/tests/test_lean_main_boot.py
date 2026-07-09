# tests/app/test_lean_main_boot.py
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi.testclient import TestClient


def test_app_boots_in_memory_and_serves(monkeypatch, tmp_path):
    # Hermetic provider catalog: a keyless local "test" provider so the router
    # exposes a usable default model without any real credentials. The echo LLM
    # is registered under the router's default_model_id — the same id the
    # /v1/responses route resolves to — so no real API client is ever built.
    catalog = tmp_path / "config.yaml"
    catalog.write_text(
        "models:\n"
        "  default_model: test/echo\n"
        "  providers:\n"
        "    - name: test\n"
        "      base_url: http://test.local/v1\n"
        '      api_key_env: ""\n'
        "      models:\n"
        "        - id: echo\n"
    )
    monkeypatch.setenv("MODELS_PATH", str(catalog))
    monkeypatch.setenv("STORE_BACKEND", "memory")
    # No OPENAI_API_KEY set: boot must succeed using the catalog's `test`
    # provider alone — the legacy openai fallback client is skipped when its
    # key is absent (see _build_llm), so deployments configuring only a
    # non-OpenAI provider boot without OpenAI credentials.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEFAULT_MODEL", "test/echo")
    monkeypatch.setenv("JWT_SECRET", "boot-test-secret")
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
        # rather than state.llm; register the fake LLM under the router's default
        # (the id the route resolves to when the request omits `model`) so no real
        # API call is made.
        router = c.app.state.app_state.router
        if router is not None:
            router.register_llm(router.default_model_id, echo)
        # First-run bootstrap creates the admin and sets the auth cookie on the
        # client, exercising the real auth stack; /v1/responses now requires it.
        boot = c.post("/v1/auth/bootstrap",
                      json={"email": "admin@example.com", "password": "password123"})
        assert boot.status_code == 200
        r = c.post("/v1/responses", json={"input": "hi", "stream": False})
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "response" and body["status"] == "completed"
        # The agent prepends a `[System Time: ...]` prefix to the user turn, so the
        # echoed text contains "hi" rather than equalling it exactly.
        text = body["output"][0]["content"][0]["text"]
        assert text.startswith("echo:")
        assert "hi" in text


def test_app_boots_with_sql_db_and_runs_migrations(monkeypatch, tmp_path):
    """The persistent-DB path must migrate on boot. This exercises migrate()
    inside the app's LIVE event loop — the case where a naive command.upgrade()
    would blow up on asyncio.run-inside-a-running-loop — and then serves a turn
    off the freshly-migrated schema (auth bootstrap writes to users)."""
    catalog = tmp_path / "config.yaml"
    catalog.write_text(
        "models:\n"
        "  default_model: test/echo\n"
        "  providers:\n"
        "    - name: test\n"
        "      base_url: http://test.local/v1\n"
        '      api_key_env: ""\n'
        "      models:\n"
        "        - id: echo\n"
    )
    monkeypatch.setenv("MODELS_PATH", str(catalog))
    monkeypatch.setenv("STORE_BACKEND", "sql")
    monkeypatch.setenv("DB_URL", f"sqlite+aiosqlite:///{tmp_path / 'boot.db'}")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("DEFAULT_MODEL", "test/echo")
    monkeypatch.setenv("JWT_SECRET", "boot-test-secret")
    import importlib
    import app.lean_main as m

    importlib.reload(m)

    from common.llm.models import TextChunk
    from openai.types.chat.chat_completion_chunk import CompletionUsage

    class _EchoLLM:
        async def astream(self, messages, tools=None, **kwargs):
            last = ""
            for msg in messages:
                if msg.get("role") == "user":
                    last = msg.get("content") or ""
            usage = CompletionUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)

            async def gen():
                yield TextChunk(delta=f"echo:{last}", usage=None)
                yield TextChunk(delta="", usage=usage)

            return gen()

    with TestClient(m.app) as c:  # lifespan runs migrate() here
        echo = _EchoLLM()
        c.app.state.app_state.llm = echo
        router = c.app.state.app_state.router
        if router is not None:
            router.register_llm(router.default_model_id, echo)
        # Bootstrap writes to the migrated users table — the query that used to
        # fail with "no such column: users.email" on an un-migrated DB.
        boot = c.post("/v1/auth/bootstrap",
                      json={"email": "admin@example.com", "password": "password123"})
        assert boot.status_code == 200
        r = c.post("/v1/responses", json={"input": "hi", "stream": False})
        assert r.status_code == 200
        assert r.json()["status"] == "completed"
