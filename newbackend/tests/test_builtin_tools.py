import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import httpx
from agent.tools.builtin.datetime_tool import make_current_datetime_tool
from agent.tools.builtin.web_fetch import make_web_fetch_tool
from agent.tools.builtin.web_search import make_web_search_tool
from agent.tools.builtin.code_sandbox import make_code_sandbox_tool
from agent.tools.builtin.install_skill import make_install_skill_tool, _validate_source_allowed
from agent.tools.defaults import build_default_registry
from agent.message import ToolCall
from agent.tools.base import ToolBox
from agent.tools.sandbox_providers import AgentRunRestSandboxProvider
from agent.tools.scope import ToolScope, reset_current_tool_scope, set_current_tool_scope
from app.agent_config import AgentConfigDocument


def test_current_datetime_tool_returns_a_time_string():
    t = make_current_datetime_tool()
    assert t.name == "current_datetime"
    out = asyncio.run(t.fn())
    assert isinstance(out, str) and len(out) >= 8


class _FakeResp:
    def __init__(self, text):
        self.text = text
    def raise_for_status(self):
        return None


class _FakeClient:
    def __init__(self, resp, boom=False):
        self._resp = resp
        self._boom = boom
    async def __aenter__(self):
        return self
    async def __aexit__(self, *a):
        return False
    async def get(self, url):
        if self._boom:
            raise RuntimeError("network down")
        return self._resp


def test_web_fetch_extracts_text_and_truncates():
    html = "<html><head><style>x{}</style></head><body><h1>Hi</h1><p>World &amp; more</p><script>bad()</script></body></html>"
    t = make_web_fetch_tool(client_factory=lambda: _FakeClient(_FakeResp(html)), limit=50)
    out = asyncio.run(t.fn(url="http://x"))
    assert "Hi" in out and "World" in out
    assert "bad()" not in out and "<" not in out
    assert len(out) <= 50


def test_web_fetch_returns_error_string_on_failure():
    t = make_web_fetch_tool(client_factory=lambda: _FakeClient(None, boom=True))
    out = asyncio.run(t.fn(url="http://x"))
    assert out.startswith("web_fetch failed:")


class _FakeProvider:
    async def search(self, query, num_results):
        return [{"title": "T1", "url": "http://1", "snippet": "S1"},
                {"title": "T2", "url": "http://2", "snippet": "S2"}][:num_results]


def test_web_search_formats_results():
    t = make_web_search_tool(_FakeProvider())
    out = asyncio.run(t.fn(query="hello", num_results=2))
    assert "T1" in out and "http://2" in out and "S2" in out


def test_web_search_returns_error_string_on_provider_failure():
    class _Boom:
        async def search(self, q, n):
            raise RuntimeError("boom")
    t = make_web_search_tool(_Boom())
    out = asyncio.run(t.fn(query="x"))
    assert out.startswith("web_search failed:")


class _FakeSandboxProvider:
    default_timeout_seconds = 12

    async def run_code(self, *, code, language, timeout, context_id=None, cwd=None):
        return {
            "stdout": f"{language}:{code}:{timeout}:{cwd or ''}",
            "stderr": "",
            "exit_code": 0,
        }


def test_code_sandbox_formats_provider_result():
    t = make_code_sandbox_tool(_FakeSandboxProvider(), default_timeout=12)
    out = asyncio.run(t.fn(code="print(1)", cwd="/home/user"))
    assert '"exit_code": 0' in out
    assert "python:print(1):12:/home/user" in out


class _Settings:
    search_provider = "none"


def test_default_registry_omits_search_when_unconfigured():
    reg = build_default_registry(_Settings())
    assert set(reg.names()) == {"current_datetime", "web_fetch"}


def test_default_registry_includes_injected_search_provider():
    reg = build_default_registry(_Settings(), search_provider=_FakeProvider())
    assert "web_search" in reg.names()


def test_default_registry_includes_configured_tavily_search():
    doc = AgentConfigDocument(**{
        "providers": [{
            "id": "search.default",
            "type": "search",
            "name": "tavily",
            "settings": {"provider": "tavily", "api_key": "tvly", "max_results": 5},
            "used_by": ["search"],
        }],
        "capabilities": [{
            "id": "search",
            "kind": "core_tool",
            "name": "Search",
            "enabled": True,
            "permission": "auto",
            "provider_refs": ["search.default"],
        }],
    })
    reg = build_default_registry(_Settings(), agent_config=doc)
    assert "web_search" in reg.names()


def test_default_registry_includes_configured_agentrun_sandbox(monkeypatch):
    monkeypatch.setenv("AGENTRUN_ACCESS_KEY_ID", "id")
    monkeypatch.setenv("AGENTRUN_ACCESS_KEY_SECRET", "secret")
    monkeypatch.setenv("AGENTRUN_ACCOUNT_ID", "account")
    doc = AgentConfigDocument(**{
        "providers": [{
            "id": "sandbox.default",
            "type": "sandbox",
            "name": "AgentRun",
            "settings": {
                "provider": "agentrun",
                "template_name": "code-template",
                "access_key_id_env": "AGENTRUN_ACCESS_KEY_ID",
                "access_key_secret_env": "AGENTRUN_ACCESS_KEY_SECRET",
                "account_id_env": "AGENTRUN_ACCOUNT_ID",
            },
            "used_by": ["sandbox"],
        }],
        "capabilities": [{
            "id": "sandbox",
            "kind": "core_tool",
            "name": "Sandbox",
            "enabled": True,
            "permission": "auto",
            "provider_refs": ["sandbox.default"],
        }],
    })
    reg = build_default_registry(_Settings(), agent_config=doc)
    assert "code_sandbox" in reg.names()


def test_default_registry_includes_configured_agentrun_rest_sandbox():
    doc = AgentConfigDocument(**{
        "providers": [{
            "id": "sandbox.default",
            "type": "sandbox",
            "name": "AgentRun REST",
            "settings": {
                "provider": "agentrun_rest",
                "endpoint": "https://sandbox-gateway.example.com",
                "template_name": "code-template",
                "account_id": "acct-1",
            },
            "used_by": ["sandbox"],
        }],
        "capabilities": [{
            "id": "sandbox",
            "kind": "core_tool",
            "name": "Sandbox",
            "enabled": True,
            "permission": "auto",
            "provider_refs": ["sandbox.default"],
        }],
    })
    reg = build_default_registry(_Settings(), agent_config=doc)
    assert "code_sandbox" in reg.names()


def test_agentrun_rest_provider_creates_per_user_scope(monkeypatch):
    requests = []

    class _Resp:
        def __init__(self, body, status_code=200):
            self._body = body
            self.content = b"{}"
            self.status_code = status_code

        def raise_for_status(self):
            return None

        def json(self):
            return self._body

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def request(self, method, url, headers=None, json=None):
            requests.append((method, url, headers or {}, json or {}))
            if url.endswith("/sandboxes"):
                return _Resp({"code": "SUCCESS", "data": {"sandboxId": "sb-doc-create"}})
            return _Resp({
                "contextId": "ctx-1",
                "results": [
                    {"type": "stdout", "text": "ok"},
                    {"type": "endOfExecution", "status": "ok"},
                ],
            })

    monkeypatch.setattr("agent.tools.sandbox_providers.httpx.AsyncClient", _Client)
    provider = AgentRunRestSandboxProvider({
        "endpoint": "https://gateway.test",
        "template_name": "code-template",
        "api_key": "secret",
        "account_id": "acct-1",
        "oss_mount_config": {"mount_points": []},
        "nas_config": {"mount_points": []},
    })
    token = set_current_tool_scope(ToolScope(
        user_id="u1",
        conversation_id="c1",
        agent_id="main",
        skill_fingerprint="skills123",
        skill_mounts=[{
            "id": "skill.writer",
            "oss": {
                "bucketName": "agent-skills",
                "bucketPath": "/skills/writer@1.0.0",
                "endpoint": "oss-cn-hangzhou.aliyuncs.com",
                "mountDir": "/mnt/skills/writer",
                "readOnly": True,
            },
        }],
    ))
    try:
        result = asyncio.run(provider.run_code(code="print(1)", language="python", timeout=60))
    finally:
        reset_current_tool_scope(token)

    assert result["stdout"] == "ok"
    create = requests[0]
    execute = requests[1]
    assert create[0] == "POST"
    assert create[1] == "https://gateway.test/sandboxes"
    assert create[3] == {
        "templateName": "code-template",
        "ossMountConfig": {
            "mountPoints": [{
                "bucketName": "agent-skills",
                "bucketPath": "/skills/writer@1.0.0",
                "endpoint": "oss-cn-hangzhou.aliyuncs.com",
                "mountDir": "/mnt/skills/writer",
                "readOnly": True,
            }]
        },
    }
    assert execute[0] == "POST"
    assert execute[1] == "https://gateway.test/sandboxes/sb-doc-create/contexts/execute"
    assert execute[2]["X-Acs-Parent-Id"] == "acct-1"
    assert execute[2]["X-API-Key"] == "secret"
    assert "Authorization" not in execute[2]
    assert execute[3]["timeout"] == 30
    assert execute[3]["scope_key"] == "conversation:c1:agent:main:skills:skills123"


def test_agentrun_rest_provider_reuses_user_session(monkeypatch):
    requests = []

    class _Resp:
        content = b"{}"

        def __init__(self, body, status_code=200):
            self._body = body
            self.status_code = status_code

        def raise_for_status(self):
            return None

        def json(self):
            return self._body

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def request(self, method, url, headers=None, json=None):
            requests.append((method, url, json or {}))
            if url.endswith("/sandboxes"):
                return _Resp({"code": "SUCCESS", "data": {"sandboxId": "sb-reused"}})
            return _Resp({"stdout": "ok", "stderr": "", "exit_code": 0})

    monkeypatch.setattr("agent.tools.sandbox_providers.httpx.AsyncClient", _Client)
    provider = AgentRunRestSandboxProvider({
        "endpoint": "https://gateway.test",
        "template_name": "code-template",
        "account_id": "acct-1",
    })
    token = set_current_tool_scope(ToolScope(user_id="u1"))
    try:
        asyncio.run(provider.run_code(code="a=1", language="python", timeout=3))
        asyncio.run(provider.run_code(code="a+1", language="python", timeout=3))
    finally:
        reset_current_tool_scope(token)

    create_calls = [item for item in requests if item[1].endswith("/sandboxes")]
    execute_calls = [item for item in requests if item[1].endswith("/contexts/execute")]
    assert len(create_calls) == 1
    assert len(execute_calls) == 2


def test_agentrun_rest_provider_recreates_expired_cached_sandbox(monkeypatch):
    requests = []
    execute_attempts = 0

    class _Resp:
        content = b"{}"

        def __init__(self, body=None, status_code=200):
            self._body = body or {}
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                request = httpx.Request("POST", "https://gateway.test")
                response = httpx.Response(self.status_code, request=request)
                raise httpx.HTTPStatusError("expired", request=request, response=response)

        def json(self):
            return self._body

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def request(self, method, url, headers=None, json=None):
            nonlocal execute_attempts
            requests.append((method, url, json or {}))
            if url.endswith("/sandboxes"):
                sandbox_id = f"sb-{len([r for r in requests if r[1].endswith('/sandboxes')])}"
                return _Resp({"code": "SUCCESS", "data": {"sandboxId": sandbox_id}})
            if url.endswith("/contexts/execute"):
                execute_attempts += 1
                if execute_attempts == 1:
                    return _Resp(status_code=404)
            return _Resp({"stdout": "ok", "stderr": "", "exit_code": 0})

    monkeypatch.setattr("agent.tools.sandbox_providers.httpx.AsyncClient", _Client)
    provider = AgentRunRestSandboxProvider({
        "endpoint": "https://gateway.test",
        "template_name": "code-template",
        "account_id": "acct-1",
    })
    token = set_current_tool_scope(ToolScope(user_id="u1"))
    try:
        result = asyncio.run(provider.run_code(code="a=1", language="python", timeout=3))
    finally:
        reset_current_tool_scope(token)

    create_calls = [item for item in requests if item[1].endswith("/sandboxes")]
    execute_calls = [item for item in requests if item[1].endswith("/contexts/execute")]
    assert result["stdout"] == "ok"
    assert len(create_calls) == 2
    assert len(execute_calls) == 2
    assert execute_calls[0][1] == "https://gateway.test/sandboxes/sb-1/contexts/execute"
    assert execute_calls[1][1] == "https://gateway.test/sandboxes/sb-2/contexts/execute"


def test_install_skill_tool_requires_admin(monkeypatch, tmp_path):
    calls = []

    def _fake_install(*args, **kwargs):
        calls.append((args, kwargs))
        return {"id": "skill.demo", "status": "ready"}

    monkeypatch.setattr("agent.tools.builtin.install_skill._install_skill_sync", _fake_install)
    doc = AgentConfigDocument(**{"skills": {"root": str(tmp_path)}})
    tool = make_install_skill_tool(type("Settings", (), {"app_env": "development"})(), doc)
    box = ToolBox([tool])
    tc = ToolCall(
        id="call_1",
        name="install_skill",
        arguments='{"source":{"type":"git","url":"https://example.com/skills.git"}}',
    )

    denied = asyncio.run(box.dispatch(tc, scope=ToolScope(metadata={"role": "user"})))
    assert not denied.ok
    assert "requires admin permission" in denied.error
    assert calls == []

    allowed = asyncio.run(box.dispatch(tc, scope=ToolScope(metadata={"role": "admin"})))
    assert allowed.ok
    assert "skill.demo" in allowed.content
    assert len(calls) == 1


def test_install_skill_url_git_disabled_in_production():
    install_config = {
        "allow_sources": ["zip_upload", "url", "git"],
        "production_allow_sources": ["zip_upload"],
    }
    for source_type in ("url", "git"):
        try:
            _validate_source_allowed(source_type, install_config, "production")
        except PermissionError as exc:
            assert source_type in str(exc)
        else:
            raise AssertionError(f"{source_type} should be disabled in production")

    _validate_source_allowed("zip_upload", install_config, "production")
