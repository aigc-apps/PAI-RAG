import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import httpx
from agent.tools.builtin.datetime_tool import make_current_datetime_tool
from agent.tools.builtin.web_fetch import make_web_fetch_tool
from agent.tools.builtin.web_search import make_web_search_tool
from agent.tools.builtin.code_interpreter import make_code_interpreter_tool
from agent.tools.builtin.shell import make_shell_tool
from agent.tools.builtin.install_skill import (
    make_install_skill_tool,
    _validate_source_allowed,
    _find_skill_dir,
    _read_skill_package,
    _dependency_summary,
)
from agent.tools.builtin.load_skill import make_load_skill_tool
from agent.tools.builtin.read_skill_resource import make_read_skill_resource_tool
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

    async def run_command(self, *, command, cwd=None, timeout=None):
        return {
            "stdout": f"$ {command} @ {cwd or ''}",
            "stderr": "",
            "exit_code": 0,
            "cwd": cwd or "/home/user",
        }


def test_code_interpreter_formats_provider_result():
    t = make_code_interpreter_tool(_FakeSandboxProvider(), default_timeout=12)
    out = asyncio.run(t.fn(code="print(1)", cwd="/home/user"))
    assert '"exit_code": 0' in out
    assert "python:print(1):12:/home/user" in out


def test_shell_tool_formats_provider_result():
    t = make_shell_tool(_FakeSandboxProvider(), default_timeout=12)
    out = asyncio.run(t.fn(command="ls -la", cwd="/home/user"))
    assert '"exit_code": 0' in out
    assert "$ ls -la @ /home/user" in out
    assert '"cwd": "/home/user"' in out


def test_shell_tool_reports_nonzero_exit():
    class _Boom:
        default_timeout_seconds = 12

        async def run_command(self, *, command, cwd=None, timeout=None):
            return {"stdout": "", "stderr": "not found", "exit_code": 127}

    t = make_shell_tool(_Boom(), default_timeout=12)
    out = asyncio.run(t.fn(command="nope"))
    assert '"exit_code": 127' in out
    assert "not found" in out


def test_shell_tool_returns_error_string_on_provider_failure():
    class _Boom:
        default_timeout_seconds = 12

        async def run_command(self, *, command, cwd=None, timeout=None):
            raise RuntimeError("sandbox gone")

    t = make_shell_tool(_Boom(), default_timeout=12)
    out = asyncio.run(t.fn(command="ls"))
    assert out.startswith("shell failed:")


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
    assert "code_interpreter" in reg.names()
    assert "shell" in reg.names()


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
                "api_key": "secret",
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
    assert "code_interpreter" in reg.names()
    assert "shell" in reg.names()


def test_agentrun_rest_provider_derives_endpoint_from_account_id():
    provider = AgentRunRestSandboxProvider({
        "template_name": "code-template",
        "api_key": "secret",
        "account_id": "acct-1",
        "region": "cn-hangzhou",
    })
    assert provider.endpoint == "https://acct-1.agentrun-data.cn-hangzhou.aliyuncs.com"


def test_agentrun_rest_provider_skipped_without_api_key():
    doc = AgentConfigDocument(**{
        "providers": [{
            "id": "sandbox.default",
            "type": "sandbox",
            "name": "AgentRun REST",
            "settings": {
                "provider": "agentrun_rest",
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
    assert "code_interpreter" not in reg.names()
    assert "shell" not in reg.names()


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
        "nas_config": {
            "user_id": 1000,
            "group_id": 1000,
            "user_server_addr": "nas-cn-hangzhou.aliyuncs.com:/",
            "user_remote_path_template": "/users/{user_id}",
            "user_read_only": False,
        },
    })
    token = set_current_tool_scope(ToolScope(
        user_id="u1",
        conversation_id="c1",
        agent_id="main",
        skill_fingerprint="skills123",
        skill_mounts=[{
            "id": "skill.writer",
            "nas": {
                "serverAddr": "nas-cn-hangzhou.aliyuncs.com:/skills/writer@1.0.0",
                "remotePath": "/skills/writer@1.0.0",
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
        "templateType": "CodeInterpreter",
        "nasConfig": {
            "userId": 1000,
            "groupId": 1000,
            "mountPoints": [
                {
                    "serverAddr": "nas-cn-hangzhou.aliyuncs.com:/skills/writer@1.0.0",
                    "mountDir": "/mnt/skills/writer",
                    "readOnly": True,
                },
                {
                    "serverAddr": "nas-cn-hangzhou.aliyuncs.com:/users/u1",
                    "mountDir": "/mnt/user",
                    "readOnly": False,
                },
            ],
        },
        "envs": {
            "AGENT_SYSTEM_PATH": "/mnt/system",
            "AGENT_SKILL_PATH": "/mnt/skills",
            "AGENT_USER_PATH": "/mnt/user",
            "AGENT_USER_ID": "u1",
            "AGENT_SESSION_ID": "conversation:c1:agent:main:skills:skills123",
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


def test_agentrun_rest_provider_runs_shell_command(monkeypatch):
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

        async def __aexit__(self, *a):
            return False

        async def request(self, method, url, headers=None, json=None):
            requests.append((method, url, headers or {}, json or {}))
            if url.endswith("/sandboxes"):
                return _Resp({"code": "SUCCESS", "data": {"sandboxId": "sb-cmd"}})
            if url.endswith("/processes/cmd"):
                return _Resp({
                    "executionId": "tty_exec_001",
                    "status": "completed",
                    "result": {
                        "exitCode": 0,
                        "stdout": "total 24\ndrwxr-xr-x 3 user user 4096 Jan 15 10:30 .",
                        "stderr": "",
                        "cwd": "/home/user",
                        "executionTimeMs": 150,
                    },
                    "executionTimeMs": 150,
                })
            return _Resp({})

    monkeypatch.setattr("agent.tools.sandbox_providers.httpx.AsyncClient", _Client)
    provider = AgentRunRestSandboxProvider({
        "endpoint": "https://gateway.test",
        "template_name": "code-template",
        "api_key": "secret",
        "account_id": "acct-1",
    })
    token = set_current_tool_scope(ToolScope(user_id="u1"))
    try:
        result = asyncio.run(provider.run_command(command="ls -la", cwd="/home/user"))
    finally:
        reset_current_tool_scope(token)

    assert result["stdout"].startswith("total 24")
    assert result["stderr"] == ""
    assert result["exit_code"] == 0
    cmd = requests[-1]
    assert cmd[0] == "POST"
    assert cmd[1] == "https://gateway.test/sandboxes/sb-cmd/processes/cmd"
    assert cmd[2]["X-Acs-Parent-Id"] == "acct-1"
    # processes/cmd takes only {command, cwd} — no timeout/contextId fields.
    assert set(cmd[3].keys()) == {"command", "cwd"}
    assert cmd[3]["command"] == "ls -la"
    assert cmd[3]["cwd"] == "/home/user"


def test_agentrun_rest_provider_shell_command_failure_status(monkeypatch):
    requests = []

    class _Resp:
        content = b"{}"

        def __init__(self, body):
            self._body = body
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._body

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def request(self, method, url, headers=None, json=None):
            requests.append((method, url, json or {}))
            if url.endswith("/sandboxes"):
                return _Resp({"code": "SUCCESS", "data": {"sandboxId": "sb-fail"}})
            return _Resp({
                "status": "timeout",
                "result": {"exitCode": 0, "stdout": "", "stderr": ""},
            })

    monkeypatch.setattr("agent.tools.sandbox_providers.httpx.AsyncClient", _Client)
    provider = AgentRunRestSandboxProvider({
        "endpoint": "https://gateway.test",
        "template_name": "code-template",
        "api_key": "secret",
        "account_id": "acct-1",
    })
    token = set_current_tool_scope(ToolScope(user_id="u1"))
    try:
        result = asyncio.run(provider.run_command(command="sleep 999", cwd="/home/user"))
    finally:
        reset_current_tool_scope(token)

    # Non-terminal status with zero exitCode must still surface as a failure.
    assert result["exit_code"] == 1


def _make_fake_rest_client(requests):
    class _Resp:
        content = b"{}"

        def __init__(self, body):
            self._body = body
            self.status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return self._body

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def request(self, method, url, headers=None, json=None):
            requests.append((method, url, json or {}))
            if url.endswith("/sandboxes"):
                return _Resp({"code": "SUCCESS", "data": {"sandboxId": "sb-x"}})
            return _Resp({"stdout": "ok", "stderr": "", "exit_code": 0})

    return _Client


def test_agentrun_rest_provider_omits_nasconfig_and_envs_when_disabled(monkeypatch):
    requests = []
    monkeypatch.setattr(
        "agent.tools.sandbox_providers.httpx.AsyncClient",
        _make_fake_rest_client(requests),
    )
    provider = AgentRunRestSandboxProvider({
        "endpoint": "https://gateway.test",
        "template_name": "code-template",
        "account_id": "acct-1",
        "inject_env_contract": False,
    })
    token = set_current_tool_scope(ToolScope(user_id="u1"))
    try:
        asyncio.run(provider.run_code(code="print(1)", language="python", timeout=3))
    finally:
        reset_current_tool_scope(token)

    create_payload = requests[0][2]
    assert "nasConfig" not in create_payload
    assert "envs" not in create_payload
    assert create_payload["templateName"] == "code-template"


def test_agentrun_rest_provider_user_path_fallback_without_user_id(monkeypatch):
    requests = []
    monkeypatch.setattr(
        "agent.tools.sandbox_providers.httpx.AsyncClient",
        _make_fake_rest_client(requests),
    )
    provider = AgentRunRestSandboxProvider({
        "endpoint": "https://gateway.test",
        "template_name": "code-template",
        "account_id": "acct-1",
        "nas_config": {
            "user_server_addr": "nas-cn-hangzhou.aliyuncs.com:/",
            "user_remote_path_template": "/users/{user_id}",
        },
    })
    # No user_id, no conversation_id -> scope falls back to anonymous; the user
    # NAS remote path must still be templated from a stable fallback id.
    token = set_current_tool_scope(ToolScope())
    try:
        asyncio.run(provider.run_code(code="print(1)", language="python", timeout=3))
    finally:
        reset_current_tool_scope(token)

    create_payload = requests[0][2]
    nas = create_payload["nasConfig"]
    user_mount = nas["mountPoints"][-1]
    assert user_mount["mountDir"] == "/mnt/user"
    assert user_mount["readOnly"] is False
    assert user_mount["serverAddr"].startswith("nas-cn-hangzhou.aliyuncs.com:/users/")
    assert user_mount["serverAddr"] != "nas-cn-hangzhou.aliyuncs.com:/users/"
    assert create_payload["envs"]["AGENT_USER_ID"] == ""


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


def test_enable_skill_for_agent_tool_requires_admin_and_persists(tmp_path):
    from agent.tools.builtin.enable_skill import make_enable_skill_for_agent_tool
    from app.agent_config import load_agent_config

    config_path = str(tmp_path / "config.yaml")
    settings = type("Settings", (), {"config_path": config_path})()
    reloaded = []
    doc = load_agent_config(config_path)  # -> DEFAULT_DOCUMENT (agent "main", skill.writing ready+enabled)
    tool = make_enable_skill_for_agent_tool(
        settings, doc, on_config_change=lambda: reloaded.append(True)
    )
    box = ToolBox([tool])

    def _call(args, role):
        tc = ToolCall(id="c", name="enable_skill_for_agent", arguments=args)
        return asyncio.run(box.dispatch(tc, scope=ToolScope(metadata={"role": role})))

    # Non-admin is blocked before any mutation.
    denied = _call('{"skill_id":"skill.writing","enabled":false}', "user")
    assert not denied.ok
    assert "requires admin permission" in denied.error
    assert reloaded == []

    # Admin disables the (default-enabled) skill: a real change, persisted + reloaded.
    off = _call('{"skill_id":"skill.writing","enabled":false}', "admin")
    assert off.ok
    assert '"changed": true' in off.content
    assert '"runtime_reloaded": true' in off.content
    persisted = load_agent_config(config_path)
    assert "skill.writing" not in next(a for a in persisted.agents if a.id == "main").skills.enabled

    # Admin re-enables it.
    on = _call('{"skill_id":"skill.writing"}', "admin")
    assert on.ok
    assert '"changed": true' in on.content
    persisted = load_agent_config(config_path)
    assert "skill.writing" in next(a for a in persisted.agents if a.id == "main").skills.enabled
    assert reloaded == [True, True]


def test_enable_skill_for_agent_tool_rejects_not_ready_skill(tmp_path):
    from agent.tools.builtin.enable_skill import make_enable_skill_for_agent_tool
    from app.agent_config import load_agent_config

    config_path = str(tmp_path / "config.yaml")
    settings = type("Settings", (), {"config_path": config_path})()
    doc = load_agent_config(config_path)
    # skill.data_analysis depends on the (disabled) sandbox capability -> not ready.
    tool = make_enable_skill_for_agent_tool(settings, doc)
    box = ToolBox([tool])
    tc = ToolCall(
        id="call_2",
        name="enable_skill_for_agent",
        arguments='{"skill_id":"skill.data_analysis"}',
    )
    result = asyncio.run(box.dispatch(tc, scope=ToolScope(metadata={"role": "admin"})))
    assert not result.ok
    assert "not ready" in result.error


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


def test_install_finds_and_reads_skill_md_only_package(tmp_path):
    """A community skill shipped as just SKILL.md (no skill.yaml) must be
    located by _find_skill_dir and read by _read_skill_package."""
    wrapper = tmp_path / "archive-root"
    pkg = wrapper / "skill-creator"
    pkg.mkdir(parents=True)
    (pkg / "SKILL.md").write_text(
        "---\n"
        "name: skill-creator\n"
        "description: Create new skills.\n"
        "metadata:\n  version: \"1.4.0\"\n"
        "---\n\n# Skill Creator\n\nDraft, eval, iterate.\n",
        encoding="utf-8",
    )

    found = _find_skill_dir(wrapper)
    assert found == pkg
    package = _read_skill_package(found)
    assert package.capability_id == "skill.skill-creator"
    assert package.name == "skill-creator"
    assert package.version == "1.4.0"
    assert "Draft, eval, iterate." in package.instructions
    # No skill.yaml -> runtime block absent, but no crash.
    deps = _dependency_summary(pkg)
    assert deps["runtime"] == {}
    assert deps["has_dependencies"] is False


def test_install_rejects_dir_with_no_manifest(tmp_path):
    bad = tmp_path / "not-a-skill"
    bad.mkdir()
    (bad / "README.md").write_text("nothing here", encoding="utf-8")
    try:
        _find_skill_dir(tmp_path)
    except ValueError as exc:
        assert "skill.yaml or SKILL.md" in str(exc)
    else:
        raise AssertionError("expected missing-manifest error")


# --- Progressive-disclosure skill tools: load_skill + read_skill_resource ---

def _make_skill(tmp_path, skill_id="architecture-diagram"):
    skill_dir = tmp_path / skill_id
    (skill_dir / "resources").mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {skill_id}\ndescription: Draw diagrams.\n---\n\n"
        "Copy the template at resources/template.html.\n",
        encoding="utf-8",
    )
    (skill_dir / "resources" / "template.html").write_text(
        "<html>TEMPLATE</html>", encoding="utf-8"
    )
    return skill_dir


def _scope_with_skill(skill_dir, skill_id="architecture-diagram"):
    return ToolScope(
        user_id="u1",
        agent_id="main",
        skill_mounts=[{
            "id": f"skill.{skill_id}",
            "version": "0.0.0",
            "source_path": str(skill_dir),
            "mount_path": f"/mnt/skills/{skill_id}",
        }],
    )


def test_load_skill_returns_full_instructions_and_file_manifest(tmp_path):
    skill_dir = _make_skill(tmp_path)
    tool = make_load_skill_tool()
    token = set_current_tool_scope(_scope_with_skill(skill_dir))
    try:
        out = asyncio.run(tool.fn(skill_id="skill.architecture-diagram"))
    finally:
        reset_current_tool_scope(token)
    assert "# Skill: architecture-diagram" in out
    assert "Copy the template at resources/template.html." in out
    assert "resources/template.html" in out  # bundled-file manifest


def test_load_skill_accepts_id_without_prefix(tmp_path):
    skill_dir = _make_skill(tmp_path)
    tool = make_load_skill_tool()
    token = set_current_tool_scope(_scope_with_skill(skill_dir))
    try:
        out = asyncio.run(tool.fn(skill_id="architecture-diagram"))
    finally:
        reset_current_tool_scope(token)
    assert "# Skill: architecture-diagram" in out


def test_load_skill_rejects_skill_not_enabled_for_agent(tmp_path):
    skill_dir = _make_skill(tmp_path)
    tool = make_load_skill_tool()
    token = set_current_tool_scope(_scope_with_skill(skill_dir))
    try:
        out = asyncio.run(tool.fn(skill_id="skill.other"))
    finally:
        reset_current_tool_scope(token)
    assert "not an available skill" in out
    assert "skill.architecture-diagram" in out  # lists what IS available


def test_read_skill_resource_reads_bundled_file(tmp_path):
    skill_dir = _make_skill(tmp_path)
    tool = make_read_skill_resource_tool()
    token = set_current_tool_scope(_scope_with_skill(skill_dir))
    try:
        out = asyncio.run(tool.fn(
            skill_id="skill.architecture-diagram", path="resources/template.html"
        ))
    finally:
        reset_current_tool_scope(token)
    assert out == "<html>TEMPLATE</html>"


def test_read_skill_resource_blocks_path_traversal(tmp_path):
    skill_dir = _make_skill(tmp_path)
    # A secret sibling file outside the skill dir.
    (tmp_path / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
    tool = make_read_skill_resource_tool()
    token = set_current_tool_scope(_scope_with_skill(skill_dir))
    try:
        out = asyncio.run(tool.fn(
            skill_id="skill.architecture-diagram", path="../secret.txt"
        ))
    finally:
        reset_current_tool_scope(token)
    assert "escapes the skill directory" in out
    assert "TOP SECRET" not in out


def test_read_skill_resource_rejects_skill_not_enabled(tmp_path):
    skill_dir = _make_skill(tmp_path)
    tool = make_read_skill_resource_tool()
    token = set_current_tool_scope(_scope_with_skill(skill_dir))
    try:
        out = asyncio.run(tool.fn(skill_id="skill.other", path="resources/template.html"))
    finally:
        reset_current_tool_scope(token)
    assert "not an available skill" in out


def test_read_skill_resource_missing_file(tmp_path):
    skill_dir = _make_skill(tmp_path)
    tool = make_read_skill_resource_tool()
    token = set_current_tool_scope(_scope_with_skill(skill_dir))
    try:
        out = asyncio.run(tool.fn(
            skill_id="skill.architecture-diagram", path="resources/nope.txt"
        ))
    finally:
        reset_current_tool_scope(token)
    assert "is not a file" in out


def test_skill_tools_registered_when_skills_configured(tmp_path):
    doc = AgentConfigDocument(**{"skills": {"root": str(tmp_path)}})
    reg = build_default_registry(
        type("S", (), {"search_provider": "none"})(), agent_config=doc
    )
    assert "load_skill" in reg.names()
    assert "read_skill_resource" in reg.names()
