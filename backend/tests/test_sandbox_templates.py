import asyncio
import base64
import datetime
import shlex
import time

import pytest

from agent.tools.sandbox_providers import (
    AgentRunRestSandboxProvider,
    _SandboxSession,
    _template_env_refs,
    make_sandbox_provider,
)
from agent.tools.scope import ToolScope, reset_current_tool_scope, set_current_tool_scope
from app.agent_config import AgentConfigDocument


def _provider() -> AgentRunRestSandboxProvider:
    return AgentRunRestSandboxProvider({
        "api_key": "secret",
        "account_id": "acct-1",
        "region": "cn-hangzhou",
        "templates": {
            "pairec": {"name": "sandbox-code-feiyue"},
            "turbox": {
                "name": "sandbox-turbox-feiyue",
                "code_writable": True,
                "env_refs": {"GITLAB_TOKEN": "GITLAB_TOKEN"},
            },
        },
        "default_template": "pairec",
    })


def _with_scope(template: str):
    return set_current_tool_scope(
        ToolScope(agent_id="a1", metadata={"sandbox_template": template})
    )


def test_resolve_uses_scope_template():
    token = _with_scope("turbox")
    try:
        key, tpl = _provider()._resolve_template()
    finally:
        reset_current_tool_scope(token)
    assert key == "turbox"
    assert tpl["name"] == "sandbox-turbox-feiyue"


def test_resolve_falls_back_to_default_when_scope_is_blank():
    token = _with_scope("")
    try:
        key, tpl = _provider()._resolve_template()
    finally:
        reset_current_tool_scope(token)
    assert key == "pairec"
    assert tpl["name"] == "sandbox-code-feiyue"


def test_resolve_unknown_key_raises_and_lists_valid_keys():
    token = _with_scope("nope")
    try:
        with pytest.raises(RuntimeError) as exc:
            _provider()._resolve_template()
    finally:
        reset_current_tool_scope(token)
    message = str(exc.value)
    assert "'nope'" in message
    assert "pairec" in message and "turbox" in message


def test_env_refs_resolve_from_environment(monkeypatch):
    monkeypatch.setenv("GITLAB_TOKEN", "glpat-xyz")
    tpl = _provider().templates["turbox"]
    assert _template_env_refs(tpl) == {"GITLAB_TOKEN": "glpat-xyz"}


def test_pairec_template_carries_no_env_refs(monkeypatch):
    monkeypatch.setenv("GITLAB_TOKEN", "glpat-xyz")
    tpl = _provider().templates["pairec"]
    assert _template_env_refs(tpl) == {}


def test_unset_env_ref_is_skipped_with_a_warning(monkeypatch):
    """Unset source var: skipped, never raises. The provider logs through loguru,
    which does not propagate to pytest's caplog, so capture the sink directly
    rather than asserting on caplog.records."""
    from loguru import logger

    monkeypatch.delenv("GITLAB_TOKEN", raising=False)
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(str(m)), format="{message}", level="WARNING")
    try:
        assert _template_env_refs(_provider().templates["turbox"]) == {}
    finally:
        logger.remove(sink_id)
    assert any("GITLAB_TOKEN" in m for m in messages)


def _bootstrap_source(command: str) -> str:
    """Decode the base64 Python payload that `_bootstrap_env_async` sends to
    `processes/cmd` (built by `_env_bootstrap_command`). This is the *real*
    env delivery path -- AgentRun's CreateSandbox has no `envs` field, so the
    contract only reaches the sandbox via this command writing ~/.bash_env.
    The command has the shape `printf %s <base64blob> | base64 -d | python3 -`
    with no shell-unsafe characters in the base64 alphabet, so a plain
    whitespace split (via shlex, to stay robust to quoting) picks out the
    blob at index 2."""
    blob = shlex.split(command)[2]
    return base64.b64decode(blob).decode("utf-8")


def test_create_sandbox_async_delivers_env_ref_to_payload_and_bootstrap(monkeypatch):
    """End-to-end regression for the env_refs security wiring.

    A template's env_refs must reach BOTH the create payload's `envs` (kept
    only for forward-compat -- AgentRun ignores it) AND the env_contract
    handed to `_bootstrap_env_async`, which is what actually delivers the
    token into the sandbox (via ~/.bash_env). The isolated unit tests for
    `_resolve_template`/`_template_env_refs` cannot catch a regression where
    a future change silently drops env_refs between resolution and either of
    these two hand-off points -- e.g. the payload line already went through
    `env_contract` -> `or {}` -> `... or None` once. This test drives the
    real `_create_sandbox_async` with only the HTTP seam (`_request_async`)
    stubbed, so both hand-offs are pinned against regression.
    """
    monkeypatch.setenv("GITLAB_TOKEN", "glpat-e2e-secret")
    provider = _provider()
    calls: list[dict] = []

    async def fake_request_async(method, path, *, json=None, sensitive=False):
        calls.append({"method": method, "path": path, "json": json})
        if path == provider.create_path:
            return {"data": {"sandboxId": "sb-e2e-1"}}
        return {}

    monkeypatch.setattr(provider, "_request_async", fake_request_async)

    token = _with_scope("turbox")
    try:
        sandbox_id = asyncio.run(provider._create_sandbox_async("scope-e2e-1"))
    finally:
        reset_current_tool_scope(token)

    assert sandbox_id == "sb-e2e-1"
    create_calls = [c for c in calls if c["path"] == provider.create_path]
    bootstrap_calls = [c for c in calls if c["path"].endswith("/processes/cmd")]
    assert len(create_calls) == 1
    assert len(bootstrap_calls) == 1

    payload = create_calls[0]["json"]
    # The resolved template's `name`, not the lookup key ("turbox").
    assert payload["templateName"] == "sandbox-turbox-feiyue"
    # Forward-compat only (AgentRun ignores this field), but it must still
    # carry the resolved env_refs value.
    assert payload["envs"]["GITLAB_TOKEN"] == "glpat-e2e-secret"

    # The real delivery path: the env_contract handed to _bootstrap_env_async,
    # which writes it into ~/.bash_env inside the freshly created sandbox.
    bootstrap_source = _bootstrap_source(bootstrap_calls[0]["json"]["command"])
    assert "GITLAB_TOKEN" in bootstrap_source
    assert "glpat-e2e-secret" in bootstrap_source


def test_create_sandbox_async_pairec_template_never_carries_gitlab_token(monkeypatch):
    """Isolation guarantee, the flip side of the test above: a template with
    no env_refs (pairec, the public-network template) must never carry
    GITLAB_TOKEN in either the create payload or the env_contract delivered
    to _bootstrap_env_async -- even though GITLAB_TOKEN IS set in the
    environment for this test. That last part is what proves the *template*
    gates the token rather than the token merely being absent: a
    public-network sandbox must never carry an intranet gitlab token.
    """
    monkeypatch.setenv("GITLAB_TOKEN", "glpat-should-not-leak")
    provider = _provider()
    calls: list[dict] = []

    async def fake_request_async(method, path, *, json=None, sensitive=False):
        calls.append({"method": method, "path": path, "json": json})
        if path == provider.create_path:
            return {"data": {"sandboxId": "sb-e2e-2"}}
        return {}

    monkeypatch.setattr(provider, "_request_async", fake_request_async)

    token = _with_scope("pairec")
    try:
        sandbox_id = asyncio.run(provider._create_sandbox_async("scope-e2e-2"))
    finally:
        reset_current_tool_scope(token)

    assert sandbox_id == "sb-e2e-2"
    create_calls = [c for c in calls if c["path"] == provider.create_path]
    bootstrap_calls = [c for c in calls if c["path"].endswith("/processes/cmd")]
    assert len(create_calls) == 1
    assert len(bootstrap_calls) == 1

    payload = create_calls[0]["json"]
    assert payload["templateName"] == "sandbox-code-feiyue"
    envs = payload.get("envs") or {}
    assert "GITLAB_TOKEN" not in envs

    bootstrap_source = _bootstrap_source(bootstrap_calls[0]["json"]["command"])
    assert "GITLAB_TOKEN" not in bootstrap_source
    assert "glpat-should-not-leak" not in bootstrap_source


def test_maybe_refresh_env_async_carries_template_env_refs(monkeypatch):
    """Regression: the STS-refresh path re-bootstraps the env contract
    independently of `_create_sandbox_async`, and `_bootstrap_env_async`'s
    injected script REPLACES (not appends to) the marker block in
    ~/.bash_env. If the refreshed contract omits `env_refs`, the refresh
    silently deletes whatever `_create_sandbox_async` originally injected --
    e.g. GITLAB_TOKEN vanishes the moment a long-running conversation's STS
    creds cross the refresh margin, and the agent's next `git fetch` gets a
    401 with nothing in the logs explaining it. This drives
    `_maybe_refresh_env_async` directly (not `_create_sandbox_async`) so it
    pins the *refresh* hand-off, not the create hand-off already covered
    above.
    """
    monkeypatch.setenv("GITLAB_TOKEN", "glpat-refresh-secret")
    provider = _provider()
    captured: list[dict] = []

    async def fake_bootstrap(sandbox_id, env_contract):
        captured.append(env_contract or {})

    monkeypatch.setattr(provider, "_bootstrap_env_async", fake_bootstrap)

    # Session already inside the refresh margin (_ENV_REFRESH_MARGIN_SECONDS
    # = 300s), so _maybe_refresh_env_async actually fires the re-inject.
    session = _SandboxSession(
        handle="sb-refresh-1",
        last_used=0.0,
        last_checked=0.0,
        env_expires_at=time.time() + 10,
    )

    expiry = (
        datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    token = set_current_tool_scope(
        ToolScope(
            agent_id="a1",
            metadata={
                "sandbox_template": "turbox",
                "aliyun_sandbox_env": {
                    "ALIBABACLOUD_ACCESS_KEY_ID": "ak",
                    "ALIBABACLOUD_ACCESS_KEY_SECRET": "sk",
                    "ALIBABACLOUD_SECURITY_TOKEN": "tok",
                    "ALIBABACLOUD_SESSION_EXPIRATION": expiry,
                },
            },
        )
    )
    try:
        asyncio.run(provider._maybe_refresh_env_async(session, "scope-refresh-1"))
    finally:
        reset_current_tool_scope(token)

    assert len(captured) == 1
    assert captured[0].get("GITLAB_TOKEN") == "glpat-refresh-secret"


def _doc_with(settings: dict) -> AgentConfigDocument:
    return AgentConfigDocument(**{
        "providers": [{
            "id": "sandbox.default",
            "type": "sandbox",
            "name": "AgentRun REST",
            "settings": {"provider": "agentrun_rest", **settings},
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


def test_make_sandbox_provider_returns_none_without_templates():
    doc = _doc_with({"templates": {}, "api_key": "secret", "account_id": "acct-1"})
    assert make_sandbox_provider(doc) is None


def test_make_sandbox_provider_builds_with_templates():
    doc = _doc_with({
        "templates": {"pairec": {"name": "sandbox-code-feiyue"}},
        "api_key": "secret",
        "account_id": "acct-1",
    })
    assert make_sandbox_provider(doc) is not None
