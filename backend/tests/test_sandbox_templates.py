import pytest

from agent.tools.sandbox_providers import (
    AgentRunRestSandboxProvider,
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
