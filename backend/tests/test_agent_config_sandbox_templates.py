from app.agent_config import (
    DEFAULT_DOCUMENT,
    AgentConfigDocument,
    AgentProfile,
    apply_runtime_status,
)


def test_agent_profile_defaults_to_empty_sandbox_template():
    profile = AgentProfile(id="main", name="Main")
    assert profile.sandbox.template == ""


def test_agent_profile_accepts_sandbox_template():
    profile = AgentProfile(**{
        "id": "turbox-helper",
        "name": "Turbo-X",
        "sandbox": {"template": "turbox"},
    })
    assert profile.sandbox.template == "turbox"


def test_default_document_ships_no_template_name():
    provider = next(p for p in DEFAULT_DOCUMENT.providers if p.id == "sandbox.default")
    assert "template_name" not in provider.settings
    assert provider.settings["templates"] == {}
    assert provider.settings["default_template"] == ""


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


def test_runtime_status_healthy_with_templates():
    doc = _doc_with({
        "templates": {"pairec": {"name": "sandbox-code-feiyue"}},
        "api_key": "secret",
        "account_id": "acct-1",
    })
    doc = apply_runtime_status(doc)
    provider = next(p for p in doc.providers if p.id == "sandbox.default")
    assert provider.status == "healthy"


def test_runtime_status_missing_config_with_empty_templates():
    doc = _doc_with({"templates": {}, "api_key": "secret", "account_id": "acct-1"})
    doc = apply_runtime_status(doc)
    provider = next(p for p in doc.providers if p.id == "sandbox.default")
    assert provider.status == "missing_config"
