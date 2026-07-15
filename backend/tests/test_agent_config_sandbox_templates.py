import pytest

from app.agent_config import (
    DEFAULT_DOCUMENT,
    AgentConfigDocument,
    AgentProfile,
    apply_runtime_status,
    validate_sandbox_bindings,
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


def _settings():
    return type("Settings", (), {
        "openai_api_key": "",
        "default_model": "m",
        "search_provider": "none",
        "search_api_key": "",
        "search_endpoint": "",
    })()


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
    doc = apply_runtime_status(doc, _settings(), None)
    provider = next(p for p in doc.providers if p.id == "sandbox.default")
    assert provider.status == "healthy"


def test_runtime_status_missing_config_with_empty_templates():
    doc = _doc_with({"templates": {}, "api_key": "secret", "account_id": "acct-1"})
    doc = apply_runtime_status(doc, _settings(), None)
    provider = next(p for p in doc.providers if p.id == "sandbox.default")
    assert provider.status == "missing_config"


def _doc_with_agent(*, template_key: str, templates: dict, default_template: str = "",
                     code_enabled: bool = True, manifest: str = "") -> AgentConfigDocument:
    return AgentConfigDocument(**{
        "agents": [{
            "id": "turbox-helper",
            "name": "Turbo-X",
            "code": {"enabled": code_enabled, "manifest": manifest},
            "sandbox": {"template": template_key},
        }],
        "providers": [{
            "id": "sandbox.default",
            "type": "sandbox",
            "name": "AgentRun REST",
            "settings": {
                "provider": "agentrun_rest",
                "templates": templates,
                "default_template": default_template,
            },
            "used_by": ["sandbox"],
        }],
    })


def test_validate_sandbox_bindings_rejects_writable_template_with_blank_manifest():
    doc = _doc_with_agent(
        template_key="turbox",
        templates={"turbox": {"name": "turbox image", "code_writable": True}},
        code_enabled=True,
        manifest="",
    )
    with pytest.raises(ValueError, match="turbox-helper"):
        validate_sandbox_bindings(doc)


def test_validate_sandbox_bindings_allows_writable_template_with_manifest():
    doc = _doc_with_agent(
        template_key="turbox",
        templates={"turbox": {"name": "turbox image", "code_writable": True}},
        code_enabled=True,
        manifest="- repo-a — the API server",
    )
    validate_sandbox_bindings(doc)  # no raise


def test_validate_sandbox_bindings_allows_writable_template_when_code_disabled():
    doc = _doc_with_agent(
        template_key="turbox",
        templates={"turbox": {"name": "turbox image", "code_writable": True}},
        code_enabled=False,
        manifest="",
    )
    validate_sandbox_bindings(doc)  # no raise: guidance block is never rendered


def test_validate_sandbox_bindings_allows_read_only_template_with_blank_manifest():
    doc = _doc_with_agent(
        template_key="pairec",
        templates={"pairec": {"name": "pairec image", "code_writable": False}},
        code_enabled=True,
        manifest="",
    )
    validate_sandbox_bindings(doc)  # no raise


def test_validate_sandbox_bindings_rejects_via_default_template_resolution():
    # Blank agent.sandbox.template resolves through settings.default_template.
    doc = _doc_with_agent(
        template_key="",
        templates={"turbox": {"name": "turbox image", "code_writable": True}},
        default_template="turbox",
        code_enabled=True,
        manifest="",
    )
    with pytest.raises(ValueError, match="turbox-helper"):
        validate_sandbox_bindings(doc)


def test_validate_sandbox_bindings_ignores_unknown_template_key():
    # An unresolvable/unknown template key is the sandbox provider's job to
    # reject at create time, not this validator's.
    doc = _doc_with_agent(
        template_key="does-not-exist",
        templates={"turbox": {"name": "turbox image", "code_writable": True}},
        code_enabled=True,
        manifest="",
    )
    validate_sandbox_bindings(doc)  # no raise


def test_validate_sandbox_bindings_rejects_blank_default_template_when_agent_would_resolve_through_it():
    # templates is non-empty, default_template is blank, and the agent's own
    # sandbox.template is blank too -- it would resolve through
    # default_template, which resolves to nothing. Every sandbox create for
    # this agent fails even though the runtime status grading (templates
    # non-empty) reports healthy.
    doc = _doc_with_agent(
        template_key="",
        templates={"turbox": {"name": "turbox image", "code_writable": True}},
        default_template="",
        code_enabled=False,
        manifest="",
    )
    with pytest.raises(ValueError, match="default_template"):
        validate_sandbox_bindings(doc)


def test_validate_sandbox_bindings_allows_blank_default_template_when_every_agent_is_explicitly_bound():
    # Same blank default_template, but the only agent binds its own
    # sandbox.template explicitly -- nothing ever resolves through the blank
    # default, so it is not a live misconfiguration.
    doc = _doc_with_agent(
        template_key="turbox",
        templates={"turbox": {"name": "turbox image", "code_writable": True}},
        default_template="",
        code_enabled=False,
        manifest="",
    )
    validate_sandbox_bindings(doc)  # no raise


def test_validate_sandbox_bindings_rejects_default_template_naming_absent_key():
    doc = _doc_with_agent(
        template_key="turbox",
        templates={"turbox": {"name": "turbox image", "code_writable": True}},
        default_template="does-not-exist",
        code_enabled=False,
        manifest="",
    )
    with pytest.raises(ValueError, match="does-not-exist"):
        validate_sandbox_bindings(doc)
