from app.agent_config import AgentConfigDocument
from app.builder import _apply_sandbox_template, _resolve_agent_template


def _doc() -> AgentConfigDocument:
    return AgentConfigDocument(**{
        "default_agent": "main",
        "agents": [
            {"id": "main", "name": "Main"},
            {"id": "turbox-helper", "name": "Turbo-X", "sandbox": {"template": "turbox"}},
        ],
        "providers": [{
            "id": "sandbox.default",
            "type": "sandbox",
            "name": "AgentRun REST",
            "settings": {
                "provider": "agentrun_rest",
                "templates": {
                    "pairec": {"name": "sandbox-code-feiyue"},
                    "turbox": {"name": "sandbox-turbox-feiyue", "code_writable": True},
                },
                "default_template": "pairec",
            },
        }],
    })


def test_blank_agent_template_resolves_to_default_and_read_only():
    doc = _doc()
    profile = next(a for a in doc.agents if a.id == "main")
    key, writable = _resolve_agent_template(doc, profile)
    assert key == ""          # metadata stays blank; the provider applies its default
    assert writable is False  # ...but the prompt must match the default template


def test_bound_agent_template_resolves_to_writable():
    doc = _doc()
    profile = next(a for a in doc.agents if a.id == "turbox-helper")
    key, writable = _resolve_agent_template(doc, profile)
    assert key == "turbox"
    assert writable is True


def test_unknown_template_is_not_writable():
    doc = _doc()
    profile = next(a for a in doc.agents if a.id == "turbox-helper")
    profile.sandbox.template = "nope"
    key, writable = _resolve_agent_template(doc, profile)
    assert key == "nope"      # passed through; the provider raises on create
    assert writable is False


def test_missing_provider_resolves_to_read_only():
    doc = AgentConfigDocument(**{"agents": [{"id": "main", "name": "Main"}]})
    profile = doc.agents[0]
    assert _resolve_agent_template(doc, profile) == ("", False)


def test_apply_sets_the_childs_own_template_over_the_parents():
    metadata = {"sandbox_template": "pairec", "subagent_depth": 0}
    _apply_sandbox_template(metadata, "turbox")
    assert metadata["sandbox_template"] == "turbox"


def test_apply_clears_an_inherited_template_when_the_child_has_none():
    """A subagent's metadata starts as a copy of the parent's scope. Without the
    pop, an unbound child silently runs on the parent's image."""
    metadata = {"sandbox_template": "pairec", "subagent_depth": 0}
    _apply_sandbox_template(metadata, "")
    assert "sandbox_template" not in metadata
    assert metadata["subagent_depth"] == 0  # unrelated keys survive


def test_apply_is_a_noop_when_there_is_nothing_to_inherit_or_set():
    metadata = {"subagent_depth": 0}
    _apply_sandbox_template(metadata, "")
    assert metadata == {"subagent_depth": 0}
