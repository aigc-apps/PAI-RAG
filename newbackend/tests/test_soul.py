import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.soul import Soul, DEFAULT_SOUL, render_stable_system_prompt, render_context_block


def test_default_soul_has_identity_and_name():
    assert DEFAULT_SOUL.name
    assert DEFAULT_SOUL.role
    assert DEFAULT_SOUL.identity
    assert DEFAULT_SOUL.personality and DEFAULT_SOUL.principles
    assert DEFAULT_SOUL.tools_enabled is None


def test_merge_replaces_known_fields_and_ignores_none_and_unknown():
    merged = DEFAULT_SOUL.merge({"name": "Helper", "role": None, "bogus": "x"})
    assert merged.name == "Helper"
    assert merged.role == DEFAULT_SOUL.role  # None override ignored
    assert not hasattr(merged, "bogus")
    assert DEFAULT_SOUL.name != "Helper"  # original unchanged


def test_stable_prompt_has_persona_tools_safety_not_memory_or_instructions():
    out = render_stable_system_prompt(DEFAULT_SOUL, tool_names=["web_fetch"])
    assert DEFAULT_SOUL.name in out and "# Identity" in out
    assert "# Personality" in out and "# Operating principles" in out
    assert "# Safety" in out and "web_fetch" in out and "# Tools" in out
    assert "# Memory" not in out and "# Additional instructions" not in out


def test_publish_artifact_guidance_only_when_tool_enabled():
    without = render_stable_system_prompt(DEFAULT_SOUL, tool_names=["web_fetch"])
    assert "publish_artifact" not in without
    with_tool = render_stable_system_prompt(
        DEFAULT_SOUL, tool_names=["web_fetch", "publish_artifact"]
    )
    assert "publish_artifact" in with_tool and "/mnt/user" in with_tool


def test_aliyun_cli_guidance_only_when_capability_and_shell_present():
    # Gated on both the capability flag and a sandbox shell tool.
    assert "aliyun" not in render_stable_system_prompt(
        DEFAULT_SOUL, tool_names=["shell"]).lower()
    assert "aliyun" not in render_stable_system_prompt(
        DEFAULT_SOUL, tool_names=["web_fetch"], aliyun_pai_enabled=True).lower()
    on = render_stable_system_prompt(
        DEFAULT_SOUL, tool_names=["shell"], aliyun_pai_enabled=True)
    assert "aliyun" in on.lower()
    # Steers the model away from self-configuring the CLI.
    assert "aliyun configure" in on
    # Tells the model not to discard stderr, so auth errors stay diagnosable.
    assert "2>/dev/null" in on


def test_stable_prompt_lists_no_tools_when_empty_and_project_when_set():
    assert "no tools" in render_stable_system_prompt(DEFAULT_SOUL, tool_names=[]).lower()
    out = render_stable_system_prompt(DEFAULT_SOUL, tool_names=[], project_context="Repo: PAI-RAG")
    assert "# Project context" in out and "Repo: PAI-RAG" in out
    assert "# Project context" not in render_stable_system_prompt(DEFAULT_SOUL, tool_names=[])


def test_context_block_renders_memory_summary_instructions_and_empty():
    assert render_context_block() == ""
    out = render_context_block(memories=["likes tea"], summary="talked about X",
                               instructions="be terse")
    assert "# Memory" in out and "likes tea" in out
    assert "# Conversation summary" in out and "talked about X" in out
    assert "# Additional instructions" in out and "be terse" in out


def test_stable_prompt_includes_expertise_when_set():
    soul = DEFAULT_SOUL.merge({"expertise": ["tax law", "accounting"]})
    out = render_stable_system_prompt(soul, tool_names=[])
    assert "tax law" in out and "accounting" in out


def test_merge_revalidates_and_rejects_bad_types():
    import pytest
    with pytest.raises(ValueError):
        DEFAULT_SOUL.merge({"principles": 123})
    # original remains usable/unchanged
    assert isinstance(DEFAULT_SOUL.principles, list)
