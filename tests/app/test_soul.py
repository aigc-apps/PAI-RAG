import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from agent.soul import Soul, DEFAULT_SOUL, render_system_prompt


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


def test_render_includes_identity_personality_principles_safety():
    out = render_system_prompt(DEFAULT_SOUL, tool_names=[])
    assert DEFAULT_SOUL.name in out
    assert DEFAULT_SOUL.role in out
    assert "# Identity" in out
    assert "# Personality" in out
    assert "# Operating principles" in out
    assert "# Safety" in out
    # the first principle text shows up
    assert DEFAULT_SOUL.principles[0] in out


def test_render_lists_tools_when_present_and_says_none_when_empty():
    none_out = render_system_prompt(DEFAULT_SOUL, tool_names=[])
    assert "no tools" in none_out.lower()
    tools_out = render_system_prompt(DEFAULT_SOUL, tool_names=["web_search", "web_fetch"])
    assert "web_search" in tools_out and "web_fetch" in tools_out
    assert "# Tools" in tools_out


def test_render_includes_extra_instructions_only_when_present():
    soul = DEFAULT_SOUL.merge({"extra_instructions": "Always answer in French."})
    out = render_system_prompt(soul, tool_names=[])
    assert "Always answer in French." in out
    assert "# Additional instructions" in out
    blank = render_system_prompt(DEFAULT_SOUL, tool_names=[])
    assert "# Additional instructions" not in blank


def test_render_includes_expertise_when_set():
    soul = DEFAULT_SOUL.merge({"expertise": ["tax law", "accounting"]})
    out = render_system_prompt(soul, tool_names=[])
    assert "tax law" in out
