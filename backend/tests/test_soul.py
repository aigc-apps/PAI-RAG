import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.soul import (  # noqa: E402
    DEFAULT_INSTRUCTIONS,
    render_context_block,
    render_stable_system_prompt,
)
from agent.tools.knowledge_bundle import (  # noqa: E402
    KNOWLEDGE_TOOL_NAMES,
    LEGACY_KNOWLEDGE_TOOL_MAP,
)


# A sample author-written persona, used across the tool-gating cases below. The
# `instructions` markdown IS the base system prompt — it appears verbatim.
PERSONA = "# Ada\nYou are Ada, a meticulous research assistant.\n\n## Voice\nTerse."


def test_default_instructions_is_nonempty_markdown():
    assert DEFAULT_INSTRUCTIONS.strip()
    # A persona, not a tool section — the engine layer is appended separately.
    assert "# Tools" not in DEFAULT_INSTRUCTIONS


def test_stable_prompt_uses_instructions_verbatim_and_appends_tools():
    out = render_stable_system_prompt(PERSONA, tool_names=["web_fetch"])
    assert PERSONA in out
    assert "web_fetch" in out and "# Tools" in out
    # Volatile content is never in the stable layer.
    assert "# Memory" not in out and "# Additional instructions" not in out


def test_blank_instructions_falls_back_to_default():
    for blank in ("", "   ", "\n\t"):
        out = render_stable_system_prompt(blank, tool_names=[])
        assert DEFAULT_INSTRUCTIONS.strip() in out


def test_tool_protocol_carries_execution_bias():
    # Persistence + recover-from-weak-results + verify-before-finalize live in the
    # always-on engine layer, NOT the author's markdown, so a custom persona can't
    # drop them. Present even in a no-tools session.
    for tools in (["web_fetch"], []):
        out = render_stable_system_prompt(PERSONA, tool_names=tools)
        assert "keep going until the request is fully handled" in out
        assert "try another angle" in out  # recover from weak/empty tool results
        assert "check that what you produced actually answers" in out  # verify


def test_publish_artifact_guidance_only_when_tool_enabled():
    without = render_stable_system_prompt(PERSONA, tool_names=["web_fetch"])
    assert "publish_artifact" not in without
    with_tool = render_stable_system_prompt(
        PERSONA, tool_names=["web_fetch", "publish_artifact"]
    )
    assert "publish_artifact" in with_tool and "/mnt/user" in with_tool


def test_web_search_guidance_only_when_tool_enabled():
    without = render_stable_system_prompt(PERSONA, tool_names=["web_fetch"])
    assert "Web search is available" not in without
    with_tool = render_stable_system_prompt(PERSONA, tool_names=["web_search", "web_fetch"])
    assert "Web search is available" in with_tool
    assert "web_fetch on a specific result URL" in with_tool


def test_aliyun_cli_guidance_only_when_capability_and_shell_present():
    # Gated on both the capability flag and a sandbox shell tool.
    assert "aliyun" not in render_stable_system_prompt(
        PERSONA, tool_names=["shell"]).lower()
    assert "aliyun" not in render_stable_system_prompt(
        PERSONA, tool_names=["web_fetch"], aliyun_pai_enabled=True).lower()
    on = render_stable_system_prompt(
        PERSONA, tool_names=["shell"], aliyun_pai_enabled=True)
    assert "aliyun" in on.lower()
    # Steers the model away from self-configuring the CLI.
    assert "aliyun configure" in on
    # Tells the model not to discard stderr, so auth errors stay diagnosable.
    assert "2>/dev/null" in on


def test_knowledge_guidance_only_when_knowledge_search_present():
    without = render_stable_system_prompt(PERSONA, tool_names=["web_fetch"])
    assert "knowledge base" not in without.lower()
    on = render_stable_system_prompt(PERSONA, tool_names=["knowledge_search"])
    # Reflex ("ground your answer") without naming unavailable sibling tools.
    assert "ground your answer" in on
    assert all(name not in on for name in KNOWLEDGE_TOOL_NAMES[1:])

    with_aux = render_stable_system_prompt(
        PERSONA, tool_names=list(KNOWLEDGE_TOOL_NAMES)
    )
    assert all(name in with_aux for name in KNOWLEDGE_TOOL_NAMES)
    assert all(
        legacy not in with_aux
        for legacy in LEGACY_KNOWLEDGE_TOOL_MAP
    )
    assert "incomplete" in with_aux
    assert "document_id" in with_aux and "chunk_id" in with_aux
    assert "[n]" in with_aux
    assert "never expose" in with_aux


def test_knowledge_guidance_defines_concrete_search_and_skip_categories():
    out = render_stable_system_prompt(PERSONA, tool_names=["knowledge_search"])

    for category in (
        "product behavior",
        "configuration",
        "procedures",
        "APIs",
        "error messages or codes",
        "troubleshooting",
        "policies",
    ):
        assert category in out
    for skip in (
        "greetings",
        "identity questions",
        "casual conversation",
        "writing or translation tasks",
    ):
        assert skip in out
    assert "call knowledge_search first" in out
    assert "product or service names" in out
    assert "TurboX license_check 失败" in out


def test_sandbox_guidance_when_code_interpreter_or_shell_present():
    assert "runs real code" not in render_stable_system_prompt(
        PERSONA, tool_names=["web_fetch"])
    for tools in (["code_interpreter"], ["shell"]):
        out = render_stable_system_prompt(PERSONA, tool_names=tools)
        assert "runs real code" in out and "/mnt/user" in out


def test_code_guidance_gated_on_agent_setting_and_sandbox_tool():
    # Off by default even with a sandbox tool present.
    assert "/opt/code" not in render_stable_system_prompt(
        PERSONA, tool_names=["shell"])
    # Enabled flag but no sandbox tool to explore with -> still off.
    assert "/opt/code" not in render_stable_system_prompt(
        PERSONA, tool_names=["web_fetch"], code_enabled=True)
    # Flag + sandbox tool -> the fallback-to-code guidance appears.
    for tools in (["shell"], ["code_interpreter"]):
        out = render_stable_system_prompt(
            PERSONA, tool_names=tools, code_enabled=True)
        assert 'ls /opt/code' in out


def test_code_manifest_injected_when_present_and_layer_enabled():
    manifest = "- repo-a — the API server\n- repo-b — the ingest worker"
    # Manifest verbatim in the prompt, and still points at ls for the uncovered case.
    out = render_stable_system_prompt(
        PERSONA, tool_names=["shell"], code_enabled=True,
        code_manifest=manifest)
    assert manifest in out
    assert '/opt/code' in out
    # Empty manifest -> falls back to the pure discover-by-ls guidance (no leftover
    # "available repositories" header, but /opt/code still mentioned).
    empty = render_stable_system_prompt(
        PERSONA, tool_names=["shell"], code_enabled=True,
        code_manifest="")
    assert "repo-a" not in empty and "/opt/code" in empty
    # Layer off -> a manifest is never advertised.
    off = render_stable_system_prompt(
        PERSONA, tool_names=["shell"], code_manifest=manifest)
    assert manifest not in off and "/opt/code" not in off


def test_stable_prompt_lists_no_tools_when_empty_and_project_when_set():
    assert "no tools" in render_stable_system_prompt(PERSONA, tool_names=[]).lower()
    out = render_stable_system_prompt(PERSONA, tool_names=[], project_context="Repo: PAI-Loop")
    assert "# Project context" in out and "Repo: PAI-Loop" in out
    assert "# Project context" not in render_stable_system_prompt(PERSONA, tool_names=[])


def test_context_block_renders_memory_summary_instructions_and_empty():
    assert render_context_block() == ""
    out = render_context_block(memories=["likes tea"], summary="talked about X",
                               instructions="be terse")
    assert "# Memory" in out and "likes tea" in out
    assert "# Conversation summary" in out and "talked about X" in out
    assert "# Additional instructions" in out and "be terse" in out
