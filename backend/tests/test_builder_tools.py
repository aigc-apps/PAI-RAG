# ruff: noqa: E402
import asyncio
import os
import sys
import types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.schemas import ResponsesRequest
from app.builder import build_context
from app.store.memory import InMemoryStore
from agent.tools.defaults import build_default_registry
from agent.tools.base import Tool
from agent.tools.knowledge_bundle import (
    KNOWLEDGE_TOOL_NAMES,
    LEGACY_KNOWLEDGE_TOOL_MAP,
)
from agent.tools.registry import ToolRegistry
from agent.message import ToolCall


class _Settings:
    search_provider = "none"


def _agent_config(skills_root):
    return types.SimpleNamespace(
        skills=types.SimpleNamespace(root=skills_root), capabilities=[], providers=[]
    )


def test_build_context_wires_registry_tools_and_names_them_in_prompt():
    async def run():
        reg = build_default_registry(_Settings())
        ctx, _ = await build_context(
            ResponsesRequest(model="m", input="hi"), InMemoryStore(), registry=reg
        )
        names = {t.name for t in ctx.tools.tools}
        assert names == {"current_datetime", "web_fetch"}
        assert "current_datetime" in ctx.system_prompt
        assert "web_fetch" in ctx.system_prompt

    asyncio.run(run())


def test_knowledge_service_registers_only_complete_canonical_bundle():
    reg = build_default_registry(_Settings(), knowledge_service=object())

    names = set(reg.names())
    assert set(KNOWLEDGE_TOOL_NAMES) <= names
    assert set(LEGACY_KNOWLEDGE_TOOL_MAP).isdisjoint(names)


def test_disabled_knowledge_capability_registers_no_knowledge_tools():
    config = types.SimpleNamespace(
        skills=types.SimpleNamespace(root="/nonexistent"),
        capabilities=[
            types.SimpleNamespace(
                id="knowledge", enabled=False, permission="disabled"
            )
        ],
        providers=[],
    )

    reg = build_default_registry(
        _Settings(), agent_config=config, knowledge_service=object()
    )

    assert set(KNOWLEDGE_TOOL_NAMES).isdisjoint(reg.names())


def test_profile_include_filters_the_toolbox():
    async def run():
        from app.agent_config import AgentProfile, AgentToolsConfig
        reg = build_default_registry(_Settings())
        cfg = types.SimpleNamespace(
            agents=[AgentProfile(id="main", name="Main",
                                 tools=AgentToolsConfig(include=["current_datetime"]))],
            default_agent="main",
            skills=types.SimpleNamespace(root="/nonexistent"),
            capabilities=[], providers=[],
        )
        req = ResponsesRequest(model="m", input="hi")
        ctx, _ = await build_context(req, InMemoryStore(), registry=reg, agent_config=cfg)
        assert [t.name for t in ctx.tools.tools] == ["current_datetime"]
        assert "web_fetch" not in ctx.system_prompt

    asyncio.run(run())


def test_partial_knowledge_include_selects_complete_bundle():
    async def noop():
        return "ok"

    registry = ToolRegistry()
    for name in KNOWLEDGE_TOOL_NAMES:
        registry.register(
            Tool(
                name=name,
                description=name,
                parameters={"type": "object", "properties": {}},
                fn=noop,
            )
        )

    from app.agent_config import AgentProfile, AgentToolsConfig
    from app.builder import _select_tool_names

    profile = AgentProfile(
        id="main",
        name="Main",
        tools=AgentToolsConfig(include=["knowledge_search"]),
    )
    toolbox = registry.build_toolbox(_select_tool_names(registry, profile))

    assert [tool.name for tool in toolbox.tools] == list(KNOWLEDGE_TOOL_NAMES)


def test_no_registry_means_no_tools():
    async def run():
        ctx, _ = await build_context(
            ResponsesRequest(model="m", input="hi"), InMemoryStore()
        )
        assert ctx.tools.tools == []
        assert "no tools" in ctx.system_prompt.lower()

    asyncio.run(run())


def test_load_skill_not_registered_when_no_skill_packages(tmp_path):
    # skills.root is configured but the dir holds no packages. load_skill must
    # NOT be registered — otherwise the model sees the tool with an empty catalog
    # and hallucinates a skill id (e.g. "skill.frontend-design") to call.
    reg = build_default_registry(_Settings(), agent_config=_agent_config(str(tmp_path)))
    assert "load_skill" not in reg.names()
    assert "read_skill_resource" not in reg.names()


def test_load_skill_registered_when_a_skill_package_exists(tmp_path):
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: Demo Skill\ndescription: does demo things\n---\nbody\n",
        encoding="utf-8",
    )
    reg = build_default_registry(_Settings(), agent_config=_agent_config(str(tmp_path)))
    assert "load_skill" in reg.names()
    assert "read_skill_resource" not in reg.names()


def test_skill_loader_survives_include_whitelist_when_skills_active(tmp_path):
    # Regression: an agent that uses an include-whitelist (only its domain tools)
    # AND has an enabled skill must still get load_skill — the injected catalog tells
    # the model to call it, so filtering it out strands the Agent
    # ("load_skill 工具在当前会话中不可用"). The two halves of progressive disclosure
    # (catalog + loaders) must stay coupled.
    from app.agent_config import AgentProfile, AgentToolsConfig, AgentSkillsConfig
    from agent.custom_skills import discover_skill_packages, skill_sources

    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: Demo Skill\ndescription: does demo things\n---\nbody\n", encoding="utf-8"
    )
    sid = discover_skill_packages(
        skill_sources(types.SimpleNamespace(root=str(tmp_path)))
    )[0].capability_id

    async def run():
        reg = build_default_registry(_Settings(), agent_config=_agent_config(str(tmp_path)))
        assert "load_skill" in reg.names()  # precondition: registered because a package exists
        cfg = types.SimpleNamespace(
            agents=[AgentProfile(
                id="main", name="Main",
                tools=AgentToolsConfig(include=["current_datetime"]),  # whitelist omits the loaders
                skills=AgentSkillsConfig(enabled=[sid]),
            )],
            default_agent="main",
            capabilities=[types.SimpleNamespace(id=sid, kind="skill", enabled=True, status="ready")],
            providers=[],
            skills=types.SimpleNamespace(root=str(tmp_path)),
        )
        ctx, _ = await build_context(
            ResponsesRequest(model="m", input="hi", agent_id="main"),
            InMemoryStore(), registry=reg, agent_config=cfg,
        )
        names = {t.name for t in ctx.tools.tools}
        assert "current_datetime" in names          # the whitelisted domain tool
        assert "load_skill" in names                 # forced in: a skill is active
        assert "read_skill_resource" not in names
        assert "web_fetch" not in names              # whitelist still excludes everything else
        assert "Demo Skill" in ctx.context_block     # catalog injected → coupling holds

    asyncio.run(run())


def test_skill_loaders_absent_when_no_skill_enabled(tmp_path):
    # Control: a package exists (loaders registered) but the agent enables no skill →
    # no catalog, so the loaders stay out of an include-whitelist toolbox.
    from app.agent_config import AgentProfile, AgentToolsConfig, AgentSkillsConfig

    skill_dir = tmp_path / "demo"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: Demo Skill\ndescription: does demo things\n---\nbody\n", encoding="utf-8"
    )

    async def run():
        reg = build_default_registry(_Settings(), agent_config=_agent_config(str(tmp_path)))
        cfg = types.SimpleNamespace(
            agents=[AgentProfile(
                id="main", name="Main",
                tools=AgentToolsConfig(include=["current_datetime"]),
                skills=AgentSkillsConfig(enabled=[]),  # nothing enabled
            )],
            default_agent="main", capabilities=[], providers=[],
            skills=types.SimpleNamespace(root=str(tmp_path)),
        )
        ctx, _ = await build_context(
            ResponsesRequest(model="m", input="hi", agent_id="main"),
            InMemoryStore(), registry=reg, agent_config=cfg,
        )
        names = {t.name for t in ctx.tools.tools}
        assert names == {"current_datetime"}

    asyncio.run(run())


def test_wired_tool_is_dispatchable():
    async def run():
        reg = build_default_registry(_Settings())
        ctx, _ = await build_context(
            ResponsesRequest(model="m", input="hi"), InMemoryStore(), registry=reg
        )
        result = await ctx.tools.dispatch(
            ToolCall(id="c1", name="current_datetime", arguments="{}")
        )
        assert result.ok and isinstance(result.content, str) and len(result.content) >= 8

    asyncio.run(run())
