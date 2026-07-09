import sys, os, asyncio, types
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.schemas import ResponsesRequest
from app.builder import build_context
from app.store.memory import InMemoryStore
from agent.tools.defaults import build_default_registry
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


def test_soul_tools_enabled_filters_the_toolbox():
    async def run():
        reg = build_default_registry(_Settings())
        req = ResponsesRequest(
            model="m", input="hi", soul={"tools_enabled": ["current_datetime"]}
        )
        ctx, _ = await build_context(req, InMemoryStore(), registry=reg)
        assert [t.name for t in ctx.tools.tools] == ["current_datetime"]
        assert "web_fetch" not in ctx.system_prompt

    asyncio.run(run())


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
    assert "read_skill_resource" in reg.names()


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
