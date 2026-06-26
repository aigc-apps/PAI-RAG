import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.schemas import ResponsesRequest
from app.builder import build_context
from app.store.memory import InMemoryStore
from agent.tools.defaults import build_default_registry
from agent.message import ToolCall


class _Settings:
    search_provider = "none"


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
