import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from agent.soul import DEFAULT_SOUL, render_system_prompt
from app.builder import build_context
from app.schemas import ResponsesRequest
from app.store.memory import InMemoryStore
from app.store.base import MemoryItem


def test_render_includes_memory_section_when_present():
    out = render_system_prompt(DEFAULT_SOUL, tool_names=[], memories=["likes tea", "in NYC"])
    assert "# Memory" in out and "likes tea" in out and "in NYC" in out
    assert "# Memory" not in render_system_prompt(DEFAULT_SOUL, tool_names=[], memories=[])


def test_build_context_injects_user_memories():
    async def run():
        st = InMemoryStore()
        await st.add_memory(MemoryItem(user_id="u1", text="prefers Python"))
        req = ResponsesRequest(model="m", input="hi", user="u1")
        ctx, _ = await build_context(req, st)
        assert "prefers Python" in ctx.system_prompt
        # no user -> no memory section
        ctx2, _ = await build_context(ResponsesRequest(model="m", input="hi"), st)
        assert "# Memory" not in ctx2.system_prompt
    asyncio.run(run())
