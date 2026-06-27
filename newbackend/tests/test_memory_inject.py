import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.soul import render_context_block
from app.builder import build_context
from app.schemas import ResponsesRequest
from app.store.memory import InMemoryStore
from app.store.base import MemoryItem


def test_context_block_includes_memory_when_present():
    out = render_context_block(memories=["likes tea", "in NYC"])
    assert "# Memory" in out and "likes tea" in out and "in NYC" in out
    assert render_context_block(memories=[]) == ""


def test_build_context_injects_user_memories_into_context_block():
    async def run():
        st = InMemoryStore()
        await st.add_memory(MemoryItem(user_id="u1", text="prefers Python"))
        ctx, _ = await build_context(ResponsesRequest(model="m", input="hi", user="u1"), st)
        assert "prefers Python" in ctx.context_block
        ctx2, _ = await build_context(ResponsesRequest(model="m", input="hi"), st)
        assert "# Memory" not in ctx2.context_block
    asyncio.run(run())
