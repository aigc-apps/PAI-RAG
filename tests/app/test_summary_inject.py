import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.builder import build_context
from app.schemas import ResponsesRequest
from app.store.memory import InMemoryStore
from app.store.base import Item, StoredResponse


def test_build_context_injects_summary_and_trims_old_history():
    async def run():
        st = InMemoryStore()
        conv = await st.create_conversation()
        await st.append_items(conv.id, [
            Item(type="message", role="user", content={"text": "old q"}, response_id="r0"),
            Item(type="message", role="assistant", content={"text": "old a"}, response_id="r0"),
            Item(type="message", role="user", content={"text": "recent q"}, response_id="r1"),
            Item(type="message", role="assistant", content={"text": "recent a"}, response_id="r1"),
        ])  # seq 0..3
        await st.save_response(StoredResponse(id="r1", conversation_id=conv.id, model="m", status="completed"))
        await st.update_conversation_summary(conv.id, "earlier: discussed old q", 1)  # fold seq 0,1
        req = ResponsesRequest(model="m", input="next", conversation=conv.id)
        ctx, _ = await build_context(req, st)
        # summary present in the volatile block
        assert "earlier: discussed old q" in ctx.context_block
        assert "# Conversation summary" in ctx.context_block
        # only items with seq>1 remain in history (old q/old a dropped)
        hist_text = " ".join(m.content for m in ctx.history if isinstance(m.content, str))
        assert "recent q" in hist_text and "old q" not in hist_text
    asyncio.run(run())


def test_no_summary_serves_full_history():
    async def run():
        st = InMemoryStore()
        conv = await st.create_conversation()
        await st.append_items(conv.id, [
            Item(type="message", role="user", content={"text": "q1"}, response_id="r0"),
        ])
        await st.save_response(StoredResponse(id="r0", conversation_id=conv.id, model="m", status="completed"))
        ctx, _ = await build_context(ResponsesRequest(model="m", input="x", conversation=conv.id), st)
        assert "# Conversation summary" not in ctx.context_block
        assert any("q1" in (m.content or "") for m in ctx.history if isinstance(m.content, str))
    asyncio.run(run())
