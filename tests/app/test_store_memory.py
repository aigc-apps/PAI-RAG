# tests/app/test_store_memory.py
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.store.memory import InMemoryStore
from app.store.base import Item, StoredResponse


def _store(): return InMemoryStore()


def test_create_conversation_and_append_items_ordered():
    async def run():
        st = _store()
        conv = await st.create_conversation(user_id="u1")
        await st.append_items(conv.id, [Item(type="message", role="user", content={"text": "q1"})])
        await st.append_items(conv.id, [Item(type="message", role="assistant", content={"text": "a1"})])
        items = await st.get_conversation_items(conv.id)
        assert [i.seq for i in items] == [0, 1]
        assert items[0].content["text"] == "q1"
    asyncio.run(run())


def test_save_get_delete_response():
    async def run():
        st = _store()
        conv = await st.create_conversation()
        r = await st.save_response(StoredResponse(id="resp_1", conversation_id=conv.id,
                                                  model="m", status="completed"))
        assert (await st.get_response("resp_1")).id == "resp_1"
        await st.delete_response("resp_1")
        assert await st.get_response("resp_1") is None
    asyncio.run(run())


def test_resolve_history_by_previous_response_id():
    async def run():
        st = _store()
        conv = await st.create_conversation()
        await st.append_items(conv.id, [Item(type="message", role="user", content={"text": "q1"}, response_id="resp_1")])
        await st.save_response(StoredResponse(id="resp_1", conversation_id=conv.id, model="m", status="completed"))
        hist = await st.resolve_history(previous_response_id="resp_1", conversation=None)
        assert any(i.content.get("text") == "q1" for i in hist)
    asyncio.run(run())


def test_resolve_history_conflicting_ids_raises():
    async def run():
        st = _store()
        c1 = await st.create_conversation(); c2 = await st.create_conversation()
        await st.save_response(StoredResponse(id="resp_x", conversation_id=c1.id, model="m", status="completed"))
        import pytest
        with pytest.raises(ValueError):
            await st.resolve_history(previous_response_id="resp_x", conversation=c2.id)
    asyncio.run(run())
