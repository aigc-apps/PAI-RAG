# tests/app/test_store_sql.py
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
from app.db import make_engine, create_all
from app.store.sql import SqlStore
from app.store.base import Item, StoredResponse


def _fresh_store():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    return engine


def test_sql_append_order_and_response_crud_and_resolve():
    async def run():
        engine = _fresh_store()
        await create_all(engine)
        st = SqlStore(engine)
        conv = await st.create_conversation(user_id="u1")
        await st.append_items(conv.id, [Item(type="message", role="user", content={"text": "q1"}, response_id="resp_1")])
        await st.append_items(conv.id, [Item(type="message", role="assistant", content={"text": "a1"})])
        items = await st.get_conversation_items(conv.id)
        assert [i.seq for i in items] == [0, 1] and items[0].content["text"] == "q1"
        await st.save_response(StoredResponse(id="resp_1", conversation_id=conv.id, model="m", status="completed"))
        assert (await st.get_response("resp_1")).status == "completed"
        hist = await st.resolve_history(previous_response_id="resp_1", conversation=None)
        assert any(i.content.get("text") == "q1" for i in hist)
        await st.delete_response("resp_1")
        assert await st.get_response("resp_1") is None
    asyncio.run(run())


def test_sql_conflicting_ids_raise():
    async def run():
        engine = _fresh_store()
        await create_all(engine)
        st = SqlStore(engine)
        c1 = await st.create_conversation(); c2 = await st.create_conversation()
        await st.save_response(StoredResponse(id="resp_x", conversation_id=c1.id, model="m", status="completed"))
        import pytest
        with pytest.raises(ValueError):
            await st.resolve_history(previous_response_id="resp_x", conversation=c2.id)
    asyncio.run(run())
