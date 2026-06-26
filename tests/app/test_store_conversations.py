import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
import pytest
from app.store.memory import InMemoryStore
from app.store.sql import SqlStore
from app.db import make_engine, create_all
from app.store.base import Item, StoredResponse


async def _mem():
    return InMemoryStore()


async def _sql():
    engine = make_engine("sqlite+aiosqlite:///:memory:")
    await create_all(engine)
    return SqlStore(engine)


# Run every test against both store backends.
STORES = [_mem, _sql]


def _run(coro_factory):
    asyncio.run(coro_factory())


@pytest.mark.parametrize("make_store", STORES)
def test_ensure_conversation_is_idempotent_and_keeps_title(make_store):
    async def run():
        st = await make_store()
        c1 = await st.ensure_conversation("conv_1", user_id="u1", title="first title")
        assert c1.id == "conv_1" and c1.title == "first title" and c1.user_id == "u1"
        # second call must NOT overwrite title or user_id
        c2 = await st.ensure_conversation("conv_1", user_id="u2", title="second title")
        assert c2.title == "first title" and c2.user_id == "u1"
        got = await st.get_conversation("conv_1")
        assert got is not None and got.title == "first title"
    _run(run)


@pytest.mark.parametrize("make_store", STORES)
def test_touch_sets_last_response_id_and_bumps_updated_at(make_store):
    async def run():
        st = await make_store()
        c = await st.ensure_conversation("conv_1", user_id=None, title="t")
        before = c.updated_at
        await asyncio.sleep(0.01)
        await st.touch_conversation("conv_1", last_response_id="resp_9")
        got = await st.get_conversation("conv_1")
        assert got.last_response_id == "resp_9"
        assert got.updated_at >= before
    _run(run)


@pytest.mark.parametrize("make_store", STORES)
def test_list_conversations_filters_by_user_and_orders_newest_first(make_store):
    async def run():
        st = await make_store()
        await st.ensure_conversation("conv_a", user_id="u1", title="a")
        await asyncio.sleep(0.01)
        await st.ensure_conversation("conv_b", user_id="u1", title="b")
        await asyncio.sleep(0.01)
        await st.ensure_conversation("conv_c", user_id="u2", title="c")
        # touch conv_a so it becomes the most-recently-updated for u1
        await asyncio.sleep(0.01)
        await st.touch_conversation("conv_a", last_response_id="resp_1")
        u1 = await st.list_conversations(user_id="u1", limit=50, offset=0)
        assert [c.id for c in u1] == ["conv_a", "conv_b"]
        # pagination
        page = await st.list_conversations(user_id="u1", limit=1, offset=1)
        assert [c.id for c in page] == ["conv_b"]
        # no filter -> all three, newest first
        allc = await st.list_conversations(user_id=None, limit=50, offset=0)
        assert allc[0].id == "conv_a" and len(allc) == 3
    _run(run)


@pytest.mark.parametrize("make_store", STORES)
def test_list_conversation_responses(make_store):
    async def run():
        st = await make_store()
        await st.ensure_conversation("conv_1", user_id=None, title="t")
        await st.save_response(StoredResponse(id="resp_1", conversation_id="conv_1", model="m", status="completed"))
        await st.save_response(StoredResponse(id="resp_2", conversation_id="conv_1", model="m", status="failed"))
        await st.save_response(StoredResponse(id="resp_x", conversation_id="conv_other", model="m", status="completed"))
        rs = await st.list_conversation_responses("conv_1")
        assert {r.id for r in rs} == {"resp_1", "resp_2"}
    _run(run)


@pytest.mark.parametrize("make_store", STORES)
def test_delete_conversation_cascades(make_store):
    async def run():
        st = await make_store()
        await st.ensure_conversation("conv_1", user_id=None, title="t")
        await st.append_items("conv_1", [Item(type="message", role="user", content={"text": "q"}, response_id="resp_1")])
        await st.save_response(StoredResponse(id="resp_1", conversation_id="conv_1", model="m", status="completed"))
        await st.delete_conversation("conv_1")
        assert await st.get_conversation("conv_1") is None
        assert await st.get_conversation_items("conv_1") == []
        assert await st.get_response("resp_1") is None
    _run(run)


@pytest.mark.parametrize("make_store", STORES)
def test_get_conversation_absent_returns_none(make_store):
    async def run():
        st = await make_store()
        assert await st.get_conversation("nope") is None
    _run(run)
