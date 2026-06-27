import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
import pytest
from app.store.memory import InMemoryStore
from app.store.sql import SqlStore
from app.db import make_engine, create_all
from app.store.base import MemoryItem


async def _mem():
    return InMemoryStore()


async def _sql():
    e = make_engine("sqlite+aiosqlite:///:memory:")
    await create_all(e)
    return SqlStore(e)


STORES = [_mem, _sql]


@pytest.mark.parametrize("make_store", STORES)
def test_add_list_update_delete(make_store):
    async def run():
        st = await make_store()
        a = await st.add_memory(MemoryItem(user_id="u1", text="likes tea"))
        await asyncio.sleep(0.01)
        b = await st.add_memory(MemoryItem(user_id="u1", text="lives in NYC"))
        await st.add_memory(MemoryItem(user_id="u2", text="other user"))
        mems = await st.list_memories("u1")
        assert [m.text for m in mems] == ["lives in NYC", "likes tea"]  # newest first
        await st.update_memory(b.id, "lives in Boston")
        assert (await st.list_memories("u1"))[0].text == "lives in Boston"
        await st.delete_memory(a.id)
        assert [m.text for m in await st.list_memories("u1")] == ["lives in Boston"]
    asyncio.run(run())


@pytest.mark.parametrize("make_store", STORES)
def test_limit_and_delete_user(make_store):
    async def run():
        st = await make_store()
        for i in range(5):
            await st.add_memory(MemoryItem(user_id="u1", text=f"f{i}"))
            await asyncio.sleep(0.001)
        assert len(await st.list_memories("u1", limit=3)) == 3
        await st.delete_user_memories("u1")
        assert await st.list_memories("u1") == []
    asyncio.run(run())
