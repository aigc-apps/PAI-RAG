import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))
import pytest
from app.store.memory import InMemoryStore
from app.store.sql import SqlStore
from app.db import make_engine, create_all


async def _mem():
    return InMemoryStore()


async def _sql():
    e = make_engine("sqlite+aiosqlite:///:memory:")
    await create_all(e)
    return SqlStore(e)


STORES = [_mem, _sql]


@pytest.mark.parametrize("make_store", STORES)
def test_summary_defaults_and_update(make_store):
    async def run():
        st = await make_store()
        await st.ensure_conversation("c1", user_id="u1", title="t")
        conv = await st.get_conversation("c1")
        assert conv.summary is None and conv.summarized_seq == -1
        await st.update_conversation_summary("c1", "they discussed hiking", 7)
        conv = await st.get_conversation("c1")
        assert conv.summary == "they discussed hiking" and conv.summarized_seq == 7
    asyncio.run(run())
