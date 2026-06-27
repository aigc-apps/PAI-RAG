import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest
from app.schemas import ResponsesRequest
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


def test_resolved_user_id_precedence():
    assert ResponsesRequest(input="x", user_id="a", user="b").resolved_user_id == "a"
    assert ResponsesRequest(input="x", user="b").resolved_user_id == "b"
    assert ResponsesRequest(input="x", safety_identifier="s").resolved_user_id == "s"
    assert ResponsesRequest(input="x").resolved_user_id is None


@pytest.mark.parametrize("make_store", STORES)
def test_ensure_user_idempotent_and_get(make_store):
    async def run():
        st = await make_store()
        u1 = await st.ensure_user("u1", display_name="Alice")
        assert u1.id == "u1" and u1.display_name == "Alice"
        u2 = await st.ensure_user("u1", display_name="ignored")
        assert u2.display_name == "Alice"  # not overwritten
        assert (await st.get_user("u1")).id == "u1"
        assert await st.get_user("nope") is None
    asyncio.run(run())
