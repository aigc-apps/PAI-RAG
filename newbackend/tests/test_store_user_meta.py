import sys, os, asyncio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db import make_engine, create_all
from app.store.sql import SqlStore
from app.store.memory import InMemoryStore


async def _roundtrip(store):
    await store.ensure_user("u_1")
    u = await store.update_user_meta("u_1", {"aliyun_pai": {"role_arn": "acs:ram::17:role/r"}})
    assert u.meta["aliyun_pai"]["role_arn"] == "acs:ram::17:role/r"
    # shallow merge preserves other keys
    await store.update_user_meta("u_1", {"other": 1})
    u = await store.get_user("u_1")
    assert u.meta["aliyun_pai"]["role_arn"] == "acs:ram::17:role/r"
    assert u.meta["other"] == 1
    # deauthorize keeps the key present as null
    await store.update_user_meta("u_1", {"aliyun_pai": None})
    u = await store.get_user("u_1")
    assert u.meta["aliyun_pai"] is None


def test_memory_store_user_meta_roundtrip():
    asyncio.run(_roundtrip(InMemoryStore()))


def test_memory_store_creates_user_on_update():
    async def run():
        store = InMemoryStore()
        u = await store.update_user_meta("u_new", {"aliyun_pai": {"x": 1}})
        assert u.id == "u_new" and u.meta["aliyun_pai"]["x"] == 1
    asyncio.run(run())


def test_sql_store_user_meta_roundtrip():
    async def run():
        engine = make_engine("sqlite+aiosqlite:///:memory:")
        await create_all(engine)
        await _roundtrip(SqlStore(engine))
    asyncio.run(run())


def test_sql_store_meta_survives_new_session(tmp_path):
    """JSON mutation must be flag_modified'd or the write silently no-ops.
    A second, independent engine reads the committed row from disk (no shared
    identity map), so a missing flag_modified would surface as stale data."""
    db = f"sqlite+aiosqlite:///{tmp_path}/u.db"

    async def run():
        w = make_engine(db)
        await create_all(w)
        st = SqlStore(w)
        await st.ensure_user("u_1")
        await st.update_user_meta("u_1", {"aliyun_pai": {"role_arn": "acs:ram::17:role/r"}})
        await w.dispose()

        r = make_engine(db)
        u = await SqlStore(r).get_user("u_1")
        await r.dispose()
        return u

    u = asyncio.run(run())
    assert u.meta["aliyun_pai"]["role_arn"] == "acs:ram::17:role/r"
