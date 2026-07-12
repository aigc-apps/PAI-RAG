# ruff: noqa: E402
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alembic.script import ScriptDirectory
from sqlalchemy import inspect
from app.db import make_engine, migrate, _alembic_config
from app.store.sql import SqlStore


def _head_revision(db_url: str) -> str:
    return ScriptDirectory.from_config(_alembic_config(db_url)).get_current_head()


async def _columns(engine, table):
    async with engine.connect() as conn:
        return await conn.run_sync(
            lambda c: {col["name"] for col in inspect(c).get_columns(table)}
        )


async def _tables(engine):
    async with engine.connect() as conn:
        return await conn.run_sync(lambda c: set(inspect(c).get_table_names()))


def test_migrate_builds_full_schema_on_fresh_db(tmp_path):
    """A brand-new persistent DB is created entirely by `alembic upgrade head`,
    with the auth columns present and the version stamped at head."""
    async def run():
        url = f"sqlite+aiosqlite:///{tmp_path / 'fresh.db'}"
        await migrate(url)

        engine = make_engine(url)
        tables = await _tables(engine)
        for t in ("users", "conversations", "conversation_items", "responses",
                  "knowledge_bases", "knowledge_documents", "knowledge_chunks",
                  "app_config_documents", "app_config_revisions"):
            assert t in tables, f"{t} missing"

        user_cols = await _columns(engine, "users")
        for c in ("email", "password_hash", "role", "status",
                  "invite_token_hash", "invite_expires_at"):
            assert c in user_cols, f"users.{c} missing"

        # Version table stamped at head.
        assert "alembic_version" in tables
        async with engine.connect() as conn:
            ver = await conn.exec_driver_sql("SELECT version_num FROM alembic_version")
            rows = ver.fetchall()
        assert [r[0] for r in rows] == [_head_revision(url)]

        # The query that used to crash on the un-migrated DB now runs.
        store = SqlStore(engine)
        assert await store.get_user_auth("feiyue@pku.edu.cn") is None
        await engine.dispose()

    asyncio.run(run())


def test_migrate_is_idempotent(tmp_path):
    async def run():
        url = f"sqlite+aiosqlite:///{tmp_path / 'idem.db'}"
        await migrate(url)
        await migrate(url)  # second run is a no-op, not an error

        engine = make_engine(url)
        async with engine.connect() as conn:
            rows = (await conn.exec_driver_sql(
                "SELECT version_num FROM alembic_version")).fetchall()
        assert [r[0] for r in rows] == [_head_revision(url)]
        await engine.dispose()

    asyncio.run(run())
