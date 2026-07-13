# ruff: noqa: E402
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from alembic.script import ScriptDirectory
from sqlalchemy import inspect
from app.db import make_engine, migrate, _alembic_config
from app.store.sql import SqlStore


MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "alembic" / "versions"


def _head_revision(db_url: str) -> str:
    return ScriptDirectory.from_config(_alembic_config(db_url)).get_current_head()


def test_alembic_config_preserves_percent_encoded_runtime_url():
    url = (
        "postgresql+asyncpg://pairag:Test1234%40%25@"
        "pgm.example.com:5432/loop0713"
    )

    config = _alembic_config(url)

    assert config.attributes["db_url"] == url
    assert config.get_main_option("sqlalchemy.url") is None


def test_migrations_do_not_create_timezone_naive_datetime_columns():
    offenders = [
        path.name
        for path in MIGRATIONS_DIR.glob("*.py")
        if "sa.DateTime()" in path.read_text()
    ]

    assert offenders == []


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

        job_cols = await _columns(engine, "background_jobs")
        assert {
            "progress", "heartbeat_at", "lease_expires_at", "cancel_requested_at",
        } <= job_cols
        datasource_cols = await _columns(engine, "knowledge_data_sources")
        assert "active_job_id" in datasource_cols
        document_cols = await _columns(engine, "knowledge_documents")
        assert {
            "search_index_status", "search_index_error", "search_index_attempts",
        } <= document_cols

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
