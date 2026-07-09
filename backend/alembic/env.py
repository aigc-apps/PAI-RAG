"""Alembic environment — async engine, schema sourced from SQLModel metadata.

The URL comes from the app settings (DB_URL env var), never from alembic.ini, so
`alembic upgrade head` on the CLI and the app's boot-time migrate() hit the same
database. Works for both sqlite (batch mode) and postgres (native ALTER)."""
from __future__ import annotations

import asyncio

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

# Import the models so every table registers on SQLModel.metadata — this is what
# autogenerate diffs against. Keep the import even though it looks unused.
import app.models  # noqa: F401
from app.config import get_settings

config = context.config
target_metadata = SQLModel.metadata


def _db_url() -> str:
    # Priority: `-x db_url=...` (one-off CLI targets) > a URL injected on the
    # Config by the app's boot-time migrate() (sqlalchemy.url main option) >
    # the app settings, so the CLI and the app stay in lockstep by default.
    x_args = context.get_x_argument(as_dictionary=True)
    if x_args.get("db_url"):
        return x_args["db_url"]
    injected = config.get_main_option("sqlalchemy.url")
    if injected:
        return injected
    return get_settings().db_url


def _configure(connection) -> None:
    is_sqlite = connection.dialect.name == "sqlite"
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # SQLite can't ALTER in place — batch mode rebuilds the table. Harmless
        # to skip on postgres, which alters natively.
        render_as_batch=is_sqlite,
        compare_type=True,
        compare_server_default=True,
    )


def run_migrations_offline() -> None:
    context.configure(
        url=_db_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=_db_url().startswith("sqlite"),
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _do_run_migrations(connection) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_db_url(), future=True)
    async with engine.connect() as connection:
        await connection.run_sync(_do_run_migrations)
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
