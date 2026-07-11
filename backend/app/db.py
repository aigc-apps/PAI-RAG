from __future__ import annotations
import asyncio
import logging
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import event
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlmodel import SQLModel
import app.models  # noqa: F401  (register tables on SQLModel.metadata)

log = logging.getLogger("app.db")

_BACKEND_DIR = Path(__file__).resolve().parent.parent  # backend/
_ALEMBIC_INI = _BACKEND_DIR / "alembic.ini"


def make_engine(db_url: str) -> AsyncEngine:
    # An in-memory SQLite DB lives inside a single connection; pooling normally
    # hands out fresh (empty) connections per session. StaticPool keeps one
    # shared connection so create_all and all store sessions see the same DB.
    if ":memory:" in db_url:
        return create_async_engine(
            db_url,
            future=True,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    if db_url.startswith("sqlite"):
        # Dev / single-node fallback. SQLite is a single-writer store, so the
        # first-token path must never stall behind a peer's write: WAL lets
        # readers run concurrently with a writer, synchronous=NORMAL trims fsyncs
        # (durable enough under WAL), and busy_timeout makes a would-be writer
        # wait out a lock instead of failing with "database is locked" — needed
        # now the background worker pool writes the same file. Production uses
        # PostgreSQL (below); this branch is not the throughput target.
        engine = create_async_engine(
            db_url, future=True, connect_args={"timeout": 30}
        )

        @event.listens_for(engine.sync_engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _rec):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            try:
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.execute("PRAGMA busy_timeout=30000")
                cur.execute("PRAGMA foreign_keys=ON")
            finally:
                cur.close()

        return engine
    # PostgreSQL (production) and other server-backed DBs. An explicitly sized,
    # health-checked pool: up to ~30 in-flight connections (pool_size 10 + overflow
    # 20) back ~100 concurrent conversations, since each store op borrows a
    # connection only for its brief query and returns it. Bump PAIRAG_DB_POOL_SIZE
    # if steady-state concurrency needs a larger warm pool. pool_pre_ping discards a
    # connection killed by a server restart / idle timeout instead of erroring the
    # turn's first query on it; pool_recycle refreshes long-lived connections.
    return create_async_engine(
        db_url,
        future=True,
        pool_size=int(os.getenv("PAIRAG_DB_POOL_SIZE", "10")),
        max_overflow=int(os.getenv("PAIRAG_DB_MAX_OVERFLOW", "20")),
        pool_pre_ping=True,
        pool_recycle=int(os.getenv("PAIRAG_DB_POOL_RECYCLE", "1800")),
    )


async def create_all(engine: AsyncEngine) -> None:
    """Create every table from the model metadata.

    Used for ephemeral databases only — the in-memory store and the test suite,
    where there's nothing to migrate and a version history has no value. Real
    persistent databases go through Alembic via ``migrate()``.
    """
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


def _alembic_config(db_url: str) -> Config:
    cfg = Config(str(_ALEMBIC_INI))
    # Absolute script location so it resolves regardless of the process cwd.
    cfg.set_main_option("script_location", str(_BACKEND_DIR / "alembic"))
    # env.py reads this as the target DB (see its _db_url priority order).
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def _upgrade_head(db_url: str) -> None:
    command.upgrade(_alembic_config(db_url), "head")


async def migrate(db_url: str) -> None:
    """Bring a persistent database up to the latest schema (``alembic upgrade head``).

    A fresh DB gets every table from the baseline migration; an already-managed
    DB gets any new revisions. Alembic's command API is synchronous and its
    async env.py calls ``asyncio.run`` internally, which would explode inside the
    app's running event loop — so run it in a worker thread where no loop is live.
    """
    log.info("running database migrations (alembic upgrade head)")
    await asyncio.to_thread(_upgrade_head, db_url)
