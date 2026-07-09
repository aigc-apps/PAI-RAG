from __future__ import annotations
import asyncio
import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
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
    return create_async_engine(db_url, future=True)


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
