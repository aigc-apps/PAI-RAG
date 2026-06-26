from __future__ import annotations
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlmodel import SQLModel
import app.models  # noqa: F401  (register tables on SQLModel.metadata)


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
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
