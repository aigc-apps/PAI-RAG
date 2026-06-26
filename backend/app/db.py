from __future__ import annotations
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlmodel import SQLModel
import app.models  # noqa: F401  (register tables on SQLModel.metadata)


def make_engine(db_url: str) -> AsyncEngine:
    return create_async_engine(db_url, future=True)


async def create_all(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
