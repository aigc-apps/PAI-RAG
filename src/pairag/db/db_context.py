from contextlib import asynccontextmanager
from sqlmodel import SQLModel
from sqlalchemy.orm import sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.ext.asyncio.engine import AsyncEngine

from urllib.parse import quote_plus
import os


def create_db_engine(
    db_user: str,
    db_password: str,
    db_host: str,
    db_port: int,
    db_name: str,
) -> AsyncEngine:
    encoded_db_user = quote_plus(db_user)
    encoded_db_password = quote_plus(db_password)

    db_url = f"postgresql+asyncpg://{encoded_db_user}:{encoded_db_password}@{db_host}:{db_port}/{db_name}"
    async_engine = create_async_engine(db_url)
    return async_engine


class DbContext:
    def __init__(self):
        # 从环境变量中读取数据库配置
        db_name = os.getenv("DB_NAME")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", 5432)

        if not all([db_name, db_user, db_password, db_host, db_port]):
            raise ValueError("One or more database environment variables are missing.")

        self.async_engine = create_db_engine()

    async def init_db(self):
        async with self.async_engine.begin() as conn:
            # await conn.run_sync(SQLModel.metadata.drop_all)
            await conn.run_sync(SQLModel.metadata.create_all)

    @asynccontextmanager
    async def get_session(self):
        AsyncSessionLocal = sessionmaker(
            self.async_engine, class_=AsyncSession, expire_on_commit=False
        )
        async with AsyncSessionLocal() as session:
            yield session


db_context = DbContext()
