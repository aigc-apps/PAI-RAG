
# flake8: noqa: E402
import dotenv
dotenv.load_dotenv()

from loguru import logger
from sqlmodel import SQLModel
from sqlalchemy.orm import sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from functools import wraps

from urllib.parse import quote_plus
import os


def get_async_db_angine():
    # 从环境变量中读取数据库配置
    if not os.path.exists("./localdata"):
        os.makedirs("./localdata")
    db_type= os.getenv("DB_TYPE", "sqlite")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", 5432)

    if db_type == "postgresql":
        assert db_name, "Postgres db_name不能为空。"
        assert db_host, "Postgres db_host不能为空。"
        assert db_name, "Postgres db_name不能为空。"
        assert db_user, "Postgres db_user你们为空。"
        assert db_password, "Postgres db_password不能为空。"
        assert db_port, "Postgres db_port不能为空。"

        encoded_db_user = quote_plus(db_user)
        encoded_db_password = quote_plus(db_password)

        db_url = f"postgresql+asyncpg://{encoded_db_user}:{encoded_db_password}@{db_host}:{db_port}/{db_name}"
        async_engine = create_async_engine(db_url, echo=False)
        logger.info(
            f"created async engine with {db_user}@{db_host}:{db_port}/{db_name}"
        )

        return async_engine
    else:
        local_db_url = os.getenv("SQLITE_URL", "sqlite+aiosqlite:///./tmp/sqlite/local.db")
        logger.warning(
            f"Created db engine with sqlite {local_db_url}."
        )
        async_engine = create_async_engine(
            local_db_url,
            echo=False,  # 输出执行的 SQL 语句
            connect_args={"check_same_thread": False},  # SQLite 特有参数
        )

        return async_engine


async_engine = get_async_db_angine()


async def init_db():
    async with async_engine.begin() as conn:
        # await conn.run_sync(SQLModel.metadata.drop_all)
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session():
    AsyncSessionLocal = sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with AsyncSessionLocal() as session:
        yield session


def with_async_db_session(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        AsyncSessionLocal = sessionmaker(
            async_engine, class_=AsyncSession, expire_on_commit=False
        )
        async with AsyncSessionLocal() as session:
            kwargs["session"] = session
            return await func(*args, **kwargs)

    return wrapper
