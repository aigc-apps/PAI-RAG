
# flake8: noqa: E402
import dotenv
dotenv.load_dotenv()

from loguru import logger
from sqlmodel import SQLModel
from sqlalchemy import event
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncEngine
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from contextlib import asynccontextmanager
from functools import wraps
from typing import AsyncGenerator, Optional

from urllib.parse import quote_plus
import os


def get_async_db_engine() -> AsyncEngine:
    # 从环境变量中读取数据库配置
    if not os.path.exists("./localdata"):
        os.makedirs("./localdata")
    db_type = os.getenv("DB_TYPE", "sqlite")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT")

    if db_type == "postgresql":
        assert db_name, "Postgres db_name不能为空。"
        assert db_host, "Postgres db_host不能为空。"
        assert db_user, "Postgres db_user不能为空。"
        assert db_password, "Postgres db_password不能为空。"
        db_port = int(db_port) if db_port else 5432

        encoded_db_user = quote_plus(db_user)
        encoded_db_password = quote_plus(db_password)

        db_url = f"postgresql+asyncpg://{encoded_db_user}:{encoded_db_password}@{db_host}:{db_port}/{db_name}"
        async_engine = create_async_engine(
            db_url,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=300)
        logger.info(
            f"created async engine with {db_user}@{db_host}:{db_port}/{db_name}"
        )

        return async_engine
    elif db_type == "mysql":
        assert db_name, "MySQL db_name不能为空。"
        assert db_host, "MySQL db_host不能为空。"
        assert db_user, "MySQL db_user不能为空。"
        assert db_password, "MySQL db_password不能为空。"
        db_port = int(db_port) if db_port else 3306

        encoded_db_user = quote_plus(db_user)
        encoded_db_password = quote_plus(db_password)

        db_url = f"mysql+aiomysql://{encoded_db_user}:{encoded_db_password}@{db_host}:{db_port}/{db_name}?charset=utf8mb4"
        async_engine = create_async_engine(
            db_url,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=10,
            max_overflow=20,
            connect_args={
                "charset": "utf8mb4",
                "use_unicode": True,
            },
        )
        logger.info(
            f"created async engine with MySQL {db_user}@{db_host}:{db_port}/{db_name}"
        )

        return async_engine
    else:
        db_url = os.getenv("SQLITE_URL", "sqlite+aiosqlite:///./tmp/sqlite/local.db")
        logger.info(f"Creating SQLite engine: {db_url}")

        async_engine = create_async_engine(
            db_url,
            echo=False,
            connect_args={
                "check_same_thread": False,
                "timeout": 60,
            },
            pool_pre_ping=True,
        )

        @event.listens_for(async_engine.sync_engine, "connect")
        def _set_pragma(dbapi_conn, _):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=60000")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA cache_size=-64000")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()
            # logger.debug("SQLite PRAGMA applied: WAL, busy_timeout=60s")

        return async_engine


# ---------------------------------------------------------------------------
# Lazy singletons
# ---------------------------------------------------------------------------
# The engine and session factory are constructed on first use rather than at
# module-import time. This keeps `import db.db_context` (and any module that
# transitively imports it, such as FastAPI routers under `api.*`) free of side
# effects, which makes unit tests, CLI tools, and Alembic significantly easier
# to reason about.
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker] = None


def get_engine() -> AsyncEngine:
    """Return the process-wide async engine, creating it on first access."""
    global _engine
    if _engine is None:
        _engine = get_async_db_engine()
    return _engine


def get_session_factory() -> async_sessionmaker:
    """Return the process-wide async session factory, bound to the lazy engine."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


# ---------------------------------------------------------------------------
# Test hooks
# ---------------------------------------------------------------------------
def reset_engine_for_test() -> None:
    """Drop any cached engine/session factory so the next call rebuilds them.

    Intended for unit tests that mutate `DB_TYPE`/`SQLITE_URL` and need a fresh
    engine. Not safe for production use.
    """
    global _engine, _session_factory
    _engine = None
    _session_factory = None


def set_engine_for_test(engine: AsyncEngine) -> None:
    """Inject a pre-built engine (e.g. an in-memory mock) for tests."""
    global _engine, _session_factory
    _engine = engine
    _session_factory = None  # force rebuild against the injected engine


def set_session_factory_for_test(factory) -> None:
    """Inject a session factory (e.g. one that yields AsyncMock sessions)."""
    global _session_factory
    _session_factory = factory


# ---------------------------------------------------------------------------
# Backwards compatibility
# ---------------------------------------------------------------------------
# Historical callers and tests reference `async_engine` and `AsyncSessionLocal`
# as module-level attributes. Expose them via PEP 562 `__getattr__` so the
# import surface is unchanged while the actual construction stays lazy.
def __getattr__(name):
    if name == "async_engine":
        return get_engine()
    if name == "AsyncSessionLocal":
        return get_session_factory()
    raise AttributeError(f"module 'db.db_context' has no attribute {name!r}")


# Backwards-compatible alias for the historical misspelling.
# Deprecated: use get_async_db_engine() instead. Will be removed in a future release.
get_async_db_angine = get_async_db_engine


async def init_db():
    async with get_engine().begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


@asynccontextmanager
async def create_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager that owns a SQLAlchemy AsyncSession lifecycle.

    Behavior:
        - Creates a new Session from the lazy session factory.
        - On normal exit: commits the session.
        - On exception: rolls back and re-raises.
        - Always closes the session.

    This is the single source of truth for session lifecycle management;
    `get_db_session` (FastAPI dependency) and `with_async_db_session`
    (decorator) both delegate to it to avoid duplication.
    """
    session = get_session_factory()()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency: yields an AsyncSession per request with
    commit/rollback/close managed by `create_db_session`.
    """
    async with create_db_session() as session:
        yield session


def with_async_db_session(func):
    """Decorator that injects a managed AsyncSession as `session` kwarg."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        async with create_db_session() as session:
            kwargs["session"] = session
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Execution error: {e}")
                raise

    return wrapper
