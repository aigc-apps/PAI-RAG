import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestGetAsyncDbEngine:
    @patch.dict(os.environ, {"DB_TYPE": "sqlite", "SQLITE_URL": "sqlite+aiosqlite:///:memory:"}, clear=False)
    @patch("db.db_context.event")
    def test_sqlite_engine(self, mock_event):
        with patch("db.db_context.create_async_engine") as mock_create:
            mock_engine = MagicMock()
            mock_engine.sync_engine = MagicMock()
            mock_create.return_value = mock_engine
            from db.db_context import get_async_db_angine
            engine = get_async_db_angine()
            assert engine is not None

    @patch.dict(os.environ, {
        "DB_TYPE": "postgresql",
        "DB_NAME": "testdb",
        "DB_HOST": "localhost",
        "DB_USER": "user",
        "DB_PASSWORD": "pass",
    }, clear=False)
    def test_postgresql_engine(self):
        with patch("db.db_context.create_async_engine") as mock_create:
            mock_create.return_value = MagicMock()
            from db.db_context import get_async_db_angine
            engine = get_async_db_angine()
            assert engine is not None
            call_args = mock_create.call_args[0][0]
            assert "postgresql+asyncpg" in call_args

    @patch.dict(os.environ, {
        "DB_TYPE": "mysql",
        "DB_NAME": "testdb",
        "DB_HOST": "localhost",
        "DB_USER": "user",
        "DB_PASSWORD": "pass",
    }, clear=False)
    def test_mysql_engine(self):
        with patch("db.db_context.create_async_engine") as mock_create:
            mock_create.return_value = MagicMock()
            from db.db_context import get_async_db_angine
            engine = get_async_db_angine()
            assert engine is not None
            call_args = mock_create.call_args[0][0]
            assert "mysql+aiomysql" in call_args


class TestGetDbSession:
    async def test_session_commits_on_success(self):
        from db.db_context import get_db_session
        mock_session = AsyncMock()
        with patch("db.db_context.AsyncSessionLocal", return_value=mock_session):
            gen = get_db_session()
            session = await gen.__anext__()
            assert session == mock_session
            try:
                await gen.__anext__()
            except StopAsyncIteration:
                pass
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()

    async def test_session_rollbacks_on_error(self):
        from db.db_context import get_db_session
        mock_session = AsyncMock()
        with patch("db.db_context.AsyncSessionLocal", return_value=mock_session):
            gen = get_db_session()
            session = await gen.__anext__()
            with pytest.raises(ValueError):
                await gen.athrow(ValueError("test error"))
            mock_session.rollback.assert_called_once()
            mock_session.close.assert_called_once()


class TestCreateDbSession:
    async def test_context_manager_commits(self):
        from db.db_context import create_db_session
        mock_session = AsyncMock()
        with patch("db.db_context.AsyncSessionLocal", return_value=mock_session):
            async with create_db_session() as session:
                assert session == mock_session
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()

    async def test_context_manager_rollbacks_on_error(self):
        from db.db_context import create_db_session
        mock_session = AsyncMock()
        with patch("db.db_context.AsyncSessionLocal", return_value=mock_session):
            with pytest.raises(ValueError):
                async with create_db_session() as session:
                    raise ValueError("test")
            mock_session.rollback.assert_called_once()
            mock_session.close.assert_called_once()


class TestWithAsyncDbSession:
    async def test_decorator_injects_session(self):
        from db.db_context import with_async_db_session
        mock_session = AsyncMock()
        with patch("db.db_context.AsyncSessionLocal", return_value=mock_session):
            @with_async_db_session
            async def my_func(session=None):
                return "result"
            result = await my_func()
            assert result == "result"
            mock_session.commit.assert_called_once()
            mock_session.close.assert_called_once()

    async def test_decorator_rollbacks_on_error(self):
        from db.db_context import with_async_db_session
        mock_session = AsyncMock()
        with patch("db.db_context.AsyncSessionLocal", return_value=mock_session):
            @with_async_db_session
            async def my_func(session=None):
                raise ValueError("test")
            with pytest.raises(ValueError):
                await my_func()
            mock_session.rollback.assert_called_once()
            mock_session.close.assert_called_once()
