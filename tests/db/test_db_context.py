import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(autouse=True)
def _reset_engine_state():
    """Each test starts with a fresh lazy engine/session-factory cache."""
    from db.db_context import reset_engine_for_test
    reset_engine_for_test()
    yield
    reset_engine_for_test()


class TestGetAsyncDbEngine:
    @patch.dict(os.environ, {"DB_TYPE": "sqlite", "SQLITE_URL": "sqlite+aiosqlite:///:memory:"}, clear=False)
    @patch("db.db_context.event")
    def test_sqlite_engine(self, mock_event):
        with patch("db.db_context.create_async_engine") as mock_create:
            mock_engine = MagicMock()
            mock_engine.sync_engine = MagicMock()
            mock_create.return_value = mock_engine
            from db.db_context import get_async_db_engine
            engine = get_async_db_engine()
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
            from db.db_context import get_async_db_engine
            engine = get_async_db_engine()
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
            from db.db_context import get_async_db_engine
            engine = get_async_db_engine()
            assert engine is not None
            call_args = mock_create.call_args[0][0]
            assert "mysql+aiomysql" in call_args

    def test_deprecated_alias_still_works(self):
        """The historical misspelling `get_async_db_angine` must remain
        importable as an alias to `get_async_db_engine` for backwards
        compatibility. Remove this test when the alias is dropped."""
        from db.db_context import get_async_db_angine, get_async_db_engine
        assert get_async_db_angine is get_async_db_engine


class TestLazyEngineInit:
    """Regression tests ensuring `db.db_context` import is side-effect free."""

    def test_get_engine_is_cached(self):
        from db import db_context
        with patch("db.db_context.get_async_db_engine") as factory:
            sentinel = MagicMock(name="engine")
            factory.return_value = sentinel
            assert db_context.get_engine() is sentinel
            assert db_context.get_engine() is sentinel  # cached
            factory.assert_called_once()

    def test_get_session_factory_is_cached(self):
        from db import db_context
        sentinel_engine = MagicMock(name="engine")
        db_context.set_engine_for_test(sentinel_engine)
        f1 = db_context.get_session_factory()
        f2 = db_context.get_session_factory()
        assert f1 is f2

    def test_set_engine_for_test_rebuilds_session_factory(self):
        from db import db_context
        db_context.set_engine_for_test(MagicMock(name="engine-a"))
        first_factory = db_context.get_session_factory()
        db_context.set_engine_for_test(MagicMock(name="engine-b"))
        second_factory = db_context.get_session_factory()
        assert first_factory is not second_factory

    def test_async_engine_attr_resolves_via_getattr(self):
        from db import db_context
        sentinel = MagicMock(name="engine")
        db_context.set_engine_for_test(sentinel)
        assert db_context.async_engine is sentinel

    def test_async_session_local_attr_resolves_via_getattr(self):
        from db import db_context
        db_context.set_engine_for_test(MagicMock(name="engine"))
        assert db_context.AsyncSessionLocal is db_context.get_session_factory()


class TestGetDbSession:
    async def test_session_commits_on_success(self):
        from db import db_context
        mock_session = AsyncMock()
        db_context.set_session_factory_for_test(lambda: mock_session)
        gen = db_context.get_db_session()
        session = await gen.__anext__()
        assert session is mock_session
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    async def test_session_rollbacks_on_error(self):
        from db import db_context
        mock_session = AsyncMock()
        db_context.set_session_factory_for_test(lambda: mock_session)
        gen = db_context.get_db_session()
        session = await gen.__anext__()
        assert session is mock_session
        with pytest.raises(ValueError):
            await gen.athrow(ValueError("test error"))
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()


class TestCreateDbSession:
    async def test_context_manager_commits(self):
        from db import db_context
        mock_session = AsyncMock()
        db_context.set_session_factory_for_test(lambda: mock_session)
        async with db_context.create_db_session() as session:
            assert session is mock_session
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    async def test_context_manager_rollbacks_on_error(self):
        from db import db_context
        mock_session = AsyncMock()
        db_context.set_session_factory_for_test(lambda: mock_session)
        with pytest.raises(ValueError):
            async with db_context.create_db_session():
                raise ValueError("test")
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()


class TestWithAsyncDbSession:
    async def test_decorator_injects_session(self):
        from db import db_context
        mock_session = AsyncMock()
        db_context.set_session_factory_for_test(lambda: mock_session)

        @db_context.with_async_db_session
        async def my_func(session=None):
            assert session is mock_session
            return "result"

        result = await my_func()
        assert result == "result"
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    async def test_decorator_rollbacks_on_error(self):
        from db import db_context
        mock_session = AsyncMock()
        db_context.set_session_factory_for_test(lambda: mock_session)

        @db_context.with_async_db_session
        async def my_func(session=None):
            raise ValueError("test")

        with pytest.raises(ValueError):
            await my_func()
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()
