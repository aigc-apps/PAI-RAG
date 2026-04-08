"""Shared fixtures for config API tests using dependency overrides."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../backend"))

# Set up test environment before importing app
os.environ.setdefault("SQLITE_URL", "sqlite+aiosqlite:///./localdata/pytest_api.db")
os.environ.setdefault("DB_TYPE", "sqlite")
os.environ.setdefault("DISABLE_REDIS_CACHE_IN_TESTS", "true")

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from db.db_context import get_db_session
from service.injection import (
    get_trace_service,
    get_guardrail_service,
    get_websearch_service,
    get_vectordb_service,
    get_codesandbox_service,
    get_mcpserver_service,
    get_thread_service,
    get_message_service,
    get_llm_service,
    get_tenant_id,
)


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def override_tenant():
    return "test-tenant"


@pytest.fixture
def api_client(mock_session, override_tenant):
    """Create a test client with dependency overrides for session and tenant."""

    async def _get_session():
        yield mock_session

    app.dependency_overrides[get_db_session] = _get_session
    app.dependency_overrides[get_tenant_id] = lambda: override_tenant

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
