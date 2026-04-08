"""Shared fixtures for service layer tests."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../backend"))

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_session():
    """Create a mock AsyncSession for service tests."""
    session = AsyncMock()

    # Mock exec to return a result proxy with .first() and .all()
    mock_result = MagicMock()
    mock_result.first.return_value = None
    mock_result.all.return_value = []
    mock_result.one_or_none.return_value = 0
    session.exec.return_value = mock_result

    # Mock flush, refresh, commit, delete, add, get
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    session.commit = AsyncMock()
    session.delete = AsyncMock()
    session.add = MagicMock()
    session.get = AsyncMock(return_value=None)

    return session


def make_mock_result(first_value=None, all_values=None, one_or_none_value=None):
    """Helper to create a mock result with custom return values."""
    result = MagicMock()
    result.first.return_value = first_value
    result.all.return_value = all_values or []
    result.one_or_none.return_value = one_or_none_value
    return result
