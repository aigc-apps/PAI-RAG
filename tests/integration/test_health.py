"""Integration tests for health check endpoint."""

import pytest
from tests.integration.conftest import pytestmark  # noqa: F401


class TestHealthCheck:
    """Smoke tests to verify the app starts correctly."""

    def test_health_returns_200(self, client):
        """GET /health returns 200 with healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        # Response is wrapped: {"code": 200, "data": {"status": "healthy", ...}, "message": "..."}
        assert data["data"]["status"] == "healthy"

    def test_health_response_structure(self, client):
        """Health response contains expected fields."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "code" in data
        assert "data" in data
        assert "status" in data["data"]
        assert data["data"]["service"] == "PAI-RAG"
